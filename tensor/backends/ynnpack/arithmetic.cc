/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/ynnpack/arithmetic.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/backends/common_nnpack/utils.h"
#include "tensor/backends/ynnpack/conversion.h"
#include "tensor/backends/ynnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/type_id.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::graph {

namespace {

// YNNPACK creates the value backing a node output when the in/out id is
// `YNN_INVALID_VALUE_ID`. Intermediate values introduced by a lowering use this
// so that YNNPACK infers their shape and type.
constexpr uint32_t kInferredValueId = YnnpackBuildContext::kInferredValueId;

struct BinaryIOIds {
  uint32_t lhs;
  uint32_t rhs;
  uint32_t output;
};

struct UnaryIOIds {
  uint32_t input;
  uint32_t output;
};

// Returns the single output value id of `op`.
absl::StatusOr<uint32_t> DefineSingleOutput(const Operation& op,
                                            YnnpackBuildContext& ctx,
                                            absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  return ctx.DefineValue(outputs.front());
}

// Marks untouched dimensions in `DefineBroadcastOperand` broadcast shape.
constexpr int32_t kKeepDim = -1;

// Marks the dimensions of `operand` that can be broadcasted.
//
// - `ctx`: the YNNPACK build context.
// - `value_id`: the YNNPACK value of `operand`.
// - `operand`: input tensor of an op that supports broadcasting.
// - `broadcast_shape`: the shape `operand` is broadcasted to, used to find
//   which dimensions are broadcasted. **`kKeepDim` elements are skipped.**
// - `op_name`: the operation name, for error handling.
//
// Note: The operand and broadcast shapes are aligned from the innermost
// dimension.
//
// Returns `value_id` if no broadcasting is needed, otherwise the id of the
// broadcast result.
//
// YNNPACK decides which dimensions broadcast when the graph is defined.
// Dimensions that can be broadcasted are either:
//
//  - declared with a static extent of 1,
//  - created by a YNNPACK op,
//  - marked with `ynn_define_broadcast`.
//
// Values such as external inputs are declared with dynamic extents, so a
// dimension of theirs that is 1 has to be marked explicitly.
//
// ```
// broadcast_shape: [2, 3, kKeepDim, kKeepDim]
// operand:            [1,        8,        5]
//                      |         └─────────┴ left alone
//                      └ 1 vs 3: marked
// ```
absl::StatusOr<uint32_t> DefineBroadcastOperand(
    YnnpackBuildContext& ctx, uint32_t value_id, const graph::Tensor& operand,
    absl::Span<const int32_t> broadcast_shape, absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& info,
                              graph::GetInfo(operand));
  const absl::Span<const int32_t> shape = info.shape;
  const size_t common_rank = std::min(shape.size(), broadcast_shape.size());
  const size_t operand_offset = shape.size() - common_rank;
  const size_t broadcast_offset = broadcast_shape.size() - common_rank;
  std::vector<int32_t> axes;
  for (size_t i = 0; i < common_rank; ++i) {
    const int32_t extent = broadcast_shape[broadcast_offset + i];
    if (shape[operand_offset + i] == 1 && extent != 1 && extent != kKeepDim) {
      axes.push_back(static_cast<int32_t>(operand_offset + i));
    }
  }
  if (axes.empty()) {
    return value_id;
  }
  uint32_t broadcast_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_broadcast(ctx.subgraph(), axes.size(),
                                                  axes.data(), value_id,
                                                  &broadcast_id, /*flags=*/0))
      << op_name;
  return broadcast_id;
}

absl::StatusOr<BinaryIOIds> PrepareBinaryIO(const Operation& op,
                                            YnnpackBuildContext& ctx,
                                            absl::string_view op_name) {
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects two inputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t lhs_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t rhs_id, ctx.DefineValue(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      lhs_id, DefineBroadcastOperand(ctx, lhs_id, op.inputs[0],
                                     output_info.shape, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      rhs_id, DefineBroadcastOperand(ctx, rhs_id, op.inputs[1],
                                     output_info.shape, op_name));
  return BinaryIOIds{lhs_id, rhs_id, output_id};
}

absl::StatusOr<UnaryIOIds> PrepareUnaryIO(const Operation& op,
                                          YnnpackBuildContext& ctx,
                                          absl::string_view op_name) {
  if (op.inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects one input", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  return UnaryIOIds{input_id, output_id};
}

// YNNPACK nodes have no fused activation parameters. Clamping is expressed with
// explicit max/min nodes, which the subgraph fusion pass folds back into the
// producing node.
bool NeedsClamp(const NnpackActivationBounds& bounds) {
  return bounds.min > -std::numeric_limits<float>::infinity() ||
         bounds.max < std::numeric_limits<float>::infinity();
}

absl::StatusOr<uint32_t> DefineScalarConstant(YnnpackBuildContext& ctx,
                                              float value) {
  return ctx.DefineConstant(&value, sizeof(value), Type::kFP32, /*shape=*/{});
}

// Emits `output = min(max(input, bounds.min), bounds.max)`, skipping the
// clamps whose bound is infinite.
//
// Warning: this should only be called when `NeedsClamp` holds, otherwise
// nothing would write to `output_id`.
absl::Status ApplyActivationBounds(YnnpackBuildContext& ctx, uint32_t input_id,
                                   const NnpackActivationBounds& bounds,
                                   uint32_t output_id,
                                   absl::string_view op_name) {
  const bool has_min = bounds.min > -std::numeric_limits<float>::infinity();
  const bool has_max = bounds.max < std::numeric_limits<float>::infinity();
  uint32_t current_id = input_id;
  if (has_min) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t min_id,
                                DefineScalarConstant(ctx, bounds.min));
    uint32_t clamped_id = has_max ? kInferredValueId : output_id;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(ctx.subgraph(), ynn_binary_max,
                                                 current_id, min_id,
                                                 &clamped_id, /*flags=*/0))
        << op_name;
    current_id = clamped_id;
  }
  if (has_max) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t max_id,
                                DefineScalarConstant(ctx, bounds.max));
    uint32_t clamped_id = output_id;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(ctx.subgraph(), ynn_binary_min,
                                                 current_id, max_id,
                                                 &clamped_id, /*flags=*/0))
        << op_name;
  }
  return absl::OkStatus();
}

absl::Status AddBinaryNode(ynn_binary_operator op_type, const BinaryIOIds& io,
                           FusedActivation activation, YnnpackBuildContext& ctx,
                           absl::string_view op_name) {
  const NnpackActivationBounds bounds = GetActivationBounds(activation);
  const bool needs_clamp = NeedsClamp(bounds);
  uint32_t result_id = needs_clamp ? kInferredValueId : io.output;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(ctx.subgraph(), op_type, io.lhs,
                                               io.rhs, &result_id,
                                               /*flags=*/0))
      << op_name;
  if (needs_clamp) {
    return ApplyActivationBounds(ctx, result_id, bounds, io.output, op_name);
  }
  return absl::OkStatus();
}

absl::Status AddUnaryNode(ynn_unary_operator op_type, const UnaryIOIds& io,
                          YnnpackBuildContext& ctx, absl::string_view op_name) {
  uint32_t output_id = io.output;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_unary(ctx.subgraph(), op_type, io.input,
                                              &output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

// Reads a constant int32 tensor into a vector.
absl::StatusOr<std::vector<int32_t>> GetConstantInt32Vector(
    const graph::Tensor& tensor, absl::string_view op_name,
    absl::string_view field_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& info,
                              graph::GetInfo(tensor));
  if (info.type != Type::kI32) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: %s must be int32. Got type id %d.", op_name,
                        field_name, static_cast<int>(info.type)));
  }
  if (info.buffer == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: %s must be a constant tensor", op_name, field_name));
  }
  LockedBufferSpan<const int32_t> locked =
      info.buffer->Lock().As<const int32_t>();
  return std::vector<int32_t>(locked.begin(), locked.end());
}

// Emits a transpose of the two innermost dimensions of `value_id`.
//
// Returns the transpose output id.
absl::StatusOr<uint32_t> TransposeInnermostDims(YnnpackBuildContext& ctx,
                                                uint32_t value_id,
                                                absl::string_view op_name) {
  static constexpr int32_t kSwapLastTwo[] = {-1, -2};
  uint32_t transposed_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_transpose(
      ctx.subgraph(), /*num_axes=*/2, kSwapLastTwo, value_id, &transposed_id,
      YNN_NODE_FLAG_KEEP_DIMS))
      << op_name;
  return transposed_id;
}

int32_t ReadSignedQuantizedElement(const uint8_t* raw, ynn_type storage_type,
                                   size_t linear_idx) {
  switch (storage_type) {
    case ynn_type_int8:
      return static_cast<int32_t>(static_cast<int8_t>(raw[linear_idx]));
    case ynn_type_int4: {
      const uint8_t byte = raw[linear_idx >> 1];
      if ((linear_idx & 1) == 0) {
        return static_cast<int32_t>(static_cast<int8_t>(byte << 4) >> 4);
      }
      return static_cast<int32_t>(static_cast<int8_t>(byte) >> 4);
    }
    case ynn_type_int2: {
      const uint8_t byte = raw[linear_idx >> 2];
      const int shift = static_cast<int>(6 - 2 * (linear_idx & 3));
      return static_cast<int32_t>(static_cast<int8_t>(byte << shift) >> 6);
    }
    default:
      return 0;
  }
}

// Checks if `weights` can be consumed by `DefineQuantizedFullyConnected`.
//
// - `weights`: the tensor info of the quantized tensor.
// - `weights_are_kn`: tells whether the weights are laid out as [K, N] (e.g.
//   the right-hand side of a batch matmul without `adj_y`) rather than [N, K]
//   (e.g. the weights of a fully connected op).
bool CanUseQuantizedDot(const graph::TensorInformation& weights,
                        bool weights_are_kn) {
  if (weights.shape.size() != 2 || weights.shape[0] <= 0 ||
      weights.shape[1] <= 0 || weights.quantization == nullptr) {
    return false;
  }
  const Quantization& quantization = *weights.quantization;
  // The quantized dots consume signed integers.
  const ynn_type storage_type =
      YnnpackBuildContext::GetQuantizedStorageType(weights);
  if (storage_type != ynn_type_int2 && storage_type != ynn_type_int4 &&
      storage_type != ynn_type_int8) {
    return false;
  }
  // The dots need the weights as [K, N], with the scales along N. Per-tensor
  // quantization has no quantized dimension (-1).
  const int channel_dim = weights_are_kn ? 1 : 0;
  const int quantized_dimension =
      YnnpackBuildContext::GetQuantizedDimension(quantization);
  if (quantized_dimension >= 0 && quantized_dimension != channel_dim) {
    return false;
  }
  // Sub-byte dimensions that appear innermost before or after transposing
  // [N, K] to [K, N] must fill whole bytes.
  const size_t elems_per_byte =
      YnnpackBuildContext::ElementsPerByte(storage_type);
  if (elems_per_byte == 0) {
    return false;
  }
  if (elems_per_byte > 1 &&
      (static_cast<size_t>(weights.shape[0]) % elems_per_byte != 0 ||
       static_cast<size_t>(weights.shape[1]) % elems_per_byte != 0)) {
    return false;
  }
  // Asymmetric blockwise weights factor zero points using a host-precomputed
  // [1, N] column-sum vector, which requires static constant weight data.
  if (YnnpackBuildContext::GetBlockSize(quantization) > 0 &&
      !YnnpackBuildContext::IsSymmetric(quantization) &&
      weights.buffer == nullptr) {
    return false;
  }
  return true;
}

// Lowers a fully connected op with an fp32 `input_id` and quantized weights.
//
// We operate in the same way as YNNPACK's XNNPACK shim for dynamically
// quantized fully connected ops: the input is quantized to int8 with parameters
// computed per row, and the dot runs on the integer weights, which are static
// and therefore packed only once.
//
// - `weights`: weights tensor information.
// - `weights_data_id`: YNNPACK value holding the weights' data.
// - `bias_id`: (optional) a value of `YNN_INVALID_VALUE_ID` is ignored.
// - `result_id`: in/out, like the output of YNNPACK node definitions.
//
// Warning: `CanUseQuantizedDot(weights, weights_are_kn)` must hold.
absl::Status DefineQuantizedFullyConnected(
    YnnpackBuildContext& ctx, const graph::Tensor& input_tensor,
    uint32_t input_id, const graph::TensorInformation& weights,
    uint32_t weights_data_id, bool weights_are_kn, uint32_t bias_id,
    uint32_t& result_id, absl::string_view op_name) {
  ynn_subgraph_t sg = ctx.subgraph();
  const Quantization& quantization = *weights.quantization;

  // Quantize the input.
  static constexpr int32_t kReductionAxis[] = {-1};
  uint32_t min_max_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_reduce(
      sg, ynn_reduce_min_max, /*num_axes=*/1, kReductionAxis, input_id,
      /*input_b_id=*/YNN_INVALID_VALUE_ID, &min_max_id,
      YNN_NODE_FLAG_KEEP_DIMS))
      << op_name;
  uint32_t input_zero_point_id = kInferredValueId;
  uint32_t input_scale_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dynamic_quantization(
      sg, min_max_id, ynn_type_int8, &input_zero_point_id, &input_scale_id,
      /*flags=*/0))
      << op_name;
  uint32_t quantized_input_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_quantize(sg, input_id, ynn_type_int8, input_zero_point_id,
                          input_scale_id, &quantized_input_id, /*flags=*/0))
      << op_name;

  // The dots expect [K, N] weights.
  uint32_t weights_id = weights_data_id;
  if (!weights_are_kn) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        weights_id, TransposeInnermostDims(ctx, weights_data_id, op_name));
  }
  const size_t channels = weights.shape[weights_are_kn ? 1 : 0];
  const size_t reduction = weights.shape[weights_are_kn ? 0 : 1];
  if (channels == 0 || reduction == 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: expected positive weight dimensions, got [%d, %d]",
                        op_name, weights.shape[0], weights.shape[1]));
  }
  const absl::Span<const float> stored_scales =
      YnnpackBuildContext::GetScales(quantization);

  if (const size_t block_size = YnnpackBuildContext::GetBlockSize(quantization);
      block_size > 0) {
    const size_t num_blocks = stored_scales.size() / channels;
    // Preserve the native block size (`32`, `64`, etc.) unless `input_tensor`
    // is shared across blockwise dots with distinct multiple-of-16 block sizes
    // (such as GGUF `Q4_K_M`'s `attn_norm` feeding `Q4_K` block-32 alongside
    // `Q6_K` block-16, where `transpose_a` CSE requires matching block grids).
    const size_t eff_block_size =
        (block_size > 16 && block_size % 16 == 0 &&
         ctx.ShouldCanonicalizeToBlock16(input_tensor))
            ? 16
            : block_size;
    const size_t subblocks = block_size / eff_block_size;
    const size_t eff_num_blocks = num_blocks * subblocks;
    if (weights.buffer == nullptr) {
      // One scale per channel and block: [N, K / eff_block_size]. The scales
      // are stored channel major whatever the layout of the weights.
      std::vector<float> expanded_scales;
      absl::Span<const float> eff_scales = stored_scales;
      if (subblocks > 1) {
        expanded_scales.resize(channels * eff_num_blocks);
        for (size_t n = 0; n < channels; ++n) {
          for (size_t b = 0; b < num_blocks; ++b) {
            const float scale = stored_scales[n * num_blocks + b];
            for (size_t s = 0; s < subblocks; ++s) {
              expanded_scales[n * eff_num_blocks + b * subblocks + s] = scale;
            }
          }
        }
        eff_scales = absl::MakeConstSpan(expanded_scales);
      }
      LRT_TENSOR_ASSIGN_OR_RETURN(
          const uint32_t weights_scale_id,
          ctx.DefineConstant(eff_scales.data(),
                             eff_scales.size() * sizeof(float), Type::kFP32,
                             /*shape=*/{channels, eff_num_blocks}));
      LRT_TENSOR_RETURN_IF_ERROR(ynn::define_blockwise_dot(
          sg, quantized_input_id, input_zero_point_id, input_scale_id,
          weights_id, /*b_zero_point_id=*/YNN_INVALID_VALUE_ID,
          weights_scale_id, eff_block_size, bias_id, ynn_type_fp32, result_id))
          << op_name;
      return absl::OkStatus();
    }

    // Precompute the [1, N] effective column sums `C[n] = sum_b(s[n,b] *
    // sum_j(b[n,b,j]) + B * min[n,b])`, [eff_num_blocks, 1, N] block scales,
    // and (when asymmetric) [eff_num_blocks, 1, N] block offsets `min[n,b] =
    // -s[n,b] * zp[n,b]` on the host. Adding `sum_a * mins` to the scaled
    // per-block dots before the block reduction avoids a second FP32 GEMM over
    // a dense [eff_num_blocks, N] offset matrix (and its packed copy).
    const ynn_type storage_type =
        YnnpackBuildContext::GetQuantizedStorageType(weights);
    const bool is_symmetric = YnnpackBuildContext::IsSymmetric(quantization);
    const absl::StatusOr<const BlockwiseQuantization&> bwq =
        quantization.As<const BlockwiseQuantization>();
    const LockedBufferSpan<const uint8_t> locked =
        weights.buffer->Lock().As<const uint8_t>();
    const uint8_t* raw = locked.data();

    std::vector<float> scales_b1n(eff_num_blocks * channels);
    std::vector<float> col_sums(channels, 0.0f);
    std::vector<float> mins_b1n;
    std::vector<float> zps_float;
    if (!is_symmetric) {
      mins_b1n.resize(eff_num_blocks * channels, 0.0f);
      zps_float.resize(num_blocks * channels, 0.0f);
      if (bwq.ok()) {
        LRT_TENSOR_RETURN_IF_ERROR(
            bwq->zero_points.CopyTo(absl::MakeSpan(zps_float)));
      }
    }
    for (size_t n = 0; n < channels; ++n) {
      double col_sum_n = 0.0;
      for (size_t b = 0; b < num_blocks; ++b) {
        const size_t param_idx = n * num_blocks + b;
        const float scale = stored_scales[param_idx];
        const float zp = is_symmetric ? 0.0f : zps_float[param_idx];
        const float min_nb = is_symmetric ? 0.0f : (-scale * zp);
        for (size_t s = 0; s < subblocks; ++s) {
          const size_t eff_idx = (b * subblocks + s) * channels + n;
          scales_b1n[eff_idx] = scale;
          if (!is_symmetric) {
            mins_b1n[eff_idx] = min_nb;
          }
        }
        int64_t block_q_sum = 0;
        const size_t k_base = b * block_size;
        for (size_t j = 0; j < block_size; ++j) {
          const size_t k = k_base + j;
          const size_t linear_idx =
              weights_are_kn ? (k * channels + n) : (n * reduction + k);
          block_q_sum +=
              ReadSignedQuantizedElement(raw, storage_type, linear_idx);
        }
        col_sum_n +=
            static_cast<double>(scale) * static_cast<double>(block_q_sum) +
            static_cast<double>(block_size) * static_cast<double>(min_nb);
      }
      col_sums[n] = static_cast<float>(col_sum_n);
    }

    const size_t split_dims[2] = {0, eff_block_size};
    uint32_t a_split_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn_define_split_dim(sg, /*axis=*/-1, /*num_splits=*/2, split_dims,
                             quantized_input_id, &a_split_id, /*flags=*/0))
        << op_name;
    static constexpr int32_t kPermA[2] = {-2, -3};
    uint32_t a_inner_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn_define_static_transpose(sg, /*num_axes=*/2, kPermA, a_split_id,
                                    &a_inner_id, YNN_NODE_FLAG_KEEP_DIMS))
        << op_name;

    uint32_t b_inner_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_split_dim(
        sg, /*axis=*/-2, /*num_splits=*/2, split_dims, weights_id, &b_inner_id,
        /*flags=*/0))
        << op_name;

    uint32_t dot_blocks_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dot(
        sg, /*num_k_dims=*/1, a_inner_id, b_inner_id,
        /*input_c_id=*/YNN_INVALID_VALUE_ID, &dot_blocks_id, /*flags=*/0))
        << op_name;

    LRT_TENSOR_ASSIGN_OR_RETURN(
        const uint32_t scales_b1n_id,
        ctx.DefineConstant(scales_b1n.data(), scales_b1n.size() * sizeof(float),
                           Type::kFP32,
                           /*shape=*/{eff_num_blocks, 1, channels}));
    uint32_t scaled_blocks_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn_define_binary(sg, ynn_binary_multiply, dot_blocks_id, scales_b1n_id,
                          &scaled_blocks_id, /*flags=*/0))
        << op_name;

    if (!is_symmetric) {
      uint32_t sum_a_i32_id = kInferredValueId;
      LRT_TENSOR_RETURN_IF_ERROR(ynn_define_reduce(
          sg, ynn_reduce_sum, /*num_axes=*/1, kReductionAxis, a_inner_id,
          /*input_b_id=*/YNN_INVALID_VALUE_ID, &sum_a_i32_id,
          YNN_NODE_FLAG_KEEP_DIMS))
          << op_name;
      LRT_TENSOR_ASSIGN_OR_RETURN(
          const uint32_t mins_b1n_id,
          ctx.DefineConstant(mins_b1n.data(), mins_b1n.size() * sizeof(float),
                             Type::kFP32,
                             /*shape=*/{eff_num_blocks, 1, channels}));
      uint32_t zp_blocks_id = kInferredValueId;
      LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(
          sg, ynn_binary_multiply, sum_a_i32_id, mins_b1n_id, &zp_blocks_id,
          /*flags=*/0))
          << op_name;
      uint32_t affine_blocks_id = kInferredValueId;
      LRT_TENSOR_RETURN_IF_ERROR(
          ynn_define_binary(sg, ynn_binary_add, scaled_blocks_id, zp_blocks_id,
                            &affine_blocks_id, /*flags=*/0))
          << op_name;
      scaled_blocks_id = affine_blocks_id;
    }

    static constexpr int32_t kBlockReduceAxis[] = {-3};
    uint32_t dot_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_reduce(
        sg, ynn_reduce_sum, /*num_axes=*/1, kBlockReduceAxis, scaled_blocks_id,
        /*input_b_id=*/YNN_INVALID_VALUE_ID, &dot_id, /*flags=*/0))
        << op_name;

    LRT_TENSOR_ASSIGN_OR_RETURN(
        const uint32_t col_sums_id,
        ctx.DefineConstant(col_sums.data(), col_sums.size() * sizeof(float),
                           Type::kFP32, /*shape=*/{1, channels}));
    uint32_t zp_correction_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn_define_binary(sg, ynn_binary_multiply, input_zero_point_id,
                          col_sums_id, &zp_correction_id, /*flags=*/0))
        << op_name;
    uint32_t unscaled_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(
        sg, ynn_binary_subtract, dot_id, zp_correction_id, &unscaled_id,
        /*flags=*/0))
        << op_name;

    const bool has_bias = bias_id != YNN_INVALID_VALUE_ID;
    uint32_t scaled_id = has_bias ? kInferredValueId : result_id;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(
        sg, ynn_binary_multiply, unscaled_id, input_scale_id, &scaled_id,
        /*flags=*/0))
        << op_name;
    if (!has_bias) {
      result_id = scaled_id;
      return absl::OkStatus();
    }
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(
        sg, ynn_binary_add, scaled_id, bias_id, &result_id, /*flags=*/0))
        << op_name;
    return absl::OkStatus();
  }

  // One scale (and zero point) per channel: [1, N]. A per-tensor scale is
  // repeated for every channel.
  std::vector<float> scales(stored_scales.begin(), stored_scales.end());
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<int32_t> zero_points,
      YnnpackBuildContext::GetZeroPoints(quantization));
  if (YnnpackBuildContext::GetQuantizedDimension(quantization) < 0) {
    scales.assign(channels, stored_scales[0]);
    // Copied first: `assign` may free the storage `zero_points[0]` lives in.
    const int32_t zero_point = zero_points[0];
    zero_points.assign(channels, zero_point);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const uint32_t weights_scale_id,
      ctx.DefineConstant(scales.data(), scales.size() * sizeof(float),
                         Type::kFP32, /*shape=*/{1, channels}));
  // The zero points of asymmetric weights only add a `b_zero_point * sum(a)`
  // term, which is computed from the activations.
  uint32_t weights_zero_point_id = YNN_INVALID_VALUE_ID;
  if (!YnnpackBuildContext::IsSymmetric(quantization)) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        weights_zero_point_id,
        ctx.DefineConstant(zero_points.data(),
                           zero_points.size() * sizeof(int32_t), Type::kI32,
                           /*shape=*/{1, channels}));
  }

  // Integer dot, with the accumulator initialized to `-zero_point`.
  uint32_t dot_zero_point_id = YNN_INVALID_VALUE_ID;
  uint32_t dot_scale_id = YNN_INVALID_VALUE_ID;
  LRT_TENSOR_RETURN_IF_ERROR(ynn::define_dot_quantization(
      sg, /*num_k_dims=*/1, quantized_input_id, input_zero_point_id,
      input_scale_id, weights_id, weights_zero_point_id, weights_scale_id,
      dot_zero_point_id, dot_scale_id))
      << op_name;
  uint32_t init_id = YNN_INVALID_VALUE_ID;
  if (dot_zero_point_id != YNN_INVALID_VALUE_ID) {
    init_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_unary(sg, ynn_unary_negate,
                                                dot_zero_point_id, &init_id,
                                                /*flags=*/0))
        << op_name;
  }
  uint32_t accumulator_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dot(sg, /*num_k_dims=*/1,
                                            quantized_input_id, weights_id,
                                            init_id, &accumulator_id,
                                            /*flags=*/0))
      << op_name;

  // Back to fp32. The bias is a float, so it can't initialize the integer
  // accumulator and is added afterwards.
  const bool has_bias = bias_id != YNN_INVALID_VALUE_ID;
  uint32_t dequantized_id = has_bias ? kInferredValueId : result_id;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dequantize(
      sg, accumulator_id, /*zero_point_id=*/YNN_INVALID_VALUE_ID, dot_scale_id,
      ynn_type_fp32, &dequantized_id, /*flags=*/0))
      << op_name;
  if (!has_bias) {
    result_id = dequantized_id;
    return absl::OkStatus();
  }
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(
      sg, ynn_binary_add, dequantized_id, bias_id, &result_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

}  // namespace

absl::Status OpMixin<AddOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const AddOperation& op_data,
                              op.As<AddOperation>());
  return AddBinaryNode(ynn_binary_add, io, op_data.activation, ctx, op_name);
}

absl::Status OpMixin<MulOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const MulOperation& op_data,
                              op.As<MulOperation>());
  return AddBinaryNode(ynn_binary_multiply, io, op_data.activation, ctx,
                       op_name);
}

absl::Status OpMixin<SubOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const SubOperation& op_data,
                              op.As<SubOperation>());
  return AddBinaryNode(ynn_binary_subtract, io, op_data.activation, ctx,
                       op_name);
}

absl::Status OpMixin<DivOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const DivOperation& op_data,
                              op.As<DivOperation>());
  return AddBinaryNode(ynn_binary_divide, io, op_data.activation, ctx, op_name);
}

absl::Status OpMixin<MaximumOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  return AddBinaryNode(ynn_binary_max, io, kActNone, ctx, op_name);
}

absl::Status OpMixin<MinimumOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  return AddBinaryNode(ynn_binary_min, io, kActNone, ctx, op_name);
}

absl::Status OpMixin<PowOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  return AddBinaryNode(ynn_binary_pow, io, kActNone, ctx, op_name);
}

absl::Status OpMixin<PReluOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  return AddBinaryNode(ynn_binary_leaky_relu, io, kActNone, ctx, op_name);
}

absl::Status OpMixin<AbsOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_abs, io, ctx, op_name);
}

absl::Status OpMixin<SquareOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_square, io, ctx, op_name);
}

absl::Status OpMixin<RsqrtOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_rsqrt, io, ctx, op_name);
}

absl::Status OpMixin<SqrtOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_sqrt, io, ctx, op_name);
}

absl::Status OpMixin<ExpOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_exp, io, ctx, op_name);
}

absl::Status OpMixin<LogOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_log, io, ctx, op_name);
}

absl::Status OpMixin<CeilOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_ceil, io, ctx, op_name);
}

absl::Status OpMixin<FloorOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_floor, io, ctx, op_name);
}

absl::Status OpMixin<SignOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_sign, io, ctx, op_name);
}

absl::Status OpMixin<RoundOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_round, io, ctx, op_name);
}

absl::Status OpMixin<NegOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_negate, io, ctx, op_name);
}

absl::Status OpMixin<TanhOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_tanh, io, ctx, op_name);
}

absl::Status OpMixin<LogisticOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_sigmoid, io, ctx, op_name);
}

absl::Status OpMixin<CosOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_cos, io, ctx, op_name);
}

absl::Status OpMixin<SinOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_sin, io, ctx, op_name);
}

absl::Status OpMixin<HardSwishOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_hardswish, io, ctx, op_name);
}

absl::Status OpMixin<CastOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_convert, io, ctx, op_name);
}

absl::Status OpMixin<DequantizeOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  // Quantization parameters are attached to the value by the conversion layer,
  // which already emits the dequantize node. What is left here is a plain type
  // conversion.
  return AddUnaryNode(ynn_unary_convert, io, ctx, op_name);
}

absl::Status OpMixin<ReluOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return ApplyActivationBounds(ctx, io.input, GetActivationBounds(kActRelu),
                               io.output, op_name);
}

absl::Status OpMixin<Relu6Operation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return ApplyActivationBounds(ctx, io.input, GetActivationBounds(kActRelu6),
                               io.output, op_name);
}

absl::Status OpMixin<LeakyReluOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const LeakyReluOperation& op_data,
                              op.As<LeakyReluOperation>());
  uint32_t output_id = io.output;
  LRT_TENSOR_RETURN_IF_ERROR(ynn::define_leaky_relu(ctx.subgraph(), io.input,
                                                    op_data.alpha, output_id))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<EluOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  uint32_t output_id = io.output;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn::define_elu(ctx.subgraph(), io.input, /*alpha=*/1.0f, output_id))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<GeluOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const GeluOperation& op_data,
                              op.As<GeluOperation>());
  uint32_t output_id = io.output;
  if (op_data.approximate) {
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn::define_approx_gelu(ctx.subgraph(), io.input, output_id))
        << op_name;
  } else {
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn::define_gelu(ctx.subgraph(), io.input, output_id))
        << op_name;
  }
  return absl::OkStatus();
}

absl::Status OpMixin<SoftmaxOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const SoftmaxOperation& op_data,
                              op.As<SoftmaxOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  uint32_t output_id = io.output;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn::define_softmax(ctx.subgraph(), io.input, op_data.beta, output_id))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<L2NormalizationOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "L2Normalization is not supported in YNNPACK.");
}

absl::Status OpMixin<AveragePool2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "AveragePool2D";
  LRT_TENSOR_ASSIGN_OR_RETURN(const AveragePool2DOperation& op_data,
                              op.As<AveragePool2DOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));
  const ynn_type type = ToYnnType(input_info.type);
  if (type == ynn_type_invalid) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: unsupported input type", op_name));
  }
  const NnpackActivationBounds bounds = GetActivationBounds(op_data.activation);
  const bool needs_clamp = NeedsClamp(bounds);
  uint32_t result_id = needs_clamp ? kInferredValueId : io.output;
  LRT_TENSOR_RETURN_IF_ERROR(ynn::define_average_pool_2d(
      ctx.subgraph(), io.input, type, op_data.padding == kPaddingSame,
      op_data.filter_height, op_data.filter_width, op_data.stride_h,
      op_data.stride_w, result_id))
      << op_name;
  if (!needs_clamp) {
    return absl::OkStatus();
  }
  return ApplyActivationBounds(ctx, result_id, bounds, io.output, op_name);
}

namespace {

absl::StatusOr<uint32_t> DefineConstantOfType(YnnpackBuildContext& ctx,
                                              float value, ynn_type type,
                                              absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t fp32_id,
                              DefineScalarConstant(ctx, value));
  if (type == ynn_type_fp32) {
    return fp32_id;
  }
  uint32_t converted_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_convert(ctx.subgraph(), fp32_id, type,
                                                &converted_id, /*flags=*/0))
      << op_name;
  return converted_id;
}

// Extracts `[N, H_out, W_out, kh, kw, C_in]` sliding windows from an NHWC
// `input_id` using `ynn_define_stencil_copy`.
absl::StatusOr<uint32_t> DefineStencil2D(YnnpackBuildContext& ctx,
                                         uint32_t input_id, ynn_type input_type,
                                         Padding padding, float padding_value,
                                         int filter_h, int filter_w,
                                         int stride_h, int stride_w,
                                         int dilation_h, int dilation_w,
                                         absl::string_view op_name) {
  if (filter_h <= 0 || filter_w <= 0 || stride_h <= 0 || stride_w <= 0 ||
      dilation_h <= 0 || dilation_w <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: filter (%d, %d), stride (%d, %d), and dilation (%d, %d) must be "
        "positive",
        op_name, filter_h, filter_w, stride_h, stride_w, dilation_h,
        dilation_w));
  }
  uint32_t padding_id = YNN_INVALID_VALUE_ID;
  if (padding == kPaddingSame) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        padding_id,
        DefineConstantOfType(ctx, padding_value, input_type, op_name));
  } else if (padding != kPaddingValid) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: unsupported padding mode", op_name));
  }

  const int32_t stencil_axes[] = {1, 2};
  const int32_t new_axes[] = {3, 4};
  const size_t stencil_dims[] = {static_cast<size_t>(filter_h),
                                 static_cast<size_t>(filter_w)};
  const size_t stencil_strides[] = {static_cast<size_t>(stride_h),
                                    static_cast<size_t>(stride_w)};
  const size_t stencil_dilations[] = {static_cast<size_t>(dilation_h),
                                      static_cast<size_t>(dilation_w)};
  uint32_t stencil_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_stencil_copy(
      ctx.subgraph(), /*num_stencils=*/2, stencil_axes, new_axes, stencil_dims,
      stencil_strides, stencil_dilations, input_id, padding_id, &stencil_id,
      /*flags=*/0))
      << op_name;
  return stencil_id;
}

// Lowers standard (`groups == 1`) or grouped (`groups > 1`) 2D convolution
// where `filter_id` has shape `[C_out, kh, kw, C_in / groups]`.
absl::Status DefineConv2DImpl(YnnpackBuildContext& ctx, uint32_t input_id,
                              ynn_type input_type, ynn_type output_type,
                              uint32_t filter_id, uint32_t bias_id,
                              int filter_h, int filter_w, size_t groups,
                              size_t group_input_channels,
                              size_t group_output_channels, Padding padding,
                              int stride_h, int stride_w, int dilation_h,
                              int dilation_w, FusedActivation activation,
                              uint32_t output_id, absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(
      uint32_t stenciled_id,
      DefineStencil2D(ctx, input_id, input_type, padding, /*padding_value=*/0.f,
                      filter_h, filter_w, stride_h, stride_w, dilation_h,
                      dilation_w, op_name));

  uint32_t transposed_filter_id = kInferredValueId;
  if (groups == 1) {
    const int32_t swap_co_ci[4] = {1, 2, 3, 0};
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_transpose(
        ctx.subgraph(), /*num_dims=*/4, swap_co_ci, filter_id,
        &transposed_filter_id, /*flags=*/0))
        << op_name;
  } else {
    // `[N, H, W, kh, kw, C_in]` -> `[N, H, W, kh, kw, G, 1, C_in/G]` ->
    // `[N, H, W, G, 1, kh, kw, C_in/G]`.
    const size_t input_split[] = {groups, 1, group_input_channels};
    uint32_t split_input_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_split_dim(
        ctx.subgraph(), /*axis=*/5, /*num_splits=*/3, input_split, stenciled_id,
        &split_input_id, /*flags=*/0))
        << op_name;
    const int32_t input_perm[8] = {0, 1, 2, 5, 6, 3, 4, 7};
    uint32_t transposed_input_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_transpose(
        ctx.subgraph(), /*num_dims=*/8, input_perm, split_input_id,
        &transposed_input_id, /*flags=*/0))
        << op_name;
    stenciled_id = transposed_input_id;

    // `[C_out, kh, kw, C_in/G]` -> `[G, C_out/G, kh, kw, C_in/G]` ->
    // `[G, kh, kw, C_in/G, C_out/G]`.
    const size_t filter_split[] = {groups, group_output_channels};
    uint32_t split_filter_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_split_dim(
        ctx.subgraph(), /*axis=*/0, /*num_splits=*/2, filter_split, filter_id,
        &split_filter_id, /*flags=*/0))
        << op_name;
    const int32_t swap_co_ci[5] = {0, 2, 3, 4, 1};
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_transpose(
        ctx.subgraph(), /*num_dims=*/5, swap_co_ci, split_filter_id,
        &transposed_filter_id, /*flags=*/0))
        << op_name;

    if (bias_id != YNN_INVALID_VALUE_ID) {
      const size_t bias_split[] = {groups, 1, group_output_channels};
      uint32_t split_bias_id = kInferredValueId;
      LRT_TENSOR_RETURN_IF_ERROR(ynn_define_split_dim(
          ctx.subgraph(), /*axis=*/0, /*num_splits=*/3, bias_split, bias_id,
          &split_bias_id, /*flags=*/0))
          << op_name;
      bias_id = split_bias_id;
    }
  }

  const NnpackActivationBounds bounds = GetActivationBounds(activation);
  const bool needs_clamp = NeedsClamp(bounds);
  const bool needs_convert = output_type != ynn_type_fp32;
  const bool needs_fuse = groups != 1;

  const uint32_t conv_out_id = needs_clamp ? kInferredValueId : output_id;
  uint32_t dot_out_id =
      (needs_fuse || needs_convert) ? kInferredValueId : conv_out_id;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dot(ctx.subgraph(), /*num_k_dims=*/3,
                                            stenciled_id, transposed_filter_id,
                                            bias_id, &dot_out_id, /*flags=*/0))
      << op_name;

  uint32_t post_dot_id = dot_out_id;
  if (needs_convert) {
    uint32_t converted_id = needs_fuse ? kInferredValueId : conv_out_id;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_convert(
        ctx.subgraph(), post_dot_id, output_type, &converted_id, /*flags=*/0))
        << op_name;
    post_dot_id = converted_id;
  }
  if (needs_fuse) {
    uint32_t fused_id = conv_out_id;
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn_define_fuse_dim(ctx.subgraph(), /*first_axis=*/3, /*axes_count=*/3,
                            post_dot_id, &fused_id, /*flags=*/0))
        << op_name;
    post_dot_id = fused_id;
  }
  if (needs_clamp) {
    return ApplyActivationBounds(ctx, post_dot_id, bounds, output_id, op_name);
  }
  return absl::OkStatus();
}

struct Conv2DIO {
  const graph::TensorInformation* input_info = nullptr;
  const graph::TensorInformation* filter_info = nullptr;
  uint32_t input_id = YNN_INVALID_VALUE_ID;
  uint32_t filter_id = YNN_INVALID_VALUE_ID;
  uint32_t bias_id = YNN_INVALID_VALUE_ID;
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  ynn_type input_type = ynn_type_invalid;
  ynn_type output_type = ynn_type_invalid;
};

absl::StatusOr<Conv2DIO> PrepareConv2DIO(const graph::Operation& op,
                                         YnnpackBuildContext& ctx,
                                         absl::string_view op_name) {
  if (op.inputs.size() < 2 || op.inputs.size() > 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 2 or 3 inputs (input, filter[, bias])", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& filter_info,
                              graph::GetInfo(op.inputs[1]));
  if (input_info.shape.size() != 4 || filter_info.shape.size() != 4) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects rank-4 NHWC input and rank-4 filter", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t input_id,
                              ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t filter_id,
                              ctx.DefineValue(op.inputs[1]));
  uint32_t bias_id = YNN_INVALID_VALUE_ID;
  if (op.inputs.size() == 3 && op.inputs[2].group != nullptr) {
    LRT_TENSOR_ASSIGN_OR_RETURN(bias_id, ctx.DefineValue(op.inputs[2]));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));
  const ynn_type input_type = input_info.quantization != nullptr
                                  ? ynn_type_fp32
                                  : ToYnnType(input_info.type);
  const ynn_type output_type = ToYnnType(output_info.type);
  if (input_type == ynn_type_invalid || output_type == ynn_type_invalid) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: unsupported tensor type", op_name));
  }
  return Conv2DIO{
      .input_info = &input_info,
      .filter_info = &filter_info,
      .input_id = input_id,
      .filter_id = filter_id,
      .bias_id = bias_id,
      .output_id = output_id,
      .input_type = input_type,
      .output_type = output_type,
  };
}

}  // namespace

absl::Status OpMixin<MaxPool2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "MaxPool2D";
  LRT_TENSOR_ASSIGN_OR_RETURN(const MaxPool2DOperation& op_data,
                              op.As<MaxPool2DOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));
  if (input_info.shape.size() != 4) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects rank-4 NHWC input, got rank %d", op_name,
                        input_info.shape.size()));
  }
  const ynn_type type = ToYnnType(input_info.type);
  if (type != ynn_type_fp32 && type != ynn_type_fp16 && type != ynn_type_bf16) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: unsupported input type (expected float)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const uint32_t stencil_id,
      DefineStencil2D(ctx, io.input, type, op_data.padding,
                      -std::numeric_limits<float>::infinity(),
                      op_data.filter_height, op_data.filter_width,
                      op_data.stride_h, op_data.stride_w, /*dilation_h=*/1,
                      /*dilation_w=*/1, op_name));
  const NnpackActivationBounds bounds = GetActivationBounds(op_data.activation);
  const bool needs_clamp = NeedsClamp(bounds);
  uint32_t result_id = needs_clamp ? kInferredValueId : io.output;
  const int32_t reduce_axes[] = {3, 4};
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_reduce(
      ctx.subgraph(), ynn_reduce_max, /*num_axes=*/2, reduce_axes, stencil_id,
      /*input_b_id=*/YNN_INVALID_VALUE_ID, &result_id, /*flags=*/0))
      << op_name;
  if (!needs_clamp) {
    return absl::OkStatus();
  }
  return ApplyActivationBounds(ctx, result_id, bounds, io.output, op_name);
}

absl::Status OpMixin<Conv2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Conv2D";
  LRT_TENSOR_ASSIGN_OR_RETURN(const Conv2DOperation& op_data,
                              op.As<Conv2DOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(const Conv2DIO io,
                              PrepareConv2DIO(op, ctx, op_name));
  const int32_t input_channels = io.input_info->shape[3];
  const int32_t output_channels = io.filter_info->shape[0];
  const int32_t filter_h = io.filter_info->shape[1];
  const int32_t filter_w = io.filter_info->shape[2];
  const int32_t group_ci = io.filter_info->shape[3];
  if (input_channels <= 0 || output_channels <= 0 || group_ci <= 0 ||
      input_channels % group_ci != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: invalid channels input=%d, output=%d, filter_ci=%d", op_name,
        input_channels, output_channels, group_ci));
  }
  const int32_t groups = input_channels / group_ci;
  if (output_channels % groups != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: output channels (%d) must be divisible by groups (%d)", op_name,
        output_channels, groups));
  }

  return DefineConv2DImpl(
      ctx, io.input_id, io.input_type, io.output_type, io.filter_id, io.bias_id,
      filter_h, filter_w, static_cast<size_t>(groups),
      static_cast<size_t>(group_ci),
      static_cast<size_t>(output_channels / groups), op_data.padding,
      op_data.stride_h, op_data.stride_w, op_data.dilation_h_factor,
      op_data.dilation_w_factor, op_data.activation, io.output_id, op_name);
}

absl::Status OpMixin<DepthwiseConv2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "DepthwiseConv2D";
  LRT_TENSOR_ASSIGN_OR_RETURN(const DepthwiseConv2DOperation& op_data,
                              op.As<DepthwiseConv2DOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(const Conv2DIO io,
                              PrepareConv2DIO(op, ctx, op_name));
  if (io.filter_info->shape[0] != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects rank-4 NHWC input and [1, kh, kw, C_out] filter", op_name));
  }
  const int32_t input_channels = io.input_info->shape[3];
  const int32_t filter_h = io.filter_info->shape[1];
  const int32_t filter_w = io.filter_info->shape[2];
  const int32_t output_channels = io.filter_info->shape[3];
  if (input_channels <= 0 || output_channels <= 0 ||
      output_channels % input_channels != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: invalid channels input=%d, output=%d", op_name,
                        input_channels, output_channels));
  }
  const int32_t depth_multiplier = output_channels / input_channels;

  // `[1, kh, kw, C_in * dm]` -> `[C_in * dm, kh, kw, 1]`.
  const int32_t swap_dims[4] = {3, 1, 2, 0};
  uint32_t transposed_filter_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_transpose(
      ctx.subgraph(), /*num_dims=*/4, swap_dims, io.filter_id,
      &transposed_filter_id, /*flags=*/0))
      << op_name;

  return DefineConv2DImpl(
      ctx, io.input_id, io.input_type, io.output_type, transposed_filter_id,
      io.bias_id, filter_h, filter_w, static_cast<size_t>(input_channels),
      /*group_input_channels=*/1, static_cast<size_t>(depth_multiplier),
      op_data.padding, op_data.stride_h, op_data.stride_w,
      op_data.dilation_h_factor, op_data.dilation_w_factor, op_data.activation,
      io.output_id, op_name);
}

absl::Status OpMixin<TransposeConvOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("TransposeConv is not supported in YNNPACK.");
}

absl::Status OpMixin<TransposeConv2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "TransposeConv2D is not supported in YNNPACK.");
}

absl::Status OpMixin<FullyConnectedOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const FullyConnectedOperation& op_data,
                              op.As<FullyConnectedOperation>());
  if (op.inputs.size() < 2 || op.inputs.size() > 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 2 or 3 inputs (input, weights[, bias])", op_name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t weights_id,
                              ctx.DefineValue(op.inputs[1]));
  uint32_t bias_id = YNN_INVALID_VALUE_ID;
  if (op.inputs.size() == 3) {
    LRT_TENSOR_ASSIGN_OR_RETURN(bias_id, ctx.DefineValue(op.inputs[2]));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));

  const NnpackActivationBounds bounds = GetActivationBounds(op_data.activation);
  const bool needs_clamp = NeedsClamp(bounds);
  uint32_t result_id = needs_clamp ? kInferredValueId : output_id;

  // Quantized weights are consumed as integers by a dynamically quantized dot.
  // Going through the fp32 value `weights_id` would make YNNPACK dequantize,
  // transpose and pack the full weights on every invoke: its constant folding
  // refuses to fold nodes that grow their static inputs by 2x or more, which an
  // int4 -> fp32 dequantize does.
  const uint32_t weights_data_id = ctx.FindQuantizedDataId(op.inputs[1]);
  absl::StatusOr<const TensorInformation&> input_info = GetInfo(op.inputs[0]);
  absl::StatusOr<const TensorInformation&> weights_info = GetInfo(op.inputs[1]);
  if (weights_data_id != YNN_INVALID_VALUE_ID && input_info.ok() &&
      weights_info.ok() && input_info->type == Type::kFP32 &&
      input_info->shape.size() >= 2 &&
      CanUseQuantizedDot(*weights_info, /*weights_are_kn=*/false)) {
    LRT_TENSOR_RETURN_IF_ERROR(DefineQuantizedFullyConnected(
        ctx, op.inputs[0], input_id, *weights_info, weights_data_id,
        /*weights_are_kn=*/false, bias_id, result_id, op_name));
  } else {
    // `ynn_define_dot` expects `b` with shape [..., K, N], while
    // FullyConnected stores the weights as [N, K].
    LRT_TENSOR_ASSIGN_OR_RETURN(
        const uint32_t transposed_weights_id,
        TransposeInnermostDims(ctx, weights_id, op_name));
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dot(ctx.subgraph(), /*num_k_dims=*/1,
                                              input_id, transposed_weights_id,
                                              bias_id, &result_id, /*flags=*/0))
        << op_name;
  }
  if (!needs_clamp) {
    return absl::OkStatus();
  }
  return ApplyActivationBounds(ctx, result_id, bounds, output_id, op_name);
}

absl::Status OpMixin<BatchMatMulOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "BatchMatMul";
  LRT_TENSOR_ASSIGN_OR_RETURN(const BatchMatMulOperation& op_data,
                              op.As<BatchMatMulOperation>());
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs", op_name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t lhs_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t rhs_id, ctx.DefineValue(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));

  const uint32_t rhs_data_id = ctx.FindQuantizedDataId(op.inputs[1]);
  absl::StatusOr<const TensorInformation&> lhs_info = GetInfo(op.inputs[0]);
  absl::StatusOr<const TensorInformation&> rhs_info = GetInfo(op.inputs[1]);
  if (rhs_data_id != YNN_INVALID_VALUE_ID && !op_data.adj_x && lhs_info.ok() &&
      rhs_info.ok() && lhs_info->type == Type::kFP32 &&
      lhs_info->shape.size() >= 2 &&
      CanUseQuantizedDot(*rhs_info, /*weights_are_kn=*/!op_data.adj_y)) {
    return DefineQuantizedFullyConnected(
        ctx, op.inputs[0], lhs_id, *rhs_info, rhs_data_id,
        /*weights_are_kn=*/!op_data.adj_y,
        /*bias_id=*/YNN_INVALID_VALUE_ID, output_id, op_name);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));
  // Only the batch dimensions broadcast: the matrix dimensions of the operands
  // don't line up with the output's.
  std::vector<int32_t> batch_shape(output_info.shape.begin(),
                                   output_info.shape.end());
  std::fill(batch_shape.end() - std::min<size_t>(2, batch_shape.size()),
            batch_shape.end(), kKeepDim);
  LRT_TENSOR_ASSIGN_OR_RETURN(
      lhs_id,
      DefineBroadcastOperand(ctx, lhs_id, op.inputs[0], batch_shape, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      rhs_id,
      DefineBroadcastOperand(ctx, rhs_id, op.inputs[1], batch_shape, op_name));

  // `ynn_define_dot` computes `a[..., M, K] . b[..., K, N]`.
  if (op_data.adj_x) {
    LRT_TENSOR_ASSIGN_OR_RETURN(lhs_id,
                                TransposeInnermostDims(ctx, lhs_id, op_name));
  }
  if (op_data.adj_y) {
    LRT_TENSOR_ASSIGN_OR_RETURN(rhs_id,
                                TransposeInnermostDims(ctx, rhs_id, op_name));
  }

  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dot(
      ctx.subgraph(), /*num_k_dims=*/1, lhs_id, rhs_id,
      /*input_c_id=*/YNN_INVALID_VALUE_ID, &output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<TransposeOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Transpose";
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, perm)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> perm,
      GetConstantInt32Vector(op.inputs[1], op_name, "permutation"));

  // The Tensor API keeps the quantization of a transposed tensor, with the
  // quantized dimension following the permutation. When the input has an
  // integer form, the transpose moves the integers, and the output gets an
  // integer form too so that a dot consuming it can still use them.
  if (const uint32_t input_data_id = ctx.FindQuantizedDataId(op.inputs[0]);
      input_data_id != YNN_INVALID_VALUE_ID) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                                graph::GetInfo(op.inputs[0]));
    LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                                graph::GetOutputs(op));
    LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                                graph::GetInfo(outputs.front()));
    if (output_info.quantization == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s must maintain quantization: output isn't quantized.", op_name));
    }
    const ynn_type storage_type =
        YnnpackBuildContext::GetQuantizedStorageType(input_info);
    if (YnnpackBuildContext::GetQuantizedStorageType(output_info) !=
        storage_type) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s must maintain quantization: output has a different storage type.",
          op_name));
    }
    // Sub-byte integers are packed along the innermost dimension, so YNNPACK
    // requires the transposed innermost dimension to fill whole bytes. Other
    // shapes are transposed as fp32.
    const size_t elements_per_byte =
        std::max<size_t>(1, YnnpackBuildContext::ElementsPerByte(storage_type));
    if (!output_info.shape.empty() &&
        output_info.shape.back() % elements_per_byte == 0) {
      uint32_t output_data_id = kInferredValueId;
      LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_transpose(
          ctx.subgraph(), perm.size(), perm.data(), input_data_id,
          &output_data_id, /*flags=*/0))
          << op_name;
      return ctx
          .DefineDequantizedValue(outputs.front(), output_data_id, output_id)
          .status();
    }
  }

  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_static_transpose(ctx.subgraph(), perm.size(), perm.data(),
                                  input_id, &output_id, /*flags=*/0));
  return absl::OkStatus();
}

absl::Status OpMixin<MeanOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Mean";
  LRT_TENSOR_ASSIGN_OR_RETURN(const MeanOperation& op_data,
                              op.As<MeanOperation>());
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, axes)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              ctx.DefineValue(outputs.front()));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> axes,
      GetConstantInt32Vector(op.inputs[1], op_name, "axes"));

  const ynn_type output_type = ToYnnType(output_info.type);
  if (output_type == ynn_type_invalid) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: unsupported output type", op_name));
  }
  // Note: we don't currently support emitting a quantized mean.
  LRT_TENSOR_RETURN_IF_ERROR(ynn::define_reduce_sum(
      ctx.subgraph(), axes.size(), axes.data(), input_id,
      /*input_zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*input_scale_id=*/YNN_INVALID_VALUE_ID, op_data.keep_dims,
      /*mean=*/true, /*squared=*/false, output_type,
      /*output_zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*output_scale_id=*/YNN_INVALID_VALUE_ID, output_id))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<SliceOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Slice";
  if (op.inputs.size() != 3) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 3 inputs (input, begin, size)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 output", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> begins,
      GetConstantInt32Vector(op.inputs[1], op_name, "begin"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> sizes,
      GetConstantInt32Vector(op.inputs[2], op_name, "size"));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));

  if (begins.size() != sizes.size() ||
      begins.size() != input_info.shape.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: begin (%d) and size (%d) must have one entry per input dimension "
        "(%d)",
        op_name, begins.size(), sizes.size(), input_info.shape.size()));
  }

  const size_t num_axes = begins.size();
  const bool input_has_static_shape = ctx.HasStaticShape(op.inputs[0]);
  std::vector<int32_t> axes;
  std::vector<int64_t> slice_begins;
  std::vector<int64_t> slice_ends;
  axes.reserve(num_axes);
  slice_begins.reserve(num_axes);
  slice_ends.reserve(num_axes);
  for (size_t i = 0; i < num_axes; ++i) {
    if (begins[i] == 0 && (sizes[i] < 0 || (input_has_static_shape &&
                                            sizes[i] == input_info.shape[i]))) {
      continue;
    }
    axes.push_back(static_cast<int32_t>(i));
    slice_begins.push_back(begins[i]);
    if (sizes[i] < 0) {
      // A negative size means "up to the end of this dimension". Spelling that
      // out as the extent the graph happens to have been built with would
      // freeze the dimension, which breaks graphs that are re-run with a
      // different shape, e.g. attention over a KV cache that grows by one
      // token per decode step. YNNPACK interprets an end of 0 or less as
      // relative to the (possibly symbolic) extent, so 0 keeps it dynamic.
      slice_ends.push_back(0);
    } else {
      slice_ends.push_back(static_cast<int64_t>(begins[i]) + sizes[i]);
    }
  }

  if (axes.empty()) {
    if (!ctx.IsExternalOutput(outputs.front()) &&
        !ctx.IsDefined(outputs.front())) {
      return ctx.AliasValue(outputs.front(), op.inputs[0]);
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                                ctx.DefineValue(outputs.front()));
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn_define_copy(ctx.subgraph(), input_id, &output_id, /*flags=*/0))
        << op_name;
    return absl::OkStatus();
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              ctx.DefineValue(outputs.front()));
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_slice(
      ctx.subgraph(), axes.size(), axes.data(), slice_begins.data(),
      slice_ends.data(), /*strides=*/nullptr, input_id, &output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<ConcatenationOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Concatenation";
  LRT_TENSOR_ASSIGN_OR_RETURN(const ConcatenationOperation& op_data,
                              op.As<ConcatenationOperation>());
  if (op.inputs.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects at least 1 input", op_name));
  }

  std::vector<uint32_t> input_ids;
  input_ids.reserve(op.inputs.size());
  for (const graph::Tensor& input : op.inputs) {
    LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));
    input_ids.push_back(input_id);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));

  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_concatenate(ctx.subgraph(), op_data.axis, input_ids.size(),
                             input_ids.data(), &output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<SqueezeOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Squeeze";
  LRT_TENSOR_ASSIGN_OR_RETURN(const SqueezeOperation& op_data,
                              op.As<SqueezeOperation>());
  if (op.inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 input", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));

  // Squeezing is a transpose that drops the squeezed dimensions. Unlike a
  // reshape, it doesn't need the full output shape, so the dimensions that are
  // kept stay dynamic.
  const int rank = static_cast<int>(input_info.shape.size());
  std::vector<bool> squeezed(rank, false);
  if (op_data.squeeze_dims.empty()) {
    for (int i = 0; i < rank; ++i) {
      squeezed[i] = input_info.shape[i] == 1;
    }
  } else {
    for (int dim : op_data.squeeze_dims) {
      const int axis = dim < 0 ? dim + rank : dim;
      if (axis < 0 || axis >= rank) {
        return absl::InvalidArgumentError(
            absl::StrFormat("%s: dimension %d is out of range for rank %d",
                            op_name, dim, rank));
      }
      if (input_info.shape[axis] != 1) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "%s: dimension %d has extent %d, only extent 1 dimensions can be "
            "squeezed",
            op_name, dim, input_info.shape[axis]));
      }
      squeezed[axis] = true;
    }
  }

  std::vector<int32_t> squeezed_axes;
  std::vector<int32_t> kept_axes;
  for (int i = 0; i < rank; ++i) {
    (squeezed[i] ? squeezed_axes : kept_axes).push_back(i);
  }

  // The transpose doesn't check the extent of the dimensions it drops, so they
  // are turned into broadcast dimensions first, which checks that their extent
  // is still 1 when the runtime is reshaped.
  uint32_t broadcast_id = input_id;
  if (!squeezed_axes.empty()) {
    broadcast_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_broadcast(
        ctx.subgraph(), squeezed_axes.size(), squeezed_axes.data(), input_id,
        &broadcast_id, /*flags=*/0))
        << op_name;
  }
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_transpose(
      ctx.subgraph(), kept_axes.size(),
      kept_axes.empty() ? nullptr : kept_axes.data(), broadcast_id, &output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<ExpandDimsOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "ExpandDims";
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> axes,
      GetConstantInt32Vector(op.inputs[1], op_name, "axis"));
  if (axes.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis tensor must not be empty", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));

  const int rank = static_cast<int>(input_info.shape.size());
  int axis = axes[0];
  if (axis < 0) {
    axis += rank + 1;
  }
  if (axis < 0 || axis > rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis %d is out of range for input of rank %d",
                        op_name, axes[0], rank));
  }

  const int32_t new_axis = axis;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
      ctx.subgraph(), /*num_new_axes=*/1, &new_axis, input_id, &output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<ReshapeOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Reshape";
  LRT_TENSOR_ASSIGN_OR_RETURN(const ReshapeOperation& op_data,
                              op.As<ReshapeOperation>());
  if (op.inputs.size() != 1 && op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 or 2 inputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));

  std::vector<size_t> new_shape;
  new_shape.reserve(op_data.new_shape.size());
  for (const int dim : op_data.new_shape) {
    new_shape.push_back(dim < 0 ? 0 : static_cast<size_t>(dim));
  }
  if (op_data.inferred_axis >= 0) {
    if (op_data.inferred_axis >= new_shape.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s: inferred axis (%v) is out of range for shape of rank %v",
          op_name, op_data.inferred_axis, new_shape.size()));
    }
    new_shape[op_data.inferred_axis] = 0;
  }
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_reshape(
      ctx.subgraph(), new_shape.size(),
      new_shape.empty() ? nullptr : new_shape.data(), input_id, &output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<TileOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Tile";
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, multiples)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> multiples,
      GetConstantInt32Vector(op.inputs[1], op_name, "multiples"));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));

  if (input_info.shape.size() != multiples.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: input rank (%d) != multiples size (%d)", op_name,
                        input_info.shape.size(), multiples.size()));
  }

  // `ynn_define_static_broadcast` treats a zero entry in `new_dims` as "pass
  // this dimension through unchanged". We must use that for untiled
  // dimensions rather than spelling out their current extent: an extent that
  // is only known symbolically (e.g. a KV cache sequence length that grows on
  // every decode step) would otherwise be frozen at graph-build time and the
  // broadcast would reject the tensor on the next reshape.
  std::vector<size_t> new_shape(multiples.size());
  for (size_t i = 0; i < multiples.size(); ++i) {
    const int dim = input_info.shape[i];
    if (multiples[i] > 1 && dim != 1) {
      return absl::UnimplementedError(absl::StrFormat(
          "%s: YNNPACK broadcast can only tile dimensions of size 1. "
          "Dimension %d has size %d but multiples[%d]=%d",
          op_name, i, dim, i, multiples[i]));
    }
    new_shape[i] =
        multiples[i] == 1 ? 0 : static_cast<size_t>(dim * multiples[i]);
  }

  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_broadcast(
      ctx.subgraph(), new_shape.size(), new_shape.data(), input_id, &output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<SplitOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Split";
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const SplitOperation& op_data,
                              op.As<SplitOperation>());
  if (outputs.size() != op_data.num_splits) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects %d outputs, but got %d", op_name,
                        op_data.num_splits, outputs.size()));
  }

  std::vector<uint32_t> output_ids(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    LRT_TENSOR_ASSIGN_OR_RETURN(output_ids[i], ctx.DefineValue(outputs[i]));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> axes,
      GetConstantInt32Vector(op.inputs[1], op_name, "axis"));
  if (axes.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis tensor must not be empty", op_name));
  }

  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_even_split(ctx.subgraph(), axes[0], input_id,
                            output_ids.size(), output_ids.data(), /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

namespace {

// Emits a YNNPACK gather of `input_id` (rank `input_rank`) along `axis`
// (`0 <= axis < input_rank`) using `indices_id` (rank `indices_rank`) and
// `batch_dims`, aligning multi-rank indices and trailing slice dimensions.
absl::Status DefineGatherValues(ynn_subgraph_t sg, uint32_t input_id,
                                int input_rank, uint32_t indices_id,
                                int indices_rank, int axis, int batch_dims,
                                uint32_t& output_id,
                                absl::string_view op_name) {
  const int b_g = axis - batch_dims;
  const int s = input_rank - 1 - axis;
  const int k_g = indices_rank - batch_dims;
  const bool gather_is_scalar = (k_g == 0);

  uint32_t final_input_id = input_id;
  if (k_g > 1) {
    std::vector<int32_t> new_input_axes(k_g - 1);
    std::iota(new_input_axes.begin(), new_input_axes.end(), axis + 1);
    uint32_t expanded_input_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
        sg, new_input_axes.size(), new_input_axes.data(), input_id,
        &expanded_input_id, /*flags=*/0))
        << op_name;
    final_input_id = expanded_input_id;
  }

  uint32_t final_indices_id = indices_id;
  if (b_g > 0 || s > 0 || gather_is_scalar) {
    std::vector<int32_t> new_axes;
    if (!gather_is_scalar) {
      new_axes.reserve(b_g + s);
      for (int i = 0; i < b_g; ++i) {
        new_axes.push_back(batch_dims + i);
      }
      for (int i = 0; i < s; ++i) {
        new_axes.push_back(batch_dims + b_g + k_g + i);
      }
    } else {
      new_axes.resize(b_g + 1 + s);
      std::iota(new_axes.begin(), new_axes.end(), batch_dims);
    }
    uint32_t expanded_indices_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
        sg, new_axes.size(), new_axes.data(), indices_id, &expanded_indices_id,
        /*flags=*/0))
        << op_name;
    final_indices_id = expanded_indices_id;
  }

  uint32_t gather_output_id = gather_is_scalar ? kInferredValueId : output_id;
  const size_t gather_output_rank =
      gather_is_scalar
          ? static_cast<size_t>(input_rank)
          : static_cast<size_t>(input_rank + indices_rank - batch_dims - 1);
  const int32_t gather_axis = axis;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_gather(
      sg, /*num_axes=*/1, &gather_axis, gather_output_rank, final_input_id,
      final_indices_id, &gather_output_id, /*flags=*/0))
      << op_name;

  if (gather_is_scalar) {
    const int32_t slice_axes[] = {axis};
    const int64_t slice_begins[] = {0};
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_slice(
        sg, /*num_axes=*/1, slice_axes, slice_begins, /*ends=*/nullptr,
        /*strides=*/nullptr, gather_output_id, &output_id,
        YNN_NODE_FLAG_SLICE_DIMS))
        << op_name;
  } else {
    output_id = gather_output_id;
  }
  return absl::OkStatus();
}

bool IsFloatOutputType(ynn_type type) {
  return type == ynn_type_fp32 || type == ynn_type_fp16 ||
         type == ynn_type_bf16;
}

bool CanUseQuantizedRowGather(const graph::TensorInformation& table_info) {
  if (table_info.shape.size() != 2 || table_info.quantization == nullptr) {
    return false;
  }
  const ynn_type storage_type =
      YnnpackBuildContext::GetQuantizedStorageType(table_info);
  const size_t elems_per_byte =
      YnnpackBuildContext::ElementsPerByte(storage_type);
  if (elems_per_byte == 0) {
    return false;
  }
  const size_t row_width = static_cast<size_t>(table_info.shape[1]);
  if (row_width == 0 || row_width % elems_per_byte != 0) {
    return false;
  }
  const Quantization& q = *table_info.quantization;
  const size_t block_size = YnnpackBuildContext::GetBlockSize(q);
  const int qdim = YnnpackBuildContext::GetQuantizedDimension(q);
  if (block_size > 0) {
    if (qdim != 0 || row_width % block_size != 0) {
      return false;
    }
    if (elems_per_byte > 1 && block_size % elems_per_byte != 0) {
      return false;
    }
  }
  return true;
}

// Gathers rows along axis 0 of a 2-D quantized `[V, D]` table in packed
// integer form and dequantizes only the `[indices..., D]` output slice.
absl::Status DefineQuantizedRowGather(
    YnnpackBuildContext& ctx, const graph::TensorInformation& table_info,
    uint32_t table_data_id, uint32_t indices_id, int indices_rank,
    ynn_type output_type, uint32_t& output_id, absl::string_view op_name) {
  LRT_TENSOR_RETURN_IF_ERROR(
      YnnpackBuildContext::ValidateQuantization(table_info));
  ynn_subgraph_t sg = ctx.subgraph();
  const Quantization& q = *table_info.quantization;
  const size_t vocab_size = static_cast<size_t>(table_info.shape[0]);
  const size_t row_width = static_cast<size_t>(table_info.shape[1]);

  uint32_t gathered_data_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(DefineGatherValues(
      sg, table_data_id, /*input_rank=*/2, indices_id, indices_rank,
      /*axis=*/0, /*batch_dims=*/0, gathered_data_id, op_name));

  uint32_t fp32_out_id =
      output_type == ynn_type_fp32 ? output_id : kInferredValueId;

  if (const size_t block_size = YnnpackBuildContext::GetBlockSize(q);
      block_size > 0) {
    const size_t num_blocks = row_width / block_size;
    const size_t splits[] = {num_blocks, block_size};
    uint32_t split_data_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn_define_split_dim(sg, /*axis=*/-1, /*num_splits=*/2, splits,
                             gathered_data_id, &split_data_id, /*flags=*/0))
        << op_name;

    const absl::Span<const float> scales = YnnpackBuildContext::GetScales(q);
    LRT_TENSOR_ASSIGN_OR_RETURN(
        const uint32_t table_scale_id,
        ctx.DefineConstant(scales.data(), scales.size() * sizeof(float),
                           Type::kFP32, /*shape=*/{vocab_size, num_blocks}));
    uint32_t gathered_scale_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(DefineGatherValues(
        sg, table_scale_id, /*input_rank=*/2, indices_id, indices_rank,
        /*axis=*/0, /*batch_dims=*/0, gathered_scale_id, op_name));
    const int32_t expand_axis = indices_rank + 1;
    uint32_t scale_exp_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
        sg, /*num_new_axes=*/1, &expand_axis, gathered_scale_id, &scale_exp_id,
        /*flags=*/0))
        << op_name;

    uint32_t deq_split_id = kInferredValueId;
    if (absl::StatusOr<const BlockwiseQuantization&> bwq =
            q.As<const BlockwiseQuantization>();
        bwq.ok() && !bwq->zero_points.IsSymmetric() &&
        bwq->zero_points.Data<float>().ok()) {
      std::vector<float> mins(scales.size());
      LRT_TENSOR_RETURN_IF_ERROR(
          bwq->zero_points.Visit(scales.size(), [&](auto zp_view) {
            for (size_t i = 0; i < scales.size(); ++i) {
              mins[i] = -scales[i] * static_cast<float>(zp_view[i]);
            }
          }));
      LRT_TENSOR_ASSIGN_OR_RETURN(
          const uint32_t table_min_id,
          ctx.DefineConstant(mins.data(), mins.size() * sizeof(float),
                             Type::kFP32, /*shape=*/{vocab_size, num_blocks}));
      uint32_t gathered_min_id = kInferredValueId;
      LRT_TENSOR_RETURN_IF_ERROR(DefineGatherValues(
          sg, table_min_id, /*input_rank=*/2, indices_id, indices_rank,
          /*axis=*/0, /*batch_dims=*/0, gathered_min_id, op_name));
      uint32_t min_exp_id = kInferredValueId;
      LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
          sg, /*num_new_axes=*/1, &expand_axis, gathered_min_id, &min_exp_id,
          /*flags=*/0))
          << op_name;
      uint32_t scaled_split_id = kInferredValueId;
      LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dequantize(
          sg, split_data_id, /*zero_point_id=*/YNN_INVALID_VALUE_ID,
          scale_exp_id, ynn_type_fp32, &scaled_split_id, /*flags=*/0))
          << op_name;
      LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(
          sg, ynn_binary_add, scaled_split_id, min_exp_id, &deq_split_id,
          /*flags=*/0))
          << op_name;
    } else {
      uint32_t zp_exp_id = YNN_INVALID_VALUE_ID;
      if (!YnnpackBuildContext::IsSymmetric(q)) {
        LRT_TENSOR_ASSIGN_OR_RETURN(const std::vector<int32_t> zps,
                                    YnnpackBuildContext::GetZeroPoints(q));
        LRT_TENSOR_ASSIGN_OR_RETURN(
            const uint32_t table_zp_id,
            ctx.DefineConstant(zps.data(), zps.size() * sizeof(int32_t),
                               Type::kI32, /*shape=*/{vocab_size, num_blocks}));
        uint32_t gathered_zp_id = kInferredValueId;
        LRT_TENSOR_RETURN_IF_ERROR(DefineGatherValues(
            sg, table_zp_id, /*input_rank=*/2, indices_id, indices_rank,
            /*axis=*/0, /*batch_dims=*/0, gathered_zp_id, op_name));
        zp_exp_id = kInferredValueId;
        LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
            sg, /*num_new_axes=*/1, &expand_axis, gathered_zp_id, &zp_exp_id,
            /*flags=*/0))
            << op_name;
      }
      LRT_TENSOR_RETURN_IF_ERROR(
          ynn_define_dequantize(sg, split_data_id, zp_exp_id, scale_exp_id,
                                ynn_type_fp32, &deq_split_id, /*flags=*/0))
          << op_name;
    }

    LRT_TENSOR_RETURN_IF_ERROR(
        ynn_define_fuse_dim(sg, /*first_axis=*/indices_rank, /*axes_count=*/2,
                            deq_split_id, &fp32_out_id, /*flags=*/0))
        << op_name;
  } else {
    const absl::Span<const float> scales = YnnpackBuildContext::GetScales(q);
    LRT_TENSOR_ASSIGN_OR_RETURN(const std::vector<int32_t> zps,
                                YnnpackBuildContext::GetZeroPoints(q));
    const int qdim = YnnpackBuildContext::GetQuantizedDimension(q);
    const bool is_symmetric = YnnpackBuildContext::IsSymmetric(q);
    uint32_t scale_id = YNN_INVALID_VALUE_ID;
    uint32_t zp_id = YNN_INVALID_VALUE_ID;

    if (qdim < 0) {
      LRT_TENSOR_ASSIGN_OR_RETURN(
          scale_id,
          ctx.DefineConstant(&scales[0], sizeof(float), Type::kFP32, {}));
      if (!is_symmetric) {
        LRT_TENSOR_ASSIGN_OR_RETURN(
            zp_id,
            ctx.DefineConstant(&zps[0], sizeof(int32_t), Type::kI32, {}));
      }
    } else if (qdim == 0) {
      LRT_TENSOR_ASSIGN_OR_RETURN(
          const uint32_t table_scale_id,
          ctx.DefineConstant(scales.data(), scales.size() * sizeof(float),
                             Type::kFP32, /*shape=*/{vocab_size}));
      uint32_t gathered_scale_id = kInferredValueId;
      LRT_TENSOR_RETURN_IF_ERROR(DefineGatherValues(
          sg, table_scale_id, /*input_rank=*/1, indices_id, indices_rank,
          /*axis=*/0, /*batch_dims=*/0, gathered_scale_id, op_name));
      const int32_t expand_axis = indices_rank;
      scale_id = kInferredValueId;
      LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
          sg, /*num_new_axes=*/1, &expand_axis, gathered_scale_id, &scale_id,
          /*flags=*/0))
          << op_name;
      if (!is_symmetric) {
        LRT_TENSOR_ASSIGN_OR_RETURN(
            const uint32_t table_zp_id,
            ctx.DefineConstant(zps.data(), zps.size() * sizeof(int32_t),
                               Type::kI32, /*shape=*/{vocab_size}));
        uint32_t gathered_zp_id = kInferredValueId;
        LRT_TENSOR_RETURN_IF_ERROR(DefineGatherValues(
            sg, table_zp_id, /*input_rank=*/1, indices_id, indices_rank,
            /*axis=*/0, /*batch_dims=*/0, gathered_zp_id, op_name));
        zp_id = kInferredValueId;
        LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
            sg, /*num_new_axes=*/1, &expand_axis, gathered_zp_id, &zp_id,
            /*flags=*/0))
            << op_name;
      }
    } else {
      std::vector<size_t> param_shape(static_cast<size_t>(indices_rank) + 1, 1);
      param_shape.back() = row_width;
      LRT_TENSOR_ASSIGN_OR_RETURN(
          scale_id,
          ctx.DefineConstant(scales.data(), scales.size() * sizeof(float),
                             Type::kFP32, param_shape));
      if (!is_symmetric) {
        LRT_TENSOR_ASSIGN_OR_RETURN(
            zp_id, ctx.DefineConstant(zps.data(), zps.size() * sizeof(int32_t),
                                      Type::kI32, param_shape));
      }
    }

    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dequantize(
        sg, gathered_data_id, zp_id, scale_id, ynn_type_fp32, &fp32_out_id,
        /*flags=*/0))
        << op_name;
  }

  if (output_type != ynn_type_fp32) {
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_convert(sg, fp32_out_id, output_type,
                                                  &output_id, /*flags=*/0))
        << op_name;
  } else {
    output_id = fp32_out_id;
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status OpMixin<GatherOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Gather";
  LRT_TENSOR_ASSIGN_OR_RETURN(const GatherOperation& op_data,
                              op.As<GatherOperation>());
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, indices)", op_name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t index_id, ctx.DefineValue(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              ctx.DefineValue(outputs.front()));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& indices_info,
                              graph::GetInfo(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));

  const int input_rank = static_cast<int>(input_info.shape.size());
  const int indices_rank = static_cast<int>(indices_info.shape.size());
  int axis = op_data.axis;
  if (axis < 0) {
    axis += input_rank;
  }
  if (axis < 0 || axis >= input_rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis %d is out of range for rank %d", op_name,
                        op_data.axis, input_rank));
  }
  int batch_dims = op_data.batch_dims;
  if (batch_dims < 0) {
    batch_dims += indices_rank;
  }
  if (batch_dims < 0 || batch_dims > axis || batch_dims > indices_rank) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: batch_dims %d is out of range", op_name, op_data.batch_dims));
  }

  const ynn_type output_type = ToYnnType(output_info.type);
  // Peephole `Gather(Dequantize(table), indices, axis=0)`: gather the packed
  // quantized rows first and dequantize only `[indices..., D]` so YNNPACK
  // prunes the unused full-table `Dequantize(table)` node.
  if (axis == 0 && batch_dims == 0 && IsFloatOutputType(output_type)) {
    if (absl::StatusOr<std::shared_ptr<graph::Operation>> producer =
            graph::GetProducer(op.inputs[0]);
        producer.ok() && *producer != nullptr &&
        (*producer)->IsA(internal::TypeId::Get<DequantizeOperation>()) &&
        (*producer)->inputs.size() == 1) {
      const graph::Tensor& q_table = (*producer)->inputs[0];
      const uint32_t q_data_id = ctx.FindQuantizedDataId(q_table);
      if (absl::StatusOr<const graph::TensorInformation&> q_info =
              graph::GetInfo(q_table);
          q_data_id != YNN_INVALID_VALUE_ID && q_info.ok() &&
          CanUseQuantizedRowGather(*q_info)) {
        return DefineQuantizedRowGather(ctx, *q_info, q_data_id, index_id,
                                        indices_rank, output_type, output_id,
                                        op_name);
      }
    }
  }

  // Raw integer gather when `Gather(quantized_table, indices, ...)` produces an
  // unquantized integer tensor of the storage type.
  if (const uint32_t data_id = ctx.FindQuantizedDataId(op.inputs[0]);
      data_id != YNN_INVALID_VALUE_ID && output_info.quantization == nullptr &&
      output_type == YnnpackBuildContext::GetQuantizedStorageType(input_info)) {
    const size_t elems_per_byte =
        std::max<size_t>(1, YnnpackBuildContext::ElementsPerByte(output_type));
    if (input_info.shape.empty() ||
        input_info.shape.back() % elems_per_byte != 0 ||
        (elems_per_byte > 1 &&
         (axis >= input_rank - 1 || output_info.shape.empty() ||
          output_info.shape.back() % elems_per_byte != 0))) {
      return absl::UnimplementedError(absl::StrFormat(
          "%s: raw sub-byte gather requires axis < input_rank - 1, non-scalar "
          "output, and byte-aligned innermost dimensions",
          op_name));
    }
    return DefineGatherValues(ctx.subgraph(), data_id, input_rank, index_id,
                              indices_rank, axis, batch_dims, output_id,
                              op_name);
  }

  return DefineGatherValues(ctx.subgraph(), input_id, input_rank, index_id,
                            indices_rank, axis, batch_dims, output_id, op_name);
}

absl::Status OpMixin<EmbeddingLookupOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "EmbeddingLookup";
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (ids, value)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t ids_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t value_id, ctx.DefineValue(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              ctx.DefineValue(outputs.front()));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& ids_info,
                              graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& value_info,
                              graph::GetInfo(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));
  if (value_info.shape.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: value must have rank >= 1", op_name));
  }
  const ynn_type output_type = ToYnnType(output_info.type);
  if (output_type == ynn_type_invalid) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: unsupported output type", op_name));
  }
  const int indices_rank = static_cast<int>(ids_info.shape.size());

  const uint32_t value_data_id = ctx.FindQuantizedDataId(op.inputs[1]);
  if (value_data_id != YNN_INVALID_VALUE_ID && IsFloatOutputType(output_type) &&
      CanUseQuantizedRowGather(value_info)) {
    return DefineQuantizedRowGather(ctx, value_info, value_data_id, ids_id,
                                    indices_rank, output_type, output_id,
                                    op_name);
  }

  const ynn_type gathered_type = value_info.quantization != nullptr
                                     ? ynn_type_fp32
                                     : ToYnnType(value_info.type);
  uint32_t gather_out_id =
      gathered_type == output_type ? output_id : kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(DefineGatherValues(
      ctx.subgraph(), value_id, static_cast<int>(value_info.shape.size()),
      ids_id, indices_rank, /*axis=*/0, /*batch_dims=*/0, gather_out_id,
      op_name));
  if (gathered_type != output_type) {
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_convert(
        ctx.subgraph(), gather_out_id, output_type, &output_id, /*flags=*/0))
        << op_name;
  }
  return absl::OkStatus();
}

absl::Status OpMixin<SpaceToDepthOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("SpaceToDepth is not supported in YNNPACK.");
}

absl::Status OpMixin<DepthToSpaceOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("DepthToSpace is not supported in YNNPACK.");
}

absl::Status OpMixin<ResizeBilinearOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "ResizeBilinear is not supported in YNNPACK.");
}

absl::Status
OpMixin<ResizeNearestNeighborOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "ResizeNearestNeighbor is not supported in YNNPACK.");
}

absl::Status OpMixin<RopeOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("Rope is not supported in YNNPACK.");
}

namespace {

absl::StatusOr<uint32_t> ExpandLeadingDimsToRank(YnnpackBuildContext& ctx,
                                                 uint32_t value_id,
                                                 int current_rank,
                                                 int target_rank,
                                                 absl::string_view op_name) {
  if (current_rank >= target_rank) {
    return value_id;
  }
  const int diff = target_rank - current_rank;
  std::vector<int32_t> new_axes(diff);
  std::iota(new_axes.begin(), new_axes.end(), 0);
  uint32_t expanded_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
      ctx.subgraph(), new_axes.size(), new_axes.data(), value_id, &expanded_id,
      /*flags=*/0))
      << op_name;
  return expanded_id;
}

absl::Status DefineSelectImpl(const graph::Operation& op,
                              YnnpackBuildContext& ctx,
                              bool allow_general_broadcast,
                              absl::string_view op_name) {
  if (op.inputs.size() != 3) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 3 inputs (condition, a, b)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& cond_info,
                              graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& a_info,
                              graph::GetInfo(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& b_info,
                              graph::GetInfo(op.inputs[2]));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));
  if (cond_info.type != Type::kBOOL) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s condition must be BOOL", op_name));
  }
  if (a_info.type != b_info.type || a_info.type != output_info.type ||
      ToYnnType(output_info.type) == ynn_type_invalid) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: unsupported or mismatched operand types", op_name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t cond_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t a_id, ctx.DefineValue(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t b_id, ctx.DefineValue(op.inputs[2]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              ctx.DefineValue(outputs.front()));

  const int output_rank = static_cast<int>(output_info.shape.size());
  const int cond_rank = static_cast<int>(cond_info.shape.size());

  // Normalize non-zero boolean bytes (`uint8` -> `int32`) to `{0, 1}` so
  // `cond == true` indexes slot `1` (`a`) and `cond == false` indexes slot `0`
  // (`b`) in `ynn_define_gather`.
  uint32_t cond_i32_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_convert(
      ctx.subgraph(), cond_id, ynn_type_int32, &cond_i32_id, /*flags=*/0))
      << op_name;
  const int32_t one_i32 = 1;
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const uint32_t one_id,
      ctx.DefineConstant(&one_i32, sizeof(one_i32), Type::kI32, /*shape=*/{}));
  uint32_t norm_cond_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(ctx.subgraph(), ynn_binary_min,
                                               cond_i32_id, one_id,
                                               &norm_cond_id, /*flags=*/0))
      << op_name;

  if (!allow_general_broadcast && cond_rank == 1 && output_rank > 1) {
    // TFLite `Select` allows a 1-D `[D_0]` condition to select whole slices
    // along the outermost dimension.
    LRT_TENSOR_ASSIGN_OR_RETURN(
        norm_cond_id,
        DefineBroadcastOperand(ctx, norm_cond_id, op.inputs[0],
                               absl::MakeConstSpan(output_info.shape).first(1),
                               op_name));
    std::vector<int32_t> trailing_axes(output_rank - 1);
    std::iota(trailing_axes.begin(), trailing_axes.end(), 1);
    uint32_t expanded_cond_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
        ctx.subgraph(), trailing_axes.size(), trailing_axes.data(),
        norm_cond_id, &expanded_cond_id, /*flags=*/0))
        << op_name;
    norm_cond_id = expanded_cond_id;
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        norm_cond_id, DefineBroadcastOperand(ctx, norm_cond_id, op.inputs[0],
                                             output_info.shape, op_name));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        norm_cond_id, ExpandLeadingDimsToRank(ctx, norm_cond_id, cond_rank,
                                              output_rank, op_name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      a_id, DefineBroadcastOperand(ctx, a_id, op.inputs[1], output_info.shape,
                                   op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      a_id,
      ExpandLeadingDimsToRank(ctx, a_id, static_cast<int>(a_info.shape.size()),
                              output_rank, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      b_id, DefineBroadcastOperand(ctx, b_id, op.inputs[2], output_info.shape,
                                   op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      b_id,
      ExpandLeadingDimsToRank(ctx, b_id, static_cast<int>(b_info.shape.size()),
                              output_rank, op_name));

  // Align non-concatenated extents between `a` and `b` before concatenating
  // `[1, ...]` slices into `[2, ...]`.
  if (output_rank > 0) {
    std::vector<int32_t> all_axes(output_rank);
    std::iota(all_axes.begin(), all_axes.end(), 0);
    uint32_t a_bcast_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_broadcast_like(
        ctx.subgraph(), all_axes.size(), all_axes.data(), a_id, b_id,
        &a_bcast_id, /*flags=*/0))
        << op_name;
    uint32_t b_bcast_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_broadcast_like(
        ctx.subgraph(), all_axes.size(), all_axes.data(), b_id, a_bcast_id,
        &b_bcast_id, /*flags=*/0))
        << op_name;
    a_id = a_bcast_id;
    b_id = b_bcast_id;
  }

  const int32_t leading_axis = 0;
  uint32_t b_expanded_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
      ctx.subgraph(), /*num_new_axes=*/1, &leading_axis, b_id, &b_expanded_id,
      /*flags=*/0))
      << op_name;
  uint32_t a_expanded_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
      ctx.subgraph(), /*num_new_axes=*/1, &leading_axis, a_id, &a_expanded_id,
      /*flags=*/0))
      << op_name;
  uint32_t cond_expanded_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
      ctx.subgraph(), /*num_new_axes=*/1, &leading_axis, norm_cond_id,
      &cond_expanded_id, /*flags=*/0))
      << op_name;

  const uint32_t choices[2] = {b_expanded_id, a_expanded_id};
  uint32_t stacked_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_concatenate(
      ctx.subgraph(), /*axis=*/0, /*num_inputs=*/2, choices, &stacked_id,
      /*flags=*/0))
      << op_name;

  // `ynn_define_gather` replaces the gathered axis in place rather than
  // squeezing it, so `stacked_id` and `gathered_id` both have rank
  // `output_rank + 1`, with axis 0 holding the selected extent-1 broadcast
  // slice that is then dropped by `ynn_define_static_transpose`.
  const int32_t gather_axis = 0;
  uint32_t gathered_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_gather(ctx.subgraph(), /*num_axes=*/1, &gather_axis,
                        static_cast<size_t>(output_rank + 1), stacked_id,
                        cond_expanded_id, &gathered_id, /*flags=*/0))
      << op_name;

  std::vector<int32_t> kept_axes(output_rank);
  std::iota(kept_axes.begin(), kept_axes.end(), 1);
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_transpose(
      ctx.subgraph(), kept_axes.size(),
      kept_axes.empty() ? nullptr : kept_axes.data(), gathered_id, &output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

}  // namespace

absl::Status OpMixin<SelectOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return DefineSelectImpl(op, ctx, /*allow_general_broadcast=*/false, "Select");
}

absl::Status OpMixin<SelectV2Operation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return DefineSelectImpl(op, ctx, /*allow_general_broadcast=*/true,
                          "SelectV2");
}

}  // namespace litert::tensor::graph
