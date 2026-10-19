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

// Checks if `weights` can be consumed by `DefineQuantizedFullyConnected`.
//
// - `weights`: the tensor info of the quantized tensor.
// - `weights_are_kn`: tells whether the weights are laid out as [K, N] (e.g.
//   the right-hand side of a batch matmul without `adj_y`) rather than [N, K]
//   (e.g. the weights of a fully connected op).
bool CanUseQuantizedDot(const graph::TensorInformation& weights,
                        bool weights_are_kn) {
  if (weights.shape.size() != 2 || weights.quantization == nullptr) {
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
  // [N, K] weights are transposed to [K, N], which makes N the packed
  // dimension of sub-byte types. YNNPACK requires packed dimensions to fill
  // whole bytes.
  const size_t channels = weights.shape[channel_dim];
  if (!weights_are_kn &&
      channels % YnnpackBuildContext::ElementsPerByte(storage_type) != 0) {
    return false;
  }
  // Asymmetric blockwise weights would require the dot to subtract the zero
  // points from the full weights, which YNNPACK doesn't constant fold.
  if (YnnpackBuildContext::GetBlockSize(quantization) > 0 &&
      !YnnpackBuildContext::IsSymmetric(quantization)) {
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
    YnnpackBuildContext& ctx, uint32_t input_id,
    const graph::TensorInformation& weights, uint32_t weights_data_id,
    bool weights_are_kn, uint32_t bias_id, uint32_t& result_id,
    absl::string_view op_name) {
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
  const absl::Span<const float> stored_scales =
      YnnpackBuildContext::GetScales(quantization);

  if (const size_t block_size = YnnpackBuildContext::GetBlockSize(quantization);
      block_size > 0) {
    // One scale per channel and block: [N, K / block_size]. The scales are
    // stored channel major whatever the layout of the weights.
    LRT_TENSOR_ASSIGN_OR_RETURN(
        const uint32_t weights_scale_id,
        ctx.DefineConstant(
            stored_scales.data(), stored_scales.size() * sizeof(float),
            Type::kFP32,
            /*shape=*/{channels, stored_scales.size() / channels}));
    LRT_TENSOR_RETURN_IF_ERROR(ynn::define_blockwise_dot(
        sg, quantized_input_id, input_zero_point_id, input_scale_id, weights_id,
        /*b_zero_point_id=*/YNN_INVALID_VALUE_ID, weights_scale_id, block_size,
        bias_id, ynn_type_fp32, result_id))
        << op_name;
    return absl::OkStatus();
  }

  // One scale (and zero point) per channel: [1, N]. A per-tensor scale is
  // repeated for every channel.
  std::vector<float> scales(stored_scales.begin(), stored_scales.end());
  std::vector<int32_t> zero_points =
      YnnpackBuildContext::GetZeroPoints(quantization);
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

absl::Status OpMixin<MaxPool2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("MaxPool2D is not supported in YNNPACK.");
}

absl::Status OpMixin<Conv2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("Conv2D is not supported in YNNPACK.");
}

absl::Status OpMixin<DepthwiseConv2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "DepthwiseConv2D is not supported in YNNPACK.");
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
      CanUseQuantizedDot(*weights_info, /*weights_are_kn=*/false)) {
    LRT_TENSOR_RETURN_IF_ERROR(DefineQuantizedFullyConnected(
        ctx, input_id, *weights_info, weights_data_id,
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
      CanUseQuantizedDot(*rhs_info, /*weights_are_kn=*/!op_data.adj_y)) {
    return DefineQuantizedFullyConnected(
        ctx, lhs_id, *rhs_info, rhs_data_id, /*weights_are_kn=*/!op_data.adj_y,
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
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
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
  std::vector<int32_t> axes(num_axes);
  std::iota(axes.begin(), axes.end(), 0);
  std::vector<int64_t> slice_begins(num_axes);
  std::vector<int64_t> slice_ends(num_axes);
  for (size_t i = 0; i < num_axes; ++i) {
    slice_begins[i] = begins[i];
    if (sizes[i] < 0) {
      // A negative size means "up to the end of this dimension". Spelling that
      // out as the extent the graph happens to have been built with would
      // freeze the dimension, which breaks graphs that are re-run with a
      // different shape, e.g. attention over a KV cache that grows by one
      // token per decode step. YNNPACK interprets an end of 0 or less as
      // relative to the (possibly symbolic) extent, so 0 keeps it dynamic.
      slice_ends[i] = 0;
    } else {
      slice_ends[i] = begins[i] + sizes[i];
    }
  }

  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_slice(
      ctx.subgraph(), num_axes, axes.data(), slice_begins.data(),
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

absl::Status OpMixin<GatherOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Gather";
  LRT_TENSOR_ASSIGN_OR_RETURN(const GatherOperation& op_data,
                              op.As<GatherOperation>());
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, indices)", op_name));
  }
  if (op_data.batch_dims != 0) {
    return absl::UnimplementedError(absl::StrFormat(
        "%s: batch_dims != 0 is not supported in YNNPACK.", op_name));
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
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));

  const int32_t axis = op_data.axis;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_gather(
      ctx.subgraph(), /*num_axes=*/1, &axis, output_info.shape.size(), input_id,
      index_id, &output_id, /*flags=*/0))
      << op_name;
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

}  // namespace litert::tensor::graph
