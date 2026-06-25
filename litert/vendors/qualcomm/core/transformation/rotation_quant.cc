// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/rotation_quant.h"

#include <cstddef>
#include <cstring>
#include <functional>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/hadamard_transform_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"

namespace qnn {
namespace {

// Set to 1 to use a balanced adder tree for the down-projection partial-sum
// reduction in ParallelizeSwiGLUHadamardTransform (depth = ceil(log2(N))).
// Set to 0 (default) for a sequential linear chain (depth = N-1).
#define QNN_G2G_SWIGLU_ADDER_TREE 1

void CloneNamespace(const OpWrapper& source, OpWrapper& destination,
                    absl::string_view additional_namespace = {}) {
  absl::string_view start_op_name = source.GetName();
  size_t pos = start_op_name.rfind('/');
  if (pos == absl::string_view::npos) {
    return;
  }
  if (additional_namespace.empty())
    destination.AddPrefixToName(
        absl::StrCat(start_op_name.substr(0, pos), "/"));
  else
    destination.AddPrefixToName(absl::StrCat(start_op_name.substr(0, pos), "/",
                                             additional_namespace, "/"));
}

}  // namespace

size_t ConvertFcToHadamardTransform(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Attempt to replace FC with HadamardTransform.
  const auto& weight = ops[start_index].GetInputTensor(1);
  const auto scale_from_weight = ::qnn::GetSylvesterHadamardScale(weight);
  if (!scale_from_weight.has_value()) return 1;

  QNN_LOG_INFO("[G2G] FC -> HadamardTransform");
  auto op = ::qnn::CreateHadamardTransformOp(
      ops[start_index].GetInputTensor(0), ops[start_index].GetOutputTensor(0),
      scale_from_weight.value());
  // Validate new graph.
  if (validate_op_config(op)) {
    CloneNamespace(ops[start_index], op);
    // Replace the matched pattern with a newly generated subgraph.
    ops[start_index] = std::move(op);
    return 1;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

size_t ParallelizeSwiGLUHadamardTransform(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  static constexpr size_t kFcGateIdx       = 0;
  static constexpr size_t kReshapeGateIdx  = 1;
  static constexpr size_t kLogisticIdx     = 2;
  static constexpr size_t kMul1Idx         = 3;
  static constexpr size_t kFcUpIdx         = 4;
  static constexpr size_t kReshapeUpIdx    = 5;
  static constexpr size_t kMul2Idx         = 6;
  static constexpr size_t kPreReshapeIdx   = 7;
  static constexpr size_t kHadamardIdx     = 8;
  static constexpr size_t kPostReshapeIdx  = 9;
  static constexpr size_t kFcDownIdx       = 10;
  static constexpr size_t kReshapeDownIdx  = 11;

  const auto& fc_gate       = ops[start_index + kFcGateIdx];
  const auto& reshape_gate  = ops[start_index + kReshapeGateIdx];
  const auto& logistic      = ops[start_index + kLogisticIdx];
  const auto& mul1          = ops[start_index + kMul1Idx];
  const auto& fc_up         = ops[start_index + kFcUpIdx];
  const auto& reshape_up    = ops[start_index + kReshapeUpIdx];
  const auto& mul2          = ops[start_index + kMul2Idx];
  const auto& hadamard      = ops[start_index + kHadamardIdx];
  const auto& post_reshape  = ops[start_index + kPostReshapeIdx];

  // Check tensor connections.
  const auto& fc_gate_out      = fc_gate.GetOutputTensor(0);
  const auto& reshape_gate_out = reshape_gate.GetOutputTensor(0);
  const auto& logistic_out     = logistic.GetOutputTensor(0);
  const auto& mul1_out         = mul1.GetOutputTensor(0);
  const auto& fc_up_out        = fc_up.GetOutputTensor(0);
  const auto& reshape_up_out   = reshape_up.GetOutputTensor(0);
  const auto& mul2_out         = mul2.GetOutputTensor(0);
  const auto& h_input          = hadamard.GetInputTensor(0);
  const auto& h_output         = hadamard.GetOutputTensor(0);
  const auto& final_output     = post_reshape.GetOutputTensor(0);

  QNN_LOG_INFO("[G2G] ParallelizeSwiGLUHadamardTransform: checking @ %d (%s)",
               start_index, fc_gate.GetName().data());

  if (reshape_gate.GetInputTensor(0) != fc_gate_out) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: reshape_gate input != fc_gate output");
    return 1;
  }
  if (logistic.GetInputTensor(0) != reshape_gate_out) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: logistic input != reshape_gate output");
    return 1;
  }
  if (!((mul1.GetInputTensor(0) == reshape_gate_out &&
         mul1.GetInputTensor(1) == logistic_out) ||
        (mul1.GetInputTensor(0) == logistic_out &&
         mul1.GetInputTensor(1) == reshape_gate_out))) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: mul1 inputs don't match reshape_gate/logistic outputs");
    return 1;
  }
  if (reshape_up.GetInputTensor(0) != fc_up_out) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: reshape_up input != fc_up output");
    return 1;
  }
  if (!((mul2.GetInputTensor(0) == mul1_out &&
         mul2.GetInputTensor(1) == reshape_up_out) ||
        (mul2.GetInputTensor(0) == reshape_up_out &&
         mul2.GetInputTensor(1) == mul1_out))) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: mul2 inputs don't match mul1/reshape_up outputs");
    return 1;
  }
  if (ops[start_index + kPreReshapeIdx].GetInputTensor(0) != mul2_out) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: pre_reshape input != mul2 output");
    return 1;
  }
  if (ops[start_index + kPreReshapeIdx].GetOutputTensor(0) != h_input) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: pre_reshape output != hadamard input");
    return 1;
  }
  if (post_reshape.GetInputTensor(0) != h_output) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: post_reshape input != hadamard output");
    return 1;
  }
  if (fc_gate.GetInputTensor(0) != fc_up.GetInputTensor(0)) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: fc_gate and fc_up don't share the same input");
    return 1;
  }

  // Validate FC weights.
  const auto& w_gate = fc_gate.GetInputTensor(1);
  const auto& w_up   = fc_up.GetInputTensor(1);
  if (w_gate.GetRank() != 2 || !w_gate.IsTensorStatic()) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: w_gate rank=%d static=%d",
                    w_gate.GetRank(), w_gate.IsTensorStatic());
    return 1;
  }
  if (w_up.GetRank() != 2 || !w_up.IsTensorStatic()) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: w_up rank=%d static=%d",
                    w_up.GetRank(), w_up.IsTensorStatic());
    return 1;
  }
  if (w_gate.GetDimension(0) != w_up.GetDimension(0)) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: w_gate dim0=%d != w_up dim0=%d",
                    w_gate.GetDimension(0), w_up.GetDimension(0));
    return 1;
  }
  if (w_gate.GetDimension(1) != w_up.GetDimension(1)) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: w_gate dim1=%d != w_up dim1=%d",
                    w_gate.GetDimension(1), w_up.GetDimension(1));
    return 1;
  }
  // Per-channel quant slicing along axis 0 is not yet supported.
  if (w_gate.IsPerChannelQuant() || w_up.IsPerChannelQuant()) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: per-channel quant not yet supported "
                    "(gate=%d up=%d)",
                    w_gate.IsPerChannelQuant(), w_up.IsPerChannelQuant());
    return 1;
  }

  // Compute N (number of Proj blocks) and H (Hadamard size).
  // fc_gate_out has the pre-reshape shape, e.g. [1, N*H].
  const std::uint32_t H = h_input.GetDimension(h_input.GetRank() - 1);
  const std::uint32_t NxH =
      fc_gate_out.GetDimension(fc_gate_out.GetRank() - 1);
  if (NxH <= H || NxH % H != 0) {
    QNN_LOG_WARNING("[G2G] SwiGLU bail: NxH=%d H=%d (NxH must be a multiple of H > H)",
                    NxH, H);
    return 1;
  }
  const std::uint32_t N = NxH / H;

  // Early validation for the 12-op case (down FC + reshape).
  if (pattern_size >= 12) {
    const auto& fc_down      = ops[start_index + kFcDownIdx];
    const auto& reshape_down = ops[start_index + kReshapeDownIdx];
    if (fc_down.GetInputTensor(0) != final_output) {
      QNN_LOG_WARNING("[G2G] SwiGLU bail: fc_down input != post-hadamard reshape output");
      return 1;
    }
    if (reshape_down.GetInputTensor(0) != fc_down.GetOutputTensor(0)) {
      QNN_LOG_WARNING("[G2G] SwiGLU bail: reshape_down input != fc_down output");
      return 1;
    }
    const auto& w_down = fc_down.GetInputTensor(1);
    if (w_down.GetRank() != 2 || !w_down.IsTensorStatic()) {
      QNN_LOG_WARNING("[G2G] SwiGLU bail: w_down rank=%d static=%d",
                      w_down.GetRank(), w_down.IsTensorStatic());
      return 1;
    }
    if (w_down.GetDimension(1) != NxH) {
      QNN_LOG_WARNING("[G2G] SwiGLU bail: w_down in_features=%d != NxH=%d",
                      w_down.GetDimension(1), NxH);
      return 1;
    }
    if (w_down.IsPerChannelQuant()) {
      const auto* aq = std::get_if<AxisScaleOffsetQuantizeParamsWrapper>(
          &w_down.GetQuantParams());
      if (aq == nullptr || aq->GetAxis() != 0) {
        QNN_LOG_WARNING("[G2G] SwiGLU bail: w_down per-channel quant axis != 0");
        return 1;
      }
    }
  }

  // Weight byte-slicing helpers (axis 0: output-channel dimension).
  auto make_weight_bytes = [](const TensorWrapper& weight, std::uint32_t chunk,
                               std::uint32_t num_rows,
                               std::uint32_t H) -> std::vector<std::byte> {
    const Qnn_Tensor_t& w_qnn = weight.GetQnnTensor();
    const auto* raw = static_cast<const std::byte*>(w_qnn.v2.clientBuf.data);
    const size_t total = w_qnn.v2.clientBuf.dataSize;
    if (raw == nullptr || total == 0 || total % num_rows != 0) return {};
    const size_t bpr = total / num_rows;  // bytes per output-channel row
    std::vector<std::byte> buf(H * bpr);
    std::memcpy(buf.data(), raw + chunk * H * bpr, H * bpr);
    return buf;
  };

  const std::uint32_t w_num_rows = w_gate.GetDimension(0);  // = N * H

  // Build N Proj blocks.
  std::vector<qnn::ConstTensorWrapperRef> proj_outputs;
  proj_outputs.reserve(N);
  std::vector<OpWrapper> new_ops;
  new_ops.reserve(N * 8 + 1);

  // FC outputs keep the pre-reshape rank (e.g. [1, H]).
  auto fc_out_dims = fc_gate_out.GetDimensions();
  fc_out_dims[fc_gate_out.GetRank() - 1] = H;
  // All ops after reshape use the restored rank (e.g. [1, 1, H]).
  auto io_dims = reshape_gate_out.GetDimensions();
  io_dims[reshape_gate_out.GetRank() - 1] = H;
  auto had_out_dims = final_output.GetDimensions();
  had_out_dims[final_output.GetRank() - 1] = H;

  // Helper to create a reshape op in a Proj block. Handles the case where
  // the reshape op encodes its target shape via a second input tensor.
  auto make_reshape_op = [&](const OpWrapper& src_reshape,
                              const TensorWrapper& data_in,
                              const TensorWrapper& data_out,
                              std::uint32_t block_idx) -> OpWrapper& {
    std::vector<qnn::ConstTensorWrapperRef> reshape_inputs;
    reshape_inputs.emplace_back(data_in);
    if (src_reshape.GetInputTensorCount() >= 2) {
      auto out_dims = data_out.GetDimensions();
      std::vector<std::uint32_t> shape_vals(out_dims.begin(), out_dims.end());
      auto& shape_t = tensor_pool.CreateStaticTensor(
          QNN_DATATYPE_UINT_32, {},
          {static_cast<std::uint32_t>(shape_vals.size())},
          static_cast<std::uint32_t>(sizeof(std::uint32_t) * shape_vals.size()),
          shape_vals.data());
      reshape_inputs.emplace_back(shape_t);
    }
    auto& op = new_ops.emplace_back(
        CreateOpWithSameParams(src_reshape, reshape_inputs, {data_out}));
    CloneNamespace(fc_gate, op, std::to_string(block_idx));
    return op;
  };

  for (std::uint32_t i = 0; i < N; ++i) {
    // Slice weights for block i.
    auto gate_bytes = make_weight_bytes(w_gate, i, w_num_rows, H);
    auto up_bytes   = make_weight_bytes(w_up,   i, w_num_rows, H);
    if (gate_bytes.empty() || up_bytes.empty()) return 1;

    auto& w_gate_i = tensor_pool.CreateStaticTensor(
        w_gate.GetDataType(), w_gate.GetQuantParams(),
        {H, w_gate.GetDimension(1)},
        static_cast<std::uint32_t>(gate_bytes.size()), gate_bytes.data());
    auto& w_up_i = tensor_pool.CreateStaticTensor(
        w_up.GetDataType(), w_up.GetQuantParams(),
        {H, w_up.GetDimension(1)},
        static_cast<std::uint32_t>(up_bytes.size()), up_bytes.data());

    // Intermediate tensors for this Proj block.
    auto& fc_gate_out_i      = tensor_pool.CloneNativeTensorFrom(fc_gate_out,      fc_out_dims);
    auto& reshape_gate_out_i = tensor_pool.CloneNativeTensorFrom(reshape_gate_out, io_dims);
    auto& logistic_out_i     = tensor_pool.CloneNativeTensorFrom(logistic_out,     io_dims);
    auto& mul1_out_i         = tensor_pool.CloneNativeTensorFrom(mul1_out,         io_dims);
    auto& fc_up_out_i        = tensor_pool.CloneNativeTensorFrom(fc_up_out,        fc_out_dims);
    auto& reshape_up_out_i   = tensor_pool.CloneNativeTensorFrom(reshape_up_out,   io_dims);
    auto& mul2_out_i         = tensor_pool.CloneNativeTensorFrom(mul2_out,         io_dims);
    auto& had_out_i          = tensor_pool.CloneNativeTensorFrom(final_output,     had_out_dims);

    // FC_gate_i
    std::vector<qnn::ConstTensorWrapperRef> gate_inputs;
    gate_inputs.reserve(3);
    gate_inputs.emplace_back(fc_gate.GetInputTensor(0));
    gate_inputs.emplace_back(w_gate_i);
    if (fc_gate.GetInputTensorCount() >= 3) {
      gate_inputs.emplace_back(fc_gate.GetInputTensor(2));
    }
    auto& op_fc_gate = new_ops.emplace_back(
        CreateOpWithSameParams(fc_gate, gate_inputs, {fc_gate_out_i}));
    CloneNamespace(fc_gate, op_fc_gate, std::to_string(i));

    // Reshape_gate_i
    make_reshape_op(reshape_gate, fc_gate_out_i, reshape_gate_out_i, i);

    // Logistic_i
    auto& op_logistic = new_ops.emplace_back(
        CreateOpWithSameParams(logistic, {reshape_gate_out_i}, {logistic_out_i}));
    CloneNamespace(fc_gate, op_logistic, std::to_string(i));

    // Mul1_i: preserve original input order.
    std::vector<qnn::ConstTensorWrapperRef> mul1_inputs;
    mul1_inputs.reserve(2);
    if (mul1.GetInputTensor(0) == reshape_gate_out) {
      mul1_inputs.emplace_back(reshape_gate_out_i);
      mul1_inputs.emplace_back(logistic_out_i);
    } else {
      mul1_inputs.emplace_back(logistic_out_i);
      mul1_inputs.emplace_back(reshape_gate_out_i);
    }
    auto& op_mul1 = new_ops.emplace_back(
        CreateOpWithSameParams(mul1, mul1_inputs, {mul1_out_i}));
    CloneNamespace(fc_gate, op_mul1, std::to_string(i));

    // FC_up_i
    std::vector<qnn::ConstTensorWrapperRef> up_inputs;
    up_inputs.reserve(3);
    up_inputs.emplace_back(fc_up.GetInputTensor(0));
    up_inputs.emplace_back(w_up_i);
    if (fc_up.GetInputTensorCount() >= 3) {
      up_inputs.emplace_back(fc_up.GetInputTensor(2));
    }
    auto& op_fc_up = new_ops.emplace_back(
        CreateOpWithSameParams(fc_up, up_inputs, {fc_up_out_i}));
    CloneNamespace(fc_gate, op_fc_up, std::to_string(i));

    // Reshape_up_i
    make_reshape_op(reshape_up, fc_up_out_i, reshape_up_out_i, i);

    // Mul2_i: preserve original input order.
    std::vector<qnn::ConstTensorWrapperRef> mul2_inputs;
    mul2_inputs.reserve(2);
    if (mul2.GetInputTensor(0) == mul1_out) {
      mul2_inputs.emplace_back(mul1_out_i);
      mul2_inputs.emplace_back(reshape_up_out_i);
    } else {
      mul2_inputs.emplace_back(reshape_up_out_i);
      mul2_inputs.emplace_back(mul1_out_i);
    }
    auto& op_mul2 = new_ops.emplace_back(
        CreateOpWithSameParams(mul2, mul2_inputs, {mul2_out_i}));
    CloneNamespace(fc_gate, op_mul2, std::to_string(i));

    // Hadamard_i (directly on [1,1,H] — no reshape needed).
    auto& op_hadamard = new_ops.emplace_back(
        CreateOpWithSameParams(hadamard, {mul2_out_i}, {had_out_i}));
    CloneNamespace(fc_gate, op_hadamard, std::to_string(i));

    proj_outputs.emplace_back(had_out_i);
  }

  if (pattern_size >= 12) {
    // Down-projection FC splitting + Add accumulation.
    const auto& fc_down       = ops[start_index + kFcDownIdx];
    const auto& reshape_down  = ops[start_index + kReshapeDownIdx];
    const auto& w_down        = fc_down.GetInputTensor(1);
    const std::uint32_t num_units    = w_down.GetDimension(0);
    const auto& fc_down_output       = fc_down.GetOutputTensor(0);
    const auto& final_reshape_output = reshape_down.GetOutputTensor(0);

    const Qnn_Tensor_t& w_qnn    = w_down.GetQnnTensor();
    const auto* raw_bytes = static_cast<const std::byte*>(w_qnn.v2.clientBuf.data);
    const size_t total_bytes     = w_qnn.v2.clientBuf.dataSize;
    if (raw_bytes == nullptr || total_bytes == 0 || total_bytes % num_units != 0) return 1;
    const size_t bytes_per_row   = total_bytes / num_units;
    if (bytes_per_row % N != 0) return 1;
    const size_t bytes_per_chunk = bytes_per_row / N;

    const bool has_bias = fc_down.GetInputTensorCount() >= 3;

    std::vector<qnn::ConstTensorWrapperRef> partial_outputs;
    partial_outputs.reserve(N);
    for (std::uint32_t i = 0; i < N; ++i) {
      std::vector<std::byte> chunk(num_units * bytes_per_chunk);
      for (std::uint32_t r = 0; r < num_units; ++r) {
        std::memcpy(chunk.data() + r * bytes_per_chunk,
                    raw_bytes + r * bytes_per_row + i * bytes_per_chunk,
                    bytes_per_chunk);
      }
      auto& sliced_w = tensor_pool.CreateStaticTensor(
          w_down.GetDataType(), w_down.GetQuantParams(),
          {num_units, H},
          static_cast<std::uint32_t>(chunk.size()), chunk.data());

      const auto& partial_out = tensor_pool.CloneNativeTensorFrom(fc_down_output);
      std::vector<qnn::ConstTensorWrapperRef> fc_inputs;
      fc_inputs.emplace_back(proj_outputs[i]);
      fc_inputs.emplace_back(sliced_w);
      if (i == 0 && has_bias) fc_inputs.emplace_back(fc_down.GetInputTensor(2));
      auto& partial_fc = new_ops.emplace_back(
          CreateOpWithSameParams(fc_down, fc_inputs, {partial_out}));
      CloneNamespace(fc_gate, partial_fc, std::to_string(i));
      partial_outputs.emplace_back(partial_out);
    }

    // Accumulate partial FC outputs into fc_down_output.
#if QNN_G2G_SWIGLU_ADDER_TREE
    // Balanced adder tree: depth = ceil(log2(N)).
    std::vector<qnn::ConstTensorWrapperRef> level_inputs = partial_outputs;
    while (level_inputs.size() > 1) {
      std::vector<qnn::ConstTensorWrapperRef> level_outputs;
      level_outputs.reserve((level_inputs.size() + 1) / 2);
      for (size_t j = 0; j + 1 < level_inputs.size(); j += 2) {
        const bool is_very_last = (level_inputs.size() == 2 && j == 0);
        const TensorWrapper& add_out =
            is_very_last ? fc_down_output
                         : tensor_pool.CloneNativeTensorFrom(fc_down_output);
        auto& add = new_ops.emplace_back(CreateElementWiseAddOp(
            level_inputs[j].get(), level_inputs[j + 1].get(), add_out));
        CloneNamespace(fc_gate, add, absl::StrCat("tree_", j));
        level_outputs.emplace_back(add_out);
      }
      if (level_inputs.size() % 2 == 1) {
        level_outputs.emplace_back(level_inputs.back());
      }
      level_inputs = std::move(level_outputs);
    }
#else
    // Linear chain: depth = N-1.
    qnn::ConstTensorWrapperRef running_sum = partial_outputs[0];
    for (std::uint32_t i = 1; i < N; ++i) {
      const bool is_last = (i + 1 == N);
      const TensorWrapper& add_out =
          is_last ? fc_down_output
                  : tensor_pool.CloneNativeTensorFrom(fc_down_output);
      auto& add = new_ops.emplace_back(
          CreateElementWiseAddOp(running_sum.get(), partial_outputs[i].get(), add_out));
      CloneNamespace(fc_gate, add, std::to_string(i));
      running_sum = add_out;
    }
#endif

    // Preserve the reshape that follows the down FC.
    make_reshape_op(reshape_down, fc_down_output, final_reshape_output,
                    static_cast<std::uint32_t>(start_index));
  } else {
    // Concat all Hadamard outputs → final_output.
    const std::uint32_t concat_axis = final_output.GetRank() - 1;
    auto& concat = new_ops.emplace_back(
        CreateConcatenationOp(proj_outputs, final_output, concat_axis));
    CloneNamespace(fc_gate, concat, std::to_string(start_index));
  }

  // Validate new graph.
  const bool is_valid = true;
  std::all_of(new_ops.begin(), new_ops.end(),
              [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                return validate_op_config(op_wrapper);
              });
  if (is_valid) {
    QNN_LOG_INFO("[G2G] ParallelizeSwiGLUHadamardTransform @ %d, N=%d, with_down_fc=%d",
                 start_index, N, pattern_size >= 12 ? 1 : 0);
    for (size_t i = 0; i < new_ops.size(); ++i) {
      new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
    }
    size_t step_size = new_ops.size();
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    return step_size;
  }

  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

size_t ParallelizeHadamardTransform(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  static constexpr size_t kPreReshapeIndex = 0;
  static constexpr size_t kHadamardTransformIndex = 1;
  static constexpr size_t kPostReshapeIndex = 2;
  // Check connection.
  const auto& hadamard_transform = ops[start_index + kHadamardTransformIndex];
  const auto& h_input = hadamard_transform.GetInputTensor(0);
  if (!(ops[start_index + kPreReshapeIndex].GetOutputTensor(0) == h_input &&
        hadamard_transform.GetOutputTensor(0) ==
            ops[start_index + kPostReshapeIndex].GetInputTensor(0))) {
    return 1;
  }

  const auto& input = ops[start_index + kPreReshapeIndex].GetInputTensor(0);
  const auto& output = ops[start_index + kPostReshapeIndex].GetOutputTensor(0);
  const std::uint32_t input_size = input.GetDimension(input.GetRank() - 1);
  const std::uint32_t hadamard_size =
      h_input.GetDimension(h_input.GetRank() - 1);

  // Check both rehsape ops are for non-power-of-two Hadamard.
  if (!(input.GetDimensions() == output.GetDimensions() &&
        input_size > hadamard_size && input_size % hadamard_size == 0)) {
    return 1;
  }

  std::vector<OpWrapper> new_ops;

  // Compute the number of HadamardTransform ops.
  const std::uint32_t num_h = input_size / hadamard_size;

  // Prepare inputs for parrallel Hadamard Transform.
  std::vector<qnn::ConstTensorWrapperRef> hadamard_transform_inputs;
  hadamard_transform_inputs.reserve(num_h);
  auto io_dims = input.GetDimensions();
  io_dims[input.GetRank() - 1] /= num_h;
  for (std::uint32_t i = 0; i < num_h; ++i) {
    hadamard_transform_inputs.emplace_back(
        tensor_pool.CloneNativeTensorFrom(input, io_dims));
  }

  // Split
  size_t axis = input.GetRank() - 1;
  std::vector<std::uint32_t> split_indice;
  split_indice.reserve(num_h - 1);
  for (std::uint32_t i = 1; i < num_h; i++) {
    split_indice.emplace_back(i * input.GetDimension(axis) / num_h);
  }
  const auto& split_indice_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {},
      {static_cast<std::uint32_t>(split_indice.size())},
      sizeof(std::uint32_t) * split_indice.size(), split_indice.data());
  auto& split = new_ops.emplace_back(
      CreateSplitOp(input, hadamard_transform_inputs, 2, split_indice_tensor));
  CloneNamespace(ops[start_index + kHadamardTransformIndex], split,
                 std::to_string(start_index));

  // HadamardTransform
  std::vector<qnn::ConstTensorWrapperRef> hadamard_transform_outputs;
  hadamard_transform_outputs.reserve(hadamard_transform_inputs.size());
  for (const auto& input_tensor : hadamard_transform_inputs) {
    const auto& output_tensor = hadamard_transform_outputs.emplace_back(
        tensor_pool.CloneNativeTensorFrom(output, io_dims));
    auto& hadamard = new_ops.emplace_back(CreateOpWithSameParams(
        hadamard_transform, {input_tensor}, {output_tensor}));
    CloneNamespace(ops[start_index + kHadamardTransformIndex], hadamard,
                   std::to_string(start_index));
  }
  if (pattern_size == 4) {
    // Replace `concat -> FC(W)` with `num_h` parallel `FC(W_i)` ops whose
    // outputs are summed with a chain of element-wise Add ops. W is sliced
    // along its in-features axis (axis 1) into `num_h` chunks of shape
    // [num_units, hadamard_size]. Bias, if present, is attached to the first
    // partial FC so it is accumulated exactly once.
    //
    // Example shapes:
    //   Hadamard outputs : num_h * [..., hadamard_size]
    //   Original FC      : in [..., num_h * hadamard_size]
    //                      W  [num_units, num_h * hadamard_size]
    //                      out [..., num_units]
    //   New              : num_h * FC( h_out_i, W_i ) -> partial_i
    //                      partial_0 + partial_1 + ... + partial_{N-1} -> out
    static constexpr size_t kPostFCIndex = 3;
    const auto& fc_op = ops[start_index + kPostFCIndex];
    // Bail out if the FC's input is not produced by the post-reshape we just
    // eliminated.
    if (fc_op.GetInputTensor(0) != output) {
      QNN_LOG_WARNING("[G2G] FC input does not match post-reshape output.");
      return 1;
    }
    // The post-reshape output is being dropped (no concat is emitted in this
    // branch). If any op other than the trailing FC also consumes it, we
    // would orphan that consumer and produce an unconnected QNN graph.
    for (size_t j = 0; j < ops.size(); ++j) {
      if (j == start_index + kPostFCIndex) continue;
      const auto& other = ops[j];
      for (size_t k = 0; k < other.GetInputTensorCount(); ++k) {
        if (other.GetInputTensor(k) == output) {
          QNN_LOG_WARNING(
              "[G2G] Post-reshape output has additional consumers.",
              start_index);
          return 1;
        }
      }
    }
    QNN_LOG_INFO("[G2G] ParallelizeHadamardTransform @ %d w/ FC", start_index);
    const auto& weight = fc_op.GetInputTensor(1);
    if (weight.GetRank() != 2 || !weight.IsTensorStatic()) {
      return 1;
    }
    const std::uint32_t num_units = weight.GetDimension(0);
    const std::uint32_t in_features = weight.GetDimension(1);
    if (in_features != input_size || in_features % num_h != 0) {
      return 1;
    }
    // Per-channel quantization is only safe to share across slices when the
    // quant axis is the output-channel axis (axis 0), since we slice axis 1.
    if (weight.IsPerChannelQuant()) {
      const auto* axis_quant =
          std::get_if<AxisScaleOffsetQuantizeParamsWrapper>(
              &weight.GetQuantParams());
      if (axis_quant == nullptr || axis_quant->GetAxis() != 0) {
        return 1;
      }
    }

    const Qnn_Tensor_t& w_qnn = weight.GetQnnTensor();
    const auto* raw_bytes =
        static_cast<const std::byte*>(w_qnn.v2.clientBuf.data);
    const size_t total_bytes = w_qnn.v2.clientBuf.dataSize;
    if (raw_bytes == nullptr || total_bytes == 0 ||
        total_bytes % num_units != 0) {
      return 1;
    }
    const size_t bytes_per_row = total_bytes / num_units;
    if (bytes_per_row % num_h != 0) {
      // Packed sub-byte weights whose chunk does not land on a byte boundary
      // cannot be split byte-wise.
      return 1;
    }
    const size_t bytes_per_chunk_row = bytes_per_row / num_h;

    const bool has_bias = fc_op.GetInputTensorCount() >= 3;
    const auto& fc_output = fc_op.GetOutputTensor(0);

    std::vector<qnn::ConstTensorWrapperRef> partial_outputs;
    partial_outputs.reserve(num_h);

    for (std::uint32_t i = 0; i < num_h; ++i) {
      std::vector<std::byte> chunk_bytes(num_units * bytes_per_chunk_row);
      for (std::uint32_t r = 0; r < num_units; ++r) {
        std::memcpy(chunk_bytes.data() + r * bytes_per_chunk_row,
                    raw_bytes + r * bytes_per_row + i * bytes_per_chunk_row,
                    bytes_per_chunk_row);
      }
      auto& sliced_weight = tensor_pool.CreateStaticTensor(
          weight.GetDataType(), weight.GetQuantParams(),
          {num_units, hadamard_size},
          static_cast<std::uint32_t>(chunk_bytes.size()), chunk_bytes.data());

      // For all but the final partial FC, allocate a native intermediate
      // tensor to hold the partial output. The very last partial FC writes
      // directly into the original FC output only if there are no Add ops
      // following — which is not the case here (num_h >= 2), so always use
      // intermediates.
      const auto& partial_out = tensor_pool.CloneNativeTensorFrom(fc_output);

      std::vector<qnn::ConstTensorWrapperRef> fc_inputs;
      fc_inputs.reserve(3);
      fc_inputs.emplace_back(hadamard_transform_outputs[i]);
      fc_inputs.emplace_back(sliced_weight);
      if (i == 0 && has_bias) {
        fc_inputs.emplace_back(fc_op.GetInputTensor(2));
      }
      auto& partial_fc = new_ops.emplace_back(
          CreateOpWithSameParams(fc_op, fc_inputs, {partial_out}));
      CloneNamespace(fc_op, partial_fc, std::to_string(start_index));
      partial_outputs.emplace_back(partial_out);
    }

    // Chain Add ops to reduce partial outputs into the original FC output.
    // For num_h partials we need num_h - 1 Adds. The first num_h - 2 Adds
    // produce native intermediates; the final Add writes to `fc_output`.
    qnn::ConstTensorWrapperRef running_sum = partial_outputs[0];
    for (std::uint32_t i = 1; i < num_h; ++i) {
      const bool is_last = (i + 1 == num_h);
      const TensorWrapper& add_out =
          is_last ? fc_output : tensor_pool.CloneNativeTensorFrom(fc_output);
      auto& add = new_ops.emplace_back(CreateElementWiseAddOp(
          running_sum.get(), partial_outputs[i].get(), add_out));
      CloneNamespace(fc_op, add, std::to_string(start_index));
      running_sum = add_out;
    }
  } else {
    QNN_LOG_INFO("[G2G] ParallelizeHadamardTransform @ %d", start_index);
    // Concat
    std::uint32_t adjusted_axis = output.GetRank() - 1;
    auto& concat = new_ops.emplace_back(CreateConcatenationOp(
        hadamard_transform_outputs, output, adjusted_axis));
    CloneNamespace(ops[start_index + kHadamardTransformIndex], concat,
                   std::to_string(start_index));
  }

  // Validate new graph.
  const bool is_valid = true;
  std::all_of(new_ops.begin(), new_ops.end(),
              [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                return validate_op_config(op_wrapper);
              });
  if (is_valid) {
    // Adjust the name to avoid a name collision in the Qnn JSON dump.
    for (size_t i = 0; i < new_ops.size(); ++i) {
      new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
    }
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size();
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    return step_size;
  }

  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

}  // namespace qnn
