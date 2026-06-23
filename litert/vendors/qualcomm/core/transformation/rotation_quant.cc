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
