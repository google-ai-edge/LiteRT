// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/rotation_quant.h"

#include <cstddef>
#include <functional>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {
namespace {

constexpr size_t kPreReshapeIndex = 0;
constexpr size_t kHadamardTransformIndex = 1;
constexpr size_t kPostReshapeIndex = 2;

void CloneNamespace(OpWrapper& source, std::vector<OpWrapper>& ops) {
  absl::string_view start_op_name = source.GetOpConfig().v1.name;
  size_t pos = start_op_name.rfind('/');
  if (pos == absl::string_view::npos) {
    return;
  }
  for (auto& op : ops) {
    op.AddPrefixToName(absl::StrCat(start_op_name.substr(0, pos), "/"));
  }
}

// TODO (jiunkaiy): Merge similar utility function.
OpWrapper& EmplaceOpWithIO(
    std::vector<OpWrapper>& new_ops, const OpWrapper& source_op,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& inputs,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& outputs) {
  auto& ret = new_ops.emplace_back(source_op);
  ret.UpdateTensors(inputs, outputs);
  return ret;
}

}  // namespace

size_t ParallelizeHadamardTransform(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
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
  const std::uint32_t input_size = input.GetDim(input.GetRank() - 1);
  const std::uint32_t hadamard_size = h_input.GetDim(h_input.GetRank() - 1);

  // Check both rehsape ops are for non-power-of-two Hadamard.
  if (!(input.GetDims() == output.GetDims() && input_size > hadamard_size &&
        input_size % hadamard_size == 0)) {
    return 1;
  }

  QNN_LOG_INFO("[G2G] ParallelizeHadamardTransform @ %d", start_index);

  std::vector<OpWrapper> new_ops;

  // Compute the number of HadamardTransform ops.
  const std::uint32_t num_h = input_size / hadamard_size;

  // Prepare inputs for parrallel Hadamard Transform.
  std::vector<::qnn::TensorWrapperRef> hadamard_transform_inputs;
  hadamard_transform_inputs.reserve(num_h);
  auto io_dims = input.GetDims();
  io_dims[input.GetRank() - 1] /= num_h;
  for (std::uint32_t i = 0; i < num_h; ++i) {
    hadamard_transform_inputs.emplace_back(
        tensor_pool.CloneNativeTensorFrom(input, io_dims));
  }

  // Split
  const std::array<std::int32_t, 1> split_axis_data{
      static_cast<std::int32_t>(input.GetRank() - 1)};
  auto& split_axis = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {split_axis_data.size()},
      split_axis_data.size() * sizeof(split_axis_data[0]),
      split_axis_data.data());
  auto split = BuildSplitOp(
      tensor_pool, {split_axis, const_cast<::qnn::TensorWrapper&>(input)},
      hadamard_transform_inputs, num_h);
  CloneNamespace(ops[start_index + kHadamardTransformIndex], split);
  std::move(split.begin(), split.end(), std::back_inserter(new_ops));

  // HadamardTransform
  std::vector<::qnn::TensorWrapperRef> hadamard_transform_outputs;
  hadamard_transform_outputs.reserve(hadamard_transform_inputs.size());
  for (const auto& input_tensor : hadamard_transform_inputs) {
    const auto& output_tensor = hadamard_transform_outputs.emplace_back(
        tensor_pool.CloneNativeTensorFrom(output, io_dims));
    EmplaceOpWithIO(new_ops, hadamard_transform, {input_tensor},
                    {output_tensor});
  }

  // Concat
  std::uint32_t adjusted_axis = output.GetRank() - 1;
  auto concat = BuildConcatenationOp(
      tensor_pool, hadamard_transform_outputs,
      {const_cast<::qnn::TensorWrapper&>(output)}, adjusted_axis);
  CloneNamespace(ops[start_index + kHadamardTransformIndex], concat);
  std::move(concat.begin(), concat.end(), std::back_inserter(new_ops));

  // Validate new graph.
  const bool is_valid = true;
  std::all_of(new_ops.begin(), new_ops.end(),
              [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                return op_wrapper.IsOpCode(QnnOpCode::kHadamardTransform) ||
                       validate_op_config(op_wrapper);
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
