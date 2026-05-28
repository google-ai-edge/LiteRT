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
  const std::uint32_t input_size = input.GetDimension(input.GetRank() - 1);
  const std::uint32_t hadamard_size =
      h_input.GetDimension(h_input.GetRank() - 1);

  // Check both rehsape ops are for non-power-of-two Hadamard.
  if (!(input.GetDimensions() == output.GetDimensions() &&
        input_size > hadamard_size && input_size % hadamard_size == 0)) {
    return 1;
  }

  QNN_LOG_INFO("[G2G] ParallelizeHadamardTransform @ %d", start_index);

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

  // Concat
  std::uint32_t adjusted_axis = output.GetRank() - 1;
  auto& concat = new_ops.emplace_back(
      CreateConcatenationOp(hadamard_transform_outputs, output, adjusted_axis));
  CloneNamespace(ops[start_index + kHadamardTransformIndex], concat,
                 std::to_string(start_index));

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
