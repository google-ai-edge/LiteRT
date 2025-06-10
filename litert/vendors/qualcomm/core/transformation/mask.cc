// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/mask.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> TransformToSelectOp(
    const std::vector<OpWrapper>& original_ops, size_t start_index,
    TensorPool& tensor_pool, size_t pattern_size) {
  // const_cast for BuildSelectOp, can be removed if builders are refined
  auto& pattern_input =
      const_cast<TensorWrapper&>(original_ops[start_index].GetInputTensor(0));
  auto& pattern_output = const_cast<TensorWrapper&>(
      original_ops[start_index + pattern_size - 1].GetOutputTensor(0));
  const auto& quant_param = pattern_output.GetQuantParams();
  const auto& tensor_dims = pattern_input.GetDims();
  const std::uint32_t num_element = pattern_input.GetTensorNumElements();

  std::vector<std::int16_t> all_zero_data(num_element, 0);
  auto& input_1 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, tensor_dims,
      all_zero_data.size() * sizeof(all_zero_data[0]), all_zero_data.data());

  auto& mul_static_tensor =
      original_ops[start_index + pattern_size - 1].GetInputTensor(1);
  auto static_tensor_data =
      mul_static_tensor.GetTensorData<std::int16_t>();
  if (!static_tensor_data) {
    QNN_LOG_ERROR("[G2G] Get tensor data failed when transforming mask model.");
    return {};
  }
  std::vector<std::int16_t> mask_data(num_element,
                                      static_tensor_data.value()[0]);
  auto& input_2 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, tensor_dims,
      mask_data.size() * sizeof(mask_data[0]), mask_data.data());

  return BuildSelectOp(tensor_pool, {pattern_input, input_1, input_2},
                       {pattern_output});
}

size_t TransformQuantizeInMask(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Connection check
  bool is_connected = ops[start_index + 0].GetOutputTensor(0) ==
                          ops[start_index + 1].GetInputTensor(0) &&
                      ops[start_index + 1].GetOutputTensor(0) ==
                          ops[start_index + 2].GetInputTensor(0);
  if (!is_connected) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] Transform quant ops in Gemma3 mask models");
  // Construct the new subgraph
  auto new_ops =
      TransformToSelectOp(ops, start_index, tensor_pool, pattern_size);
  if (new_ops.empty()) {
    QNN_LOG_WARNING(
        "[G2G] Transformation failed. Rolling back to the original graph.");
    return 1;
  }
  // Validate new graph.
  bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                    return validate_op_config(op_wrapper);
                  });
  if (is_valid) {
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
