// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/mask.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {
std::optional<OpWrapper> CreateGemma3SelectOp(
    const std::vector<OpWrapper>& original_ops, size_t start_index,
    TensorPool& tensor_pool, size_t pattern_size) {
  const auto& pattern_input = original_ops[start_index].GetInputTensor(0);
  const auto& pattern_output =
      original_ops[start_index + pattern_size - 1].GetOutputTensor(0);
  const auto& quant_param = pattern_output.GetQuantParams();
  const auto& tensor_dims = pattern_input.GetDimensions();
  const std::uint32_t num_element = pattern_input.GetTensorNumElements();

  std::vector<std::int16_t> all_zero_data(num_element, 0);
  const auto& input_1 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, tensor_dims,
      all_zero_data.size() * sizeof(all_zero_data[0]), all_zero_data.data());

  const auto& mul_static_tensor =
      original_ops[start_index + pattern_size - 1].GetInputTensor(1);
  auto static_tensor_data = mul_static_tensor.GetTensorData<std::int16_t>();
  if (!static_tensor_data) {
    QNN_LOG_ERROR("[G2G] Get tensor data failed when transforming mask model.");
    return std::nullopt;
  }
  std::vector<std::int16_t> mask_data(num_element,
                                      static_tensor_data.value()[0]);
  const auto& input_2 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, tensor_dims,
      mask_data.size() * sizeof(mask_data[0]), mask_data.data());
  return CreateSelectOp(pattern_input, input_1, input_2, pattern_output);
}

std::optional<OpWrapper> CreateGemma4SelectOp(
    const std::vector<OpWrapper>& original_ops, size_t start_index,
    TensorPool& tensor_pool, size_t pattern_size) {
  const auto& pattern_input = original_ops[start_index].GetInputTensor(0);
  const auto& tensor_dims = pattern_input.GetDimensions();
  const std::uint32_t num_element = pattern_input.GetTensorNumElements();

  const auto& pattern_output =
      original_ops[start_index + pattern_size - 1].GetOutputTensor(0);
  const auto& quant_param = pattern_output.GetQuantParams();

  // Verify the static tensor for the Mul op.
  const auto& mul_static_tensor =
      original_ops[start_index + pattern_size - 2].GetInputTensor(1);
  auto static_tensor_data = mul_static_tensor.GetTensorData<float>();
  if (!static_tensor_data || static_tensor_data->empty()) {
    QNN_LOG_ERROR("[G2G] Get tensor data failed when transforming mask model.");
    return std::nullopt;
  }

  if (auto scale_offset =
          std::get_if<ScaleOffsetQuantizeParamsWrapper>(&quant_param)) {
    const float actual_min = (*static_tensor_data)[0];
    const float expected_min =
        scale_offset->GetScale() * (std::numeric_limits<std::int8_t>::min() -
                                    scale_offset->GetZeroPoint());
    if (expected_min != actual_min) {
      QNN_LOG_WARNING(
          "[G2G] Static min value in Mul (%f) does not match expected value "
          "(%f)",
          actual_min, expected_min)
      return std::nullopt;
    }
  }

  std::vector<std::int8_t> all_max_data(
      num_element, std::numeric_limits<std::int8_t>::max());
  const auto& input_1 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, quant_param, tensor_dims,
      all_max_data.size() * sizeof(all_max_data[0]), all_max_data.data());

  const std::vector<std::int8_t> all_min_data(
      num_element, std::numeric_limits<std::int8_t>::min());
  const auto& input_2 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, quant_param, tensor_dims,
      all_min_data.size() * sizeof(all_min_data[0]), all_min_data.data());
  return CreateSelectOp(pattern_input, input_1, input_2, pattern_output);
}
}  // namespace

size_t TransformQuantizeInMask(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Connection check
  bool is_connected = ops[start_index + 0].GetOutputTensor(0) ==
                          ops[start_index + 1].GetInputTensor(0) &&
                      ops[start_index + 1].GetOutputTensor(0) ==
                          ops[start_index + 2].GetInputTensor(0) &&
                      ops[start_index + 2].GetOutputTensor(0) ==
                          ops[start_index + 3].GetInputTensor(0);
  if (!is_connected) {
    return 1;
  }

  if (!IsElementWiseNot(ops[start_index + 0])) {
    return 1;
  }

  // Construct the new subgraph.
  QNN_LOG_INFO("[G2G] Transform quant ops in Gemma mask models");
  std::optional<OpWrapper> select{};
  if (IsElementWiseMultiply(ops[start_index + 2])) {
    select = CreateGemma4SelectOp(ops, start_index, tensor_pool, pattern_size);
  } else if (IsElementWiseMultiply(ops[start_index + 3])) {
    select = CreateGemma3SelectOp(ops, start_index, tensor_pool, pattern_size);
  }
  if (!select) {
    QNN_LOG_WARNING("[G2G] Failed to construct transformed Select op.");
    return 1;
  }

  // Validate new graph.
  if (validate_op_config(select.value())) {
    // Replace the matched pattern with a newly generated subgraph.
    ops.insert(ops.begin() + start_index + pattern_size, std::move(*select));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    return 1;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

}  // namespace qnn
