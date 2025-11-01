// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/tensor_pool.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

TensorPool::TensorPool() = default;

TensorWrapper& TensorPool::CreateInputTensorWithSuffix(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::string_view suffix) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + std::string(suffix);
  return tensor_wrappers_.emplace_back(std::move(tensor_name),
                                       QNN_TENSOR_TYPE_APP_WRITE, data_type,
                                       quant_params, dimentions);
}

TensorWrapper& TensorPool::CreateOutpuTensorWithSuffix(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::string_view suffix) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + std::string(suffix);
  return tensor_wrappers_.emplace_back(std::move(tensor_name),
                                       QNN_TENSOR_TYPE_APP_READ, data_type,
                                       quant_params, dimentions);
}

TensorWrapper& TensorPool::CreateNativeTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  return tensor_wrappers_.emplace_back(std::move(tensor_name),
                                       QNN_TENSOR_TYPE_NATIVE, data_type,
                                       quant_params, dimentions);
}

TensorWrapper& TensorPool::CreateNativeTensorWithSuffix(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::string_view suffix) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + std::string(suffix);
  return tensor_wrappers_.emplace_back(std::move(tensor_name),
                                       QNN_TENSOR_TYPE_NATIVE, data_type,
                                       quant_params, dimentions);
}

TensorWrapper& TensorPool::CreateStaticTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t bytes,
    const void* data) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  return tensor_wrappers_.emplace_back(std::move(tensor_name),
                                       QNN_TENSOR_TYPE_STATIC, data_type,
                                       quant_params, dimentions, bytes, data);
}

TensorWrapper* TensorPool::CreateStaticTensorWithValue(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, float fill_value) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;

  std::uint32_t num_data_element = std::accumulate(
      dimentions.begin(), dimentions.end(), 1, std::multiplies<>());
  const std::uint32_t data_bytes =
      num_data_element * GetDataTypeSize(data_type);

  using TensorDataVariant =
      std::variant<std::vector<std::int8_t>, std::vector<std::uint8_t>,
                   std::vector<std::int16_t>, std::vector<std::uint16_t>,
                   std::vector<std::int32_t>, std::vector<std::uint32_t>,
                   std::vector<float>>;

  TensorDataVariant tensor_data;
  const void* data_ptr;
  if (std::holds_alternative<UndefinedQuantizeParamsWrapper>(quant_params)) {
    tensor_data = std::vector<float>(num_data_element, fill_value);
    data_ptr = std::get<std::vector<float>>(tensor_data).data();
  } else if (std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
                 quant_params)) {
    std::int32_t offset =
        std::get<ScaleOffsetQuantizeParamsWrapper>(quant_params).GetOffset();

    float scale =
        std::get<ScaleOffsetQuantizeParamsWrapper>(quant_params).GetScale();

    switch (data_type) {
      case Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8: {
        std::int8_t pad_const_value = std::round(fill_value / scale + offset);

        tensor_data =
            std::vector<std::int8_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::int8_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_8: {
        std::uint8_t pad_const_value = std::round(fill_value / scale + offset);
        tensor_data =
            std::vector<std::uint8_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::uint8_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_16: {
        std::int16_t pad_const_value = std::round(fill_value / scale + offset);

        tensor_data =
            std::vector<std::int16_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::int16_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_16: {
        std::uint16_t pad_const_value = std::round(fill_value / scale + offset);
        tensor_data =
            std::vector<std::uint16_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::uint16_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_32: {
        std::int32_t pad_const_value = std::round(fill_value / scale + offset);
        tensor_data =
            std::vector<std::int32_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::int32_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_32: {
        std::uint32_t pad_const_value = std::round(fill_value / scale + offset);
        tensor_data =
            std::vector<std::uint32_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::uint32_t>>(tensor_data).data();
        break;
      }
      default: {
        // Unsupported qnn data type
        QNN_LOG_ERROR(
            "Unsupported QNN data type when CreateStaticTensorWithValue.");
        return nullptr;
      }
    }
  } else {
    // Unsuported quantization encoding
    QNN_LOG_ERROR(
        "Unsupported quantization encoding when CreateStaticTensorWithValue.");
    return nullptr;
  }

  return &tensor_wrappers_.emplace_back(
      std::move(tensor_name), QNN_TENSOR_TYPE_STATIC, data_type, quant_params,
      dimentions, data_bytes, data_ptr);
}

TensorWrapper& TensorPool::CreateStaticTensorWithSuffix(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::string_view suffix,
    std::uint32_t bytes, const void* data) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + std::string(suffix);
  return tensor_wrappers_.emplace_back(std::move(tensor_name),
                                       QNN_TENSOR_TYPE_STATIC, data_type,
                                       quant_params, dimentions, bytes, data);
}

TensorWrapper& TensorPool::CloneNativeTensorFrom(const TensorWrapper& src) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  return tensor_wrappers_.emplace_back(
      std::move(tensor_name), QNN_TENSOR_TYPE_NATIVE, src.GetDataType(),
      src.quantize_params_, src.dimentions_);
}

TensorWrapper& TensorPool::CloneNativeTensorFrom(
    const TensorWrapper& src, const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  return tensor_wrappers_.emplace_back(
      std::move(tensor_name), QNN_TENSOR_TYPE_NATIVE, src.GetDataType(),
      src.quantize_params_, dimentions);
}

TensorWrapper& TensorPool::CloneStaticTensorFrom(const TensorWrapper& src,
                                                 Qnn_DataType_t data_type) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  return tensor_wrappers_.emplace_back(
      std::move(tensor_name), QNN_TENSOR_TYPE_STATIC, data_type,
      src.quantize_params_, src.dimentions_, src.owned_data_.size(),
      src.owned_data_.data());
}

TensorWrapper& TensorPool::CloneStaticTensorFrom(
    const TensorWrapper& src, const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  return tensor_wrappers_.emplace_back(
      std::move(tensor_name), QNN_TENSOR_TYPE_STATIC,
      src.qnn_tensor_.v2.dataType, src.quantize_params_, dimentions,
      src.qnn_tensor_.v2.clientBuf.dataSize, src.qnn_tensor_.v2.clientBuf.data);
}

}  // namespace qnn
