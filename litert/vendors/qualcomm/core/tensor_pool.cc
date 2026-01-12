// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/tensor_pool.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

TensorPool::TensorPool() = default;

TensorWrapper& TensorPool::CreateInputTensorWithSuffix(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimensions, std::string_view suffix) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + std::string(suffix);
  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_APP_WRITE);
  tensor.SetDataType(data_type);
  tensor.SetQuantizeParams(quant_params);
  tensor.SetDimensions(dimensions);
  return tensor;
}

TensorWrapper& TensorPool::CreateOutpuTensorWithSuffix(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimensions, std::string_view suffix) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + std::string(suffix);
  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_APP_READ);
  tensor.SetDataType(data_type);
  tensor.SetQuantizeParams(quant_params);
  tensor.SetDimensions(dimensions);
  return tensor;
}

TensorWrapper& TensorPool::CreateNativeTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimensions) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_NATIVE);
  tensor.SetDataType(data_type);
  tensor.SetQuantizeParams(quant_params);
  tensor.SetDimensions(dimensions);
  return tensor;
}

TensorWrapper& TensorPool::CreateNativeTensorWithSuffix(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimensions, std::string_view suffix) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + std::string(suffix);
  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_NATIVE);
  tensor.SetDataType(data_type);
  tensor.SetQuantizeParams(quant_params);
  tensor.SetDimensions(dimensions);
  return tensor;
}

TensorWrapper& TensorPool::CreateStaticTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimensions, std::uint32_t bytes,
    const void* data) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_STATIC);
  tensor.SetDataType(data_type);
  tensor.SetQuantizeParams(quant_params);
  tensor.SetDimensions(dimensions);
  tensor.SetData(bytes, data, true);
  return tensor;
}

TensorWrapper* TensorPool::CreateStaticTensorWithValue(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimensions, float fill_value) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;

  std::uint32_t num_data_element = std::accumulate(
      dimensions.begin(), dimensions.end(), 1, std::multiplies<>());
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
    switch (data_type) {
      case Qnn_DataType_t::QNN_DATATYPE_INT_32: {
        if (fill_value < std::numeric_limits<std::int32_t>::min() ||
            fill_value > std::numeric_limits<std::int32_t>::max()) {
          QNN_LOG_ERROR(
              "Fill value out of range when CreateStaticTensorWithValue.");
        }
        tensor_data = std::vector<std::int32_t>(num_data_element, fill_value);
        data_ptr = std::get<std::vector<std::int32_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_UINT_32: {
        if (fill_value < std::numeric_limits<std::uint32_t>::min() ||
            fill_value > std::numeric_limits<std::uint32_t>::max()) {
          QNN_LOG_ERROR(
              "Fill value out of range when CreateStaticTensorWithValue.");
        }
        tensor_data = std::vector<std::uint32_t>(num_data_element, fill_value);
        data_ptr = std::get<std::vector<std::uint32_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_FLOAT_32: {
        tensor_data = std::vector<float>(num_data_element, fill_value);
        data_ptr = std::get<std::vector<float>>(tensor_data).data();
        break;
      }
      default: {
        QNN_LOG_ERROR(
            "Unsupported QNN data type when CreateStaticTensorWithValue.");
        return nullptr;
      }
    }
  } else if (std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
                 quant_params)) {
    std::int32_t zero_point =
        std::get<ScaleOffsetQuantizeParamsWrapper>(quant_params).GetZeroPoint();
    float scale =
        std::get<ScaleOffsetQuantizeParamsWrapper>(quant_params).GetScale();
    switch (data_type) {
      case Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8: {
        std::int8_t pad_const_value =
            Quantize<std::int8_t>(fill_value, scale, zero_point);
        tensor_data =
            std::vector<std::int8_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::int8_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_8: {
        std::uint8_t pad_const_value =
            Quantize<std::uint8_t>(fill_value, scale, zero_point);
        tensor_data =
            std::vector<std::uint8_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::uint8_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_16: {
        std::int16_t pad_const_value =
            Quantize<std::int16_t>(fill_value, scale, zero_point);
        tensor_data =
            std::vector<std::int16_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::int16_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_16: {
        std::uint16_t pad_const_value =
            Quantize<std::uint16_t>(fill_value, scale, zero_point);
        tensor_data =
            std::vector<std::uint16_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::uint16_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_32: {
        std::int32_t pad_const_value =
            Quantize<std::int32_t>(fill_value, scale, zero_point);
        tensor_data =
            std::vector<std::int32_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::int32_t>>(tensor_data).data();
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_32: {
        std::uint32_t pad_const_value =
            Quantize<std::uint32_t>(fill_value, scale, zero_point);
        tensor_data =
            std::vector<std::uint32_t>(num_data_element, pad_const_value);
        data_ptr = std::get<std::vector<std::uint32_t>>(tensor_data).data();
        break;
      }
      default: {
        QNN_LOG_ERROR(
            "Unsupported QNN data type when CreateStaticTensorWithValue.");
        return nullptr;
      }
    }
  } else {
    QNN_LOG_ERROR(
        "Unsupported quantization encoding when CreateStaticTensorWithValue.");
    return nullptr;
  }

  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_STATIC);
  tensor.SetDataType(data_type);
  tensor.SetQuantizeParams(quant_params);
  tensor.SetDimensions(dimensions);
  tensor.SetData(data_bytes, data_ptr, true);
  return &tensor;
}

TensorWrapper& TensorPool::CreateStaticTensorWithSuffix(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimensions, std::string_view suffix,
    std::uint32_t bytes, const void* data, bool copy_data) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + std::string(suffix);
  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_STATIC);
  tensor.SetDataType(data_type);
  tensor.SetQuantizeParams(quant_params);
  tensor.SetDimensions(dimensions);
  tensor.SetData(bytes, data, copy_data);
  return tensor;
}

TensorWrapper& TensorPool::CloneNativeTensorFrom(const TensorWrapper& src) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_NATIVE);
  tensor.SetDataType(src.GetDataType());
  tensor.SetQuantizeParams(src.quantize_params_);
  tensor.SetDimensions(src.dimensions_);
  return tensor;
}

TensorWrapper& TensorPool::CloneNativeTensorFrom(
    const TensorWrapper& src, const std::vector<std::uint32_t>& dimensions) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_NATIVE);
  tensor.SetDataType(src.GetDataType());
  tensor.SetQuantizeParams(src.quantize_params_);
  tensor.SetDimensions(dimensions);
  return tensor;
}

TensorWrapper& TensorPool::CloneStaticTensorFrom(const TensorWrapper& src,
                                                 Qnn_DataType_t data_type) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_STATIC);
  tensor.SetDataType(data_type);
  tensor.SetQuantizeParams(src.quantize_params_);
  tensor.SetDimensions(src.dimensions_);
  tensor.SetData(src.qnn_tensor_.v2.clientBuf.dataSize,
                 src.qnn_tensor_.v2.clientBuf.data, true);
  return tensor;
}

TensorWrapper& TensorPool::CloneStaticTensorFrom(
    const TensorWrapper& src, const std::vector<std::uint32_t>& dimensions) {
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  auto& tensor = tensor_wrappers_.emplace_back();
  tensor.SetName(std::move(tensor_name));
  tensor.SetTensorType(QNN_TENSOR_TYPE_STATIC);
  tensor.SetDataType(src.qnn_tensor_.v2.dataType);
  tensor.SetQuantizeParams(src.quantize_params_);
  tensor.SetDimensions(dimensions);
  tensor.SetData(src.qnn_tensor_.v2.clientBuf.dataSize,
                 src.qnn_tensor_.v2.clientBuf.data, true);
  return tensor;
}

}  // namespace qnn
