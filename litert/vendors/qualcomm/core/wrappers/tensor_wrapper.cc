// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

std::size_t GetDataTypeSize(const Qnn_DataType_t data_type) {
  std::size_t bytes = 0;
  switch (data_type) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      bytes = 1;
      break;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      bytes = 2;
      break;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      bytes = 4;
      break;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_FLOAT_64:
      bytes = 8;
      break;
    case QNN_DATATYPE_UNDEFINED:
    case QNN_DATATYPE_SFIXED_POINT_4:
    case QNN_DATATYPE_UFIXED_POINT_4:
    default:
      bytes = 0;
      break;
  }
  return bytes;
}

TensorWrapper::TensorWrapper() = default;

TensorWrapper::TensorWrapper(
    std::string name, Qnn_TensorType_t tensor_type, Qnn_DataType_t data_type,
    const QuantizeParamsWrapperVariant& quantize_params,
    const std::vector<std::uint32_t>& dimentions)
    : name_{std::move(name)},
      dimentions_{dimentions},
      quantize_params_{quantize_params} {
  qnn_tensor_.v2.name = name_.c_str();
  qnn_tensor_.v2.type = tensor_type;
  qnn_tensor_.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  qnn_tensor_.v2.dataType = data_type;
  UpdateQnnQuantParams();
  qnn_tensor_.v2.rank = dimentions_.size();
  qnn_tensor_.v2.dimensions = dimentions_.data();
  qnn_tensor_.v2.memType = QNN_TENSORMEMTYPE_RAW;
}

void TensorWrapper::ConvertAxisScaleOffsetToScaleOffset() {
  if (!std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
          quantize_params_)) {
    return;
  }
  quantize_params_.emplace<ScaleOffsetQuantizeParamsWrapper>(0.0, 0);
  UpdateQnnQuantParams();
}

TensorWrapper::TensorWrapper(
    std::string name, Qnn_TensorType_t tensor_type, Qnn_DataType_t data_type,
    const QuantizeParamsWrapperVariant& quantize_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t bytes,
    const void* data)
    : TensorWrapper(std::move(name), tensor_type, data_type, quantize_params,
                    dimentions) {
  // Use QNN_DATATYPE_SFIXED_POINT_8 for 4 bit quantization
  if (data_type == QNN_DATATYPE_SFIXED_POINT_4) {
    QNN_LOG_DEBUG("4bit Qunat, converting 4bit data to 8bit for QNN.");
    SetDataType(QNN_DATATYPE_SFIXED_POINT_8);
    std::vector<std::int8_t> int8_data;
    ConvertDataFromInt4ToInt8(data, int8_data, bytes);
    SetDataBy(GetTensorBytes(), int8_data.data());
  } else {
    SetDataBy(bytes, data);
  }
}

TensorWrapper::TensorWrapper(const TensorWrapper& other)
    : qnn_tensor_{other.qnn_tensor_},
      name_{other.name_},
      dimentions_{other.dimentions_},
      quantize_params_{other.quantize_params_},
      owned_data_{other.owned_data_} {
  qnn_tensor_.v2.name = name_.c_str();
  qnn_tensor_.v2.dimensions = dimentions_.data();
  qnn_tensor_.v2.clientBuf.data = owned_data_.data();
  UpdateQnnQuantParams();
}

TensorWrapper::TensorWrapper(TensorWrapper&& other)
    : qnn_tensor_{other.qnn_tensor_},
      name_{std::move(other.name_)},
      dimentions_{std::move(other.dimentions_)},
      quantize_params_{std::move(other.quantize_params_)},
      owned_data_{std::move(other.owned_data_)} {
  qnn_tensor_.v2.name = name_.c_str();
  qnn_tensor_.v2.dimensions = dimentions_.data();
  qnn_tensor_.v2.clientBuf.data = owned_data_.data();
  UpdateQnnQuantParams();
}

TensorWrapper::~TensorWrapper() = default;

std::uint32_t TensorWrapper::GetDim(size_t index) const {
  return dimentions_[index];
}

Qnn_DataType_t TensorWrapper::GetDataType() const {
  return qnn_tensor_.v2.dataType;
}

void TensorWrapper::CloneTo(Qnn_Tensor_t& dst) const { dst = qnn_tensor_; }

std::uint32_t TensorWrapper::GetRank() const { return qnn_tensor_.v2.rank; }

Qnn_TensorType_t TensorWrapper::GetTensorType() const {
  return qnn_tensor_.v2.type;
}

std::uint32_t TensorWrapper::GetTensorNumElements() const {
  return GetDims().empty() ? 0
                           : std::accumulate(GetDims().begin(), GetDims().end(),
                                             1, std::multiplies<>());
}

size_t TensorWrapper::GetTensorBytes() const {
  return GetDataTypeSize(GetDataType()) * GetTensorNumElements();
}

bool TensorWrapper::IsPerTensorQuantWithOffsetDiff(
    const TensorWrapper& rhs) const {
  const auto& lhs_quant = qnn_tensor_.v2.quantizeParams;
  const auto& rhs_quant = rhs.qnn_tensor_.v2.quantizeParams;

  if (lhs_quant.encodingDefinition != QNN_DEFINITION_DEFINED ||
      rhs_quant.encodingDefinition != QNN_DEFINITION_DEFINED) {
    return false;
  }

  if (lhs_quant.quantizationEncoding !=
          QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
      rhs_quant.quantizationEncoding !=
          QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    return false;
  }

  const auto lhs_scale = lhs_quant.scaleOffsetEncoding.scale;
  const auto lhs_offset = lhs_quant.scaleOffsetEncoding.offset;
  const auto rhs_scale = rhs_quant.scaleOffsetEncoding.scale;
  const auto rhs_offset = rhs_quant.scaleOffsetEncoding.offset;
  if ((GetDataType() == QNN_DATATYPE_SFIXED_POINT_8 &&
       rhs.GetDataType() == QNN_DATATYPE_UFIXED_POINT_8) ||
      (GetDataType() == QNN_DATATYPE_UFIXED_POINT_8 &&
       rhs.GetDataType() == QNN_DATATYPE_SFIXED_POINT_8)) {
    constexpr int kSUFixed8OffsetDiff = 128;
    if (std::fabs(lhs_scale - rhs_scale) <
            std::numeric_limits<float>::epsilon() &&
        std::abs(lhs_offset - rhs_offset) == kSUFixed8OffsetDiff) {
      return true;
    }
  } else if ((GetDataType() == QNN_DATATYPE_SFIXED_POINT_16 &&
              rhs.GetDataType() == QNN_DATATYPE_UFIXED_POINT_16) ||
             (GetDataType() == QNN_DATATYPE_UFIXED_POINT_16 &&
              rhs.GetDataType() == QNN_DATATYPE_SFIXED_POINT_16)) {
    constexpr int kSUFixed16OffsetDiff = 32768;
    if (std::fabs(lhs_scale - rhs_scale) <
            std::numeric_limits<float>::epsilon() &&
        std::abs(lhs_offset - rhs_offset) == kSUFixed16OffsetDiff) {
      return true;
    }
  }
  return false;
}

void TensorWrapper::SetDataBy(std::uint32_t bytes, const void* data) {
  if (bytes != GetTensorBytes()) {
    QNN_LOG_WARNING(
        "Bytes: %d != GetTensorBytes(): %d, use GetTensorBytes() instead.",
        bytes, GetTensorBytes());
    bytes = GetTensorBytes();
  }
  owned_data_.resize(bytes);
  std::memcpy(owned_data_.data(), reinterpret_cast<const char*>(data), bytes);
  qnn_tensor_.v2.clientBuf.dataSize = owned_data_.size();
  qnn_tensor_.v2.clientBuf.data = owned_data_.data();
}

void TensorWrapper::ConvertQint16ToQuint16() {
  if (GetDataType() != QNN_DATATYPE_SFIXED_POINT_16) {
    return;
  }

  // adjust static data
  if (IsTensorStatic()) {
    auto int16_data = GetTensorData<std::int16_t>();
    if (!int16_data.has_value()) {
      QNN_LOG_ERROR(
          "Cannot convert static QInt16 data to QUint16 data failed since "
          "GetTensorData failed.");
      return;
    }
    QNN_LOG_DEBUG("Converting static tensor data from QInt16 to QUint16...");
    std::vector<std::uint16_t> uint16_data;
    ConvertDataFromInt16toUInt16((*int16_data), uint16_data);
    std::memcpy(owned_data_.data(),
                reinterpret_cast<const char*>(uint16_data.data()),
                GetTensorBytes());
    qnn_tensor_.v2.clientBuf.dataSize = owned_data_.size();
    qnn_tensor_.v2.clientBuf.data = owned_data_.data();
  }

  // adjust quant param;
  if (IsPerTensorQuant()) {
    const auto& q_param =
        std::get<ScaleOffsetQuantizeParamsWrapper>(GetQuantParams());
    quantize_params_.emplace<ScaleOffsetQuantizeParamsWrapper>(
        q_param.GetScale(), q_param.GetZeroPoint() + kUint16ZeroPoint);

  } else if (IsPerChannelQuant()) {
    const auto& q_param =
        std::get<AxisScaleOffsetQuantizeParamsWrapper>(GetQuantParams());
    std::int32_t axis = q_param.GetAxis();
    std::vector<float> scales;
    q_param.GetScales(scales);
    std::vector<std::int32_t> zero_points;
    q_param.GetZeroPoints(zero_points);
    std::for_each(zero_points.begin(), zero_points.end(),
                  [](std::int32_t& val) { val += kUint16ZeroPoint; });
    quantize_params_.emplace<AxisScaleOffsetQuantizeParamsWrapper>(
        axis, absl::MakeSpan(scales), absl::MakeSpan(zero_points));
  }

  UpdateQnnQuantParams();

  // change data type here since GetTensorData checks data type
  qnn_tensor_.v2.dataType = QNN_DATATYPE_UFIXED_POINT_16;
  QNN_LOG_DEBUG(
      "QNN does not fully support QInt16 now, converting to QUint16 for better "
      "compatibility.");
}

TensorWrapper::TensorWrapper(const Qnn_Tensor_t& qnn_tensor)
    : qnn_tensor_{qnn_tensor} {
  if (qnn_tensor_.version == QNN_TENSOR_VERSION_1) {
    name_ = qnn_tensor_.v1.name;
    qnn_tensor_.v1.name = name_.data();
    dimentions_.reserve(qnn_tensor_.v1.rank);
    std::copy(
        qnn_tensor_.v1.dimensions,
        qnn_tensor_.v1.dimensions + qnn_tensor_.v1.rank,
        std::back_insert_iterator<std::vector<std::uint32_t>>(dimentions_));
    qnn_tensor_.v1.dimensions = dimentions_.data();
    if (const auto& quant_params = qnn_tensor_.v1.quantizeParams;
        quant_params.encodingDefinition == QNN_DEFINITION_DEFINED) {
      if (quant_params.quantizationEncoding ==
          QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        quantize_params_.emplace<ScaleOffsetQuantizeParamsWrapper>(
            quant_params.scaleOffsetEncoding);
      } else if (quant_params.quantizationEncoding ==
                 QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        quantize_params_.emplace<AxisScaleOffsetQuantizeParamsWrapper>(
            quant_params.axisScaleOffsetEncoding);
      } else {
        QNN_LOG_ERROR("Unsupported quantization encoding: %d",
                      quant_params.quantizationEncoding);
      }
    }
    UpdateQnnQuantParams();
  } else if (qnn_tensor_.version == Qnn_TensorVersion_t::QNN_TENSOR_VERSION_2) {
    // TODO: support v2 only
    name_ = qnn_tensor_.v2.name;
    qnn_tensor_.v2.name = name_.data();
    dimentions_.reserve(qnn_tensor_.v2.rank);
    std::copy(
        qnn_tensor_.v2.dimensions,
        qnn_tensor_.v2.dimensions + qnn_tensor_.v2.rank,
        std::back_insert_iterator<std::vector<std::uint32_t>>(dimentions_));
    qnn_tensor_.v2.dimensions = dimentions_.data();
    if (const auto& quant_params = qnn_tensor_.v2.quantizeParams;
        quant_params.encodingDefinition == QNN_DEFINITION_DEFINED) {
      if (quant_params.quantizationEncoding ==
          QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        quantize_params_.emplace<ScaleOffsetQuantizeParamsWrapper>(
            quant_params.scaleOffsetEncoding);
      } else if (quant_params.quantizationEncoding ==
                 QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        quantize_params_.emplace<AxisScaleOffsetQuantizeParamsWrapper>(
            quant_params.axisScaleOffsetEncoding);
      } else {
        QNN_LOG_ERROR("Unsupported quantization encoding: %d",
                      quant_params.quantizationEncoding);
      }
    }
    std::visit(
        [this](auto&& quantize_params) -> void {
          quantize_params.CloneTo(qnn_tensor_.v2.quantizeParams);
        },
        quantize_params_);
  } else {
    // TODO: tensor.v3
  }
}

}  // namespace qnn
