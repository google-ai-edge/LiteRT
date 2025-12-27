// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/param_wrapper.h"

#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

bool ScalarParamWrapper::operator==(const ScalarParamWrapper& other) const {
  if (qnn_scalar_.dataType != other.qnn_scalar_.dataType) return false;

  switch (qnn_scalar_.dataType) {
    case QNN_DATATYPE_FLOAT_32:
      if (qnn_scalar_.floatValue != other.qnn_scalar_.floatValue) return false;
      break;
    case QNN_DATATYPE_BOOL_8:
      if (qnn_scalar_.bool8Value != other.qnn_scalar_.bool8Value) return false;
      break;
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_UINT_8:
      if (qnn_scalar_.uint8Value != other.qnn_scalar_.uint8Value) return false;
      break;
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_INT_8:
      if (qnn_scalar_.int8Value != other.qnn_scalar_.int8Value) return false;
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_UINT_16:
      if (qnn_scalar_.uint16Value != other.qnn_scalar_.uint16Value)
        return false;
      break;
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_INT_16:
      if (qnn_scalar_.int16Value != other.qnn_scalar_.int16Value) return false;
      break;
    case QNN_DATATYPE_UFIXED_POINT_32:
    case QNN_DATATYPE_UINT_32:
      if (qnn_scalar_.uint32Value != other.qnn_scalar_.uint32Value)
        return false;
      break;
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_INT_32:
      if (qnn_scalar_.int32Value != other.qnn_scalar_.int32Value) return false;
      break;
    default:
      QNN_LOG_ERROR(
          "Unsupported data type for comparing scalar param: input: %#x, "
          "golden: %#x",
          qnn_scalar_.dataType, other.qnn_scalar_.dataType);
      return false;
  }

  return true;
}

void ScalarParamWrapper::CloneTo(Qnn_Param_t& dst) const {
  dst.name = name_;
  dst.paramType = QNN_PARAMTYPE_SCALAR;
  dst.scalarParam = qnn_scalar_;
}

TensorParamWrapper::TensorParamWrapper(const char* name,
                                       const TensorWrapper& tensor)
    : name_{name}, tensor_{tensor} {}

bool TensorParamWrapper::operator==(const TensorParamWrapper& other) const {
  return tensor_ == other.tensor_;
}

void TensorParamWrapper::CloneTo(Qnn_Param_t& dst) const {
  dst.name = name_;
  dst.paramType = QNN_PARAMTYPE_TENSOR;
  tensor_.CloneTo(dst.tensorParam);
}

const TensorWrapper& TensorParamWrapper::GetTensor() const { return tensor_; }

}  // namespace qnn
