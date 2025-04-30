// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/tensor_pool.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

TensorPool::TensorPool() = default;

TensorWrapper& TensorPool::CreateInputTensorWithId(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t framework_id) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(
      id, kLiteRtPrefix + std::to_string(framework_id),
      QNN_TENSOR_TYPE_APP_WRITE, data_type, quant_params, dimentions);
}

TensorWrapper& TensorPool::CreateOutpuTensorWithId(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t framework_id) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(
      id, kLiteRtPrefix + std::to_string(framework_id),
      QNN_TENSOR_TYPE_APP_READ, data_type, quant_params, dimentions);
}

TensorWrapper& TensorPool::CreateNativeTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, kQnnPrefix, QNN_TENSOR_TYPE_NATIVE,
                                       data_type, quant_params, dimentions);
}

TensorWrapper& TensorPool::CreateNativeTensorWithId(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t framework_id) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(
      id, kLiteRtPrefix + std::to_string(framework_id), QNN_TENSOR_TYPE_NATIVE,
      data_type, quant_params, dimentions);
}

TensorWrapper& TensorPool::CreateStaticTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t bytes,
    const void* data) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, kQnnPrefix, QNN_TENSOR_TYPE_STATIC,
                                       data_type, quant_params, dimentions,
                                       bytes, data);
}

TensorWrapper& TensorPool::CreateStaticTensorWithId(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t framework_id,
    std::uint32_t bytes, const void* data) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(
      id, kLiteRtPrefix + std::to_string(framework_id), QNN_TENSOR_TYPE_STATIC,
      data_type, quant_params, dimentions, bytes, data);
}

TensorWrapper& TensorPool::CloneNativeTensorFrom(const TensorWrapper& src) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, kQnnPrefix, QNN_TENSOR_TYPE_NATIVE,
                                       src.GetDataType(), src.quantize_params_,
                                       src.dimentions_);
}

TensorWrapper& TensorPool::CloneNativeTensorFrom(
    const TensorWrapper& src, const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, kQnnPrefix, QNN_TENSOR_TYPE_NATIVE,
                                       src.GetDataType(), src.quantize_params_,
                                       dimentions);
}

TensorWrapper& TensorPool::CloneStaticTensorFrom(const TensorWrapper& src,
                                                 Qnn_DataType_t data_type) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(
      id, kQnnPrefix, QNN_TENSOR_TYPE_STATIC, data_type, src.quantize_params_,
      src.dimentions_, src.owned_data_.size(), src.owned_data_.data());
}

TensorWrapper& TensorPool::CloneStaticTensorFrom(
    const TensorWrapper& src, const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(
      id, kQnnPrefix, QNN_TENSOR_TYPE_STATIC, src.qnn_tensor_.v2.dataType,
      src.quantize_params_, dimentions, src.qnn_tensor_.v2.clientBuf.dataSize,
      src.qnn_tensor_.v2.clientBuf.data);
}

}  // namespace qnn
