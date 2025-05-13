// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/tensor_pool.h"

#include <atomic>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
std::atomic<std::uint32_t> qnn_tensor_id{
    std::numeric_limits<std::uint32_t>::max()};

std::uint32_t CreateQnnTensorId() { return --qnn_tensor_id; }

TensorPool::TensorPool() = default;

// prefix: Tensor created from framework will have no prefix (""), tensor
// created by qnn will have prefix kQnnPrefix.
// id: Used to ensure uniqueness of qnn tensor name.
// framework_id: The tensor index imported from framework.
TensorWrapper& TensorPool::CreateInputTensorWithId(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t framework_id) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, "", QNN_TENSOR_TYPE_APP_WRITE,
                                       data_type, quant_params, dimentions,
                                       framework_id);
}

TensorWrapper& TensorPool::CreateOutpuTensorWithId(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t framework_id) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, "", QNN_TENSOR_TYPE_APP_READ,
                                       data_type, quant_params, dimentions,
                                       framework_id);
}

TensorWrapper& TensorPool::CreateNativeTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, kQnnPrefix, QNN_TENSOR_TYPE_NATIVE,
                                       data_type, quant_params, dimentions,
                                       CreateQnnTensorId());
}

TensorWrapper& TensorPool::CreateNativeTensorWithId(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t framework_id) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, "", QNN_TENSOR_TYPE_NATIVE,
                                       data_type, quant_params, dimentions,
                                       framework_id);
}

TensorWrapper& TensorPool::CreateStaticTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t bytes,
    const void* data) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, kQnnPrefix, QNN_TENSOR_TYPE_STATIC,
                                       data_type, quant_params, dimentions,
                                       CreateQnnTensorId(), bytes, data);
}

TensorWrapper& TensorPool::CreateStaticTensorWithId(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t framework_id,
    std::uint32_t bytes, const void* data) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, "", QNN_TENSOR_TYPE_STATIC,
                                       data_type, quant_params, dimentions,
                                       framework_id, bytes, data);
}

TensorWrapper& TensorPool::CloneNativeTensorFrom(const TensorWrapper& src) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, kQnnPrefix, QNN_TENSOR_TYPE_NATIVE,
                                       src.GetDataType(), src.quantize_params_,
                                       src.dimentions_, CreateQnnTensorId());
}

TensorWrapper& TensorPool::CloneNativeTensorFrom(
    const TensorWrapper& src, const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(id, kQnnPrefix, QNN_TENSOR_TYPE_NATIVE,
                                       src.GetDataType(), src.quantize_params_,
                                       dimentions, CreateQnnTensorId());
}

TensorWrapper& TensorPool::CloneStaticTensorFrom(const TensorWrapper& src,
                                                 Qnn_DataType_t data_type) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(
      id, kQnnPrefix, QNN_TENSOR_TYPE_STATIC, data_type, src.quantize_params_,
      src.dimentions_, CreateQnnTensorId(), src.owned_data_.size(),
      src.owned_data_.data());
}

TensorWrapper& TensorPool::CloneStaticTensorFrom(
    const TensorWrapper& src, const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  return tensor_wrappers_.emplace_back(
      id, kQnnPrefix, QNN_TENSOR_TYPE_STATIC, src.qnn_tensor_.v2.dataType,
      src.quantize_params_, dimentions, CreateQnnTensorId(),
      src.qnn_tensor_.v2.clientBuf.dataSize, src.qnn_tensor_.v2.clientBuf.data);
}

}  // namespace qnn
