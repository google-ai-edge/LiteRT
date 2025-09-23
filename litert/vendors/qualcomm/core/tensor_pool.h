// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TENSOR_POOL_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TENSOR_POOL_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
static const char* kQnnSuffix = "_qnn";
class TensorPool {
 public:
  TensorPool();
  // id: Number of tensors in TensorPool, used for ensure uniqueness of tensor
  // name Tensors directly converted from framework: <id>_suffix_from_framework
  // Tensors created by builder: <id>_qnn
  TensorWrapper& CreateInputTensorWithSuffix(
      Qnn_DataType_t data_type,
      const QuantizeParamsWrapperVariant& quant_params,
      const std::vector<std::uint32_t>& dimentions, std::string_view suffix);

  TensorWrapper& CreateOutpuTensorWithSuffix(
      Qnn_DataType_t data_type,
      const QuantizeParamsWrapperVariant& quant_params,
      const std::vector<std::uint32_t>& dimentions, std::string_view suffix);

  TensorWrapper& CreateNativeTensor(
      Qnn_DataType_t data_type,
      const QuantizeParamsWrapperVariant& quant_params,
      const std::vector<std::uint32_t>& dimentions);

  TensorWrapper& CreateNativeTensorWithSuffix(
      Qnn_DataType_t data_type,
      const QuantizeParamsWrapperVariant& quant_params,
      const std::vector<std::uint32_t>& dimentions, std::string_view suffix);

  TensorWrapper& CreateStaticTensor(
      Qnn_DataType_t data_type,
      const QuantizeParamsWrapperVariant& quant_params,
      const std::vector<std::uint32_t>& dimentions, std::uint32_t bytes,
      const void* data);

  TensorWrapper& CreateStaticTensorWithSuffix(
      Qnn_DataType_t data_type,
      const QuantizeParamsWrapperVariant& quant_params,
      const std::vector<std::uint32_t>& dimentions, std::string_view suffix,
      std::uint32_t bytes, const void* data);

  TensorWrapper& CloneNativeTensorFrom(const TensorWrapper& src);

  TensorWrapper& CloneNativeTensorFrom(
      const TensorWrapper& src, const std::vector<std::uint32_t>& dimentions);

  TensorWrapper& CloneStaticTensorFrom(const TensorWrapper& src,
                                       Qnn_DataType_t data_type);

  TensorWrapper& CloneStaticTensorFrom(
      const TensorWrapper& src, const std::vector<std::uint32_t>& dimentions);

  template <typename T>
  TensorWrapper* ConvertStaticTensorFrom(const TensorWrapper& src_tensor);

  template <typename UnaryFunc>
  void ForEach(UnaryFunc f) {
    for (auto& tensor_wrapper : tensor_wrappers_) {
      f(tensor_wrapper);
    }
  }

 private:
  std::list<TensorWrapper> tensor_wrappers_{};
};

namespace {

template <typename Src, typename Dst>
bool FillData(const TensorWrapper& src_tensor, std::vector<Dst>& dst_data) {
  const auto src_data = src_tensor.GetTensorData<Src>();
  if (!src_data.has_value()) {
    QNN_LOG_ERROR("Failed to get static tensor data when filling data.");
    return false;
  }

  dst_data.clear();
  dst_data.reserve(src_data->size());
  for (size_t i = 0; i < src_data->size(); ++i) {
    if ((*src_data)[i] > std::numeric_limits<Dst>::max() ||
        (*src_data)[i] < std::numeric_limits<Dst>::lowest()) {
      QNN_LOG_ERROR("Source data exceeds the range of destination data type.");

      dst_data.clear();
      return false;
    }

    dst_data.emplace_back((*src_data)[i]);
  }
  return true;
}

}  // namespace

template <typename T>
TensorWrapper* TensorPool::ConvertStaticTensorFrom(
    const TensorWrapper& src_tensor) {
  if (!src_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("Cannot convert non-static tensor to static tensor.");
    return nullptr;
  }

  std::vector<T> dst_data{};
  bool fill_result = true;
  if (const auto src_data_type = src_tensor.GetDataType();
      src_data_type == QNN_DATATYPE_BOOL_8) {
    fill_result = FillData<bool, T>(src_tensor, dst_data);
  } else if (src_data_type == QNN_DATATYPE_INT_8 ||
             src_data_type == QNN_DATATYPE_SFIXED_POINT_8) {
    fill_result = FillData<std::int8_t, T>(src_tensor, dst_data);
  } else if (src_data_type == QNN_DATATYPE_UINT_8 ||
             src_data_type == QNN_DATATYPE_UFIXED_POINT_8) {
    fill_result = FillData<std::uint8_t, T>(src_tensor, dst_data);
  } else if (src_data_type == QNN_DATATYPE_INT_16 ||
             src_data_type == QNN_DATATYPE_SFIXED_POINT_16) {
    fill_result = FillData<std::int16_t, T>(src_tensor, dst_data);
  } else if (src_data_type == QNN_DATATYPE_UINT_16 ||
             src_data_type == QNN_DATATYPE_UFIXED_POINT_16) {
    fill_result = FillData<std::uint16_t, T>(src_tensor, dst_data);
  } else if (src_data_type == QNN_DATATYPE_INT_32 ||
             src_data_type == QNN_DATATYPE_SFIXED_POINT_32) {
    fill_result = FillData<std::int32_t, T>(src_tensor, dst_data);
  } else if (src_data_type == QNN_DATATYPE_UINT_32 ||
             src_data_type == QNN_DATATYPE_UFIXED_POINT_32) {
    fill_result = FillData<std::uint32_t, T>(src_tensor, dst_data);
  } else if (src_data_type == QNN_DATATYPE_INT_64) {
    fill_result = FillData<std::int64_t, T>(src_tensor, dst_data);
  } else if (src_data_type == QNN_DATATYPE_UINT_64) {
    fill_result = FillData<std::uint64_t, T>(src_tensor, dst_data);
  } else if (src_data_type == QNN_DATATYPE_FLOAT_32) {
    fill_result = FillData<float, T>(src_tensor, dst_data);
  } else if (src_data_type == QNN_DATATYPE_FLOAT_64) {
    fill_result = FillData<double, T>(src_tensor, dst_data);
  } else {
    QNN_LOG_ERROR("Unsupported QNN type for conversion.");
    fill_result = false;
  }

  if (!fill_result) {
    return nullptr;
  }
  const auto id = tensor_wrappers_.size();
  auto tensor_name = std::to_string(id) + kQnnSuffix;
  auto& back = tensor_wrappers_.emplace_back(
      std::move(tensor_name), QNN_TENSOR_TYPE_STATIC,
      GetQnnDataType<T>(src_tensor.IsQuant()), src_tensor.GetQuantParams(),
      src_tensor.GetDims(), sizeof(T) * dst_data.size(), dst_data.data());
  return &back;
}

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TENSOR_POOL_H_
