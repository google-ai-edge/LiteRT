// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_PARAM_WRAPPER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_PARAM_WRAPPER_H_

#include <cstdint>
#include <string_view>
#include <type_traits>

#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

class ScalarParamWrapper {
 public:
  template <typename T>
  explicit ScalarParamWrapper(const char* name, const T data,
                              const bool is_quant)
      : name_{name} {
    if constexpr (std::is_same_v<T, bool>) {
      qnn_scalar_.dataType = QNN_DATATYPE_BOOL_8;
      qnn_scalar_.bool8Value = data;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
      qnn_scalar_.dataType =
          is_quant ? QNN_DATATYPE_UFIXED_POINT_8 : QNN_DATATYPE_UINT_8;
      qnn_scalar_.uint8Value = data;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
      qnn_scalar_.dataType =
          is_quant ? QNN_DATATYPE_SFIXED_POINT_8 : QNN_DATATYPE_INT_8;
      qnn_scalar_.int8Value = data;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
      qnn_scalar_.dataType =
          is_quant ? QNN_DATATYPE_UFIXED_POINT_16 : QNN_DATATYPE_UINT_16;
      qnn_scalar_.uint16Value = data;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
      qnn_scalar_.dataType =
          is_quant ? QNN_DATATYPE_SFIXED_POINT_16 : QNN_DATATYPE_INT_16;
      qnn_scalar_.int16Value = data;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
      qnn_scalar_.dataType =
          is_quant ? QNN_DATATYPE_UFIXED_POINT_32 : QNN_DATATYPE_UINT_32;
      qnn_scalar_.uint32Value = data;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      qnn_scalar_.dataType =
          is_quant ? QNN_DATATYPE_SFIXED_POINT_32 : QNN_DATATYPE_INT_32;
      qnn_scalar_.int32Value = data;
    } else if constexpr (std::is_same_v<T, float>) {
      qnn_scalar_.dataType = QNN_DATATYPE_FLOAT_32;
      qnn_scalar_.floatValue = data;
    } else {
      static_assert(::qnn::always_false<T>,
                    "Unsupported data type for scalar param.");
    }
  }

  bool operator==(const ScalarParamWrapper& other) const;

  void CloneTo(Qnn_Param_t& dst) const;

  std::string_view GetName() const {
    if (!name_) {
      return std::string_view{};
    }
    return name_;
  }

  template <typename T>
  T GetValue() const {
    if constexpr (std::is_same_v<T, bool>) {
      return qnn_scalar_.bool8Value;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
      return qnn_scalar_.uint8Value;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
      return qnn_scalar_.int8Value;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
      return qnn_scalar_.uint16Value;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
      return qnn_scalar_.int16Value;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
      return qnn_scalar_.uint32Value;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      return qnn_scalar_.int32Value;
    } else if constexpr (std::is_same_v<T, float>) {
      return qnn_scalar_.floatValue;
    } else {
      static_assert(::qnn::always_false<T>,
                    "Unsupported data type for scalar param.");
    }
  }

 private:
  const char* name_ = nullptr;
  Qnn_Scalar_t qnn_scalar_ = QNN_SCALAR_INIT;
};

class TensorParamWrapper {
 public:
  explicit TensorParamWrapper(const char* name, const TensorWrapper& tensor);

  bool operator==(const TensorParamWrapper& other) const;

  void CloneTo(Qnn_Param_t& dst) const;

  const TensorWrapper& GetTensor() const;

 private:
  const char* name_ = nullptr;
  const TensorWrapper& tensor_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_PARAM_WRAPPER_H_
