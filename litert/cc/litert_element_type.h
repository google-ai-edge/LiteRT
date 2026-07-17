// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_CC_LITERT_ELEMENT_TYPE_H_
#define ODML_LITERT_LITERT_CC_LITERT_ELEMENT_TYPE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_numerics.h"

/// @file
/// @brief Defines the C++ `ElementType` enum and related utility functions for
/// LiteRT.

namespace litert {

/// @brief The data type of tensor elements. This is the C++ equivalent of
/// `LiteRtElementType`.
enum class ElementType {
  None = kLiteRtElementTypeNone,
  Bool = kLiteRtElementTypeBool,
  Int2 = kLiteRtElementTypeInt2,
  Int4 = kLiteRtElementTypeInt4,
  Int8 = kLiteRtElementTypeInt8,
  Int16 = kLiteRtElementTypeInt16,
  Int32 = kLiteRtElementTypeInt32,
  Int64 = kLiteRtElementTypeInt64,
  UInt4 = kLiteRtElementTypeUInt4,
  UInt8 = kLiteRtElementTypeUInt8,
  UInt16 = kLiteRtElementTypeUInt16,
  UInt32 = kLiteRtElementTypeUInt32,
  UInt64 = kLiteRtElementTypeUInt64,
  Float8E4M3FN = kLiteRtElementTypeFloat8E4M3FN,
  Float8E5M2 = kLiteRtElementTypeFloat8E5M2,
  Float16 = kLiteRtElementTypeFloat16,
  BFloat16 = kLiteRtElementTypeBFloat16,
  Float32 = kLiteRtElementTypeFloat32,
  Float64 = kLiteRtElementTypeFloat64,
  Complex64 = kLiteRtElementTypeComplex64,
  Complex128 = kLiteRtElementTypeComplex128,
  TfResource = kLiteRtElementTypeTfResource,
  TfString = kLiteRtElementTypeTfString,
  TfVariant = kLiteRtElementTypeTfVariant,
};

/// @brief Gets the number of bytes of a single element of a given type.
constexpr std::optional<ByteWidth> GetByteWidth(ElementType ty) {
  if (ty == ElementType::Bool)
    return ByteWidth(1);
  else if (ty == ElementType::Int4)
    return ByteWidth(1, 2);
  else if (ty == ElementType::UInt4)
    return ByteWidth(1, 2);
  else if (ty == ElementType::Int8)
    return ByteWidth(1);
  else if (ty == ElementType::Int16)
    return ByteWidth(2);
  else if (ty == ElementType::Int32)
    return ByteWidth(4);
  else if (ty == ElementType::Int64)
    return ByteWidth(8);
  else if (ty == ElementType::UInt8)
    return ByteWidth(1);
  else if (ty == ElementType::UInt16)
    return ByteWidth(2);
  else if (ty == ElementType::UInt32)
    return ByteWidth(4);
  else if (ty == ElementType::UInt64)
    return ByteWidth(8);
  else if (ty == ElementType::Float16)
    return ByteWidth(2);
  else if (ty == ElementType::BFloat16)
    return ByteWidth(2);
  else if (ty == ElementType::Float8E4M3FN)
    return ByteWidth(1);
  else if (ty == ElementType::Float8E5M2)
    return ByteWidth(1);
  else if (ty == ElementType::Float32)
    return ByteWidth(4);
  else if (ty == ElementType::Float64)
    return ByteWidth(8);
  else if (ty == ElementType::Complex64)
    return ByteWidth(8);
  else if (ty == ElementType::Int2)
    return ByteWidth(1, 4);
  else
    return std::nullopt;
}

/// @brief Gets the number of bytes of a single element of a given type via a
/// template parameter.
template <ElementType Ty>
constexpr ByteWidth GetByteWidth() {
  constexpr auto byte_width = GetByteWidth(Ty);
  static_assert(byte_width.has_value(), "Type does not have byte width");
  return byte_width.value();
}

template <class>
constexpr bool dependent_false = false;  // workaround before CWG2518/P2593R1

/// @brief Gets the `litert::ElementType` associated with a given C++ type.
template <typename T>
constexpr ElementType GetElementType() {
  static_assert(dependent_false<T>, "Uknown C++ type");
  return ElementType::None;
}

template <>
constexpr ElementType GetElementType<bool>() {
  return ElementType::Bool;
}

template <>
constexpr ElementType GetElementType<int8_t>() {
  return ElementType::Int8;
}

template <>
constexpr ElementType GetElementType<uint8_t>() {
  return ElementType::UInt8;
}

template <>
constexpr ElementType GetElementType<int16_t>() {
  return ElementType::Int16;
}

template <>
constexpr ElementType GetElementType<uint16_t>() {
  return ElementType::UInt16;
}

template <>
constexpr ElementType GetElementType<int32_t>() {
  return ElementType::Int32;
}

template <>
constexpr ElementType GetElementType<uint32_t>() {
  return ElementType::UInt32;
}

template <>
constexpr ElementType GetElementType<int64_t>() {
  return ElementType::Int64;
}

template <>
constexpr ElementType GetElementType<uint64_t>() {
  return ElementType::UInt64;
}

template <>
constexpr ElementType GetElementType<float>() {
  return ElementType::Float32;
}

template <>
constexpr ElementType GetElementType<double>() {
  return ElementType::Float64;
}

}  // namespace litert

namespace tflite { struct half; }
namespace litert::tensor { struct int4_t; struct int2_t; struct bf16_t; }
struct TfLiteComplex64;

namespace litert {

template <>
constexpr ElementType GetElementType<litert::tensor::bf16_t>() {
  return ElementType::BFloat16;
}

template <>
constexpr ElementType GetElementType<litert::tensor::int4_t>() {
  return ElementType::Int4;
}

template <>
constexpr ElementType GetElementType<litert::tensor::int2_t>() {
  return ElementType::Int2;
}

template <>
constexpr ElementType GetElementType<TfLiteComplex64>() {
  return ElementType::Complex64;
}

// clang format off
template <ElementType Ty>
using GetCCType =
    SelectT<std::bool_constant<Ty == ElementType::Bool>, bool,
            std::bool_constant<Ty == ElementType::Int8>, int8_t,
            std::bool_constant<Ty == ElementType::Int16>, int16_t,
            std::bool_constant<Ty == ElementType::Int32>, int32_t,
            std::bool_constant<Ty == ElementType::Int64>, int64_t,
            std::bool_constant<Ty == ElementType::UInt8>, uint8_t,
            std::bool_constant<Ty == ElementType::UInt16>, uint16_t,
            std::bool_constant<Ty == ElementType::UInt32>, uint32_t,
            std::bool_constant<Ty == ElementType::UInt64>, uint64_t,
            // uint16_t is the opaque representation of tflite::half to avoid
            // leaking tflite headers into public SDK.
            std::bool_constant<Ty == ElementType::Float16>, uint16_t,
            std::bool_constant<Ty == ElementType::Float32>, float,
            std::bool_constant<Ty == ElementType::Float64>, double>;
// clang format on

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_ELEMENT_TYPE_H_
