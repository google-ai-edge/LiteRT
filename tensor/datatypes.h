/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_DATATYPES_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_DATATYPES_H_

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <ratio>  // NOLINT: used as a tag.
#include <type_traits>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/internal/fp16.h"

namespace litert::tensor {

enum class Type {
  kUnknown = 0,
  kBOOL,
  kI2,
  kI4,
  kI8,
  kI16,
  kI32,
  kI64,
  kU4,
  kU8,
  kU16,
  kU32,
  kU64,
  kFP16,
  kFP32,
  kFP64,
  kBF16,
};

struct int2_t {
  int8_t a : 2;
  int8_t b : 2;
  int8_t c : 2;
  int8_t d : 2;

  friend bool operator==(const int2_t lhs, const int2_t rhs) {
    return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c && lhs.d == rhs.d;
  }

  friend bool operator!=(const int2_t lhs, const int2_t rhs) {
    return !(lhs == rhs);
  }
};

static_assert(sizeof(int2_t) == sizeof(int8_t));
static_assert(alignof(int2_t) == alignof(int8_t));

struct int4_t {
  int8_t a : 4;
  int8_t b : 4;

  friend bool operator==(const int4_t lhs, const int4_t rhs) {
    return lhs.a == rhs.a && lhs.b == rhs.b;
  }

  friend bool operator!=(const int4_t lhs, const int4_t rhs) {
    return !(lhs == rhs);
  }
};

static_assert(sizeof(int4_t) == sizeof(int8_t));
static_assert(alignof(int4_t) == alignof(int8_t));

struct uint4_t {
  uint8_t a : 4;
  uint8_t b : 4;

  friend bool operator==(const uint4_t lhs, const uint4_t rhs) {
    return lhs.a == rhs.a && lhs.b == rhs.b;
  }

  friend bool operator!=(const uint4_t lhs, const uint4_t rhs) {
    return !(lhs == rhs);
  }
};

static_assert(sizeof(uint4_t) == sizeof(uint8_t));
static_assert(alignof(uint4_t) == alignof(uint8_t));

struct bf16_t {
  constexpr bf16_t() = default;
  constexpr bf16_t(const bf16_t&) = default;
  constexpr bf16_t& operator=(const bf16_t&) = default;

  // NOLINTNEXTLINE(*-explicit-constructor): bf16_t can be built from a float.
  bf16_t(float v) : val(fp32_to_bf16(v)) {}
  // NOLINTNEXTLINE(*-explicit-constructor): bf16_t can be converted to a float.
  operator float() const {
    return bit_cast<float>(static_cast<uint32_t>(val) << 16);
  }

  friend bool operator==(bf16_t a, bf16_t b) { return a.val == b.val; }
  friend bool operator==(bf16_t a, float b) {
    return static_cast<float>(a) == b;
  }
  friend bool operator==(float a, bf16_t b) {
    return a == static_cast<float>(b);
  }

  uint16_t val = 0;

  // Converts an fp32 value to bf16 and rounds to nearest even.
  static uint16_t fp32_to_bf16(float value) {
    const uint32_t float_bits = bit_cast<uint32_t>(value);
    if ((float_bits & 0x7f800000) == 0x7f800000 && (float_bits & 0x007fffff)) {
      return 0x7fc0;
    }
    const uint32_t rounding_bias = 0x7fff + ((float_bits >> 16) & 1);
    return static_cast<uint16_t>((float_bits + rounding_bias) >> 16);
  }
};

static_assert(sizeof(bf16_t) == sizeof(int16_t));
static_assert(alignof(bf16_t) == alignof(int16_t));

struct fp16_t {
  constexpr fp16_t() = default;
  constexpr fp16_t(const fp16_t&) = default;
  constexpr fp16_t& operator=(const fp16_t&) = default;

  // NOLINTNEXTLINE(*-explicit-constructor): fp16_t can be built from a float.
  fp16_t(float f) : val(fp16_ieee_from_fp32_value(f)) {}
  // NOLINTNEXTLINE(*-explicit-constructor): fp16_t can be converted to a float.
  operator float() const { return fp16_ieee_to_fp32_value(val); }

  friend bool operator==(fp16_t a, fp16_t b) { return a.val == b.val; }
  friend bool operator==(fp16_t a, float b) {
    return static_cast<float>(a) == b;
  }
  friend bool operator==(float a, fp16_t b) {
    return a == static_cast<float>(b);
  }

  uint16_t val = 0;
};

static_assert(sizeof(fp16_t) == sizeof(int16_t));
static_assert(alignof(fp16_t) == alignof(int16_t));

template <Type>
struct NativeStorage;

namespace internal {

// Implements the NativeStorage interface.
//
// The NativeStorage specializations inherit from this template to avoid code
// duplication.
//
// - T: the native type used to store data.
// - Bits: a power of 2, for data that is smaller than 1 byte.
template <Type t, class T, uint64_t Bits = 8 * sizeof(T)>
struct StorageImpl {
  static_assert((Bits & (Bits - 1)) == 0, "Bits must be a power of 2.");
  static_assert(Bits != 0, "Bits must be non-zero.");
  static_assert(!std::is_same_v<T, void>, "Storage type cannot be void.");
  using type = T;
  static constexpr Type value = t;
  static constexpr uint64_t kNumElements = 8 * sizeof(T) / Bits;
  static constexpr uint64_t BufferSize(size_t count) {
    return (Bits * count + 7) / 8;
  }
};

}  // namespace internal

// We're defining this for consistency. Actually using it for anything else is
// an error.
template <>
struct NativeStorage<Type::kUnknown> {
  using type = void;
  static constexpr Type value = Type::kUnknown;
  static constexpr uint64_t kNumElements = 0;
  static constexpr uint64_t BufferSize(size_t count) { return 0; }
};

template <>
struct NativeStorage<Type::kI2>
    : internal::StorageImpl<Type::kI2, int2_t, /*Bits=*/2> {};

template <>
struct NativeStorage<Type::kI4>
    : internal::StorageImpl<Type::kI4, int4_t, /*Bits=*/4> {};

template <>
struct NativeStorage<Type::kI8> : internal::StorageImpl<Type::kI8, int8_t> {};

template <>
struct NativeStorage<Type::kI16> : internal::StorageImpl<Type::kI16, int16_t> {
};

template <>
struct NativeStorage<Type::kI32> : internal::StorageImpl<Type::kI32, int32_t> {
};

template <>
struct NativeStorage<Type::kI64> : internal::StorageImpl<Type::kI64, int64_t> {
};

template <>
struct NativeStorage<Type::kU4>
    : internal::StorageImpl<Type::kU4, uint4_t, /*Bits=*/4> {};

template <>
struct NativeStorage<Type::kU8> : internal::StorageImpl<Type::kU8, uint8_t> {};

template <>
struct NativeStorage<Type::kU16> : internal::StorageImpl<Type::kU16, uint16_t> {
};

template <>
struct NativeStorage<Type::kU32> : internal::StorageImpl<Type::kU32, uint32_t> {
};

template <>
struct NativeStorage<Type::kU64> : internal::StorageImpl<Type::kU64, uint64_t> {
};

template <>
struct NativeStorage<Type::kBF16> : internal::StorageImpl<Type::kBF16, bf16_t> {
};

template <>
struct NativeStorage<Type::kFP16> : internal::StorageImpl<Type::kFP16, fp16_t> {
};

template <>
struct NativeStorage<Type::kFP32> : internal::StorageImpl<Type::kFP32, float> {
};

template <>
struct NativeStorage<Type::kFP64> : internal::StorageImpl<Type::kFP64, double> {
};

template <>
struct NativeStorage<Type::kBOOL> : internal::StorageImpl<Type::kBOOL, bool> {};

template <class T>
struct ApiType;

template <>
struct ApiType<int2_t> : internal::StorageImpl<Type::kI2, int2_t, /*Bits=*/2> {
};

template <>
struct ApiType<int4_t> : internal::StorageImpl<Type::kI4, int4_t, /*Bits=*/4> {
};

template <>
struct ApiType<int8_t> : internal::StorageImpl<Type::kI8, int8_t> {};

template <>
struct ApiType<int16_t> : internal::StorageImpl<Type::kI16, int16_t> {};

template <>
struct ApiType<int32_t> : internal::StorageImpl<Type::kI32, int32_t> {};

template <>
struct ApiType<int64_t> : internal::StorageImpl<Type::kI64, int64_t> {};

template <>
struct ApiType<uint4_t>
    : internal::StorageImpl<Type::kU4, uint4_t, /*Bits=*/4> {};

template <>
struct ApiType<uint8_t> : internal::StorageImpl<Type::kU8, uint8_t> {};

template <>
struct ApiType<uint16_t> : internal::StorageImpl<Type::kU16, uint16_t> {};

template <>
struct ApiType<uint32_t> : internal::StorageImpl<Type::kU32, uint32_t> {};

template <>
struct ApiType<uint64_t> : internal::StorageImpl<Type::kU64, uint64_t> {};

template <>
struct ApiType<bf16_t> : internal::StorageImpl<Type::kBF16, bf16_t> {};

template <>
struct ApiType<fp16_t> : internal::StorageImpl<Type::kFP16, fp16_t> {};

template <>
struct ApiType<float> : internal::StorageImpl<Type::kFP32, float> {};

template <>
struct ApiType<double> : internal::StorageImpl<Type::kFP64, double> {};

template <>
struct ApiType<bool> : internal::StorageImpl<Type::kBOOL, bool> {};

namespace internal {

template <class T>
constexpr auto GetPackedValue(std::integral_constant<int, 1>, const T& val,
                              size_t /*index*/) {
  return val;
}

template <class T>
inline auto GetPackedValue(std::integral_constant<int, 2>, const T& val,
                           size_t index) {
  return index == 0 ? val.a : val.b;
}

template <class T>
inline auto GetPackedValue(std::integral_constant<int, 4>, const T& val,
                           size_t index) {
  switch (index) {
    case 0:
      return val.a;
    case 1:
      return val.b;
    case 2:
      return val.c;
    default:
      return val.d;
  }
}

template <class T>
constexpr auto GetPackedValue(const T& val, size_t index) {
  return GetPackedValue(std::integral_constant<int, ApiType<T>::kNumElements>{},
                        val, index);
}

template <class T, class U>
constexpr void SetPackedValue(std::integral_constant<int, 1>, T& dest,
                              size_t index, U&& val) {
  dest = std::forward<U>(val);
}

template <class T, class U>
constexpr void SetPackedValue(std::integral_constant<int, 2>, T& dest,
                              size_t index, U&& val) {
  if (index == 0) {
    dest.a = static_cast<int8_t>(std::forward<U>(val));
  } else {
    dest.b = static_cast<int8_t>(std::forward<U>(val));
  }
}

template <class T, class U>
constexpr void SetPackedValue(std::integral_constant<int, 4>, T& dest,
                              size_t index, U&& val) {
  switch (index) {
    case 0:
      dest.a = static_cast<int8_t>(std::forward<U>(val));
      break;
    case 1:
      dest.b = static_cast<int8_t>(std::forward<U>(val));
      break;
    case 2:
      dest.c = static_cast<int8_t>(std::forward<U>(val));
      break;
    default:
      dest.d = static_cast<int8_t>(std::forward<U>(val));
      break;
  }
}

template <class T, class U>
constexpr void SetPackedValue(T& dest, size_t index, U&& val) {
  SetPackedValue(std::integral_constant<int, ApiType<T>::kNumElements>{}, dest,
                 index, std::forward<U>(val));
}

template <class T>
using ElementOf = std::decay_t<decltype(GetPackedValue(std::declval<T>(), 0))>;

template <Type t>
using ElementType = ElementOf<typename NativeStorage<t>::type>;

template <class Cvt, class InIt, class OutIt>
void UnrolledConvert(std::ratio<1, 4>, InIt& in, OutIt& out) {
  *out = Cvt::Call(in->a);
  *++out = Cvt::Call(in->b);
  *++out = Cvt::Call(in->c);
  *++out = Cvt::Call(in->d);
}

template <class Cvt, class InIt, class OutIt>
void UnrolledConvert(std::ratio<2, 4>, InIt& in, OutIt& out) {
  *out = {Cvt::Call(in->a), Cvt::Call(in->b)};
  *++out = {Cvt::Call(in->c), Cvt::Call(in->d)};
}

template <class Cvt, class InIt, class OutIt>
void UnrolledConvert(std::ratio<4, 4>, InIt& in, OutIt& out) {
  out = {Cvt::Call(in->a), Cvt::Call(in->b), Cvt::Call(in->c),
         Cvt::Call(in->d)};
}

template <class Cvt, class InIt, class OutIt>
void UnrolledConvert(std::ratio<1, 2>, InIt& in, OutIt& out) {
  *out = Cvt::Call(in->a);
  *++out = Cvt::Call(in->b);
}

template <class Cvt, class InIt, class OutIt>
void UnrolledConvert(std::ratio<2, 2>, InIt& in, OutIt& out) {
  *out = {Cvt::Call(in->a), Cvt::Call(in->b)};
}

template <class Cvt, class InIt, class OutIt>
void UnrolledConvert(std::ratio<4, 2>, InIt& in, OutIt& out) {
  *out = {Cvt::Call(in->a), Cvt::Call(in->b), Cvt::Call((++in)->a),
          Cvt::Call(in->b)};
}

template <class Cvt, class InIt, class OutIt>
void UnrolledConvert(std::ratio<1, 1>, InIt& in, OutIt& out) {
  *out = Cvt::Call(*in);
}

template <class Cvt, class InIt, class OutIt>
void UnrolledConvert(std::ratio<2, 1>, InIt& in, OutIt& out) {
  *out = {Cvt::Call(*in), Cvt::Call(*++in)};
}

template <class Cvt, class InIt, class OutIt>
void UnrolledConvert(std::ratio<4, 1>, InIt& in, OutIt& out) {
  *out = {Cvt::Call(*in), Cvt::Call(*++in), Cvt::Call(*++in), Cvt::Call(*++in)};
}

}  // namespace internal

template <Type out, Type in, class = void>
struct Conversion {
  using OutElement = internal::ElementType<out>;
  using InElement = internal::ElementType<in>;

  static constexpr OutElement Call(InElement val) {
    return static_cast<OutElement>(val);
  }
};

template <Type inout>
struct Conversion<inout, inout, void> {
  using Element = internal::ElementType<inout>;
  static constexpr Element Call(Element val) { return val; }
};

template <Type type>
struct Conversion<
    Type::kBF16, type,
    std::enable_if_t<type != Type::kFP32 && type != Type::kBF16>> {
  using InElement = internal::ElementType<type>;
  static constexpr typename NativeStorage<Type::kBF16>::type Call(
      InElement val) {
    return Conversion<Type::kBF16, Type::kFP32>::Call(static_cast<float>(val));
  }
};

template <Type type>
struct Conversion<
    type, Type::kBF16,
    std::enable_if_t<type != Type::kFP32 && type != Type::kBF16>> {
  using OutElement = internal::ElementType<type>;
  static constexpr OutElement Call(
      typename NativeStorage<Type::kBF16>::type val) {
    return Conversion<type, Type::kFP32>::Call(static_cast<float>(val));
  }
};

template <Type type>
struct Conversion<Type::kFP16, type,
                  std::enable_if_t<type != Type::kFP32 && type != Type::kFP16 &&
                                   type != Type::kBF16>> {
  using InElement = internal::ElementType<type>;
  static constexpr typename NativeStorage<Type::kFP16>::type Call(
      InElement val) {
    return Conversion<Type::kFP16, Type::kFP32>::Call(static_cast<float>(val));
  }
};

template <Type type>
struct Conversion<type, Type::kFP16,
                  std::enable_if_t<type != Type::kFP32 && type != Type::kFP16 &&
                                   type != Type::kBF16>> {
  using OutElement = internal::ElementType<type>;
  static constexpr OutElement Call(
      typename NativeStorage<Type::kFP16>::type val) {
    return Conversion<type, Type::kFP32>::Call(static_cast<float>(val));
  }
};

// Converts a sequence of elements from one type to another, supporting packed
// and unpacked types.
template <class InSequence, class OutSequence,
          Type from = ApiType<std::decay_t<
              decltype(*std::begin(std::declval<InSequence&>()))>>::value,
          Type to = ApiType<std::decay_t<
              decltype(*std::begin(std::declval<OutSequence&>()))>>::value>
absl::Status Convert(InSequence&& in, OutSequence&& out) {
  constexpr uint64_t in_knum = NativeStorage<from>::kNumElements;
  constexpr uint64_t out_knum = NativeStorage<to>::kNumElements;
  using std::begin, std::end, std::size;
  auto src_it = begin(in);
  auto src_end = end(in);
  auto dest_it = begin(out);
  auto dest_end = end(out);

  const size_t unroll_end =
      std::min<size_t>(size(in) / in_knum, size(out) / out_knum);
  for (size_t i = 0; i < unroll_end; ++i, ++src_it, ++dest_it) {
    internal::UnrolledConvert<Conversion<to, from>>(
        std::ratio<out_knum, in_knum>(), src_it, dest_it);
  }

  // Handle the tail.
  size_t src_index = 0;
  size_t dest_index = 0;
  while (src_it != src_end && dest_it != dest_end) {
    auto val = internal::GetPackedValue(*src_it, src_index);
    internal::SetPackedValue(*dest_it, dest_index,
                             Conversion<to, from>::Call(val));
    if (++src_index >= in_knum) {
      ++src_it;
      src_index = 0;
    }
    if (++dest_index >= out_knum) {
      ++dest_it;
      dest_index = 0;
    }
  }

  return absl::OkStatus();
}

template <Type to, class T>
auto ConvertTo(T value) {
  constexpr Type from = ApiType<T>::value;
  return Conversion<to, from>::Call(value);
}

inline const char* ToString(Type t) {
#define LITERT_TENSOR_TYPE_TO_STRING_CASE(name) \
  case Type::k##name:                           \
    return #name
  switch (t) {
    LITERT_TENSOR_TYPE_TO_STRING_CASE(Unknown);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(BOOL);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I2);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I4);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I8);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I16);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I32);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I64);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U4);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U8);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U16);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U32);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U64);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(FP16);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(FP32);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(FP64);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(BF16);
  }
#undef LITERT_TENSOR_TYPE_TO_STRING_CASE
  // This return should never be reached.
  return "ERROR: litert::tensor::ToString(Type) failed.";
}

template <class Sink>
void AbslStringify(Sink& sink, Type t) {
  sink.Append(ToString(t));
}

inline std::ostream& operator<<(std::ostream& os, const Type t) {
  return os << ToString(t);
}

inline constexpr size_t BufferSize(Type t, size_t count) {
#define LITERT_TENSOR_TYPE_BUFFER_SIZE(name) \
  case Type::k##name:                        \
    return NativeStorage<Type::k##name>::BufferSize(count);
  switch (t) {
    LITERT_TENSOR_TYPE_BUFFER_SIZE(Unknown);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(BOOL);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I2);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I4);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I8);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I16);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I32);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I64);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(U4);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(U8);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(U16);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(U32);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(U64);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(FP16);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(FP32);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(FP64);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(BF16);
  }
#undef LITERT_TENSOR_TYPE_BUFFER_SIZE
  return 0;
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_DATATYPES_H_
