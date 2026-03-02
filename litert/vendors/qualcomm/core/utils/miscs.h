// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_MISCS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_MISCS_H_
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
namespace qnn {
namespace miscs {

inline bool IsStrEq(const char* input, const char* golden) {
  if (input == nullptr && golden == nullptr) return true;
  if (input == nullptr || golden == nullptr) return false;

  return std::strcmp(input, golden) == 0;
}

}  // namespace miscs

constexpr uint32_t kUint16ZeroPoint = -std::numeric_limits<std::int16_t>::min();
constexpr uint32_t kQuantBitWidth4 = 4;
constexpr uint32_t kQuantBitWidth2 = 2;

constexpr std::array<int8_t, 256 * 4> MakeInt2LUT()
{
   std::array<int8_t, 256 * 4> lut{};
   for (int b = 0; b < 256; b++) {
       for (int i = 0; i < 4; i++) {
           int v = (b >> (i * 2)) & 3;
           if (v & 2) v |= ~3;
           lut[b * 4 + i] = static_cast<int8_t>(v);
       }
   }
   return lut;
}
alignas(64) inline constexpr auto kInt2LUT = MakeInt2LUT();

constexpr std::array<int8_t, 256 * 2> MakeInt4LUT() {
  std::array<int8_t, 256 * 2> lut{};
  for (int b = 0; b < 256; b++) {
    for (int i = 0; i < 2; i++) {
      int v = (b >> (i * 4)) & 0xF;
      if (v & 0x8) v |= ~0xF;
      lut[b * 2 + i] = static_cast<int8_t>(v);
    }
  }
  return lut;
}
alignas(64) inline constexpr auto kInt4LUT = MakeInt4LUT();

std::vector<std::int8_t> UnpackIntData(const void* src, size_t src_bytes,
                                       uint32_t bit_width);

template <typename...>
inline constexpr bool always_false = false;
template <typename T>
T Quantize(const float val, const float scale, const int32_t zero_point) {
  static_assert(std::is_integral<T>::value,
                "Integral required in Quantize function.");
  return std::round(val / scale) + zero_point;
}

template <typename T>
float Dequantize(const T val, const float scale, const int32_t zero_point) {
  static_assert(std::is_integral<T>::value,
                "Integral required in Dequantize function.");
  return scale * (val - zero_point);
}

template <typename T>
void DequantizeInto(const absl::Span<const T>& in, const float scale,
                    const int32_t zero_point, std::vector<float>& out) {
  for (size_t i = 0; i < in.size(); ++i) {
    out.emplace_back(Dequantize(in[i], scale, zero_point));
  }
}

void ConvertDataFromInt16toUInt16(absl::Span<const std::int16_t> src,
                                  std::vector<std::uint16_t>& dst);

void ConvertDataFromUInt16toInt16(absl::Span<const std::uint16_t> src,
                                  std::vector<std::int16_t>& dst);

void ConvertDataFromInt8ToInt2(const std::vector<std::int8_t>& src,
                               std::vector<std::int8_t>& dst);

bool CreateDirectoryRecursive(const std::filesystem::path& dir_name);

#if !defined(_WIN32)
struct DlCloser {
  void operator()(void* handle) const;
};

using DLHandle = std::unique_ptr<void, DlCloser>;

DLHandle CreateDLHandle(const char* path);

const QNN_INTERFACE_VER_TYPE* ResolveQnnApi(void* handle,
                                            Qnn_Version_t expected_qnn_version);
#else   // _WIN32
struct DlCloser {
  void operator()(void* handle) const {}
};
using DLHandle = std::unique_ptr<void, DlCloser>;
inline DLHandle CreateDLHandle(const char* path) { return DLHandle(nullptr); }
inline const QNN_INTERFACE_VER_TYPE* ResolveQnnApi(
    void* handle, Qnn_Version_t expected_qnn_version) {
  return nullptr;
}
#endif  // !defined(_WIN32)

std::optional<::qnn::SocInfo> FindSocModel(std::string_view soc_model_name);
}  // namespace qnn
#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_MISCS_H_
