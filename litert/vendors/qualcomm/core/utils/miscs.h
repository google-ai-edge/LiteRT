// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_MISCS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_MISCS_H_
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
namespace qnn {
namespace {

#define CHECK_STR_EQ(INPUT, GOLDEN, MSG)                                       \
  do {                                                                         \
    const char* _input = (INPUT);                                              \
    const char* _golden = (GOLDEN);                                            \
    std::string_view _msg(MSG);                                                \
    if ((_input == nullptr && _golden != nullptr) ||                           \
        (_input != nullptr && _golden == nullptr) ||                           \
        (_input && _golden && std::strcmp(_input, _golden) != 0)) {            \
      QNN_LOG_ERROR("%.*s mismatch. Input: %s, Golden: %s",                    \
                    static_cast<int>(_msg.size()), _msg.data(),                \
                    _input ? _input : "(null)", _golden ? _golden : "(null)"); \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define CHECK_VALUE_EQ(INPUT, GOLDEN, MSG)                      \
  do {                                                          \
    std::string_view _msg(MSG);                                 \
    if ((INPUT) != (GOLDEN)) {                                  \
      QNN_LOG_ERROR("%.*s mismatch. Input: %lld, Golden: %lld", \
                    static_cast<int>(_msg.size()), _msg.data(), \
                    static_cast<long long>(INPUT),              \
                    static_cast<long long>(GOLDEN));            \
      return false;                                             \
    }                                                           \
  } while (0)

#define CHECK_TYPE_EQ(INPUT, GOLDEN, MSG)                       \
  do {                                                          \
    std::string_view _msg(MSG);                                 \
    if ((INPUT) != (GOLDEN)) {                                  \
      QNN_LOG_ERROR("%.*s mismatch. Input: %#x, Golden: %#x",   \
                    static_cast<int>(_msg.size()), _msg.data(), \
                    static_cast<unsigned>(INPUT),               \
                    static_cast<unsigned>(GOLDEN));             \
      return false;                                             \
    }                                                           \
  } while (0)

}  // namespace

constexpr uint32_t kUint16ZeroPoint = -std::numeric_limits<std::int16_t>::min();
constexpr uint32_t kQuantBitWidth4 = 4;

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

void ConvertDataFromInt16toUInt16(absl::Span<const std::int16_t> src,
                                  std::vector<std::uint16_t>& dst);

void ConvertDataFromUInt16toInt16(absl::Span<const std::uint16_t> src,
                                  std::vector<std::int16_t>& dst);

void ConvertDataFromInt4ToInt8(const void* src, std::vector<std::int8_t>& dst,
                               size_t num_bytes);

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

inline bool CompareConfig(const Qnn_OpConfigV1_t& input,
                          const Qnn_OpConfigV1_t& golden) {
  CHECK_STR_EQ(input.packageName, golden.packageName, "Op package name");
  CHECK_STR_EQ(input.typeName, golden.typeName, "Op type name");
  CHECK_VALUE_EQ(input.numOfParams, golden.numOfParams, "Number of op params");
  CHECK_VALUE_EQ(input.numOfInputs, golden.numOfInputs, "Number of op inputs");
  CHECK_VALUE_EQ(input.numOfOutputs, golden.numOfOutputs,
                 "Number of op outputs");
  return true;
}
}  // namespace qnn
#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_MISCS_H_
