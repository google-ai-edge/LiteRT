// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include "litert/vendors/qualcomm/core/utils/miscs.h"

#include <cassert>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>
#include <system_error>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
namespace qnn {
namespace {
static constexpr int kRequiredNumProviders{1};
}
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** provider_list, uint32_t* num_providers);

void ConvertDataFromInt16toUInt16(absl::Span<const std::int16_t> src,
                                  std::vector<std::uint16_t>& dst) {
  dst.clear();
  dst.reserve(src.size());
  for (const auto& data : src) {
    dst.emplace_back(data + kUint16ZeroPoint);
  }
}
void ConvertDataFromUInt16toInt16(absl::Span<const std::uint16_t> src,
                                  std::vector<std::int16_t>& dst) {
  dst.clear();
  dst.reserve(src.size());
  for (const auto& data : src) {
    dst.emplace_back(data - kUint16ZeroPoint);
  }
}

void ConvertDataFromInt4ToInt8(const void* src, size_t num_bytes,
                               std::vector<std::int8_t>& dst) {
  dst.clear();
  dst.reserve(num_bytes * 2);
  const std::uint8_t* byte_data = static_cast<const std::uint8_t*>(src);
  for (size_t i = 0; i < num_bytes; i++) {
    std::uint8_t byte = byte_data[i];
    std::int8_t lower = byte & 0x0F;
    std::int8_t upper = (byte >> 4) & 0x0F;
    if (lower & 0x08) lower |= 0xF0;
    if (upper & 0x08) upper |= 0xF0;
    dst.emplace_back(lower);
    dst.emplace_back(upper);
  }
}

void ConvertDataFromInt2ToInt8(const void* src, size_t num_bytes,
                               std::vector<std::int8_t>& dst) {
  dst.clear();
  dst.reserve(num_bytes * 4);
  const std::uint8_t* byte_data = static_cast<const std::uint8_t*>(src);
  for (size_t i = 0; i < num_bytes; i++) {
    std::uint8_t byte = byte_data[i];

    for (size_t j = 0; j < 4; j++) {
      // Mask: 0000 0011
      std::int8_t num = byte & 0x03;

      // Perform sign extension on all four numbers
      // The sign bit for a 2-bit number is the 2nd bit (mask 0x02)
      // The sign extension mask is 0xFC (binary 1111 1100)
      if (num & 0x02) num |= 0xFC;

      dst.emplace_back(num);

      byte >>= 2;
    }
  }
}

void ConvertDataFromInt8ToInt2(const std::vector<std::int8_t>& src,
                               std::vector<std::int8_t>& dst) {
  // The source vector size must be a multiple of 4.
  assert(src.size() % 4 == 0);

  dst.clear();
  dst.reserve(src.size() / 4);

  // Process the source vector in chunks of 4.
  for (size_t i = 0; i < src.size(); i += 4) {
    // Mask each int8_t to get its 2-bit representation, discarding sign bits.
    // Mask: 0000 0011
    std::int8_t num1 = src[i] & 0x03;
    std::int8_t num2 = src[i + 1] & 0x03;
    std::int8_t num3 = src[i + 2] & 0x03;
    std::int8_t num4 = src[i + 3] & 0x03;

    // Combine the four 2-bit numbers into a single byte.
    // num4 is placed in the most significant bits, num1 in the least.
    std::int8_t byte = num1 | (num2 << 2) | (num3 << 4) | (num4 << 6);
    dst.emplace_back(byte);
  }
}

bool CreateDirectoryRecursive(const std::filesystem::path& dir_name) {
  std::error_code err;
  err.clear();
  if (!std::filesystem::create_directories(dir_name, err)) {
    if (std::filesystem::exists(dir_name)) {
      err.clear();
      return true;
    }
    if (err) {
      QNN_LOG_ERROR("%s", err.message().c_str());
    }
    return false;
  }
  return true;
}

#if !defined(_WIN32)
void DlCloser::operator()(void* handle) const {
  if (handle) {
    dlclose(handle);
    if (const char* error = dlerror(); error) {
      QNN_LOG_ERROR("dlclose failed: %s", error);
    }
  }
}

DLHandle CreateDLHandle(const char* path) {
  void* handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    QNN_LOG_ERROR("dlopen failed: %s", dlerror());
  }
  return DLHandle(handle);
}

const QNN_INTERFACE_VER_TYPE* ResolveQnnApi(
    void* handle, Qnn_Version_t expected_qnn_version) {
  const QnnInterface_t** providers = nullptr;
  uint32_t num_providers = 0;

  if (!handle) {
    QNN_LOG_ERROR("Can not resolve QNN API from null handle");
    return nullptr;
  }

  auto get_providers = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
      dlsym(handle, "QnnInterface_getProviders"));

  if (const char* error = dlerror(); error) {
    QNN_LOG_ERROR("dlsym failed: %s", error);
    return nullptr;
  }

  if (get_providers(&providers, &num_providers) != 0 || providers == nullptr ||
      num_providers == 0) {
    QNN_LOG_ERROR("QnnInterface_GetProviders failed or returned no providers");
    return nullptr;
  }

  if (num_providers != kRequiredNumProviders) {
    QNN_LOG_ERROR("Found %u providers, expected %u", num_providers,
                  kRequiredNumProviders);
    return nullptr;
  }

  auto qnn_version = providers[0]->apiVersion;

  // Core API version check
  if (qnn_version.coreApiVersion.major != QNN_API_VERSION_MAJOR ||
      (qnn_version.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
       qnn_version.coreApiVersion.minor < QNN_API_VERSION_MINOR)) {
    QNN_LOG_ERROR(
        "Qnn core API version %u.%u.%u is not supported. Minimum required is "
        "%u.%u.%u",
        qnn_version.coreApiVersion.major, qnn_version.coreApiVersion.minor,
        qnn_version.coreApiVersion.patch, QNN_API_VERSION_MAJOR,
        QNN_API_VERSION_MINOR, QNN_API_VERSION_PATCH);
    return nullptr;
  }

  // Backend API version check
  if (qnn_version.backendApiVersion.major != expected_qnn_version.major ||
      (qnn_version.backendApiVersion.major == expected_qnn_version.major &&
       qnn_version.backendApiVersion.minor < expected_qnn_version.minor)) {
    QNN_LOG_ERROR(
        "Qnn backend API version %u.%u.%u is not supported. Minimum required "
        "is %u.%u.%u",
        qnn_version.backendApiVersion.major,
        qnn_version.backendApiVersion.minor,
        qnn_version.backendApiVersion.patch, expected_qnn_version.major,
        expected_qnn_version.minor, expected_qnn_version.patch);
    return nullptr;
  }

  if (!providers[0]) {
    QNN_LOG_ERROR("No valid interface was provided");
    return nullptr;
  }

  return &providers[0]->QNN_INTERFACE_VER_NAME;
}
#endif  // !defined(_WIN32)

std::optional<::qnn::SocInfo> FindSocModel(std::string_view soc_model_name) {
  std::optional<::qnn::SocInfo> soc_model;
  for (auto i = 0; i < ::qnn::kNumSocInfos; ++i) {
    if (soc_model_name == ::qnn::kSocInfos[i].soc_name) {
      soc_model = ::qnn::kSocInfos[i];
      break;
    }
  }
  return soc_model;
}

bool IsHtpBackend() {
  // If env variable is not set, default to HTP (return true).
  if (const char* env_p = std::getenv("GTEST_TARGET_BACKEND")) {
    // If env variable is set, it must be HTP to return true.
    return std::string(env_p) == "Htp";
  }
  return true;
}

}  // namespace qnn
