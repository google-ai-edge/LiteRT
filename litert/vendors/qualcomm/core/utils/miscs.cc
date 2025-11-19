// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include "litert/vendors/qualcomm/core/utils/miscs.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <cstddef>
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

void ConvertDataFromInt4ToInt8(const void* src, std::vector<std::int8_t>& dst,
                               size_t num_bytes) {
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
}  // namespace qnn
