// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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

#ifndef LITERT_VENDORS_SAMSUNG_ENN_MANAGER_H_
#define LITERT_VENDORS_SAMSUNG_ENN_MANAGER_H_

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/samsung/dispatch/enn_type.h"

namespace litert::samsung {

class EnnManager {
 public:
  using UniquePtr = std::unique_ptr<EnnManager>;
  using Ptr = EnnManager*;
  struct PublicApi;

  EnnManager(EnnManager&) = delete;
  EnnManager(EnnManager&&) = delete;
  EnnManager& operator=(const EnnManager&) = delete;
  EnnManager& operator=(EnnManager&&) = delete;

  static Expected<EnnManager::UniquePtr> Create();

  const PublicApi& Api() const;
  ~EnnManager();

  // Guard flag for cross-destructor safety.
  // WeightBinaryManager sets this in its destructor to call EnnDeinitialize()
  // before EnnManager is destroyed (static destruction order is unspecified).
  bool _enn_deinitialized_ = false;

 private:
  EnnManager();
  // Loads and resolve compiler related api
  LiteRtStatus LoadEnnRuntimeLibrary(absl::string_view path);

  SharedLibrary enn_runtime_lib_;
  std::unique_ptr<PublicApi> api_;
};

struct EnnManager::PublicApi {
  EnnReturn (*EnnInitialize)(void);
  EnnReturn (*EnnOpenModelFromMemory)(const char* va, const uint32_t size,
                                      EnnModelId* model_id);
  EnnReturn (*EnnOpenModelFromMemoryWithWeight)(const char* va,
                                                const uint32_t size,
                                                EnnBufferPtr* weights,
                                                uint32_t n_weights,
                                                EnnModelId* model_id);
  EnnReturn (*EnnOpenModelFromFdWithWeight)(
      const uint32_t fd, const uint32_t size, const uint32_t offset,
      EnnBufferPtr* weights, uint32_t n_weights, EnnModelId* model_id);
  EnnReturn (*EnnOpenModelWithFileOpenFdWeight)(
      const int fd, const uint32_t size, const uint32_t offset,
      EnnBufferPtr* weights, uint32_t n_weights, EnnModelId* model_id);
  EnnReturn (*EnnOpenModelWithFileOpenFd)(const int fd, const uint32_t size,
                                          const uint32_t offset,
                                          EnnModelId* model_id);
  EnnReturn (*EnnCreateBufferFromFdWithOffset)(const uint32_t fd,
                                               const uint32_t size,
                                               const uint32_t offset,
                                               EnnBufferPtr* out);
  EnnReturn (*EnnSetPreferencePerfMode)(const uint32_t val);
  EnnReturn (*EnnSetPreferencePerfConfigId)(uint32_t val);
  EnnReturn (*EnnCreateBufferCache)(const uint32_t req_size, EnnBufferPtr* out);
  EnnReturn (*EnnAllocateAllBuffers)(const EnnModelId model_id,
                                     EnnBufferPtr** out_buffers,
                                     NumberOfBuffersInfo* out_buffers_info);
  EnnReturn (*EnnBufferCommit)(const EnnModelId model_id);
  EnnReturn (*EnnGetBuffersInfo)(const EnnModelId model_id,
                                 NumberOfBuffersInfo* buffers_info);
  EnnReturn (*EnnSetBufferByIndex)(const EnnModelId model_id,
                                   const enn_buf_dir_e direction,
                                   const uint32_t index, EnnBufferPtr buf);
  EnnReturn (*EnnReleaseBuffer)(EnnBufferPtr buffer);
  EnnReturn (*EnnExecuteModel)(const EnnModelId model_id);
  EnnReturn (*EnnBufferUncommit)(const EnnModelId model_id);
  EnnReturn (*EnnUnsetBuffers)(const EnnModelId model_id);
  EnnReturn (*EnnCloseModel)(const EnnModelId model_id);
  EnnReturn (*EnnDeinitialize)(void);
};

static constexpr uint32_t PERF_CONFIG_ID_DEFAULT = 0;

struct PerfConfig {
  PerfModePreference mode;
  uint32_t configId;
};

// Get hardware name from system property (primary)
inline std::string GetHwName() {
#if defined(__ANDROID__)
  char prop_value[256] = {};

  if (__system_property_get("ro.hardware", prop_value) > 0) {
    return std::string(prop_value);
  }
#endif
  return "";
}

// Get SOC name from system property (fallback)
inline std::string GetSocName() {
#if defined(__ANDROID__)
  char prop_value[256] = {};

  if (__system_property_get("ro.soc.model", prop_value) > 0) {
    return std::string(prop_value);
  }
#endif
  return "";
}

// Extract trailing digits from string (e.g., "exynos9955" -> "9955", "s5e9965"
// -> "9965")
inline bool GetTrimNumber(std::string& value) {
  const char* digits = "0123456789";
  auto lp = value.find_last_of(digits);
  if (lp == std::string::npos) return false;

  // Find the start of the trailing digit block
  auto fp = lp;
  while (fp > 0 && std::strchr(digits, value[fp - 1]) != nullptr) {
    fp--;
  }

  value = value.substr(fp, lp - fp + 1);
  return true;
}

// SOC-to-performance-config map
inline const std::unordered_map<std::string, PerfConfig>&
GetSocPerfConfigMap() {
  static const std::unordered_map<std::string, PerfConfig> kSocPerfConfig = {
      {"9955", {ENN_PREF_MODE_PERFORMANCE, PERF_CONFIG_ID_DEFAULT}},
      {"9965", {ENN_PREF_MODE_PERFORMANCE, PERF_CONFIG_ID_DEFAULT}},
  };
  return kSocPerfConfig;
}

inline PerfConfig GetPerfConfigFromSoc(absl::string_view soc_name) {
  std::string key = "";

  // Try ro.hardware first (primary)
  key = GetHwName();
  if (!key.empty()) {
    GetTrimNumber(key);
  }

  // Fallback to ro.soc.model
  if (key.empty()) {
    key = GetSocName();
    if (!key.empty()) {
      GetTrimNumber(key);
    }
  }

  // Lookup in map
  const auto& map = GetSocPerfConfigMap();
  auto it = map.find(key);
  if (it != map.end()) {
    return it->second;
  }

  return {ENN_PREF_MODE_PERFORMANCE, PERF_CONFIG_ID_DEFAULT};
}

inline LiteRtStatus SetGenAiPerfConfig(
    const EnnManager::PublicApi& api, PerfModePreference mode,
    uint32_t configId = PERF_CONFIG_ID_DEFAULT) {
  EnnReturn ret_mode =
      api.EnnSetPreferencePerfMode(static_cast<uint32_t>(mode));
  if (ret_mode != ENN_RET_SUCCESS) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  EnnReturn ret_config = api.EnnSetPreferencePerfConfigId(configId);
  if (ret_config != ENN_RET_SUCCESS) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus SetGenAiPerfConfigFromSoc(const EnnManager::PublicApi& api);

}  // namespace litert::samsung

#endif  // LITERT_VENDORS_SAMSUNG_ENN_MANAGER_H_
