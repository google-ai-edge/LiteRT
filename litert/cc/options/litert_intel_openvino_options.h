// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_

#include <memory>
#include <string>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::intel_openvino {

/// @brief Defines the C++ wrapper for Intel OpenVINO-specific LiteRT options.
class IntelOpenVinoOptions {
public:
  IntelOpenVinoOptions() = delete;

  // Non-copyable, only movable
  IntelOpenVinoOptions(const IntelOpenVinoOptions &) = delete;
  IntelOpenVinoOptions &operator=(const IntelOpenVinoOptions &) = delete;
  IntelOpenVinoOptions(IntelOpenVinoOptions &&) = default;
  IntelOpenVinoOptions &operator=(IntelOpenVinoOptions &&) = default;

  static Expected<IntelOpenVinoOptions> Create() {
    LrtIntelOpenVinoOptions options;
    LITERT_RETURN_IF_ERROR(LrtIntelOpenVinoOptionsCreate(&options));
    return IntelOpenVinoOptions(options);
  }

  static IntelOpenVinoOptions
  CreateFromOwnedHandle(LrtIntelOpenVinoOptions options) {
    return IntelOpenVinoOptions(options);
  }

  void SetDeviceType(LiteRtIntelOpenVinoDeviceType device_type) {
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsSetDeviceType(Get(), device_type));
  }

  LiteRtIntelOpenVinoDeviceType GetDeviceType() const {
    LiteRtIntelOpenVinoDeviceType device_type;
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsGetDeviceType(Get(), &device_type));
    return device_type;
  }

  void SetPerformanceMode(LiteRtIntelOpenVinoPerformanceMode performance_mode) {
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsSetPerformanceMode(Get(), performance_mode));
  }

  LiteRtIntelOpenVinoPerformanceMode GetPerformanceMode() const {
    LiteRtIntelOpenVinoPerformanceMode performance_mode;
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsGetPerformanceMode(Get(), &performance_mode));
    return performance_mode;
  }

  void SetConfigsMapOption(const char *key, const char *value) {
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsSetConfigsMapOption(Get(), key, value));
  }

  int GetNumConfigsMapOptions() const {
    int num_options;
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsGetNumConfigsMapOptions(Get(), &num_options));
    return num_options;
  }

  std::pair<std::string, std::string> GetConfigsMapOption(int index) const {
    const char *key = nullptr;
    const char *value = nullptr;
    auto status =
        LrtIntelOpenVinoOptionsGetConfigsMapOption(Get(), index, &key, &value);

    if (status != kLiteRtStatusOk) {
      return std::make_pair(std::string(), std::string());
    }

    return std::make_pair(std::string(key), std::string(value));
  }

  LrtIntelOpenVinoOptions Get() const { return options_.get(); }

private:
  explicit IntelOpenVinoOptions(LrtIntelOpenVinoOptions options)
      : options_(options) {}

  struct IntelOpenVinoOptionsDeleter {
    void operator()(LrtIntelOpenVinoOptions options) const {
      LrtDestroyIntelOpenVinoOptions(options);
    }
  };

  std::unique_ptr<LrtIntelOpenVinoOptionsT, IntelOpenVinoOptionsDeleter>
      options_;
};

} // namespace litert::intel_openvino

#endif // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
