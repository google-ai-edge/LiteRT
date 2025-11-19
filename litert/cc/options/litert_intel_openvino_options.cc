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

#include "litert/cc/options/litert_intel_openvino_options.h"

#include <string>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

// C++ WRAPPERS ////////////////////////////////////////////////////////////////
namespace litert::intel_openvino {

const char* IntelOpenVinoOptions::Discriminator() {
  return LiteRtIntelOpenVinoOptionsGetIdentifier();
}

Expected<IntelOpenVinoOptions> IntelOpenVinoOptions::Create(
    OpaqueOptions& options) {
  const auto id = options.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return IntelOpenVinoOptions(options.Get(), OwnHandle::kNo);
}

Expected<IntelOpenVinoOptions> IntelOpenVinoOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtIntelOpenVinoOptionsCreate(&options));
  return IntelOpenVinoOptions(options, OwnHandle::kYes);
}

void IntelOpenVinoOptions::SetDeviceType(
    LiteRtIntelOpenVinoDeviceType device_type) {
  LITERT_ABORT_IF_ERROR(
      LiteRtIntelOpenVinoOptionsSetDeviceType(Data(), device_type));
}

LiteRtIntelOpenVinoDeviceType IntelOpenVinoOptions::GetDeviceType() const {
  LiteRtIntelOpenVinoDeviceType device_type;
  LITERT_ABORT_IF_ERROR(
      LiteRtIntelOpenVinoOptionsGetDeviceType(Data(), &device_type));
  return device_type;
}

void IntelOpenVinoOptions::SetPerformanceMode(
    LiteRtIntelOpenVinoPerformanceMode performance_mode) {
  LITERT_ABORT_IF_ERROR(
      LiteRtIntelOpenVinoOptionsSetPerformanceMode(Data(), performance_mode));
}

LiteRtIntelOpenVinoPerformanceMode IntelOpenVinoOptions::GetPerformanceMode()
    const {
  LiteRtIntelOpenVinoPerformanceMode performance_mode;
  LITERT_ABORT_IF_ERROR(
      LiteRtIntelOpenVinoOptionsGetPerformanceMode(Data(), &performance_mode));
  return performance_mode;
}

void IntelOpenVinoOptions::SetConfigsMapOption(const char* key,
                                               const char* value) {
  LITERT_ABORT_IF_ERROR(
      LiteRtIntelOpenVinoOptionsSetConfigsMapOption(Data(), key, value));
}

int IntelOpenVinoOptions::GetNumConfigsMapOptions() const {
  int num_options;
  LITERT_ABORT_IF_ERROR(
      LiteRtIntelOpenVinoOptionsGetNumConfigsMapOptions(Data(), &num_options));
  return num_options;
}

std::pair<std::string, std::string> IntelOpenVinoOptions::GetConfigsMapOption(
    int index) const {
  const char* key = nullptr;
  const char* value = nullptr;
  auto status = LiteRtIntelOpenVinoOptionsGetConfigsMapOption(Data(), index,
                                                              &key, &value);

  if (status != kLiteRtStatusOk) {
    // Return empty strings on error
    return std::make_pair(std::string(), std::string());
  }

  // Create string copies and return
  return std::make_pair(std::string(key), std::string(value));
}

LiteRtIntelOpenVinoOptions IntelOpenVinoOptions::Data() const {
  LiteRtIntelOpenVinoOptions options_data;
  LITERT_ABORT_IF_ERROR(LiteRtIntelOpenVinoOptionsGet(Get(), &options_data));
  return options_data;
}

}  // namespace litert::intel_openvino
