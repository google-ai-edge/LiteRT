// Copyright (C) 2025 Intel Corporation
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

#include "litert/vendors/intel_openvino/compiler/openvino_soc_config.h"

#include <cstddef>
#include <string>
#include <unordered_map>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "openvino/core/any.hpp"

namespace litert::openvino {

// Map from SoC model codename to the NPU platform string used by OpenVINO.
const std::unordered_map<std::string, const char*> kSocModelConfigs = {
    {"LNL", "NPU4000"},
    {"PTL", "NPU5010"}};

size_t GetNumSocModels() { return kSocModelConfigs.size(); }

const char* GetSocModelName(size_t index) {
  if (index >= kSocModelConfigs.size()) return nullptr;
  return std::next(kSocModelConfigs.begin(), index)->first.c_str();
}

LiteRtStatus GetSocModelConfig(absl::string_view codename,
                               ov::AnyMap& config_map) {
  auto it = kSocModelConfigs.find(std::string(codename));
  if (it != kSocModelConfigs.end()) {
    config_map["NPU_PLATFORM"] = it->second;
    return kLiteRtStatusOk;
  }
  LITERT_LOG(LITERT_ERROR, "Unrecognized SoC model: %s", codename.data());
  return kLiteRtStatusErrorInvalidArgument;
}

LiteRtStatus ConfigureCompilationParams(const char* soc_model,
                                        ov::AnyMap& configs_map) {
  if (soc_model == nullptr) {
    LITERT_LOG(LITERT_INFO,
               "Soc model is null, Compile will proceed with default NPU "
               "compile.");
  } else {
    LITERT_RETURN_IF_ERROR(GetSocModelConfig(soc_model, configs_map));
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::openvino
