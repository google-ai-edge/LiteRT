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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_OPENVINO_SOC_CONFIG_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_OPENVINO_SOC_CONFIG_H_

#include <cstddef>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "openvino/core/any.hpp"

namespace litert::openvino {

// Returns the number of supported SoC models.
size_t GetNumSocModels();

// Returns the SoC model codename at the given index, or nullptr if out of range.
const char* GetSocModelName(size_t index);

// Looks up the platform codename in the SoC-model-to-NPU-config map and
// sets the "NPU_PLATFORM" key in config_map to the corresponding NPU version
// string (e.g. "NPU4000" for "LNL").
// Returns kLiteRtStatusErrorInvalidArgument if the codename is not recognised.
LiteRtStatus GetSocModelConfig(absl::string_view codename,
                               ov::AnyMap& config_map);

// Applies SoC-model-specific compilation parameters to configs_map.
// If soc_model is nullptr, compilation proceeds with default NPU settings.
LiteRtStatus ConfigureCompilationParams(const char* soc_model,
                                        ov::AnyMap& configs_map);

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_OPENVINO_SOC_CONFIG_H_
