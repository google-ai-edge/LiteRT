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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_INTEL_OPENVINO_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_INTEL_OPENVINO_FLAGS_H_

#include <string>

#include "absl/flags/declare.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_intel_openvino_options.h"

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

#if defined(INCLUDE_INTEL_OPENVINO_COMPILE_FLAGS)

ABSL_DECLARE_FLAG(LiteRtIntelOpenVinoDeviceType, intel_openvino_device_type);

ABSL_DECLARE_FLAG(LiteRtIntelOpenVinoPerformanceMode,
                  intel_openvino_performance_mode);

ABSL_DECLARE_FLAG(std::string, intel_openvino_configs_map);

bool AbslParseFlag(absl::string_view text,
                   LiteRtIntelOpenVinoDeviceType* options, std::string* error);

std::string AbslUnparseFlag(LiteRtIntelOpenVinoDeviceType options);

bool AbslParseFlag(absl::string_view text,
                   LiteRtIntelOpenVinoPerformanceMode* options,
                   std::string* error);

std::string AbslUnparseFlag(LiteRtIntelOpenVinoPerformanceMode options);

#endif

// PARSERS (internal) //////////////////////////////////////////////////////////

#if defined(INCLUDE_INTEL_OPENVINO_COMPILE_FLAGS) || \
    defined(INCLUDE_INTEL_OPENVINO_RUNTIME_FLAGS)

namespace litert::intel_openvino {

Expected<IntelOpenVinoOptions> IntelOpenVinoOptionsFromFlags();

}  // namespace litert::intel_openvino

#endif  // INCLUDE_INTEL_OPENVINO_COMPILE_FLAGS ||
        // INCLUDE_INTEL_OPENVINO_RUNTIME_FLAGS
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_INTEL_OPENVINO_FLAGS_H_
