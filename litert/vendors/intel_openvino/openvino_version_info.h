// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_
#define ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_

#include <string>

#include "openvino/core/any.hpp"

namespace litert::openvino {

// Returns a formatted OpenVINO runtime version string.
std::string GetOpenVINOVersionString();

// Returns the NPU platform to log: NPU_PLATFORM from |compile_properties| if
// set, else |global_platform|. Empty if only the AUTO_DETECT default is known.
std::string ResolveNpuPlatformForLog(const ov::AnyMap& compile_properties,
                                     const std::string& global_platform);

// Queries and logs OpenVINO runtime version and plugin/driver versions for
// a specific target device (e.g., "CPU", "GPU", "NPU", or empty/null for all
// enumerated devices).
// |compile_properties| are the properties passed to ov::Core::compile_model;
// their NPU_PLATFORM is logged instead of the plugin-wide default.
// Logging is performed at most once per unique (context_tag, target_device,
// NPU_PLATFORM) combination.
void LogOpenVINOVersionInfoOnce(const char* context_tag = "OpenVINO",
                                const char* target_device = nullptr,
                                const ov::AnyMap& compile_properties = {});

}  // namespace litert::openvino

#endif  // ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_
