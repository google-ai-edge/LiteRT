// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_
#define ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_

#include <string>

namespace litert::openvino {

// Returns a formatted OpenVINO runtime version string.
std::string GetOpenVINOVersionString();

// Queries and logs OpenVINO runtime version and plugin/driver versions for
// a specific target device (e.g., "CPU", "GPU", "NPU", or empty/null for all
// enumerated devices).
// Logging is performed at most once per unique (context_tag, target_device)
// pair.
void LogOpenVINOVersionInfoOnce(const char* context_tag = "OpenVINO",
                                const char* target_device = nullptr);

}  // namespace litert::openvino

#endif  // ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_
