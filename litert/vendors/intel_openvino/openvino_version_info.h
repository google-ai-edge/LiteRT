// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_
#define ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_

#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"

namespace litert::openvino {

// Returns a formatted OpenVINO runtime version string.
inline std::string GetOpenVINOVersionString() {
  try {
    const auto ov_ver = ov::get_openvino_version();
    std::string res = "OpenVINO ";
    if (ov_ver.buildNumber) {
      res += ov_ver.buildNumber;
    }
    if (ov_ver.description) {
      res += " (";
      res += ov_ver.description;
      res += ")";
    }
    return res;
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_WARNING, "Failed to get OpenVINO version: %s", e.what());
    return "OpenVINO (unknown version)";
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_WARNING, "Failed to get OpenVINO version: %s", e.what());
    return "OpenVINO (unknown version)";
  }
}

// Queries and logs OpenVINO version, device plugin versions, Level Zero driver version,
// and NPU compiler version once.
inline void LogOpenVINOVersionInfoOnce(const char* context_tag = "OpenVINO") {
  static std::once_flag flag;
  std::call_once(flag, [context_tag]() {
    const std::string ov_ver_str = GetOpenVINOVersionString();
    LITERT_LOG(LITERT_INFO, "[%s] Runtime version: %s", context_tag,
               ov_ver_str.c_str());

    try {
      ov::Core core;
      std::vector<std::string> devices = core.get_available_devices();
      for (const auto& dev : devices) {
        LITERT_LOG(LITERT_INFO, "[%s] Found device: %s", context_tag,
                   dev.c_str());

        // Query device plugin versions
        try {
          auto plugin_versions = core.get_versions(dev);
          for (const auto& [plugin_name, ver] : plugin_versions) {
            LITERT_LOG(LITERT_INFO,
                       "[%s]   Device '%s' plugin [%s] version: %s (%s)",
                       context_tag, dev.c_str(), plugin_name.c_str(),
                       ver.buildNumber ? ver.buildNumber : "unknown",
                       ver.description ? ver.description : "");
          }
        } catch (const ov::Exception& e) {
          LITERT_LOG(LITERT_WARNING,
                     "[%s]   Failed to query plugin versions for device '%s': %s",
                     context_tag, dev.c_str(), e.what());
        } catch (const std::exception& e) {
          LITERT_LOG(LITERT_WARNING,
                     "[%s]   Failed to query plugin versions for device '%s': %s",
                     context_tag, dev.c_str(), e.what());
        }

        // Query device full name
        try {
          std::string full_name =
              core.get_property(dev, ov::device::full_name);
          if (!full_name.empty()) {
            LITERT_LOG(LITERT_INFO, "[%s]   Device '%s' full name: %s",
                       context_tag, dev.c_str(), full_name.c_str());
          }
        } catch (const ov::Exception& e) {
          LITERT_LOG(LITERT_WARNING,
                     "[%s]   Failed to query full name for device '%s': %s",
                     context_tag, dev.c_str(), e.what());
        } catch (const std::exception& e) {
          LITERT_LOG(LITERT_WARNING,
                     "[%s]   Failed to query full name for device '%s': %s",
                     context_tag, dev.c_str(), e.what());
        }

        // NPU specific properties
        if (dev.find("NPU") != std::string::npos) {
          // Level Zero driver version
          try {
            uint32_t drv_ver =
                core.get_property(dev, ov::intel_npu::driver_version);
            LITERT_LOG(
                LITERT_INFO,
                "[%s]   Device '%s' Level Zero / NPU driver version: %u (0x%X)",
                context_tag, dev.c_str(), drv_ver, drv_ver);
          } catch (const ov::Exception& e) {
            LITERT_LOG(LITERT_WARNING,
                       "[%s]   Failed to query driver version for device '%s': %s",
                       context_tag, dev.c_str(), e.what());
          } catch (const std::exception& e) {
            LITERT_LOG(LITERT_WARNING,
                       "[%s]   Failed to query driver version for device '%s': %s",
                       context_tag, dev.c_str(), e.what());
          }

          // NPU compiler version
          try {
            uint32_t comp_ver =
                core.get_property(dev, ov::intel_npu::compiler_version);
            uint16_t comp_major = static_cast<uint16_t>(comp_ver >> 16);
            uint16_t comp_minor = static_cast<uint16_t>(comp_ver & 0xFFFF);
            LITERT_LOG(
                LITERT_INFO,
                "[%s]   Device '%s' NPU compiler version: %u.%u (raw: 0x%08X)",
                context_tag, dev.c_str(), comp_major, comp_minor, comp_ver);
          } catch (const ov::Exception& e) {
            LITERT_LOG(LITERT_WARNING,
                       "[%s]   Failed to query compiler version for device '%s': %s",
                       context_tag, dev.c_str(), e.what());
          } catch (const std::exception& e) {
            LITERT_LOG(LITERT_WARNING,
                       "[%s]   Failed to query compiler version for device '%s': %s",
                       context_tag, dev.c_str(), e.what());
          }
        }
      }
    } catch (const ov::Exception& e) {
      LITERT_LOG(LITERT_WARNING,
                 "[%s] Failed to query OpenVINO Core devices or properties: %s",
                 context_tag, e.what());
    } catch (const std::exception& e) {
      LITERT_LOG(LITERT_WARNING,
                 "[%s] Failed to query OpenVINO Core devices or properties: %s",
                 context_tag, e.what());
    }
  });
}

}  // namespace litert::openvino

#endif  // ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_
