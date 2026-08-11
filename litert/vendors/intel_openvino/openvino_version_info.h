// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_
#define ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_

#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
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

namespace internal {

// Log specific CPU device/backend properties.
inline void LogCpuProperties(ov::Core& core, const std::string& dev,
                             const char* context_tag,
                             bool is_device_instance = true) {
  const char* type_label = is_device_instance ? "Device" : "Backend";
  try {
    std::string full_name = core.get_property(dev, ov::device::full_name);
    if (!full_name.empty()) {
      LITERT_LOG(LITERT_INFO, "[%s]   %s '%s' full name: %s",
                 context_tag, type_label, dev.c_str(), full_name.c_str());
    }
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query full name for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query full name for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  }
}

// Log specific GPU device/backend properties.
inline void LogGpuProperties(ov::Core& core, const std::string& dev,
                             const char* context_tag,
                             bool is_device_instance = true) {
  const char* type_label = is_device_instance ? "Device" : "Backend";
  try {
    std::string full_name = core.get_property(dev, ov::device::full_name);
    if (!full_name.empty()) {
      LITERT_LOG(LITERT_INFO, "[%s]   %s '%s' full name: %s",
                 context_tag, type_label, dev.c_str(), full_name.c_str());
    }
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query full name for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query full name for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  }

  try {
    std::string uarch = core.get_property(dev, ov::intel_gpu::uarch_version);
    if (!uarch.empty()) {
      LITERT_LOG(LITERT_INFO, "[%s]   %s '%s' GPU uarch version: %s",
                 context_tag, type_label, dev.c_str(), uarch.c_str());
    }
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query GPU uarch version for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query GPU uarch version for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  }

  try {
    int32_t eu_count =
        core.get_property(dev, ov::intel_gpu::execution_units_count);
    LITERT_LOG(LITERT_INFO, "[%s]   %s '%s' execution units count: %d",
               context_tag, type_label, dev.c_str(), eu_count);
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query execution units count for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query execution units count for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  }

  try {
    uint64_t total_mem =
        core.get_property(dev, ov::intel_gpu::device_total_mem_size);
    LITERT_LOG(LITERT_INFO,
               "[%s]   %s '%s' GPU total memory: %llu MB", context_tag,
               type_label, dev.c_str(),
               static_cast<unsigned long long>(total_mem / (1024 * 1024)));
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query GPU total memory for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query GPU total memory for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  }
}

// Log specific NPU device/backend properties.
inline void LogNpuProperties(ov::Core& core, const std::string& dev,
                             const char* context_tag,
                             bool is_device_instance = true) {
  const char* type_label = is_device_instance ? "Device" : "Backend";
  try {
    std::string full_name = core.get_property(dev, ov::device::full_name);
    if (!full_name.empty()) {
      LITERT_LOG(LITERT_INFO, "[%s]   %s '%s' full name: %s",
                 context_tag, type_label, dev.c_str(), full_name.c_str());
    }
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query full name for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query full name for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  }

  try {
    std::string platform = core.get_property(dev, ov::intel_npu::platform);
    if (!platform.empty()) {
      LITERT_LOG(LITERT_INFO, "[%s]   %s '%s' NPU platform: %s",
                 context_tag, type_label, dev.c_str(), platform.c_str());
    }
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query NPU platform for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query NPU platform for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  }

  // Level Zero driver version
  try {
    uint32_t drv_ver = core.get_property(dev, ov::intel_npu::driver_version);
    LITERT_LOG(LITERT_INFO,
               "[%s]   %s '%s' Level Zero / NPU driver version: %u (0x%X)",
               context_tag, type_label, dev.c_str(), drv_ver, drv_ver);
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query driver version for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query driver version for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  }

  // NPU compiler version
  try {
    uint32_t comp_ver = core.get_property(dev, ov::intel_npu::compiler_version);
    uint16_t comp_major = static_cast<uint16_t>(comp_ver >> 16);
    uint16_t comp_minor = static_cast<uint16_t>(comp_ver & 0xFFFF);
    LITERT_LOG(LITERT_INFO,
               "[%s]   %s '%s' NPU compiler version: %u.%u (raw: 0x%08X)",
               context_tag, type_label, dev.c_str(), comp_major, comp_minor,
               comp_ver);
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query compiler version for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query compiler version for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  }

  try {
    int64_t max_tiles = core.get_property(dev, ov::intel_npu::max_tiles);
    LITERT_LOG(LITERT_INFO, "[%s]   %s '%s' NPU max tiles: %lld",
               context_tag, type_label, dev.c_str(),
               static_cast<long long>(max_tiles));
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query max tiles for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_DEBUG,
               "[%s]   Failed to query max tiles for %s '%s': %s",
               context_tag, type_label, dev.c_str(), e.what());
  }
}

// Log general device properties (full name, etc.).
inline void LogGeneralDeviceProperties(ov::Core& core, const std::string& dev,
                                       const char* context_tag,
                                       bool is_device_instance = true) {
  if (dev.find("CPU") != std::string::npos) {
    LogCpuProperties(core, dev, context_tag, is_device_instance);
  } else if (dev.find("GPU") != std::string::npos) {
    LogGpuProperties(core, dev, context_tag, is_device_instance);
  } else if (dev.find("NPU") != std::string::npos) {
    LogNpuProperties(core, dev, context_tag, is_device_instance);
  } else {
    const char* type_label = is_device_instance ? "Device" : "Backend";
    try {
      std::string full_name = core.get_property(dev, ov::device::full_name);
      if (!full_name.empty()) {
        LITERT_LOG(LITERT_INFO, "[%s]   %s '%s' full name: %s",
                   context_tag, type_label, dev.c_str(), full_name.c_str());
      }
    } catch (const ov::Exception& e) {
      LITERT_LOG(LITERT_DEBUG,
                 "[%s]   Failed to query full name for %s '%s': %s",
                 context_tag, type_label, dev.c_str(), e.what());
    } catch (const std::exception& e) {
      LITERT_LOG(LITERT_DEBUG,
                 "[%s]   Failed to query full name for %s '%s': %s",
                 context_tag, type_label, dev.c_str(), e.what());
    }
  }
}

}  // namespace internal

// Queries and logs OpenVINO runtime version and plugin/driver versions for
// a specific target device (e.g., "CPU", "GPU", "NPU", or empty/null for all
// enumerated devices).
// Logging is performed at most once per unique (context_tag, target_device) pair.
inline void LogOpenVINOVersionInfoOnce(const char* context_tag = "OpenVINO",
                                      const char* target_device = nullptr) {
  static std::mutex mutex;
  static std::set<std::pair<std::string, std::string>> logged_targets;

  std::string ctx = context_tag ? context_tag : "OpenVINO";
  std::string target = target_device ? target_device : "";

  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!logged_targets.emplace(ctx, target).second) {
      return;
    }
  }

  const std::string ov_ver_str = GetOpenVINOVersionString();
  LITERT_LOG(LITERT_INFO, "[%s] Runtime version: %s", ctx.c_str(),
             ov_ver_str.c_str());

  try {
    ov::Core core;

    // If a specific target device is specified (e.g. "CPU", "GPU", "NPU")
    if (!target.empty()) {
      std::vector<std::string> available_devices;
      try {
        available_devices = core.get_available_devices();
      } catch (const ov::Exception& e) {
        LITERT_LOG(LITERT_DEBUG,
                   "[%s] Could not enumerate available devices: %s",
                   ctx.c_str(), e.what());
      } catch (const std::exception& e) {
        LITERT_LOG(LITERT_DEBUG,
                   "[%s] Could not enumerate available devices: %s",
                   ctx.c_str(), e.what());
      }

      bool found_device = false;
      for (const auto& dev : available_devices) {
        if (dev.find(target) != std::string::npos) {
          found_device = true;
          LITERT_LOG(LITERT_INFO, "[%s] Found device: %s", ctx.c_str(),
                     dev.c_str());
          internal::LogGeneralDeviceProperties(core, dev, ctx.c_str(),
                                               /*is_device_instance=*/true);
        }
      }

      // If target device was not in enumerated list, query candidate backend properties directly
      if (!found_device) {
        internal::LogGeneralDeviceProperties(core, target, ctx.c_str(),
                                             /*is_device_instance=*/false);
      }
      return;
    }

    // When no specific target is provided, query all enumerated devices
    std::vector<std::string> devices;
    try {
      devices = core.get_available_devices();
    } catch (const ov::Exception& e) {
      LITERT_LOG(LITERT_INFO,
                 "[%s] Could not enumerate available devices: %s",
                 ctx.c_str(), e.what());
    } catch (const std::exception& e) {
      LITERT_LOG(LITERT_INFO,
                 "[%s] Could not enumerate available devices: %s",
                 ctx.c_str(), e.what());
    }

    bool has_cpu = false;
    bool has_gpu = false;
    bool has_npu = false;

    if (!devices.empty()) {
      for (const auto& dev : devices) {
        LITERT_LOG(LITERT_INFO, "[%s] Found device: %s", ctx.c_str(),
                   dev.c_str());
        if (dev.find("CPU") != std::string::npos) {
          has_cpu = true;
        }
        if (dev.find("GPU") != std::string::npos) {
          has_gpu = true;
        }
        if (dev.find("NPU") != std::string::npos) {
          has_npu = true;
        }

        // Query device specific properties
        internal::LogGeneralDeviceProperties(core, dev, ctx.c_str(),
                                             /*is_device_instance=*/true);
      }
    } else {
      LITERT_LOG(LITERT_INFO,
                 "[%s] No active hardware devices enumerated; querying backend properties directly.",
                 ctx.c_str());
    }

    // Query candidate backend properties (CPU, GPU, NPU) if not covered by enumerated devices
    const std::vector<std::pair<std::string, bool>> candidate_backends = {
        {"CPU", has_cpu},
        {"GPU", has_gpu},
        {"NPU", has_npu},
    };

    for (const auto& [backend_name, already_found] : candidate_backends) {
      if (!already_found) {
        internal::LogGeneralDeviceProperties(core, backend_name, ctx.c_str(),
                                             /*is_device_instance=*/false);
      }
    }
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_WARNING,
               "[%s] Failed to query OpenVINO Core devices or properties: %s",
               ctx.c_str(), e.what());
  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_WARNING,
               "[%s] Failed to query OpenVINO Core devices or properties: %s",
               ctx.c_str(), e.what());
  }
}

}  // namespace litert::openvino

#endif  // ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_OPENVINO_VERSION_INFO_H_
