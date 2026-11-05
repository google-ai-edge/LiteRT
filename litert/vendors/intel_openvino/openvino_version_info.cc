// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/intel_openvino/openvino_version_info.h"

#include <cstdint>
#include <exception>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

#include "openvino/core/except.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"  // NOLINT
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"  // NOLINT

namespace litert::openvino {

namespace {

// Log specific CPU device/backend properties.
void LogCpuProperties(ov::Core& core, const std::string& dev,
                      const char* context_tag, bool is_device_instance) {
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
void LogGpuProperties(ov::Core& core, const std::string& dev,
                      const char* context_tag, bool is_device_instance) {
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
void LogNpuProperties(ov::Core& core, const std::string& dev,
                      const char* context_tag, bool is_device_instance,
                      const ov::AnyMap& compile_properties) {
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
    const std::string platform = ResolveNpuPlatformForLog(
        compile_properties, core.get_property(dev, ov::intel_npu::platform));
    if (!platform.empty()) {
      LITERT_LOG(LITERT_INFO, "[%s]   %s '%s' NPU platform: %s",
                 context_tag, type_label, dev.c_str(), platform.c_str());
    } else {
      LITERT_LOG(LITERT_INFO,
                 "[%s]   %s '%s' NPU platform: not specified (AUTO_DETECT)",
                 context_tag, type_label, dev.c_str());
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
void LogGeneralDeviceProperties(ov::Core& core, const std::string& dev,
                                const char* context_tag,
                                bool is_device_instance,
                                const ov::AnyMap& compile_properties) {
  if (absl::StrContains(dev, "CPU")) {
    LogCpuProperties(core, dev, context_tag, is_device_instance);
  } else if (absl::StrContains(dev, "GPU")) {
    LogGpuProperties(core, dev, context_tag, is_device_instance);
  } else if (absl::StrContains(dev, "NPU")) {
    LogNpuProperties(core, dev, context_tag, is_device_instance,
                     compile_properties);
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

}  // namespace

std::string GetOpenVINOVersionString() {
  try {
    const auto ov_ver = ov::get_openvino_version();
    std::string res = "OpenVINO ";
    if (ov_ver.buildNumber) {
      res += ov_ver.buildNumber;
    }
    if (ov_ver.description) {
      res += " (";
      res += ov_ver.description;
      res += ')';
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

std::string ResolveNpuPlatformForLog(const ov::AnyMap& compile_properties,
                                     const std::string& global_platform) {
  // The plugin-wide default; it is not a platform.
  const std::string kAutoDetect = "AUTO_DETECT";
  try {
    const auto it = compile_properties.find(ov::intel_npu::platform.name());
    if (it != compile_properties.end()) {
      const std::string platform = it->second.as<std::string>();
      if (!platform.empty() && platform != kAutoDetect) {
        return platform;
      }
    }
  } catch (const std::exception&) {
    // A malformed value falls back to the plugin-wide platform.
  }
  return global_platform == kAutoDetect ? std::string() : global_platform;
}

void LogOpenVINOVersionInfoOnce(const char* context_tag,
                                const char* target_device,
                                const ov::AnyMap& compile_properties) {
  static auto* const mutex = new absl::Mutex();
  static auto* const logged_targets =
      new std::set<std::pair<std::string, std::string>>();

  std::string ctx = context_tag ? context_tag : "OpenVINO";
  std::string target = target_device ? target_device : "";

  {
    absl::MutexLock lock(mutex);
    const std::string key =
        target + "|" + ResolveNpuPlatformForLog(compile_properties, "");
    if (!logged_targets->emplace(ctx, key).second) {
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
          LogGeneralDeviceProperties(core, dev, ctx.c_str(),
                                     /*is_device_instance=*/true,
                                     compile_properties);
        }
      }

      // If target device was not in enumerated list, query candidate backend
      // properties directly.
      if (!found_device) {
        LogGeneralDeviceProperties(core, target, ctx.c_str(),
                                   /*is_device_instance=*/false,
                                   compile_properties);
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
        if (absl::StrContains(dev, "CPU")) {
          has_cpu = true;
        }
        if (absl::StrContains(dev, "GPU")) {
          has_gpu = true;
        }
        if (absl::StrContains(dev, "NPU")) {
          has_npu = true;
        }

        // Query device specific properties
        LogGeneralDeviceProperties(core, dev, ctx.c_str(),
                                   /*is_device_instance=*/true,
                                   compile_properties);
      }
    } else {
      LITERT_LOG(
          LITERT_INFO,
          "[%s] No active hardware devices enumerated; querying backend "
          "properties directly.",
          ctx.c_str());
    }

    // Query candidate backend properties (CPU, GPU, NPU) if not covered by
    // enumerated devices.
    const std::vector<std::pair<std::string, bool>> candidate_backends = {
        {"CPU", has_cpu},
        {"GPU", has_gpu},
        {"NPU", has_npu},
    };

    for (const auto& [backend_name, already_found] : candidate_backends) {
      if (!already_found) {
        LogGeneralDeviceProperties(core, backend_name, ctx.c_str(),
                                   /*is_device_instance=*/false,
                                   compile_properties);
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
