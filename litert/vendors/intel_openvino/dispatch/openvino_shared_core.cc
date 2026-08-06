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

#include "litert/vendors/intel_openvino//dispatch/openvino_shared_core.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>  // NOLINT
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "litert/c/internal/litert_logging.h"

OpenVINOSharedCore::OpenVINOSharedCore()
    : core_(std::make_shared<ov::Core>()) {}

OpenVINOSharedCore::~OpenVINOSharedCore() {
  // Unlink the staged weights bank. The singleton outlives every compiled
  // model that imported it, so this only runs at process teardown.
  std::string path;
  {
    std::lock_guard<std::mutex> lock(bank_mu_);
    path = bank_path_;
  }
  if (!path.empty()) {
    std::error_code ec;
    std::filesystem::remove(std::filesystem::path(path), ec);
  }
}

// static
OpenVINOSharedCore* OpenVINOSharedCore::GetInstance() {
  static OpenVINOSharedCore instance;
  return &instance;
}

const std::vector<std::string>& OpenVINOSharedCore::GetAvailableDevices() {
  std::call_once(available_devices_once_, [this]() {
    try {
      available_devices_ = core_->get_available_devices();
    } catch (const std::exception&) {
      available_devices_.clear();
    }
  });
  return available_devices_;
}

std::string OpenVINOSharedCore::GetBankPath() {
  std::lock_guard<std::mutex> lock(bank_mu_);
  return bank_path_;
}

std::string OpenVINOSharedCore::EnsureBankOnDisk(const void* data,
                                                 size_t size) {
  std::lock_guard<std::mutex> lock(bank_mu_);
  if (!bank_path_.empty()) return bank_path_;  // write once, reuse

  // Guard against a null/empty pool: writing 0 bytes yields an empty file that
  // NPUW would mmap to a zero-length region, so data()+bin_offset is undefined.
  if (data == nullptr || size == 0) {
    LITERT_LOG(LITERT_ERROR,
               "EnsureBankOnDisk: refusing to stage null/empty pool "
               "(data=%p size=%zu)",
               data, size);
    return {};
  }

  // Deployment override: point at a pre-staged bank and skip the write.
  if (const char* override_path = std::getenv("LITERT_OV_WEIGHTS_PATH");
      override_path != nullptr && override_path[0] != '\0') {
    bank_path_ = override_path;
    LITERT_LOG(LITERT_INFO,
               "NPU weight sharing: using LITERT_OV_WEIGHTS_PATH override '%s'",
               bank_path_.c_str());
    return bank_path_;
  }

  std::error_code ec;
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path(ec);
  if (ec) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: no temp directory available: %s",
               ec.message().c_str());
    return {};
  }

  // std::filesystem has no mkstemp equivalent; probe random names in the system
  // temp dir. std::random_device + exists() is portable across Windows / Linux
  // / Android (unlike mkstemp, which is POSIX-only).
  std::random_device rd;
  std::filesystem::path file_path;
  bool found = false;
  for (int attempt = 0; attempt < 64; ++attempt) {
    const uint64_t r = (static_cast<uint64_t>(rd()) << 32) ^
                       static_cast<uint64_t>(rd());
    char name[40];
    std::snprintf(name, sizeof(name), "litert_ov_bank_%016llx.bin",
                  static_cast<unsigned long long>(r));
    file_path = tmp_dir / name;
    if (!std::filesystem::exists(file_path, ec)) {
      found = true;
      break;
    }
  }
  if (!found) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: could not find a free temp file name");
    return {};
  }

  std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: failed to open temp file '%s'",
               file_path.string().c_str());
    return {};
  }
  out.write(static_cast<const char*>(data),
            static_cast<std::streamsize>(size));
  out.close();
  if (!out) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: failed to write temp file '%s'",
               file_path.string().c_str());
    std::filesystem::remove(file_path, ec);
    return {};
  }

  bank_path_ = file_path.string();
  LITERT_LOG(LITERT_INFO,
             "NPU weight sharing: staged %zu-byte weights bank to '%s'", size,
             bank_path_.c_str());
  return bank_path_;
}
