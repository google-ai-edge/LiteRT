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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_CPU_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_CPU_OPTIONS_H_

#include <cstdint>
#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

struct LrtCpuOptions;

namespace litert {

/// @brief Defines the C++ wrapper for LiteRT CPU options.
class CpuOptions {
 public:
  /// @brief Creates a new CPU options instance.
  static Expected<CpuOptions> Create() {
    LrtCpuOptions* options = nullptr;
    LITERT_RETURN_IF_ERROR(LrtCreateCpuOptions(&options));
    return CpuOptions(options);
  }

  /// @brief Sets the number of threads for the CPU backend.
  Expected<void> SetNumThreads(int num_threads) {
    LITERT_RETURN_IF_ERROR(
        LrtSetCpuOptionsNumThread(options_.get(), num_threads));
    return {};
  }

  /// @brief Gets the number of threads for the CPU backend that was set.
  Expected<int> GetNumThreads() const {
    int num_threads;
    auto s = LrtGetCpuOptionsNumThread(options_.get(), &num_threads);
    if (s == kLiteRtStatusErrorNotFound) {
      return 0;
    }
    LITERT_RETURN_IF_ERROR(s);
    return num_threads;
  }

  /// @brief Selects which CPU kernel mode LiteRT should use.
  Expected<void> SetKernelMode(LiteRtCpuKernelMode mode) {
    LITERT_RETURN_IF_ERROR(LrtSetCpuOptionsKernelMode(options_.get(), mode));
    return {};
  }

  /// @brief Gets the configured CPU kernel mode.
  Expected<LiteRtCpuKernelMode> GetKernelMode() const {
    LiteRtCpuKernelMode mode;
    auto s = LrtGetCpuOptionsKernelMode(options_.get(), &mode);
    if (s == kLiteRtStatusErrorNotFound) {
      return kLiteRtCpuKernelModeXnnpack;
    }
    LITERT_RETURN_IF_ERROR(s);
    return mode;
  }

  /// @brief Sets the XNNPack flags.
  Expected<void> SetXNNPackFlags(uint32_t flags) {
    LITERT_RETURN_IF_ERROR(LrtSetCpuOptionsXNNPackFlags(options_.get(), flags));
    return {};
  }

  /// @brief Gets the XNNPack flags that were set.
  ///
  /// To get a default XNNPack flags, use `TfLiteXNNPackDelegateOptionsDefault`.
  Expected<uint32_t> GetXNNPackFlags() const {
    uint32_t flags;
    auto s = LrtGetCpuOptionsXNNPackFlags(options_.get(), &flags);
    if (s == kLiteRtStatusErrorNotFound) {
      return 0;
    }
    LITERT_RETURN_IF_ERROR(s);
    return flags;
  }

  /// @brief Sets the XNNPack weight cache file path.
  Expected<void> SetXNNPackWeightCachePath(const char* path) {
    LITERT_RETURN_IF_ERROR(
        LrtSetCpuOptionsXnnPackWeightCachePath(options_.get(), path));
    return {};
  }

  /// @brief Gets the XNNPack weight cache file path.
  Expected<absl::string_view> GetXNNPackWeightCachePath() const {
    const char* path;
    auto s = LrtGetCpuOptionsXnnPackWeightCachePath(options_.get(), &path);
    if (s == kLiteRtStatusErrorNotFound) {
      return absl::string_view();
    }
    LITERT_RETURN_IF_ERROR(s);
    return absl::NullSafeStringView(path);
  }

  /// @brief Sets the XNNPack weight cache file descriptor.
  Expected<void> SetXNNPackWeightCacheFileDescriptor(int fd) {
    LITERT_RETURN_IF_ERROR(
        LrtSetCpuOptionsXnnPackWeightCacheFileDescriptor(options_.get(), fd));
    return {};
  }

  /// @brief Gets the XNNPack weight cache file descriptor.
  Expected<int> GetXNNPackWeightCacheFileDescriptor() const {
    int fd;
    auto s =
        LrtGetCpuOptionsXnnPackWeightCacheFileDescriptor(options_.get(), &fd);
    if (s == kLiteRtStatusErrorNotFound) {
      return -1;
    }
    LITERT_RETURN_IF_ERROR(s);
    return fd;
  }

  LrtCpuOptions* Get() { return options_.get(); }
  const LrtCpuOptions* Get() const { return options_.get(); }

 private:
  explicit CpuOptions(LrtCpuOptions* options) : options_(options) {}

  struct Deleter {
    void operator()(LrtCpuOptions* ptr) const { LrtDestroyCpuOptions(ptr); }
  };
  std::unique_ptr<LrtCpuOptions, Deleter> options_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_CPU_OPTIONS_H_
