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
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/litert_expected.h"

struct LrtCpuOptions;

namespace litert {

/// @brief Defines the C++ wrapper for LiteRT CPU options.
class CpuOptions {
 public:
  /// @brief Creates a new CPU options instance.
  static Expected<CpuOptions> Create();

  /// @brief Sets the number of threads for the CPU backend.
  Expected<void> SetNumThreads(int num_threads);

  /// @brief Gets the number of threads for the CPU backend that was set.
  Expected<int> GetNumThreads() const;

  /// @brief Sets the XNNPack flags.
  Expected<void> SetXNNPackFlags(uint32_t flags);

  /// @brief Gets the XNNPack flags that were set.
  ///
  /// To get a default XNNPack flags, use `TfLiteXNNPackDelegateOptionsDefault`.
  Expected<uint32_t> GetXNNPackFlags() const;

  /// @brief Sets the XNNPack weight cache file path.
  Expected<void> SetXNNPackWeightCachePath(const char* path);

  /// @brief Gets the XNNPack weight cache file path.
  Expected<absl::string_view> GetXNNPackWeightCachePath() const;

  /// @brief Sets the XNNPack weight cache file descriptor.
  Expected<void> SetXNNPackWeightCacheFileDescriptor(int fd);

  /// @brief Gets the XNNPack weight cache file descriptor.
  Expected<int> GetXNNPackWeightCacheFileDescriptor() const;

  LrtCpuOptions* Get() { return options_.get(); }
  const LrtCpuOptions* Get() const { return options_.get(); }

 private:
  explicit CpuOptions(LrtCpuOptions* options);

  struct Deleter {
    void operator()(LrtCpuOptions* ptr) const { LrtDestroyCpuOptions(ptr); }
  };
  std::unique_ptr<LrtCpuOptions, Deleter> options_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_CPU_OPTIONS_H_
