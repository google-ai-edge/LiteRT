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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_

#include <memory>

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_runtime_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

/// @brief Defines the C++ wrapper for LiteRT runtime options.
class RuntimeOptions {
public:
  /// @brief Creates a new `RuntimeOptions` instance with default values.
  static Expected<RuntimeOptions> Create() {
    LrtRuntimeOptions *options = nullptr;
    LITERT_RETURN_IF_ERROR(LrtCreateRuntimeOptions(&options));
    return RuntimeOptions(options);
  }

  /// @brief Sets the profiling enablement flag.
  /// @param enable_profiling If true, profiling will be enabled.
  Expected<void> SetEnableProfiling(bool enable_profiling) {
    LITERT_RETURN_IF_ERROR(
        LrtSetRuntimeOptionsEnableProfiling(options_.get(), enable_profiling));
    return {};
  }

  /// @brief Gets the current profiling enablement flag.
  Expected<bool> GetEnableProfiling() const {
    bool enable_profiling;
    LITERT_RETURN_IF_ERROR(
        LrtGetRuntimeOptionsEnableProfiling(options_.get(), &enable_profiling));
    return enable_profiling;
  }

  /// @brief Sets the error reporter mode.
  /// @param error_reporter_mode The mode for error reporting (e.g., stderr,
  /// buffer).
  Expected<void>
  SetErrorReporterMode(LiteRtErrorReporterMode error_reporter_mode) {
    LITERT_RETURN_IF_ERROR(LrtSetRuntimeOptionsErrorReporterMode(
        options_.get(), error_reporter_mode));
    return {};
  }

  /// @brief Gets the current error reporter mode.
  Expected<LiteRtErrorReporterMode> GetErrorReporterMode() const {
    LiteRtErrorReporterMode error_reporter_mode;
    LITERT_RETURN_IF_ERROR(LrtGetRuntimeOptionsErrorReporterMode(
        options_.get(), &error_reporter_mode));
    return error_reporter_mode;
  }

  /// @brief Sets the flag for compressing quantization zero points.
  /// @param compress_zero_points If true, identical per-channel quantization
  /// zero-points will be compressed.
  Expected<void> SetCompressQuantizationZeroPoints(bool compress_zero_points) {
    LITERT_RETURN_IF_ERROR(LrtSetRuntimeOptionsCompressQuantizationZeroPoints(
        options_.get(), compress_zero_points));
    return {};
  }

  /// @brief Gets the current flag for compressing quantization zero points.
  Expected<bool> GetCompressQuantizationZeroPoints() const {
    bool compress_zero_points;
    LITERT_RETURN_IF_ERROR(LrtGetRuntimeOptionsCompressQuantizationZeroPoints(
        options_.get(), &compress_zero_points));
    return compress_zero_points;
  }

  /// @brief Sets the flag for disabling delegate clustering.
  /// @param disable_delegate_clustering If true, delegate clustering will be
  /// disabled.
  Expected<void>
  SetDisableDelegateClustering(bool disable_delegate_clustering) {
    LITERT_RETURN_IF_ERROR(LrtSetRuntimeOptionsDisableDelegateClustering(
        options_.get(), disable_delegate_clustering));
    return {};
  }

  /// @brief Gets the current flag for disabling delegate clustering.
  Expected<bool> GetDisableDelegateClustering() const {
    bool disable_delegate_clustering;
    LITERT_RETURN_IF_ERROR(LrtGetRuntimeOptionsDisableDelegateClustering(
        options_.get(), &disable_delegate_clustering));
    return disable_delegate_clustering;
  }

  /// @brief Gets the underlying C options object.
  LrtRuntimeOptions *Get() { return options_.get(); }

  /// @brief Gets the underlying C options object.
  const LrtRuntimeOptions *Get() const { return options_.get(); }

private:
  explicit RuntimeOptions(LrtRuntimeOptions *options) : options_(options) {}

  struct Deleter {
    void operator()(LrtRuntimeOptions *ptr) const {
      LrtDestroyRuntimeOptions(ptr);
    }
  };
  std::unique_ptr<LrtRuntimeOptions, Deleter> options_;
};

} // namespace litert

#endif // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
