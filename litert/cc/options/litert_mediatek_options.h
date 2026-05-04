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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::mediatek {

/// @brief Defines the C++ wrapper for MediaTek-specific LiteRT options.
class MediatekOptions {
 public:
  MediatekOptions() = delete;
  explicit MediatekOptions(LrtMediatekOptions* options,
                           OwnHandle own = OwnHandle::kYes)
      : options_(
            options, own == OwnHandle::kYes ? LrtDestroyMediatekOptions
                                            : [](LrtMediatekOptions*) {}) {}

  static Expected<MediatekOptions> Create() {
    LrtMediatekOptions* options = nullptr;
    LITERT_RETURN_IF_ERROR(LrtCreateMediatekOptions(&options));
    return MediatekOptions(options, OwnHandle::kYes);
  }

  LrtMediatekOptions* Get() const { return options_.get(); }
  LrtMediatekOptions* Release() { return options_.release(); }

  void SetNeronSDKVersionType(
      LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type) {
    internal::AssertOk(LrtSetMediatekOptionsNeronSDKVersionType, Get(),
                       sdk_version_type);
  }

  LiteRtMediatekOptionsNeronSDKVersionType GetNeronSDKVersionType() {
    LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type;
    internal::AssertOk(LrtGetMediatekOptionsNeronSDKVersionType, Get(),
                       &sdk_version_type);
    return sdk_version_type;
  }

  void SetEnableGemmaCompilerOptimizations(
      bool enable_gemma_compiler_optimizations) {
    internal::AssertOk(LrtSetMediatekOptionsGemmaCompilerOptimizations, Get(),
                       enable_gemma_compiler_optimizations);
  }

  bool GetEnableGemmaCompilerOptimizations() {
    bool enable_gemma_compiler_optimizations;
    internal::AssertOk(LrtGetMediatekOptionsGemmaCompilerOptimizations, Get(),
                       &enable_gemma_compiler_optimizations);
    return enable_gemma_compiler_optimizations;
  }

  void SetPerformanceMode(
      LiteRtMediatekNeuronAdapterPerformanceMode performance_mode) {
    internal::AssertOk(LrtSetMediatekOptionsPerformanceMode, Get(),
                       performance_mode);
  }

  LiteRtMediatekNeuronAdapterPerformanceMode GetPerformanceMode() {
    LiteRtMediatekNeuronAdapterPerformanceMode performance_mode;
    internal::AssertOk(LrtGetMediatekOptionsPerformanceMode, Get(),
                       &performance_mode);
    return performance_mode;
  }

  void SetEnableL1CacheOptimizations(bool enable_l1_cache_optimizations) {
    internal::AssertOk(LrtSetMediatekOptionsL1CacheOptimizations, Get(),
                       enable_l1_cache_optimizations);
  }

  bool GetEnableL1CacheOptimizations() {
    bool enable_l1_cache_optimizations;
    internal::AssertOk(LrtGetMediatekOptionsL1CacheOptimizations, Get(),
                       &enable_l1_cache_optimizations);
    return enable_l1_cache_optimizations;
  }

  void SetOptimizationHint(
      LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint) {
    internal::AssertOk(LrtSetMediatekOptionsOptimizationHint, Get(),
                       optimization_hint);
  }

  LiteRtMediatekNeuronAdapterOptimizationHint GetOptimizationHint() {
    LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint;
    internal::AssertOk(LrtGetMediatekOptionsOptimizationHint, Get(),
                       &optimization_hint);
    return optimization_hint;
  }

  void SetDisableDlaDirRemoval(bool disable_dla_dir_removal) {
    internal::AssertOk(LrtSetMediatekOptionsDisableDlaDirRemoval, Get(),
                       disable_dla_dir_removal);
  }

  bool GetDisableDlaDirRemoval() {
    bool disable_dla_dir_removal;
    internal::AssertOk(LrtGetMediatekOptionsDisableDlaDirRemoval, Get(),
                       &disable_dla_dir_removal);
    return disable_dla_dir_removal;
  }

  void SetMediatekDlaDir(const std::string& mediatek_dla_dir) {
    internal::AssertOk(LrtSetMediatekOptionsMediatekDlaDir, Get(),
                       mediatek_dla_dir.c_str());
  }

  absl::string_view GetMediatekDlaDir() {
    const char* mediatek_dla_dir;
    internal::AssertOk(LrtGetMediatekOptionsMediatekDlaDir, Get(),
                       &mediatek_dla_dir);
    return absl::string_view(mediatek_dla_dir);
  }

  void SetAotCompilationOptions(const std::string& aot_compilation_options) {
    internal::AssertOk(LrtSetMediatekOptionsAotCompilationOptions, Get(),
                       aot_compilation_options.c_str());
  }

  absl::string_view GetAotCompilationOptions() {
    const char* aot_compilation_options;
    internal::AssertOk(LrtGetMediatekOptionsAotCompilationOptions, Get(),
                       &aot_compilation_options);
    return absl::string_view(aot_compilation_options);
  }

 private:
  std::unique_ptr<LrtMediatekOptions, void (*)(LrtMediatekOptions*)> options_;
};

}  // namespace litert::mediatek

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_
