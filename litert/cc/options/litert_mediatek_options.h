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

#include <string>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::mediatek {

// Wraps a LiteRtMediatekOptions object for convenience.
class MediatekOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  MediatekOptions() = delete;

  static const char* Discriminator();

  static Expected<MediatekOptions> Create(OpaqueOptions& options);

  static Expected<MediatekOptions> Create();

  void SetNeronSDKVersionType(
      LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type);

  LiteRtMediatekOptionsNeronSDKVersionType GetNeronSDKVersionType();

  void SetEnableGemmaCompilerOptimizations(
      bool enable_gemma_compiler_optimizations);

  bool GetEnableGemmaCompilerOptimizations();

  void SetPerformanceMode(
      LiteRtMediatekNeuronAdapterPerformanceMode performance_mode);

  LiteRtMediatekNeuronAdapterPerformanceMode GetPerformanceMode();

  void SetEnableL1CacheOptimizations(bool enable_l1_cache_optimizations);

  bool GetEnableL1CacheOptimizations();

  void SetOptimizationHint(
      LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint);

  LiteRtMediatekNeuronAdapterOptimizationHint GetOptimizationHint();

  void SetDisableDlaDirRemoval(bool disable_dla_dir_removal);

  bool GetDisableDlaDirRemoval();

  void SetMediatekDlaDir(const std::string& mediatek_dla_dir);

  absl::string_view GetMediatekDlaDir();

 private:
  LiteRtMediatekOptions Data() const;
};

}  // namespace litert::mediatek

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_
