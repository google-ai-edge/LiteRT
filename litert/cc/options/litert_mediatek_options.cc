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
#include "litert/cc/options/litert_mediatek_options.h"

#include <memory>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

// C++ WRAPPERS ////////////////////////////////////////////////////////////////
namespace litert::mediatek {
const char* MediatekOptions::Discriminator() {
  return LiteRtMediatekOptionsGetIdentifier();
}
Expected<MediatekOptions> MediatekOptions::Create(OpaqueOptions& options) {
  const auto id = options.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return MediatekOptions(options.Get(), OwnHandle::kNo);
}
Expected<MediatekOptions> MediatekOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtMediatekOptionsCreate(&options));
  return MediatekOptions(options, OwnHandle::kYes);
}

void MediatekOptions::SetNeronSDKVersionType(
    LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type) {
  internal::AssertOk(LiteRtMediatekOptionsSetNeronSDKVersionType, Data(),
                     sdk_version_type);
}

LiteRtMediatekOptionsNeronSDKVersionType
MediatekOptions::GetNeronSDKVersionType() {
  LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type;
  LiteRtMediatekOptions options_data = Data();
  internal::AssertOk(LiteRtMediatekOptionsGetNeronSDKVersionType, options_data,
                     &sdk_version_type);
  return sdk_version_type;
}

LiteRtMediatekOptions MediatekOptions::Data() const {
  LiteRtMediatekOptions options_data;
  internal::AssertOk(LiteRtMediatekOptionsGet, Get(), &options_data);
  return options_data;
}

void MediatekOptions::SetEnableGemmaCompilerOptimizations(
    bool enable_gemma_compiler_optimizations) {
  internal::AssertOk(LiteRtMediatekOptionsSetGemmaCompilerOptimizations, Data(),
                     enable_gemma_compiler_optimizations);
}

bool MediatekOptions::GetEnableGemmaCompilerOptimizations() {
  bool enable_gemma_compiler_optimizations;
  internal::AssertOk(LiteRtMediatekOptionsGetGemmaCompilerOptimizations, Data(),
                     &enable_gemma_compiler_optimizations);
  return enable_gemma_compiler_optimizations;
}

void MediatekOptions::SetPerformanceMode(
    LiteRtMediatekNeuronAdapterPerformanceMode performance_mode) {
  internal::AssertOk(LiteRtMediatekOptionsSetPerformanceMode, Data(),
                     performance_mode);
}

LiteRtMediatekNeuronAdapterPerformanceMode
MediatekOptions::GetPerformanceMode() {
  LiteRtMediatekNeuronAdapterPerformanceMode performance_mode;
  internal::AssertOk(LiteRtMediatekOptionsGetPerformanceMode, Data(),
                     &performance_mode);
  return performance_mode;
}

void MediatekOptions::SetEnableL1CacheOptimizations(
    bool enable_l1_cache_optimizations) {
  internal::AssertOk(LiteRtMediatekOptionsSetL1CacheOptimizations, Data(),
                     enable_l1_cache_optimizations);
}

bool MediatekOptions::GetEnableL1CacheOptimizations() {
  bool enable_l1_cache_optimizations;
  internal::AssertOk(LiteRtMediatekOptionsGetL1CacheOptimizations, Data(),
                     &enable_l1_cache_optimizations);
  return enable_l1_cache_optimizations;
}

void MediatekOptions::SetOptimizationHint(
    LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint) {
  internal::AssertOk(LiteRtMediatekOptionsSetOptimizationHint, Data(),
                     optimization_hint);
}

LiteRtMediatekNeuronAdapterOptimizationHint
MediatekOptions::GetOptimizationHint() {
  LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint;
  internal::AssertOk(LiteRtMediatekOptionsGetOptimizationHint, Data(),
                     &optimization_hint);
  return optimization_hint;
}

void MediatekOptions::SetDisableDlaDirRemoval(
    bool disable_dla_dir_removal) {
  internal::AssertOk(LiteRtMediatekOptionsSetDisableDlaDirRemoval, Data(),
                     disable_dla_dir_removal);
}

bool MediatekOptions::GetDisableDlaDirRemoval() {
  bool disable_dla_dir_removal;
  internal::AssertOk(LiteRtMediatekOptionsGetDisableDlaDirRemoval, Data(),
                     &disable_dla_dir_removal);
  return disable_dla_dir_removal;
}

void MediatekOptions::SetMediatekDlaDir(const std::string& mediatek_dla_dir) {
  internal::AssertOk(LiteRtMediatekOptionsSetMediatekDlaDir, Data(),
                     mediatek_dla_dir.c_str());
}

absl::string_view MediatekOptions::GetMediatekDlaDir() {
  const char* mediatek_dla_dir;
  internal::AssertOk(LiteRtMediatekOptionsGetMediatekDlaDir, Data(),
                     &mediatek_dla_dir);
  return absl::string_view(mediatek_dla_dir);
}

}  // namespace litert::mediatek
