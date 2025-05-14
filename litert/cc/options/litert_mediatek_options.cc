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

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
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

}  // namespace litert::mediatek
