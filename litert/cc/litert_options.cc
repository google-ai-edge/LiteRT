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

#include "litert/cc/litert_options.h"

#include <optional>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/litert_options.h"
#include "litert/cc/internal/scoped_file.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#if defined(LITERT_WITH_EXTERNAL_WEIGHT_LOADER)
#include "litert/core/options.h"
#endif  // defined(LITERT_WITH_EXTERNAL_WEIGHT_LOADER)

namespace litert {

namespace {

template <typename OptionType>
Expected<OptionType&> EnsureOption(std::optional<OptionType>& slot) {
  if (!slot) {
    LITERT_ASSIGN_OR_RETURN(auto option, OptionType::Create());
    slot.emplace(std::move(option));
  }
  return slot.value();
}

template <typename OptionType>
LiteRtStatus AppendAndReset(LiteRtOptions options,
                            std::optional<OptionType>& slot) {
  if (!slot) {
    return kLiteRtStatusOk;
  }
  LiteRtOpaqueOptions opaque = slot->Release();
  slot.reset();
  return LiteRtAddOpaqueOptions(options, opaque);
}

}  // namespace

Expected<GpuOptions&> Options::GetGpuOptions() {
  return EnsureOption(gpu_options_);
}

Expected<CpuOptions&> Options::GetCpuOptions() {
  return EnsureOption(cpu_options_);
}

Expected<qualcomm::QualcommOptions&> Options::GetQualcommOptions() {
  return EnsureOption(qualcomm_options_);
}

Expected<mediatek::MediatekOptions&> Options::GetMediatekOptions() {
  return EnsureOption(mediatek_options_);
}

Expected<google_tensor::GoogleTensorOptions&>
Options::GetGoogleTensorOptions() {
  return EnsureOption(google_tensor_options_);
}

Expected<intel_openvino::IntelOpenVinoOptions&>
Options::GetIntelOpenVinoOptions() {
  return EnsureOption(intel_openvino_options_);
}

Expected<RuntimeOptions&> Options::GetRuntimeOptions() {
  return EnsureOption(runtime_options_);
}

Expected<CompilerOptions&> Options::GetCompilerOptions() {
  return EnsureOption(compiler_options_);
}

Expected<void> Options::Build() {
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), gpu_options_));
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), cpu_options_));
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), qualcomm_options_));
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), mediatek_options_));
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), google_tensor_options_));
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), intel_openvino_options_));
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), runtime_options_));
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), compiler_options_));
  return {};
}

Expected<void> Options::SetExternalWeightScopedFile(
    ScopedFile& scoped_file, ScopedWeightSectionMap sections) {
#if defined(LITERT_WITH_EXTERNAL_WEIGHT_LOADER)
  if (!scoped_file.IsValid()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Scoped file handle must be valid");
  }
  if (sections.empty()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "At least one external buffer group must be provided");
  }
  for (const auto& [name, section] : sections) {
    if (section.length == 0) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Section length must be positive for group " + name);
    }
  }
  auto* options_impl = reinterpret_cast<LiteRtOptionsT*>(Get());
  if (!options_impl) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Options handle must not be null");
  }
  options_impl->scoped_weight_source = std::make_unique<ScopedWeightSource>(
      std::move(scoped_file), std::move(sections));
  return {};
#else
  return Unexpected(kLiteRtStatusErrorInvalidArgument,
                    "LiteRT was built without external weight loader support");
#endif
}

}  // namespace litert
