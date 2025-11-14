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
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"

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

Expected<qualcomm::QualcommOptions&> Options::GetQualcommOptions() {
  return EnsureOption(qualcomm_options_);
}

Expected<mediatek::MediatekOptions&> Options::GetMediatekOptions() {
  return EnsureOption(mediatek_options_);
}

Expected<void> Options::Build() {
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), gpu_options_));
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), qualcomm_options_));
  LITERT_RETURN_IF_ERROR(AppendAndReset(Get(), mediatek_options_));
  return {};
}

}  // namespace litert
