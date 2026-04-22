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

#include <memory>
#include <optional>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/c/options/litert_runtime_options.h"
#include "litert/c/options/litert_samsung_options.h"
#include "litert/cc/internal/litert_runtime_proxy.h"
#include "litert/cc/internal/scoped_file.h"
#include "litert/cc/internal/scoped_weight_source.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/cc/options/litert_samsung_options.h"
#include "litert/core/options.h"

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
LiteRtStatus AppendAndReset(internal::RuntimeProxy* runtime,
                            LiteRtOptions options,
                            std::optional<OptionType>& slot) {
  if (!slot) {
    return kLiteRtStatusOk;
  }
  LiteRtOpaqueOptions opaque = slot->Release();
  slot.reset();
  return runtime->AddOpaqueOptions(options, opaque);
}

template <typename OptionType, typename GetDataFunc>
LiteRtStatus AppendAndResetOpaqueData(internal::RuntimeProxy* runtime,
                                      LiteRtOptions options,
                                      const std::optional<OptionType>& slot,
                                      GetDataFunc get_data_func) {
  if (!slot) {
    return kLiteRtStatusOk;
  }
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_RETURN_IF_ERROR(
      get_data_func(slot->Get(), &identifier, &payload, &payload_deleter));
  LiteRtOpaqueOptions opaque_opts = nullptr;
  LITERT_RETURN_IF_ERROR(runtime->CreateOpaqueOptions(
      identifier, payload, payload_deleter, &opaque_opts));
  LITERT_RETURN_IF_ERROR(runtime->AddOpaqueOptions(options, opaque_opts));
  return kLiteRtStatusOk;
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

Expected<samsung::SamsungOptions&> Options::GetSamsungOptions() {
  return EnsureOption(samsung_options_);
}

Expected<RuntimeOptions&> Options::GetRuntimeOptions() {
  return EnsureOption(runtime_options_);
}

Expected<CompilerOptions&> Options::GetCompilerOptions() {
  return EnsureOption(compiler_options_);
}

Expected<internal::LiteRtOptionsPtr> Options::Build(
    const Options& options, const internal::EnvironmentHolder& env) {
  auto* runtime = env.runtime;
  LiteRtOptions litert_options;
  LITERT_RETURN_IF_ERROR(runtime->CreateOptions(&litert_options));

  if (options.lite_rt_hw_accelerator_set_.has_value()) {
    LITERT_RETURN_IF_ERROR(runtime->SetOptionsHardwareAccelerators(
        litert_options, *options.lite_rt_hw_accelerator_set_));
  }

  for (auto& litert_opaque_options : options.opaque_options_) {
    LITERT_RETURN_IF_ERROR(
        runtime->AddOpaqueOptions(litert_options, litert_opaque_options));
  }

  for (const auto& action : options.build_actions_) {
    LITERT_RETURN_IF_ERROR(action(runtime, litert_options));
  }

  LITERT_RETURN_IF_ERROR(AppendAndResetOpaqueData(runtime, litert_options,
                                                  options.gpu_options_,
                                                  LrtGetOpaqueGpuOptionsData));
  LITERT_RETURN_IF_ERROR(AppendAndResetOpaqueData(runtime, litert_options,
                                                  options.cpu_options_,
                                                  LrtGetOpaqueCpuOptionsData));
  LITERT_RETURN_IF_ERROR(AppendAndResetOpaqueData(
      runtime, litert_options, options.qualcomm_options_,
      LrtGetOpaqueQualcommOptionsData));
  LITERT_RETURN_IF_ERROR(AppendAndResetOpaqueData(
      runtime, litert_options, options.mediatek_options_,
      LrtGetOpaqueMediatekOptionsData));
  LITERT_RETURN_IF_ERROR(AppendAndResetOpaqueData(
      runtime, litert_options, options.google_tensor_options_,
      LrtGetOpaqueGoogleTensorOptionsData));
  LITERT_RETURN_IF_ERROR(AppendAndResetOpaqueData(
      runtime, litert_options, options.intel_openvino_options_,
      LrtGetOpaqueIntelOpenVinoOptionsData));
  LITERT_RETURN_IF_ERROR(AppendAndResetOpaqueData(
      runtime, litert_options, options.samsung_options_,
      LrtGetOpaqueSamsungOptionsData));
  LITERT_RETURN_IF_ERROR(AppendAndResetOpaqueData(
      runtime, litert_options, options.runtime_options_,
      LrtGetOpaqueRuntimeOptionsData));
  LITERT_RETURN_IF_ERROR(AppendAndResetOpaqueData(
      runtime, litert_options, options.compiler_options_,
      LrtGetOpaqueCompilerOptionsData));

  return internal::LiteRtOptionsPtr(
      litert_options, internal::LiteRtDestroyOptionsDeleter{
                          runtime->runtime_c_api_->litert_destroy_options});
}

Expected<void> Options::SetExternalWeightScopedFile(
    ScopedFile& scoped_file, ScopedWeightSectionMap sections) {
  if (!scoped_file.IsValid()) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Scoped file handle must be valid");
  }
  if (sections.empty()) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "At least one external buffer group must be provided");
  }
  for (const auto& [name, section] : sections) {
    if (section.length == 0) {
      return Unexpected(Status::kErrorInvalidArgument,
                        "Section length must be positive for group " + name);
    }
  }

  auto scoped_weight_source = std::make_unique<ScopedWeightSource>(
      std::move(scoped_file), std::move(sections));
  build_actions_.push_back(
      [scoped_weight_source_ptr = scoped_weight_source.release()](
          internal::RuntimeProxy* runtime, LiteRtOptions options) {
        auto* options_impl = reinterpret_cast<LiteRtOptionsT*>(options);
        if (!options_impl) {
          return kLiteRtStatusErrorRuntimeFailure;
        }
        options_impl->scoped_weight_source.reset(scoped_weight_source_ptr);
        return kLiteRtStatusOk;
      });
  return {};
}

}  // namespace litert
