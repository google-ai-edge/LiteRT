// Copyright 2024 Google LLC.
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

#include "litert/c/litert_environment.h"

#include <algorithm>
#include <array>
#include <utility>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerators/auto_registration.h"
#if !defined(LITERT_DISABLE_GPU)
#include "litert/runtime/gpu_environment.h"
#endif  // !defined(LITERT_DISABLE_GPU)

#ifdef __cplusplus
extern "C" {
#endif

LiteRtStatus LiteRtCreateEnvironment(int num_options,
                                     const LiteRtEnvOption* options,
                                     LiteRtEnvironment* environment) {
  LITERT_RETURN_IF_ERROR(environment != nullptr,
                         kLiteRtStatusErrorInvalidArgument);

  auto options_span = absl::MakeSpan(options, num_options);
  LITERT_ASSIGN_OR_RETURN(auto env,
                          LiteRtEnvironmentT::CreateWithOptions(options_span));
  litert::TriggerAcceleratorAutomaticRegistration(*env);

  // Check if any GPU-related options are present using modern C++ algorithms
  constexpr std::array<LiteRtEnvOptionTag, 7> kGpuOptionTags = {
      kLiteRtEnvOptionTagOpenClDeviceId, kLiteRtEnvOptionTagOpenClPlatformId,
      kLiteRtEnvOptionTagOpenClContext,  kLiteRtEnvOptionTagOpenClCommandQueue,
      kLiteRtEnvOptionTagEglContext,     kLiteRtEnvOptionTagEglDisplay,
      kLiteRtEnvOptionTagMetalDevice};

  const bool has_gpu_options = std::any_of(
      options_span.begin(), options_span.end(),
      [&kGpuOptionTags](const LiteRtEnvOption& option) {
        return std::find(kGpuOptionTags.begin(), kGpuOptionTags.end(),
                         option.tag) != kGpuOptionTags.end();
      });

  if (has_gpu_options) {
    LITERT_ASSIGN_OR_RETURN(
        auto gpu_env, litert::internal::GpuEnvironment::Create(env.get()));
    LITERT_RETURN_IF_ERROR(env->SetGpuEnvironment(std::move(gpu_env)));
  }

  *environment = env.release();
  return kLiteRtStatusOk;
}

void LiteRtDestroyEnvironment(LiteRtEnvironment environment) {
  if (environment != nullptr) {
    delete environment;
  }
}

LiteRtStatus LiteRtGetEnvironmentOptions(LiteRtEnvironment environment,
                                         LiteRtEnvironmentOptions* options) {
  LITERT_RETURN_IF_ERROR(
      environment, litert::ErrorStatusBuilder(kLiteRtStatusErrorInvalidArgument)
                       << "Environment pointer is null.");
  LITERT_RETURN_IF_ERROR(
      options, litert::ErrorStatusBuilder(kLiteRtStatusErrorInvalidArgument)
                   << "Options pointer is null.");
  *options = &environment->GetOptions();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAddEnvironmentOptions(LiteRtEnvironment environment,
                                         int num_options,
                                         const LiteRtEnvOption* options,
                                         bool overwrite) {
  LITERT_RETURN_IF_ERROR(
      environment, litert::ErrorStatusBuilder(kLiteRtStatusErrorInvalidArgument)
                       << "Environment pointer is null.");
  LITERT_RETURN_IF_ERROR(
      options, litert::ErrorStatusBuilder(kLiteRtStatusErrorInvalidArgument)
                   << "Options pointer is null.");
  LITERT_RETURN_IF_ERROR(
      environment->AddOptions(absl::MakeSpan(options, num_options), overwrite));
#if !defined(LITERT_DISABLE_GPU)
  if (environment->HasGpuEnvironment()) {
    LITERT_ASSIGN_OR_RETURN(litert::internal::GpuEnvironment * gpu_env,
                            environment->GetGpuEnvironment());
    LITERT_RETURN_IF_ERROR(gpu_env->AddEnvironmentOptions(
        absl::MakeSpan(options, num_options)));
  }
#endif  // !defined(LITERT_DISABLE_GPU)
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGpuEnvironmentCreate(LiteRtEnvironment environment,
                                        int num_options,
                                        const LiteRtEnvOption* options) {
  LITERT_RETURN_IF_ERROR(
      environment->AddOptions(absl::MakeSpan(options, num_options)));
  LITERT_ASSIGN_OR_RETURN(
      auto gpu_env, litert::internal::GpuEnvironment::Create(environment));
  LITERT_RETURN_IF_ERROR(environment->SetGpuEnvironment(std::move(gpu_env)));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtEnvironmentSupportsClGlInterop(LiteRtEnvironment environment,
                                                  bool* is_supported) {
  LITERT_RETURN_IF_ERROR(environment != nullptr)
      << "Environment pointer is null.";
  *is_supported = environment->SupportsClGlInterop();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtEnvironmentSupportsAhwbClInterop(
    LiteRtEnvironment environment, bool* is_supported) {
  LITERT_RETURN_IF_ERROR(environment != nullptr)
      << "Environment pointer is null.";
  *is_supported = environment->SupportsAhwbClInterop();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtEnvironmentSupportsAhwbGlInterop(
    LiteRtEnvironment environment, bool* is_supported) {
  LITERT_RETURN_IF_ERROR(environment != nullptr)
      << "Environment pointer is null.";
  *is_supported = environment->SupportsAhwbGlInterop();
  return kLiteRtStatusOk;
}

void LiteRtEnvironmentHasGpuEnvironment(LiteRtEnvironment environment,
                             bool* has_gpu_environment) {
  if (environment == nullptr) {
    *has_gpu_environment = false;
    return;
  }
  *has_gpu_environment = environment->HasGpuEnvironment();
}

#ifdef __cplusplus
}  // extern "C"
#endif
