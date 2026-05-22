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

#include "litert/runtime/accelerators/auto_registration.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerators/cpu_registry.h"
#include "litert/runtime/accelerators/gpu_registry.h"
#include "litert/runtime/accelerators/npu_registry.h"
#include "litert/runtime/accelerators/webnn_registry.h"

namespace litert {
namespace {

constexpr LiteRtHwAcceleratorSet kDefaultAutoRegisterAccelerators =
    kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu |
    kLiteRtHwAcceleratorNpu | litert::internal::kLiteRtHwAcceleratorWebNnAlias;

LiteRtHwAcceleratorSet GetAutoRegisterAccelerators(
    const LiteRtEnvironmentT& environment) {
  auto option =
      environment.GetOption(kLiteRtEnvOptionTagAutoRegisterAccelerators);
  if (!option.has_value()) {
    return kDefaultAutoRegisterAccelerators;
  }
  if (option->type != kLiteRtAnyTypeInt) {
    LITERT_LOG(LITERT_WARNING,
               "Auto-register accelerators option must be an integer bitmask. "
               "Using default accelerator auto-registration.");
    return kDefaultAutoRegisterAccelerators;
  }
  return static_cast<LiteRtHwAcceleratorSet>(option->int_value);
}

}  // namespace

Expected<void> TriggerAcceleratorAutomaticRegistration(
    LiteRtEnvironmentT& environment) {
  const LiteRtHwAcceleratorSet auto_register_accelerators =
      GetAutoRegisterAccelerators(environment);

  // Register the NPU accelerator.
  if (auto_register_accelerators & kLiteRtHwAcceleratorNpu) {
    litert::internal::RegisterNpuAccelerator(&environment);
  }

  // Register the WebNN accelerator.
  if (auto_register_accelerators &
      litert::internal::kLiteRtHwAcceleratorWebNnAlias) {
    litert::internal::RegisterWebNnAccelerator(&environment);
  }

  // Register the GPU accelerator.
  if (auto_register_accelerators & kLiteRtHwAcceleratorGpu) {
    litert::internal::RegisterGpuAccelerator(&environment);
  }

  // Register the CPU accelerator.
  if (auto_register_accelerators & kLiteRtHwAcceleratorCpu) {
    litert::internal::RegisterCpuAccelerator(&environment);
  }

  return {};
};

}  // namespace litert
