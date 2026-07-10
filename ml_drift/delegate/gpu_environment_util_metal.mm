// Copyright 2026 Google LLC.
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

#include "third_party/odml/litert/ml_drift/delegate/gpu_environment_util.h"

#include <variant>

#import <Metal/Metal.h>

#include "absl/status/status.h"  // from @com_google_absl
#if LITERT_HAS_METAL_SUPPORT
#include "ml_drift/metal/environment.h"  // from @ml_drift
#endif  // LITERT_HAS_METAL_SUPPORT
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_any.h"
#include "litert/core/environment.h"
#include "litert/runtime/gpu_environment.h"

namespace litert {
namespace ml_drift {

#if LITERT_HAS_METAL_SUPPORT
absl::Status UpdateGpuEnvironmentMetal(LiteRtEnvironment environment) {
  const auto& env_options = environment->GetOptions();

  auto metal_device_res = env_options.GetOption(kLiteRtEnvOptionTagMetalDevice);
  if (!metal_device_res.HasValue()) return absl::OkStatus();

  id<MTLDevice> metal_device =
      (__bridge id<MTLDevice>)(std::get<void*>(::litert::ToStdAny(metal_device_res.Value())));

  if (metal_device == nullptr) return absl::OkStatus();

  ::ml_drift::metal::Environment metal_env(metal_device);

  auto gpu_env_res = environment->GetGpuEnvironment();
  if (gpu_env_res.HasValue()) {
    gpu_env_res.Value()->SetFP16Supported(metal_env.GetInfo().SupportsFP16());
  }

  return absl::OkStatus();
}
#endif  // LITERT_HAS_METAL_SUPPORT

}  // namespace ml_drift
}  // namespace litert
