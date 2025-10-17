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

#include "litert/core/environment.h"

#include <memory>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/gpu_environment.h"

litert::Expected<LiteRtEnvironmentT::Ptr> LiteRtEnvironmentT::CreateWithOptions(
    absl::Span<const LiteRtEnvOption> options) {
  LITERT_LOG(LITERT_INFO, "Creating LiteRT environment with options");
  auto env = std::make_unique<LiteRtEnvironmentT>();
  for (const auto& opt : options) {
    env->options_.SetOption(opt);
  }

  return env;
}

litert::Expected<void> LiteRtEnvironmentT::AddOptions(
    absl::Span<const LiteRtEnvOption> options, bool overwrite) {
  LITERT_LOG(LITERT_INFO, "Adding options to the existing LiteRT environment");
  for (const auto& opt : options) {
    LITERT_RETURN_IF_ERROR(options_.SetOption(opt, overwrite));
  }
  return {};
}

// C API to workaround Windows build issue.
// This function is only used in tensor_buffer.cc.
extern "C" litert::internal::GpuEnvironment* LiteRtGetGpuEnvironment(
    LiteRtEnvironment env) {
  if (env == nullptr) {
    return nullptr;
  }
  LITERT_ASSIGN_OR_RETURN(auto gpu_env, env->GetGpuEnvironment(), nullptr);

  return gpu_env;
}
