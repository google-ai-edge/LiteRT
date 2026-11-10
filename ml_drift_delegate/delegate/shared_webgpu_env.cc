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

#include "ml_drift_delegate/delegate/shared_webgpu_env.h"

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift

namespace litert::ml_drift {

SharedWebGpuEnv& SharedWebGpuEnv::Get() {
  static absl::NoDestructor<SharedWebGpuEnv> instance;
  return *instance;
}

GpuInferenceContextWebGpu* SharedWebGpuEnv::GetLastContext(
    const ::ml_drift::webgpu::ExecutionEnvironment* env) const {
  absl::MutexLock lock(mutex_);
  auto it = last_contexts_.find(env);
  return it == last_contexts_.end() ? nullptr : it->second;
}

void SharedWebGpuEnv::SetLastContext(
    const ::ml_drift::webgpu::ExecutionEnvironment* env,
    GpuInferenceContextWebGpu* ctx) {
  absl::MutexLock lock(mutex_);
  last_contexts_[env] = ctx;
}

void SharedWebGpuEnv::ClearLastContextIfMatched(
    const ::ml_drift::webgpu::ExecutionEnvironment* env,
    GpuInferenceContextWebGpu* ctx) {
  absl::MutexLock lock(mutex_);
  auto it = last_contexts_.find(env);
  if (it != last_contexts_.end() && it->second == ctx) {
    last_contexts_.erase(it);
  }
}

}  // namespace litert::ml_drift
