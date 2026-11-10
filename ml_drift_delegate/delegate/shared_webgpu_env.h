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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_WEBGPU_ENV_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_WEBGPU_ENV_H_

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift

namespace litert::ml_drift {

class GpuInferenceContextWebGpu;

// Process-wide state shared by all GpuBackendWebGpu instances, and the
// inference contexts created from them, which wrap the same
// ::ml_drift::webgpu::ExecutionEnvironment. For example, the prefill and
// decode delegates of an LLM each have their own backend but share one
// environment (and GPU memory), whereas independent engines, which may run on
// different threads, have their own environments.
//
// It tracks the last context which dispatched on each environment. A context
// uses it to invalidate the command buffers it prepared in advance when a
// different context dispatched on the same environment in between, since the
// GPU resource bindings would have changed. For example, during multiple
// conversations with an LLM, prefill inference starts after decode and would
// otherwise reuse the command buffers prepared by the previous prefill
// inference. The state used to be a process-wide static, which made
// independent engines running on different threads race on it and invalidate
// each other's prepared command buffers on every step.
//
// Thread-safe.
class SharedWebGpuEnv {
 public:
  // Returns the singleton instance.
  static SharedWebGpuEnv& Get();

  // Returns the last context which dispatched on `env`, or nullptr if there is
  // none. The returned pointer may only be compared, never dereferenced, since
  // the context may have been destroyed since.
  GpuInferenceContextWebGpu* GetLastContext(
      const ::ml_drift::webgpu::ExecutionEnvironment* env) const;

  // Records `ctx` as the last context which dispatched on `env`.
  void SetLastContext(const ::ml_drift::webgpu::ExecutionEnvironment* env,
                      GpuInferenceContextWebGpu* ctx);

  // Forgets the last context of `env` if it is `ctx`. Called when `ctx` is
  // destroyed so that a context allocated at the same address later is not
  // mistaken for it, and so that no state is kept for destroyed environments.
  void ClearLastContextIfMatched(
      const ::ml_drift::webgpu::ExecutionEnvironment* env,
      GpuInferenceContextWebGpu* ctx);

 private:
  friend class absl::NoDestructor<SharedWebGpuEnv>;
  SharedWebGpuEnv() = default;

  mutable absl::Mutex mutex_;
  // The last context which dispatched on each environment, keyed by the
  // environment. Entries are erased by ClearLastContextIfMatched().
  absl::flat_hash_map<const ::ml_drift::webgpu::ExecutionEnvironment*,
                      GpuInferenceContextWebGpu*>
      last_contexts_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_WEBGPU_ENV_H_
