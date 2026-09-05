/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/runners/xnnpack/runner.h"

#include <cstddef>
#include <memory>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/utils/macros.h"
#include "pthreadpool.h"  // from @pthreadpool

namespace litert::tensor {

absl::Status XnnpackTraits::CreateRuntime(
    const NnpackRunner<XnnpackTraits>& runner, SubgraphType subgraph,
    size_t num_threads, std::unique_ptr<RuntimeType, RuntimeDeleter>& runtime) {
  pthreadpool_t threadpool = nullptr;
  if (const auto* xnn_runner = dynamic_cast<const XnnpackRunner*>(&runner)) {
    if (xnn_runner->threadpool() != nullptr) {
      threadpool = xnn_runner->threadpool();
    }
  }
  xnn_weights_cache_t weights_cache = nullptr;
  if (const auto* xnn_runner = dynamic_cast<const XnnpackRunner*>(&runner)) {
    weights_cache = xnn_runner->weights_cache();
  }
  xnn_runtime* raw_runtime = nullptr;
  LRT_TENSOR_RETURN_IF_ERROR(
      XnnStatusToAbsl(xnn_create_runtime_v3(subgraph, weights_cache, threadpool,
                                            /*flags=*/0, &raw_runtime),
                      "xnn_create_runtime_v3"));
  runtime.reset(raw_runtime);
  return absl::OkStatus();
}

}  // namespace litert::tensor
