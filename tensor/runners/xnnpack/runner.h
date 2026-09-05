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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_XNNPACK_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_XNNPACK_RUNNER_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/runners/nnpack_common/runner.h"
#include "tensor/tensor.h"
#include "pthreadpool.h"  // from @pthreadpool

namespace litert::tensor {

// XnnpackRunner is a class that runs an XNNPACK graph.
class XnnpackRunner : public NnpackRunner<XnnpackTraits> {
 public:
  static absl::StatusOr<XnnpackRunner> Create(
      std::vector<TensorHandle> outputs) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto graph,
                                BuildXnnpackGraph(std::move(outputs)));
    return XnnpackRunner(std::move(graph));
  }

  explicit XnnpackRunner(std::unique_ptr<XnnpackGraph> graph)
      : NnpackRunner<XnnpackTraits>(std::move(graph)) {}

  ~XnnpackRunner() override {
    if (threadpool_ != nullptr) {
      pthreadpool_destroy(threadpool_);
    }
  }

  XnnpackRunner(XnnpackRunner&& other) noexcept
      : NnpackRunner<XnnpackTraits>(std::move(other)),
        weights_cache_(other.weights_cache_),
        threadpool_(std::exchange(other.threadpool_, nullptr)) {}

  XnnpackRunner& operator=(XnnpackRunner&& other) noexcept {
    if (this != &other) {
      if (threadpool_ != nullptr) {
        pthreadpool_destroy(threadpool_);
      }
      NnpackRunner<XnnpackTraits>::operator=(std::move(other));
      weights_cache_ = other.weights_cache_;
      threadpool_ = std::exchange(other.threadpool_, nullptr);
    }
    return *this;
  }

  void SetNumThreads(size_t num_threads) {
    NnpackRunner<XnnpackTraits>::SetNumThreads(num_threads);
    if (threadpool_ != nullptr) {
      pthreadpool_destroy(threadpool_);
      threadpool_ = nullptr;
    }
    if (num_threads > 1) {
      threadpool_ = pthreadpool_create(num_threads);
    }
  }

  void SetWeightsCache(xnn_weights_cache_t weights_cache) {
    weights_cache_ = weights_cache;
  }

  xnn_weights_cache_t weights_cache() const { return weights_cache_; }
  pthreadpool_t threadpool() const { return threadpool_; }

 private:
  xnn_weights_cache_t weights_cache_ = nullptr;
  pthreadpool_t threadpool_ = nullptr;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_XNNPACK_RUNNER_H_
