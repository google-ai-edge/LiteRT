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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/backends/xnnpack/graph.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/tensor.h"
#include "pthreadpool.h"  // from @pthreadpool

namespace litert::tensor {

// XnnpackRunner is a class that runs an XNNPACK graph.
class XnnpackRunner : public NnpackRunner {
 public:
  struct RuntimeDeleter {
    void operator()(::xnn_runtime* ptr) const {
      if (ptr) {
        xnn_delete_runtime(ptr);
      }
    }
  };
  using RuntimePtr = std::unique_ptr<::xnn_runtime, RuntimeDeleter>;

  static absl::StatusOr<XnnpackRunner> Create(
      std::vector<TensorHandle> outputs) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto graph,
                                BuildXnnpackGraph(std::move(outputs)));
    return XnnpackRunner(std::move(graph));
  }

  explicit XnnpackRunner(std::unique_ptr<XnnpackGraph> graph)
      : NnpackRunner(std::move(graph)) {}

  ~XnnpackRunner() override {
    if (threadpool_ != nullptr) {
      pthreadpool_destroy(threadpool_);
    }
  }

  XnnpackRunner(XnnpackRunner&& other) noexcept
      : NnpackRunner(std::move(other)),
        runtime_(std::move(other.runtime_)),
        weights_cache_(other.weights_cache_),
        threadpool_(std::exchange(other.threadpool_, nullptr)) {}

  XnnpackRunner& operator=(XnnpackRunner&& other) noexcept {
    if (this != &other) {
      if (threadpool_ != nullptr) {
        pthreadpool_destroy(threadpool_);
      }
      NnpackRunner::operator=(std::move(other));
      runtime_ = std::move(other.runtime_);
      weights_cache_ = other.weights_cache_;
      threadpool_ = std::exchange(other.threadpool_, nullptr);
    }
    return *this;
  }

  void SetNumThreads(size_t num_threads) override {
    NnpackRunner::SetNumThreads(num_threads);
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
  xnn_runtime_t runtime() const { return runtime_.get(); }

 protected:
  uint32_t FlagExternalInput() const override {
    return XNN_VALUE_FLAG_EXTERNAL_INPUT;
  }
  uint32_t FlagExternalOutput() const override {
    return XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }

  absl::Status CreateRuntime(size_t num_threads) override;
  absl::Status SetExternalValueShape(uint32_t id,
                                     absl::Span<const size_t> dims) override;
  absl::Status ReshapeRuntime() override;
  absl::Status GetExternalValueShape(uint32_t id,
                                     std::vector<size_t>& dims) override;
  absl::Status SetupExternalValues(
      absl::Span<NnpackValue> values,
      const absl::flat_hash_map<uint32_t, std::shared_ptr<Buffer>>&
          external_buffers,
      std::vector<LockedBufferSpan<const std::byte>>& locks) override;
  absl::Status InvokeRuntime() override;

 private:
  RuntimePtr runtime_ = nullptr;
  xnn_weights_cache_t weights_cache_ = nullptr;
  pthreadpool_t threadpool_ = nullptr;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_XNNPACK_RUNNER_H_
