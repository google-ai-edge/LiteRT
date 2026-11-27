/* Copyright 2026 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_YNNPACK_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_YNNPACK_RUNNER_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/ynnpack/conversion.h"
#include "tensor/backends/ynnpack/graph.h"
#include "tensor/buffer.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

// Adapts a spin-before-sleep FIFO worker pool to the `ynn_scheduler` interface
// YNNPACK expects.
//
// YNNPACK doesn't own threads: it asks a scheduler to run tasks and joins the
// work itself from the invoking thread. As a result the scheduler only needs to
// provide `num_threads - 1` background threads. Workers briefly spin-poll
// (`spin_duration`, 1 ms by default) before sleeping on a condition variable so
// consecutive short parallel loops during token decoding avoid futex wakeups.
class YnnpackScheduler {
 public:
  explicit YnnpackScheduler(
      int num_background_threads,
      std::chrono::nanoseconds spin_duration = std::chrono::milliseconds(1));

  ~YnnpackScheduler();

  YnnpackScheduler(const YnnpackScheduler&) = delete;
  YnnpackScheduler& operator=(const YnnpackScheduler&) = delete;

  int num_background_threads() const {
    return static_cast<int>(workers_.size());
  }

  // Returns the singleton vtable YNNPACK dispatches through. The scheduler
  // instance itself is passed as the opaque context.
  static const ynn_scheduler* Vtable();

 private:
  struct Task {
    void (*fn)(void*) = nullptr;
    void* context = nullptr;
  };

  void WorkerLoop();
  void Enqueue(void (*fn)(void*), void* context);

  static int NumThreads(void* self);
  static void Schedule(void* self, void* context, void (*task)(void* context));

  std::chrono::nanoseconds spin_duration_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Task> queue_;
  std::atomic<size_t> pending_tasks_{0};
  std::atomic<int> sleeping_workers_{0};
  std::atomic<bool> stop_{false};
  std::vector<std::thread> workers_;
};

// YnnpackRunner is a class that runs a YNNPACK graph.
class YnnpackRunner : public NnpackRunner {
 public:
  struct RuntimeDeleter {
    void operator()(::ynn_runtime* ptr) const {
      if (ptr) {
        ynn_delete_runtime(ptr);
      }
    }
  };
  using RuntimePtr = std::unique_ptr<::ynn_runtime, RuntimeDeleter>;

  struct ThreadpoolDeleter {
    void operator()(::ynn_threadpool* ptr) const {
      if (ptr) {
        ynn_delete_threadpool(ptr);
      }
    }
  };
  using ThreadpoolPtr = std::unique_ptr<::ynn_threadpool, ThreadpoolDeleter>;

  static absl::StatusOr<YnnpackRunner> Create(
      std::vector<TensorHandle> outputs) {
    LRT_TENSOR_ASSIGN_OR_RETURN(std::unique_ptr<YnnpackGraph> graph,
                                BuildYnnpackGraph(std::move(outputs)));
    return YnnpackRunner(std::move(graph));
  }

  explicit YnnpackRunner(std::unique_ptr<YnnpackGraph> graph)
      : NnpackRunner(std::move(graph)) {}

  ~YnnpackRunner() override {
    // Destroy the runtime before releasing any owned threadpool/scheduler or
    // the base class `graph_` (YNNPACK runtimes keep a raw pointer to the
    // subgraph).
    runtime_.reset();
    threadpool_.reset();
    scheduler_.reset();
  }

  YnnpackRunner(YnnpackRunner&& other) noexcept = default;
  YnnpackRunner& operator=(YnnpackRunner&& other) noexcept {
    if (this != &other) {
      runtime_.reset();
      threadpool_.reset();
      scheduler_.reset();
      NnpackRunner::operator=(std::move(other));
      scheduler_ = std::move(other.scheduler_);
      threadpool_ = std::move(other.threadpool_);
      external_threadpool_ = std::exchange(other.external_threadpool_, nullptr);
      runtime_ = std::move(other.runtime_);
    }
    return *this;
  }

  // Binds an externally owned `ynn_threadpool_t` (must outlive this runner).
  // Must be called before `PrepareRuntime()` / `Run()`.
  absl::Status SetThreadpool(ynn_threadpool_t threadpool) {
    if (runtime_prepared_) {
      return absl::FailedPreconditionError(
          "SetThreadpool must be called before the runtime is prepared.");
    }
    external_threadpool_ = threadpool;
    return absl::OkStatus();
  }

  ynn_runtime_t runtime() const { return runtime_.get(); }
  ynn_threadpool_t threadpool() const {
    return external_threadpool_ != nullptr ? external_threadpool_
                                           : threadpool_.get();
  }

 protected:
  uint32_t FlagExternalInput() const override {
    return YNN_VALUE_FLAG_EXTERNAL_INPUT;
  }
  uint32_t FlagExternalOutput() const override {
    return YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }

  absl::Status CreateRuntime() override;
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
  std::unique_ptr<YnnpackScheduler> scheduler_;
  ThreadpoolPtr threadpool_ = nullptr;
  ynn_threadpool_t external_threadpool_ = nullptr;
  RuntimePtr runtime_ = nullptr;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_YNNPACK_RUNNER_H_
