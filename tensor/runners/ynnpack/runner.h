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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/ynnpack/conversion.h"
#include "tensor/backends/ynnpack/graph.h"
#include "tensor/buffer.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

// Runs `fn` on a worker thread with a custom stack size (`stack_bytes == 0`
// selects 64 MB on 64-bit targets and 16 MB on 32-bit targets) so recursive
// Slinky AST simplification inside `ynn_create_runtime` cannot overflow
// constrained caller stacks.
absl::Status RunWithLargeStack(absl::FunctionRef<absl::Status()> fn,
                               size_t stack_bytes = 0);

// Adapts a thread pool to the `ynn_scheduler` interface YNNPACK expects.
//
// YNNPACK doesn't own threads: it asks a scheduler to run tasks and joins the
// work itself from the invoking thread. As a result the scheduler only needs to
// provide `num_threads - 1` background threads.
//
// YNNPACK schedules one task per idle thread whenever it starts a parallel
// loop, and small inferences such as LLM decoding run many short loops. Waking
// a sleeping thread can take longer than such a loop, so idle threads poll for
// new tasks for `spin_duration` before they sleep.
class YnnpackScheduler {
 public:
  static constexpr absl::Duration kDefaultSpinDuration = absl::Milliseconds(1);

  explicit YnnpackScheduler(
      int num_background_threads,
      absl::Duration spin_duration = kDefaultSpinDuration);

  // Runs the remaining tasks before joining the threads.
  ~YnnpackScheduler();

  YnnpackScheduler(const YnnpackScheduler&) = delete;
  YnnpackScheduler& operator=(const YnnpackScheduler&) = delete;

  // Returns the singleton vtable YNNPACK dispatches through. The scheduler
  // instance itself is passed as the opaque context.
  static const ynn_scheduler* Vtable();

 private:
  struct Task {
    void (*fn)(void* context);
    void* context;
  };

  static int NumThreads(void* self);
  static void Schedule(void* self, void* context, void (*task)(void* context));

  void RunThread();
  // Returns when a task may be available, the scheduler is stopping, or after
  // `spin_duration_`.
  void SpinWhileIdle() const;

  const absl::Duration spin_duration_;
  absl::Mutex mutex_;
  absl::CondVar task_available_;
  std::deque<Task> tasks_ ABSL_GUARDED_BY(mutex_);
  int sleeping_threads_ ABSL_GUARDED_BY(mutex_) = 0;
  bool stop_ ABSL_GUARDED_BY(mutex_) = false;
  // Mirror `tasks_.size()` and `stop_` so spinning threads can poll them
  // without the lock.
  std::atomic<size_t> num_tasks_{0};
  std::atomic<bool> stopping_{false};
  std::vector<std::thread> threads_;
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
      std::vector<TensorHandle> outputs,
      const YnnpackGraphOptions& options = {}) {
    LRT_TENSOR_ASSIGN_OR_RETURN(std::unique_ptr<YnnpackGraph> graph,
                                BuildYnnpackGraph(std::move(outputs), options));
    return YnnpackRunner(std::move(graph));
  }

  explicit YnnpackRunner(std::unique_ptr<YnnpackGraph> graph)
      : NnpackRunner(std::move(graph)) {}

  ~YnnpackRunner() override = default;

  YnnpackRunner(YnnpackRunner&& other) noexcept = default;
  YnnpackRunner& operator=(YnnpackRunner&& other) noexcept = default;

  ynn_runtime_t runtime() const { return runtime_.get(); }
  ynn_threadpool_t threadpool() const {
    return external_threadpool_ != nullptr ? external_threadpool_
                                           : threadpool_.get();
  }

  absl::Status SetNumThreads(size_t num_threads) override {
    LRT_TENSOR_RETURN_IF_ERROR(NnpackRunner::SetNumThreads(num_threads));
    external_threadpool_ = nullptr;
    return absl::OkStatus();
  }

  // Uses a caller-owned `ynn_threadpool_t` instead of creating a private
  // scheduler and threadpool per runner. Must be called before the runtime is
  // prepared, and `threadpool` must outlive this `YnnpackRunner`.
  absl::Status SetThreadpool(ynn_threadpool_t threadpool) {
    if (runtime_prepared_) {
      return absl::FailedPreconditionError(
          "SetThreadpool must be called before the runtime is prepared.");
    }
    external_threadpool_ = threadpool;
    threadpool_.reset();
    scheduler_.reset();
    return absl::OkStatus();
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
