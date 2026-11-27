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

#include "tensor/runners/ynnpack/runner.h"

#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/ynnpack/graph.h"
#include "tensor/backends/ynnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

absl::Status RunWithLargeStack(absl::FunctionRef<absl::Status()> fn,
                               size_t stack_bytes) {
#if defined(__unix__) || defined(__APPLE__)
  constexpr size_t kDefaultCompilerStackBytes =
      sizeof(void*) >= 8 ? (64ULL * 1024 * 1024) : (16ULL * 1024 * 1024);
  const size_t target_stack_bytes =
      stack_bytes > 0 ? stack_bytes : kDefaultCompilerStackBytes;
  pthread_attr_t attr;
  if (pthread_attr_init(&attr) != 0) {
    return fn();
  }
  if (pthread_attr_setstacksize(&attr, target_stack_bytes) != 0) {
    pthread_attr_destroy(&attr);
    return fn();
  }
  struct ThreadContext {
    absl::FunctionRef<absl::Status()> fn;
    absl::Status status;
  } ctx{fn, absl::OkStatus()};
  pthread_t thread;
  const int rc = pthread_create(
      &thread, &attr,
      +[](void* arg) -> void* {
        auto* c = static_cast<ThreadContext*>(arg);
        c->status = c->fn();
        return nullptr;
      },
      &ctx);
  pthread_attr_destroy(&attr);
  if (rc != 0) {
    return fn();
  }
  pthread_join(thread, nullptr);
  return ctx.status;
#else
  (void)stack_bytes;
  return fn();
#endif
}

YnnpackScheduler::YnnpackScheduler(int num_background_threads,
                                   absl::Duration spin_duration)
    : spin_duration_(spin_duration) {
  threads_.reserve(std::max(0, num_background_threads));
  for (int i = 0; i < num_background_threads; ++i) {
    threads_.emplace_back([this] { RunThread(); });
  }
}

YnnpackScheduler::~YnnpackScheduler() {
  {
    absl::MutexLock lock(mutex_);
    stop_ = true;
  }
  stopping_.store(true, std::memory_order_relaxed);
  task_available_.SignalAll();
  for (std::thread& thread : threads_) {
    thread.join();
  }
}

void YnnpackScheduler::SpinWhileIdle() const {
  if (spin_duration_ <= absl::ZeroDuration()) return;
  const absl::Time deadline = absl::Now() + spin_duration_;
  for (int i = 1; num_tasks_.load(std::memory_order_acquire) == 0 &&
                  !stopping_.load(std::memory_order_relaxed);
       ++i) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    asm volatile("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#endif
    // Reading the clock costs more than a pause.
    if (i % 64 == 0 && absl::Now() >= deadline) return;
  }
}

void YnnpackScheduler::RunThread() {
  while (true) {
    SpinWhileIdle();
    Task task = {};
    {
      absl::MutexLock lock(mutex_);
      while (tasks_.empty() && !stop_) {
        ++sleeping_threads_;
        task_available_.Wait(&mutex_);
        --sleeping_threads_;
      }
      if (tasks_.empty()) return;
      task = tasks_.front();
      tasks_.pop_front();
      num_tasks_.fetch_sub(1, std::memory_order_relaxed);
    }
    task.fn(task.context);
  }
}

int YnnpackScheduler::NumThreads(void* self) {
  return static_cast<int>(
      static_cast<YnnpackScheduler*>(self)->threads_.size());
}

void YnnpackScheduler::Schedule(void* self, void* context,
                                void (*task)(void* context)) {
  YnnpackScheduler* scheduler = static_cast<YnnpackScheduler*>(self);
  bool wake = false;
  {
    absl::MutexLock lock(scheduler->mutex_);
    scheduler->tasks_.push_back({task, context});
    scheduler->num_tasks_.fetch_add(1, std::memory_order_release);
    wake = scheduler->sleeping_threads_ > 0;
  }
  if (wake) {
    scheduler->task_available_.Signal();
  }
}

const ynn_scheduler* YnnpackScheduler::Vtable() {
  static const ynn_scheduler kScheduler = {NumThreads, Schedule};
  return &kScheduler;
}

absl::Status YnnpackRunner::CreateRuntime() {
  // `NnpackRunner` only calls this once, and `SetNumThreads` / `SetThreadpool`
  // fail from then on, so any runner-owned thread pool is created here, right
  // before the runtime that uses it, and lives as long as the runner.
  if (external_threadpool_ == nullptr && num_threads_ > 1 &&
      scheduler_ == nullptr) {
    scheduler_ =
        std::make_unique<YnnpackScheduler>(static_cast<int>(num_threads_) - 1);
  }
  if (external_threadpool_ == nullptr && scheduler_ != nullptr &&
      threadpool_ == nullptr) {
    ynn_threadpool_t raw_threadpool = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(YnnStatusToAbsl(
        ynn_create_threadpool(YnnpackScheduler::Vtable(), scheduler_.get(),
                              /*flags=*/0, &raw_threadpool),
        "ynn_create_threadpool"));
    threadpool_.reset(raw_threadpool);
  }

  ynn_threadpool_t active_pool = threadpool();
  ynn_subgraph_t sg = static_cast<YnnpackGraph&>(*graph_).GetSubgraph();
  ynn_runtime* raw_runtime = nullptr;
  LRT_TENSOR_RETURN_IF_ERROR(RunWithLargeStack([&]() -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(
        YnnStatusToAbsl(ynn_optimize_subgraph(sg, active_pool, /*flags=*/0),
                        "ynn_optimize_subgraph"));
    return YnnStatusToAbsl(
        ynn_create_runtime(sg, active_pool, /*flags=*/0, &raw_runtime),
        "ynn_create_runtime");
  }));
  runtime_.reset(raw_runtime);
  return absl::OkStatus();
}

absl::Status YnnpackRunner::SetExternalValueShape(
    uint32_t id, absl::Span<const size_t> dims) {
  return YnnStatusToAbsl(
      ynn_set_external_value_shape(runtime_.get(), id, dims.size(),
                                   dims.empty() ? nullptr : dims.data()),
      "ynn_set_external_value_shape");
}

absl::Status YnnpackRunner::ReshapeRuntime() {
  return YnnStatusToAbsl(ynn_reshape_runtime(runtime_.get()),
                         "ynn_reshape_runtime");
}

absl::Status YnnpackRunner::GetExternalValueShape(uint32_t id,
                                                  std::vector<size_t>& dims) {
  // `rank` is an in/out parameter: on input it is the capacity of `shape_arr`.
  size_t rank = YNN_MAX_TENSOR_RANK;
  std::array<size_t, YNN_MAX_TENSOR_RANK> shape_arr{};
  LRT_TENSOR_RETURN_IF_ERROR(YnnStatusToAbsl(
      ynn_get_external_value_shape(runtime_.get(), id, &rank, shape_arr.data()),
      "ynn_get_external_value_shape"));
  dims.assign(shape_arr.begin(), shape_arr.begin() + rank);
  return absl::OkStatus();
}

absl::Status YnnpackRunner::SetupExternalValues(
    absl::Span<NnpackValue> values,
    const absl::flat_hash_map<uint32_t, std::shared_ptr<Buffer>>&
        external_buffers,
    std::vector<LockedBufferSpan<const std::byte>>& locks) {
  locks.reserve(values.size());
  for (NnpackValue& value : values) {
    if (value.flags == 0) {
      continue;
    }
    const auto it = external_buffers.find(value.id);
    if (it == external_buffers.end() || it->second == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrFormat("External value %u missing host buffer", value.id));
    }
    LockedBufferSpan<const std::byte> lock = it->second->Lock();
    if (lock.data() == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrFormat("External value %u could not be locked", value.id));
    }
    LRT_TENSOR_RETURN_IF_ERROR(YnnStatusToAbsl(
        ynn_set_external_value_data(runtime_.get(), value.id,
                                    const_cast<std::byte*>(lock.data())),
        "ynn_set_external_value_data"));
    locks.push_back(std::move(lock));
  }
  return absl::OkStatus();
}

absl::Status YnnpackRunner::InvokeRuntime() {
  return YnnStatusToAbsl(ynn_invoke_runtime(runtime_.get()),
                         "ynn_invoke_runtime");
}

}  // namespace litert::tensor
