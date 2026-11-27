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

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/ynnpack/graph.h"
#include "tensor/backends/ynnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {
namespace {

inline void CpuRelax() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

}  // namespace

YnnpackScheduler::YnnpackScheduler(int num_background_threads,
                                   std::chrono::nanoseconds spin_duration)
    : spin_duration_(spin_duration) {
  if (num_background_threads <= 0) return;
  workers_.reserve(static_cast<size_t>(num_background_threads));
  for (int i = 0; i < num_background_threads; ++i) {
    workers_.emplace_back([this]() { WorkerLoop(); });
  }
}

YnnpackScheduler::~YnnpackScheduler() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_.store(true, std::memory_order_release);
  }
  cv_.notify_all();
  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void YnnpackScheduler::WorkerLoop() {
  while (true) {
    Task task;
    bool got_task = false;
    if (pending_tasks_.load(std::memory_order_acquire) > 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!queue_.empty()) {
        task = queue_.front();
        queue_.pop_front();
        pending_tasks_.fetch_sub(1, std::memory_order_relaxed);
        got_task = true;
      }
    }
    if (got_task) {
      task.fn(task.context);
      continue;
    }
    if (stop_.load(std::memory_order_acquire)) {
      return;
    }
    if (spin_duration_.count() > 0) {
      const auto deadline = std::chrono::steady_clock::now() + spin_duration_;
      while (pending_tasks_.load(std::memory_order_acquire) == 0 &&
             !stop_.load(std::memory_order_acquire)) {
        for (int i = 0; i < 64; ++i) {
          if (pending_tasks_.load(std::memory_order_acquire) != 0 ||
              stop_.load(std::memory_order_acquire)) {
            break;
          }
          CpuRelax();
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          break;
        }
      }
      if (pending_tasks_.load(std::memory_order_acquire) != 0) {
        continue;
      }
      if (stop_.load(std::memory_order_acquire)) {
        return;
      }
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty() && !stop_.load(std::memory_order_acquire)) {
      sleeping_workers_.fetch_add(1, std::memory_order_acq_rel);
      cv_.wait(lock, [this]() {
        return !queue_.empty() || stop_.load(std::memory_order_acquire);
      });
      sleeping_workers_.fetch_sub(1, std::memory_order_acq_rel);
    }
    if (!queue_.empty()) {
      task = queue_.front();
      queue_.pop_front();
      pending_tasks_.fetch_sub(1, std::memory_order_relaxed);
      lock.unlock();
      task.fn(task.context);
    } else if (stop_.load(std::memory_order_acquire)) {
      return;
    }
  }
}

void YnnpackScheduler::Enqueue(void (*fn)(void*), void* context) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(Task{.fn = fn, .context = context});
    pending_tasks_.fetch_add(1, std::memory_order_release);
  }
  if (sleeping_workers_.load(std::memory_order_acquire) > 0) {
    cv_.notify_one();
  }
}

int YnnpackScheduler::NumThreads(void* self) {
  return reinterpret_cast<YnnpackScheduler*>(self)->num_background_threads();
}

void YnnpackScheduler::Schedule(void* self, void* context,
                                void (*task)(void* context)) {
  reinterpret_cast<YnnpackScheduler*>(self)->Enqueue(task, context);
}

const ynn_scheduler* YnnpackScheduler::Vtable() {
  static const ynn_scheduler kScheduler = {NumThreads, Schedule};
  return &kScheduler;
}

absl::Status YnnpackRunner::CreateRuntime() {
  // `NnpackRunner` only calls this once, and `SetNumThreads` / `SetThreadpool`
  // fail from then on, so any owned thread pool is created right before the
  // runtime that uses it, and lives as long as the runner.
  ynn_threadpool_t active_pool = external_threadpool_;
  if (active_pool == nullptr) {
    if (num_threads_ > 1 && scheduler_ == nullptr) {
      scheduler_ =
          std::make_unique<YnnpackScheduler>(static_cast<int>(num_threads_) - 1);
    }
    if (scheduler_ != nullptr && threadpool_ == nullptr) {
      ynn_threadpool_t raw_threadpool = nullptr;
      LRT_TENSOR_RETURN_IF_ERROR(YnnStatusToAbsl(
          ynn_create_threadpool(YnnpackScheduler::Vtable(), scheduler_.get(),
                                /*flags=*/0, &raw_threadpool),
          "ynn_create_threadpool"));
      threadpool_.reset(raw_threadpool);
    }
    active_pool = threadpool_.get();
  }

  ynn_subgraph_t sg = static_cast<YnnpackGraph&>(*graph_).GetSubgraph();
  LRT_TENSOR_RETURN_IF_ERROR(
      YnnStatusToAbsl(ynn_optimize_subgraph(sg, active_pool, /*flags=*/0),
                      "ynn_optimize_subgraph"));

  ynn_runtime* raw_runtime = nullptr;
  LRT_TENSOR_RETURN_IF_ERROR(YnnStatusToAbsl(
      ynn_create_runtime(sg, active_pool, /*flags=*/0, &raw_runtime),
      "ynn_create_runtime"));
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
