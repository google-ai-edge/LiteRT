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

#include "tensor/utils/parallel.h"

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "pthreadpool.h"  // from @pthreadpool

namespace litert::tensor::internal {
namespace {

pthreadpool_t CreateCappedThreadpool(size_t max_threads) {
  size_t threads = max_threads;
#if defined(__unix__) || defined(__APPLE__)
  const int64_t online = sysconf(_SC_NPROCESSORS_ONLN);
  if (online > 0) {
    threads = std::min(threads, static_cast<size_t>(online));
  }
#endif
  return pthreadpool_create(std::max<size_t>(1, threads));
}

pthreadpool_t GetSharedCappedThreadpool(size_t max_threads) {
  if (max_threads <= 8) {
    static pthreadpool_t pool8 = CreateCappedThreadpool(8);
    return pool8;
  }
  if (max_threads <= 16) {
    static pthreadpool_t pool16 = CreateCappedThreadpool(16);
    return pool16;
  }
  static pthreadpool_t pool24 = CreateCappedThreadpool(24);
  return pool24;
}

thread_local bool in_parallel_for_worker = false;

// Restores `in_parallel_for_worker` on scope exit when a worker task returns.
class ScopedWorkerGuard {
 public:
  ScopedWorkerGuard() : prev_(in_parallel_for_worker) {
    in_parallel_for_worker = true;
  }
  ScopedWorkerGuard(const ScopedWorkerGuard&) = delete;
  ScopedWorkerGuard& operator=(const ScopedWorkerGuard&) = delete;
  ~ScopedWorkerGuard() { in_parallel_for_worker = prev_; }

 private:
  bool prev_;
};

}  // namespace

void ParallelForImpl(size_t count, size_t min_grain, size_t max_threads,
                     void* ctx,
                     void (*fn)(void* ctx, size_t begin, size_t end)) {
  if (count == 0) return;
  if (in_parallel_for_worker || count <= min_grain || max_threads <= 1) {
    fn(ctx, 0, count);
    return;
  }
  pthreadpool_t pool = GetSharedCappedThreadpool(max_threads);
  const size_t pool_threads =
      pool != nullptr ? pthreadpool_get_threads_count(pool) : 1;
  const size_t active_threads = std::min(pool_threads, max_threads);
  if (active_threads <= 1) {
    fn(ctx, 0, count);
    return;
  }
  // When `max_threads < pool_threads` (e.g. `max_threads = 2` on the shared
  // 8-thread pool), cap total tiles at `active_threads` so at most
  // `max_threads` workers can execute concurrently; otherwise use 4x dynamic
  // tiling bounded by `pool_threads <= max_threads`.
  const size_t max_tasks =
      max_threads < pool_threads ? active_threads : active_threads * 4;
  const size_t safe_min_grain = std::max<size_t>(1, min_grain);
  const size_t effective_grain =
      std::max(safe_min_grain, 1 + (count - 1) / max_tasks);
  if (count <= effective_grain) {
    fn(ctx, 0, count);
    return;
  }
  pthreadpool_parallelize_1d_tile_1d(
      pool,
      [&](size_t offset, size_t tile) {
        const ScopedWorkerGuard guard;
        fn(ctx, offset, offset + tile);
      },
      count, effective_grain, /*flags=*/0);
}

}  // namespace litert::tensor::internal
