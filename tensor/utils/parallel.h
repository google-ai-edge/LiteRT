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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_PARALLEL_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_PARALLEL_H_

#include <cstddef>
#include <type_traits>

namespace litert::tensor {
namespace internal {

void ParallelForImpl(size_t count, size_t min_grain, size_t max_threads,
                     void* ctx,
                     void (*fn)(void* ctx, size_t begin, size_t end));

}  // namespace internal

// Executes `fn(size_t begin, size_t end)` over `[0, count)` in tiles of at
// least `min_grain` elements, using at most `max_threads` concurrent worker
// tasks drawn from process-wide shared 8/16/24-thread pools (capped at 24 and
// the online CPU count). Falls back to a single synchronous call `fn(0, count)`
// when `count <= min_grain`, `max_threads <= 1`, only one worker thread is
// available, or on nested `ParallelFor` invocations.
template <typename Fn>
void ParallelFor(size_t count, size_t min_grain, Fn&& fn,
                 size_t max_threads = 16) {
  if (count == 0) return;
  if (count <= min_grain || max_threads <= 1) {
    fn(size_t{0}, count);
    return;
  }
  using FnType = std::remove_reference_t<Fn>;
  internal::ParallelForImpl(count, min_grain, max_threads,
                            const_cast<void*>(static_cast<const void*>(&fn)),
                            [](void* ctx, size_t begin, size_t end) {
                              (*static_cast<FnType*>(ctx))(begin, end);
                            });
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_PARALLEL_H_
