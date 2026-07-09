/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_LITE_KERNELS_FUZZING_FUZZER_QUOTA_ALLOCATOR_H_
#define TENSORFLOW_LITE_KERNELS_FUZZING_FUZZER_QUOTA_ALLOCATOR_H_

#include <cstddef>

#include "tflite/core/c/common.h"

namespace tflite {
namespace fuzzing {

// Quota allocator for fuzzing. It implements TFLite's custom allocator
// callback table and tracks cumulative live bytes allocated through it.
class FuzzerQuotaAllocator final {
 public:
  explicit FuzzerQuotaAllocator(size_t max_live_bytes);

  FuzzerQuotaAllocator(const FuzzerQuotaAllocator&) = delete;
  FuzzerQuotaAllocator& operator=(const FuzzerQuotaAllocator&) = delete;

  TfLiteAllocator* allocator() { return &allocator_; }

  size_t live_bytes() const { return live_bytes_; }
  size_t max_live_bytes() const { return max_live_bytes_; }

 private:
  struct AllocationHeader;

  static void* AllocateCallback(void* data, size_t bytes, size_t alignment);
  static void* ReallocateCallback(void* data, void* ptr, size_t old_bytes,
                                  size_t new_bytes, size_t alignment);
  static void DeallocateCallback(void* data, void* ptr, size_t bytes,
                                 size_t alignment);

  void* Allocate(size_t bytes, size_t alignment);
  void* Reallocate(void* ptr, size_t old_bytes, size_t new_bytes,
                   size_t alignment);
  void Deallocate(void* ptr);

  bool Reserve(size_t num_bytes);
  void Release(size_t num_bytes);

  TfLiteAllocator allocator_;
  const size_t max_live_bytes_;
  size_t live_bytes_ = 0;
};

}  // namespace fuzzing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_FUZZING_FUZZER_QUOTA_ALLOCATOR_H_
