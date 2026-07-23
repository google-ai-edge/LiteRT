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

#include "tflite/kernels/fuzzing/fuzzer_quota_allocator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace tflite {
namespace fuzzing {

FuzzerQuotaAllocator::FuzzerQuotaAllocator(size_t max_live_bytes)
    : allocator_{this, &FuzzerQuotaAllocator::AllocateCallback,
                 &FuzzerQuotaAllocator::ReallocateCallback,
                 &FuzzerQuotaAllocator::DeallocateCallback},
      max_live_bytes_(max_live_bytes) {}

struct FuzzerQuotaAllocator::AllocationHeader {
  void* base;
  size_t bytes;
};

void* FuzzerQuotaAllocator::AllocateCallback(void* data, size_t bytes,
                                             size_t alignment) {
  return static_cast<FuzzerQuotaAllocator*>(data)->Allocate(bytes, alignment);
}

void* FuzzerQuotaAllocator::ReallocateCallback(void* data, void* ptr,
                                               size_t old_bytes,
                                               size_t new_bytes,
                                               size_t alignment) {
  return static_cast<FuzzerQuotaAllocator*>(data)->Reallocate(
      ptr, old_bytes, new_bytes, alignment);
}

void FuzzerQuotaAllocator::DeallocateCallback(void* data, void* ptr,
                                              size_t bytes, size_t alignment) {
  (void)bytes;
  (void)alignment;
  static_cast<FuzzerQuotaAllocator*>(data)->Deallocate(ptr);
}

void* FuzzerQuotaAllocator::Allocate(size_t bytes, size_t alignment) {
  if (bytes == 0) {
    return nullptr;
  }
  if (alignment == 0) {
    alignment = alignof(std::max_align_t);
  }
  if ((alignment & (alignment - 1)) != 0) {
    return nullptr;
  }
  alignment = std::max(alignment, alignof(AllocationHeader));

  if (!Reserve(bytes)) {
    return nullptr;
  }

  constexpr size_t kHeaderBytes = sizeof(AllocationHeader);
  if (bytes > std::numeric_limits<size_t>::max() - alignment + 1 ||
      bytes + alignment - 1 >
          std::numeric_limits<size_t>::max() - kHeaderBytes) {
    Release(bytes);
    return nullptr;
  }
  const size_t total_bytes = bytes + alignment - 1 + kHeaderBytes;
  void* base = std::malloc(total_bytes);
  if (base == nullptr) {
    Release(bytes);
    return nullptr;
  }

  const uintptr_t unaligned =
      reinterpret_cast<uintptr_t>(base) + kHeaderBytes;
  const uintptr_t aligned = (unaligned + alignment - 1) & ~(alignment - 1);
  auto* header = reinterpret_cast<AllocationHeader*>(aligned - kHeaderBytes);
  header->base = base;
  header->bytes = bytes;
  return reinterpret_cast<void*>(aligned);
}

void* FuzzerQuotaAllocator::Reallocate(void* ptr, size_t old_bytes,
                                       size_t new_bytes, size_t alignment) {
  if (ptr == nullptr) {
    return Allocate(new_bytes, alignment);
  }
  void* new_ptr = Allocate(new_bytes, alignment);
  if (new_ptr == nullptr) {
    return nullptr;
  }
  std::memcpy(new_ptr, ptr, std::min(old_bytes, new_bytes));
  Deallocate(ptr);
  return new_ptr;
}

void FuzzerQuotaAllocator::Deallocate(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  auto* header = reinterpret_cast<AllocationHeader*>(
      reinterpret_cast<uintptr_t>(ptr) - sizeof(AllocationHeader));
  Release(header->bytes);
  std::free(header->base);
}

bool FuzzerQuotaAllocator::Reserve(size_t num_bytes) {
  if (num_bytes > max_live_bytes_ ||
      live_bytes_ > max_live_bytes_ - num_bytes) {
    return false;
  }
  live_bytes_ += num_bytes;
  return true;
}

void FuzzerQuotaAllocator::Release(size_t num_bytes) {
  if (num_bytes > live_bytes_) {
    live_bytes_ = 0;
  } else {
    live_bytes_ -= num_bytes;
  }
}

}  // namespace fuzzing
}  // namespace tflite
