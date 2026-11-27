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

#include "tensor/buffer.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::tensor {

absl::StatusOr<std::shared_ptr<OwningCpuBuffer>> OwningCpuBuffer::Own(
    OwningCpuBuffer::CustomAllocPtr data, size_t bytes) {
  if (!IsAligned(data.get())) {
    return absl::InvalidArgumentError(
        "Data isn't aligned to kCpuBufferAlignment.");
  }
  return std::make_shared<OwningCpuBuffer>(kPass, std::move(data), bytes);
}

absl::StatusOr<std::shared_ptr<OwningCpuBuffer>> OwningCpuBuffer::Own(
    std::shared_ptr<std::byte> data, size_t bytes) {
  if (!IsAligned(data.get())) {
    return absl::InvalidArgumentError(
        "Data isn't aligned to kCpuBufferAlignment.");
  }
  return std::make_shared<OwningCpuBuffer>(kPass, std::move(data), bytes);
}

std::shared_ptr<OwningCpuBuffer> OwningCpuBuffer::Copy(const char* data,
                                                       size_t bytes) {
  CustomAllocPtr copied_data = AlignedAlloc(bytes);
  std::memcpy(copied_data.get(), data, bytes);
  return std::make_shared<OwningCpuBuffer>(kPass, std::move(copied_data),
                                           bytes);
}

OwningCpuBuffer::CustomAllocPtr OwningCpuBuffer::AlignedAlloc(size_t bytes) {
  static_assert(
      kCpuBufferAlignment >= alignof(ptrdiff_t) &&
          kCpuBufferAlignment % alignof(ptrdiff_t) == 0,
      "kCpuBufferAlignment must be a multiple of the alignment of the "
      "offset prefix type.");
  static_assert(kCpuBufferAlignment >= sizeof(ptrdiff_t),
                "kCpuBufferAlignment must fit the offset/size header.");
#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
  const size_t max_mmap_bytes =
      static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) -
      kCpuBufferAlignment;
  if (bytes >= kMmapAllocationThresholdBytes && bytes <= max_mmap_bytes) {
    const size_t total_bytes = bytes + kCpuBufferAlignment;
    void* raw = nullptr;
#if defined(_WIN32)
    raw = VirtualAlloc(nullptr, total_bytes, MEM_COMMIT | MEM_RESERVE,
                       PAGE_READWRITE);
#else
    void* mapped = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped != MAP_FAILED) {
      raw = mapped;
    }
#endif
    if (raw != nullptr) {
      // OS page mappings are aligned to at least 4 KiB (a multiple of
      // `kCpuBufferAlignment`), so advancing by `kCpuBufferAlignment` stays
      // aligned while reserving space for the negative mapped-size tag.
      std::byte* const aligned_data =
          static_cast<std::byte*>(raw) + kCpuBufferAlignment;
      reinterpret_cast<ptrdiff_t*>(aligned_data)[-1] =
          -static_cast<ptrdiff_t>(total_bytes);
      return CustomAllocPtr(aligned_data, &AlignedFree);
    }
  }
#endif
  std::byte* data =
      new std::byte[bytes + kCpuBufferAlignment + sizeof(ptrdiff_t)];
  // We must move `data` by at least the size of the offset prefix.
  const uintptr_t min_data_offset =
      reinterpret_cast<uintptr_t>(data) + sizeof(ptrdiff_t);
  const ptrdiff_t displacement =
      kCpuBufferAlignment - (min_data_offset % kCpuBufferAlignment);
  std::byte* const aligned_data = data + sizeof(ptrdiff_t) + displacement;
  reinterpret_cast<ptrdiff_t*>(aligned_data)[-1] = aligned_data - data;
  return CustomAllocPtr(aligned_data, &AlignedFree);
}

void OwningCpuBuffer::AlignedFree(std::byte* ptr) {
  if (ptr == nullptr) return;
  const ptrdiff_t tag = *(reinterpret_cast<ptrdiff_t*>(ptr) - 1);
  if (tag < 0) {
    std::byte* const base = ptr - kCpuBufferAlignment;
#if defined(_WIN32)
    VirtualFree(base, 0, MEM_RELEASE);
#elif defined(__linux__) || defined(__APPLE__)
    const size_t total_bytes = static_cast<size_t>(-tag);
    (void)munmap(base, total_bytes);
#else
    (void)base;
#endif
    return;
  }
  delete[] (ptr - tag);
}

}  // namespace litert::tensor
