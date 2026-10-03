// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/mediatek/dispatch/file_backed_pages.h"

#include <cstddef>
#include <cstdint>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace litert::mediatek {
namespace {

bool IsPowerOfTwo(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

// Returns the page size, or 0 if it is unavailable on this platform.
size_t SystemPageSize() {
#if defined(_WIN32)
  return 0;
#else
  const int64_t page_size = sysconf(_SC_PAGESIZE);  // NOLINT(runtime/int)
  return page_size > 0 ? static_cast<size_t>(page_size) : 0;
#endif
}

}  // namespace

PageRange InnerPageRange(const void* addr, size_t size, size_t page_size) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(addr);
  const uintptr_t end = begin + size;
  if (!IsPowerOfTwo(page_size) || end < begin) {
    return {};
  }
  const uintptr_t mask = page_size - 1;
  if (begin > UINTPTR_MAX - mask) {
    return {};
  }
  const PageRange range = {(begin + mask) & ~mask, end & ~mask};
  return range.empty() ? PageRange{} : range;
}

PageRange OuterPageRange(const void* addr, size_t size, size_t page_size) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(addr);
  const uintptr_t end = begin + size;
  if (!IsPowerOfTwo(page_size) || size == 0 || end < begin) {
    return {};
  }
  const uintptr_t mask = page_size - 1;
  if (end > UINTPTR_MAX - mask) {
    return {};
  }
  return {begin & ~mask, (end + mask) & ~mask};
}

void AdviseNoHugePages(int fd, const void* addr, size_t size) {
#if defined(MADV_NOHUGEPAGE)
  if (fd < 0 || addr == nullptr) {
    return;
  }
  const PageRange range = OuterPageRange(addr, size, SystemPageSize());
  if (!range.empty()) {
    // Best effort: this is only a hint, so failures are ignored.
    madvise(reinterpret_cast<void*>(range.begin), range.size(),
            MADV_NOHUGEPAGE);
  }
#endif
}

bool ReleaseFileBackedPages(int fd, const void* addr, size_t size) {
#if defined(_WIN32)
  return false;
#else
  if (fd < 0 || addr == nullptr) {
    return false;
  }
  const PageRange range = InnerPageRange(addr, size, SystemPageSize());
  if (range.empty()) {
    return false;
  }
  return madvise(reinterpret_cast<void*>(range.begin), range.size(),
                 MADV_DONTNEED) == 0;
#endif
}

}  // namespace litert::mediatek
