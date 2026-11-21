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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_FILE_BACKED_PAGES_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_FILE_BACKED_PAGES_H_

#include <cstddef>
#include <cstdint>

namespace litert::mediatek {

// A half-open, page-aligned virtual address range `[begin, end)`.
struct PageRange {
  uintptr_t begin = 0;
  uintptr_t end = 0;

  bool empty() const { return end <= begin; }
  size_t size() const { return empty() ? 0 : end - begin; }
};

// Returns the largest page-aligned range contained in `[addr, addr + size)`.
// The result is empty if no whole page fits, if `page_size` is not a power of
// two, or if the input range overflows the address space.
PageRange InnerPageRange(const void* addr, size_t size, size_t page_size);

// Returns the smallest page-aligned range that covers `[addr, addr + size)`.
// The result is empty if `size` is zero, if `page_size` is not a power of two,
// or if the input range overflows the address space.
PageRange OuterPageRange(const void* addr, size_t size, size_t page_size);

// Helpers for a read-only region of a file mapping, such as the NPU bytecode
// inside an mmapped model file.
//
// `fd` is the descriptor of the file that backs the mapping; a negative value
// means the memory is not known to be file-backed (e.g. a model loaded from a
// heap buffer). Every helper is a no-op in that case because
// `MADV_DONTNEED` zero-fills private anonymous memory instead of re-reading
// it, which would corrupt the buffer.

// Advises the kernel not to back the pages covering the region with
// transparent huge pages. Call this before a one-shot read of the region so
// that the pages can later be released individually.
void AdviseNoHugePages(int fd, const void* addr, size_t size);

// Drops the whole pages inside the region from this process's page tables.
// Partial pages at either end are kept because they may hold neighboring
// data. The region must be a clean file mapping: later accesses transparently
// fault the file contents back in.
//
// Returns true if pages were released; false if nothing was done (see above)
// or if `madvise` failed, in which case `errno` is set.
bool ReleaseFileBackedPages(int fd, const void* addr, size_t size);

// Copies `size` bytes from the region at `src` to `dst` in chunks of
// `chunk_size` bytes (the whole region at once if `chunk_size` is zero). After
// each chunk is copied, the whole source pages that it completes are released
// as in `ReleaseFileBackedPages`, so a large copy does not keep the whole
// source resident at once. `dst` must not overlap the source region.
//
// Returns the number of source bytes that were released; this is 0 if `fd` is
// negative, in which case the source is left untouched.
size_t CopyAndReleaseFileBackedPages(int fd, const void* src, void* dst,
                                     size_t size, size_t chunk_size);

}  // namespace litert::mediatek

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_FILE_BACKED_PAGES_H_
