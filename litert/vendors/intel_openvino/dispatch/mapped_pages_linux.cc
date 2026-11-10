// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>  // NOLINT

#include "litert/vendors/intel_openvino/dispatch/mapped_pages.h"

namespace litert::openvino {

void ReleaseMappedPagesFromWorkingSet(const void* data, size_t size) {
#if defined(MADV_PAGEOUT)
  // Whole pages strictly inside the range. MADV_PAGEOUT reclaims without
  // discarding (unlike MADV_DONTNEED on a private mapping), so it is safe
  // whatever kind of memory the range lives in.
  const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
  const uintptr_t begin =
      (reinterpret_cast<uintptr_t>(data) + page - 1) & ~(page - 1);
  const uintptr_t end =
      (reinterpret_cast<uintptr_t>(data) + size) & ~(page - 1);
  if (end > begin) {
    madvise(reinterpret_cast<void*>(begin), end - begin, MADV_PAGEOUT);
  }
#else
  (void)data;
  (void)size;
#endif
}

}  // namespace litert::openvino
