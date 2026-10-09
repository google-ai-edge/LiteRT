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

#include <cstddef>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "litert/vendors/intel_openvino/dispatch/mapped_pages.h"

namespace litert::openvino {

void ReleaseMappedPagesFromWorkingSet(const void* data, size_t size) {
  // Only for a view of a file. VirtualUnlock on pages that are not locked
  // removes them from the working set to the standby list (it then reports
  // ERROR_NOT_LOCKED, which is expected); on private memory it would push
  // them toward the page file instead, so skip anything else.
  MEMORY_BASIC_INFORMATION mbi;
  if (VirtualQuery(data, &mbi, sizeof(mbi)) == sizeof(mbi) &&
      mbi.Type == MEM_MAPPED) {
    VirtualUnlock(const_cast<void*>(data), size);
  }
}

}  // namespace litert::openvino
