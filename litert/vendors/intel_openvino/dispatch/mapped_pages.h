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

#ifndef ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_MAPPED_PAGES_H_
#define ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_MAPPED_PAGES_H_

#include <cstddef>

namespace litert::openvino {

// Lets the OS drop [data, data + size) -- part of a read-only view of a file --
// from this process's working set. Nothing is discarded: if the range is read
// again, the pages come back from the file (or the OS file cache).
//
// Used for a dispatch node's payload, which import_model reads once from the
// memory-mapped model and never touches again; otherwise its pages stay
// resident (on the gemma4-12b 4-tier bundle every payload stayed 100%
// resident, ~0.8 GB after three tier crossings).
//
// One implementation per OS: mapped_pages_win.cc, mapped_pages_linux.cc.
void ReleaseMappedPagesFromWorkingSet(const void* data, size_t size);

}  // namespace litert::openvino

#endif  // ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_MAPPED_PAGES_H_
