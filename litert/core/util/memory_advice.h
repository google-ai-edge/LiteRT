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

#ifndef ODML_LITERT_LITERT_CORE_UTIL_MEMORY_ADVICE_H_
#define ODML_LITERT_LITERT_CORE_UTIL_MEMORY_ADVICE_H_

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"

namespace litert::internal {

// Page-aligned half-open address range [begin, end).
struct PageRange {
  uintptr_t begin;
  uintptr_t end;
  bool operator==(const PageRange& other) const {
    return begin == other.begin && end == other.end;
  }
};

// The OS page size (a power of two; 4096 if the query fails).
size_t GetPageSize();

// Issues the OS readahead hint for `ranges`: madvise(MADV_WILLNEED) on Linux
// and Android, PrefetchVirtualMemory on Windows (resolved at runtime, Windows
// 8+). Returns Unsupported where no hint exists so callers can stop calling,
// and RuntimeFailure with the first failed call's error (for example EINVAL or
// ENOMEM for a range that is not mapped) so a wrong address does not pass
// silently; the remaining ranges are then not hinted. Ranges must be
// page-aligned. An empty `ranges` issues nothing and only reports
// availability (Ok or Unsupported), which callers use as a probe.
Expected<void> AdviseWillNeed(absl::Span<const PageRange> ranges);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_UTIL_MEMORY_ADVICE_H_
