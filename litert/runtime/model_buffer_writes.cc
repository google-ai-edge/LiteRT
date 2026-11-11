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

#include "litert/runtime/model_buffer_writes.h"

#include <cstddef>

#include "litert/cc/litert_expected.h"

#ifdef _WIN32
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <mutex>  // NOLINT(build/c++11)

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#endif

namespace litert::internal {

#ifdef _WIN32
namespace {

// Serializes the protection changes. A range made copy-on-write gets its
// private copies before the mutex is released, so that a view's unwritten
// pages can be made read-only at any time without taking back a range that
// was asked for but not written to yet.
std::mutex& ProtectionMutex() {
  static auto* const mutex = new std::mutex();
  return *mutex;
}

uintptr_t RegionEnd(const MEMORY_BASIC_INFORMATION& region) {
  return reinterpret_cast<uintptr_t>(region.BaseAddress) + region.RegionSize;
}

// Makes the pages of the copy-on-write view at `base` that nothing has written
// to read-only. A page that was written to has its own copy and reads
// PAGE_READWRITE; it keeps its protection and its charge. The caller holds
// ProtectionMutex().
void LowerUnwrittenPages(void* base) {
  size_t lowered = 0;
  MEMORY_BASIC_INFORMATION region;
  uintptr_t cursor = reinterpret_cast<uintptr_t>(base);
  while (VirtualQuery(reinterpret_cast<const void*>(cursor), &region,
                      sizeof(region)) == sizeof(region) &&
         region.AllocationBase == base) {
    DWORD previous;
    if (region.State == MEM_COMMIT && region.Protect == PAGE_WRITECOPY &&
        VirtualProtect(region.BaseAddress, region.RegionSize, PAGE_READONLY,
                       &previous)) {
      lowered += region.RegionSize;
    }
    cursor = RegionEnd(region);
  }
  if (lowered > 0) {
    LITERT_LOG(LITERT_INFO,
               "Made %zu MiB of a copy-on-write model mapping read-only until "
               "written to",
               lowered >> 20);
  }
}

bool IsCopyOnWriteView(const MEMORY_BASIC_INFORMATION& region) {
  return region.Type == MEM_MAPPED && region.State == MEM_COMMIT &&
         region.AllocationProtect == PAGE_WRITECOPY;
}

// Gives each page of [begin, end) its private copy now, by writing back a byte
// it holds, so that the pages read PAGE_READWRITE from here on.
void CopyPages(uintptr_t begin, uintptr_t end, uintptr_t page) {
  for (uintptr_t at = begin; at < end; at += page) {
    volatile uint8_t* byte = reinterpret_cast<volatile uint8_t*>(at);
    *byte = *byte;
  }
}

}  // namespace
#endif  // _WIN32

void KeepModelBufferReadOnlyUntilWritten(const void* data) {
#ifdef _WIN32
  MEMORY_BASIC_INFORMATION region;
  if (data == nullptr ||
      VirtualQuery(data, &region, sizeof(region)) != sizeof(region) ||
      !IsCopyOnWriteView(region)) {
    return;
  }
  std::lock_guard<std::mutex> lock(ProtectionMutex());
  LowerUnwrittenPages(region.AllocationBase);
#endif  // _WIN32
}

Expected<void> MakeModelBufferWritable(const void* data, size_t size) {
#ifdef _WIN32
  MEMORY_BASIC_INFORMATION region;
  if (size == 0 ||
      VirtualQuery(data, &region, sizeof(region)) != sizeof(region) ||
      region.Type != MEM_MAPPED) {
    return {};
  }
  std::lock_guard<std::mutex> lock(ProtectionMutex());
  // A page still copy-on-write and unwritten: the view was not made read-only
  // yet (KeepModelBufferReadOnlyUntilWritten), so it is now.
  if (IsCopyOnWriteView(region) && region.Protect == PAGE_WRITECOPY) {
    LowerUnwrittenPages(region.AllocationBase);
  }
  SYSTEM_INFO system;
  GetSystemInfo(&system);
  const uintptr_t page = system.dwPageSize;
  const uintptr_t end =
      (reinterpret_cast<uintptr_t>(data) + size + page - 1) / page * page;
  uintptr_t cursor = reinterpret_cast<uintptr_t>(data) / page * page;
  while (cursor < end) {
    if (VirtualQuery(reinterpret_cast<const void*>(cursor), &region,
                     sizeof(region)) != sizeof(region) ||
        region.State != MEM_COMMIT) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "A model buffer to write to is not mapped");
    }
    const uintptr_t part_end = std::min(RegionEnd(region), end);
    DWORD previous;
    switch (region.Protect) {
      case PAGE_READWRITE:
        break;
      case PAGE_READONLY:
        if (region.Type != MEM_MAPPED ||
            !VirtualProtect(reinterpret_cast<void*>(cursor), part_end - cursor,
                            PAGE_WRITECOPY, &previous)) {
          return Unexpected(
              kLiteRtStatusErrorRuntimeFailure,
              "A model buffer to write to cannot be made writable");
        }
        [[fallthrough]];
      case PAGE_WRITECOPY:
        CopyPages(cursor, part_end, page);
        break;
      default:
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "A model buffer to write to cannot be made writable");
    }
    cursor = part_end;
  }
#endif  // _WIN32
  return {};
}

}  // namespace litert::internal
