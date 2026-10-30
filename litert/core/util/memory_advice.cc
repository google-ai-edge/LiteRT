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

#include "litert/core/util/memory_advice.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(LITERT_WINDOWS_OS)
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif  // defined(LITERT_WINDOWS_OS)
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#endif  // defined(__linux__) || defined(__APPLE__)
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/vm_map.h>
#include <mach/vm_region.h>

#include <atomic>
#endif  // defined(__APPLE__)

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::internal {
namespace {

// SplitPageRanges() keeps a single slice unless every slice would hold at
// least this many ranges; fanning out fewer would cost more than it saves.
constexpr size_t kMinRangesPerSlice = 16;

// Upper bound on the slices SplitPageRanges() produces. A guard only: the
// one caller, AdviseWillNeed() on Apple, asks for kAppleAdviseSlices.
constexpr size_t kMaxSlices = 8;

#if defined(__APPLE__)
// Slices AdviseWillNeed() issues concurrently on Apple. XNU serves
// madvise(MADV_WILLNEED) as a synchronous per-page pre-fault on the calling
// thread (vm_map_willneed), so a single caller faults the rows in one page at
// a time; the count of two comes from the prototype measurements
// (b/576340655).
constexpr int kAppleAdviseSlices = 2;
#endif  // defined(__APPLE__)

#if defined(LITERT_WINDOWS_OS)
// Toolchains targeting Windows 7 hide WIN32_MEMORY_RANGE_ENTRY and
// PrefetchVirtualMemory in <memoryapi.h>; resolve it (Windows 8+) at runtime.
struct Win32MemoryRangeEntry {
  PVOID VirtualAddress;
  SIZE_T NumberOfBytes;
};
using PrefetchVirtualMemoryFn = BOOL(WINAPI*)(HANDLE, ULONG_PTR,
                                              Win32MemoryRangeEntry*, ULONG);

// Returns the PrefetchVirtualMemory entry point, or null when the running
// Windows does not provide it.
PrefetchVirtualMemoryFn GetPrefetchVirtualMemory() {
  static const PrefetchVirtualMemoryFn kPrefetchVirtualMemory =
      reinterpret_cast<PrefetchVirtualMemoryFn>(GetProcAddress(
          GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory"));
  return kPrefetchVirtualMemory;
}
#endif  // defined(LITERT_WINDOWS_OS)

#if defined(__linux__) || defined(__APPLE__)
// Issues madvise(MADV_WILLNEED) for each range, in order, and stops at the
// first failure. Returns 0, or the errno of the failed call (EINVAL for an
// unaligned or wrapped range, ENOMEM for an unmapped one).
int MadviseWillNeed(absl::Span<const PageRange> ranges) {
  for (const PageRange& range : ranges) {
    // madvise takes a non-const pointer; the memory is only read.
    if (madvise(reinterpret_cast<void*>(range.begin), range.end - range.begin,
                MADV_WILLNEED) != 0) {
      return errno;
    }
  }
  return 0;
}
#endif  // defined(__linux__) || defined(__APPLE__)

}  // namespace

#if defined(__APPLE__)
bool IsReadOnlyMapping(const void* data, size_t bytes) {
  if (data == nullptr || bytes == 0) return false;
  vm_address_t address = reinterpret_cast<vm_address_t>(data);
  const vm_address_t end = address + bytes;
  if (end < address) return false;  // Wrapped around.
  while (address < end) {
    vm_address_t region_address = address;
    vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object_name = MACH_PORT_NULL;
    const kern_return_t kr = vm_region_64(
        mach_task_self(), &region_address, &region_size,
        VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info),
        &info_count, &object_name);
    if (object_name != MACH_PORT_NULL) {
      mach_port_deallocate(mach_task_self(), object_name);
    }
    if (kr != KERN_SUCCESS) return false;
    // vm_region_64 returns the first region at or above `address`; one that
    // starts past it means `address` itself is unmapped. The size guard keeps
    // the walk strictly moving forward.
    if (region_address > address || region_size == 0 ||
        region_address + region_size <= address) {
      return false;
    }
    if ((info.protection & VM_PROT_READ) == 0 ||
        (info.protection & VM_PROT_WRITE) != 0) {
      return false;
    }
    address = region_address + region_size;
  }
  return true;
}
#endif  // defined(__APPLE__)

size_t GetPageSize() {
  static const size_t kPageSize = [] {
#if defined(LITERT_WINDOWS_OS)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwPageSize > 0 ? static_cast<size_t>(info.dwPageSize)
                               : size_t{4096};
#else
    const int64_t size = sysconf(_SC_PAGESIZE);
    return size > 0 ? static_cast<size_t>(size) : size_t{4096};
#endif
  }();
  return kPageSize;
}

// Linux queues the readahead asynchronously inside the call, so one thread
// suffices. XNU serves madvise(MADV_WILLNEED) as a synchronous per-page
// pre-fault on the calling thread, so on Apple the ranges are split into up to
// kAppleAdviseSlices contiguous slices that dispatch_apply_f() issues
// concurrently, each in address order.
Expected<void> AdviseWillNeed(absl::Span<const PageRange> ranges) {
#if defined(__linux__) || defined(__APPLE__)
  int first_errno = 0;
#if defined(__APPLE__)
  using Slice = absl::Span<const PageRange>;
  struct Context {
    std::vector<Slice> slices;
    std::atomic<int> first_errno{0};
  } context;
  context.slices = SplitPageRanges(ranges, kAppleAdviseSlices);
  if (context.slices.empty()) return {};
  // Synchronous: returns once every slice is done. The calling thread runs
  // slices too and GCD worker threads, at the caller's QoS, run the rest; if
  // no worker is free the caller runs them all. Unlike starting a thread this
  // cannot fail, so the advisory hint can never abort the process.
  dispatch_apply_f(context.slices.size(), DISPATCH_APPLY_AUTO, &context,
                   [](void* context_ptr, size_t i) {
                     Context& ctx = *static_cast<Context*>(context_ptr);
                     if (const int err = MadviseWillNeed(ctx.slices[i]);
                         err != 0) {
                       int expected = 0;
                       ctx.first_errno.compare_exchange_strong(expected, err);
                     }
                   });
  first_errno = context.first_errno.load();
#else
  first_errno = MadviseWillNeed(ranges);
#endif  // defined(__APPLE__)
  if (first_errno != 0) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrCat("madvise(MADV_WILLNEED) failed: ", strerror(first_errno)));
  }
  return {};
#elif defined(LITERT_WINDOWS_OS)
  const PrefetchVirtualMemoryFn prefetch_virtual_memory =
      GetPrefetchVirtualMemory();
  if (prefetch_virtual_memory == nullptr) {
    return Unexpected(kLiteRtStatusErrorUnsupported,
                      "PrefetchVirtualMemory is not available.");
  }
  if (ranges.empty()) {
    return {};
  }
  std::vector<Win32MemoryRangeEntry> entries;
  entries.reserve(ranges.size());
  for (const PageRange& range : ranges) {
    entries.push_back(
        {reinterpret_cast<PVOID>(range.begin), range.end - range.begin});
  }
  if (!prefetch_virtual_memory(GetCurrentProcess(), entries.size(),
                               entries.data(), /*Flags=*/0)) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrCat("PrefetchVirtualMemory failed: error ", GetLastError()));
  }
  return {};
#else
  (void)ranges;
  return Unexpected(
      kLiteRtStatusErrorUnsupported,
      "No readahead hint (madvise/PrefetchVirtualMemory) on this platform.");
#endif
}

std::vector<absl::Span<const PageRange>> SplitPageRanges(
    absl::Span<const PageRange> ranges, int max_slices) {
  std::vector<absl::Span<const PageRange>> slices;
  if (ranges.empty()) return slices;
  const size_t requested = max_slices > 1 ? static_cast<size_t>(max_slices) : 1;
  const size_t num_slices = std::max<size_t>(
      1, std::min({requested, kMaxSlices, ranges.size() / kMinRangesPerSlice}));
  slices.reserve(num_slices);
  for (size_t i = 0; i < num_slices; ++i) {
    const size_t begin = ranges.size() * i / num_slices;
    const size_t end = ranges.size() * (i + 1) / num_slices;
    slices.push_back(ranges.subspan(begin, end - begin));
  }
  return slices;
}

}  // namespace litert::internal
