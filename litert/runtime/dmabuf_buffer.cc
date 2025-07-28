// Copyright 2024 Google LLC.
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

#include "litert/runtime/dmabuf_buffer.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/container/node_hash_map.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

#if LITERT_HAS_DMABUF_SUPPORT
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif  // LITERT_HAS_DMABUF_SUPPORT

namespace litert::internal {

#if LITERT_HAS_DMABUF_SUPPORT
namespace {

int DmabufHeapAlloc(int heap_fd, size_t len) {
  dma_heap_allocation_data data = {};
  data.len = len;  // Length of data to be allocated in bytes.
  data.fd = 0;     // Output parameter.
  data.fd_flags =
      O_RDWR | O_CLOEXEC;  // Permissions for the memory to be allocated.
  data.heap_flags = 0;
  int ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data);
  if (ret < 0) {
    return ret;
  }
  return data.fd;
}

class DmaBufLibrary {
 public:
  using Ptr = std::unique_ptr<DmaBufLibrary>;

  DmaBufLibrary(const DmaBufLibrary&) = delete;
  DmaBufLibrary& operator=(const DmaBufLibrary&) = delete;
  DmaBufLibrary(DmaBufLibrary&&) = default;
  DmaBufLibrary& operator=(DmaBufLibrary&&) = default;

  ~DmaBufLibrary() { close(heap_fd_); }

  static Expected<Ptr> Create() {
    int heap_fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (heap_fd < 0) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to open DMA-BUF system heap.");
    }
    return Ptr(new DmaBufLibrary(heap_fd));
  }

  Expected<DmaBufBuffer> Alloc(size_t size) {
    int fd = DmabufHeapAlloc(heap_fd_, size);
    if (fd < 0) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to allocate DMA-BUF buffer");
    }
    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to mem-map DMA-BUF buffer");
    }
    records_[addr] = Record{.fd = fd, .addr = addr, .size = size};
    return DmaBufBuffer{.fd = fd, .addr = addr};
  }

  void Free(void* addr) {
    auto iter = records_.find(addr);
    if (iter == records_.end()) {
      return;
    }
    auto& record = iter->second;
    munmap(record.addr, record.size);
    close(record.fd);
    records_.erase(iter);
  }

 private:
  struct Record {
    int fd;
    void* addr;
    size_t size;
  };

  explicit DmaBufLibrary(int heap_fd) : heap_fd_(heap_fd) {}

  int heap_fd_;
  absl::node_hash_map<void*, Record> records_;
};

DmaBufLibrary* TheDmaBufLibrary;
ABSL_CONST_INIT absl::Mutex TheMutex(absl::kConstInit);

Expected<void> InitLibraryIfNeededUnlocked() {
  if (!TheDmaBufLibrary) {
    if (auto library = DmaBufLibrary::Create(); library) {
      TheDmaBufLibrary = library->release();
    } else {
      return Unexpected(library.Error());
    }
  }
  return {};
}

}  // namespace
#endif  // LITERT_HAS_DMABUF_SUPPORT

bool DmaBufBuffer::IsSupported() {
#if LITERT_HAS_DMABUF_SUPPORT
  absl::MutexLock lock(&TheMutex);
  auto status = InitLibraryIfNeededUnlocked();
  return static_cast<bool>(status);
#else   // LITERT_HAS_DMABUF_SUPPORT
  return false;
#endif  // LITERT_HAS_DMABUF_SUPPORT
}

Expected<DmaBufBuffer> DmaBufBuffer::Alloc(size_t size) {
#if LITERT_HAS_DMABUF_SUPPORT
  absl::MutexLock lock(&TheMutex);
  if (auto status = InitLibraryIfNeededUnlocked(); !status) {
    return Unexpected(status.Error());
  }
  return TheDmaBufLibrary->Alloc(size);
#else   // LITERT_HAS_DMABUF_SUPPORT
  return Unexpected(kLiteRtStatusErrorUnsupported,
                    "DmaBufBuffer::Alloc not implemented for this platform");
#endif  // LITERT_HAS_DMABUF_SUPPORT
}

void DmaBufBuffer::Free(void* addr) {
#if LITERT_HAS_DMABUF_SUPPORT
  absl::MutexLock lock(&TheMutex);
  if (TheDmaBufLibrary) {
    TheDmaBufLibrary->Free(addr);
  }
#endif  // LITERT_HAS_DMABUF_SUPPORT
}

}  // namespace litert::internal
