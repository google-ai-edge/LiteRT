// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/tools/ai_core_sample/dma_buf_provider.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <mutex>  // NOLINT

#if __has_include(<linux/dma-heap.h>)
#include <linux/dma-heap.h>
#else
struct dma_heap_allocation_data {
  uint64_t len;
  uint32_t fd;
  uint32_t fd_flags;
  uint64_t heap_flags;
};
#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC \
  _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)
#endif

#include <sys/ioctl.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"

namespace litert::tools {
namespace {
constexpr uint64_t kPageSize = 4096;
uint64_t AlignUp(uint64_t v) { return (v + kPageSize - 1) & ~(kPageSize - 1); }
}  // namespace

LiteRtDmaContextBinaryCallback DmaBufProvider::AsCallback() {
  LiteRtDmaContextBinaryCallback callback{};
  callback.provide = &DmaBufProvider::Provide;
  callback.release = &DmaBufProvider::Release;
  callback.user_data = this;
  return callback;
}

LiteRtStatus DmaBufProvider::Provide(LiteRtDmaContextDataRequest request,
                                     LiteRtDmaContextDataResponse* response,
                                     void* user_data) {
  auto* self = static_cast<DmaBufProvider*>(user_data);
  if (self == nullptr || response == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return self->ProvideImpl(request, response);
}

LiteRtStatus DmaBufProvider::Release(LiteRtDmaContextDataResponse response,
                                     void* user_data) {
  auto* self = static_cast<DmaBufProvider*>(user_data);
  if (self == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return self->ReleaseImpl(response);
}

LiteRtStatus DmaBufProvider::ProvideImpl(
    LiteRtDmaContextDataRequest request,
    LiteRtDmaContextDataResponse* response) {
  if (request.size == 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const uint64_t aligned_size = AlignUp(request.size);

  // Each request must get a fresh, distinct DMA-BUF: the HTP backend tracks
  // SMMU imports by DMA-BUF identity, so returning the same fd (or a dup())
  // across multiple requests fails the second import.
  int heap_fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
  if (heap_fd < 0) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  struct dma_heap_allocation_data alloc {};
  alloc.len = aligned_size;
  alloc.fd_flags = O_RDWR | O_CLOEXEC;
  int rc = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc);
  close(heap_fd);
  if (rc < 0) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  int dma_fd = static_cast<int>(alloc.fd);
  void* addr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    dma_fd, 0);
  if (addr == MAP_FAILED) {
    close(dma_fd);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  ssize_t n = pread(model_fd_, addr, request.size,
                    static_cast<off_t>(request.offset));
  if (n < 0 || static_cast<uint64_t>(n) != request.size) {
    munmap(addr, aligned_size);
    close(dma_fd);
    return kLiteRtStatusErrorFileIO;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_[dma_fd] = Buffer{dma_fd, addr, aligned_size};
  }
  response->fd = dma_fd;
  response->data = addr;
  response->data_start_offset = 0;
  response->aligned_size = aligned_size;
  return kLiteRtStatusOk;
}

LiteRtStatus DmaBufProvider::ReleaseImpl(
    LiteRtDmaContextDataResponse response) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(response.fd);
  if (it == buffers_.end()) {
    return kLiteRtStatusOk;
  }
  if (it->second.addr != nullptr) munmap(it->second.addr, it->second.size);
  if (it->second.fd >= 0) close(it->second.fd);
  buffers_.erase(it);
  return kLiteRtStatusOk;
}

}  // namespace litert::tools
