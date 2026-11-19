// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/tools/ai_core_sample/dma_buf_provider.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
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

LiteRtDmaContextBinaryCallback PreloadedDmaBufProvider::AsCallback() {
  LiteRtDmaContextBinaryCallback callback{};
  callback.provide = &PreloadedDmaBufProvider::Provide;
  callback.release = &PreloadedDmaBufProvider::Release;
  callback.user_data = this;
  return callback;
}

PreloadedDmaBufProvider::~PreloadedDmaBufProvider() {
  if (base_ != nullptr) munmap(base_, size_);
  if (dmabuf_fd_ >= 0) close(dmabuf_fd_);
}

LiteRtStatus PreloadedDmaBufProvider::Stage(uint64_t pad) {
  const int file_fd = open(model_path_.c_str(), O_RDONLY | O_CLOEXEC);
  if (file_fd < 0) return kLiteRtStatusErrorFileIO;
  struct stat st {};
  if (fstat(file_fd, &st) != 0) {
    close(file_fd);
    return kLiteRtStatusErrorFileIO;
  }
  file_size_ = static_cast<uint64_t>(st.st_size);
  size_ = AlignUp(pad + file_size_);
  const int heap_fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
  if (heap_fd < 0) {
    close(file_fd);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  struct dma_heap_allocation_data alloc {};
  alloc.len = size_;
  alloc.fd_flags = O_RDWR | O_CLOEXEC;
  const int rc = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc);
  close(heap_fd);
  if (rc < 0) {
    close(file_fd);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  dmabuf_fd_ = static_cast<int>(alloc.fd);
  void* addr =
      mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd_, 0);
  if (addr == MAP_FAILED) {
    close(file_fd);
    close(dmabuf_fd_);
    dmabuf_fd_ = -1;
    return kLiteRtStatusErrorRuntimeFailure;
  }
  uint64_t done = 0;
  while (done < file_size_) {
    const ssize_t n = read(file_fd, static_cast<char*>(addr) + pad + done,
                           file_size_ - done);
    if (n <= 0) break;
    done += static_cast<uint64_t>(n);
  }
  close(file_fd);
  if (done != file_size_) {
    munmap(addr, size_);
    close(dmabuf_fd_);
    dmabuf_fd_ = -1;
    return kLiteRtStatusErrorFileIO;
  }
  base_ = addr;
  pad_ = pad;
  return kLiteRtStatusOk;
}

LiteRtStatus PreloadedDmaBufProvider::Provide(
    LiteRtDmaContextDataRequest request, LiteRtDmaContextDataResponse* response,
    void* user_data) {
  auto* self = static_cast<PreloadedDmaBufProvider*>(user_data);
  if (self == nullptr || response == nullptr || request.size == 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  std::lock_guard<std::mutex> lock(self->mutex_);
  if (self->dmabuf_fd_ < 0) {
    // HTP needs weights to start on a 4KB boundary. Place the model so the
    // first requested range does; later ranges share the same file layout.
    const uint64_t pad = (kPageSize - request.offset % kPageSize) % kPageSize;
    if (const auto status = self->Stage(pad); status != kLiteRtStatusOk) {
      return status;
    }
  }
  if (request.offset + request.size > self->file_size_) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // The data is already in the DMA-BUF: nothing to allocate, read or copy.
  const int fd = dup(self->dmabuf_fd_);
  if (fd < 0) return kLiteRtStatusErrorRuntimeFailure;
  response->fd = fd;
  response->data = self->base_;
  response->data_start_offset = self->pad_ + request.offset;
  // Keep the mapped length equal to the requested size so QNN's unmap length
  // matches the registered length.
  response->aligned_size = request.size;
  return kLiteRtStatusOk;
}

LiteRtStatus PreloadedDmaBufProvider::Release(
    LiteRtDmaContextDataResponse response, void* /*user_data*/) {
  // Only the dup'd fd handed out by Provide(); the DMA-BUF and its mapping
  // stay with the provider until it is destroyed.
  if (response.fd >= 0) close(response.fd);
  return kLiteRtStatusOk;
}

}  // namespace litert::tools
