// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/dma_buf_context_callback.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <sys/stat.h>

#include <cerrno>
#include <cstdint>
#include <memory>
#include <mutex>  // NOLINT
#include <string>

#if __has_include(<linux/dma-heap.h>)
#include <linux/dma-heap.h>
#else
// The hermetic x86 host sysroot (building this file's host-only caller,
// the compiler plugin) lacks this kernel UAPI header; define it locally.
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

#include "QnnTypes.h"  // from @qairt
#include "litert/c/internal/litert_logging.h"

namespace litert::qnn {
namespace {
constexpr uint64_t kPageSize = 4096;
uint64_t AlignUp(uint64_t v) { return (v + kPageSize - 1) & ~(kPageSize - 1); }
}  // namespace

std::unique_ptr<DmaBufContextCallback> DmaBufContextCallback::Create(
    int fd, uint64_t fd_offset, uint64_t size) {
  return std::unique_ptr<DmaBufContextCallback>(
      new DmaBufContextCallback(fd, fd_offset, size));
}

bool DmaBufContextCallback::IsDmaBuf(int fd) {
  char link[64];
  char target[256];
  snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
  const ssize_t n = readlink(link, target, sizeof(target) - 1);
  if (n <= 0) return false;
  target[n] = '\0';
  return std::string(target).find("dmabuf") != std::string::npos;
}

DmaBufContextCallback::~DmaBufContextCallback() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (dma_buf_base_ != nullptr) munmap(dma_buf_base_, dma_buf_size_);
  for (auto& [dma_fd, buf] : buffers_) {
    if (buf.addr != nullptr) munmap(buf.addr, buf.size);
    if (buf.fd >= 0) close(buf.fd);
  }
  buffers_.clear();
}

Qnn_ContextBinaryCallback_t DmaBufContextCallback::AsQnnCallback() {
  Qnn_ContextBinaryCallback_t callback{};
  callback.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
  callback.dmaBufferCallback.version =
      QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
  callback.dmaBufferCallback.v1.dataProvide = &DmaBufContextCallback::Provide;
  callback.dmaBufferCallback.v1.dataRelease = &DmaBufContextCallback::Release;
  callback.dmaBufferCallback.v1.notifyParam = this;
  return callback;
}

Qnn_ErrorHandle_t DmaBufContextCallback::Provide(
    Qnn_ContextBinaryDataRequest_t req,
    Qnn_ContextBinaryDmaDataResponse_t* resp, void* notify_param) {
  auto* self = static_cast<DmaBufContextCallback*>(notify_param);
  if (self == nullptr || resp == nullptr) {
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  return self->ProvideImpl(req, resp);
}

Qnn_ErrorHandle_t DmaBufContextCallback::Release(
    Qnn_ContextBinaryDmaDataMem_t mem, void* notify_param) {
  auto* self = static_cast<DmaBufContextCallback*>(notify_param);
  if (self == nullptr) {
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  return self->ReleaseImpl(mem);
}

Qnn_ErrorHandle_t DmaBufContextCallback::ProvideImpl(
    Qnn_ContextBinaryDataRequest_t req,
    Qnn_ContextBinaryDmaDataResponse_t* resp) {
  if (req.size == 0 || req.offset + req.size > size_) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "DmaBufContextCallback: data request out of range");
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  if (fd_is_dma_buf_) return ProvideFromPreloaded(req, resp);
  const uint64_t aligned_size = AlignUp(req.size);

  // Allocate a DMA buffer sized for this request.
  int heap_fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
  if (heap_fd < 0) {
    LITERT_LOG(LITERT_ERROR, "%s", "DmaBufContextCallback: open dma_heap");
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }
  struct dma_heap_allocation_data alloc {};
  alloc.len = aligned_size;
  alloc.fd_flags = O_RDWR | O_CLOEXEC;
  int rc = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc);
  close(heap_fd);
  if (rc < 0) {
    LITERT_LOG(LITERT_ERROR, "DmaBufContextCallback: alloc failed (errno=%d)",
               errno);
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }
  int dma_fd = static_cast<int>(alloc.fd);
  void* addr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    dma_fd, 0);
  if (addr == MAP_FAILED) {
    close(dma_fd);
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }

  // Load the requested range directly from the model file into the buffer.
  // The valid data begins at the buffer start (dataStartOffset = 0).
  ssize_t n = pread(fd_, addr, req.size,
                    static_cast<off_t>(fd_offset_ + req.offset));
  if (n < 0 || static_cast<uint64_t>(n) != req.size) {
    munmap(addr, aligned_size);
    close(dma_fd);
    LITERT_LOG(LITERT_ERROR, "%s", "DmaBufContextCallback: pread failed");
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_[dma_fd] = Buffer{dma_fd, addr, aligned_size};
  }
  resp->dmaBuffer.fd = dma_fd;
  resp->dmaBuffer.data = addr;
  resp->dataStartOffset = 0;
  resp->alignedSize = aligned_size;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t DmaBufContextCallback::ProvideFromPreloaded(
    Qnn_ContextBinaryDataRequest_t req,
    Qnn_ContextBinaryDmaDataResponse_t* resp) {
  const uint64_t start = fd_offset_ + req.offset;
  if (start % kPageSize != 0) {
    LITERT_LOG(LITERT_ERROR,
               "DmaBufContextCallback: weights start at fd offset %llu, not "
               "4KB aligned; HTP would read wrong data",
               static_cast<unsigned long long>(start));
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (dma_buf_base_ == nullptr) {
    struct stat st {};
    if (fstat(fd_, &st) != 0 || st.st_size <= 0) {
      return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
    }
    dma_buf_size_ = static_cast<uint64_t>(st.st_size);
    void* base = mmap(nullptr, dma_buf_size_, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_, 0);
    if (base == MAP_FAILED) return QNN_CONTEXT_ERROR_MEM_ALLOC;
    dma_buf_base_ = base;
  }
  if (start + req.size > dma_buf_size_) {
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  // HTP needs a distinct fd per request; a dup() of the same DMA-BUF is one.
  const int dup_fd = dup(fd_);
  if (dup_fd < 0) return QNN_CONTEXT_ERROR_MEM_ALLOC;
  resp->dmaBuffer.fd = dup_fd;
  resp->dmaBuffer.data = dma_buf_base_;
  resp->dataStartOffset = start;
  // Must equal the requested size so QNN's unmap length matches.
  resp->alignedSize = req.size;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t DmaBufContextCallback::ReleaseImpl(
    Qnn_ContextBinaryDmaDataMem_t mem) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(mem.dmaBuffer.fd);
  if (it == buffers_.end()) {
    // Preloaded mode: only the dup()'d fd is ours to close.
    if (fd_is_dma_buf_ && mem.dmaBuffer.fd >= 0) close(mem.dmaBuffer.fd);
    return QNN_SUCCESS;
  }
  if (it->second.addr != nullptr) munmap(it->second.addr, it->second.size);
  if (it->second.fd >= 0) close(it->second.fd);
  buffers_.erase(it);
  return QNN_SUCCESS;
}

}  // namespace litert::qnn
