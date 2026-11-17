// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_DMA_BUF_CONTEXT_CALLBACK_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_DMA_BUF_CONTEXT_CALLBACK_H_

#include <cstdint>
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>

#include "QnnContext.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace litert::qnn {

// Serves QnnContext_createFromBinaryWithCallback()'s DMA-BUF data requests by
// loading the requested range from `fd` into a freshly allocated DMA buffer.
// `fd_offset` is the context binary's absolute offset within `fd`. One
// instance serves exactly one QnnContext_createFromBinaryWithCallback() call
// and must outlive the resulting QNN context.
class DmaBufContextCallback {
 public:
  static std::unique_ptr<DmaBufContextCallback> Create(int fd,
                                                       uint64_t fd_offset,
                                                       uint64_t size);

  ~DmaBufContextCallback();

  DmaBufContextCallback(const DmaBufContextCallback&) = delete;
  DmaBufContextCallback& operator=(const DmaBufContextCallback&) = delete;

  // Returns a QNN callback struct wired to this instance.
  Qnn_ContextBinaryCallback_t AsQnnCallback();

 private:
  struct Buffer {
    int fd = -1;
    void* addr = nullptr;
    uint64_t size = 0;
  };

  DmaBufContextCallback(int fd, uint64_t fd_offset, uint64_t size)
      : fd_(fd), fd_offset_(fd_offset), size_(size) {}

  static Qnn_ErrorHandle_t Provide(Qnn_ContextBinaryDataRequest_t req,
                                   Qnn_ContextBinaryDmaDataResponse_t* resp,
                                   void* notify_param);
  static Qnn_ErrorHandle_t Release(Qnn_ContextBinaryDmaDataMem_t mem,
                                   void* notify_param);

  Qnn_ErrorHandle_t ProvideImpl(Qnn_ContextBinaryDataRequest_t req,
                                Qnn_ContextBinaryDmaDataResponse_t* resp);
  Qnn_ErrorHandle_t ReleaseImpl(Qnn_ContextBinaryDmaDataMem_t mem);

  const int fd_;
  const uint64_t fd_offset_;
  const uint64_t size_;
  std::mutex mutex_;
  std::unordered_map<int, Buffer> buffers_;  // keyed by dma-buf fd
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_DMA_BUF_CONTEXT_CALLBACK_H_
