// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_TOOLS_AI_CORE_SAMPLE_DMA_BUF_PROVIDER_H_
#define ODML_LITERT_LITERT_TOOLS_AI_CORE_SAMPLE_DMA_BUF_PROVIDER_H_

#include <cstdint>
#include <mutex>  // NOLINT
#include <unordered_map>

#include "litert/c/litert_model_types.h"

namespace litert::tools {

// Reference implementation of the caller-side (AICore) half of
// LiteRtCreateModelFromFdWithDmaContextCallback: for each requested byte
// range, allocates a fresh page-aligned DMA buffer and loads the range
// directly from `model_fd` into it with pread. This is what a real AICore
// implementation would replace with its own allocator (e.g. a per-request
// VM-donated DMA-BUF), since each request must return a DMA-BUF with a
// distinct identity -- see the QAIRT HTP backend guide's
// "QnnContext_createFromBinaryWithCallback API" section.
//
// `model_fd` is not owned; the caller keeps it open for the instance's
// lifetime.
class DmaBufProvider {
 public:
  explicit DmaBufProvider(int model_fd) : model_fd_(model_fd) {}

  DmaBufProvider(const DmaBufProvider&) = delete;
  DmaBufProvider& operator=(const DmaBufProvider&) = delete;

  // Returns a LiteRtDmaContextBinaryCallback wired to this instance.
  LiteRtDmaContextBinaryCallback AsCallback();

 private:
  struct Buffer {
    int fd = -1;
    void* addr = nullptr;
    uint64_t size = 0;
  };

  static LiteRtStatus Provide(LiteRtDmaContextDataRequest request,
                              LiteRtDmaContextDataResponse* response,
                              void* user_data);
  static LiteRtStatus Release(LiteRtDmaContextDataResponse response,
                              void* user_data);

  LiteRtStatus ProvideImpl(LiteRtDmaContextDataRequest request,
                           LiteRtDmaContextDataResponse* response);
  LiteRtStatus ReleaseImpl(LiteRtDmaContextDataResponse response);

  const int model_fd_;
  std::mutex mutex_;
  std::unordered_map<int, Buffer> buffers_;  // keyed by dma-buf fd
};

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_TOOLS_AI_CORE_SAMPLE_DMA_BUF_PROVIDER_H_
