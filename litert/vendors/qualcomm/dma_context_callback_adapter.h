// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_DMA_CONTEXT_CALLBACK_ADAPTER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_DMA_CONTEXT_CALLBACK_ADAPTER_H_

#include <cstdint>
#include <memory>

#include "QnnContext.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "litert/c/litert_model_types.h"

namespace litert::qnn {

// Translates QnnContext_createFromBinaryWithCallback()'s DMA-BUF requests
// into the caller-neutral LiteRtDmaContextBinaryCallback the model was
// loaded with, and translates the caller's response back into QNN's ABI.
// Does no allocation or file I/O itself; that is the caller's
// responsibility. `fd_offset` is the context binary's absolute offset
// within the model file the caller's callback reads from. One instance
// serves exactly one QnnContext_createFromBinaryWithCallback() call and
// must outlive the resulting QNN context.
class DmaContextCallbackAdapter {
 public:
  static std::unique_ptr<DmaContextCallbackAdapter> Create(
      uint64_t fd_offset, LiteRtDmaContextBinaryCallback callback);

  DmaContextCallbackAdapter(const DmaContextCallbackAdapter&) = delete;
  DmaContextCallbackAdapter& operator=(const DmaContextCallbackAdapter&) =
      delete;

  // Returns a QNN callback struct wired to this instance.
  Qnn_ContextBinaryCallback_t AsQnnCallback();

 private:
  DmaContextCallbackAdapter(uint64_t fd_offset,
                            LiteRtDmaContextBinaryCallback callback)
      : fd_offset_(fd_offset), callback_(callback) {}

  static Qnn_ErrorHandle_t Provide(Qnn_ContextBinaryDataRequest_t req,
                                   Qnn_ContextBinaryDmaDataResponse_t* resp,
                                   void* notify_param);
  static Qnn_ErrorHandle_t Release(Qnn_ContextBinaryDmaDataMem_t mem,
                                   void* notify_param);

  Qnn_ErrorHandle_t ProvideImpl(Qnn_ContextBinaryDataRequest_t req,
                                Qnn_ContextBinaryDmaDataResponse_t* resp);
  Qnn_ErrorHandle_t ReleaseImpl(Qnn_ContextBinaryDmaDataMem_t mem);

  const uint64_t fd_offset_;
  const LiteRtDmaContextBinaryCallback callback_;
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_DMA_CONTEXT_CALLBACK_ADAPTER_H_
