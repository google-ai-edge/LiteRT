// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/dma_context_callback_adapter.h"

#include <cstdint>
#include <memory>

#include "QnnTypes.h"  // from @qairt
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"

namespace litert::qnn {

std::unique_ptr<DmaContextCallbackAdapter> DmaContextCallbackAdapter::Create(
    uint64_t fd_offset, LiteRtDmaContextBinaryCallback callback) {
  return std::unique_ptr<DmaContextCallbackAdapter>(
      new DmaContextCallbackAdapter(fd_offset, callback));
}

Qnn_ContextBinaryCallback_t DmaContextCallbackAdapter::AsQnnCallback() {
  Qnn_ContextBinaryCallback_t callback{};
  callback.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
  callback.dmaBufferCallback.version =
      QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
  callback.dmaBufferCallback.v1.dataProvide =
      &DmaContextCallbackAdapter::Provide;
  callback.dmaBufferCallback.v1.dataRelease =
      &DmaContextCallbackAdapter::Release;
  callback.dmaBufferCallback.v1.notifyParam = this;
  return callback;
}

Qnn_ErrorHandle_t DmaContextCallbackAdapter::Provide(
    Qnn_ContextBinaryDataRequest_t req,
    Qnn_ContextBinaryDmaDataResponse_t* resp, void* notify_param) {
  auto* self = static_cast<DmaContextCallbackAdapter*>(notify_param);
  if (self == nullptr || resp == nullptr) {
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  return self->ProvideImpl(req, resp);
}

Qnn_ErrorHandle_t DmaContextCallbackAdapter::Release(
    Qnn_ContextBinaryDmaDataMem_t mem, void* notify_param) {
  auto* self = static_cast<DmaContextCallbackAdapter*>(notify_param);
  if (self == nullptr) {
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  return self->ReleaseImpl(mem);
}

Qnn_ErrorHandle_t DmaContextCallbackAdapter::ProvideImpl(
    Qnn_ContextBinaryDataRequest_t req,
    Qnn_ContextBinaryDmaDataResponse_t* resp) {
  if (callback_.provide == nullptr) {
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  // req.offset is relative to the start of the context binary; the caller's
  // callback deals in absolute offsets within the model file.
  LiteRtDmaContextDataRequest lrt_req{
      /*.offset=*/fd_offset_ + req.offset,
      /*.size=*/req.size,
  };
  LiteRtDmaContextDataResponse lrt_resp{};
  if (callback_.provide(lrt_req, &lrt_resp, callback_.user_data) !=
      kLiteRtStatusOk) {
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }

  resp->dmaBuffer.fd = lrt_resp.fd;
  resp->dmaBuffer.data = lrt_resp.data;
  resp->dataStartOffset = lrt_resp.data_start_offset;
  resp->alignedSize = lrt_resp.aligned_size;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t DmaContextCallbackAdapter::ReleaseImpl(
    Qnn_ContextBinaryDmaDataMem_t mem) {
  if (callback_.release == nullptr) {
    return QNN_SUCCESS;
  }
  LiteRtDmaContextDataResponse lrt_resp{
      /*.fd=*/mem.dmaBuffer.fd,
      /*.data=*/mem.dmaBuffer.data,
      /*.data_start_offset=*/0,
      /*.aligned_size=*/mem.memSize,
  };
  if (callback_.release(lrt_resp, callback_.user_data) != kLiteRtStatusOk) {
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }
  return QNN_SUCCESS;
}

}  // namespace litert::qnn
