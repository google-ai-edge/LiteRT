/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ML_DRIFT_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ML_DRIFT_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/cl_memory.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/type_id.h"
#include <CL/cl.h>

namespace litert::tensor {

class MlDriftBuffer : public Buffer {
 public:
  MlDriftBuffer(ml_drift::ValueId value_id,
                ml_drift::cl::InferenceContext& context,
                ml_drift::cl::CLCommandQueue& queue, size_t size_bytes);

  // Constructor for standalone memory.
  MlDriftBuffer(ml_drift::cl::CLMemory memory,
                ml_drift::cl::CLCommandQueue& queue, size_t size_bytes);

  // Creates an MlDriftBuffer attached to an InferenceContext.
  //
  // - value_id: The ID of the tensor in the ML Drift inference context.
  // - context: The inference context managing the tensor.
  // - queue: The command queue for synchronization.
  // - size_bytes: The size of the buffer in bytes.
  static absl::StatusOr<std::unique_ptr<MlDriftBuffer>> Create(
      ml_drift::ValueId value_id, ml_drift::cl::InferenceContext& context,
      ml_drift::cl::CLCommandQueue& queue, size_t size_bytes);

  // Creates an MlDriftBuffer with standalone allocated CL memory.
  static absl::StatusOr<std::unique_ptr<MlDriftBuffer>> Create(
      ml_drift::cl::CLContext& context, ml_drift::cl::CLCommandQueue& queue,
      size_t size_bytes);

  ~MlDriftBuffer() override = default;

  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<MlDriftBuffer>();
  }
  bool IsA(internal::TypeId id) const override {
    return id == internal::TypeId::Get<MlDriftBuffer>();
  }

  // Returns the size of the buffer in bytes. Known without touching the GPU.
  absl::StatusOr<size_t> ByteSize() const override { return GetSizeBytes(); }

  LockedBufferSpan<const std::byte> Lock() override;
  LockedBufferSpan<std::byte> LockMutable() override;

  ml_drift::ValueId value_id() const { return value_id_; }

  cl_mem GetMemory() const;

  // Gets the underlying memory pointer for writing.
  // This correctly returns the underlying buffer for IMAGE_BUFFER storage types
  // to ensure zero-copy checks succeed when comparing src and dst memory
  // pointers.
  cl_mem GetMemoryForWriting() const;
  absl::Status WriteDataAsync(const void* ptr, size_t offset,
                              size_t size_bytes);
  absl::Status CopyFromAsync(const MlDriftBuffer& src, size_t src_offset,
                             size_t dst_offset, size_t size_bytes);
  absl::Status ZeroInit();

  ml_drift::cl::InferenceContext* context() const { return context_; }

  size_t GetSizeBytes() const { return size_bytes_; }

  // Invalidates the host buffer cache, forcing the next Lock() to read from
  // the GPU.
  void Invalidate() { host_buffer_ = nullptr; }

 private:
  ml_drift::ValueId value_id_ = 0;
  ml_drift::cl::InferenceContext* context_ = nullptr;
  ml_drift::cl::CLMemory cl_memory_;
  ml_drift::cl::CLCommandQueue* queue_;
  size_t size_bytes_;
  std::shared_ptr<OwningCpuBuffer> host_buffer_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ML_DRIFT_BUFFER_H_
