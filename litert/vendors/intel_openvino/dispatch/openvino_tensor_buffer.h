// Copyright 2025 Google LLC.
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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_TENSOR_BUFFER_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_TENSOR_BUFFER_H_

#include <atomic>
#include <cstddef>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"

// One LiteRT tensor buffer of type kLiteRtTensorBufferTypeOpenVINOTensorBuffer:
// an ov::Tensor in host memory shared with the device (zero-copy on NPU/GPU).
//
// Memory is created lazily. Alloc() records only the element type and shape;
// the host tensor is created, zero-filled, on first use: a Lock()
// (GetTensorData) or a dispatch node using the buffer (GetOVTensor). A
// multi-context model creates every tier's I/O buffers at init, so a tier a
// session never reaches costs nothing.
//
// Clear() zeroes the buffer. On a buffer not created yet it does nothing: a new
// buffer starts zeroed anyway.
//
// Wrap() makes the buffer a non-owning view of memory owned elsewhere (a
// smaller tier's KV-cache alias onto the largest tier's buffer). It exists from
// the start and is never created or freed here.
//
// All methods are thread-safe (one mutex per buffer).
class OpenVinoTensorBuffer {
 public:
  OpenVinoTensorBuffer(const OpenVinoTensorBuffer&) = delete;
  OpenVinoTensorBuffer& operator=(const OpenVinoTensorBuffer&) = delete;

  OpenVinoTensorBuffer() = default;
  ~OpenVinoTensorBuffer() = default;

  litert::Expected<void> Alloc(const LiteRtRankedTensorType& tensor_type,
                               size_t size);

  // Wraps `host_memory`, the CPU-visible data pointer of an existing
  // OpenVINO tensor (for the NPU and GPU devices, a host tensor allocated
  // by the device's remote context, so the device can use it in place), as
  // an ov::Tensor of `tensor_type` without taking ownership. Used to give a
  // smaller-context KV-cache signature a view onto the largest context's
  // KV-cache buffer for multi-signature NPU models.
  litert::Expected<void> Wrap(const LiteRtRankedTensorType& tensor_type,
                              void* host_memory, size_t size);

  // Whether any buffer in this process has been Wrap()ed, i.e. some buffers
  // alias memory owned by others.
  static bool AnyWrapped();

  litert::Expected<void*> GetTensorData();

  litert::Expected<ov::Tensor> GetOVTensor();

  // Zero the buffer (the LiteRT custom-buffer clear hook).
  litert::Expected<void> Clear();

 private:
  // Creates ov_tensor_ if it does not exist yet. `why` is for the log.
  litert::Expected<void> Materialize(const char* why)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  static std::atomic<bool>& AnyWrappedFlag();

  absl::Mutex mutex_;
  bool allocated_ ABSL_GUARDED_BY(mutex_) = false;  // Alloc() has run
  ov::element::Type element_type_ ABSL_GUARDED_BY(mutex_);
  ov::Shape shape_ ABSL_GUARDED_BY(mutex_);
  ov::Tensor ov_tensor_ ABSL_GUARDED_BY(mutex_);  // empty until Materialize()
  bool wrapped_ ABSL_GUARDED_BY(mutex_) = false;  // Wrap(): external memory
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_TENSOR_BUFFER_H_
