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

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/intel_openvino/dispatch/d3d12_shared_buffer.h"

// A dispatch-owned tensor buffer for an OpenVINO partition I/O tensor.
//
// There are two allocation strategies:
//   * Single-device (default): one OpenVINO host tensor created from the target
//     device's remote context (NPU/GPU) or a plain host tensor (CPU). This is
//     the historical behaviour.
//   * Cross-device shared (Windows, model spans both NPU and GPU): one D3D12
//     shared allocation whose CPU pointer serves Lock and whose NT handle is
//     imported into each device's remote context on demand. Both devices then
//     alias one physical allocation, so an NPU->GPU (or GPU->NPU) partition
//     boundary is zero-copy regardless of which partition was created first.
//
// In both cases GetTensorData() returns a host pointer that never throws,
// preserving the LiteRT custom-buffer Lock/Unlock contract.
class OpenVinoTensorBuffer {
 public:
  OpenVinoTensorBuffer(const OpenVinoTensorBuffer&) = delete;
  OpenVinoTensorBuffer& operator=(const OpenVinoTensorBuffer&) = delete;
  OpenVinoTensorBuffer(OpenVinoTensorBuffer&&) = default;
  OpenVinoTensorBuffer& operator=(OpenVinoTensorBuffer&&) = default;

  OpenVinoTensorBuffer() : host_tensor_(), allocated_(false) {};
  ~OpenVinoTensorBuffer() = default;

  litert::Expected<void> Alloc(const LiteRtRankedTensorType& tensor_type,
                               size_t size);

  litert::Expected<void*> GetTensorData();

  // Locks the buffer for CPU access and returns a host pointer. For the shared
  // (D3D12) path a read lock first copies device->host; the mode is remembered
  // so Unlock() can copy host->device after a write. For the single-device path
  // this is equivalent to GetTensorData() and Unlock() is a no-op.
  litert::Expected<void*> Lock(LiteRtTensorBufferLockMode mode);
  litert::Expected<void> Unlock();

  // Returns the OpenVINO tensor to bind for a given target device. For the
  // shared path the NT handle is imported into that device's remote context
  // (lazily, cached); for the single-device path the device argument is ignored
  // and the host tensor is returned.
  litert::Expected<ov::Tensor> GetOVTensor(const std::string& device);

  // True when this buffer uses the cross-device D3D12 shared allocation.
  bool is_shared() const { return shared_path_; }

 private:
  ov::Tensor host_tensor_;
  bool allocated_;

  // Cross-device shared allocation state (populated only on the shared path).
  bool shared_path_ = false;
  LiteRtTensorBufferLockMode last_lock_mode_ = kLiteRtTensorBufferLockModeRead;
  ov::element::Type element_type_;
  ov::Shape shape_;
  // Imported remote tensors keyed by device ("NPU"/"GPU"), aliasing shared_.
  std::unordered_map<std::string, ov::Tensor> device_imports_;
#if defined(LITERT_WINDOWS_OS)
  std::unique_ptr<litert::openvino::D3D12SharedBuffer> shared_;
#endif
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_TENSOR_BUFFER_H_
