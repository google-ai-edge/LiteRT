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

#include "litert/vendors/intel_openvino/dispatch/openvino_tensor_buffer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
// Only the lightweight property headers are needed (they pull in just
// properties.hpp). The typed ocl.hpp / level_zero.hpp wrappers are avoided on
// purpose: ocl.hpp transitively requires the OpenCL SDK headers (CL/cl2.hpp),
// which are not on the include path. The generic
// RemoteContext::create_tensor(type, shape, AnyMap) below does exactly what
// those wrappers do internally.
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/intel_openvino/dispatch/openvino_shared_core.h"
#include "litert/vendors/intel_openvino/utils.h"

litert::Expected<void> OpenVinoTensorBuffer::Alloc(
    const LiteRtRankedTensorType& tensor_type, size_t size) {
  if (allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The OpenVino tensor has been allocated.");
  }

  // TODO:: Release the shared OpenVINO Core.
  auto* shared_core = OpenVINOSharedCore::GetInstance();
  std::string device = shared_core->GetDevice();
  element_type_ = litert::openvino::MapLiteTypeToOV(tensor_type.element_type);

  std::vector<size_t> ov_shape_vec(tensor_type.layout.rank);
  for (size_t i = 0; i < ov_shape_vec.size(); i++)
    ov_shape_vec[i] = tensor_type.layout.dimensions[i];
  shape_ = ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()};

#if defined(LITERT_WINDOWS_OS)
  // Cross-device shared allocation: only when the model spans both NPU and GPU.
  // Do NOT fall back to a host tensor on failure -- the NPU must use the exact
  // same shared allocation as the GPU so any import problem surfaces loudly and
  // both devices stay aligned for debugging.
  if (shared_core->SpansNpuAndGpu()) {
    auto shared = litert::openvino::D3D12SharedBuffer::Create(size);
    if (!shared) {
      return litert::Unexpected(shared.Error());
    }
    shared_ = std::move(*shared);
    shared_path_ = true;
    allocated_ = true;
    return {};
  }
#endif  // LITERT_WINDOWS_OS

  if (device == "NPU" || device == "GPU") {
    auto context = shared_core->GetRemoteContext();
    host_tensor_ = context.create_host_tensor(element_type_, shape_);
  } else if (device == "CPU") {
    host_tensor_ = ov::Tensor(element_type_, shape_);
  } else {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Unsupported OpenVINO device: " + device);
  }
  allocated_ = true;

  return {};
}

litert::Expected<void*> OpenVinoTensorBuffer::GetTensorData() {
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The tensor didn't allocate.");
  }
#if defined(LITERT_WINDOWS_OS)
  if (shared_path_) {
    // The mapped D3D12 resource is host-visible; its pointer never throws
    // (unlike ov::RemoteTensor::data()).
    return shared_->cpu_ptr();
  }
#endif  // LITERT_WINDOWS_OS
  return host_tensor_.data();
}

litert::Expected<void*> OpenVinoTensorBuffer::Lock(
    LiteRtTensorBufferLockMode mode) {
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The tensor didn't allocate.");
  }
#if defined(LITERT_WINDOWS_OS)
  if (shared_path_) {
    last_lock_mode_ = mode;
    // A read (or read-write) lock needs the device's latest bytes copied into
    // the CPU-visible staging resource first.
    if (mode == kLiteRtTensorBufferLockModeRead ||
        mode == kLiteRtTensorBufferLockModeReadWrite) {
      LITERT_RETURN_IF_ERROR(shared_->SyncToHost());
    }
    return shared_->cpu_ptr();
  }
#endif  // LITERT_WINDOWS_OS
  return host_tensor_.data();
}

litert::Expected<void> OpenVinoTensorBuffer::Unlock() {
#if defined(LITERT_WINDOWS_OS)
  if (shared_path_) {
    // Push CPU writes back to the device-local shared resource.
    if (last_lock_mode_ == kLiteRtTensorBufferLockModeWrite ||
        last_lock_mode_ == kLiteRtTensorBufferLockModeReadWrite) {
      LITERT_RETURN_IF_ERROR(shared_->SyncFromHost());
    }
  }
#endif  // LITERT_WINDOWS_OS
  return {};
}

litert::Expected<ov::Tensor> OpenVinoTensorBuffer::GetOVTensor(
    const std::string& device) {
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Failed to get tensor.");
  }

#if defined(LITERT_WINDOWS_OS)
  if (shared_path_) {
    if (auto it = device_imports_.find(device); it != device_imports_.end()) {
      return it->second;
    }
    try {
      auto* shared_core = OpenVINOSharedCore::GetInstance();
      void* handle = shared_->nt_handle();
      ov::Tensor imported;
      if (device == "NPU") {
        // Import the shared NT handle as a Level Zero SHARED_BUF tensor usable
        // as both input and output (BINDED).
        ov::AnyMap params = {
            {ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::SHARED_BUF},
            {ov::intel_npu::mem_handle.name(),
             static_cast<ov::intel_npu::npu_handle_param>(handle)},
            {ov::intel_npu::tensor_type.name(),
             ov::intel_npu::TensorType::BINDED},
        };
        imported = shared_core->GetRemoteContext(device).create_tensor(
            element_type_, shape_, params);
      } else if (device == "GPU") {
        // Import the shared NT handle via the GPU BUFFER_FROM_HANDLE path.
        ov::AnyMap params = {
            {ov::intel_gpu::shared_mem_type.name(),
             ov::intel_gpu::SharedMemType::BUFFER_FROM_HANDLE},
            {ov::intel_gpu::os_handle.name(),
             static_cast<ov::intel_gpu::os_handle_param>(handle)},
        };
        imported = shared_core->GetRemoteContext(device).create_tensor(
            element_type_, shape_, params);
      } else if (device == "CPU") {
        // A CPU partition reads/writes the shared allocation directly through
        // its host mapping (no remote import / remote context needed).
        imported = ov::Tensor(element_type_, shape_, shared_->cpu_ptr());
      } else {
        return litert::Unexpected(
            kLiteRtStatusErrorInvalidArgument,
            "Unsupported device for shared OpenVINO tensor: " + device);
      }
      auto [it, inserted] = device_imports_.emplace(device, imported);
      return it->second;
    } catch (const ov::Exception& e) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, e.what());
    }
  }
#endif  // LITERT_WINDOWS_OS

  // Single-device path: the host tensor is device independent.
  (void)device;
  return host_tensor_;
}
