// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "openvino/runtime/tensor.hpp"
#if !LITERT_WINDOWS_OS
#include <sys/mman.h>
#endif

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"

class LiteRtDispatchDeviceContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchDeviceContextT>;

  ~LiteRtDispatchDeviceContextT() = default;
  static litert::Expected<Ptr> Create();
  litert::Expected<LiteRtTensorBufferHandle> RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer);

  litert::Expected<void> UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

#if defined(LITERT_WINDOWS_OS)
  litert::Expected<ov::intel_npu::level_zero::ZeroBufferTensor> getOvTensor(
#else
  litert::Expected<ov::Tensor> getOvTensor(
#endif
      const LiteRtTensorBufferHandle& handle) const {
    auto it = tensor_handle_map_.find(handle);
    if (it != tensor_handle_map_.end()) {
      return it->second;
    } else {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Failed to get Remote Tensor");
    }
  }

  // Return the core shared_pointer.
  std::shared_ptr<ov::Core> getCore() const { return core_; }

 private:
  explicit LiteRtDispatchDeviceContextT()
      : core_(std::make_shared<ov::Core>()), next_handle_(0) {}
  std::shared_ptr<ov::Core> core_;
#if defined(LITERT_WINDOWS_OS)
  std::unordered_map<LiteRtTensorBufferHandle,
                     ov::intel_npu::level_zero::ZeroBufferTensor>
      tensor_handle_map_;
#else
  std::unordered_map<LiteRtTensorBufferHandle, ov::Tensor> tensor_handle_map_;
#endif
  uint64_t next_handle_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
