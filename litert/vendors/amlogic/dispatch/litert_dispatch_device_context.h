// Copyright 2026 Amlogic, Inc.
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

#ifndef ODML_LITERT_LITERT_VENDORS_AMLOGIC_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_AMLOGIC_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_

#include <cstdint>
#include <memory>

#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/amlogic/dispatch/registry.h"

class LiteRtDispatchDeviceContextT {
public:
  using Ptr = std::unique_ptr<LiteRtDispatchDeviceContextT>;

  ~LiteRtDispatchDeviceContextT();

  /**
   * @brief Create a DeviceContext (tensor-buffer registry).
   */
  static litert::Expected<Ptr> Create();

  litert::Expected<LiteRtTensorBufferHandle> RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer);

  litert::Expected<void> UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<LiteRtTensorBuffer> GetTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

private:
  struct TensorBufferRegistryEntry {
    LiteRtTensorBuffer tensor_buffer;
    explicit TensorBufferRegistryEntry(LiteRtTensorBuffer tensor_buffer_)
        : tensor_buffer(tensor_buffer_) {}
    bool operator==(const TensorBufferRegistryEntry &other) const {
      return tensor_buffer == other.tensor_buffer;
    }
  };

  using TensorBufferRegistry =
      litert::amlogic::Registry<LiteRtTensorBufferHandle, TensorBufferRegistryEntry>;

  LiteRtDispatchDeviceContextT() = default;

  TensorBufferRegistry tensor_buffer_registry_;
};

#endif // ODML_LITERT_LITERT_VENDORS_AMLOGIC_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
