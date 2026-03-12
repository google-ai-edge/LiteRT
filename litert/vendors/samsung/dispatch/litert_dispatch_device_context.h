// Copyright 2024 Google LLC.
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
//
// Copyright (C) 2026 Samsung Electronics Co. LTD. 
// SPDX-License-Identifier: Apache-2.0


#ifndef LITERT_VENDORS_SAMSUNG_DISPATCH_DEVICE_CONTEXT_H_
#define LITERT_VENDORS_SAMSUNG_DISPATCH_DEVICE_CONTEXT_H_

#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/samsung/dispatch/enn_manager.h"
#include "litert/vendors/samsung/dispatch/enn_type.h"

class LiteRtDispatchDeviceContextT {
public:
  using UniquePtr = std::unique_ptr<LiteRtDispatchDeviceContextT>;

  ~LiteRtDispatchDeviceContextT() = default;

  static litert::Expected<LiteRtDispatchDeviceContextT::UniquePtr>
  Create(const litert::samsung::EnnManager *enn_manager);

  litert::Expected<LiteRtTensorBufferHandle>
  RegisterTensorBuffer(LiteRtTensorBuffer tensor_buffer);

  litert::Expected<void>
  UnregisterTensorBuffer(LiteRtTensorBufferHandle tensor_buffer_handle) {
    return tensor_buffer_registry_.Unregister(tensor_buffer_handle);
  }

  litert::Expected<EnnBufferPtr>
  GetEnnBuffer(LiteRtTensorBufferHandle tensor_buffer_handle) {
    return tensor_buffer_registry_.Find(tensor_buffer_handle);
  }

private:
  class EnnBufferRegistry {
  public:
    explicit EnnBufferRegistry(const litert::samsung::EnnManager *enn_manager)
        : enn_manager_(enn_manager) {}

    ~EnnBufferRegistry();
    LiteRtTensorBufferHandle Register(EnnBufferPtr enn_buffer);

    litert::Expected<void>
    Unregister(LiteRtTensorBufferHandle tensor_buffer_handle);

    litert::Expected<EnnBufferPtr>
    Find(LiteRtTensorBufferHandle tensor_buffer_handle);

  private:
    const litert::samsung::EnnManager *enn_manager_;
    std::vector<EnnBufferPtr> buffers_;
  };

  LiteRtDispatchDeviceContextT(const litert::samsung::EnnManager *enn_manager);

  const litert::samsung::EnnManager *enn_manager_;
  EnnBufferRegistry tensor_buffer_registry_;
};

#endif // LITERT_VENDORS_SAMSUNG_DISPATCH_DEVICE_CONTEXT_H_
