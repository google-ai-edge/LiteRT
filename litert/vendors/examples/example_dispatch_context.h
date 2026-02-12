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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_EXAMPLE_DISPATCH_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_EXAMPLE_DISPATCH_CONTEXT_H_

#include <list>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/vendors/examples/example_common.h"

// Forward declare for friend/usage if needed, or just define it here.
// Since LiteRtDispatchDeviceContext is struct LiteRtDispatchDeviceContextT*,
// we define the struct/class here.

class LiteRtDispatchDeviceContextT {
 public:
  using Buffer = ::litert::example::Data;
  using BufferHandle = Buffer*;

  LiteRtDispatchDeviceContextT() = default;
  ~LiteRtDispatchDeviceContextT() = default;

  ::litert::Expected<BufferHandle> RegisterBuffer(LiteRtTensorBuffer b) {
    auto* handle = &buffers_.emplace_back();
    registered_buffers_[handle] = b;
    return handle;
  }

  ::litert::Expected<void> UnregisterBuffer(BufferHandle handle) {
    registered_buffers_.erase(handle);
    for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
      if (&*it == handle) {
        buffers_.erase(it);
        break;
      }
    }
    return {};
  }

  ::litert::TensorBuffer Lookup(BufferHandle handle) {
    return ::litert::TensorBuffer::WrapCObject(registered_buffers_[handle],
                                               ::litert::OwnHandle::kNo);
  }

  void SetCustomOpAsset(absl::string_view custom_op_asset) {
    custom_op_asset_ = custom_op_asset;
  }

  absl::string_view GetCustomOpAsset() const { return custom_op_asset_; }

 private:
  using RegistredBuffers =
      absl::flat_hash_map<BufferHandle, LiteRtTensorBuffer>;
  std::list<Buffer> buffers_;
  RegistredBuffers registered_buffers_;
  absl::string_view custom_op_asset_;
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_EXAMPLE_DISPATCH_CONTEXT_H_
