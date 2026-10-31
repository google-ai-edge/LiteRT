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

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "QnnCommon.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/dispatch/registry.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

class LiteRtDispatchDeviceContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchDeviceContextT>;

  // Deregisters the QNN memory handles that are still registered against QNN
  // contexts owned by this device context (they are freed right after).
  ~LiteRtDispatchDeviceContextT();

  static litert::Expected<Ptr> Create(
      const LiteRtRuntimeContext* runtime_context,
      litert::qnn::QnnManager& qnn_manager, ::qnn::QnnBackend& qnn_backend);

  litert::Expected<LiteRtTensorBufferHandle> RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer) {
    return tensor_buffer_registry_.Register(
        TensorBufferRegistryEntry(tensor_buffer));
  }

  // Unregisters the tensor buffer and releases its QNN memory handles, one per
  // QNN context it was attached to.
  litert::Expected<void> UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<LiteRtTensorBuffer> GetTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

  // Returns the QNN memory handle of the tensor buffer registered against
  // `context_handle`, i.e. the QNN context of the invocation context the
  // buffer is attached to, registering the buffer with that context on first
  // use. QNN memory handles are per context, and contexts of one model may
  // live in different HTP process domains (PDs), so a buffer shared by several
  // contexts gets one memory handle per context.
  litert::Expected<Qnn_MemHandle_t> GetMemHandle(
      LiteRtTensorBufferHandle tensor_buffer_handle, const Qnn_Tensor_t& tensor,
      Qnn_ContextHandle_t context_handle);

  litert::Expected<const litert::qnn::QnnManager::ContextHandle&>
  GetOrCreateContext(const void* bytecode_ptr, size_t bytecode_size,
                     Qnn_ProfileHandle_t profile_handle);

  const LiteRtRuntimeContext* runtime_context() const {
    return runtime_context_;
  }

 private:
  // A QNN memory handle of a tensor buffer, registered against one QNN
  // context.
  struct QnnMemHandleEntry {
    Qnn_ContextHandle_t context_handle = nullptr;
    Qnn_MemHandle_t mem_handle = nullptr;
    // False when registering against `context_handle` failed and the handle
    // registered against another context is reused instead (this only works
    // when both contexts are in the same HTP PD); such an entry never
    // deregisters `mem_handle` itself.
    bool owned = true;
  };

  struct TensorBufferRegistryEntry {
    LiteRtTensorBuffer tensor_buffer;
    std::vector<QnnMemHandleEntry> qnn_mem_handles;
    explicit TensorBufferRegistryEntry(LiteRtTensorBuffer tensor_buffer_)
        : tensor_buffer(tensor_buffer_) {}
    bool operator==(const TensorBufferRegistryEntry& other) const {
      return tensor_buffer == other.tensor_buffer;
    }
  };

  using TensorBufferRegistry = litert::qnn::Registry<LiteRtTensorBufferHandle,
                                                     TensorBufferRegistryEntry>;

  explicit LiteRtDispatchDeviceContextT(
      const LiteRtRuntimeContext* runtime_context,
      litert::qnn::QnnManager& qnn_manager, ::qnn::QnnBackend& qnn_backend)
      : runtime_context_(runtime_context),
        qnn_manager_(qnn_manager),
        qnn_backend_(qnn_backend) {}

  litert::Expected<Qnn_MemHandle_t> RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer, const Qnn_Tensor_t& tensor,
      Qnn_ContextHandle_t context_handle);

  // Deregisters the memory handles of `entry` that were registered against
  // `context_handle` (or against any context if `context_handle` is null) and
  // removes them, together with the entries that reuse them, from `entry`.
  // Returns the first QNN error.
  litert::Expected<void> DeregisterMemHandles(
      TensorBufferRegistryEntry& entry, Qnn_ContextHandle_t context_handle);

  bool OwnsContext(Qnn_ContextHandle_t context_handle) const;

  const LiteRtRuntimeContext* runtime_context_;
  litert::qnn::QnnManager& qnn_manager_;
  ::qnn::QnnBackend& qnn_backend_;
  TensorBufferRegistry tensor_buffer_registry_;

  struct ContextCacheKey {
    const void* ptr;
    size_t size;

    bool operator==(const ContextCacheKey& other) const {
      return ptr == other.ptr && size == other.size;
    }
    template <typename H>
    friend H AbslHashValue(H h, const ContextCacheKey& k) {
      return H::combine(std::move(h), k.ptr, k.size);
    }
  };
  using UniqueContextHandle =
      std::unique_ptr<litert::qnn::QnnManager::ContextHandle>;
  // Lifetime of the context cache is the same as the device context.
  absl::flat_hash_map<ContextCacheKey, UniqueContextHandle> context_cache_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
