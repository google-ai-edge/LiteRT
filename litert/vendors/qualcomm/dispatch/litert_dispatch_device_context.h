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
#include <string>
#include <utility>
#include <vector>

#include "QnnCommon.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
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

  ~LiteRtDispatchDeviceContextT() = default;

  static litert::Expected<Ptr> Create(
      const LiteRtRuntimeContext* runtime_context,
      litert::qnn::QnnManager& qnn_manager, ::qnn::QnnBackend& qnn_backend);

  // Sets the dispatch functions that may be invoked on this device context.
  // When non-empty, QNN contexts are created with only the active graphs
  // enabled (QNN_CONTEXT_CONFIG_ENABLE_GRAPHS), so that the other graphs of a
  // multi-graph context binary are not loaded. Must be called before any QNN
  // context is created.
  litert::Expected<void> SetActiveFunctions(
      absl::flat_hash_set<std::string> active_functions);

  litert::Expected<LiteRtTensorBufferHandle> RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer) {
    return tensor_buffer_registry_.Register(
        TensorBufferRegistryEntry(tensor_buffer));
  }

  litert::Expected<void> UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle) {
    return tensor_buffer_registry_.Unregister(tensor_buffer_handle);
  }

  litert::Expected<void> UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle,
      const Qnn_Tensor_t& tensor);

  litert::Expected<LiteRtTensorBuffer> GetTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<Qnn_MemHandle_t> GetMemHandle(
      LiteRtTensorBufferHandle tensor_buffer_handle,
      const Qnn_Tensor_t& tensor);

  void SetInvocationContext(
      LiteRtDispatchInvocationContextT* invocation_context) {
    invocation_context_ = invocation_context;
  }

  // Returns a QNN context, created from the given context binary, in which
  // graph `function_name` is enabled. `binary_graphs` are the names of all the
  // graphs stored in the context binary. Contexts are cached and shared by all
  // the functions of the same binary.
  litert::Expected<const litert::qnn::QnnManager::ContextHandle&>
  GetOrCreateContext(const void* bytecode_ptr, size_t bytecode_size,
                     absl::Span<const std::string> binary_graphs,
                     absl::string_view function_name,
                     Qnn_ProfileHandle_t profile_handle);

  const LiteRtRuntimeContext* runtime_context() const {
    return runtime_context_;
  }

 private:
  struct TensorBufferRegistryEntry {
    LiteRtTensorBuffer tensor_buffer;
    Qnn_MemHandle_t qnn_mem_handle = nullptr;
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
      LiteRtTensorBuffer tensor_buffer, const Qnn_Tensor_t& tensor);

  const LiteRtRuntimeContext* runtime_context_;
  litert::qnn::QnnManager& qnn_manager_;
  ::qnn::QnnBackend& qnn_backend_;
  // Empty means that all functions are active.
  absl::flat_hash_set<std::string> active_functions_;
  TensorBufferRegistry tensor_buffer_registry_;
  LiteRtDispatchInvocationContextT* invocation_context_ = nullptr;

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
  struct CachedContext {
    std::unique_ptr<litert::qnn::QnnManager::ContextHandle> handle;
    // Graphs enabled in the context. Empty means that all graphs are enabled.
    absl::flat_hash_set<std::string> enabled_graphs;
  };
  // Lifetime of the context cache is the same as the device context. There is
  // normally a single context per binary. An extra context is only created if
  // a function that was not declared active gets invoked.
  absl::flat_hash_map<ContextCacheKey, std::vector<CachedContext>>
      context_cache_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
