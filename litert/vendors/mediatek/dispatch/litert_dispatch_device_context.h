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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "neuron/api/NeuronAdapter.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

class LiteRtDispatchDeviceContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchDeviceContextT>;
  struct NeuronMemoryInfo {
    NeuronMemory* neuron_memory;
    size_t size;
    size_t offset;
  };

  ~LiteRtDispatchDeviceContextT();

  static litert::Expected<Ptr> Create(
      const LiteRtRuntimeContext* runtime_context,
      const litert::mediatek::NeuronAdapterApi& neuron_adapter_api);

  litert::Expected<LiteRtTensorBufferHandle> RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer);

  litert::Expected<void> UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle) {
    return neuron_memory_registry_.Unregister(tensor_buffer_handle);
  }

  litert::Expected<NeuronMemoryInfo> GetNeuronMemoryInfo(
      LiteRtTensorBufferHandle tensor_buffer_handle) {
    auto record = neuron_memory_registry_.Find(tensor_buffer_handle);
    if (!record) {
      return record.Error();
    } else {
      return NeuronMemoryInfo(**record);
    }
  }

  // Returns device memory holding a copy of the `weight_size` bytes at
  // `weight_data`, zero-padded to `padded_size` bytes if that is larger. The
  // memory is allocated and filled on the first request for a given
  // `(weight_data, padded_size)` and reused afterwards, so subgraphs that share
  // a weight buffer also share its device copy. `weight_data` must stay valid
  // for the lifetime of this device context.
  //
  // `fd` is the descriptor of the file mapping that holds `weight_data`, or a
  // negative value if it is not file-backed. When it is file-backed, the
  // source pages are released while they are copied; see
  // `litert::mediatek::CopyAndReleaseFileBackedPages`.
  litert::Expected<NeuronMemoryInfo> GetOrCreateSharedWeightMemory(
      int fd, const void* weight_data, size_t weight_size, size_t padded_size);

  const LiteRtRuntimeContext* runtime_context() const {
    return runtime_context_;
  }

 private:
  class NeuronMemoryRegistry {
   public:
    explicit NeuronMemoryRegistry(
        const litert::mediatek::NeuronAdapterApi& neuron_adapter_api)
        : neuron_adapter_api_(neuron_adapter_api) {}
    ~NeuronMemoryRegistry();
    LiteRtTensorBufferHandle Register(NeuronMemory* neuron_memory, size_t size,
                                      size_t offset);
    litert::Expected<void> Unregister(
        LiteRtTensorBufferHandle tensor_buffer_handle);
    litert::Expected<NeuronMemoryInfo*> Find(
        LiteRtTensorBufferHandle tensor_buffer_handle);

   private:
    const litert::mediatek::NeuronAdapterApi& neuron_adapter_api_;
    std::vector<NeuronMemoryInfo> records_;
  };

  // A device copy of a shared weight buffer that this context owns.
  struct SharedWeight {
    LiteRtTensorBuffer tensor_buffer;
    LiteRtTensorBufferHandle handle;
  };

  explicit LiteRtDispatchDeviceContextT(
      const LiteRtRuntimeContext* runtime_context,
      const litert::mediatek::NeuronAdapterApi& neuron_adapter_api)
      : runtime_context_(runtime_context),
        neuron_adapter_api_(neuron_adapter_api),
        neuron_memory_registry_(neuron_adapter_api) {}

  const LiteRtRuntimeContext* runtime_context_;
  const litert::mediatek::NeuronAdapterApi& neuron_adapter_api_;
  NeuronMemoryRegistry neuron_memory_registry_;
  // Keyed by the host address of the weights and the allocated size.
  absl::flat_hash_map<std::pair<const void*, size_t>, SharedWeight>
      shared_weights_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
