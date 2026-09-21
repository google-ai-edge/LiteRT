// Copyright (C) 2023 Amlogic, Inc. All rights reserved.
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

#ifndef ODML_LITERT_LITERT_VENDORS_AML_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_AML_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/aml/dispatch/registry.h"
#include "litert/vendors/aml/core/nnsdk_func.h"

class LiteRtDispatchInvocationContextT;

/**
 * @brief Shared ADLA runtime state for multi-partition / multi-subgraph models.
 *
 * Offline packs embed @c adla_bin only in @c dispatch_info[0]. All
 * InvocationContexts that share the same @c model_path|model_names key reuse
 * one @c amlnn_init result and bind IO by @c subgraph_idx.
 *
 * @note The registry is process-global (see device_context.cc), not owned by a
 *       single DeviceContext. LiteRT may create one DeviceContext per TFLite
 *       subgraph; a global map keeps sharing without changing libLiteRt.so.
 *
 * @var model_key       Key = model_path + "|" + model_names.
 * @var adla_bin        Embedded ADLA bytes (may be empty for path-based load).
 * @var adla_bin_size   Valid byte count of @c adla_bin.
 * @var qcontext        Shared NNSDK2 context from @c amlnn_init.
 * @var use_adla_memory True if loaded from memory; false if from file path.
 * @var model_file_path Fallback / diagnostic .adla path.
 * @var subgraph_num    Queried NNSDK2 subgraph count (AMLNN_QUERY_SUBGRAPH_NUM).
 * @var refcount        Number of live InvocationContexts holding this entry.
 */
struct AmlSharedAdlaModel {
  std::string model_key;
  std::vector<uint8_t> adla_bin;
  size_t adla_bin_size = 0;
  void* qcontext = nullptr;
  bool use_adla_memory = false;
  std::string model_file_path;
  int subgraph_num = 1;
  int refcount = 0;
};

class LiteRtDispatchDeviceContextT
{
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

  void SetInvocationContext(LiteRtDispatchInvocationContextT *invocation_context)
  {
    invocation_context_ = invocation_context;
  }

  void ClearInvocationContext(LiteRtDispatchInvocationContextT *invocation_context)
  {
    if (invocation_context_ == invocation_context)
    {
      invocation_context_ = nullptr;
    }
  }

  /**
   * @brief Lookup a shared ADLA entry without changing refcount.
   * @param model_key Key from MakeAdlaModelKey().
   * @return Shared entry, or nullptr if not registered.
   */
  std::shared_ptr<AmlSharedAdlaModel> FindSharedAdlaModel(
      const std::string& model_key) const;

  /**
   * @brief Register a newly loaded shared ADLA (refcount starts at 1).
   * @param model Pending shared state after successful amlnn_init.
   * @return Registered entry; if key exists, bumps refcount and returns it.
   */
  std::shared_ptr<AmlSharedAdlaModel> RegisterSharedAdlaModel(
      std::shared_ptr<AmlSharedAdlaModel> model);

  /**
   * @brief Acquire an existing shared ADLA and bump refcount.
   * @param model_key Key from MakeAdlaModelKey().
   * @return Shared entry, or nullptr if not registered yet.
   */
  std::shared_ptr<AmlSharedAdlaModel> AcquireSharedAdlaModel(
      const std::string& model_key);

  /**
   * @brief Drop one reference; destroy NPU context when refcount hits 0.
   * @param model_key Key previously acquired / registered.
   * @param aml_nn    NNSDK2 function table used for amlnn_destroy.
   */
  void ReleaseSharedAdlaModel(const std::string& model_key,
                              tflite::AML_NN* aml_nn);

  /**
   * @brief Build process-global shared-model key.
   * @param model_path Raw dispatch_info.model_path (may be empty).
   * @param model_names dispatch_info.model_names.
   * @return model_path + "|" + model_names (e.g. "|test").
   * @note Default search dirs (cwd, /data/vendor/nn/) are NOT part of the key.
   */
  static std::string MakeAdlaModelKey(const std::string& model_path,
                                      const std::string& model_names);

private:
  struct TensorBufferRegistryEntry
  {
    LiteRtTensorBuffer tensor_buffer;
    explicit TensorBufferRegistryEntry(LiteRtTensorBuffer tensor_buffer_)
        : tensor_buffer(tensor_buffer_) {}
    bool operator==(const TensorBufferRegistryEntry &other) const
    {
      return tensor_buffer == other.tensor_buffer;
    }
  };

  using TensorBufferRegistry =
      litert::aml::Registry<LiteRtTensorBufferHandle, TensorBufferRegistryEntry>;

  LiteRtDispatchDeviceContextT() = default;

  TensorBufferRegistry tensor_buffer_registry_;
  LiteRtDispatchInvocationContextT *invocation_context_ = nullptr;
};

#endif // ODML_LITERT_LITERT_VENDORS_AML_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
