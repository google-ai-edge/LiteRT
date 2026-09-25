/*******************************************************************************
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "litert/vendors/aml/dispatch/litert_dispatch_device_context.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/strings/str_format.h" // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "adla_log.h"

using litert::Expected;
using litert::Unexpected;

namespace {

/**
 * @brief Process-global shared ADLA registry.
 *
 * LiteRT may call DeviceContextCreate once per TFLite subgraph. Keeping the
 * map here (instead of on DeviceContext) lets partition 1..N-1 Acquire the
 * ADLA loaded by partition 0 without changing libLiteRt.so.
 */
std::unordered_map<std::string, std::shared_ptr<AmlSharedAdlaModel>>&
GlobalSharedAdlaModels() {
  static std::unordered_map<std::string, std::shared_ptr<AmlSharedAdlaModel>>
      models;
  return models;
}

}  // namespace

LiteRtDispatchDeviceContextT::~LiteRtDispatchDeviceContextT() = default;

Expected<LiteRtDispatchDeviceContextT::Ptr> LiteRtDispatchDeviceContextT::Create()
{
  LITERT_LOG(LITERT_DEBUG, "[AML] LiteRtDispatchDeviceContextT Enter");
  return Ptr(new LiteRtDispatchDeviceContextT());
}

std::string LiteRtDispatchDeviceContextT::MakeAdlaModelKey(
    const std::string& model_path, const std::string& model_names)
{
  // Key uses raw model_path from dispatch_info (empty stays empty).
  return model_path + "|" + model_names;
}

std::shared_ptr<AmlSharedAdlaModel>
LiteRtDispatchDeviceContextT::FindSharedAdlaModel(
    const std::string& model_key) const
{
  auto& models = GlobalSharedAdlaModels();
  auto it = models.find(model_key);
  if (it == models.end())
  {
    return nullptr;
  }
  return it->second;
}

std::shared_ptr<AmlSharedAdlaModel>
LiteRtDispatchDeviceContextT::RegisterSharedAdlaModel(
    std::shared_ptr<AmlSharedAdlaModel> model)
{
  if (model == nullptr || model->model_key.empty())
  {
    return nullptr;
  }
  auto& models = GlobalSharedAdlaModels();
  auto it = models.find(model->model_key);
  if (it != models.end())
  {
    // Another InvocationContext already loaded this key; share it.
    it->second->refcount += 1;
    LITERT_LOG(LITERT_INFO,
               "[AML Dispatch] shared ADLA already registered key=%s "
               "refcount=%d",
               model->model_key.c_str(), it->second->refcount);
    return it->second;
  }
  model->refcount = 1;
  models[model->model_key] = model;
  LITERT_LOG(LITERT_INFO,
             "[AML Dispatch] registered shared ADLA key=%s embed=%d "
             "path=%s size=%zu",
             model->model_key.c_str(), model->use_adla_memory ? 1 : 0,
             model->model_file_path.c_str(), model->adla_bin_size);
  return model;
}

std::shared_ptr<AmlSharedAdlaModel>
LiteRtDispatchDeviceContextT::AcquireSharedAdlaModel(
    const std::string& model_key)
{
  auto& models = GlobalSharedAdlaModels();
  auto it = models.find(model_key);
  if (it == models.end())
  {
    return nullptr;
  }
  it->second->refcount += 1;
  LITERT_LOG(LITERT_INFO,
             "[AML Dispatch] acquire shared ADLA key=%s refcount=%d",
             model_key.c_str(), it->second->refcount);
  return it->second;
}

void LiteRtDispatchDeviceContextT::ReleaseSharedAdlaModel(
    const std::string& model_key, tflite::AML_NN* aml_nn)
{
  auto& models = GlobalSharedAdlaModels();
  auto it = models.find(model_key);
  if (it == models.end())
  {
    return;
  }
  auto& model = it->second;
  if (model->refcount > 0)
  {
    model->refcount -= 1;
  }
  LITERT_LOG(LITERT_INFO,
             "[AML Dispatch] release shared ADLA key=%s refcount=%d",
             model_key.c_str(), model->refcount);
  if (model->refcount > 0)
  {
    return;
  }

  // Last holder: destroy the shared NPU context and drop the registry entry.
  if (model->qcontext != nullptr && aml_nn != nullptr)
  {
    ADLA_LOGI("[AML Dispatch] amlnn_destroy (shared)");
    if (aml_nn->nnsdk2_func_ptr.destroy != nullptr)
    {
      aml_nn->nnsdk2_func_ptr.destroy(model->qcontext);
    }
    if (aml_nn->qcontext == model->qcontext)
    {
      aml_nn->qcontext = nullptr;
    }
    model->qcontext = nullptr;
  }
  models.erase(it);
}

Expected<LiteRtTensorBuffer> LiteRtDispatchDeviceContextT::GetTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle)
{
  LITERT_LOG(LITERT_DEBUG, "[AML] GetTensorBuffer Enter");
  auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
  if (!registry_entry)
  {
    return Unexpected(registry_entry.Error());
  }

  return (*registry_entry)->tensor_buffer;
}

// Expected<void *> LiteRtDispatchDeviceContextT::GetMemHandle(
//     LiteRtTensorBufferHandle tensor_buffer_handle)
// {
//   auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
//   if (!registry_entry)
//   {
//     return Unexpected(registry_entry.Error());
//   }

//   return (*registry_entry)->tensor_buffer->data;
// }

Expected<LiteRtTensorBufferHandle> LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer)
{
  LITERT_LOG(LITERT_DEBUG, "Registering tensor buffer %p", tensor_buffer);
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status =
          LiteRtGetTensorBufferType(tensor_buffer, &tensor_buffer_type);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer type");
  }
  // std::cout << "LiteRtDispatchDeviceContextT::RegisterTensorBuffer buffer type = " << tensor_buffer_type << std::endl;

  size_t tensor_buffer_size;
  if (auto status =
          LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer size");
  }
  // std::cout << "LiteRtDispatchDeviceContextT::RegisterTensorBuffer buffer size = " << tensor_buffer_size << std::endl;

  size_t tensor_buffer_offset;
  if (auto status =
          LiteRtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer offset");
  }

  LiteRtRankedTensorType tensor_type;
  if (auto status =
          LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer's type");
  }

  auto element_type =
      static_cast<enum litert::ElementType>(tensor_type.element_type);

  // std::cout << "LiteRtDispatchDeviceContextT::RegisterTensorBuffer element_type = " << static_cast<int>(element_type) << std::endl;

  uint32_t tensor_rank = tensor_type.layout.rank;
  uint32_t *tensor_dimensions = reinterpret_cast<uint32_t *>(
      const_cast<int32_t *>(tensor_type.layout.dimensions));
  if (tensor_type.layout.has_strides)
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported by QNN");
  }

  void *buffer_host_addr;
  int buffer_fd;
  (void)buffer_host_addr;

  switch (tensor_buffer_type)
  {
  case kLiteRtTensorBufferTypeHostMemory:
    if (auto status =
            LiteRtGetTensorBufferHostMemory(tensor_buffer, &buffer_host_addr);
        status != kLiteRtStatusOk)
    {
      return Unexpected(status, "Failed to get host memory buffer");
    }
    break;

  case kLiteRtTensorBufferTypeFastRpc:
#if LITERT_HAS_FASTRPC_SUPPORT
    if (auto status = LiteRtGetTensorBufferFastRpcBuffer(
            tensor_buffer, &buffer_host_addr, &buffer_fd);
        status != kLiteRtStatusOk)
    {
      return Unexpected(status, "Failed to get FastRPC buffer");
    }
#else
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "FastRPC support is missing on this platform");
#endif // LRT_HAS_FASTRPC_SUPPORT
    break;

  case kLiteRtTensorBufferTypeDmaBuf:
#if LITERT_HAS_DMABUF_SUPPORT
    if (auto status = LiteRtGetTensorBufferDmaBufBuffer(
            tensor_buffer, &buffer_host_addr, &buffer_fd);
        status != kLiteRtStatusOk)
    {
      return Unexpected(status, "Failed to get DMA-BUF buffer");
    }
#else
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "DmaBuf support is missing on this platform");
#endif // LRT_HAS_DMABUF_SUPPORT
    break;

  default:
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unsupported tensor buffer type");
  }

  if (invocation_context_ == nullptr)
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Missing invocation context");
  }
  // std::cout << "buffer_host_addr = " << buffer_host_addr << std::endl;
  // std::cout << "buffer_fd = " << buffer_fd << std::endl;

  return tensor_buffer_registry_.Register(TensorBufferRegistryEntry(tensor_buffer));
}

litert::Expected<void> LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle)
{
  LITERT_ASSIGN_OR_RETURN(auto tensor_buffer,
                          GetTensorBuffer(tensor_buffer_handle));
  LITERT_LOG(LITERT_DEBUG, "Unregistering tensor buffer %p", tensor_buffer);
  LITERT_RETURN_IF_ERROR(
      tensor_buffer_registry_.Unregister(tensor_buffer_handle));
  // LITERT_ASSIGN_OR_RETURN(auto mem_handle,
  //                         GetMemHandle(tensor_buffer_handle, tensor));

  return {};
}
