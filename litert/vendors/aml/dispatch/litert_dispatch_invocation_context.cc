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

#include "litert/vendors/aml/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/aml/dispatch/litert_dispatch_device_context.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h" // from @com_google_absl
#include "absl/types/span.h"          // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "adla_log.h"
#include "litert/vendors/aml/compiler/aml_dispatch_info.h"
#include "litert/vendors/aml/core/nnsdk_func.h"

using litert::Expected;
using litert::Unexpected;

void SaveBufferToBin(const void *data, size_t size, const char *filename)
{
  if (!data || size == 0)
  {
    LITERT_LOG(LITERT_WARNING, "SaveBufferToBin: invalid data or size");
    return;
  }

  FILE *fp = fopen(filename, "wb");
  if (!fp)
  {
    LITERT_LOG(LITERT_ERROR, "SaveBufferToBin: fopen failed for %s", filename);
    return;
  }

  size_t written = fwrite(data, 1, size, fp);
  if (written != size)
  {
    LITERT_LOG(LITERT_WARNING,
               "SaveBufferToBin: written size (%zu) != expected size (%zu)",
               written, size);
  }

  fclose(fp);
  LITERT_LOG(LITERT_DEBUG, "SaveBufferToBin: saved to %s (%zu bytes)", filename,
             written);
}

namespace {

// Resolve .adla path for path-load (no embed).
// - If model_path is set: use that directory only.
// - If empty: prefer cwd/<name>.adla, then /data/vendor/nn/<name>.adla.
std::string ResolveAdlaFilePath(const std::string& model_path,
                                const std::string& model_names) {
  const std::string filename = model_names + ".adla";
  if (!model_path.empty()) {
    std::string dir = model_path;
    if (dir.back() != '/') {
      dir += '/';
    }
    return dir + filename;
  }

  const std::array<std::string, 2> candidates = {
      filename,
      std::string("/data/vendor/nn/") + filename,
  };
  for (const auto& path : candidates) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(path, ec)) {
      LITERT_LOG(LITERT_INFO, "[AML Dispatch] resolved ADLA path=%s",
                 path.c_str());
      return path;
    }
  }
  // Keep SDK default as the path passed to amlnn_init when neither exists.
  return std::string("/data/vendor/nn/") + filename;
}

}  // namespace

LiteRtDispatchInvocationContextT::LiteRtDispatchInvocationContextT(
    LiteRtDispatchDeviceContextT &device_context,
    int graph_index, int num_inputs, int num_outputs)
    : device_context_(device_context),
      context_handle_(nullptr),
      profile_handle_(nullptr),
      graph_index_(graph_index),
      graph_handle_(nullptr)
{
  input_buffer_handles_.resize(num_inputs);
  output_buffer_handles_.resize(num_outputs);
  // Index-based (not push_back): LiteRT may Detach+Attach new output buffers
  // every frame (e.g. camera demo). push_back would accumulate and cause
  // "NPU output count 3 < LiteRT output count 6".
  input_buffer_ptr_.assign(num_inputs, nullptr);
  input_buffer_sizes_.assign(num_inputs, 0);
  output_buffer_ptr.assign(num_outputs, nullptr);
  output_buffer_sizes_.assign(num_outputs, 0);
  // 创建nnsdk 的dlopen功能和句柄
  ADLA_LOGI("[AML_DELEGATE_Kernel] InitState");
  int err = 0;
  InitAdlaLogLevel();
  ADLA_LOGI("[AML_DELEGATE_Kernel] Start to open libnnsdk.so");
  aml_nn_ = tflite::AML_NNImplementation();
  if (aml_nn_ != nullptr)
  {
    ADLA_LOGI("[AML_DELEGATE_Kernel] aml_nn create success");
  }
  aml_nn_->qcontext = NULL;
}

LiteRtDispatchInvocationContextT::~LiteRtDispatchInvocationContextT()
{
  device_context_.ClearInvocationContext(this);
  if (!shared_model_key_.empty())
  {
    // Shared multi-subgraph path: drop ref; last holder destroys qcontext.
    device_context_.ReleaseSharedAdlaModel(shared_model_key_, aml_nn_);
    shared_adla_model_.reset();
    shared_model_key_.clear();
    return;
  }

  // Legacy single-owner path (should be rare after shared-model change).
  if (aml_nn_ != nullptr && aml_nn_->qcontext != nullptr)
  {
    ADLA_LOGI("[AML Dispatch] amlnn_destroy");
    if (aml_nn_->nnsdk2_func_ptr.destroy != nullptr)
    {
      aml_nn_->nnsdk2_func_ptr.destroy(aml_nn_->qcontext);
    }
    aml_nn_->qcontext = nullptr;
  }
}

Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    LiteRtDispatchDeviceContextT &device_context,
    const LiteRtMemBuffer *exec_bytecode_buffer, const char *function_name, int num_inputs, int num_outputs)
{
  LITERT_LOG(LITERT_DEBUG, "InvocationContext Create");
  int graph_index = 0;
  auto ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>(
      new LiteRtDispatchInvocationContextT(device_context, graph_index, num_inputs, num_outputs));

  const uint8_t *byte_code_data =
      static_cast<const uint8_t *>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  size_t byte_code_size = exec_bytecode_buffer->size;
  (void)byte_code_size;
  LITERT_LOG(LITERT_DEBUG,
             "[AML] LiteRtDispatchInvocationContext Create Start AML_Dispatch_Info");

  std::shared_ptr<AML_Dispatch_Info> dispatch_info = std::make_shared<AML_Dispatch_Info>(
      DeserializeDispatchInfo(std::string(
          reinterpret_cast<const char *>(byte_code_data), byte_code_size)));

  LITERT_LOG(LITERT_INFO, "function_name = %s", function_name);
  LITERT_LOG(LITERT_INFO,
             "dispatch_info model_names = %s, model_path = %s, graph_names = %s, "
             "subgraph_idx = %d, adla_bin_size = %zu, adla_bin.data_size = %zu",
             dispatch_info->model_names.c_str(), dispatch_info->model_path.c_str(),
             dispatch_info->graph_names.c_str(), dispatch_info->subgraph_idx,
             dispatch_info->adla_bin_size, dispatch_info->adla_bin.size());

  ptr->subgraph_idx_ = dispatch_info->subgraph_idx;
  if (!tflite::IsNnsdk2Available(ptr->aml_nn_))
  {
    LITERT_LOG(LITERT_ERROR, "NNSDK2 symbols not available");
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "NNSDK2 required but not available");
  }

  // File path is used only when this partition is the first loader and has no
  // embed. Shared key uses raw model_path (empty allowed), not the default dir.
  // When model_path is empty, search cwd then /data/vendor/nn/.
  ptr->model_file_path_ =
      ResolveAdlaFilePath(dispatch_info->model_path, dispatch_info->model_names);
  ptr->shared_model_key_ = LiteRtDispatchDeviceContextT::MakeAdlaModelKey(
      dispatch_info->model_path, dispatch_info->model_names);

  const size_t embedded_bin_size =
      dispatch_info->adla_bin_size > 0 ? dispatch_info->adla_bin_size
                                       : dispatch_info->adla_bin.size();
  const bool has_embedded_bin =
      embedded_bin_size > 0 && !dispatch_info->adla_bin.empty() &&
      dispatch_info->adla_bin.size() >= embedded_bin_size;

  // -------------------------------------------------------------------------
  // Multi-subgraph load:
  //   1) Acquire process-global shared ADLA (typical for partitions 1..N-1).
  //   2) Else first-load from embed (offline [0]) or file path, then Register.
  // -------------------------------------------------------------------------
  auto shared = device_context.AcquireSharedAdlaModel(ptr->shared_model_key_);
  if (shared != nullptr)
  {
    ptr->shared_adla_model_ = shared;
    ptr->use_adla_memory_ = shared->use_adla_memory;
    ptr->model_file_path_ = shared->model_file_path;
    ptr->subgraph_num_ = shared->subgraph_num;
    ptr->aml_nn_->qcontext = shared->qcontext;
    LITERT_LOG(LITERT_INFO,
               "[AML Dispatch] reuse shared ADLA key=%s subgraph_idx=%d "
               "subgraph_num=%d",
               ptr->shared_model_key_.c_str(), ptr->subgraph_idx_,
               ptr->subgraph_num_);
    if (auto status = ptr->BindSubgraph(ptr->subgraph_idx_); !status)
    {
      device_context.ReleaseSharedAdlaModel(ptr->shared_model_key_, ptr->aml_nn_);
      ptr->shared_adla_model_.reset();
      ptr->shared_model_key_.clear();
      return Unexpected(status.Error());
    }
    return litert::Expected<Ptr>(std::move(ptr));
  }

  // First loader for this key: keep ADLA bytes in shared_adla_model_ before
  // amlnn_init so the SDK pointer stays valid for the shared lifetime.
  auto pending_shared = std::make_shared<AmlSharedAdlaModel>();
  pending_shared->model_key = ptr->shared_model_key_;
  pending_shared->model_file_path = ptr->model_file_path_;
  if (has_embedded_bin)
  {
    pending_shared->adla_bin = std::move(dispatch_info->adla_bin);
    pending_shared->adla_bin_size = embedded_bin_size;
    pending_shared->use_adla_memory = true;
    ptr->use_adla_memory_ = true;
    ptr->adla_bin_size_ = embedded_bin_size;
    LITERT_LOG(LITERT_INFO, "[AML Dispatch] Using embedded ADLA bin, size=%zu",
               ptr->adla_bin_size_);
  }
  else
  {
    // Offline packs leave adla_bin empty on partitions 1..N-1; they should have
    // hit Acquire above. Miss + path fallback usually means registry miss.
    LITERT_LOG(LITERT_WARNING,
               "[AML Dispatch] shared ADLA miss key=%s subgraph_idx=%d; "
               "falling back to file path=%s",
               ptr->shared_model_key_.c_str(), ptr->subgraph_idx_,
               ptr->model_file_path_.c_str());
    pending_shared->use_adla_memory = false;
    ptr->use_adla_memory_ = false;
  }
  ptr->shared_adla_model_ = pending_shared;

  LITERT_LOG(LITERT_INFO, "[AML Dispatch] NNSDK2 use_adla_memory=%d",
             ptr->use_adla_memory_ ? 1 : 0);

  if (auto status = ptr->CreateModel(ptr->subgraph_idx_); !status)
  {
    if (ptr->aml_nn_->qcontext != nullptr &&
        ptr->aml_nn_->nnsdk2_func_ptr.destroy != nullptr)
    {
      ptr->aml_nn_->nnsdk2_func_ptr.destroy(ptr->aml_nn_->qcontext);
      ptr->aml_nn_->qcontext = nullptr;
    }
    return Unexpected(status.Error());
  }

  pending_shared->qcontext = ptr->aml_nn_->qcontext;
  pending_shared->subgraph_num = ptr->subgraph_num_;
  ptr->shared_adla_model_ =
      device_context.RegisterSharedAdlaModel(std::move(pending_shared));
  if (ptr->shared_adla_model_ == nullptr)
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to register shared ADLA model");
  }

  return litert::Expected<Ptr>(std::move(ptr));
}


namespace
{

  Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
      const LiteRtRankedTensorType &tensor_type)
  {
    if (tensor_type.layout.has_strides)
    {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Tensor strides are not supported by QNN");
    }

    std::vector<LiteRtTensorBufferType> supported_tensor_buffer_types = {
        // Prefer host memory for broader compatibility across builds/devices.
        kLiteRtTensorBufferTypeHostMemory,
    };
#if LITERT_HAS_FASTRPC_SUPPORT
    supported_tensor_buffer_types.push_back(kLiteRtTensorBufferTypeFastRpc);
#endif
#if LITERT_HAS_DMABUF_SUPPORT
    supported_tensor_buffer_types.push_back(kLiteRtTensorBufferTypeDmaBuf);
#endif

    auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
    if (!buffer_size)
    {
      return Unexpected(buffer_size.Error());
    }

    LiteRtTensorBufferRequirements requirements;
    if (auto status = LiteRtCreateTensorBufferRequirements(
            supported_tensor_buffer_types.size(),
            supported_tensor_buffer_types.data(), *buffer_size, /*num_strides=*/0,
            /*strides=*/nullptr, &requirements);
        status != kLiteRtStatusOk)
    {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Not implemented");
    }

    return requirements;
  }

} // namespace

Expected<void> LiteRtDispatchInvocationContextT::CreateModel(int subgraph_idx)
{
  amlnn_init_config init_config;
  memset(&init_config, 0, sizeof(init_config));
  init_config.backend_type = AMLNN_BACKEND_ADLA_NPU;

  // Prefer bytes owned by the pending/shared model (keeps pointer valid).
  const uint8_t* adla_data =
      (shared_adla_model_ != nullptr && !shared_adla_model_->adla_bin.empty())
          ? shared_adla_model_->adla_bin.data()
          : adla_bin_.data();
  const size_t adla_size =
      (shared_adla_model_ != nullptr && shared_adla_model_->adla_bin_size > 0)
          ? shared_adla_model_->adla_bin_size
          : adla_bin_size_;

  void* ctx = nullptr;
  int ret = 0;
  if (use_adla_memory_)
  {
    ret = aml_nn_->nnsdk2_func_ptr.init(
        &ctx, const_cast<uint8_t*>(adla_data),
        static_cast<uint32_t>(adla_size), &init_config);
    if (ret < 0 || ctx == nullptr)
    {
      ADLA_LOGE("amlnn_init fail from memory, size=%zu, ret=%d",
                adla_size, ret);
      return Unexpected(kLiteRtStatusErrorRuntimeFailure, "amlnn_init fail");
    }
  }
  else
  {
    ret = aml_nn_->nnsdk2_func_ptr.init(
        &ctx, const_cast<char*>(model_file_path_.c_str()), 0, &init_config);
    if (ret < 0 || ctx == nullptr)
    {
      ADLA_LOGE("amlnn_init fail, path=%s, ret=%d", model_file_path_.c_str(),
                ret);
      return Unexpected(kLiteRtStatusErrorRuntimeFailure, "amlnn_init fail");
    }
  }
  aml_nn_->qcontext = ctx;

  // Match NNSDK sample: query AMLNN_QUERY_SUBGRAPH_NUM after amlnn_init.
  subgraph_num_ = 1;
  if (tflite::Nnsdk2QuerySubgraphNum(aml_nn_, ctx, &subgraph_num_) < 0)
  {
    ADLA_LOGE("[AML Dispatch] query subgraph num fail");
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "AMLNN_QUERY_SUBGRAPH_NUM fail");
  }
  ADLA_LOGI("[AML Dispatch] NNSDK2 subgraph_num=%d", subgraph_num_);

  return BindSubgraph(subgraph_idx);
}

Expected<void> LiteRtDispatchInvocationContextT::BindSubgraph(int subgraph_idx)
{
  if (aml_nn_ == nullptr || aml_nn_->qcontext == nullptr)
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "AML NPU context is not initialized");
  }

  subgraph_idx_ = subgraph_idx;
  void* ctx = aml_nn_->qcontext;

  // Match NNSDK sample: amlnn_select_subgraph then query IO for that subgraph.
  if (tflite::Nnsdk2SelectSubgraphIfNeeded(aml_nn_, ctx, subgraph_idx_,
                                           subgraph_num_) < 0)
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "amlnn_select_subgraph fail");
  }

  memset(&v2_io_num_, 0, sizeof(v2_io_num_));
  if (aml_nn_->nnsdk2_func_ptr.query(ctx, AMLNN_QUERY_IN_OUT_NUM, &v2_io_num_,
                                     sizeof(v2_io_num_)) != AMLNN_SUCCESS)
  {
    ADLA_LOGE("amlnn_query IN_OUT_NUM fail subgraph_idx=%d", subgraph_idx_);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "amlnn_query IN_OUT_NUM fail");
  }

  ADLA_LOGI("[AML Dispatch] BindSubgraph idx=%d/%d input=%u output=%u",
            subgraph_idx_, subgraph_num_, v2_io_num_.n_input,
            v2_io_num_.n_output);

  v2_inputs_.assign(v2_io_num_.n_input, {});
  for (uint32_t i = 0; i < v2_io_num_.n_input; ++i)
  {
    amlnn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = i;
    if (aml_nn_->nnsdk2_func_ptr.query(ctx, AMLNN_QUERY_INPUT_ATTR, &input_attr,
                                       sizeof(input_attr)) != AMLNN_SUCCESS)
    {
      ADLA_LOGE("amlnn_query INPUT_ATTR[%u] fail", i);
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "amlnn_query INPUT_ATTR fail");
    }
    ADLA_LOGI("[AML Dispatch] NNSDK2 input[%u] name=%s size=%u", i,
              input_attr.name, input_attr.size);
  }

  for (uint32_t i = 0; i < v2_io_num_.n_output; ++i)
  {
    amlnn_tensor_attr output_attr;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = i;
    if (aml_nn_->nnsdk2_func_ptr.query(ctx, AMLNN_QUERY_OUTPUT_ATTR,
                                       &output_attr,
                                       sizeof(output_attr)) != AMLNN_SUCCESS)
    {
      ADLA_LOGE("amlnn_query OUTPUT_ATTR[%u] fail", i);
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "amlnn_query OUTPUT_ATTR fail");
    }
    ADLA_LOGI("[AML Dispatch] NNSDK2 output[%u] name=%s size=%u", i,
              output_attr.name, output_attr.size);
  }

  return {};
}

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType &tensor_type)
{
  return GetTensorBufferRequirements(tensor_type);
}

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType &tensor_type)
{
  return GetTensorBufferRequirements(tensor_type);
}

Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle)
{
  // std::cout << "LiteRtDispatchInvocationContextT::AttachInput *****" << std::endl;
  if (graph_input_index < 0)
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_input_index");
  }

  // auto &tensor = inputs_[graph_input_index];
  input_buffer_handles_[graph_input_index] = tensor_buffer_handle;
  // auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);

  return AttachInputBuffer(tensor_buffer_handle, graph_input_index);
}

Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle)
{
  // std::cout << "LiteRtDispatchInvocationContextT::AttachOutput *****" << std::endl;
  if (graph_output_index < 0)
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_output_index");
  }

  // auto &tensor = outputs_[graph_output_index];
  output_buffer_handles_[graph_output_index] = tensor_buffer_handle;
  return AttachOutputBuffer(tensor_buffer_handle, graph_output_index);
}

Expected<void> LiteRtDispatchInvocationContextT::DetachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle)
{
  (void)tensor_buffer_handle;
  if (graph_input_index < 0 ||
      static_cast<size_t>(graph_input_index) >= input_buffer_ptr_.size())
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_input_index in DetachInput");
  }
  input_buffer_handles_[graph_input_index] = -1;
  input_buffer_ptr_[graph_input_index] = nullptr;
  input_buffer_sizes_[graph_input_index] = 0;
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::DetachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle)
{
  (void)tensor_buffer_handle;
  if (graph_output_index < 0 ||
      static_cast<size_t>(graph_output_index) >= output_buffer_ptr.size())
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_output_index in DetachOutput");
  }
  output_buffer_handles_[graph_output_index] = -1;
  output_buffer_ptr[graph_output_index] = nullptr;
  output_buffer_sizes_[graph_output_index] = 0;
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::AttachBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle)
{
  // std::cout << "LiteRtDispatchInvocationContextT::AttachBuffer" << std::endl;
  auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);
  if (!tensor_buffer)
  {
    return Unexpected(tensor_buffer.Error());
  }

  // auto mem_handle = device_context_.GetMemHandle(tensor_buffer_handle, tensor);
  // if (!mem_handle)
  // {
  //   return Unexpected(mem_handle.Error());
  // }

  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::AttachInputBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle, int graph_index)
{
  LITERT_LOG(LITERT_DEBUG, "LiteRtDispatchInvocationContextT::AttachInputBuffer");
  auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);
  if (!tensor_buffer)
  {
    return Unexpected(tensor_buffer.Error());
  }

  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status = LiteRtGetTensorBufferType(*tensor_buffer, &tensor_buffer_type);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer type");
  }
  // std::cout << "AttachInputBuffer buffer type = " << tensor_buffer_type << std::endl;

  size_t tensor_buffer_size;
  if (auto status =
          LiteRtGetTensorBufferSize(*tensor_buffer, &tensor_buffer_size);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer size");
  }
  // std::cout << "AttachInputBuffer buffer size = " << tensor_buffer_size << std::endl;

  size_t tensor_buffer_offset;
  if (auto status =
          LiteRtGetTensorBufferOffset(*tensor_buffer, &tensor_buffer_offset);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer offset");
  }

  LiteRtRankedTensorType tensor_type;
  if (auto status =
          LiteRtGetTensorBufferTensorType(*tensor_buffer, &tensor_type);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer's type");
  }

  auto element_type =
      static_cast<enum litert::ElementType>(tensor_type.element_type);

  // std::cout << "AttachInputBuffer element_type = " << static_cast<int>(element_type) << std::endl;

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
            LiteRtGetTensorBufferHostMemory(*tensor_buffer, &buffer_host_addr);
        status != kLiteRtStatusOk)
    {
      return Unexpected(status, "Failed to get host memory buffer");
    }
    break;

  case kLiteRtTensorBufferTypeFastRpc:
#if LITERT_HAS_FASTRPC_SUPPORT
    if (auto status = LiteRtGetTensorBufferFastRpcBuffer(
            *tensor_buffer, &buffer_host_addr, &buffer_fd);
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
            *tensor_buffer, &buffer_host_addr, &buffer_fd);
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

  if (graph_index < 0 ||
      static_cast<size_t>(graph_index) >= input_buffer_ptr_.size())
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_input_index in AttachInputBuffer");
  }
  // Cache host ptr; LiteRT may skip Attach on reused buffers.
  input_buffer_ptr_[graph_index] = buffer_host_addr;
  input_buffer_sizes_[graph_index] = tensor_buffer_size;

  if (static_cast<size_t>(graph_index) >= v2_inputs_.size())
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid NNSDK2 input index");
  }
  v2_inputs_[graph_index].index = static_cast<uint32_t>(graph_index);
  v2_inputs_[graph_index].buf = buffer_host_addr;
  v2_inputs_[graph_index].size = static_cast<uint32_t>(tensor_buffer_size);
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::AttachOutputBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle, int graph_index)
{
  LITERT_LOG(LITERT_DEBUG, "LiteRtDispatchInvocationContextT::AttachOutputBuffer");
  auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);
  if (!tensor_buffer)
  {
    return Unexpected(tensor_buffer.Error());
  }
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status = LiteRtGetTensorBufferType(*tensor_buffer, &tensor_buffer_type);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer type");
  }
  // std::cout << "AttachOutputBuffer buffer type = " << tensor_buffer_type << std::endl;

  size_t tensor_buffer_size;
  if (auto status =
          LiteRtGetTensorBufferSize(*tensor_buffer, &tensor_buffer_size);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer size");
  }
  // std::cout << "AttachOutputBuffer buffer size = " << tensor_buffer_size << std::endl;

  size_t tensor_buffer_offset;
  if (auto status =
          LiteRtGetTensorBufferOffset(*tensor_buffer, &tensor_buffer_offset);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer offset");
  }

  LiteRtRankedTensorType tensor_type;
  if (auto status =
          LiteRtGetTensorBufferTensorType(*tensor_buffer, &tensor_type);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to get tensor buffer's type");
  }

  auto element_type =
      static_cast<enum litert::ElementType>(tensor_type.element_type);

  // std::cout << "AttachOutputBuffer element_type = " << static_cast<int>(element_type) << std::endl;

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
            LiteRtGetTensorBufferHostMemory(*tensor_buffer, &buffer_host_addr);
        status != kLiteRtStatusOk)
    {
      return Unexpected(status, "Failed to get host memory buffer");
    }
    break;

  case kLiteRtTensorBufferTypeFastRpc:
#if LITERT_HAS_FASTRPC_SUPPORT
    if (auto status = LiteRtGetTensorBufferFastRpcBuffer(
            *tensor_buffer, &buffer_host_addr, &buffer_fd);
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
            *tensor_buffer, &buffer_host_addr, &buffer_fd);
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

  // std::cout << "AttachOutputBuffer buffer_host_addr = " << buffer_host_addr << std::endl;
  // std::cout << "AttachOutputBuffer buffer_fd = " << buffer_fd << std::endl;

  if (graph_index < 0 ||
      static_cast<size_t>(graph_index) >= output_buffer_ptr.size())
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_output_index in AttachOutputBuffer");
  }
  output_buffer_ptr[graph_index] = buffer_host_addr;
  output_buffer_sizes_[graph_index] = tensor_buffer_size;
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::DetachBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle)
{
  // LITERT_RETURN_IF_ERROR(
  //     device_context_.UnregisterTensorBuffer(tensor_buffer_handle, tensor));
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::ExecuteModel()
{
  // Ensure the active subgraph matches this partition before set/run/get.
  if (tflite::Nnsdk2SelectSubgraphIfNeeded(aml_nn_, aml_nn_->qcontext,
                                           subgraph_idx_, subgraph_num_) < 0)
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "amlnn_select_subgraph fail");
  }

  const int set_ret = aml_nn_->nnsdk2_func_ptr.inputs_set(
      aml_nn_->qcontext, v2_io_num_.n_input, v2_inputs_.data());
  if (set_ret < 0)
  {
    ADLA_LOGE("amlnn_inputs_set fail, ret=%d", set_ret);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "amlnn_inputs_set fail");
  }

  const int run_ret = aml_nn_->nnsdk2_func_ptr.run(aml_nn_->qcontext, nullptr);
  if (run_ret < 0)
  {
    ADLA_LOGE("amlnn_run fail, ret=%d", run_ret);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "amlnn_run fail");
  }

  std::vector<amlnn_output> outputs(v2_io_num_.n_output);
  for (uint32_t i = 0; i < v2_io_num_.n_output; ++i)
  {
    memset(&outputs[i], 0, sizeof(amlnn_output));
    outputs[i].index = i;
    outputs[i].is_float = 0;
  }

  const int get_ret = aml_nn_->nnsdk2_func_ptr.outputs_get(
      aml_nn_->qcontext, v2_io_num_.n_output, outputs.data());
  if (get_ret < 0)
  {
    ADLA_LOGE("amlnn_outputs_get fail, ret=%d", get_ret);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "amlnn_outputs_get fail");
  }

  if (outputs.size() != output_buffer_ptr.size())
  {
    ADLA_LOGE("NNSDK2 output count %zu != LiteRT output count %zu",
              outputs.size(), output_buffer_ptr.size());
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "NNSDK2 output count mismatch");
  }

  for (size_t i = 0; i < output_buffer_ptr.size(); ++i)
  {
    if (outputs[i].buf == nullptr)
    {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "NNSDK2 output buffer is null");
    }
    const size_t npu_bytes = static_cast<size_t>(outputs[i].size);
    const size_t copy_bytes = std::min(output_buffer_sizes_[i], npu_bytes);
    if (copy_bytes < npu_bytes)
    {
      ADLA_LOGW("Truncate NNSDK2 output[%zu]: npu=%zu, buffer=%zu", i, npu_bytes,
                output_buffer_sizes_[i]);
    }
    memcpy(output_buffer_ptr[i], outputs[i].buf, copy_bytes);
  }

  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::Execute()
{
  LITERT_LOG(LITERT_DEBUG, "InvocationContext Execute");
  if (aml_nn_ == nullptr || aml_nn_->qcontext == nullptr)
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "AML NPU context is not initialized");
  }
  if (output_buffer_ptr.size() != output_buffer_sizes_.size())
  {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Output buffer metadata mismatch");
  }

  return ExecuteModel();
}

Expected<void> LiteRtDispatchInvocationContextT::ConvertToUint16(
    LiteRtTensorBufferHandle tensor_buffer_handle, size_t bytes)
{
  auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);
  if (!tensor_buffer)
  {
    return Unexpected(tensor_buffer.Error());
  }
  void *mem_addr;
  if (auto status = LiteRtLockTensorBuffer(
          *tensor_buffer, &mem_addr, kLiteRtTensorBufferLockModeReadWrite);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to lock the tensor buffer");
  }
  auto int16_data = absl::MakeSpan(static_cast<const std::int16_t *>(mem_addr),
                                   bytes / sizeof(std::int16_t));
  std::vector<std::uint16_t> uint16_data;
  // qnn::ConvertDataFromInt16toUInt16(int16_data, uint16_data);
  // std::memcpy(mem_addr, uint16_data.data(), bytes);
  // if (auto status = LiteRtUnlockTensorBuffer(*tensor_buffer);
  //     status != kLiteRtStatusOk)
  // {
  //   return Unexpected(status, "Failed to unlock the tensor buffer");
  // }
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::ConvertToInt16(
    LiteRtTensorBufferHandle tensor_buffer_handle, size_t bytes)
{
  auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);
  if (!tensor_buffer)
  {
    return Unexpected(tensor_buffer.Error());
  }
  void *mem_addr;
  if (auto status = LiteRtLockTensorBuffer(
          *tensor_buffer, &mem_addr, kLiteRtTensorBufferLockModeReadWrite);
      status != kLiteRtStatusOk)
  {
    return Unexpected(status, "Failed to lock the tensor buffer");
  }
  auto uint16_data = absl::MakeSpan(static_cast<const std::uint16_t *>(mem_addr),
                                    bytes / sizeof(std::uint16_t));
  std::vector<std::int16_t> int16_data;
  // qnn::ConvertDataFromUInt16toInt16(uint16_data, int16_data);
  // std::memcpy(mem_addr, int16_data.data(), bytes);
  // if (auto status = LiteRtUnlockTensorBuffer(*tensor_buffer);
  //     status != kLiteRtStatusOk)
  // {
  //   return Unexpected(status, "Failed to unlock the tensor buffer");
  // }
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::Profile()
{
  // TODO: Implement a viewer class to beautify the format of profiling output
  std::stringstream data_ss;
  data_ss << "\nExecute Stats:\n"
          << "----------------" << std::endl;
  LITERT_LOG(LITERT_DEBUG, "Profile");
  LITERT_LOG(LITERT_INFO, "%s", data_ss.str().c_str());

  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::WriteTensorTo(
    const std::filesystem::path &output_folder, int tensor_index)
{
  // qnn::CreateDirectoryRecursive(output_folder);
  // std::filesystem::path output_path =
  //     output_folder / (tensor.GetName() + ".raw");
  // std::ofstream fout(output_path, std::ios::binary);
  // if (fout.fail())
  // {
  //   LITERT_LOG(LITERT_ERROR, "Failed to write dumped tensor");
  //   return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  // }
  // fout.write(static_cast<const char *>(tensor.GetQnnTensor().v2.clientBuf.data),
  //            tensor.GetQnnTensor().v2.clientBuf.dataSize);
  // std::filesystem::path quant_param_path =
  //     output_folder / (tensor.GetName() + ".csv");
  // std::ofstream quant_file(quant_param_path);
  // float scale = 1;
  // int32_t zero_point = 0;
  // auto quant_param = tensor.GetQuantParams();
  // if (std::holds_alternative<qnn::ScaleOffsetQuantizeParamsWrapper>(
  //         quant_param))
  // {
  //   scale =
  //       std::get<qnn::ScaleOffsetQuantizeParamsWrapper>(quant_param).GetScale();
  //   zero_point = std::get<qnn::ScaleOffsetQuantizeParamsWrapper>(quant_param)
  //                    .GetZeroPoint();
  // }
  // std::stringstream quant_ss;
  // quant_ss << scale << "," << zero_point << "\n";
  // quant_file << quant_ss.str();
  // quant_file.close();
  return {};
}
