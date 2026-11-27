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

#include "litert/vendors/amlogic/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/amlogic/dispatch/litert_dispatch_device_context.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/amlogic/core/amlogic_dispatch_info.h"
#include "litert/vendors/amlogic/core/nnsdk_func.h"

using litert::Expected;
using litert::Unexpected;

LiteRtDispatchInvocationContextT::LiteRtDispatchInvocationContextT(
    LiteRtDispatchDeviceContextT &device_context, int num_inputs,
    int num_outputs)
    : device_context_(device_context) {
  input_buffer_handles_.assign(num_inputs, -1);
  output_buffer_handles_.assign(num_outputs, -1);
  // Index-based (not push_back): LiteRT may Detach+Attach new output buffers
  // every frame (e.g. camera demo). push_back would accumulate and cause
  // "NPU output count 3 < LiteRT output count 6".
  input_buffer_ptr_.assign(num_inputs, nullptr);
  input_buffer_sizes_.assign(num_inputs, 0);
  output_buffer_ptr.assign(num_outputs, nullptr);
  output_buffer_sizes_.assign(num_outputs, 0);
  LITERT_LOG(LITERT_INFO, "[AML_DELEGATE_Kernel] InitState");
  LITERT_LOG(LITERT_INFO, "[AML_DELEGATE_Kernel] Start to open libnnsdk.so");
  aml_nn_ = std::make_shared<aml_nnsdk::AML_NN>(aml_nnsdk::CreateAmlNn());
  if (aml_nn_ == nullptr || !aml_nn_->available) {
    LITERT_LOG(LITERT_ERROR, "[AML_DELEGATE_Kernel] aml_nn create failed");
    return;
  }
  LITERT_LOG(LITERT_INFO, "[AML_DELEGATE_Kernel] aml_nn create success AML_NN=%p",
            aml_nn_.get());
}

LiteRtDispatchInvocationContextT::~LiteRtDispatchInvocationContextT() {
  if (aml_nn_ != nullptr && aml_nn_->qcontext != nullptr &&
      aml_nn_->nnsdk2_func_ptr.destroy != nullptr) {
    LITERT_LOG(LITERT_INFO, "[AML Dispatch] amlnn_destroy");
    aml_nn_->nnsdk2_func_ptr.destroy(aml_nn_->qcontext);
    aml_nn_->qcontext = nullptr;
  }
  if (aml_nn_ != nullptr && aml_nn_->available) {
    aml_nnsdk::ReleaseNnsdk();
  }
}

Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    LiteRtDispatchDeviceContextT &device_context,
    const LiteRtMemBuffer *exec_bytecode_buffer, const char *function_name, int num_inputs, int num_outputs) {
  if (exec_bytecode_buffer == nullptr ||
      exec_bytecode_buffer->base_addr == nullptr ||
      exec_bytecode_buffer->size == 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "ADLA bytecode is empty");
  }
  int subgraph_idx = 0;
  if (!ParseAmlogicPartitionCallInfo(function_name, &subgraph_idx)) {
    LITERT_LOG(LITERT_ERROR, "Invalid AML call info: %s",
               function_name != nullptr ? function_name : "(null)");
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "AML call info must be aml_partition_<subgraph_idx>");
  }

  LITERT_LOG(LITERT_DEBUG, "InvocationContext Create");
  LITERT_LOG(LITERT_INFO,
             "[AML Dispatch] call_info=%s subgraph_idx=%d bytecode_size=%zu",
             function_name, subgraph_idx, exec_bytecode_buffer->size);
  auto ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>(
      new LiteRtDispatchInvocationContextT(device_context, num_inputs,
                                           num_outputs));

  const uint8_t *byte_code_data =
      static_cast<const uint8_t *>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  ptr->adla_bin_.assign(byte_code_data,
                        byte_code_data + exec_bytecode_buffer->size);
  ptr->adla_bin_size_ = exec_bytecode_buffer->size;
  ptr->subgraph_idx_ = subgraph_idx;

  if (!aml_nnsdk::IsNnsdk2Available(ptr->aml_nn_.get())) {
    LITERT_LOG(LITERT_ERROR, "NNSDK2 symbols not available");
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "NNSDK2 required but not available");
  }

  if (auto status = ptr->CreateModel(ptr->subgraph_idx_); !status) {
    if (ptr->aml_nn_ != nullptr && ptr->aml_nn_->qcontext != nullptr &&
        ptr->aml_nn_->nnsdk2_func_ptr.destroy != nullptr) {
      ptr->aml_nn_->nnsdk2_func_ptr.destroy(ptr->aml_nn_->qcontext);
      ptr->aml_nn_->qcontext = nullptr;
    }
    return Unexpected(status.Error());
  }

  return litert::Expected<Ptr>(std::move(ptr));
}


namespace {

Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
    const LiteRtRankedTensorType &tensor_type) {
  if (tensor_type.layout.has_strides) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported");
  }

  std::vector<LiteRtTensorBufferType> supported_tensor_buffer_types = {
      kLiteRtTensorBufferTypeHostMemory,
  };

  auto buffer_size = litert::RankedTensorType(tensor_type).Bytes();
  if (!buffer_size) {
    return Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  if (auto status = LiteRtCreateTensorBufferRequirements(
          supported_tensor_buffer_types.size(),
          supported_tensor_buffer_types.data(), *buffer_size, /*num_strides=*/0,
          /*strides=*/nullptr, &requirements);
      status != kLiteRtStatusOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Not implemented");
  }

  return requirements;
}

} // namespace

Expected<void> LiteRtDispatchInvocationContextT::CreateModel(int subgraph_idx) {
  amlnn_init_config init_config;
  memset(&init_config, 0, sizeof(init_config));
  init_config.backend_type = AMLNN_BACKEND_ADLA_NPU;

  if (adla_bin_.empty() || adla_bin_size_ == 0 ||
      adla_bin_size_ > UINT32_MAX) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Embedded ADLA binary is missing");
  }

  void* ctx = nullptr;
  const int ret = aml_nn_->nnsdk2_func_ptr.init(
      &ctx, adla_bin_.data(), static_cast<uint32_t>(adla_bin_size_),
      &init_config);
  if (ret < 0 || ctx == nullptr) {
    LITERT_LOG(LITERT_ERROR, "amlnn_init fail from memory, size=%zu, ret=%d",
              adla_bin_size_, ret);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "amlnn_init fail");
  }
  aml_nn_->qcontext = ctx;

  // Match NNSDK sample: query AMLNN_QUERY_SUBGRAPH_NUM after amlnn_init.
  subgraph_num_ = 1;
  if (aml_nnsdk::Nnsdk2QuerySubgraphNum(aml_nn_.get(), ctx, &subgraph_num_) < 0) {
    LITERT_LOG(LITERT_ERROR, "[AML Dispatch] query subgraph num fail");
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "AMLNN_QUERY_SUBGRAPH_NUM fail");
  }
  LITERT_LOG(LITERT_INFO, "[AML Dispatch] NNSDK2 subgraph_num=%d", subgraph_num_);

  return BindSubgraph(subgraph_idx);
}

Expected<void> LiteRtDispatchInvocationContextT::BindSubgraph(int subgraph_idx) {
  if (aml_nn_ == nullptr || aml_nn_->qcontext == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "AML NPU context is not initialized");
  }

  subgraph_idx_ = subgraph_idx;
  void* ctx = aml_nn_->qcontext;

  // Match NNSDK sample: amlnn_select_subgraph then query IO for that subgraph.
  if (aml_nnsdk::Nnsdk2SelectSubgraphIfNeeded(aml_nn_.get(), ctx, subgraph_idx_,
                                           subgraph_num_) < 0) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "amlnn_select_subgraph fail");
  }

  memset(&v2_io_num_, 0, sizeof(v2_io_num_));
  if (aml_nn_->nnsdk2_func_ptr.query(ctx, AMLNN_QUERY_IN_OUT_NUM, &v2_io_num_,
                                     sizeof(v2_io_num_)) != AMLNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "amlnn_query IN_OUT_NUM fail subgraph_idx=%d", subgraph_idx_);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "amlnn_query IN_OUT_NUM fail");
  }

  LITERT_LOG(LITERT_INFO, "[AML Dispatch] BindSubgraph idx=%d/%d input=%u output=%u",
            subgraph_idx_, subgraph_num_, v2_io_num_.n_input,
            v2_io_num_.n_output);
  if (v2_io_num_.n_input != input_buffer_handles_.size() ||
      v2_io_num_.n_output != output_buffer_handles_.size()) {
    LITERT_LOG(LITERT_ERROR, "[AML Dispatch] IO count mismatch litert in/out=%zu/%zu "
              "nnsdk in/out=%u/%u",
              input_buffer_handles_.size(), output_buffer_handles_.size(),
              v2_io_num_.n_input, v2_io_num_.n_output);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "NNSDK IO count does not match LiteRT");
  }

  v2_inputs_.assign(v2_io_num_.n_input, {});
  input_attr_sizes_.assign(v2_io_num_.n_input, 0);
  output_attr_sizes_.assign(v2_io_num_.n_output, 0);
  for (uint32_t i = 0; i < v2_io_num_.n_input; ++i) {
    amlnn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = i;
    if (aml_nn_->nnsdk2_func_ptr.query(ctx, AMLNN_QUERY_INPUT_ATTR, &input_attr,
                                       sizeof(input_attr)) != AMLNN_SUCCESS) {
      LITERT_LOG(LITERT_ERROR, "amlnn_query INPUT_ATTR[%u] fail", i);
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "amlnn_query INPUT_ATTR fail");
    }
    LITERT_LOG(LITERT_INFO, "[AML Dispatch] NNSDK2 input[%u] name=%s size=%u", i,
              input_attr.name, input_attr.size);
    input_attr_sizes_[i] = input_attr.size;
  }

  for (uint32_t i = 0; i < v2_io_num_.n_output; ++i) {
    amlnn_tensor_attr output_attr;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = i;
    if (aml_nn_->nnsdk2_func_ptr.query(ctx, AMLNN_QUERY_OUTPUT_ATTR,
                                       &output_attr,
                                       sizeof(output_attr)) != AMLNN_SUCCESS) {
      LITERT_LOG(LITERT_ERROR, "amlnn_query OUTPUT_ATTR[%u] fail", i);
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "amlnn_query OUTPUT_ATTR fail");
    }
    LITERT_LOG(LITERT_INFO, "[AML Dispatch] NNSDK2 output[%u] name=%s size=%u", i,
              output_attr.name, output_attr.size);
    output_attr_sizes_[i] = output_attr.size;
  }

  return {};
}

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType &tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType &tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_input_index < 0 ||
      static_cast<size_t>(graph_input_index) >= input_buffer_handles_.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_input_index");
  }

  input_buffer_handles_[graph_input_index] = tensor_buffer_handle;
  return AttachInputBuffer(tensor_buffer_handle, graph_input_index);
}

Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_output_index < 0 ||
      static_cast<size_t>(graph_output_index) >= output_buffer_handles_.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_output_index");
  }

  output_buffer_handles_[graph_output_index] = tensor_buffer_handle;
  return AttachOutputBuffer(tensor_buffer_handle, graph_output_index);
}

Expected<void> LiteRtDispatchInvocationContextT::DetachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  (void)tensor_buffer_handle;
  if (graph_input_index < 0 ||
      static_cast<size_t>(graph_input_index) >= input_buffer_handles_.size() ||
      static_cast<size_t>(graph_input_index) >= input_buffer_ptr_.size() ||
      static_cast<size_t>(graph_input_index) >= input_buffer_sizes_.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_input_index in DetachInput");
  }
  input_buffer_handles_[graph_input_index] = -1;
  input_buffer_ptr_[graph_input_index] = nullptr;
  input_buffer_sizes_[graph_input_index] = 0;
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::DetachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  (void)tensor_buffer_handle;
  if (graph_output_index < 0 ||
      static_cast<size_t>(graph_output_index) >= output_buffer_handles_.size() ||
      static_cast<size_t>(graph_output_index) >= output_buffer_ptr.size() ||
      static_cast<size_t>(graph_output_index) >= output_buffer_sizes_.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_output_index in DetachOutput");
  }
  output_buffer_handles_[graph_output_index] = -1;
  output_buffer_ptr[graph_output_index] = nullptr;
  output_buffer_sizes_[graph_output_index] = 0;
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::AttachInputBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle, int graph_index) {
  LITERT_LOG(LITERT_DEBUG, "LiteRtDispatchInvocationContextT::AttachInputBuffer");
  auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);
  if (!tensor_buffer) {
    return Unexpected(tensor_buffer.Error());
  }

  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status = LiteRtGetTensorBufferType(*tensor_buffer, &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer type");
  }
  size_t tensor_buffer_size;
  if (auto status =
          LiteRtGetTensorBufferSize(*tensor_buffer, &tensor_buffer_size);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer size");
  }
  LiteRtRankedTensorType tensor_type;
  if (auto status =
          LiteRtGetTensorBufferTensorType(*tensor_buffer, &tensor_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer's type");
  }

  if (tensor_type.layout.has_strides) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported");
  }

  void *buffer_host_addr;
  (void)buffer_host_addr;

  switch (tensor_buffer_type) {
  case kLiteRtTensorBufferTypeHostMemory:
    if (auto status =
            LiteRtGetTensorBufferHostMemory(*tensor_buffer, &buffer_host_addr);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get host memory buffer");
    }
    break;

  case kLiteRtTensorBufferTypeFastRpc:
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "FastRPC tensor buffers are not supported by AML");

  case kLiteRtTensorBufferTypeDmaBuf:
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "DMA-BUF tensor buffers are not supported by AML");

  default:
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unsupported tensor buffer type");
  }

  if (graph_index < 0 ||
      static_cast<size_t>(graph_index) >= input_buffer_ptr_.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_input_index in AttachInputBuffer");
  }
  if (static_cast<size_t>(graph_index) >= input_attr_sizes_.size() ||
      tensor_buffer_size != input_attr_sizes_[graph_index]) {
    LITERT_LOG(LITERT_ERROR, "input[%d] size %zu != nnsdk %u", graph_index, tensor_buffer_size,
              static_cast<size_t>(graph_index) < input_attr_sizes_.size()
                  ? input_attr_sizes_[graph_index]
                  : 0);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "LiteRT input size does not match NNSDK");
  }
  input_buffer_ptr_[graph_index] = buffer_host_addr;
  input_buffer_sizes_[graph_index] = tensor_buffer_size;

  if (static_cast<size_t>(graph_index) >= v2_inputs_.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid NNSDK2 input index");
  }
  v2_inputs_[graph_index].index = static_cast<uint32_t>(graph_index);
  v2_inputs_[graph_index].buf = buffer_host_addr;
  v2_inputs_[graph_index].size = static_cast<uint32_t>(tensor_buffer_size);
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::AttachOutputBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle, int graph_index) {
  LITERT_LOG(LITERT_DEBUG, "LiteRtDispatchInvocationContextT::AttachOutputBuffer");
  auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);
  if (!tensor_buffer) {
    return Unexpected(tensor_buffer.Error());
  }
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status = LiteRtGetTensorBufferType(*tensor_buffer, &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer type");
  }
  size_t tensor_buffer_size;
  if (auto status =
          LiteRtGetTensorBufferSize(*tensor_buffer, &tensor_buffer_size);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer size");
  }
  LiteRtRankedTensorType tensor_type;
  if (auto status =
          LiteRtGetTensorBufferTensorType(*tensor_buffer, &tensor_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer's type");
  }

  if (tensor_type.layout.has_strides) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported");
  }

  void *buffer_host_addr;
  (void)buffer_host_addr;

  switch (tensor_buffer_type) {
  case kLiteRtTensorBufferTypeHostMemory:
    if (auto status =
            LiteRtGetTensorBufferHostMemory(*tensor_buffer, &buffer_host_addr);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get host memory buffer");
    }
    break;

  case kLiteRtTensorBufferTypeFastRpc:
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "FastRPC tensor buffers are not supported by AML");

  case kLiteRtTensorBufferTypeDmaBuf:
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "DMA-BUF tensor buffers are not supported by AML");

  default:
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unsupported tensor buffer type");
  }

  if (graph_index < 0 ||
      static_cast<size_t>(graph_index) >= output_buffer_ptr.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_output_index in AttachOutputBuffer");
  }
  if (static_cast<size_t>(graph_index) >= output_attr_sizes_.size() ||
      tensor_buffer_size != output_attr_sizes_[graph_index]) {
    LITERT_LOG(LITERT_ERROR, "output[%d] size %zu != nnsdk %u", graph_index,
              tensor_buffer_size,
              static_cast<size_t>(graph_index) < output_attr_sizes_.size()
                  ? output_attr_sizes_[graph_index]
                  : 0);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "LiteRT output size does not match NNSDK");
  }
  output_buffer_ptr[graph_index] = buffer_host_addr;
  output_buffer_sizes_[graph_index] = tensor_buffer_size;
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::ExecuteModel() {
  // Ensure the active subgraph matches this partition before set/run/get.
  if (aml_nnsdk::Nnsdk2SelectSubgraphIfNeeded(aml_nn_.get(), aml_nn_->qcontext,
                                           subgraph_idx_, subgraph_num_) < 0) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "amlnn_select_subgraph fail");
  }

  if (v2_io_num_.n_input != input_buffer_handles_.size() ||
      v2_io_num_.n_output != output_buffer_handles_.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "NNSDK IO count does not match LiteRT");
  }

  struct TensorBufferLock {
    LiteRtTensorBuffer buffer = nullptr;
    ~TensorBufferLock() {
      if (buffer != nullptr) {
        LiteRtUnlockTensorBuffer(buffer);
      }
    }
  };
  std::vector<TensorBufferLock> input_locks(input_buffer_handles_.size());
  std::vector<TensorBufferLock> output_locks(output_buffer_handles_.size());
  for (size_t i = 0; i < input_buffer_handles_.size(); ++i) {
    if (input_buffer_handles_[i] < 0) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Input buffer is not attached");
    }
    auto tensor_buffer = device_context_.GetTensorBuffer(input_buffer_handles_[i]);
    if (!tensor_buffer) {
      return Unexpected(tensor_buffer.Error());
    }
    void* addr = nullptr;
    if (LiteRtLockTensorBuffer(*tensor_buffer, &addr,
                               kLiteRtTensorBufferLockModeRead) !=
            kLiteRtStatusOk ||
        addr == nullptr) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to lock input tensor buffer");
    }
    input_locks[i].buffer = *tensor_buffer;
    v2_inputs_[i].index = static_cast<uint32_t>(i);
    v2_inputs_[i].buf = addr;
    v2_inputs_[i].size = input_attr_sizes_[i];
  }
  for (size_t i = 0; i < output_buffer_handles_.size(); ++i) {
    if (output_buffer_handles_[i] < 0) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Output buffer is not attached");
    }
    auto tensor_buffer =
        device_context_.GetTensorBuffer(output_buffer_handles_[i]);
    if (!tensor_buffer) {
      return Unexpected(tensor_buffer.Error());
    }
    void* addr = nullptr;
    if (LiteRtLockTensorBuffer(*tensor_buffer, &addr,
                               kLiteRtTensorBufferLockModeWrite) !=
            kLiteRtStatusOk ||
        addr == nullptr) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to lock output tensor buffer");
    }
    output_locks[i].buffer = *tensor_buffer;
    output_buffer_ptr[i] = addr;
  }

  const int set_ret = aml_nn_->nnsdk2_func_ptr.inputs_set(
      aml_nn_->qcontext, v2_io_num_.n_input, v2_inputs_.data());
  if (set_ret < 0) {
    LITERT_LOG(LITERT_ERROR, "amlnn_inputs_set fail, ret=%d", set_ret);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "amlnn_inputs_set fail");
  }

  const int run_ret = aml_nn_->nnsdk2_func_ptr.run(aml_nn_->qcontext, nullptr);
  if (run_ret < 0) {
    LITERT_LOG(LITERT_ERROR, "amlnn_run fail, ret=%d", run_ret);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "amlnn_run fail");
  }

  std::vector<amlnn_output> outputs(v2_io_num_.n_output);
  for (uint32_t i = 0; i < v2_io_num_.n_output; ++i) {
    memset(&outputs[i], 0, sizeof(amlnn_output));
    outputs[i].index = i;
    outputs[i].is_float = 0;
  }

  const int get_ret = aml_nn_->nnsdk2_func_ptr.outputs_get(
      aml_nn_->qcontext, v2_io_num_.n_output, outputs.data());
  if (get_ret < 0) {
    LITERT_LOG(LITERT_ERROR, "amlnn_outputs_get fail, ret=%d", get_ret);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "amlnn_outputs_get fail");
  }

  if (outputs.size() != output_buffer_ptr.size()) {
    LITERT_LOG(LITERT_ERROR, "NNSDK2 output count %zu != LiteRT output count %zu",
              outputs.size(), output_buffer_ptr.size());
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "NNSDK2 output count mismatch");
  }

  for (size_t i = 0; i < output_buffer_ptr.size(); ++i) {
    if (outputs[i].buf == nullptr || output_buffer_ptr[i] == nullptr) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "NNSDK2 or LiteRT output buffer is null");
    }
    const size_t npu_bytes = static_cast<size_t>(outputs[i].size);
    if (npu_bytes != output_buffer_sizes_[i] ||
        npu_bytes != output_attr_sizes_[i]) {
      LITERT_LOG(LITERT_ERROR, "output[%zu] size mismatch npu=%zu buffer=%zu attr=%u", i,
                npu_bytes, output_buffer_sizes_[i], output_attr_sizes_[i]);
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "NNSDK output size does not match LiteRT");
    }
    memcpy(output_buffer_ptr[i], outputs[i].buf, npu_bytes);
  }

  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::Execute() {
  LITERT_LOG(LITERT_DEBUG, "InvocationContext Execute");
  if (aml_nn_ == nullptr || aml_nn_->qcontext == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "AML NPU context is not initialized");
  }
  if (output_buffer_ptr.size() != output_buffer_sizes_.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Output buffer metadata mismatch");
  }

  return ExecuteModel();
}
