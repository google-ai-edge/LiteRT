// Copyright 2025 Vivante Corporation.
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

#include "litert/vendors/verisilicon/dispatch/litert_dispatch_invocation_context.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/vendors/verisilicon/dispatch/viplite_adapter_api.h"

using litert::Error;
using litert::Expected;
using litert::Unexpected;

namespace {

template <class X, class Align>
inline constexpr auto Pad(X x, Align align) {
  return ((x + align - 1) / align) * align;
}

}  // namespace

namespace litert{
namespace verisilicon{

  litert::Expected<VipliteNetworkT::ModelPtr> VipliteNetworkT::CreateFromByteCode(
      litert::verisilicon::VipliteAdapterApi &viplite_adapter_api,
      const void* exec_bytecode_ptr,
      size_t   exec_bytecode_size,
      int num_inputs, int num_outputs, LiteRtDispatchDeviceContextT::VpmNetworkParam* vpm_param) {
     auto model_ptr = std::make_unique<VipliteNetworkT>(viplite_adapter_api, exec_bytecode_ptr,
                                                        exec_bytecode_size);
     LITERT_RETURN_IF_ERROR(model_ptr->Setup(vpm_param));
     //Identify inputs and outputs
     if (model_ptr->InputCount() != num_inputs ||
         model_ptr->OutputCount() != num_inputs ) {
          return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to identify inputs and outputs");
         }

     return model_ptr;
  }

VipliteNetworkT::~VipliteNetworkT() {
  if(network_){
    viplite_adapter_api_.api().finish_network(network_);
    viplite_adapter_api_.api().destroy_network(network_);
  }
  network_ = NULL;
}
litert::Expected<void> VipliteNetworkT::Setup(LiteRtDispatchDeviceContextT::VpmNetworkParam* vpm_param) {
    // creat viplite network for memory
    vip_create_network_param_t net_param;
    uint32_t input_count = 0;
    uint32_t output_count = 0;
    unsigned int time_out;
    unsigned int cnnprofile_level;
    bool dump_nbg;

    if(vpm_param) {
      device_index_ = vpm_param->device_index;
      core_index_ = vpm_param->core_index;
      time_out = vpm_param->time_out;
      cnnprofile_level = vpm_param->cnnprofile_level;
      dump_nbg = vpm_param->dump_nbg;
    }
    memset(&net_param, 0, sizeof(net_param));
    net_param.device_index = device_index_;
    net_param.prop = VIP_NET_CREATE_PROP_FROM_NBG;
    net_param.nbg.type = VIP_NET_CREATE_NBG_FROM_MEMORY;
    net_param.nbg.memory.nbg_memory = (void *)exec_bytecode_ptr_;
    net_param.nbg.memory.nbg_size = exec_bytecode_size_;

    if (cnnprofile_level == 1) {
        net_param.prop = (vip_net_create_prop_e)(net_param.prop | 0x00400000);
    } else if (cnnprofile_level == 2) {
        net_param.prop = (vip_net_create_prop_e)(net_param.prop | 0x00800000);
    }

    if (dump_nbg == 1) {
        net_param.prop = (vip_net_create_prop_e)(net_param.prop | 0x01000000);
    }

    if (auto result = viplite_adapter_api_.api().create_network(&net_param,sizeof(net_param),&network_);
      result != VIP_SUCCESS) {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create viplite network");
    }
    LITERT_RETURN_IF_ERROR(Query(VIP_NETWORK_PROP_CORE_COUNT,&core_count_));
    if(core_index_ >= core_count_) {
        LITERT_LOG(LITERT_WARNING, "The core index is large than core count, using default 0");
        core_index_ = 0;
    }
    LITERT_RETURN_IF_ERROR(Query(VIP_NETWORK_PROP_INPUT_COUNT,&input_count));
    LITERT_RETURN_IF_ERROR(Query(VIP_NETWORK_PROP_INPUT_COUNT,&output_count));
    LITERT_RETURN_IF_ERROR(Set(VIP_NETWORK_PROP_SET_CORE_INDEX,&core_index_));
    if(time_out > 0) {
      LITERT_RETURN_IF_ERROR(Set(VIP_NETWORK_PROP_SET_TIME_OUT,&time_out));
    }
    LITERT_RETURN_IF_ERROR(Prepare());
    input_buffers_.resize(input_count, NULL);
    output_buffers_.resize(output_count, NULL);

  return {};
}

litert::Expected<void> VipliteNetworkT::Query(VipEnum property, void *value) {
  if (auto result = viplite_adapter_api_.api().query_network(network_,property,value);
      result != VIP_SUCCESS) {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to query viplite network");
    }
    return {};
}

litert::Expected<void> VipliteNetworkT::Set(VipEnum property, void *value) {
  if (auto result = viplite_adapter_api_.api().set_network(network_, property, value);
      result != VIP_SUCCESS) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set viplite network");
  }
  return {};
}

litert::Expected<void> VipliteNetworkT::Prepare() {
  if (auto result = viplite_adapter_api_.api().prepare_network(network_);
      result != VIP_SUCCESS) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to prepare viplite network");
  }
  return {};
}

litert::Expected<void> VipliteNetworkT::Run() {
  if (auto result = viplite_adapter_api_.api().run_network(network_);
      result != VIP_SUCCESS) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to run viplite network");
  }
  return {};
}

litert::Expected<void> VipliteNetworkT::Trigger() {
  if (auto result = viplite_adapter_api_.api().trigger_network(network_);
      result != VIP_SUCCESS) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to trigger viplite network");
  }
  return {};
}

litert::Expected<void> VipliteNetworkT::Wait() {
  if (auto result = viplite_adapter_api_.api().wait_network(network_);
      result != VIP_SUCCESS) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to wait viplite network");
  }
  return {};
}

litert::Expected<void> VipliteNetworkT::Cancel() {
  if (auto result = viplite_adapter_api_.api().cancel_network(network_);
      result != VIP_SUCCESS) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to cancel viplite network");
  }
  return {};
}

litert::Expected<void> VipliteNetworkT::QueryInput(uint32_t index, VipEnum property, void *value) {
  uint32_t input_count = input_buffers_.size();
  if (index >= input_count) {
    std::stringstream msgss("Failed to query viplite network input ");
    msgss <<  std::to_string(index) << ". The index is large than input count("
          <<std::to_string(input_count) << ") of the compiled model";
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 msgss.str());
  }
  if (auto result = viplite_adapter_api_.api().query_input(network_, index, property, value);
      result != VIP_SUCCESS) {
    std::stringstream msgss("Failed to query viplite network input ");
     msgss <<  std::to_string(index);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 msgss.str());
  }
  return {};
}

litert::Expected<void> VipliteNetworkT::QueryOutput(uint32_t index, VipEnum property, void *value) {
  uint32_t output_count = output_buffers_.size();
  if (index >= output_count) {
    std::stringstream msgss("Failed to query viplite network output ");
    msgss <<  std::to_string(index) << ". The index is large than output count("
          <<std::to_string(output_count) << ") of the compiled model";
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 msgss.str());
  }
  if (auto result = viplite_adapter_api_.api().query_output(network_, index, property, value);
      result != VIP_SUCCESS) {
    std::stringstream msgss("Failed to query viplite network output ");
     msgss <<  std::to_string(index);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 msgss.str());
  }
  return {};
}

litert::Expected<void> VipliteNetworkT::SetInput(uint32_t index, VipBuffer buffer) {
  uint32_t input_count = input_buffers_.size();
  if (index >= input_count) {
    std::stringstream msgss("Failed to query viplite network input ");
    msgss <<  std::to_string(index) << ". The index is large than input count("
          <<std::to_string(input_count) << ") of the compiled model";
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 msgss.str());
  }
  if (auto result = viplite_adapter_api_.api().set_input(network_, index, buffer);
      result != VIP_SUCCESS) {
    std::stringstream msgss("Failed to set viplite network input ");
     msgss <<  std::to_string(index);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 msgss.str());
  }
  input_buffers_.at(index) = buffer;
  return {};
}
litert::Expected<void> VipliteNetworkT::SetOutput(uint32_t index, VipBuffer buffer) {
  uint32_t output_count = output_buffers_.size();
  if (index >= output_count) {
    std::stringstream msgss("Failed to query viplite network output ");
    msgss <<  std::to_string(index) << ". The index is large than output count("
          <<std::to_string(output_count) << ") of the compiled model";
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 msgss.str());
  }
  if (auto result = viplite_adapter_api_.api().set_output(network_, index, buffer);
       result != VIP_SUCCESS) {
    std::stringstream msgss("Failed to set viplite network output ");
     msgss <<  std::to_string(index);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 msgss.str());
  }
  input_buffers_.at(index) = buffer;
  return {};
}

litert::Expected<VipliteNetworkT::VipBuffer> VipliteNetworkT::GetInput(uint32_t index) {
  uint32_t input_count = input_buffers_.size();
  if (index >= input_count) {
    std::stringstream msgss("Failed to query viplite network input ");
    msgss <<  std::to_string(index) << ". The index is large than input count("
          <<std::to_string(input_count) << ") of the compiled model";
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 msgss.str());
  }
  return input_buffers_.at(index);
}

litert::Expected<VipliteNetworkT::VipBuffer> VipliteNetworkT::GetOutput(uint32_t index) {
  uint32_t output_count = output_buffers_.size();
  if (index >= output_count) {
    std::stringstream msgss("Failed to query viplite network output ");
    msgss <<  std::to_string(index) << ". The index is large than output count("
          <<std::to_string(output_count) << ") of the compiled model";
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 msgss.str());
  }
  return input_buffers_.at(index);
}

} //namespace verisilicon
} //namespace litert

Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    litert::verisilicon::VipliteAdapterApi& viplite_adapter_api,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs) {

  (void)function_name;
  const char* exec_bytecode_ptr =
      static_cast<const char*>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  auto exec_bytecode_size = exec_bytecode_buffer->size;
  LiteRtDispatchDeviceContextT::VpmNetworkParam vpm_param = {0};
  device_context->GetVpmNetworkParam(&vpm_param);

  LITERT_ASSIGN_OR_RETURN(litert::verisilicon::VipliteNetworkT::ModelPtr model,
              litert::verisilicon::VipliteNetworkT::CreateFromByteCode(viplite_adapter_api, exec_bytecode_ptr,
                                        exec_bytecode_size, num_inputs, num_outputs, &vpm_param));

  return Ptr(new LiteRtDispatchInvocationContextT(
      viplite_adapter_api,
      device_context,
      std::move(model),
      num_inputs,num_outputs));
}

namespace {

Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type) {
  if (tensor_type.layout.has_strides) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported on Verisilicon NPU");
  }

  int num_supported_tensor_buffer_types =
      sizeof(LiteRtDispatchDeviceContextT::kSupportedTensorBufferTypes) /
      sizeof(LiteRtDispatchDeviceContextT::kSupportedTensorBufferTypes[0]);

  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return Unexpected(buffer_size.Error());
  }
  // Viplite maps host memory need memory size align cache line (64) .
  size_t padded_buffer_size = Pad(*buffer_size, litert::verisilicon::kVipliteCacheLineAlignment);
  //Viplite maps host memory need memory start address align with burst size(256).
  LiteRtTensorBufferRequirements requirements;
  if (auto status = LiteRtCreateTensorBufferRequirementsWithAlignment(
              num_supported_tensor_buffer_types,
              LiteRtDispatchDeviceContextT::kSupportedTensorBufferTypes,
              padded_buffer_size, /*num_strides=*/0, /*strides=*/nullptr,
              litert::verisilicon::kVipliteAddressAlignment, &requirements);
        status != kLiteRtStatusOk) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to create tensor buffer requirements");
    }

  return requirements;
}
}  // namespace

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto viplite_memory_info =
      device_context_->GetVipliteMemoryInfo(tensor_buffer_handle);
  if (!viplite_memory_info) {
    return litert::Error(viplite_memory_info.Error());
  }
  LITERT_RETURN_IF_ERROR(model_->SetInput(graph_input_index,
                                          viplite_memory_info->buffer));
  input_buffers_handles_.at(graph_input_index) = tensor_buffer_handle;
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto viplite_memory_info =
      device_context_->GetVipliteMemoryInfo(tensor_buffer_handle);
  if (!viplite_memory_info) {
    return litert::Error(viplite_memory_info.Error());
  }
  LITERT_RETURN_IF_ERROR(model_->SetOutput(graph_output_index,
                                          viplite_memory_info->buffer));
  output_buffers_handles_.at(graph_output_index) = tensor_buffer_handle;
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::DetachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  // Nothing to do.
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::DetachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  // Nothing to do.
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::Invoke() {
  //When creating an input buffer in host memory,
  //the CPU cache must be flushed to DDR to ensure the NPU receives the correct data.
  for (size_t i = 0; i < model_->InputCount(); i++) {
      auto viplite_memory_info =
      device_context_->GetVipliteMemoryInfo(input_buffers_handles_[i]);
      if(viplite_memory_info->create_type == VIP_BUFFER_CREATE_ALLOC_MEM) {
        auto handle = viplite_adapter_api_.api().map_buffer(viplite_memory_info->buffer);
        memcpy(handle, viplite_memory_info->host_addr, viplite_memory_info->size);
        viplite_adapter_api_.api().unmap_buffer(viplite_memory_info->buffer);
      }
      if(viplite_memory_info->create_type == VIP_BUFFER_CREATE_ALLOC_MEM ||
         viplite_memory_info->create_type == VIP_BUFFER_CREATE_FROM_USER_MEM ) {
            viplite_adapter_api_.api().flush_buffer(viplite_memory_info->buffer,VIP_BUFFER_OPER_TYPE_FLUSH);
         }
  }
  LITERT_RETURN_IF_ERROR(model_->Run());
  //Once the output is written to DDR by the NPU,
  //the CPU cache must be invalidated to ensure correct data is read by the CPU.
  for (size_t i = 0; i < model_->OutputCount(); i++) {
    auto viplite_memory_info =
      device_context_->GetVipliteMemoryInfo(output_buffers_handles_[i]);
    if(viplite_memory_info->create_type == VIP_BUFFER_CREATE_ALLOC_MEM ||
         viplite_memory_info->create_type == VIP_BUFFER_CREATE_FROM_USER_MEM ) {
            viplite_adapter_api_.api().flush_buffer(viplite_memory_info->buffer,VIP_BUFFER_OPER_TYPE_INVALIDATE);
         }
      if(viplite_memory_info->create_type == VIP_BUFFER_CREATE_ALLOC_MEM) {
        auto handle = viplite_adapter_api_.api().map_buffer(viplite_memory_info->buffer);
        memcpy(viplite_memory_info->host_addr, handle, viplite_memory_info->size);
        viplite_adapter_api_.api().unmap_buffer(viplite_memory_info->buffer);
      }
  }
  return {};
}
