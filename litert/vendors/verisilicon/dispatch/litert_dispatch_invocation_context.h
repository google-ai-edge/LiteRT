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

#ifndef ODML_LITERT_LITERT_VENDORS_VERISILICON_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_VERISILICON_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/verisilicon//dispatch/litert_dispatch_device_context.h"

namespace litert::verisilicon {
class VipliteNetworkT {
public:
    using ModelPtr = std::unique_ptr<VipliteNetworkT>;
    using VipEnum = vip_enum;
    using VipBuffer = vip_buffer;
    using VipModel = vip_network;
    static litert::Expected<ModelPtr> CreateFromByteCode(
        litert::verisilicon::VipliteAdapterApi &viplite_adapter_api,
        const void *exec_bytecode_ptr,
        size_t exec_bytecode_size,
        int num_inputs, int num_outputs,
        LiteRtDispatchDeviceContextT::VpmNetworkParam* vpm_param);
    VipliteNetworkT(const VipliteNetworkT&) = delete;
    VipliteNetworkT& operator=(const VipliteNetworkT&) = delete;
    VipliteNetworkT(const litert::verisilicon::VipliteAdapterApi &viplite_adapter_api, const void *exec_bytecode_ptr,
        size_t exec_bytecode_size)
        : viplite_adapter_api_(viplite_adapter_api),
          network_(NULL),
          exec_bytecode_ptr_(exec_bytecode_ptr),
          exec_bytecode_size_(exec_bytecode_size),
          device_index_(0),
          core_index_(0), core_count_(1){};
    ~VipliteNetworkT();
    litert::Expected<void> Setup(LiteRtDispatchDeviceContextT::VpmNetworkParam* vpm_param);

    litert::Expected<void> Query(VipEnum property, void *value);
    litert::Expected<void> Set(VipEnum property, void *value);
    litert::Expected<void> Prepare();
    litert::Expected<void> Run();
    litert::Expected<void> Trigger();
    litert::Expected<void> Wait();
    litert::Expected<void> Cancel();
    litert::Expected<void> QueryInput(uint32_t index, VipEnum property, void *value);
    litert::Expected<void> QueryOutput(uint32_t index, VipEnum property, void *value);
    litert::Expected<void> SetInput(uint32_t index, VipBuffer vip_buffer);
    litert::Expected<void> SetOutput(uint32_t index, VipBuffer vip_buffer);
    litert::Expected<VipBuffer> GetInput(uint32_t index);
    litert::Expected<VipBuffer> GetOutput(uint32_t index);
    const uint32_t InputCount() const { return input_buffers_.size(); }
    const uint32_t OutputCount() const { return output_buffers_.size(); }

private:
    VipModel network_;
    const litert::verisilicon::VipliteAdapterApi &viplite_adapter_api_;
    uint32_t device_index_;
    uint32_t core_index_;
    uint32_t core_count_;
    const void *exec_bytecode_ptr_;
    size_t exec_bytecode_size_;
    std::vector<VipBuffer> input_buffers_;
    std::vector<VipBuffer> output_buffers_;
};
}  // namespace litert::verisilicon

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  static litert::Expected<Ptr> Create(
      litert::verisilicon::VipliteAdapterApi& viplite_adapter_api,
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs);

  ~LiteRtDispatchInvocationContextT(){};

  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);

  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

  litert::Expected<void> AttachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);
  litert::Expected<void> AttachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> DetachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);
  litert::Expected<void> DetachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> Invoke();

  void SetSchedulingInfo(const LiteRtSchedulingInfo* scheduling_info) {
    if (scheduling_info == nullptr) {
      scheduling_info_ = std::nullopt;
      return;
    }
    scheduling_info_ = *scheduling_info;
  }

  const LiteRtSchedulingInfo* GetSchedulingInfo() const {
    return scheduling_info_.has_value() ? &scheduling_info_.value() : nullptr;
  }
 private:
  LiteRtDispatchInvocationContextT(
      const litert::verisilicon::VipliteAdapterApi& viplite_adapter_api,
      LiteRtDispatchDeviceContext device_context,
      litert::verisilicon::VipliteNetworkT::ModelPtr model,
      int num_inputs, int num_outputs)
      : viplite_adapter_api_(viplite_adapter_api),
        device_context_(device_context),
        model_(std::move(model)){
            input_buffers_handles_.resize(num_inputs);
            output_buffers_handles_.resize(num_outputs);
        }

  const litert::verisilicon::VipliteAdapterApi& viplite_adapter_api_;
  LiteRtDispatchDeviceContext device_context_;
  litert::verisilicon::VipliteNetworkT::ModelPtr model_;
  std::vector<LiteRtTensorBufferHandle> input_buffers_handles_;
  std::vector<LiteRtTensorBufferHandle> output_buffers_handles_;
  std::optional<LiteRtSchedulingInfo> scheduling_info_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_VERISILICON_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
