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

#ifndef ODML_LITERT_LITERT_VENDORS_AMLOGIC_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_AMLOGIC_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/amlogic/core/nnsdk_func.h"
#include "litert/vendors/amlogic/dispatch/litert_dispatch_device_context.h"

class LiteRtDispatchInvocationContextT {
public:
    using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

    ~LiteRtDispatchInvocationContextT();

    static litert::Expected<Ptr> Create(
        LiteRtDispatchDeviceContextT &device_context,
        const LiteRtMemBuffer *exec_bytecode_buffer, const char *function_name,
        int num_inputs, int num_outputs);

    litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
        int input_index, const LiteRtRankedTensorType &tensor_type);
    litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
        int output_index, const LiteRtRankedTensorType &tensor_type);

    litert::Expected<void> AttachInput(
        int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);
    litert::Expected<void> AttachOutput(
        int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);

    litert::Expected<void> DetachInput(
        int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);
    litert::Expected<void> DetachOutput(
        int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);

    litert::Expected<void> Execute();

private:
    LiteRtDispatchInvocationContextT(
        LiteRtDispatchDeviceContextT &device_context, int num_inputs,
        int num_outputs);

    litert::Expected<void> AttachInputBuffer(
        LiteRtTensorBufferHandle tensor_buffer_handle, int graph_index);

    litert::Expected<void> AttachOutputBuffer(
        LiteRtTensorBufferHandle tensor_buffer_handle, int graph_index);

    /**
     * @brief amlnn_init from the embedded ADLA bytecode, then BindSubgraph.
     * @param subgraph_idx Partition index carried in the call-info string.
     */
    litert::Expected<void> CreateModel(int subgraph_idx);

    /**
     * @brief Select subgraph (if multi) and refresh IO attrs for this partition.
     * @param subgraph_idx Partition index for amlnn_select_subgraph.
     * @note Matches NNSDK sample: select_subgraph -> query IN_OUT_NUM / ATTR.
     */
    litert::Expected<void> BindSubgraph(int subgraph_idx);

    /**
     * @brief Run inference on the bound NNSDK2 context.
     */
    litert::Expected<void> ExecuteModel();

    LiteRtDispatchDeviceContextT &device_context_;

    std::shared_ptr<aml_nnsdk::AML_NN> aml_nn_;
    std::vector<uint8_t> adla_bin_;         /**< Copy of the shared ADLA bytecode. */
    size_t adla_bin_size_ = 0;
    int subgraph_idx_ = 0;                  /**< This partition's subgraph index. */
    int subgraph_num_ = 1;                  /**< Queried AMLNN_QUERY_SUBGRAPH_NUM. */
    amlnn_input_output_num v2_io_num_ = {};
    std::vector<amlnn_input> v2_inputs_;
    std::vector<uint32_t> input_attr_sizes_;
    std::vector<uint32_t> output_attr_sizes_;

    std::vector<void *> input_buffer_ptr_;
    std::vector<size_t> input_buffer_sizes_;
    std::vector<void *> output_buffer_ptr;
    std::vector<size_t> output_buffer_sizes_;

    std::vector<LiteRtTensorBufferHandle> input_buffer_handles_;
    std::vector<LiteRtTensorBufferHandle> output_buffer_handles_;
};

#endif // ODML_LITERT_LITERT_VENDORS_AMLOGIC_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
