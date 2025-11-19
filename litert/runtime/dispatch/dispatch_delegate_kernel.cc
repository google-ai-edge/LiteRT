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

#include "litert/runtime/dispatch/dispatch_delegate_kernel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/container/node_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_tflite_error_status_builder.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/runtime/dispatch/dispatch_opaque_options.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/runtime/tensor_buffer_requirements.h"
#include "litert/runtime/tfl_utils.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/core/c/c_api_opaque.h"

namespace litert::internal {

DispatchDelegateKernel::~DispatchDelegateKernel() {
  // Detach all buffer handles from invocation contexts.
  {
    for (const auto& [tfl_tensor, tensor_info] : tensor_buffer_infos_) {
      auto itpc_it = io_tensors_port_connections_.find(tfl_tensor);
      if (itpc_it == io_tensors_port_connections_.end()) {
        LITERT_LOG(LITERT_ERROR,
                   "IO tensor port connections not found for tensor %p",
                   tfl_tensor);
        continue;
      }
      const auto& port_connections = itpc_it->second;

      auto tbi_it = tensor_buffer_infos_.find(tfl_tensor);
      if (tbi_it == tensor_buffer_infos_.end()) {
        // Tensor buffer initialized but never consumed will not present in
        // tensor_buffer_infos_.
        LITERT_LOG(LITERT_WARNING, "Tensor buffer info not found for tensor %p",
                   tfl_tensor);
        continue;
      }
      const auto& tensor_buffer_info = tbi_it->second;
      for (auto& pc : port_connections) {
        auto* invocation_context = node_invocation_contexts_[pc.node_idx];
        if (pc.is_input_port) {
          (void)LiteRtDispatchDetachInput(invocation_context, pc.port_idx,
                                          tensor_buffer_info.buffer_handle);
        } else {
          (void)LiteRtDispatchDetachOutput(invocation_context, pc.port_idx,
                                           tensor_buffer_info.buffer_handle);
        }
      }
    }
  }

  // Unregister all buffer handles.
  for (const auto& p : tensor_buffer_infos_) {
    auto& tensor_buffer_info = p.second;
    (void)LiteRtDispatchUnregisterTensorBuffer(
        device_context_, tensor_buffer_info.buffer_handle);
  }

  // Destroy all invocation contexts.
  for (auto invocation_context : node_invocation_contexts_) {
    (void)LiteRtDispatchInvocationContextDestroy(invocation_context);
  }
}

Expected<DispatchDelegateKernel::Ptr> DispatchDelegateKernel::Create(
    std::string&& graph_name, LiteRtEnvironmentOptions environment_options,
    LiteRtOptions options, LiteRtDispatchDeviceContext device_context) {
  int capabilities;
  if (auto status = LiteRtDispatchGetCapabilities(&capabilities);
      status != kLiteRtStatusOk) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Failed to get Dispatch API capabilities: %d", status));
  }

  bool async_dispatch = (capabilities & kLiteRtDispatchCapabilitiesAsync);
  if (async_dispatch) {
    LITERT_LOG(LITERT_INFO, "Found async dispatch capabilities");
  }

  return Ptr(new DispatchDelegateKernel(environment_options, options,
                                        std::move(graph_name), device_context,
                                        async_dispatch));
}

Expected<const void*> DispatchDelegateKernel::FindAllocBase() const {
  auto opaque_options =
      OpaqueOptions::WrapCObject(options_->options, OwnHandle::kNo);
  LITERT_ASSIGN_OR_RETURN(
      auto dispatch_options,
      FindOpaqueOptions<DispatchDelegateOptions>(opaque_options));
  return dispatch_options.GetAllocBase();
}

Expected<int> DispatchDelegateKernel::FindAllocBaseFd() const {
  auto opaque_options =
      OpaqueOptions::WrapCObject(options_->options, OwnHandle::kNo);
  LITERT_ASSIGN_OR_RETURN(
      auto dispatch_options,
      FindOpaqueOptions<DispatchDelegateOptions>(opaque_options));
  return dispatch_options.GetAllocBaseFd();
}

TfLiteStatus DispatchDelegateKernel::Init(
    TfLiteOpaqueContext* context, const TfLiteOpaqueDelegateParams* params) {
  LITERT_RETURN_IF_ERROR(
      InitHelper(context, *params),
      AsTfLiteStatus(_ << "Couldn't initialize the dispatch delegate kernel"));
  return kTfLiteOk;
}

TfLiteStatus DispatchDelegateKernel::Prepare(TfLiteOpaqueContext* context,
                                             TfLiteOpaqueNode* node) {
  if (auto status = PrepareHelper(context, node); !status) {
    LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus DispatchDelegateKernel::Eval(TfLiteOpaqueContext* context,
                                          TfLiteOpaqueNode* node) {
  if (auto status = EvalHelper(context, node); !status) {
    LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
    return kTfLiteError;
  }
  return kTfLiteOk;
}

litert::Expected<void> DispatchDelegateKernel::StartMetricsCollection(
    int detail_level) {
  for (auto invocation_context : node_invocation_contexts_) {
    LITERT_RETURN_IF_ERROR(
        LiteRtDispatchStartMetricsCollection(invocation_context, detail_level));
  }
  return {};
}

Expected<LiteRtMetricsT> DispatchDelegateKernel::StopMetricsCollection() {
  std::vector<LiteRtMetricsT::Metric> metrics;

  for (auto invocation_context : node_invocation_contexts_) {
    LiteRtDispatchMetrics dispatch_metrics;
    LITERT_RETURN_IF_ERROR(LiteRtDispatchStopMetricsCollection(
        invocation_context, &dispatch_metrics));

    absl::Cleanup metrics_cleanup = [&dispatch_metrics] {
      if (auto status = LiteRtDispatchDestroyMetrics(dispatch_metrics);
          status != kLiteRtStatusOk) {
        LITERT_LOG(LITERT_ERROR, "Failed to destroy metrics: %d", status);
      }
    };

    int num_metrics = 0;
    LITERT_RETURN_IF_ERROR(
        LiteRtDispatchGetNumMetrics(dispatch_metrics, &num_metrics));

    for (int i = 0; i < num_metrics; ++i) {
      LiteRtMetric metric;
      LITERT_RETURN_IF_ERROR(
          LiteRtDispatchGetMetric(dispatch_metrics, i, &metric));
      metrics.push_back({/*.name=*/metric.name, /*.value=*/metric.value});
    }
  }

  return LiteRtMetricsT{/*.metrics=*/std::move(metrics)};
}

// /////////////////////////////////////////////////////////////////////////////

Expected<void> DispatchDelegateKernel::InitHelper(
    TfLiteOpaqueContext* context, const TfLiteOpaqueDelegateParams& params) {
  LITERT_ASSIGN_OR_RETURN(
      auto buffer_context,
      LiteRtExternalLiteRtBufferContextT::GetInstance(context));
  std::swap(buffer_context_, buffer_context);

  // Build the graph.

  LITERT_ASSIGN_OR_RETURN(auto nodes, GetNodes(context, params));
  std::swap(nodes_, nodes);

  for (auto& node : nodes_) {
    LITERT_ASSIGN_OR_RETURN(auto node_invocation_contexts,
                            CreateNodeInvocationContext(context, node));
    node_invocation_contexts_.push_back(node_invocation_contexts);
  }

  // Store tensor IDs for later pointer refresh
  input_tensor_ids_.clear();
  input_tensor_ids_.reserve(params.input_tensors->size);
  for (int i = 0; i < params.input_tensors->size; ++i) {
    input_tensor_ids_.push_back(params.input_tensors->data[i]);
  }

  output_tensor_ids_.clear();
  output_tensor_ids_.reserve(params.output_tensors->size);
  for (int i = 0; i < params.output_tensors->size; ++i) {
    output_tensor_ids_.push_back(params.output_tensors->data[i]);
  }

  LITERT_ASSIGN_OR_RETURN(auto input_tensors,
                          GetTensors(context, *params.input_tensors));

  LITERT_ASSIGN_OR_RETURN(auto output_tensors,
                          GetTensors(context, *params.output_tensors));

  LITERT_ASSIGN_OR_RETURN(
      auto internal_tensors,
      GetInternalTensors(context, nodes_, input_tensors, output_tensors));

  // Store internal tensor IDs
  for (auto* node : nodes_) {
    const int* node_inputs = nullptr;
    int num_inputs = 0;
    TfLiteOpaqueNodeInputs(node, &node_inputs, &num_inputs);

    const int* node_outputs = nullptr;
    int num_outputs = 0;
    TfLiteOpaqueNodeOutputs(node, &node_outputs, &num_outputs);

    // Check if any node inputs/outputs are internal tensors
    for (int i = 0; i < num_inputs; ++i) {
      int tensor_id = node_inputs[i];
      auto* tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
      if (std::find(internal_tensors.begin(), internal_tensors.end(), tensor) !=
          internal_tensors.end()) {
        if (std::find(internal_tensor_ids_.begin(), internal_tensor_ids_.end(),
                      tensor_id) == internal_tensor_ids_.end()) {
          internal_tensor_ids_.push_back(tensor_id);
        }
      }
    }
    for (int i = 0; i < num_outputs; ++i) {
      int tensor_id = node_outputs[i];
      auto* tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
      if (std::find(internal_tensors.begin(), internal_tensors.end(), tensor) !=
          internal_tensors.end()) {
        if (std::find(internal_tensor_ids_.begin(), internal_tensor_ids_.end(),
                      tensor_id) == internal_tensor_ids_.end()) {
          internal_tensor_ids_.push_back(tensor_id);
        }
      }
    }
  }

  LITERT_RETURN_IF_ERROR(ComputeTensorPortConnections(context));

  // Compute requirements across the graph.
  LITERT_RETURN_IF_ERROR(ComputeRequirements(context));

  return {};
}

Expected<void> DispatchDelegateKernel::PrepareHelper(
    TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  return {};
}

Expected<void> DispatchDelegateKernel::EvalHelper(TfLiteOpaqueContext* context,
                                                  TfLiteOpaqueNode* node) {
  LITERT_RETURN_IF_ERROR(AllocateTensorBuffersIfNeeded(context));
  LITERT_RETURN_IF_ERROR(AttachBuffersToInvocationContextsIfNeeded(context));

  // Copy input buffers from CPU, if needed.
  for (int tensor_id : input_tensor_ids_) {
    auto* tfl_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (!tfl_tensor) {
      continue;
    }

    if (auto& tensor_buffer_info =
            tensor_buffer_infos_.find(tfl_tensor)->second;
        tensor_buffer_info.maybe_sync_with_cpu) {
      void* tensor_data = TfLiteOpaqueTensorData(tfl_tensor);
      // Note that tensor_data may be null if the TFL allocated decided to not
      // allocate heap memory for the tensor and, instead, just use the attached
      // tensor buffer. No memcpy is necessary in that case.
      if (tensor_data) {
        size_t buffer_size = tensor_buffer_info.tensor_buffer_used_size;
        LITERT_ASSIGN_OR_RETURN(void* host_buffer,
                                tensor_buffer_info.tensor_buffer->Lock(
                                    kLiteRtTensorBufferLockModeRead));
        std::memcpy(host_buffer, tensor_data, buffer_size);
        LITERT_RETURN_IF_ERROR(tensor_buffer_info.tensor_buffer->Unlock());
      }
    }
  }

  if (async_dispatch_ && buffer_context_->IsAsyncExecutionMode()) {
    LITERT_RETURN_IF_ERROR(ScheduleAsyncExecution(context));
  } else {
    LITERT_RETURN_IF_ERROR(ScheduleSyncExecution(context));
  }

  for (int tensor_id : output_tensor_ids_) {
    auto* tfl_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (!tfl_tensor) {
      continue;
    }

    if (auto& tensor_buffer_info =
            tensor_buffer_infos_.find(tfl_tensor)->second;
        tensor_buffer_info.maybe_sync_with_cpu) {
      void* tensor_data = TfLiteOpaqueTensorData(tfl_tensor);
      // Note that tensor_data may be null if the TFL allocated decided to not
      // allocate heap memory for the tensor and, instead, just use the attached
      // tensor buffer. No memcpy is necessary in that case.
      if (tensor_data) {
        size_t buffer_size = tensor_buffer_info.tensor_buffer_used_size;
        LITERT_ASSIGN_OR_RETURN(void* host_buffer,
                                tensor_buffer_info.tensor_buffer->Lock(
                                    kLiteRtTensorBufferLockModeWrite));
        std::memcpy(tensor_data, host_buffer, buffer_size);
        LITERT_RETURN_IF_ERROR(tensor_buffer_info.tensor_buffer->Unlock());
      }
    }
  }

  return {};
}

// /////////////////////////////////////////////////////////////////////////////

Expected<std::vector<TfLiteOpaqueNode*>> DispatchDelegateKernel::GetNodes(
    TfLiteOpaqueContext* context, const TfLiteOpaqueDelegateParams& params) {
  auto num_nodes = params.nodes_to_replace->size;

  std::vector<TfLiteOpaqueNode*> nodes;
  nodes.reserve(num_nodes);

  for (auto i = 0; i < num_nodes; ++i) {
    auto node_id = params.nodes_to_replace->data[i];
    TfLiteOpaqueNode* node;
    TfLiteOperator* op;
    if (auto status = TfLiteOpaqueContextGetNodeAndRegistration(
            context, node_id, &node, &op);
        status != kTfLiteOk) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrFormat("Failed to get node and registration: %d", status));
    }
    nodes.push_back(node);
  }

  return nodes;
}

Expected<std::vector<TfLiteOpaqueTensor*>> DispatchDelegateKernel::GetTensors(
    const TfLiteOpaqueContext* context, const TfLiteIntArray& tensor_ids) {
  auto num_tensors = tensor_ids.size;

  std::vector<TfLiteOpaqueTensor*> tensors;
  tensors.reserve(num_tensors);

  for (auto i = 0; i < num_tensors; ++i) {
    auto tensor_id = tensor_ids.data[i];
    auto* tfl_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (!tfl_tensor) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Tensor not found");
    }
    tensors.push_back(tfl_tensor);
  }

  return tensors;
}

Expected<std::vector<TfLiteOpaqueTensor*>>
DispatchDelegateKernel::GetInternalTensors(
    TfLiteOpaqueContext* context, const std::vector<TfLiteOpaqueNode*>& nodes,
    const std::vector<TfLiteOpaqueTensor*>& input_tensors,
    const std::vector<TfLiteOpaqueTensor*>& output_tensors) {
  absl::flat_hash_set<TfLiteOpaqueTensor*> io_tensors;
  io_tensors.insert(input_tensors.begin(), input_tensors.end());
  io_tensors.insert(output_tensors.begin(), output_tensors.end());

  absl::flat_hash_set<TfLiteOpaqueTensor*> tensors;

  for (auto& node : nodes) {
    auto num_node_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
    for (auto i = 0; i < num_node_inputs; ++i) {
      auto* tfl_tensor = const_cast<TfLiteOpaqueTensor*>(
          TfLiteOpaqueNodeGetInput(context, node, i));
      if (!tfl_tensor) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Tensor not found");
      }
      if (io_tensors.find(tfl_tensor) == io_tensors.end()) {
        tensors.insert(tfl_tensor);
      }
    }
    auto num_node_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
    for (auto i = 0; i < num_node_outputs; ++i) {
      auto* tfl_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
      if (!tfl_tensor) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Tensor not found");
      }
      if (io_tensors.find(tfl_tensor) == io_tensors.end()) {
        tensors.insert(tfl_tensor);
      }
    }
  }

  std::vector<TfLiteOpaqueTensor*> tensors_vec;
  tensors_vec.reserve(tensors.size());
  std::copy(tensors.begin(), tensors.end(), std::back_inserter(tensors_vec));

  return tensors_vec;
}

Expected<LiteRtDispatchInvocationContext>
DispatchDelegateKernel::CreateNodeInvocationContext(
    TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  const void* init_data;
  int init_data_size;
  if (auto status = TfLiteOpaqueNodeGetCustomInitialData(node, &init_data,
                                                         &init_data_size);
      status != kTfLiteOk) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Failed to get custom initial data: %d", status));
  }
  if (!init_data || !init_data_size) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Found custom op with missing initial data");
  }

  BufferRef<uint8_t> custom_opts(init_data, init_data_size);

  // Read offset and size (relative to alloc_base) from the custom options (and
  // name).
  const auto dispatch_opts = GetDispatchOpOptions(custom_opts);
  if (dispatch_opts.bytecode_offset == 0) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Found dispatch op with missing bytecode offset");
  }

  // Find pointer to the start of the loaded model buffer.
  LITERT_ASSIGN_OR_RETURN(const void* alloc_base, FindAllocBase());
  LITERT_ASSIGN_OR_RETURN(const int alloc_fd, FindAllocBaseFd());

  // Get location of bytecode in the model buffer relative to alloc_base.
  LiteRtMemBuffer exec_bytecode_buffer = {
      /*.fd=*/alloc_fd,
      /*.base_addr=*/alloc_base,
      /*.offset=*/dispatch_opts.bytecode_offset,
      /*.size=*/dispatch_opts.bytecode_size};
  const auto& function_name = dispatch_opts.name;

  auto num_node_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
  auto num_node_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);

  LiteRtDispatchInvocationContext invocation_context;
  if (auto status = LiteRtDispatchInvocationContextCreate(
          device_context_, kLiteRtDispatchExecutableTypeMlModel,
          &exec_bytecode_buffer, function_name.data(), num_node_inputs,
          num_node_outputs, &invocation_context);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to create invocation context");
  }

  // Apply dispatch annotations from the compiled model
  if (buffer_context_) {
    const auto& annotations = buffer_context_->GetDispatchAnnotations();
    if (!annotations.empty()) {
      // Get the dispatch graph from the invocation context
      LiteRtDispatchGraph graph = nullptr;
      if (LiteRtDispatchInvocationContextGetGraph(invocation_context, &graph) ==
              kLiteRtStatusOk &&
          graph != nullptr) {
        // Apply annotations to the dispatch graph
        // TODO (b/436921503): Dispatch graph should map to compiled model's
        // subgraph
        for (const auto& [key, value] : annotations) {
          if (auto status = LiteRtDispatchAnnotateGraph(&graph, key.c_str(),
                                                        value.c_str());
              status != kLiteRtStatusOk) {
            LITERT_LOG(LITERT_WARNING,
                       "Failed to apply dispatch annotation %s=%s: %d",
                       key.c_str(), value.c_str(), status);
          }
        }
      }
    }
  }

  return invocation_context;
}

Expected<void> DispatchDelegateKernel::ComputeTensorPortConnections(
    TfLiteOpaqueContext* context) {
  for (auto node_idx = 0; node_idx < nodes_.size(); ++node_idx) {
    auto& node = nodes_[node_idx];

    auto num_node_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
    for (auto i = 0; i < num_node_inputs; ++i) {
      auto* tfl_tensor = const_cast<TfLiteOpaqueTensor*>(
          TfLiteOpaqueNodeGetInput(context, node, i));
      io_tensors_port_connections_[tfl_tensor].push_back(
          {/*.node_idx=*/node_idx, /*port_idx*/ i, /*is_input_port=*/true});
    }

    auto num_node_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
    for (auto i = 0; i < num_node_outputs; ++i) {
      auto* tfl_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
      io_tensors_port_connections_[tfl_tensor].push_back(
          {/*.node_idx=*/node_idx, /*port_idx*/ i, /*is_input_port=*/false});
    }
  }

  return {};
}

Expected<LiteRtTensorBufferRequirementsPtr>
DispatchDelegateKernel::GetBufferRequirements(int node_idx,
                                              TfLiteOpaqueTensor* io_tfl_tensor,
                                              int io_tensor_index,
                                              bool is_input) const {
  auto* invocation_context = node_invocation_contexts_[node_idx];

  LITERT_ASSIGN_OR_RETURN(auto tensor_type, ConvertTensorType(io_tfl_tensor));
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);

  LiteRtTensorBufferRequirements tensor_buffer_requirements;
  if (is_input) {
    LITERT_RETURN_IF_ERROR(LiteRtDispatchGetInputRequirements(
        invocation_context, /*input_index=*/io_tensor_index,
        &litert_tensor_type, &tensor_buffer_requirements))
        << "Failed to get input tensor requirements";
  } else {
    LITERT_RETURN_IF_ERROR(LiteRtDispatchGetOutputRequirements(
        invocation_context, /*output_index=*/io_tensor_index,
        &litert_tensor_type, &tensor_buffer_requirements));
  }
  // Check for MediaTek's contradictory buffer size for Float32 tensors
  if (litert_tensor_type.element_type == kLiteRtElementTypeFloat32) {
    size_t expected_elements = 1;
    for (int i = 0; i < litert_tensor_type.layout.rank; ++i) {
      expected_elements *= litert_tensor_type.layout.dimensions[i];
    }
    size_t expected_buffer_size = expected_elements * 4;  // Float32 is 4 bytes
    size_t reported_buffer_size = tensor_buffer_requirements->BufferSize();

    if (reported_buffer_size != expected_buffer_size) {
      LITERT_LOG(
          LITERT_WARNING,
          "MediaTek dispatch API returned contradictory buffer requirements "
          "for Float32 tensor %p:\n"
          "  Tensor type: Float32 (element_type=%d), dims=[%s], expected "
          "size=%zu bytes\n"
          "  But dispatch API returned: %zu bytes (%.1f bytes per element "
          "instead of 4)",
          io_tfl_tensor, litert_tensor_type.element_type,
          [&litert_tensor_type]() -> std::string {
            std::string dims;
            for (int i = 0; i < litert_tensor_type.layout.rank; ++i) {
              if (i > 0) dims += ",";
              dims += std::to_string(litert_tensor_type.layout.dimensions[i]);
            }
            return dims;
          }()
                                         .c_str(),
          expected_buffer_size, reported_buffer_size,
          static_cast<float>(reported_buffer_size) / expected_elements);
    }
  }

  return LiteRtTensorBufferRequirementsPtr(tensor_buffer_requirements);
}

Expected<void> DispatchDelegateKernel::ComputeRequirements(
    TfLiteOpaqueContext* context) {
  for (auto node_idx = 0; node_idx < nodes_.size(); ++node_idx) {
    auto& node = nodes_[node_idx];
    auto num_node_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
    for (auto i = 0; i < num_node_inputs; ++i) {
      auto* tfl_tensor = const_cast<TfLiteOpaqueTensor*>(
          TfLiteOpaqueNodeGetInput(context, node, i));
      if (!tfl_tensor) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Tensor not found");
      }
      LITERT_ASSIGN_OR_RETURN(
          LiteRtTensorBufferRequirementsPtr buffer_requirements,
          GetBufferRequirements(node_idx, tfl_tensor, i, /*is_input=*/true));
      LITERT_RETURN_IF_ERROR(buffer_context_->RegisterBufferRequirements(
          tfl_tensor, std::move(buffer_requirements)));
    }

    auto num_node_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
    for (auto i = 0; i < num_node_outputs; ++i) {
      auto* tfl_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
      if (!tfl_tensor) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Tensor not found");
      }
      LITERT_ASSIGN_OR_RETURN(
          LiteRtTensorBufferRequirementsPtr buffer_requirements,
          GetBufferRequirements(node_idx, tfl_tensor, i, /*is_input=*/false));
      LITERT_RETURN_IF_ERROR(buffer_context_->RegisterBufferRequirements(
          tfl_tensor, std::move(buffer_requirements)));
    }
  }

  return {};
}

Expected<void> DispatchDelegateKernel::AllocateTensorBuffersIfNeeded(
    TfLiteOpaqueContext* context) {
  absl::flat_hash_set<TfLiteOpaqueTensor*> io_tensors;
  // Get input tensors and add to io_tensors set
  for (int tensor_id : input_tensor_ids_) {
    auto* tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (tensor) {
      io_tensors.insert(tensor);
    }
  }
  // Get output tensors and add to io_tensors set
  for (int tensor_id : output_tensor_ids_) {
    auto* tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (tensor) {
      io_tensors.insert(tensor);
    }
  }

  // First allocate I/O tensor buffers. However, note that if a tensor buffer is
  // already associated with a given I/O, we use that one; otherwise allocate a
  // new tensor buffer.
  //
  // NOTE: Any allocated I/O tensor buffer is stored in and owned by
  // buffer_context_, so that the tensor buffer can may be retrieved by a
  // downstream delegate kernel or by the upper-level runtime. This class will
  // also keep a non-owned alias of I/O tensor buffers in tensor_buffer_infos_,
  // for internal processing.

  std::set<LiteRtTensorBufferHandle> unused_buffer_handles;

  auto allocate_and_register =
      [this, context, &unused_buffer_handles,
       &io_tensors](auto* tfl_tensor) -> Expected<void> {
    auto iter = tensor_buffer_infos_.find(tfl_tensor);
    if (iter != tensor_buffer_infos_.end()) {
      auto& tensor_buffer_info = iter->second;

      // If a new tensor buffer was attached to an I/O TFL tensor (e.g., user
      // has supplied new inputs/outputs to the model, then we need to use that
      // one.
      if (io_tensors.find(tfl_tensor) == io_tensors.end()) {
        return {};
      }

      LITERT_ASSIGN_OR_RETURN(LiteRtTensorBufferPtr tensor_buffer,
                              buffer_context_->GetTensorBuffer(tfl_tensor));
      if (tensor_buffer == tensor_buffer_info.tensor_buffer) {
        return {};
      }

      // The tensor buffer associated with tfl_tensor has changed. Consequently,
      // we must detach the old tensor buffer from the model and attach the new
      // one.

      const auto& port_connections =
          io_tensors_port_connections_.find(tfl_tensor)->second;
      for (auto& pc : port_connections) {
        auto* invocation_context = node_invocation_contexts_[pc.node_idx];
        if (pc.is_input_port) {
          LITERT_RETURN_IF_ERROR(
              LiteRtDispatchDetachInput(invocation_context, pc.port_idx,
                                        tensor_buffer_info.buffer_handle));
        } else {
          LITERT_RETURN_IF_ERROR(
              LiteRtDispatchDetachOutput(invocation_context, pc.port_idx,
                                         tensor_buffer_info.buffer_handle));
        }
      }

      tensor_buffer_info.attached = false;

      // Unregister the old tensor buffer with the Dispatch API. We do that at
      // the end after collecting all tensor buffer handles to unregister
      // because a tensor buffer may be connected to multiple TFL tensors
      // (once we implement a proper tensor buffer allocation algorithm).
      unused_buffer_handles.insert(tensor_buffer_info.buffer_handle);

      // Register the tensor buffer with the dispatch API.
      LITERT_RETURN_IF_ERROR(RegisterBufferWithDispatchApi(
          context, tfl_tensor, std::move(tensor_buffer)));

      return {};
    }

    // Allocate a tensor_buffer_info record for tfl_tensor. It will be used by
    // the calls below.
    auto& tensor_buffer_info = tensor_buffer_infos_[tfl_tensor];

    LiteRtTensorBufferT* litert_tensor_buffer = nullptr;
    if (auto tensor_buffer = buffer_context_->GetTensorBuffer(tfl_tensor);
        tensor_buffer) {
      litert_tensor_buffer = tensor_buffer->get();
    } else {
      LITERT_ASSIGN_OR_RETURN(LiteRtTensorBufferPtr new_tensor_buffer,
                              AllocateTensorBuffer(tfl_tensor));
      // Transfer ownership of the tensor buffer to buffer_context_.
      litert_tensor_buffer = new_tensor_buffer.get();
      LITERT_RETURN_IF_ERROR(buffer_context_->RegisterTensorBuffer(
          tfl_tensor, std::move(new_tensor_buffer)));
      size_t tfl_tensor_size = TfLiteOpaqueTensorByteSize(tfl_tensor);
      tensor_buffer_info.MarkAsMaybeSyncWithCpu(tfl_tensor_size);
    }

    // Register the tensor buffer with the dispatch API.
    LITERT_RETURN_IF_ERROR(RegisterBufferWithDispatchApi(
        context, tfl_tensor, LiteRtTensorBufferPtr(litert_tensor_buffer)));
    return {};
  };

  // Allocate buffers for input tensors
  for (int tensor_id : input_tensor_ids_) {
    auto* tfl_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (!tfl_tensor) {
      continue;
    }
    LITERT_RETURN_IF_ERROR(allocate_and_register(tfl_tensor));
  }

  // Allocate buffers for output tensors
  for (int tensor_id : output_tensor_ids_) {
    auto* tfl_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (!tfl_tensor) {
      continue;
    }
    LITERT_RETURN_IF_ERROR(allocate_and_register(tfl_tensor));
  }

  // Then allocate intermediate tensor buffers. They are always allocated,
  // without attempt to share them when possible.
  //
  // NOTE: Internal tensor buffers are be stored in and owned by
  // tensor_buffer_infos_. There is no need to store them in buffer_context_
  // since they will not need to be accessed outside of the scope of this class
  // instance.
  //
  // TODO: implement logic to share tensor buffers across intermediate tensors,
  // whenever possible.
  // Allocate buffers for internal tensors
  for (int tensor_id : internal_tensor_ids_) {
    auto* tfl_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (!tfl_tensor) {
      continue;
    }

    if (auto iter = tensor_buffer_infos_.find(tfl_tensor);
        iter != tensor_buffer_infos_.end()) {
      // A tensor buffer was already allocated for tfl_tensor.
      continue;
    }

    // Allocate a tensor_buffer_info record for tfl_tensor. It will be used
    // (internally) by the calls below.
    auto& tensor_buffer_info = tensor_buffer_infos_[tfl_tensor];
    (void)tensor_buffer_info;

    // For now we just allocate tensor buffers as needed, without attempting to
    // reuse them.
    LITERT_ASSIGN_OR_RETURN(LiteRtTensorBufferPtr tensor_buffer,
                            AllocateTensorBuffer(tfl_tensor));
    // Register an allocated tensor buffer with the dispatch API.
    LITERT_RETURN_IF_ERROR(RegisterBufferWithDispatchApi(
        context, tfl_tensor, std::move(tensor_buffer)));
  }

  for (auto buffer_handle : unused_buffer_handles) {
    LITERT_RETURN_IF_ERROR(
        LiteRtDispatchUnregisterTensorBuffer(device_context_, buffer_handle));
  }

  return {};
}

Expected<LiteRtTensorBufferPtr> DispatchDelegateKernel::AllocateTensorBuffer(
    TfLiteOpaqueTensor* tfl_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto requirements_ptr,
                          buffer_context_->GetBufferRequirements(tfl_tensor));
  const auto& supported_types = requirements_ptr->SupportedBufferTypes();

  LiteRtTensorBufferType buffer_type = supported_types[0];
  LITERT_ASSIGN_OR_RETURN(LiteRtRankedTensorType litert_tensor_type,
                          ConvertTensorType(tfl_tensor));
  size_t buffer_size = requirements_ptr->BufferSize();

  LiteRtTensorBufferT* tensor_buffer;
  LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
      buffer_context_->GetEnvironment(), buffer_type, &litert_tensor_type,
      buffer_size, &tensor_buffer));
  return LiteRtTensorBufferPtr(tensor_buffer);
}

Expected<void> DispatchDelegateKernel::RegisterBufferWithDispatchApi(
    TfLiteOpaqueContext* context, TfLiteOpaqueTensor* tfl_tensor,
    LiteRtTensorBufferPtr&& tensor_buffer) {
  LiteRtTensorBufferHandle buffer_handle = 0;
  if (tensor_buffer && tensor_buffer.get()) {
    LITERT_RETURN_IF_ERROR(LiteRtDispatchRegisterTensorBuffer(
        device_context_, tensor_buffer.get(), &buffer_handle));
  } else {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid tensor buffer");
  }

  auto iter = tensor_buffer_infos_.find(tfl_tensor);
  if (iter == tensor_buffer_infos_.end()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "TensorInfo not found");
  }

  auto& tensor_buffer_info = iter->second;
  tensor_buffer_info.tensor_buffer = std::move(tensor_buffer);
  tensor_buffer_info.buffer_handle = buffer_handle;

  // Check if it's an input tensor
  for (int tensor_id : input_tensor_ids_) {
    auto* tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (tensor == tfl_tensor) {
      tensor_idx_to_handle_[tensor_id] = buffer_handle;
      return {};
    }
  }

  // Check if it's an output tensor
  for (int tensor_id : output_tensor_ids_) {
    auto* tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (tensor == tfl_tensor) {
      tensor_idx_to_handle_[tensor_id] = buffer_handle;
      return {};
    }
  }

  // Check if it's an internal tensor
  for (int tensor_id : internal_tensor_ids_) {
    auto* tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (tensor == tfl_tensor) {
      tensor_idx_to_handle_[tensor_id] = buffer_handle;
      return {};
    }
  }

  return {};
}

Expected<void>
DispatchDelegateKernel::AttachBuffersToInvocationContextsIfNeeded(
    TfLiteOpaqueContext* context) {
  for (auto node_idx = 0; node_idx < nodes_.size(); ++node_idx) {
    auto* node = nodes_[node_idx];
    auto invocation_context = node_invocation_contexts_[node_idx];

    // Process inputs using tensor ID to handle mapping
    const int* input_indices = nullptr;
    int num_node_inputs = 0;
    TfLiteOpaqueNodeInputs(node, &input_indices, &num_node_inputs);
    for (auto i = 0; i < num_node_inputs; ++i) {
      int tensor_idx = input_indices[i];

      auto* tfl_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(context, tensor_idx);
      if (!tfl_tensor) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Tensor not found for input index");
      }

      auto tensor_info_iter = tensor_buffer_infos_.find(tfl_tensor);
      if (tensor_info_iter == tensor_buffer_infos_.end()) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Tensor info not found for input tensor");
      }

      auto& tensor_buffer_info = tensor_info_iter->second;
      if (tensor_buffer_info.attached) {
        continue;
      }

      // Look up buffer handle by tensor ID
      auto handle_iter = tensor_idx_to_handle_.find(tensor_idx);
      if (handle_iter == tensor_idx_to_handle_.end()) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Buffer handle not found for input tensor");
      }

      LiteRtTensorBufferHandle buffer_handle = handle_iter->second;
      LITERT_RETURN_IF_ERROR(
          LiteRtDispatchAttachInput(invocation_context, i, buffer_handle));
    }

    // Process outputs using tensor ID to handle mapping
    const int* output_indices = nullptr;
    int num_node_outputs = 0;
    TfLiteOpaqueNodeOutputs(node, &output_indices, &num_node_outputs);
    for (auto i = 0; i < num_node_outputs; ++i) {
      int tensor_idx = output_indices[i];

      auto* tfl_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(context, tensor_idx);
      if (!tfl_tensor) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Tensor not found for output index");
      }

      auto tensor_info_iter = tensor_buffer_infos_.find(tfl_tensor);
      if (tensor_info_iter == tensor_buffer_infos_.end()) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Tensor info not found for output tensor");
      }

      auto& tensor_buffer_info = tensor_info_iter->second;
      if (tensor_buffer_info.attached) {
        continue;
      }

      // Look up buffer handle by tensor ID
      auto handle_iter = tensor_idx_to_handle_.find(tensor_idx);
      if (handle_iter == tensor_idx_to_handle_.end()) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Buffer handle not found for output tensor");
      }

      LiteRtTensorBufferHandle buffer_handle = handle_iter->second;
      LITERT_RETURN_IF_ERROR(
          LiteRtDispatchAttachOutput(invocation_context, i, buffer_handle));
    }
  }

  // Mark buffers as attached only at the end since a given buffer may be
  // attached to multiple invocation context I/Os.
  for (auto& item : tensor_buffer_infos_) {
    item.second.attached = true;
  }

  return {};
}

Expected<void> DispatchDelegateKernel::ScheduleAsyncExecution(
    TfLiteOpaqueContext* context) {
  std::vector<LiteRtEvent> output_events;

  // Run NPU bytecodes asynchronously and in topological order.
  for (auto node_idx = 0; node_idx < nodes_.size(); ++node_idx) {
    auto* node = nodes_[node_idx];
    auto invocation_context = node_invocation_contexts_[node_idx];

    auto num_node_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
    for (auto i = 0; i < num_node_inputs; ++i) {
      auto* tfl_tensor = const_cast<TfLiteOpaqueTensor*>(
          TfLiteOpaqueNodeGetInput(context, node, i));
      auto& tensor_buffer_info = tensor_buffer_infos_.find(tfl_tensor)->second;
      if (tensor_buffer_info.tensor_buffer->HasEvent()) {
        LITERT_ASSIGN_OR_RETURN(LiteRtEventT * event,
                                tensor_buffer_info.tensor_buffer->GetEvent());
        LITERT_RETURN_IF_ERROR(
            LiteRtDispatchAttachInputEvent(invocation_context, i, event));
      }
    }

    auto num_node_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
    output_events.resize(num_node_outputs);
    LITERT_RETURN_IF_ERROR(LiteRtDispatchInvokeAsync(
        invocation_context, output_events.size(), output_events.data()));

    for (auto i = 0; i < num_node_outputs; ++i) {
      auto* tfl_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
      auto& tensor_buffer_info = tensor_buffer_infos_.find(tfl_tensor)->second;
      tensor_buffer_info.tensor_buffer->SetEvent(output_events[i]);
    }
  }

  return {};
}

Expected<void> DispatchDelegateKernel::ScheduleSyncExecution(
    TfLiteOpaqueContext* context) {
  // Deal with any events attached to inputs.
  for (int tensor_id : input_tensor_ids_) {
    auto* tfl_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    if (!tfl_tensor) {
      continue;
    }
    auto& tensor_buffer_info = tensor_buffer_infos_.find(tfl_tensor)->second;
    if (tensor_buffer_info.tensor_buffer->HasEvent()) {
      LITERT_ASSIGN_OR_RETURN(LiteRtEventT * event,
                              tensor_buffer_info.tensor_buffer->GetEvent());

      // If the HW supports async dispatch, then we can simply pass those events
      // to the HW. Otherwise, we'll need to wait on those events here, on the
      // main CPU, before we can dispatch the HW, which may lead to deadlocks,
      // based on how the user code is written.
      if (async_dispatch_) {
        const auto& port_connections =
            io_tensors_port_connections_.find(tfl_tensor)->second;
        for (const auto& [node_idx, port_idx, is_input_port] :
             port_connections) {
          if (is_input_port) {
            auto* invocation_context = node_invocation_contexts_[node_idx];
            (void)LiteRtDispatchAttachInputEvent(invocation_context, port_idx,
                                                 event);
          }
        }

      } else {
        LITERT_LOG(LITERT_WARNING,
                   "CPU wait for an input tensor buffer event; this could "
                   "lead to deadlock");
        event->Wait(-1);
      }
    }
  }

  // Run NPU bytecodes synchronously and in topological order.
  for (auto* invocation_context : node_invocation_contexts_) {
    LITERT_RETURN_IF_ERROR(LiteRtDispatchInvoke(invocation_context));
  }

  return {};
}

}  // namespace litert::internal
