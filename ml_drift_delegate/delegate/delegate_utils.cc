// Copyright 2025 Google LLC.
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

#include "ml_drift_delegate/delegate/delegate_utils.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/container/node_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "ml_drift_delegate/delegate/composite/ir/custom_parsers.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/support.h"
#include "weight_loader/external_weight_loader_litert.h"
#include "tflite/core/c/common.h"
#include "tflite/core/subgraph.h"

namespace litert::ml_drift {
namespace {

bool IsLiteRtExecutionContext(TfLiteContext* context) {
  return context != nullptr && context->GetExternalContext != nullptr &&
         context->GetExternalContext(context, kTfLiteLiteRtBufferContext) !=
             nullptr;
}

}  // namespace

bool IsAsyncExecutionMode(TfLiteContext* context,
                          const LiteRtRuntimeContext* runtime_context) {
  bool is_async_execution_mode = false;
  auto* buffer_context = reinterpret_cast<LiteRtExternalLiteRtBufferContext>(
      context->GetExternalContext(context, kTfLiteLiteRtBufferContext));
  if (buffer_context != nullptr &&
      runtime_context->external_litert_buffer_context_is_async_execution_mode(
          buffer_context, &is_async_execution_mode) == kLiteRtStatusOk) {
    return is_async_execution_mode;
  }
  return false;
}

TfLiteIntArray* GetIrModelOpsToReplace(TfLiteContext* context,
                                       const MlDriftDelegateData& delegate_data,
                                       int start_node_index,
                                       int end_node_index) {
  ir::IrModelBuilderOptions ir_options;
  ir_options.enable_infinite_float_capping =
      delegate_data.options->enable_infinite_float_capping;
  ir_options.enable_reduced_precision = delegate_data.calculation_precision !=
                                        ::ml_drift::CalculationsPrecision::kF32;
  ir_options.allow_quant_ops = true;
  ir_options.start_node_index = start_node_index;
  ir_options.end_node_index = end_node_index;
  ir_options.max_delegated_partitions = 1;
  auto custom_parsers = ir::GetCustomParsers();
  return ir::GetOpsToReplace(context, ir_options, &custom_parsers);
}

// NOLINTNEXTLINE(*-runtime-unneeded-pointer-stability-check)
const std::unordered_map<size_t, size_t>& GetExternalTensorBufferIdentifiers(
    TfLiteContext* context, MlDriftDelegateData& delegate_data) {
  if (!IsLiteRtExecutionContext(context) &&
      delegate_data.weight_loader == nullptr) {
    return reinterpret_cast<const tflite::Subgraph*>(context->impl_)
        ->GetExternalTensorBufferIdentifiers();
  }

  auto existing_it =
      delegate_data.context_to_external_buffer_id_map.find(context);
  if (existing_it != delegate_data.context_to_external_buffer_id_map.end()) {
    return existing_it->second;
  }

  // NOLINTNEXTLINE(*-runtime-unneeded-pointer-stability-check)
  std::unordered_map<size_t, size_t>& external_buffer_id_map =
      delegate_data.context_to_external_buffer_id_map[context];
  if (delegate_data.weight_loader == nullptr || context == nullptr) {
    return external_buffer_id_map;
  }

  const absl::Span<const weight_loader::WeightInfo> weight_infos =
      delegate_data.weight_loader->GetWeightInfo();
  if (weight_infos.empty()) {
    return external_buffer_id_map;
  }

  uint16_t matched_subgraph_index = 0;
  bool found_subgraph = false;
  bool acquire_supported = false;
  if (context->AcquireSubgraphContext != nullptr &&
      context->ReleaseSubgraphContext != nullptr) {
    absl::flat_hash_set<uint16_t> candidate_subgraphs;
    for (const auto& info : weight_infos) {
      candidate_subgraphs.insert(info.subgraph_index);
    }
    for (uint16_t sg_idx : candidate_subgraphs) {
      TfLiteContext* acquired_context = nullptr;
      if (context->AcquireSubgraphContext(context, sg_idx, &acquired_context) ==
          kTfLiteOk) {
        acquire_supported = true;
        context->ReleaseSubgraphContext(context, sg_idx);
        if (acquired_context == context) {
          matched_subgraph_index = sg_idx;
          found_subgraph = true;
          break;
        }
      }
    }
  }
  if (!acquire_supported) {
    found_subgraph = true;
    matched_subgraph_index = 0;
  }
  if (!found_subgraph) {
    return external_buffer_id_map;
  }

  for (const auto& info : weight_infos) {
    if (info.subgraph_index == matched_subgraph_index &&
        info.external_buffer_id != 0) {
      external_buffer_id_map.emplace(
          static_cast<size_t>(info.tensor_index),
          static_cast<size_t>(info.external_buffer_id));
    }
  }
  return external_buffer_id_map;
}

// NOLINTNEXTLINE(*-runtime-unneeded-pointer-stability-check)
const std::unordered_map<size_t, size_t>& GetTensorBufferIdentifiers(
    TfLiteContext* context, MlDriftDelegateData& delegate_data) {
  if (!IsLiteRtExecutionContext(context)) {
    return reinterpret_cast<const tflite::Subgraph*>(context->impl_)
        ->GetTensorBufferIdentifiers();
  }

  auto existing_it = delegate_data.context_to_buffer_id_map.find(context);
  if (existing_it != delegate_data.context_to_buffer_id_map.end()) {
    return existing_it->second;
  }

  // NOLINTNEXTLINE(*-runtime-unneeded-pointer-stability-check)
  const std::unordered_map<size_t, size_t>& external_buffer_ids =
      GetExternalTensorBufferIdentifiers(context, delegate_data);
  // NOLINTNEXTLINE(*-runtime-unneeded-pointer-stability-check)
  std::unordered_map<size_t, size_t>& buffer_id_map =
      delegate_data.context_to_buffer_id_map[context];
  if (context == nullptr || context->tensors == nullptr) {
    return buffer_id_map;
  }

  auto& host_ptr_to_buffer_id = GetHostPtrToBufferIdMap(delegate_data);
  for (int i = 0; i < context->tensors_size; ++i) {
    if (external_buffer_ids.find(static_cast<size_t>(i)) !=
        external_buffer_ids.end()) {
      continue;
    }
    const TfLiteTensor& tensor = context->tensors[i];
    if (tensor.allocation_type == kTfLiteMmapRo &&
        tensor.data.raw_const != nullptr) {
      const size_t next_id = host_ptr_to_buffer_id.size() + 1;
      auto [it, _] =
          host_ptr_to_buffer_id.try_emplace(tensor.data.raw_const, next_id);
      buffer_id_map.emplace(static_cast<size_t>(i), it->second);
    }
  }
  return buffer_id_map;
}

}  // namespace litert::ml_drift
