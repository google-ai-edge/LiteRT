// Copyright 2026 Google LLC.
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

#include "ml_drift_delegate/delegate/composite/ir/fused_sdpa_cache_update_parser.h"

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

void RetagQuantizedCacheTensor(::ml_drift::ir::IrModel& ir_model,
                               ::ml_drift::ir::IrTensorId tensor_id,
                               ::ml_drift::DataType kv_cache_dtype,
                               int cache_ts_idx) {
  ::ml_drift::ir::IrTensor* tensor = ir_model.GetMutableTensor(tensor_id);
  if (tensor->desc.GetDataType() == ::ml_drift::DataType::kUint8) {
    return;
  }
  ir_model.ResetQuantParams(tensor_id);
  tensor->desc.SetDataType(::ml_drift::DataType::kUint8);
  if (kv_cache_dtype == ::ml_drift::DataType::kInt4) {
    tensor->desc.SetBHWCShape(::litert::ml_drift::PackInt4KvCacheShape(
        tensor->desc.GetBHWCShape(), cache_ts_idx));
  }
}

// Retags an INT8 input that enters the GPU partition directly (no GPU
// producer) to raw kInt8, and returns true if it is raw INT8. IsSupported()
// checked that this node is its only consumer.
bool RetagPartitionInt8Tensor(::ml_drift::ir::IrModel& ir_model,
                              ::ml_drift::ir::IrTensorId tensor_id,
                              const TfLiteTensor& tflite_tensor) {
  if (tflite_tensor.type != kTfLiteInt8) {
    return false;
  }
  ::ml_drift::ir::IrTensor* tensor = ir_model.GetMutableTensor(tensor_id);
  if (tensor->desc.GetDataType() == ::ml_drift::DataType::kInt8) {
    return true;
  }
  if (ir_model.FindProducer(tensor_id) == nullptr) {
    ir_model.ResetQuantParams(tensor_id);
    tensor->desc.SetDataType(::ml_drift::DataType::kInt8);
    return true;
  }
  return false;
}

absl::Status FusedSdpaCacheUpdateIsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration) {
  return ::litert::ml_drift::FusedSdpaCacheUpdateOperationParser().IsSupported(
      context, tflite_node, registration);
}

void FusedSdpaCacheUpdateConvert(
    const TfLiteContext& context, const TfLiteNode& tflite_node,
    const TfLiteRegistration& /*registration*/,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& /*options*/,
    ::ml_drift::ir::IrModel& ir_model) {
  const TfLiteTensor& query_tensor =
      context.tensors[tflite_node.inputs->data[0]];
  const TfLiteTensor& key_cache_tensor =
      context.tensors[tflite_node.inputs->data[1]];
  const TfLiteTensor& value_cache_tensor =
      context.tensors[tflite_node.inputs->data[2]];
  const TfLiteTensor& key_new_tensor =
      context.tensors[tflite_node.inputs->data[3]];
  const TfLiteTensor& value_new_tensor =
      context.tensors[tflite_node.inputs->data[4]];

  const ::ml_drift::ir::IrTensorId query_id =
      tensor_map[tflite_node.inputs->data[0]];
  const ::ml_drift::ir::IrTensorId key_cache_id =
      tensor_map[tflite_node.inputs->data[1]];
  const ::ml_drift::ir::IrTensorId value_cache_id =
      tensor_map[tflite_node.inputs->data[2]];
  const ::ml_drift::ir::IrTensorId key_new_id =
      tensor_map[tflite_node.inputs->data[3]];
  const ::ml_drift::ir::IrTensorId value_new_id =
      tensor_map[tflite_node.inputs->data[4]];

  const bool query_is_raw_int8 =
      RetagPartitionInt8Tensor(ir_model, query_id, query_tensor);
  const bool key_new_is_raw_int8 =
      RetagPartitionInt8Tensor(ir_model, key_new_id, key_new_tensor);
  const bool value_new_is_raw_int8 =
      RetagPartitionInt8Tensor(ir_model, value_new_id, value_new_tensor);

  const bool quant_kv = key_cache_tensor.type == kTfLiteInt8 ||
                        key_cache_tensor.type == kTfLiteInt4;
  const ::ml_drift::DataType kv_cache_dtype =
      key_cache_tensor.type == kTfLiteInt4
          ? ::ml_drift::DataType::kInt4
          : (quant_kv ? ::ml_drift::DataType::kInt8
                      : ::ml_drift::DataType::kUnknown);
  if (quant_kv) {
    // Caches entering the GPU partition directly (IsSupported() checked that
    // they are graph inputs or odml.cache_update outputs).
    if (ir_model.FindProducer(key_cache_id) == nullptr) {
      RetagQuantizedCacheTensor(ir_model, key_cache_id, kv_cache_dtype,
                                /*cache_ts_idx=*/2);
    }
    if (ir_model.FindProducer(value_cache_id) == nullptr) {
      RetagQuantizedCacheTensor(ir_model, value_cache_id, kv_cache_dtype,
                                /*cache_ts_idx=*/3);
    }
    if (tflite_node.outputs->size == 3) {
      RetagQuantizedCacheTensor(ir_model,
                                tensor_map[tflite_node.outputs->data[1]],
                                kv_cache_dtype, /*cache_ts_idx=*/2);
      RetagQuantizedCacheTensor(ir_model,
                                tensor_map[tflite_node.outputs->data[2]],
                                kv_cache_dtype, /*cache_ts_idx=*/3);
    }
  }

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = kFusedSdpaCacheUpdateType;

  for (int i = 0; i < tflite_node.inputs->size; ++i) {
    if (tflite_node.inputs->data[i] != kTfLiteOptionalTensor) {
      ir_model.AddConsumer(tensor_map[tflite_node.inputs->data[i]], op->id);
    }
  }
  for (int i = 0; i < tflite_node.outputs->size; ++i) {
    ir_model.SetProducer(tensor_map[tflite_node.outputs->data[i]], op->id);
  }

  op->attr = ::litert::ml_drift::PopulateFusedSdpaCacheUpdateAttributes(
      tflite_node, query_tensor, key_cache_tensor, value_cache_tensor,
      key_new_tensor, value_new_tensor,
      {.query = query_is_raw_int8,
       .key_new = key_new_is_raw_int8,
       .value_new = value_new_is_raw_int8});
}

}  // namespace

CustomIrOpParser GetFusedSdpaCacheUpdateParser() {
  return {
      .is_supported = FusedSdpaCacheUpdateIsSupported,
      .convert = FusedSdpaCacheUpdateConvert,
  };
}

}  // namespace litert::ml_drift::ir
