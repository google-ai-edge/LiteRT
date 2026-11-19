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

#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

#include <any>
#include <cmath>
#include <optional>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_support.h"
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

constexpr int kActiveTokensAlignedIndex = 2;

bool IsFloatTfLiteType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteFloat16 ||
         type == kTfLiteBFloat16;
}

bool HasSymmetricPerTensorQuant(const TfLiteTensor& tensor) {
  if (tensor.quantization.type != kTfLiteAffineQuantization ||
      tensor.quantization.params == nullptr) {
    return false;
  }
  const auto* quantization =
      static_cast<const TfLiteAffineQuantization*>(tensor.quantization.params);
  if (quantization->scale == nullptr || quantization->scale->size != 1 ||
      !std::isfinite(quantization->scale->data[0]) ||
      quantization->scale->data[0] <= 0.0f) {
    return false;
  }
  if (quantization->zero_point != nullptr &&
      quantization->zero_point->size > 0 &&
      (quantization->zero_point->size != 1 ||
       quantization->zero_point->data[0] != 0)) {
    return false;
  }
  return true;
}

// Returns the dequantization scale of a quantized cache: the scale the
// `odml.cache_update` producer quantized the bytes with (it defaults to 1.0f
// in the AddValuesToCache kernel) if there is one, else the tensor's own
// quantization scale.
float GetQuantizedCacheScale(const TfLiteTensor& tensor,
                             const std::optional<float>& cache_update_scale,
                             bool from_cache_update_producer) {
  const float tensor_scale = GetAffineTensorScale(tensor);
  if (!from_cache_update_producer) {
    return tensor_scale;
  }
  const float scale = cache_update_scale.value_or(1.0f);
  if (scale != tensor_scale) {
    ABSL_LOG(WARNING) << "odml.sdpa_transposed: quantized cache '"
                      << (tensor.name ? tensor.name : "")
                      << "' is quantized by odml.cache_update with scale "
                      << scale << " but carries quantization scale "
                      << tensor_scale << "; using " << scale << ".";
  }
  return scale;
}

::ml_drift::Value* NewValueToMergeBatch(::ml_drift::GraphFloat32* graph,
                                        ::ml_drift::Value* val) {
  auto* new_val = graph->NewValue();
  new_val->tensor.type = val->tensor.type;
  new_val->tensor.shape =
      ::ml_drift::BHWC(1, val->tensor.shape.b * val->tensor.shape.h,
                       val->tensor.shape.w, val->tensor.shape.c);
  return new_val;
}

void Reshape(::ml_drift::GraphFloat32* graph, ::ml_drift::Value* before,
             ::ml_drift::Value* after) {
  ::ml_drift::Node* reshape = graph->NewNode();
  reshape->operation.type = ToString(::ml_drift::OperationType::kReshape);
  ::ml_drift::ReshapeAttributes reshape_attr;
  reshape_attr.new_shape = after->tensor.shape;
  reshape->operation.attributes = reshape_attr;
  graph->AddConsumer(reshape->id, before->id);
  graph->SetProducer(reshape->id, after->id);
}

::ml_drift::Value* ReshapeToMergeBatch(::ml_drift::GraphFloat32* graph,
                                       ::ml_drift::Value* val) {
  auto* new_val = NewValueToMergeBatch(graph, val);
  Reshape(graph, val, new_val);
  return new_val;
}

// Makes the delegate hand the raw int8/int4 cache bytes of `tflite_tensor_idx`
// to the kernel as a kUint8 tensor instead of its dequantized f32 shadow. For
// INT4 caches, halves the head_dim axis once via `PackInt4KvCacheShape` so
// external buffer allocation matches `TfLiteTensor::bytes`.
void RetagQuantizedCacheValue(::ml_drift::Value* value, int tflite_tensor_idx,
                              ::ml_drift::DataType kv_cache_dtype,
                              int cache_ts_idx) {
  if (value->tensor.type == ::ml_drift::DataType::kUint8) {
    return;
  }
  value->tensor.ref = tflite_tensor_idx;
  value->quant_params.reset();
  value->tensor.type = ::ml_drift::DataType::kUint8;
  if (kv_cache_dtype == ::ml_drift::DataType::kInt4) {
    value->tensor.shape = PackInt4KvCacheShape(value->tensor.shape,
                                               cache_ts_idx);
  }
}

}  // namespace

bool IsInt8Tensor(const TfLiteTensor& tensor) {
  return tensor.type == kTfLiteInt8 &&
         tensor.quantization.type == kTfLiteAffineQuantization &&
         tensor.quantization.params != nullptr;
}

bool IsInt4Tensor(const TfLiteTensor& tensor) {
  return tensor.type == kTfLiteInt4 &&
         tensor.quantization.type == kTfLiteAffineQuantization &&
         tensor.quantization.params != nullptr;
}

float GetAffineTensorScale(const TfLiteTensor& tensor) {
  const auto* quantization =
      static_cast<const TfLiteAffineQuantization*>(tensor.quantization.params);
  return quantization->scale->data[0];
}

absl::Status CheckSdpaTransposedQueryAndOutputTypes(
    const TfLiteContext* context, const TfLiteNode* tflite_node) {
  const TfLiteTensor* q_tensor = nullptr;
  ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &q_tensor));
  if (!IsFloatTfLiteType(q_tensor->type) &&
      !(q_tensor->type == kTfLiteInt8 &&
        HasSymmetricPerTensorQuant(*q_tensor))) {
    return absl::UnavailableError(
        "SDPA transposed expects a float or symmetric per-tensor INT8 query.");
  }
  if (tflite_node->outputs == nullptr || tflite_node->outputs->size != 1) {
    return absl::InvalidArgumentError("SDPA transposed expects 1 output.");
  }
  const int out_idx = tflite_node->outputs->data[0];
  if (out_idx < 0 || out_idx >= context->tensors_size) {
    return absl::InvalidArgumentError(
        "SDPA transposed has invalid output tensor index.");
  }
  if (!IsFloatTfLiteType(context->tensors[out_idx].type)) {
    return absl::UnavailableError(
        "SDPA transposed expects a float output tensor.");
  }
  if (q_tensor->type == kTfLiteInt8) {
    ABSL_ASSIGN_OR_RETURN(
        const TfLiteTensorUsage q_usage,
        GetTfLiteTensorUsage(context, tflite_node->inputs->data[0]));
    if (q_usage.num_consumers != 1) {
      return absl::UnavailableError(
          "SDPA transposed expects an INT8 query read only by this node.");
    }
  }
  return absl::OkStatus();
}

absl::Status CheckSdpaTransposedKvCacheTypes(const TfLiteContext* context,
                                             const TfLiteNode* tflite_node) {
  const TfLiteTensor* q_tensor = nullptr;
  ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &q_tensor));
  const TfLiteTensor* k_tensor = nullptr;
  ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &k_tensor));
  const TfLiteTensor* v_tensor = nullptr;
  ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 2, &v_tensor));
  if (k_tensor->type != v_tensor->type) {
    return absl::UnavailableError(
        "SDPA transposed expects matching K and V cache types.");
  }
  if (IsFloatTfLiteType(k_tensor->type)) {
    return absl::OkStatus();
  }
  if (k_tensor->type != kTfLiteInt8 && k_tensor->type != kTfLiteInt4) {
    return absl::UnavailableError(
        "SDPA transposed only supports float, INT8, or INT4 K/V caches.");
  }
  for (const TfLiteTensor* tensor : {k_tensor, v_tensor}) {
    if (!HasSymmetricPerTensorQuant(*tensor)) {
      return absl::UnavailableError(
          "SDPA transposed expects quantized K/V caches with symmetric "
          "per-tensor quantization.");
    }
  }
  // The packed cache layouts can not be reshaped to merge a model batch.
  if (q_tensor->dims == nullptr || q_tensor->dims->size != 4 ||
      k_tensor->dims == nullptr || k_tensor->dims->size != 4 ||
      v_tensor->dims == nullptr || v_tensor->dims->size != 4 ||
      q_tensor->dims->data[0] != 1 || k_tensor->dims->data[0] != 1 ||
      v_tensor->dims->data[0] != 1 ||
      k_tensor->dims->data[1] != v_tensor->dims->data[1]) {
    return absl::UnavailableError(
        "SDPA transposed with quantized K/V caches expects 4D Q, K, and V "
        "tensors with batch size 1 and matching K and V heads.");
  }
  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  if (params != nullptr && params->attributes != nullptr &&
      params->attributes_size > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    const int k_ts_idx = flexbuffer_map["k_ts_idx"].IsNull()
                             ? 2
                             : flexbuffer_map["k_ts_idx"].AsInt32();
    const int v_ts_idx = flexbuffer_map["v_ts_idx"].IsNull()
                             ? 3
                             : flexbuffer_map["v_ts_idx"].AsInt32();
    if (k_ts_idx != 2 || v_ts_idx != 3) {
      return absl::UnavailableError(
          "SDPA transposed quantized K/V caches require k_ts_idx=2 and "
          "v_ts_idx=3.");
    }
  }
  if (k_tensor->type == kTfLiteInt4) {
    const int cache_size = k_tensor->dims->data[2];
    const int head_dim = k_tensor->dims->data[3];
    if (head_dim <= 0 || head_dim % 8 != 0 || cache_size <= 0 ||
        cache_size % 4 != 0 || q_tensor->dims->data[3] != head_dim ||
        v_tensor->dims->data[2] != head_dim ||
        v_tensor->dims->data[3] != cache_size) {
      return absl::UnavailableError(
          "SDPA transposed with INT4 K/V caches requires cache_size divisible "
          "by 4 and head_dim divisible by 8.");
    }
  }
  // Quantized caches are read in the packed layout written by
  // `odml.cache_update`, so they must be graph inputs or its outputs.
  for (const int input_idx : {1, 2}) {
    ABSL_ASSIGN_OR_RETURN(
        const TfLiteTensorUsage usage,
        GetTfLiteTensorUsage(context, tflite_node->inputs->data[input_idx]));
    if (usage.producer != nullptr &&
        !IsStablehloComposite(usage.producer, usage.producer_registration,
                              "odml.cache_update")) {
      return absl::UnavailableError(
          "SDPA transposed expects quantized K/V caches to be graph inputs or "
          "odml.cache_update outputs.");
    }
  }
  return absl::OkStatus();
}

void SetSdpaTransposedQuantizedKvCacheWeights(
    const TfLiteTensor& k_tensor, const TfLiteTensor& v_tensor,
    const ::ml_drift::BHWC& k_shape, const ::ml_drift::BHWC& v_shape,
    ::ml_drift::DataType kv_cache_dtype,
    const AddValuesToCacheAttributes* cache_update_attr,
    SdpaTransposedAttributes& attr) {
  attr.kv_cache_dtype = kv_cache_dtype;
  const bool int4_kv = kv_cache_dtype == ::ml_drift::DataType::kInt4;
  const ::ml_drift::BHWC k_logical =
      int4_kv
          ? ::ml_drift::BHWC(k_shape.b, k_shape.h, k_shape.w, k_shape.c * 2)
          : k_shape;
  const ::ml_drift::BHWC v_logical =
      int4_kv
          ? ::ml_drift::BHWC(v_shape.b, v_shape.h, v_shape.w * 2, v_shape.c)
          : v_shape;
  attr.bmm1_weights.weights_shape =
      ::ml_drift::OHWI(k_logical.w, k_logical.h, 1, k_logical.c);
  attr.bmm1_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::kUint8, attr.bmm1_weights.weights_shape);
  attr.bmm1_weights.desc.layout =
      ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;
  attr.bmm2_weights.weights_shape =
      ::ml_drift::OHWI(v_logical.w, v_logical.h, 1, v_logical.c);
  attr.bmm2_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::kUint8, attr.bmm2_weights.weights_shape);

  const bool from_producer = cache_update_attr != nullptr;
  attr.scale_k = GetQuantizedCacheScale(
      k_tensor, from_producer ? cache_update_attr->scale_k : std::nullopt,
      from_producer);
  attr.scale_v = GetQuantizedCacheScale(
      v_tensor, from_producer ? cache_update_attr->scale_v : std::nullopt,
      from_producer);
}

absl::Status SdpaTransposedOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  int runtime_inputs = GetNumberOfRuntimeInputsForNode(context, tflite_node);
  if (runtime_inputs < 3 || runtime_inputs > 5) {
    return absl::UnavailableError(
        "SDPA transposed expects between 3 and 5 inputs.");
  }

  for (int i = 0; i < runtime_inputs; ++i) {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, i));
  }
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
  ABSL_RETURN_IF_ERROR(
      CheckSdpaTransposedQueryAndOutputTypes(context, tflite_node));
  ABSL_RETURN_IF_ERROR(CheckSdpaTransposedKvCacheTypes(context, tflite_node));

  return absl::OkStatus();
}

void SdpaTransposedOperationParser::Parse(const TfLiteNode* tflite_node,
                                          const TfLiteRegistration*,
                                          ::ml_drift::GraphFloat32* graph,
                                          ObjectReader* reader) {
  ::ml_drift::Value* query = reader->ReadValue(0);
  ::ml_drift::Value* key = reader->ReadValue(1);
  ::ml_drift::Value* value = reader->ReadValue(2);
  ::ml_drift::Value* mask = nullptr;
  ::ml_drift::Value* param_tensor = nullptr;
  if (tflite_node->inputs->size == 4) {
    auto* mask_or_param = reader->ReadValue(3);
    if (mask_or_param &&
        mask_or_param->tensor.type == ::ml_drift::DataType::kInt32) {
      param_tensor = mask_or_param;
    } else {
      mask = mask_or_param;
    }
  } else if (tflite_node->inputs->size > 4) {
    mask = reader->ReadValue(3);
    param_tensor = reader->ReadValue(4);
  }

  ::ml_drift::Value* output =
      reader->ReadValueByTensorIdx(tflite_node->outputs->data[0]);

  ::ml_drift::Node* key_producer = graph->FindProducer(key->id);
  const AddValuesToCacheAttributes* cache_update_attr = nullptr;
  if (key_producer && key_producer->operation.type == kAddValuesToCacheType) {
    cache_update_attr = &std::any_cast<const AddValuesToCacheAttributes&>(
        key_producer->operation.attributes);
  }

  // Quantized K/V caches (checked to be both INT8, both INT4, or both float in
  // IsSupported). Caches produced by odml.cache_update in this graph are
  // already kUint8 (with packed shape for INT4); external caches are retagged
  // here, before the merge-batch reshape copies their type and shape.
  const TfLiteTensor* q_tensor = reader->GetInputTensor(0);
  const TfLiteTensor* k_tensor = reader->GetInputTensor(1);
  const TfLiteTensor* v_tensor = reader->GetInputTensor(2);
  // When a symmetric per-tensor INT8 query enters the GPU partition directly as
  // a subgraph/partition input (no GPU producer), retag it to raw kInt8 so
  // BuildSdpaTransposedGpuGraph dequantizes it on GPU via `scale_q` in both
  // external and non-external tensor modes. IsSupported() checked that this
  // node is its only consumer. Interior INT8 queries (with a GPU producer) stay
  // kFloat32 and are scaled by AddQuantAdjustments.
  if (IsInt8Tensor(*q_tensor) && graph->FindProducer(query->id) == nullptr) {
    query->tensor.ref = tflite_node->inputs->data[0];
    query->quant_params.reset();
    query->tensor.type = ::ml_drift::DataType::kInt8;
  }
  const bool int8_kv = IsInt8Tensor(*k_tensor);
  const bool int4_kv = IsInt4Tensor(*k_tensor);
  const bool quant_kv = int8_kv || int4_kv;
  const ::ml_drift::DataType kv_cache_dtype =
      int4_kv ? ::ml_drift::DataType::kInt4
              : (int8_kv ? ::ml_drift::DataType::kInt8
                         : ::ml_drift::DataType::kUnknown);
  // Quantized caches entering the GPU partition directly (IsSupported() checked
  // that they are graph inputs or odml.cache_update outputs).
  if (quant_kv && graph->FindProducer(key->id) == nullptr) {
    RetagQuantizedCacheValue(key, tflite_node->inputs->data[1], kv_cache_dtype,
                             /*cache_ts_idx=*/2);
  }
  if (quant_kv && graph->FindProducer(value->id) == nullptr) {
    RetagQuantizedCacheValue(value, tflite_node->inputs->data[2],
                             kv_cache_dtype, /*cache_ts_idx=*/3);
  }

  const bool model_batch = query->tensor.shape.b != 1;
  ::ml_drift::Value* q_val = query;
  ::ml_drift::Value* k_val = key;
  ::ml_drift::Value* v_val = value;
  ::ml_drift::Value* result_val = output;
  // Reshape to merge batch dimension as the kernel expects 1xBxMxN shape.
  if (model_batch) {
    q_val = ReshapeToMergeBatch(graph, query);
    k_val = ReshapeToMergeBatch(graph, key);
    v_val = ReshapeToMergeBatch(graph, value);
    result_val = NewValueToMergeBatch(graph, output);
  }

  SdpaTransposedAttributes attr;
  if (IsInt8Tensor(*q_tensor) &&
      q_val->tensor.type == ::ml_drift::DataType::kInt8) {
    attr.scale_q = GetAffineTensorScale(*q_tensor);
  }
  if (param_tensor) {
    attr.runtime_check.src_end_ch_index = kActiveTokensAlignedIndex;
  }
  attr.is_prefill = (query->tensor.shape.w > 2 || model_batch);

  if (cache_update_attr != nullptr) {
    attr.from_cache_update = true;
  }

  if (quant_kv) {
    SetSdpaTransposedQuantizedKvCacheWeights(
        *k_tensor, *v_tensor, k_val->tensor.shape, v_val->tensor.shape,
        kv_cache_dtype, cache_update_attr, attr);
  } else {
    const ::ml_drift::BHWC right_shape_k = k_val->tensor.shape;
    attr.bmm1_weights.weights_shape =
        ::ml_drift::OHWI(right_shape_k.w, right_shape_k.h, 1, right_shape_k.c);
    attr.bmm1_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
        k_val->tensor.type, attr.bmm1_weights.weights_shape);
    attr.bmm1_weights.desc.layout =
        ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;

    const ::ml_drift::BHWC right_shape_v = v_val->tensor.shape;
    attr.bmm2_weights.weights_shape =
        ::ml_drift::OHWI(right_shape_v.w, right_shape_v.h, 1, right_shape_v.c);
    attr.bmm2_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
        v_val->tensor.type, attr.bmm2_weights.weights_shape);
  }

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  if (params && params->attributes) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (!flexbuffer_map["from_cache_update"].IsNull()) {
      attr.from_cache_update = flexbuffer_map["from_cache_update"].AsBool();
    }
    if (!flexbuffer_map["softcap"].IsNull()) {
      attr.softcap = flexbuffer_map["softcap"].AsFloat();
    } else if (!flexbuffer_map["logit_cap"].IsNull()) {
      attr.softcap = flexbuffer_map["logit_cap"].AsFloat();
    }
    if (!flexbuffer_map["is_causal"].IsNull()) {
      attr.is_causal = flexbuffer_map["is_causal"].AsBool();
    }
  }
  // Quantized cache bytes are only meaningful in the packed layout written by
  // odml.cache_update, which the FullyConnected external weights path reads.
  if (quant_kv) {
    attr.from_cache_update = true;
  }

  // When `UsesFusedSdpaKernel` holds, BuildSdpaTransposedGpuGraph dispatches a
  // fused Flash-Attention kernel (MMA prefill on Apple Metal, the work-group
  // kernel for prefill on OpenCL / WebGPU, Flash-Decode everywhere) for float
  // K/V produced by odml.cache_update. Those kernels enforce causal masking
  // (key <= q_start + X) and the active KV cache bound (key < active_tokens)
  // directly in registers from param_tensor when attr.is_causal is set, so the
  // BOOL mask is not needed. The decomposed BMM + SoftmaxReduce graph, which
  // quantized K/V caches always take, still reads the mask, and
  // BuildSdpaTransposedGpuGraph rejects a pruned mask there.
  bool skip_bool_mask_for_flash_sdpa = false;
  if (gpu_info_ != nullptr && attr.is_causal && attr.from_cache_update &&
      !quant_kv && param_tensor != nullptr && mask != nullptr &&
      mask->tensor.type == ::ml_drift::DataType::kBool) {
    skip_bool_mask_for_flash_sdpa = UsesFusedSdpaKernel(
        *gpu_info_, q_val->tensor.shape, k_val->tensor.shape, attr.is_prefill,
        attr.from_cache_update);
  }

  ::ml_drift::Node* node = graph->NewNode();
  node->operation.attributes = std::move(attr);
  node->operation.type = kSdpaTransposedType;
  graph->AddConsumer(node->id, q_val->id);
  graph->AddConsumer(node->id, k_val->id);
  graph->AddConsumer(node->id, v_val->id);
  if (mask && !skip_bool_mask_for_flash_sdpa) {
    graph->AddConsumer(node->id, mask->id);
  }
  if (param_tensor) {
    graph->AddConsumer(node->id, param_tensor->id);
  }
  graph->SetProducer(node->id, result_val->id);

  if (model_batch) {
    Reshape(graph, result_val, output);
  }
}

}  // namespace litert::ml_drift
