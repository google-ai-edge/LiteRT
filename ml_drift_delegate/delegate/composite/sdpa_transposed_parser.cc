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
#include <optional>
#include <utility>
#include <vector>

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

// Returns the per-tensor scale of an int8 affine quantized `tensor`.
float GetInt8TensorScale(const TfLiteTensor& tensor) {
  const auto* quantization =
      static_cast<const TfLiteAffineQuantization*>(tensor.quantization.params);
  return quantization->scale->data[0];
}

// Returns the dequantization scale of an int8 cache: the scale the
// `odml.cache_update` producer quantized the bytes with (it defaults to 1.0f
// in the AddValuesToCache kernel) if there is one, else the tensor's own
// quantization scale.
float GetInt8CacheScale(const TfLiteTensor& tensor,
                        const std::optional<float>& cache_update_scale,
                        bool from_cache_update_producer) {
  const float tensor_scale = GetInt8TensorScale(tensor);
  if (!from_cache_update_producer) {
    return tensor_scale;
  }
  const float scale = cache_update_scale.value_or(1.0f);
  if (scale != tensor_scale) {
    ABSL_LOG(WARNING) << "odml.sdpa_transposed: int8 cache '"
                      << (tensor.name ? tensor.name : "")
                      << "' is quantized by odml.cache_update with scale "
                      << scale << " but carries quantization scale "
                      << tensor_scale << "; using " << scale << ".";
  }
  return scale;
}

::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::kFloat32>
BroadcastScale(float scale, int channels) {
  ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::kFloat32>
      scale_tensor;
  scale_tensor.shape = ::ml_drift::Linear(channels);
  scale_tensor.data = std::vector<float>(channels, scale);
  return scale_tensor;
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

// Makes the delegate hand the raw int8 cache bytes of `tflite_tensor_idx` to
// the kernel as a kUint8 tensor instead of its dequantized f32 shadow. Same
// trick as AddValuesToCacheOperationParser (TODO(b/482104479): allow native
// int8 tensors in MLDrift).
void RetagInt8CacheValue(::ml_drift::Value* value, int tflite_tensor_idx) {
  value->tensor.ref = tflite_tensor_idx;
  value->quant_params.reset();
  value->tensor.type = ::ml_drift::DataType::kUint8;
}

}  // namespace

bool IsInt8Tensor(const TfLiteTensor& tensor) {
  return tensor.type == kTfLiteInt8 &&
         tensor.quantization.type == kTfLiteAffineQuantization &&
         tensor.quantization.params != nullptr;
}

absl::Status CheckSdpaTransposedKvCacheTypes(const TfLiteContext* context,
                                             const TfLiteNode* tflite_node) {
  const TfLiteTensor* k_tensor = nullptr;
  ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &k_tensor));
  const TfLiteTensor* v_tensor = nullptr;
  ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 2, &v_tensor));
  const bool int8_k = k_tensor->type == kTfLiteInt8;
  const bool int8_v = v_tensor->type == kTfLiteInt8;
  if (!int8_k && !int8_v) {
    return absl::OkStatus();
  }
  if (int8_k != int8_v) {
    return absl::UnavailableError(
        "SDPA transposed expects both or none of the K/V caches to be INT8.");
  }
  for (const TfLiteTensor* tensor : {k_tensor, v_tensor}) {
    if (!IsInt8Tensor(*tensor)) {
      return absl::UnavailableError(
          "SDPA transposed expects INT8 K/V caches with affine quantization.");
    }
    const auto* quantization = static_cast<const TfLiteAffineQuantization*>(
        tensor->quantization.params);
    if (quantization->scale == nullptr || quantization->scale->size != 1 ||
        (quantization->zero_point != nullptr &&
         quantization->zero_point->size == 1 &&
         quantization->zero_point->data[0] != 0)) {
      return absl::UnavailableError(
          "SDPA transposed expects INT8 K/V caches with symmetric per-tensor "
          "quantization.");
    }
  }
  return absl::OkStatus();
}

void SetSdpaTransposedInt8KvCacheWeights(
    const TfLiteTensor& k_tensor, const TfLiteTensor& v_tensor,
    const ::ml_drift::BHWC& k_shape, const ::ml_drift::BHWC& v_shape,
    const AddValuesToCacheAttributes* cache_update_attr,
    SdpaTransposedAttributes& attr) {
  attr.bmm1_weights.weights_shape =
      ::ml_drift::OHWI(k_shape.w, k_shape.h, 1, k_shape.c);
  attr.bmm1_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::kUint8, attr.bmm1_weights.weights_shape);
  attr.bmm1_weights.desc.layout =
      ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;
  attr.bmm2_weights.weights_shape =
      ::ml_drift::OHWI(v_shape.w, v_shape.h, 1, v_shape.c);
  attr.bmm2_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::kUint8, attr.bmm2_weights.weights_shape);

  const bool from_producer = cache_update_attr != nullptr;
  const float scale_k = GetInt8CacheScale(
      k_tensor, from_producer ? cache_update_attr->scale_k : std::nullopt,
      from_producer);
  const float scale_v = GetInt8CacheScale(
      v_tensor, from_producer ? cache_update_attr->scale_v : std::nullopt,
      from_producer);
  // One scale per output channel of each matmul: the cache slots for Q*K^T
  // and the head_dim elements for P*V.
  attr.scale_k = BroadcastScale(scale_k, attr.bmm1_weights.weights_shape.o);
  attr.scale_v = BroadcastScale(scale_v, attr.bmm2_weights.weights_shape.o);
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

  // INT8 K/V caches (checked to be both int8 or both float in IsSupported).
  // Caches produced by odml.cache_update in this graph are already kUint8;
  // other int8 caches are retagged here, before the merge-batch reshape copies
  // their type.
  const TfLiteTensor* k_tensor = reader->GetInputTensor(1);
  const TfLiteTensor* v_tensor = reader->GetInputTensor(2);
  const bool int8_kv = IsInt8Tensor(*k_tensor);
  if (int8_kv && cache_update_attr == nullptr) {
    RetagInt8CacheValue(key, tflite_node->inputs->data[1]);
    RetagInt8CacheValue(value, tflite_node->inputs->data[2]);
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
  if (param_tensor) {
    attr.runtime_check.src_end_ch_index = kActiveTokensAlignedIndex;
  }
  attr.is_prefill = (query->tensor.shape.w > 2 || model_batch);

  if (cache_update_attr != nullptr) {
    attr.from_cache_update = true;
  }

  if (int8_kv) {
    SetSdpaTransposedInt8KvCacheWeights(*k_tensor, *v_tensor,
                                        k_val->tensor.shape,
                                        v_val->tensor.shape, cache_update_attr,
                                        attr);
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
  // The int8 cache bytes are only meaningful in the packed layout written by
  // odml.cache_update, which the FullyConnected external weights path reads.
  if (int8_kv) {
    attr.from_cache_update = true;
  }

  // When `UsesFusedSdpaKernel` holds, BuildSdpaTransposedGpuGraph dispatches a
  // fused Flash-Attention kernel (MMA prefill on Apple Metal, the work-group
  // kernel for prefill on OpenCL / WebGPU, Flash-Decode everywhere) for float
  // K/V produced by odml.cache_update. Those kernels enforce causal masking
  // (key <= q_start + X) and the active KV cache bound (key < active_tokens)
  // directly in registers from param_tensor when attr.is_causal is set, so the
  // BOOL mask is not needed. The decomposed BMM + SoftmaxReduce graph, which
  // int8 K/V caches always take, still reads the mask, and
  // BuildSdpaTransposedGpuGraph rejects a pruned mask there.
  bool skip_bool_mask_for_flash_sdpa = false;
  if (gpu_info_ != nullptr && attr.is_causal && attr.from_cache_update &&
      !int8_kv && param_tensor != nullptr && mask != nullptr &&
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
