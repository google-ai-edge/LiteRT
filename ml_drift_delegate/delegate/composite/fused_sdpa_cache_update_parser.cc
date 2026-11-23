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

#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

constexpr int kQuery = 0;
constexpr int kKeyCache = 1;
constexpr int kValueCache = 2;
constexpr int kKeyNew = 3;
constexpr int kValueNew = 4;
constexpr int kMask = 5;
constexpr int kParam = 6;
constexpr int kNumInputs = 7;

bool IsFloatType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteFloat16;
}

std::optional<float> GetSymmetricPerTensorQuantScale(
    const TfLiteTensor& tensor) {
  if (tensor.quantization.type != kTfLiteAffineQuantization ||
      tensor.quantization.params == nullptr) {
    return std::nullopt;
  }
  const auto* affine_quant =
      static_cast<const TfLiteAffineQuantization*>(tensor.quantization.params);
  if (affine_quant->scale == nullptr || affine_quant->scale->size != 1 ||
      !std::isfinite(affine_quant->scale->data[0]) ||
      affine_quant->scale->data[0] <= 0.0f) {
    return std::nullopt;
  }
  if (affine_quant->zero_point != nullptr &&
      affine_quant->zero_point->size > 0 &&
      (affine_quant->zero_point->size != 1 ||
       affine_quant->zero_point->data[0] != 0)) {
    return std::nullopt;
  }
  return affine_quant->scale->data[0];
}

bool IsSymmetricInt8Tensor(const TfLiteTensor& tensor) {
  return tensor.type == kTfLiteInt8 &&
         GetSymmetricPerTensorQuantScale(tensor).has_value();
}

bool IsSymmetricInt4Tensor(const TfLiteTensor& tensor) {
  return tensor.type == kTfLiteInt4 &&
         GetSymmetricPerTensorQuantScale(tensor).has_value();
}

// Returns the 4D dims of `tensor`, or an error if it is not rank 4.
absl::Status Get4dDims(const TfLiteTensor& tensor, const char* name,
                       int32_t dims[4]) {
  if (tensor.dims == nullptr || tensor.dims->size != 4) {
    return absl::UnimplementedError(
        absl::StrCat("odml.fused_sdpa_cache_update expects a 4D ", name, "."));
  }
  for (int i = 0; i < 4; ++i) dims[i] = tensor.dims->data[i];
  return absl::OkStatus();
}

int64_t NumElements(const TfLiteTensor& tensor) {
  int64_t count = 1;
  for (int i = 0; tensor.dims != nullptr && i < tensor.dims->size; ++i) {
    count *= tensor.dims->data[i];
  }
  return count;
}

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
    value->tensor.shape =
        PackInt4KvCacheShape(value->tensor.shape, cache_ts_idx);
  }
}

// Retags a symmetric per-tensor INT8 input that enters the GPU partition
// directly (no GPU producer) to raw kInt8, and returns true if it is raw INT8.
// IsSupported() checked that this node is its only consumer.
bool RetagPartitionInt8Input(::ml_drift::GraphFloat32* graph,
                             ::ml_drift::Value* value,
                             const TfLiteTensor& tflite_tensor,
                             int tflite_tensor_idx) {
  if (!IsSymmetricInt8Tensor(tflite_tensor)) {
    return false;
  }
  if (value->tensor.type == ::ml_drift::DataType::kInt8) {
    return true;
  }
  if (graph->FindProducer(value->id) == nullptr) {
    value->tensor.ref = tflite_tensor_idx;
    value->quant_params.reset();
    value->tensor.type = ::ml_drift::DataType::kInt8;
    return true;
  }
  return false;
}

}  // namespace

FusedSdpaCacheUpdateAttributes PopulateFusedSdpaCacheUpdateAttributes(
    const TfLiteNode& tflite_node, const TfLiteTensor& q_tensor,
    const TfLiteTensor& k_cache_tensor, const TfLiteTensor& v_cache_tensor,
    const TfLiteTensor& k_new_tensor, const TfLiteTensor& v_new_tensor,
    RawInt8Inputs raw_int8_inputs) {
  FusedSdpaCacheUpdateAttributes attr;
  attr.update_cache = tflite_node.outputs->size == 3;
  std::optional<float> fb_scale_k;
  std::optional<float> fb_scale_v;
  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node.builtin_data);
  if (params != nullptr && params->attributes != nullptr &&
      params->attributes_size > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (!flexbuffer_map["softcap"].IsNull()) {
      attr.softcap = flexbuffer_map["softcap"].AsFloat();
    }
    if (!flexbuffer_map["scale_k"].IsNull()) {
      const float scale = flexbuffer_map["scale_k"].AsFloat();
      if (std::isfinite(scale) && scale > 0.0f) fb_scale_k = scale;
    }
    if (!flexbuffer_map["scale_v"].IsNull()) {
      const float scale = flexbuffer_map["scale_v"].AsFloat();
      if (std::isfinite(scale) && scale > 0.0f) fb_scale_v = scale;
    }
  }

  if (raw_int8_inputs.query && q_tensor.type == kTfLiteInt8) {
    attr.scale_q = GetSymmetricPerTensorQuantScale(q_tensor);
  }
  if (raw_int8_inputs.key_new && k_new_tensor.type == kTfLiteInt8) {
    attr.scale_key_new = GetSymmetricPerTensorQuantScale(k_new_tensor);
  }
  if (raw_int8_inputs.value_new && v_new_tensor.type == kTfLiteInt8) {
    attr.scale_value_new = GetSymmetricPerTensorQuantScale(v_new_tensor);
  }

  if (k_cache_tensor.type == kTfLiteInt8 ||
      k_cache_tensor.type == kTfLiteInt4) {
    attr.kv_cache_dtype = ToDataType(k_cache_tensor.type);
    const std::optional<float> scale_k =
        GetSymmetricPerTensorQuantScale(k_cache_tensor);
    attr.scale_k = scale_k.has_value() ? scale_k : fb_scale_k;
    const std::optional<float> scale_v =
        GetSymmetricPerTensorQuantScale(v_cache_tensor);
    attr.scale_v = scale_v.has_value() ? scale_v : fb_scale_v;
  }
  return attr;
}

absl::Status FusedSdpaCacheUpdateOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  if (tflite_node->inputs->size != kNumInputs ||
      GetNumberOfRuntimeInputsForNode(context, tflite_node) != kNumInputs) {
    return absl::UnavailableError(
        "odml.fused_sdpa_cache_update expects 7 runtime inputs.");
  }
  if (tflite_node->outputs->size != 1 && tflite_node->outputs->size != 3) {
    return absl::UnavailableError(
        "odml.fused_sdpa_cache_update expects 1 or 3 outputs.");
  }
  for (int i = 0; i < kNumInputs; ++i) {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, i));
  }
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

  auto input = [&](int i) -> const TfLiteTensor& {
    return context->tensors[tflite_node->inputs->data[i]];
  };
  auto output = [&](int i) -> const TfLiteTensor& {
    return context->tensors[tflite_node->outputs->data[i]];
  };

  int32_t q[4], kc[4], vc[4], kn[4], vn[4], mask[4], out[4];
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kQuery), "query", q));
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kKeyCache), "key cache", kc));
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kValueCache), "value cache", vc));
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kKeyNew), "new key", kn));
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kValueNew), "new value", vn));
  ABSL_RETURN_IF_ERROR(Get4dDims(input(kMask), "mask", mask));
  ABSL_RETURN_IF_ERROR(Get4dDims(output(0), "output", out));

  if (!IsFloatType(input(kQuery).type) &&
      !IsSymmetricInt8Tensor(input(kQuery))) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update expects a float or symmetric per-tensor "
        "INT8 query.");
  }
  if (!IsFloatType(output(0).type)) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update expects a float attention output.");
  }

  const TfLiteTensor& kc_tensor = input(kKeyCache);
  const TfLiteTensor& vc_tensor = input(kValueCache);
  if (kc_tensor.type != vc_tensor.type) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update expects matching K and V cache types.");
  }
  const std::optional<float> kc_scale =
      GetSymmetricPerTensorQuantScale(kc_tensor);
  const std::optional<float> vc_scale =
      GetSymmetricPerTensorQuantScale(vc_tensor);
  if (!IsFloatType(kc_tensor.type) && !IsSymmetricInt8Tensor(kc_tensor) &&
      !IsSymmetricInt4Tensor(kc_tensor)) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update supports only float or symmetric "
        "per-tensor INT8/INT4 KV caches.");
  }
  if (!IsFloatType(vc_tensor.type) && !vc_scale.has_value()) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update expects symmetric per-tensor "
        "quantization on V cache.");
  }

  for (int i : {kKeyNew, kValueNew}) {
    if (!IsFloatType(input(i).type) && !IsSymmetricInt8Tensor(input(i))) {
      return absl::UnimplementedError(
          "odml.fused_sdpa_cache_update supports only float or symmetric "
          "per-tensor INT8 key_new/value_new slices.");
    }
    if (IsFloatType(kc_tensor.type) && !IsFloatType(input(i).type)) {
      return absl::UnimplementedError(
          "odml.fused_sdpa_cache_update with float KV caches expects float "
          "key_new/value_new slices.");
    }
  }

  const int heads = kc[1];
  const int cache_size = kc[2];
  const int head_dim = kc[3];
  const int new_len = kn[2];
  if (q[0] != 1 || kc[0] != 1 || vc[0] != 1 || kn[0] != 1 || vn[0] != 1) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update supports only batch size 1.");
  }
  if (heads <= 0 || cache_size <= 0 || head_dim <= 0 || new_len <= 0) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update has an empty dimension.");
  }
  if (q[1] != heads || q[3] != head_dim || q[2] % new_len != 0 ||
      vc[1] != heads || vc[2] != head_dim || vc[3] != cache_size ||
      kn[1] != heads || kn[3] != head_dim || vn[1] != heads ||
      vn[2] != head_dim || vn[3] != new_len) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update has inconsistent Q/K/V shapes.");
  }
  for (int i = 0; i < 4; ++i) {
    if (out[i] != q[i]) {
      return absl::InvalidArgumentError(
          "odml.fused_sdpa_cache_update output must have the query shape.");
    }
  }
  // The packed KV cache layouts group the cache and head dims by 4 (and INT4
  // packs two 4-element slices per uchar4, requiring head_dim % 8 == 0).
  const int required_head_align = (kc_tensor.type == kTfLiteInt4) ? 8 : 4;
  if (cache_size % 4 != 0 || head_dim % required_head_align != 0) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update requires cache size divisible by 4 and "
        "head dim divisible by 4 (or 8 for INT4 KV caches).");
  }
  if (input(kMask).type != kTfLiteBool && !IsFloatType(input(kMask).type)) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update supports only bool or float masks.");
  }
  if (mask[0] != 1 || mask[1] != 1 || mask[2] != new_len ||
      mask[3] != cache_size + new_len) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update expects a [1, 1, T, W + T] mask.");
  }
  if (input(kParam).type != kTfLiteInt32 || NumElements(input(kParam)) < 2) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update expects an int32 param tensor with at "
        "least 2 elements.");
  }
  if (tflite_node->outputs->size == 3) {
    for (int i : {1, 2}) {
      const TfLiteTensor& cache_out = output(i);
      const TfLiteTensor& cache_in = input(i == 1 ? kKeyCache : kValueCache);
      if (cache_out.type != cache_in.type || cache_out.dims == nullptr ||
          cache_out.dims->size != 4) {
        return absl::InvalidArgumentError(
            "odml.fused_sdpa_cache_update cache outputs must match inputs.");
      }
      for (int d = 0; d < 4; ++d) {
        if (cache_out.dims->data[d] != cache_in.dims->data[d]) {
          return absl::InvalidArgumentError(
              "odml.fused_sdpa_cache_update cache outputs must match inputs.");
        }
      }
      if (!IsFloatType(cache_in.type)) {
        const std::optional<float> out_scale =
            GetSymmetricPerTensorQuantScale(cache_out);
        const float in_scale = (i == 1) ? *kc_scale : *vc_scale;
        if (!out_scale.has_value() ||
            std::abs(*out_scale - in_scale) > 1e-6f * in_scale) {
          return absl::InvalidArgumentError(
              "odml.fused_sdpa_cache_update quantized cache output scales must "
              "match inputs.");
        }
      }
    }
  }
  // Checks of the graph around the node, after the checks of the node itself.
  // INT8 inputs entering the GPU partition directly are retagged to raw INT8 in
  // place for all of their GPU consumers, so they must be read by this node
  // only.
  for (int i : {kQuery, kKeyNew, kValueNew}) {
    if (input(i).type != kTfLiteInt8) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(
        const TfLiteTensorUsage usage,
        GetTfLiteTensorUsage(context, tflite_node->inputs->data[i]));
    if (usage.num_consumers != 1) {
      return absl::UnimplementedError(
          "odml.fused_sdpa_cache_update expects INT8 inputs read only by this "
          "node.");
    }
  }
  // Quantized caches are read in the packed `odml.cache_update` layout, so they
  // must be graph inputs or `odml.cache_update` outputs.
  if (!IsFloatType(kc_tensor.type)) {
    for (int i : {kKeyCache, kValueCache}) {
      ABSL_ASSIGN_OR_RETURN(
          const TfLiteTensorUsage usage,
          GetTfLiteTensorUsage(context, tflite_node->inputs->data[i]));
      if (usage.producer != nullptr &&
          !IsStablehloComposite(usage.producer, usage.producer_registration,
                                "odml.cache_update")) {
        return absl::UnimplementedError(
            "odml.fused_sdpa_cache_update expects quantized KV caches to be "
            "graph inputs or odml.cache_update outputs.");
      }
    }
  }
  return absl::OkStatus();
}

void FusedSdpaCacheUpdateOperationParser::Parse(
    const TfLiteNode* tflite_node, const TfLiteRegistration*,
    ::ml_drift::GraphFloat32* graph, ObjectReader* reader) {
  ::ml_drift::Value* query_val = reader->ReadValue(kQuery);
  ::ml_drift::Value* key_cache_val = reader->ReadValue(kKeyCache);
  ::ml_drift::Value* value_cache_val = reader->ReadValue(kValueCache);
  ::ml_drift::Value* key_new_val = reader->ReadValue(kKeyNew);
  ::ml_drift::Value* value_new_val = reader->ReadValue(kValueNew);

  const TfLiteTensor* query_tensor = reader->GetInputTensor(kQuery);
  const TfLiteTensor* key_cache_tensor = reader->GetInputTensor(kKeyCache);
  const TfLiteTensor* value_cache_tensor = reader->GetInputTensor(kValueCache);
  const TfLiteTensor* key_new_tensor = reader->GetInputTensor(kKeyNew);
  const TfLiteTensor* value_new_tensor = reader->GetInputTensor(kValueNew);

  const bool query_is_raw_int8 = RetagPartitionInt8Input(
      graph, query_val, *query_tensor, tflite_node->inputs->data[kQuery]);
  const bool key_new_is_raw_int8 = RetagPartitionInt8Input(
      graph, key_new_val, *key_new_tensor, tflite_node->inputs->data[kKeyNew]);
  const bool value_new_is_raw_int8 =
      RetagPartitionInt8Input(graph, value_new_val, *value_new_tensor,
                              tflite_node->inputs->data[kValueNew]);

  const bool quant_kv = key_cache_tensor->type == kTfLiteInt8 ||
                        key_cache_tensor->type == kTfLiteInt4;
  const ::ml_drift::DataType kv_cache_dtype =
      quant_kv ? ToDataType(key_cache_tensor->type)
               : ::ml_drift::DataType::kUnknown;
  // Quantized caches entering the GPU partition directly (IsSupported() checked
  // that they are graph inputs or odml.cache_update outputs).
  if (quant_kv && graph->FindProducer(key_cache_val->id) == nullptr) {
    RetagQuantizedCacheValue(key_cache_val,
                             tflite_node->inputs->data[kKeyCache],
                             kv_cache_dtype, /*cache_ts_idx=*/2);
  }
  if (quant_kv && graph->FindProducer(value_cache_val->id) == nullptr) {
    RetagQuantizedCacheValue(value_cache_val,
                             tflite_node->inputs->data[kValueCache],
                             kv_cache_dtype, /*cache_ts_idx=*/3);
  }

  // Populate attributes before `AddInput` / `AddOutputs`: reading quantized
  // cache outputs may allocate float shadow tensors in TFLite and reallocate
  // `context->tensors`, invalidating the `TfLiteTensor*` pointers above.
  FusedSdpaCacheUpdateAttributes attr = PopulateFusedSdpaCacheUpdateAttributes(
      *tflite_node, *query_tensor, *key_cache_tensor, *value_cache_tensor,
      *key_new_tensor, *value_new_tensor,
      {.query = query_is_raw_int8,
       .key_new = key_new_is_raw_int8,
       .value_new = value_new_is_raw_int8});

  ::ml_drift::Node* node = graph->NewNode();
  node->operation.type = kFusedSdpaCacheUpdateType;
  for (int i = 0; i < kNumInputs; ++i) {
    reader->AddInput(node, i);
  }
  reader->AddOutputs(node);

  if (quant_kv && tflite_node->outputs->size == 3) {
    std::vector<::ml_drift::Value*> node_outputs = graph->FindOutputs(node->id);
    RetagQuantizedCacheValue(node_outputs[1], tflite_node->outputs->data[1],
                             kv_cache_dtype, /*cache_ts_idx=*/2);
    RetagQuantizedCacheValue(node_outputs[2], tflite_node->outputs->data[2],
                             kv_cache_dtype, /*cache_ts_idx=*/3);
  }

  node->operation.attributes = std::move(attr);
}

}  // namespace litert::ml_drift
