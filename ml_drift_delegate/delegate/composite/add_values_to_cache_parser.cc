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

#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift {
namespace {

std::optional<float> TryGetSymmetricTensorScale(const TfLiteTensor& tensor) {
  if (tensor.quantization.type != kTfLiteAffineQuantization ||
      tensor.quantization.params == nullptr) {
    return std::nullopt;
  }
  const auto* quantization =
      static_cast<const TfLiteAffineQuantization*>(tensor.quantization.params);
  if (quantization->scale == nullptr || quantization->scale->size != 1) {
    return std::nullopt;
  }
  return quantization->scale->data[0];
}

// Returns an error unless the quantization parameters of `tensor`, if present,
// are per-tensor with zero point `expected_zero_point` (an absent zero point
// counts as 0).
absl::Status CheckPerTensorQuantParamsIfPresent(const TfLiteTensor& tensor,
                                                int expected_zero_point) {
  if (tensor.quantization.type != kTfLiteAffineQuantization ||
      tensor.quantization.params == nullptr) {
    return absl::OkStatus();
  }
  const auto* quantization =
      static_cast<const TfLiteAffineQuantization*>(tensor.quantization.params);
  const bool has_zero_point = quantization->zero_point != nullptr &&
                              quantization->zero_point->size > 0;
  if (quantization->scale == nullptr || quantization->scale->size != 1 ||
      (has_zero_point && quantization->zero_point->size != 1) ||
      (has_zero_point ? quantization->zero_point->data[0] : 0) !=
          expected_zero_point) {
    return absl::UnavailableError(
        "odml.cache_update expects per-tensor quantization with zero point 0 "
        "(INT8/INT4) or 128 (UINT8) for K/V caches.");
  }
  return absl::OkStatus();
}

bool IsSupportedQuantizedCacheSourceTensor(const TfLiteTensor& tensor) {
  if (tensor.type == kTfLiteFloat32 || tensor.type == kTfLiteFloat16) {
    return true;
  }
  if ((tensor.type == kTfLiteInt8 || tensor.type == kTfLiteUInt8 ||
       tensor.type == kTfLiteInt4) &&
      tensor.quantization.type == kTfLiteAffineQuantization &&
      tensor.quantization.params != nullptr) {
    const auto* quantization =
        static_cast<const TfLiteAffineQuantization*>(
            tensor.quantization.params);
    if (quantization->scale == nullptr || quantization->scale->size != 1 ||
        !std::isfinite(quantization->scale->data[0]) ||
        quantization->scale->data[0] <= 0.0f ||
        quantization->zero_point == nullptr ||
        quantization->zero_point->size != 1) {
      return false;
    }
    if (tensor.type == kTfLiteInt4 && quantization->zero_point->data[0] != 0) {
      return false;
    }
    return true;
  }
  return false;
}

// Maximum number of RESHAPE and TRANSPOSE ops between a QUANTIZE op and a
// quantized K/V update.
constexpr int kMaxLayoutOpsAfterQuantize = 4;

// Checks that tensor `tensor_idx` is the output of a QUANTIZE op with a float
// input, optionally followed by RESHAPE and TRANSPOSE ops. The delegate
// supports these ops, so such a tensor stays inside the GPU partition.
absl::Status CheckQuantizedInGraph(const TfLiteContext* context,
                                   int tensor_idx) {
  for (int i = 0; i <= kMaxLayoutOpsAfterQuantize; ++i) {
    if (tensor_idx < 0 ||
        static_cast<size_t>(tensor_idx) >= context->tensors_size) {
      break;
    }
    const TfLiteTensor& tensor = context->tensors[tensor_idx];
    if (!IsSupportedQuantizedCacheSourceTensor(tensor) ||
        tflite::IsConstantTensor(&tensor) || tensor.dims == nullptr ||
        tensor.dims->size < 1 || tensor.dims->size > 4) {
      break;
    }
    ABSL_ASSIGN_OR_RETURN(const TfLiteTensorUsage usage,
                          GetTfLiteTensorUsage(context, tensor_idx));
    if (usage.producer == nullptr || usage.producer_registration == nullptr ||
        usage.producer->inputs == nullptr ||
        usage.producer->inputs->size < 1 ||
        usage.producer->outputs == nullptr ||
        usage.producer->outputs->size != 1) {
      break;
    }
    const int input_idx = usage.producer->inputs->data[0];
    if (input_idx < 0 ||
        static_cast<size_t>(input_idx) >= context->tensors_size) {
      break;
    }
    const TfLiteTensor& input_tensor = context->tensors[input_idx];
    if (tflite::IsConstantTensor(&input_tensor) ||
        input_tensor.dims == nullptr || input_tensor.dims->size < 1 ||
        input_tensor.dims->size > 4) {
      break;
    }
    const int builtin_code = usage.producer_registration->builtin_code;
    const int version = usage.producer_registration->version;
    if (builtin_code == kTfLiteBuiltinQuantize) {
      if (version <= 4 && usage.producer->inputs->size == 1 &&
          (input_tensor.type == kTfLiteFloat32 ||
           input_tensor.type == kTfLiteFloat16)) {
        return absl::OkStatus();
      }
      break;
    }
    if (builtin_code == kTfLiteBuiltinReshape) {
      if (version != 1 || usage.producer->inputs->size > 2 ||
          input_tensor.type != tensor.type) {
        break;
      }
    } else if (builtin_code == kTfLiteBuiltinTranspose) {
      if (version > 9 || usage.producer->inputs->size != 2 ||
          input_tensor.type != tensor.type) {
        break;
      }
      const int perm_idx = usage.producer->inputs->data[1];
      if (perm_idx < 0 ||
          static_cast<size_t>(perm_idx) >= context->tensors_size) {
        break;
      }
      const TfLiteTensor& perm_tensor = context->tensors[perm_idx];
      if (!tflite::IsConstantTensor(&perm_tensor) ||
          perm_tensor.type != kTfLiteInt32 ||
          perm_tensor.data.raw_const == nullptr ||
          perm_tensor.dims == nullptr || perm_tensor.dims->size != 1 ||
          perm_tensor.dims->data[0] < 2 || perm_tensor.dims->data[0] > 4 ||
          perm_tensor.dims->data[0] != input_tensor.dims->size ||
          perm_tensor.bytes !=
              static_cast<size_t>(perm_tensor.dims->data[0]) *
                  sizeof(int32_t)) {
        break;
      }
    } else {
      break;
    }
    tensor_idx = input_idx;
  }
  return absl::UnavailableError(
      "odml.cache_update expects quantized K/V updates that a QUANTIZE op "
      "produces in the graph.");
}

}  // namespace

absl::Status ReadAddValuesToCacheLayout(const flexbuffers::Map& attributes,
                                        AddValuesToCacheAttributes& attr) {
  const auto read = [&attributes](const char* name, int& value) {
    if (!attributes[name].IsNull()) {
      value = attributes[name].AsInt32();
    }
  };
  read("k_cache_ts_idx", attr.k_cache_ts_idx);
  read("v_cache_ts_idx", attr.v_cache_ts_idx);
  read("k_update_ts_idx", attr.k_update_ts_idx);
  read("v_update_ts_idx", attr.v_update_ts_idx);
  // The caches are bound to packed weight layouts that downstream matmuls
  // read, so only the historical cache layouts are implemented. Each update
  // may be [B, H, T, D] (ts_idx 2) or [B, H, D, T] (ts_idx 3).
  const auto valid_update = [](int ts_idx) {
    return ts_idx == 2 || ts_idx == 3;
  };
  if (attr.k_cache_ts_idx != 2 || attr.v_cache_ts_idx != 3 ||
      !valid_update(attr.k_update_ts_idx) ||
      !valid_update(attr.v_update_ts_idx)) {
    return absl::UnavailableError(
        "odml.cache_update: unsupported layout (k_cache_ts_idx, "
        "v_cache_ts_idx, k_update_ts_idx, v_update_ts_idx).");
  }
  return absl::OkStatus();
}

absl::Status CheckAddValuesToCacheOutputs(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const AddValuesToCacheAttributes& attr) {
  if (tflite_node->outputs == nullptr || tflite_node->outputs->size != 2) {
    return absl::InvalidArgumentError("odml.cache_update expects 2 outputs.");
  }
  const int out_k_idx = tflite_node->outputs->data[0];
  const int out_v_idx = tflite_node->outputs->data[1];
  if (out_k_idx < 0 || out_k_idx >= context->tensors_size || out_v_idx < 0 ||
      out_v_idx >= context->tensors_size) {
    return absl::InvalidArgumentError(
        "odml.cache_update has invalid output tensor index.");
  }
  const TfLiteTensor& out_k = context->tensors[out_k_idx];
  const TfLiteTensor& out_v = context->tensors[out_v_idx];
  if (out_k.type != out_v.type) {
    return absl::UnavailableError(
        "odml.cache_update expects matching K and V cache output types.");
  }
  if (out_k.type == kTfLiteUInt4 || out_k.type == kTfLiteInt2) {
    return absl::UnavailableError(
        "odml.cache_update does not support UInt4 or Int2 KV caches on GPU.");
  }
  if (out_k.type != kTfLiteFloat32 && out_k.type != kTfLiteFloat16 &&
      out_k.type != kTfLiteBFloat16 && out_k.type != kTfLiteInt8 &&
      out_k.type != kTfLiteUInt8 && out_k.type != kTfLiteInt4) {
    return absl::UnavailableError(
        "odml.cache_update has unsupported output tensor type.");
  }
  if (out_k.type == kTfLiteInt4 || out_k.type == kTfLiteInt8 ||
      out_k.type == kTfLiteUInt8) {
    if (attr.kv_cache_batch_size <= 0 || attr.head_size <= 0 ||
        attr.cache_size <= 0) {
      return absl::UnavailableError(
          "odml.cache_update with quantized KV cache requires positive "
          "kv_cache_batch_size, head_size, and cache_size.");
    }
    if (tflite_node->inputs == nullptr || tflite_node->inputs->size < 2) {
      return absl::InvalidArgumentError(
          "odml.cache_update expects at least 2 inputs.");
    }
    const int src_k_idx = tflite_node->inputs->data[0];
    const int src_v_idx = tflite_node->inputs->data[1];
    if (src_k_idx < 0 || src_k_idx >= context->tensors_size || src_v_idx < 0 ||
        src_v_idx >= context->tensors_size) {
      return absl::InvalidArgumentError(
          "odml.cache_update has invalid input tensor index.");
    }
    const TfLiteTensor& src_k = context->tensors[src_k_idx];
    const TfLiteTensor& src_v = context->tensors[src_v_idx];
    if (!IsSupportedQuantizedCacheSourceTensor(src_k) ||
        !IsSupportedQuantizedCacheSourceTensor(src_v)) {
      return absl::UnavailableError(
          "odml.cache_update with quantized KV cache requires float or "
          "per-tensor affine quantized INT8/UINT8/INT4 K/V update inputs.");
    }
    // Quantized K/V updates must be quantized in the graph, where ML Drift
    // reads them as dequantized float values. Quantized graph inputs and the
    // outputs of other ops would need host-side dequantization (and INT4
    // unpacking) that the delegate does not implement.
    for (const int src_idx : {src_k_idx, src_v_idx}) {
      const TfLiteType src_type = context->tensors[src_idx].type;
      if (src_type == kTfLiteFloat32 || src_type == kTfLiteFloat16) {
        continue;
      }
      ABSL_RETURN_IF_ERROR(CheckQuantizedInGraph(context, src_idx));
    }
    if (src_k.dims == nullptr || src_k.dims->size != 4 ||
        src_v.dims == nullptr || src_v.dims->size != 4 ||
        out_k.dims == nullptr || out_k.dims->size != 4 ||
        out_v.dims == nullptr || out_v.dims->size != 4) {
      return absl::UnavailableError(
          "odml.cache_update with quantized KV cache expects 4D input and "
          "cache tensors.");
    }
    const int src_k_tokens =
        attr.k_update_ts_idx == 3 ? src_k.dims->data[3] : src_k.dims->data[2];
    const int src_k_head_dim =
        attr.k_update_ts_idx == 3 ? src_k.dims->data[2] : src_k.dims->data[3];
    const int src_v_tokens =
        attr.v_update_ts_idx == 3 ? src_v.dims->data[3] : src_v.dims->data[2];
    const int src_v_head_dim =
        attr.v_update_ts_idx == 3 ? src_v.dims->data[2] : src_v.dims->data[3];
    // The kernel writes cache row `y` from update head `y % src_heads` and
    // reads `head_size` channels per row, so each update holds either one head
    // per cache row or a single head broadcast to all rows.
    const int src_heads = src_k.dims->data[1];
    if (src_k.dims->data[0] != 1 || src_v.dims->data[0] != 1 ||
        (src_heads != 1 && src_heads != attr.kv_cache_batch_size) ||
        src_v.dims->data[1] != src_heads || src_k_tokens <= 0 ||
        src_k_tokens != src_v_tokens || src_k_head_dim != attr.head_size ||
        src_v_head_dim != attr.head_size) {
      return absl::UnavailableError(
          "odml.cache_update with quantized KV cache has incompatible K/V "
          "update shapes.");
    }
    const int k_head_axis = attr.k_cache_ts_idx == 2 ? 3 : 2;
    const int k_time_axis = attr.k_cache_ts_idx;
    const int v_head_axis = attr.v_cache_ts_idx == 3 ? 2 : 3;
    const int v_time_axis = attr.v_cache_ts_idx;
    const int64_t cache_batch =
        static_cast<int64_t>(out_k.dims->data[0]) * out_k.dims->data[1];
    if (out_k.dims->data[k_head_axis] != attr.head_size ||
        out_v.dims->data[v_head_axis] != attr.head_size ||
        out_k.dims->data[k_time_axis] != attr.cache_size ||
        out_v.dims->data[v_time_axis] != attr.cache_size ||
        out_k.dims->data[0] <= 0 || out_k.dims->data[1] <= 0 ||
        out_k.dims->data[0] != out_v.dims->data[0] ||
        out_k.dims->data[1] != out_v.dims->data[1] ||
        cache_batch != attr.kv_cache_batch_size) {
      return absl::UnavailableError(
          "odml.cache_update with quantized KV cache expects K and V cache "
          "dimensions matching kv_cache_batch_size, head_size, and "
          "cache_size.");
    }
    // The kernel stores INT8/INT4 values with a +128/+8 offset, i.e. UINT8
    // caches must have zero point 128.
    const int expected_zero_point = out_k.type == kTfLiteUInt8 ? 128 : 0;
    ABSL_RETURN_IF_ERROR(
        CheckPerTensorQuantParamsIfPresent(out_k, expected_zero_point));
    ABSL_RETURN_IF_ERROR(
        CheckPerTensorQuantParamsIfPresent(out_v, expected_zero_point));
    const std::optional<float> scale_k =
        attr.scale_k.has_value() ? attr.scale_k
                                 : TryGetSymmetricTensorScale(out_k);
    const std::optional<float> scale_v =
        attr.scale_v.has_value() ? attr.scale_v
                                 : TryGetSymmetricTensorScale(out_v);
    if ((scale_k.has_value() &&
         (!std::isfinite(*scale_k) || *scale_k <= 0.0f)) ||
        (scale_v.has_value() &&
         (!std::isfinite(*scale_v) || *scale_v <= 0.0f))) {
      return absl::UnavailableError(
          "odml.cache_update with quantized KV cache requires finite positive "
          "scale_k and scale_v.");
    }
    if (out_k.type == kTfLiteInt4) {
      if (attr.head_size % 8 != 0) {
        return absl::UnavailableError(
            "odml.cache_update with INT4 KV cache requires head_size divisible "
            "by 8.");
      }
      if (attr.cache_size % 4 != 0) {
        return absl::UnavailableError(
            "odml.cache_update with INT4 KV cache requires cache_size "
            "divisible by 4.");
      }
      if (!scale_k.has_value() || !scale_v.has_value()) {
        return absl::UnavailableError(
            "odml.cache_update with INT4 KV cache requires positive scale_k "
            "and scale_v.");
      }
    }
  }
  return absl::OkStatus();
}

void PopulateAddValuesToCacheQuantAttributes(const TfLiteTensor& out_k,
                                             const TfLiteTensor& out_v,
                                             AddValuesToCacheAttributes& attr) {
  const bool quantized_cache = out_k.type == kTfLiteInt4 ||
                               out_k.type == kTfLiteInt8 ||
                               out_k.type == kTfLiteUInt8;
  attr.kv_cache_dtype =
      quantized_cache ? ToDataType(out_k.type) : ::ml_drift::DataType::kUnknown;
  if (!attr.scale_k.has_value()) {
    attr.scale_k = TryGetSymmetricTensorScale(out_k);
  }
  if (!attr.scale_v.has_value()) {
    attr.scale_v = TryGetSymmetricTensorScale(out_v);
  }
}

::ml_drift::BHWC PackInt4KvCacheShape(const ::ml_drift::BHWC& logical_shape,
                                      int cache_ts_idx) {
  ::ml_drift::BHWC packed = logical_shape;
  if (cache_ts_idx == 2) {
    // [B, K, S, H] -> [B, K, S, H / 2]
    packed.c /= 2;
  } else {
    // [B, K, H, S] -> [B, K, H / 2, S]
    packed.w /= 2;
  }
  return packed;
}

absl::StatusOr<TfLiteTensorUsage> GetTfLiteTensorUsage(
    const TfLiteContext* context, int tensor_idx) {
  // The TfLiteContext graph accessors take a mutable context but only read it.
  auto* mutable_context = const_cast<TfLiteContext*>(context);
  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan == nullptr ||
      context->GetNodeAndRegistration == nullptr ||
      context->GetExecutionPlan(mutable_context, &execution_plan) !=
          kTfLiteOk ||
      execution_plan == nullptr) {
    return absl::UnavailableError("TFLite execution plan is unavailable.");
  }
  TfLiteTensorUsage usage;
  for (int i = 0; i < execution_plan->size; ++i) {
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(mutable_context,
                                        execution_plan->data[i], &node,
                                        &registration) != kTfLiteOk) {
      return absl::UnavailableError("Failed to get a TFLite node.");
    }
    if (node->outputs != nullptr &&
        absl::c_linear_search(
            absl::MakeConstSpan(node->outputs->data, node->outputs->size),
            tensor_idx)) {
      usage.producer = node;
      usage.producer_registration = registration;
    }
    if (node->inputs != nullptr &&
        absl::c_linear_search(
            absl::MakeConstSpan(node->inputs->data, node->inputs->size),
            tensor_idx)) {
      ++usage.num_consumers;
    }
  }
  return usage;
}

bool IsStablehloComposite(const TfLiteNode* node,
                          const TfLiteRegistration* registration,
                          absl::string_view name) {
  if (node == nullptr || registration == nullptr ||
      registration->builtin_code != kTfLiteBuiltinStablehloComposite ||
      node->builtin_data == nullptr) {
    return false;
  }
  const auto* params =
      static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
  return params->name != nullptr && name == params->name;
}

absl::Status AddValuesToCacheOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  if (GetNumberOfRuntimeInputsForNode(context, tflite_node) != 3 &&
      GetNumberOfRuntimeInputsForNode(context, tflite_node) != 7) {
    return absl::UnavailableError("odml.cache_update expects 3 or 7 inputs.");
  }
  ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
  ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
  ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 2));
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

  if (tflite_node->outputs->size != 2) {
    return absl::InvalidArgumentError("odml.cache_update expects 2 outputs.");
  }

  AddValuesToCacheAttributes attr{};
  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  if (params) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (flexbuffer_map["kv_cache_batch_size"].IsNull()) {
      return absl::InvalidArgumentError(
          "odml.cache_update is missing kv_cache_batch_size.");
    }
    if (flexbuffer_map["cache_size"].IsNull()) {
      return absl::InvalidArgumentError(
          "odml.cache_update is missing cache_size.");
    }
    if (flexbuffer_map["head_size"].IsNull()) {
      return absl::InvalidArgumentError(
          "odml.cache_update is missing head_size.");
    }
    attr.kv_cache_batch_size = flexbuffer_map["kv_cache_batch_size"].AsInt32();
    attr.cache_size = flexbuffer_map["cache_size"].AsInt32();
    attr.head_size = flexbuffer_map["head_size"].AsInt32();
    if (!flexbuffer_map["scale_k"].IsNull()) {
      attr.scale_k = flexbuffer_map["scale_k"].AsFloat();
    }
    if (!flexbuffer_map["scale_v"].IsNull()) {
      attr.scale_v = flexbuffer_map["scale_v"].AsFloat();
    }
    ABSL_RETURN_IF_ERROR(ReadAddValuesToCacheLayout(flexbuffer_map, attr));
  }
  ABSL_RETURN_IF_ERROR(
      CheckAddValuesToCacheOutputs(context, tflite_node, attr));

  return absl::OkStatus();
}

void AddValuesToCacheOperationParser::Parse(const TfLiteNode* tflite_node,
                                            const TfLiteRegistration*,
                                            ::ml_drift::GraphFloat32* graph,
                                            ObjectReader* reader) {
  auto* node = graph->NewNode();
  node->operation.type = kAddValuesToCacheType;
  // The operator might have 7 inputs. We only read the first 3. It is a
  // composite op, and cpu decomposition needs additional tensors to do the same
  // thing (in-place dynamic update slice). It needs to a part of the op
  // signature, even if for the GPU we do not need those tensors.
  reader->AddInput(node, 0);  // src_k
  reader->AddInput(node, 1);  // src_v
  reader->AddInput(node, 2);  // runtime_param_tensor
  reader->AddOutputs(node);

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  const flexbuffers::Map flexbuffer_map =
      flexbuffers::GetRoot(params->attributes, params->attributes_size).AsMap();
  ml_drift::AddValuesToCacheAttributes attr;
  attr.kv_cache_batch_size = flexbuffer_map["kv_cache_batch_size"].AsInt32();
  attr.cache_size = flexbuffer_map["cache_size"].AsInt32();
  attr.head_size = flexbuffer_map["head_size"].AsInt32();
  if (!flexbuffer_map["scale_k"].IsNull()) {
    attr.scale_k = flexbuffer_map["scale_k"].AsFloat();
  }
  if (!flexbuffer_map["scale_v"].IsNull()) {
    attr.scale_v = flexbuffer_map["scale_v"].AsFloat();
  }
  if (!flexbuffer_map["is_ring_buffer"].IsNull()) {
    attr.is_ring_buffer = flexbuffer_map["is_ring_buffer"].AsBool();
  }
  // IsSupported() has already rejected unsupported layouts.
  ReadAddValuesToCacheLayout(flexbuffer_map, attr).IgnoreError();

  const TfLiteTensor* out_k_tensor = reader->GetOutputTensor(0);
  const TfLiteTensor* out_v_tensor = reader->GetOutputTensor(1);
  PopulateAddValuesToCacheQuantAttributes(*out_k_tensor, *out_v_tensor, attr);

  // TODO(b/482104479): allow native int8/int4 tensors in MLDrift. Currently,
  // MLDrift will automatically dequantize quantized tensors to float32 if
  // AffineQuantization is present.
  if (out_k_tensor->type == kTfLiteInt8 ||
      out_k_tensor->type == kTfLiteUInt8 ||
      out_k_tensor->type == kTfLiteInt4) {
    auto output_1 = graph->FindOutputs(node->id)[0];
    auto output_2 = graph->FindOutputs(node->id)[1];
    // reader->AddOutputs() created new FP32 tflite tensors, and uses those as
    // the tflite tensor reference (tensor index). We change it back to the
    // original tensor index to keep the mapping of the value to the original
    // quantized tflite tensor, without any dequantize in the middle.
    output_1->tensor.ref = tflite_node->outputs->data[0];
    output_2->tensor.ref = tflite_node->outputs->data[1];
    if (output_1->quant_params.has_value()) {
      output_1->quant_params.reset();
    }
    if (output_2->quant_params.has_value()) {
      output_2->quant_params.reset();
    }
    // Store quantized KV caches in DataType::kUint8 GPU buffers. For INT8/UInt8
    // the shape is unchanged (1 byte/elem). For INT4 we halve the head_size
    // axis so the GPU buffer allocates 0.5 bytes/elem (matching
    // TfLiteTensor::bytes) with 2 nibbles packed per byte.
    output_1->tensor.type = ::ml_drift::DataType::kUint8;
    output_2->tensor.type = ::ml_drift::DataType::kUint8;
    if (out_k_tensor->type == kTfLiteInt4) {
      output_1->tensor.shape =
          PackInt4KvCacheShape(output_1->tensor.shape, attr.k_cache_ts_idx);
      output_2->tensor.shape =
          PackInt4KvCacheShape(output_2->tensor.shape, attr.v_cache_ts_idx);
    }
  }

  node->operation.attributes = std::move(attr);
}

}  // namespace litert::ml_drift
