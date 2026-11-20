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

#include "ml_drift_delegate/delegate/composite/block_ffw_parser.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "fp16.h"  // from @FP16
#include "xnnpack.h"  // from @XNNPACK
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::BHWC;
using ::ml_drift::DataType;
using ::ml_drift::GraphFloat32;
using ::ml_drift::Node;
using ::ml_drift::OHWI;
using ::ml_drift::OperationType;
using ::ml_drift::QuantizeAndDequantizeAttributes;
using ::ml_drift::Value;

bool IsFloatTensorType(TfLiteType type) {
  return type == kTfLiteFloat16 || type == kTfLiteFloat32;
}

absl::Status ParseFlexBufferAttributes(const TfLiteNode& tflite_node,
                                       BlockFfwAttributes* out_attr) {
  *out_attr = BlockFfwAttributes{};
  const uint8_t* buffer = nullptr;
  size_t length = 0;
  if (tflite_node.custom_initial_data != nullptr &&
      tflite_node.custom_initial_data_size > 0) {
    buffer = static_cast<const uint8_t*>(tflite_node.custom_initial_data);
    length = tflite_node.custom_initial_data_size;
  } else if (tflite_node.builtin_data != nullptr) {
    const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
        tflite_node.builtin_data);
    if (params != nullptr && params->attributes != nullptr &&
        params->attributes_size > 0) {
      buffer = static_cast<const uint8_t*>(params->attributes);
      length = params->attributes_size;
    }
  }
  if (buffer == nullptr || length == 0) {
    return absl::OkStatus();
  }

  const flexbuffers::Map map = flexbuffers::GetRoot(buffer, length).AsMap();
  if (!map["activation"].IsNull()) {
    if (!map["activation"].IsString()) {
      return absl::InvalidArgumentError(
          "odml.blocked_ffw activation attribute must be a string.");
    }
    const std::string act = map["activation"].AsString().str();
    if (act != "silu" && act != "swish") {
      return absl::UnimplementedError(absl::StrCat(
          "odml.blocked_ffw only supports silu/swish activation, got: ", act));
    }
  }

  auto parse_stage = [&](absl::string_view prefix,
                         BlockFfwStageQuantScales* scales) -> absl::Status {
    const std::string spec_key = absl::StrCat(prefix, "_quant_spec");
    if (!map[spec_key.c_str()].IsNull()) {
      return absl::UnimplementedError(absl::StrCat(
          "odml.blocked_ffw dynamic quantization spec is not supported: ",
          spec_key));
    }
    for (const auto& [suffix, dst] :
         {std::pair<absl::string_view, std::optional<float>*>{
              "_in_scale", &scales->in_scale},
          std::pair<absl::string_view, std::optional<float>*>{
              "_out_scale", &scales->out_scale}}) {
      const std::string key = absl::StrCat(prefix, suffix);
      const flexbuffers::Reference ref = map[key.c_str()];
      if (ref.IsNull()) continue;
      if (!ref.IsNumeric()) {
        return absl::InvalidArgumentError(
            absl::StrCat("odml.blocked_ffw ", key, " must be numeric."));
      }
      const float val = ref.AsFloat();
      if (!std::isfinite(val) || val <= 0.0f) {
        return absl::InvalidArgumentError(
            absl::StrCat("odml.blocked_ffw ", key, " must be finite and > 0."));
      }
      *dst = val;
    }
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(parse_stage("u1", &out_attr->u1));
  ABSL_RETURN_IF_ERROR(parse_stage("u2", &out_attr->u2));
  ABSL_RETURN_IF_ERROR(parse_stage("d1", &out_attr->d1));
  ABSL_RETURN_IF_ERROR(parse_stage("d2", &out_attr->d2));
  return absl::OkStatus();
}

// Determines whether a 3-D weight tensor `[H, D1, D2]` with expected `I` input
// channels is stored as `[H, I, O]` (`oi_transposed = true`) or
// `[H, O, I]` (`oi_transposed = false`).
absl::Status Resolve3dWeightLayout(const TfLiteTensor& tensor, int expected_h,
                                   int expected_i,
                                   bool default_oi_transposed_when_square,
                                   int* out_o, bool* out_oi_transposed) {
  if (tensor.dims == nullptr || tensor.dims->size != 3) {
    return absl::InvalidArgumentError(
        "Expected a 3-D batched weight tensor in odml.blocked_ffw.");
  }
  const int h = tensor.dims->data[0];
  const int d1 = tensor.dims->data[1];
  const int d2 = tensor.dims->data[2];
  if (h != expected_h || h <= 0 || d1 <= 0 || d2 <= 0) {
    return absl::InvalidArgumentError(
        "Mismatched batch dimension in odml.blocked_ffw 3-D weight tensor.");
  }
  if (d1 != expected_i && d2 != expected_i) {
    return absl::InvalidArgumentError(absl::StrCat(
        "3-D weight tensor [", h, ", ", d1, ", ", d2,
        "] does not match expected input channels ", expected_i, "."));
  }
  if (d1 == expected_i && d2 != expected_i) {
    *out_o = d2;
    *out_oi_transposed = true;
    return absl::OkStatus();
  }
  if (d2 == expected_i && d1 != expected_i) {
    *out_o = d1;
    *out_oi_transposed = false;
    return absl::OkStatus();
  }
  // Square `[H, I, I]` matrix: disambiguate using BlockwiseV2 block_shape when
  // available (`[1, B, 1]` -> `[H, I, O]`, `[1, 1, B]` -> `[H, O, I]`).
  *out_o = expected_i;
  *out_oi_transposed = default_oi_transposed_when_square;
  if (tensor.quantization.type == kTfLiteBlockwiseQuantizationV2 &&
      tensor.quantization.params != nullptr) {
    const auto* q = static_cast<const TfLiteBlockwiseQuantizationV2*>(
        tensor.quantization.params);
    if (q->block_shape != nullptr && q->block_shape->size == 3) {
      if (q->block_shape->data[1] > 1 && q->block_shape->data[2] == 1) {
        *out_oi_transposed = true;
      } else if (q->block_shape->data[2] > 1 && q->block_shape->data[1] == 1) {
        *out_oi_transposed = false;
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateAndBuildWeightSpec(
    const TfLiteTensor* tensors, size_t tensors_size,
    const TfLiteNode& tflite_node, int node_input_index, OHWI expected_ohwi,
    bool oi_transposed, BlockFfwWeightOperandSpec* spec) {
  const int tensor_id = tflite_node.inputs->data[node_input_index];
  if (tensor_id < 0 || static_cast<size_t>(tensor_id) >= tensors_size) {
    return absl::InvalidArgumentError("Invalid weight tensor index.");
  }
  const TfLiteTensor& tensor = tensors[tensor_id];
  if (!tflite::IsConstantTensor(&tensor)) {
    return absl::UnimplementedError(
        "odml.blocked_ffw requires constant weight tensors.");
  }

  *spec = BlockFfwWeightOperandSpec{};
  spec->node_input_index = node_input_index;
  spec->tensor_id = tensor_id;
  spec->type = tensor.type;
  spec->quant_type = tensor.quantization.type;
  spec->shape = expected_ohwi;
  spec->oi_transposed = oi_transposed;

  if (IsFloatTensorType(tensor.type)) {
    if (tensor.quantization.type != kTfLiteNoQuantization) {
      return absl::UnimplementedError(
          "Unexpected quantization parameters on float weight tensor.");
    }
    return absl::OkStatus();
  }

  if (!IsBlockFfwQuantizedWeightType(tensor.type)) {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported odml.blocked_ffw weight type: ",
                     TfLiteTypeGetName(tensor.type)));
  }

  if (tensor.quantization.type == kTfLiteBlockwiseQuantizationV2) {
    ABSL_ASSIGN_OR_RETURN(
        const BlockwiseV2Params params,
        GetBlockwiseV2Params(tensors, tensors_size, tensor));
    BlockwiseV2WeightsInfo info;
    ABSL_RETURN_IF_ERROR(
        GetBlockwiseV2WeightsInfo(tensor, params, &info, oi_transposed));
    if (info.weights_shape != expected_ohwi) {
      return absl::InvalidArgumentError(
          "BlockwiseV2 weight shape does not match resolved OHWI shape.");
    }
    spec->scale_shape = info.scale_shape;
    spec->has_zero_point = info.has_zero_point;
    return absl::OkStatus();
  }

  if (expected_ohwi.h != 1) {
    return absl::UnimplementedError(
        "Batched 3-D quantized weights in odml.blocked_ffw require "
        "kTfLiteBlockwiseQuantizationV2.");
  }

  if (tensor.quantization.type == kTfLiteAffineQuantization &&
      tensor.quantization.params != nullptr) {
    const auto* q = static_cast<const TfLiteAffineQuantization*>(
        tensor.quantization.params);
    if (q->scale == nullptr ||
        (q->scale->size != 1 && q->scale->size != expected_ohwi.o)) {
      return absl::InvalidArgumentError(
          "Affine quantized weights require scalar or per-channel scales.");
    }
    if (q->zero_point == nullptr ||
        (q->zero_point->size != 1 && q->zero_point->size != q->scale->size)) {
      return absl::InvalidArgumentError(
          "Affine quantized weights require scalar or per-channel zero "
          "points.");
    }
    spec->scale_shape = OHWI(q->scale->size, 1, 1, 1);
    // SharedMemoryManager::CreateAffineQuantizationParams always attaches a
    // zero-point tensor input for affine-quantized shared weights.
    spec->has_zero_point = true;
    return absl::OkStatus();
  }

  return absl::UnimplementedError(
      "Unsupported quantization type on odml.blocked_ffw weight tensor.");
}

// Host-side fallback float weight buffer extracted before reader mutations.
struct HostWeightFallback {
  std::vector<float> data;
};

Value* MakeIntermediateValue(GraphFloat32* graph, DataType type,
                             const BHWC& shape) {
  Value* v = graph->NewValue();
  v->tensor.type = type;
  v->tensor.shape = shape;
  return v;
}

Value* MaybeAddSymmetricInt8Qdq(GraphFloat32* graph, Value* src,
                                std::optional<float> scale,
                                Value* final_output = nullptr) {
  if (!scale.has_value()) {
    return src;
  }
  Value* dst =
      final_output != nullptr
          ? final_output
          : MakeIntermediateValue(graph, src->tensor.type, src->tensor.shape);
  Node* qdq = graph->NewNode();
  qdq->operation.type = ToString(OperationType::kQuantizeAndDequantize);
  QuantizeAndDequantizeAttributes attr;
  attr.min = -128.0f * (*scale);
  attr.max = 127.0f * (*scale);
  attr.scale = *scale;
  qdq->operation.attributes = attr;
  graph->AddConsumer(qdq->id, src->id);
  graph->SetProducer(qdq->id, dst->id);
  return dst;
}

Value* AddLinearStage(GraphFloat32* graph, ObjectReader* reader, Value* src,
                      const BlockFfwWeightOperandSpec& spec, bool is_shared,
                      int preferred_global_id,
                      HostWeightFallback&& host_fallback,
                      const BHWC& dst_shape) {
  Value* dst = MakeIntermediateValue(graph, src->tensor.type, dst_shape);
  if (is_shared) {
    Node* node = graph->NewNode();
    graph->AddConsumer(node->id, src->id);
    Value* weights = nullptr;
    reader->ReadQuantizedValueByTensorIdx(
        spec.tensor_id,
        BHWC(spec.shape.o, spec.shape.h, spec.shape.w, spec.shape.i), &weights);
    graph->AddConsumer(node->id, weights->id);
    reader->SetSharedTensor(weights->id, preferred_global_id, spec.tensor_id,
                            /*dequant_forced=*/false, /*layout=*/std::nullopt,
                            spec.oi_transposed);
    auto set_quant_shapes = [&](auto& attr) {
      attr.scale.shape = spec.scale_shape;
      if (spec.has_zero_point) {
        attr.zero_point.shape = spec.scale_shape;
      }
    };
    if (spec.type == kTfLiteInt8) {
      ::ml_drift::FullyConnectedInt8Attributes attr;
      attr.weights.shape = spec.shape;
      set_quant_shapes(attr);
      node->operation.type = ToString(OperationType::kFullyConnectedInt8);
      node->operation.attributes = std::move(attr);
    } else if (spec.type == kTfLiteInt4) {
      ::ml_drift::FullyConnectedInt4Attributes attr;
      ::ml_drift::Tensor<OHWI, DataType::kInt4> w;
      w.shape = spec.shape;
      attr.weights = w;
      set_quant_shapes(attr);
      node->operation.type = ToString(OperationType::kFullyConnectedInt4);
      node->operation.attributes = std::move(attr);
    } else {
      ::ml_drift::FullyConnectedInt2Attributes attr;
      ::ml_drift::Tensor<OHWI, DataType::kInt2> w;
      w.shape = spec.shape;
      attr.weights = w;
      set_quant_shapes(attr);
      node->operation.type = ToString(OperationType::kFullyConnectedInt2);
      node->operation.attributes = std::move(attr);
    }
    graph->SetProducer(node->id, dst->id);
    return dst;
  }

  if (spec.shape.h == 1) {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::kFullyConnected);
    ::ml_drift::FullyConnectedAttributes attr;
    attr.weights.shape = spec.shape;
    attr.weights.data = std::move(host_fallback.data);
    // `ml_drift::RearrangeWeights` expects `XNN_EXTRA_BYTES` of trailing
    // padding on float weight buffers for XNNPACK transpose kernels.
    attr.weights.data.resize(
        attr.weights.shape.DimensionsProduct() +
            XNN_EXTRA_BYTES / sizeof(float),
        0.0f);
    node->operation.attributes = std::move(attr);
    graph->AddConsumer(node->id, src->id);
    graph->SetProducer(node->id, dst->id);
    return dst;
  }

  // Non-shared batched `[1, H, I, O]` fallback via Constant + BatchedMatmul so
  // `different_weights_for_height` holds for arbitrary sequence lengths T.
  const BHWC rhs_shape(1, spec.shape.h, spec.shape.i, spec.shape.o);
  Value* rhs_val = MakeIntermediateValue(graph, src->tensor.type, rhs_shape);
  Node* const_node = graph->NewNode();
  const_node->operation.type = ToString(OperationType::kConstant);
  ::ml_drift::ConstTensorAttributes const_attr;
  auto& rhs_tensor =
      const_attr.tensor.emplace<::ml_drift::Tensor<BHWC, DataType::kFloat32>>();
  rhs_tensor.shape = rhs_shape;
  rhs_tensor.data = std::move(host_fallback.data);
  const_node->operation.attributes = std::move(const_attr);
  graph->SetProducer(const_node->id, rhs_val->id);

  Node* bmm = graph->NewNode();
  bmm->operation.type = ToString(OperationType::kBatchedMatmul);
  bmm->operation.attributes = ::ml_drift::BatchedMatMulAttributes{};
  graph->AddConsumer(bmm->id, src->id);
  graph->AddConsumer(bmm->id, rhs_val->id);
  graph->SetProducer(bmm->id, dst->id);
  return dst;
}

bool SameScale(std::optional<float> a, std::optional<float> b) {
  return a.has_value() && b.has_value() && *a == *b;
}

}  // namespace

bool IsBlockFfwQuantizedWeightType(TfLiteType type) {
  return type == kTfLiteInt2 || type == kTfLiteInt4 || type == kTfLiteInt8;
}

absl::Status ResolveBlockFfwPlan(const TfLiteTensor* tensors,
                                 size_t tensors_size,
                                 const TfLiteNode& tflite_node,
                                 ResolvedBlockFfwPlan* plan) {
  *plan = ResolvedBlockFfwPlan{};
  if (tflite_node.inputs == nullptr || tflite_node.outputs == nullptr) {
    return absl::InvalidArgumentError("Null inputs/outputs on composite node.");
  }
  const int num_inputs = tflite_node.inputs->size;
  if (num_inputs < 5 || (num_inputs - 3) % 2 != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "odml.blocked_ffw expects (3 + 2 * K1) inputs, got ", num_inputs));
  }
  if (tflite_node.outputs->size != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "odml.blocked_ffw expects 1 output, got ", tflite_node.outputs->size));
  }

  ABSL_RETURN_IF_ERROR(ParseFlexBufferAttributes(tflite_node, &plan->attrs));

  const TfLiteTensor& x = tensors[tflite_node.inputs->data[0]];
  const TfLiteTensor& out = tensors[tflite_node.outputs->data[0]];
  if (!IsFloatTensorType(x.type) || out.type != x.type) {
    return absl::UnimplementedError(
        "odml.blocked_ffw expects matching float16/float32 input and output.");
  }
  if (x.dims == nullptr || out.dims == nullptr || x.dims->size != 3 ||
      out.dims->size != 3) {
    return absl::InvalidArgumentError(
        "odml.blocked_ffw expects 3-D [1, T, D] input and output tensors.");
  }
  for (int d = 0; d < 3; ++d) {
    if (x.dims->data[d] != out.dims->data[d] || x.dims->data[d] <= 0) {
      return absl::InvalidArgumentError(
          "odml.blocked_ffw input and output shapes must match and be > 0.");
    }
  }
  if (x.dims->data[0] != 1) {
    return absl::UnimplementedError(
        "odml.blocked_ffw supports only batch size 1.");
  }
  plan->tokens = x.dims->data[1];
  plan->model_dim = x.dims->data[2];
  if (plan->model_dim % 4 != 0) {
    return absl::UnimplementedError(
        "odml.blocked_ffw requires model dimension divisible by 4.");
  }

  const TfLiteTensor& first_w = tensors[tflite_node.inputs->data[1]];
  if (first_w.dims == nullptr) {
    return absl::InvalidArgumentError("Weight tensor has null dims.");
  }

  // Operands: `(x, *u1_ws, u2_w, d1_w, *d2_ws)` with `K1` 2-D weights for
  // each of the edge stages U1 and D2.
  if (first_w.dims->size != 2) {
    return absl::InvalidArgumentError(
        "odml.blocked_ffw expects 2-D edge weight tensors.");
  }
  plan->k1 = (num_inputs - 3) / 2;
  if (plan->k1 <= 0 || plan->model_dim % plan->k1 != 0) {
    return absl::InvalidArgumentError(
        "Model dimension is not divisible by K1.");
  }
  plan->i1 = plan->model_dim / plan->k1;
  plan->o1 = first_w.dims->data[0];
  if (plan->o1 <= 0 || first_w.dims->data[1] != plan->i1) {
    return absl::InvalidArgumentError("Mismatched 2-D U1 weight shape.");
  }
  plan->u1_weights.resize(plan->k1);
  for (int k = 0; k < plan->k1; ++k) {
    const TfLiteTensor& wk = tensors[tflite_node.inputs->data[1 + k]];
    if (wk.dims == nullptr || wk.dims->size != 2 ||
        wk.dims->data[0] != plan->o1 || wk.dims->data[1] != plan->i1) {
      return absl::InvalidArgumentError(
          "All U1 split weights must have shape [O1, I1].");
    }
    ABSL_RETURN_IF_ERROR(ValidateAndBuildWeightSpec(
        tensors, tensors_size, tflite_node, 1 + k,
        OHWI(plan->o1, 1, 1, plan->i1), /*oi_transposed=*/true,
        &plan->u1_weights[k]));
  }

  const int u2_idx = 1 + plan->k1;
  const TfLiteTensor& u2_tensor = tensors[tflite_node.inputs->data[u2_idx]];
  if (u2_tensor.dims == nullptr || u2_tensor.dims->size != 3) {
    return absl::InvalidArgumentError("Expected 3-D U2 weight tensor.");
  }
  plan->k2 = u2_tensor.dims->data[0];
  if (plan->k2 <= 0 || plan->o1 % plan->k2 != 0) {
    return absl::InvalidArgumentError("U1 output is not divisible by K2.");
  }
  plan->s = plan->o1 / plan->k2;
  const int i2 = plan->k1 * plan->s;
  bool u2_oi_transposed = true;
  ABSL_RETURN_IF_ERROR(Resolve3dWeightLayout(
      u2_tensor, plan->k2, i2, /*default_oi_transposed_when_square=*/true,
      &plan->o2, &u2_oi_transposed));
  ABSL_RETURN_IF_ERROR(ValidateAndBuildWeightSpec(
      tensors, tensors_size, tflite_node, u2_idx,
      OHWI(plan->o2, plan->k2, 1, i2), u2_oi_transposed, &plan->u2_weight));

  const int d1_idx = 2 + plan->k1;
  const TfLiteTensor& d1_tensor = tensors[tflite_node.inputs->data[d1_idx]];
  int o3 = 0;
  bool d1_oi_transposed = true;
  ABSL_RETURN_IF_ERROR(Resolve3dWeightLayout(
      d1_tensor, plan->k2, plan->o2, /*default_oi_transposed_when_square=*/true,
      &o3, &d1_oi_transposed));
  if (o3 != i2) {
    return absl::InvalidArgumentError(
        "D1 output channels must equal K1 * S.");
  }
  ABSL_RETURN_IF_ERROR(ValidateAndBuildWeightSpec(
      tensors, tensors_size, tflite_node, d1_idx,
      OHWI(o3, plan->k2, 1, plan->o2), d1_oi_transposed, &plan->d1_weight));

  const int i4 = plan->k2 * plan->s;
  plan->o4 = plan->model_dim / plan->k1;
  plan->d2_weights.resize(plan->k1);
  for (int k = 0; k < plan->k1; ++k) {
    const int d2_idx = 3 + plan->k1 + k;
    const TfLiteTensor& wk = tensors[tflite_node.inputs->data[d2_idx]];
    if (wk.dims == nullptr || wk.dims->size != 2 ||
        wk.dims->data[0] != plan->o4 || wk.dims->data[1] != i4) {
      return absl::InvalidArgumentError(
          "All D2 split weights must have shape [O4, I4].");
    }
    ABSL_RETURN_IF_ERROR(ValidateAndBuildWeightSpec(
        tensors, tensors_size, tflite_node, d2_idx, OHWI(plan->o4, 1, 1, i4),
        /*oi_transposed=*/true, &plan->d2_weights[k]));
  }

  if (plan->i1 % 4 != 0 || plan->s % 4 != 0 || plan->o2 % 4 != 0 ||
      plan->o4 % 4 != 0) {
    return absl::UnimplementedError(
        "odml.blocked_ffw requires block channel extents I1, S, O2, and O4 "
        "to be multiples of 4.");
  }
  return absl::OkStatus();
}

std::vector<float> ExtractBlockFfwHostFallback(
    const TfLiteTensor* tensors, size_t tensors_size,
    const BlockFfwWeightOperandSpec& spec) {
  const TfLiteTensor& tensor = tensors[spec.tensor_id];
  const size_t total = spec.shape.DimensionsProduct();
  std::vector<float> out(total);
  if (tensor.quantization.type == kTfLiteBlockwiseQuantizationV2) {
    const absl::StatusOr<BlockwiseV2Params> params =
        GetBlockwiseV2Params(tensors, tensors_size, tensor);
    ABSL_CHECK_OK(params.status());
    std::vector<float> raw(total);
    DequantizeBlockwiseV2(tensor, *params, raw.data());
    if (spec.shape.h == 1 || spec.oi_transposed) {
      return raw;
    }
    // Raw is `[H, O, I]`, transpose each batch to `[H, I, O]` for BatchMatMul.
    const int H = spec.shape.h;
    const int O = spec.shape.o;
    const int I = spec.shape.i;
    for (int h = 0; h < H; ++h) {
      for (int i = 0; i < I; ++i) {
        for (int o = 0; o < O; ++o) {
          out[(h * I + i) * O + o] = raw[(h * O + o) * I + i];
        }
      }
    }
    return out;
  }

  // Float32 / Float16 or 2-D affine quantized tensor.
  std::vector<float> raw(total);
  if (tensor.type == kTfLiteFloat32) {
    const float* src = tensor.data.f;
    raw.assign(src, src + total);
  } else if (tensor.type == kTfLiteFloat16) {
    const uint16_t* src = reinterpret_cast<const uint16_t*>(tensor.data.raw);
    for (size_t i = 0; i < total; ++i) {
      raw[i] = fp16_ieee_to_fp32_value(src[i]);
    }
  } else if (tensor.quantization.type == kTfLiteAffineQuantization) {
    const auto* q = static_cast<const TfLiteAffineQuantization*>(
        tensor.quantization.params);
    const int O = spec.shape.o;
    const int I = spec.shape.i;
    for (int o = 0; o < O; ++o) {
      const float sc =
          q->scale->size == 1 ? q->scale->data[0] : q->scale->data[o];
      const float zp = (q->zero_point != nullptr && q->zero_point->size > 0)
                           ? static_cast<float>(q->zero_point->size == 1
                                                    ? q->zero_point->data[0]
                                                    : q->zero_point->data[o])
                           : 0.0f;
      for (int i = 0; i < I; ++i) {
        const size_t idx = static_cast<size_t>(o) * I + i;
        int qval = 0;
        if (tensor.type == kTfLiteInt8) {
          qval = tensor.data.int8[idx];
        } else if (tensor.type == kTfLiteInt4) {
          const uint8_t byte = static_cast<uint8_t>(tensor.data.int8[idx / 2]);
          const int8_t nib =
              static_cast<int8_t>(((idx % 2 == 0) ? (byte << 4) : byte)) >> 4;
          qval = nib;
        } else {
          const uint8_t byte = static_cast<uint8_t>(tensor.data.int8[idx / 4]);
          const int shift = 6 - 2 * static_cast<int>(idx % 4);
          const int8_t crumb = static_cast<int8_t>(byte << shift) >> 6;
          qval = crumb;
        }
        raw[idx] = sc * (static_cast<float>(qval) - zp);
      }
    }
  }
  if (spec.shape.h == 1 || spec.oi_transposed) {
    return raw;
  }
  const int H = spec.shape.h;
  const int O = spec.shape.o;
  const int I = spec.shape.i;
  for (int h = 0; h < H; ++h) {
    for (int i = 0; i < I; ++i) {
      for (int o = 0; o < O; ++o) {
        out[(h * I + i) * O + o] = raw[(h * O + o) * I + i];
      }
    }
  }
  return out;
}

absl::Status BlockFfwOperationParser::IsSupported(const TfLiteContext* context,
                                                  const TfLiteNode* tflite_node,
                                                  const TfLiteRegistration*) {
  ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
  ResolvedBlockFfwPlan plan;
  return ResolveBlockFfwPlan(context->tensors, context->tensors_size,
                             *tflite_node, &plan);
}

void BlockFfwOperationParser::Parse(const TfLiteNode* tflite_node,
                                    const TfLiteRegistration*,
                                    GraphFloat32* graph, ObjectReader* reader) {
  int max_tensor_id = 0;
  for (int i = 0; i < tflite_node->inputs->size; ++i) {
    const int tid = tflite_node->inputs->data[i];
    if (tid > max_tensor_id) max_tensor_id = tid;
    const TfLiteTensor* t = reader->GetTensor(tid);
    if (t->quantization.type == kTfLiteBlockwiseQuantizationV2 &&
        t->quantization.params != nullptr) {
      const auto* q = static_cast<const TfLiteBlockwiseQuantizationV2*>(
          t->quantization.params);
      if (q->scale > max_tensor_id) max_tensor_id = q->scale;
      if (q->zero_point > max_tensor_id) max_tensor_id = q->zero_point;
    }
  }
  for (int i = 0; i < tflite_node->outputs->size; ++i) {
    if (tflite_node->outputs->data[i] > max_tensor_id) {
      max_tensor_id = tflite_node->outputs->data[i];
    }
  }
  const TfLiteTensor* tensors = reader->GetTensor(0);
  const size_t tensors_size = static_cast<size_t>(max_tensor_id + 1);
  ResolvedBlockFfwPlan plan;
  ABSL_CHECK_OK(
      ResolveBlockFfwPlan(tensors, tensors_size, *tflite_node, &plan));

  // Allow sharing on quantized constant weight inputs and prepare any host
  // fallbacks before `ReadValue` / `AddOutputs` can reallocate TFLite tensors.
  struct PreparedWeight {
    BlockFfwWeightOperandSpec spec;
    bool is_shared = false;
    int preferred_global_id = -1;
    HostWeightFallback fallback;
  };
  auto prepare_weight = [&](const BlockFfwWeightOperandSpec& spec) {
    PreparedWeight pw;
    pw.spec = spec;
    if (IsBlockFfwQuantizedWeightType(spec.type)) {
      reader->AllowSharingInput(spec.node_input_index);
      const ObjectReader::ConstantInputSharingInfo share =
          reader->GetSharingInfoByNodeInputIndex(spec.node_input_index);
      if (share.IsShared()) {
        pw.is_shared = true;
        pw.preferred_global_id = share.PreferredId();
      }
    }
    if (!pw.is_shared) {
      pw.fallback.data =
          ExtractBlockFfwHostFallback(tensors, tensors_size, spec);
    }
    return pw;
  };

  std::vector<PreparedWeight> u1_prepared;
  u1_prepared.reserve(plan.u1_weights.size());
  for (const auto& w : plan.u1_weights) {
    u1_prepared.push_back(prepare_weight(w));
  }
  PreparedWeight u2_prepared = prepare_weight(plan.u2_weight);
  PreparedWeight d1_prepared = prepare_weight(plan.d1_weight);
  std::vector<PreparedWeight> d2_prepared;
  d2_prepared.reserve(plan.d2_weights.size());
  for (const auto& w : plan.d2_weights) {
    d2_prepared.push_back(prepare_weight(w));
  }

  Value* x_val = reader->ReadValue(0);
  Value* out_val = reader->ReadValueByTensorIdx(tflite_node->outputs->data[0]);
  const DataType dtype = x_val->tensor.type;
  const int T = plan.tokens;
  const int i2 = plan.k1 * plan.s;
  const int i4 = plan.k2 * plan.s;

  // Stage 1 (U1) + permutation into U2's input `[1, K2, T, K1 * S]`.
  Value* u2_in = nullptr;
  {
    std::vector<Value*> parts(plan.k1);
    for (int k = 0; k < plan.k1; ++k) {
      Value* part_in =
          MakeIntermediateValue(graph, dtype, BHWC(1, 1, T, plan.i1));
      Node* unpack = graph->NewNode();
      unpack->operation.type = kBlockFfwUnpackPartType;
      unpack->operation.attributes =
          BlockFfwUnpackPartAttributes{.part = k, .num_parts = plan.k1};
      graph->AddConsumer(unpack->id, x_val->id);
      graph->SetProducer(unpack->id, part_in->id);

      Value* q_in =
          MaybeAddSymmetricInt8Qdq(graph, part_in, plan.attrs.u1.in_scale);
      Value* fc_out = AddLinearStage(
          graph, reader, q_in, u1_prepared[k].spec, u1_prepared[k].is_shared,
          u1_prepared[k].preferred_global_id,
          std::move(u1_prepared[k].fallback), BHWC(1, 1, T, plan.o1));
      parts[k] =
          MaybeAddSymmetricInt8Qdq(graph, fc_out, plan.attrs.u1.out_scale);
    }
    Value* packed =
        MakeIntermediateValue(graph, dtype, BHWC(1, plan.k2, T, i2));
    Node* pack_node = graph->NewNode();
    pack_node->operation.type = kBlockFfwPackPartsType;
    pack_node->operation.attributes =
        BlockFfwPackPartsAttributes{.num_parts = plan.k1};
    for (Value* part : parts) {
      graph->AddConsumer(pack_node->id, part->id);
    }
    graph->SetProducer(pack_node->id, packed->id);
    u2_in = packed;
  }

  // Deduplicate adjacent identical scales (`u2_in_scale == u1_out_scale`).
  if (plan.attrs.u2.in_scale.has_value() &&
      !SameScale(plan.attrs.u2.in_scale, plan.attrs.u1.out_scale)) {
    u2_in = MaybeAddSymmetricInt8Qdq(graph, u2_in, plan.attrs.u2.in_scale);
  }

  // Stage 2 (U2) + SiLU activation.
  Value* u2_raw = AddLinearStage(
      graph, reader, u2_in, u2_prepared.spec, u2_prepared.is_shared,
      u2_prepared.preferred_global_id, std::move(u2_prepared.fallback),
      BHWC(1, plan.k2, T, plan.o2));
  Value* u2_out =
      MaybeAddSymmetricInt8Qdq(graph, u2_raw, plan.attrs.u2.out_scale);

  Value* act_out =
      MakeIntermediateValue(graph, dtype, BHWC(1, plan.k2, T, plan.o2));
  Node* silu_node = graph->NewNode();
  silu_node->operation.type = kBlockFfwSiluType;
  graph->AddConsumer(silu_node->id, u2_out->id);
  graph->SetProducer(silu_node->id, act_out->id);

  // Stage 3 (D1).
  Value* d1_in =
      MaybeAddSymmetricInt8Qdq(graph, act_out, plan.attrs.d1.in_scale);
  Value* d1_raw =
      AddLinearStage(graph, reader, d1_in, d1_prepared.spec,
                     d1_prepared.is_shared, d1_prepared.preferred_global_id,
                     std::move(d1_prepared.fallback), BHWC(1, plan.k2, T, i2));
  Value* d1_out =
      MaybeAddSymmetricInt8Qdq(graph, d1_raw, plan.attrs.d1.out_scale);

  // Stage 4 (D2) + merge into `[1, 1, T, D]`.
  const std::optional<float> d2_in_scale =
      (plan.attrs.d2.in_scale.has_value() &&
       !SameScale(plan.attrs.d2.in_scale, plan.attrs.d1.out_scale))
          ? plan.attrs.d2.in_scale
          : std::nullopt;

  std::vector<Value*> parts(plan.k1);
  for (int k = 0; k < plan.k1; ++k) {
    Value* part_in = MakeIntermediateValue(graph, dtype, BHWC(1, 1, T, i4));
    Node* unpack = graph->NewNode();
    unpack->operation.type = kBlockFfwUnpackPartType;
    unpack->operation.attributes =
        BlockFfwUnpackPartAttributes{.part = k, .num_parts = plan.k1};
    graph->AddConsumer(unpack->id, d1_out->id);
    graph->SetProducer(unpack->id, part_in->id);

    Value* q_in = MaybeAddSymmetricInt8Qdq(graph, part_in, d2_in_scale);
    Value* fc_out = AddLinearStage(
        graph, reader, q_in, d2_prepared[k].spec, d2_prepared[k].is_shared,
        d2_prepared[k].preferred_global_id,
        std::move(d2_prepared[k].fallback), BHWC(1, 1, T, plan.o4));
    parts[k] =
        MaybeAddSymmetricInt8Qdq(graph, fc_out, plan.attrs.d2.out_scale);
  }
  Node* pack_node = graph->NewNode();
  pack_node->operation.type = kBlockFfwPackPartsType;
  pack_node->operation.attributes =
      BlockFfwPackPartsAttributes{.num_parts = plan.k1};
  for (Value* part : parts) {
    graph->AddConsumer(pack_node->id, part->id);
  }
  graph->SetProducer(pack_node->id, out_val->id);
}

}  // namespace litert::ml_drift
