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
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "fp16.h"  // from @FP16
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/internal/portable_tensor_utils.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/util.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::OHWI;

bool IsFloatTensorType(TfLiteType type) {
  return type == kTfLiteFloat16 || type == kTfLiteFloat32;
}

bool IsValidTensorId(int tensor_id, size_t tensors_size) {
  return tensor_id >= 0 && static_cast<size_t>(tensor_id) < tensors_size;
}

void IgnoreTfLiteError(TfLiteContext*, const char*, ...) {}

// Parses optional static symmetric activation quantization scales for a single
// stage (`u1`, `u2`, `d1`, or `d2`) from the composite FlexBuffer map.
absl::StatusOr<BlockFfwStageQuantScales> ParseStageQuantScales(
    const flexbuffers::Map& map, absl::string_view prefix) {
  const std::string spec_key = absl::StrCat(prefix, "_quant_spec");
  if (!map[spec_key.c_str()].IsNull()) {
    return absl::UnimplementedError(absl::StrCat(
        "odml.blocked_ffw dynamic quantization spec is not supported: ",
        spec_key));
  }
  BlockFfwStageQuantScales scales;
  for (const auto& [suffix, dst] :
       {std::pair<absl::string_view, std::optional<float>*>{
            "_in_scale", &scales.in_scale},
        std::pair<absl::string_view, std::optional<float>*>{
            "_out_scale", &scales.out_scale}}) {
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
  return scales;
}

// Parses optional FlexBuffer attributes from `tflite_node`. When absent,
// defaults to SiLU activation with unquantized float activations across all
// stages (matching other composite parsers such as `glu_parser`).
absl::StatusOr<BlockFfwAttributes> ParseFlexBufferAttributes(
    const TfLiteNode& tflite_node) {
  BlockFfwAttributes attrs;
  const uint8_t* buffer = nullptr;
  size_t length = 0;
  if (tflite_node.custom_initial_data != nullptr &&
      tflite_node.custom_initial_data_size != 0) {
    if (tflite_node.custom_initial_data_size < 0) {
      return absl::InvalidArgumentError(
          "Negative custom_initial_data_size in odml.blocked_ffw.");
    }
    buffer = static_cast<const uint8_t*>(tflite_node.custom_initial_data);
    length = static_cast<size_t>(tflite_node.custom_initial_data_size);
  } else if (tflite_node.builtin_data != nullptr) {
    const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
        tflite_node.builtin_data);
    if (params->attributes != nullptr && params->attributes_size > 0) {
      buffer = static_cast<const uint8_t*>(params->attributes);
      length = params->attributes_size;
    }
  }
  if (buffer == nullptr || length == 0) {
    return attrs;
  }

  constexpr size_t kMaxAttrBufferBytes = 1 << 20;
  if (length > kMaxAttrBufferBytes ||
      !flexbuffers::VerifyBuffer(buffer, length)) {
    return absl::InvalidArgumentError(
        "Malformed FlexBuffers attribute buffer in odml.blocked_ffw.");
  }
  const flexbuffers::Reference root = flexbuffers::GetRoot(buffer, length);
  if (!root.IsMap()) {
    return absl::InvalidArgumentError(
        "Expected FlexBuffers map root for odml.blocked_ffw attributes.");
  }
  const flexbuffers::Map map = root.AsMap();
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

  ABSL_ASSIGN_OR_RETURN(attrs.u1, ParseStageQuantScales(map, "u1"));
  ABSL_ASSIGN_OR_RETURN(attrs.u2, ParseStageQuantScales(map, "u2"));
  ABSL_ASSIGN_OR_RETURN(attrs.d1, ParseStageQuantScales(map, "d1"));
  ABSL_ASSIGN_OR_RETURN(attrs.d2, ParseStageQuantScales(map, "d2"));
  return attrs;
}

// Resolved output channel count and layout orientation for a 3-D weight tensor.
struct Resolved3dWeightLayout {
  int out_channels = 0;
  bool oi_transposed = true;
};

// Determines whether a 3-D weight tensor `[H, D1, D2]` with expected `I` input
// channels is stored as `[H, I, O]` (`oi_transposed = true`) or
// `[H, O, I]` (`oi_transposed = false`).
absl::StatusOr<Resolved3dWeightLayout> Resolve3dWeightLayout(
    const TfLiteTensor* tensors, size_t tensors_size,
    const TfLiteTensor& tensor, int expected_h, int expected_i,
    bool default_oi_transposed_when_square) {
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
    return Resolved3dWeightLayout{.out_channels = d2, .oi_transposed = true};
  }
  if (d2 == expected_i && d1 != expected_i) {
    return Resolved3dWeightLayout{.out_channels = d1, .oi_transposed = false};
  }
  // Square `[H, I, I]` matrix: disambiguate using BlockwiseV2 block_shape when
  // available (`[1, B, 1]` -> `[H, I, O]`, `[1, 1, B]` -> `[H, O, I]`), and
  // fall back to scale extents (`[H, 1, D]` vs `[H, D, 1]`) to stay in sync
  // with `IrModelBuilder::InferOiTransposed`.
  Resolved3dWeightLayout result{
      .out_channels = expected_i,
      .oi_transposed = default_oi_transposed_when_square,
  };
  if (tensor.quantization.type == kTfLiteBlockwiseQuantizationV2 &&
      tensor.quantization.params != nullptr) {
    const auto* q = static_cast<const TfLiteBlockwiseQuantizationV2*>(
        tensor.quantization.params);
    bool resolved_by_block_shape = false;
    if (q->block_shape != nullptr && q->block_shape->size == 3) {
      if (q->block_shape->data[1] > 1 && q->block_shape->data[2] == 1) {
        result.oi_transposed = true;
        resolved_by_block_shape = true;
      } else if (q->block_shape->data[2] > 1 && q->block_shape->data[1] == 1) {
        result.oi_transposed = false;
        resolved_by_block_shape = true;
      }
    }
    if (!resolved_by_block_shape && tensors != nullptr &&
        IsValidTensorId(q->scale, tensors_size)) {
      const TfLiteTensor& scale = tensors[q->scale];
      if (scale.dims != nullptr && scale.dims->size == 3) {
        const int s_row = scale.dims->data[1];
        const int s_col = scale.dims->data[2];
        if (s_row > 0 && s_col > 0) {
          if (s_col == d2 && s_row < d1) {
            result.oi_transposed = true;
          } else if (s_row == d1 && s_col < d2) {
            result.oi_transposed = false;
          }
        }
      }
    }
  }
  return result;
}

// Computes the minimum byte size required for a weight tensor of `type` and
// `shape`, guarding against non-positive dims and integer overflow.
absl::StatusOr<size_t> ComputeRequiredWeightBytes(TfLiteType type,
                                                  const OHWI& shape) {
  if (shape.o <= 0 || shape.h <= 0 || shape.w <= 0 || shape.i <= 0) {
    return absl::InvalidArgumentError(
        "odml.blocked_ffw weight dimensions must be positive.");
  }
  if (!IsFloatTensorType(type) && !IsBlockFfwQuantizedWeightType(type)) {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported odml.blocked_ffw weight type: ",
                     TfLiteTypeGetName(type)));
  }
  const int dims[4] = {shape.o, shape.h, shape.w, shape.i};
  int num_elements = 0;
  if (tflite::CheckedNumElements(dims, num_elements) != kTfLiteOk) {
    return absl::InvalidArgumentError(
        "odml.blocked_ffw weight element count overflows int32.");
  }
  TfLiteContext stub_context{};
  stub_context.ReportError = IgnoreTfLiteError;
  size_t required_bytes = 0;
  if (tflite::BytesRequired(type, dims, /*dims_size=*/4, &required_bytes,
                            &stub_context) != kTfLiteOk) {
    return absl::InvalidArgumentError(
        "odml.blocked_ffw weight byte size overflows.");
  }
  return required_bytes;
}

// Validates that a constant weight tensor has a non-truncated buffer for
// `expected_ohwi`. When `allow_null_host_data` is true (2-D quantized weights
// that may be externalized and bound later by SharedMemoryManager when
// shared), `data.raw_const == nullptr` is permitted with `bytes == 0` or
// `bytes >= required_bytes`. Float and 3-D batched weights always require host
// data.
absl::Status ValidateWeightBuffer(const TfLiteTensor& tensor,
                                  const OHWI& expected_ohwi,
                                  bool allow_null_host_data) {
  ABSL_ASSIGN_OR_RETURN(const size_t required_bytes,
                        ComputeRequiredWeightBytes(tensor.type, expected_ohwi));
  if (tensor.data.raw_const == nullptr) {
    if (!allow_null_host_data) {
      return absl::InvalidArgumentError(
          "odml.blocked_ffw weight tensor has null host data.");
    }
    if (tensor.bytes != 0 && tensor.bytes < required_bytes) {
      return absl::InvalidArgumentError(
          "odml.blocked_ffw weight tensor buffer is too small.");
    }
    return absl::OkStatus();
  }
  if (tensor.bytes < required_bytes) {
    return absl::InvalidArgumentError(
        "odml.blocked_ffw weight tensor buffer is too small.");
  }
  return absl::OkStatus();
}

absl::StatusOr<BlockFfwWeightOperandSpec> ValidateAndBuildWeightSpec(
    const TfLiteTensor* tensors, size_t tensors_size,
    const TfLiteNode& tflite_node, int node_input_index, OHWI expected_ohwi,
    bool is_batched, bool oi_transposed) {
  const int tensor_id = tflite_node.inputs->data[node_input_index];
  if (!IsValidTensorId(tensor_id, tensors_size)) {
    return absl::InvalidArgumentError("Invalid weight tensor index.");
  }
  const TfLiteTensor& tensor = tensors[tensor_id];
  if (!tflite::IsConstantTensor(&tensor)) {
    return absl::UnimplementedError(
        "odml.blocked_ffw requires constant weight tensors.");
  }

  BlockFfwWeightOperandSpec spec;
  spec.node_input_index = node_input_index;
  spec.tensor_id = tensor_id;
  spec.type = tensor.type;
  spec.quant_type = tensor.quantization.type;
  spec.shape = expected_ohwi;
  spec.is_batched = is_batched;
  spec.oi_transposed = oi_transposed;

  if (IsFloatTensorType(tensor.type)) {
    if (tensor.quantization.type != kTfLiteNoQuantization) {
      return absl::UnimplementedError(
          "Unexpected quantization parameters on float weight tensor.");
    }
    ABSL_RETURN_IF_ERROR(ValidateWeightBuffer(
        tensor, expected_ohwi, /*allow_null_host_data=*/false));
    return spec;
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
    // SharedMemoryManager does not support externalized or null-host 3-D
    // batched BlockwiseV2 weights, so require host data whenever `is_batched`.
    ABSL_RETURN_IF_ERROR(ValidateWeightBuffer(
        tensor, expected_ohwi, /*allow_null_host_data=*/!is_batched));
    spec.scale_shape = info.scale_shape;
    spec.has_zero_point = info.has_zero_point;
    return spec;
  }

  if (is_batched || tensor.dims == nullptr || tensor.dims->size != 2) {
    return absl::UnimplementedError(
        "Batched 3-D quantized weights in odml.blocked_ffw require "
        "kTfLiteBlockwiseQuantizationV2.");
  }

  if (tensor.quantization.type == kTfLiteAffineQuantization &&
      tensor.quantization.params != nullptr) {
    const auto* q = static_cast<const TfLiteAffineQuantization*>(
        tensor.quantization.params);
    if (q->quantized_dimension != 0) {
      return absl::InvalidArgumentError(
          "Affine quantized weights require quantized_dimension == 0.");
    }
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
    // 2-D affine quantized weights may be externalized and bound later by
    // SharedMemoryManager.
    ABSL_RETURN_IF_ERROR(ValidateWeightBuffer(
        tensor, expected_ohwi, /*allow_null_host_data=*/true));
    spec.scale_shape = OHWI(q->scale->size, 1, 1, 1);
    // SharedMemoryManager::CreateAffineQuantizationParams always attaches a
    // zero-point tensor input for affine-quantized shared weights.
    spec.has_zero_point = true;
    return spec;
  }

  return absl::UnimplementedError(
      "Unsupported quantization type on odml.blocked_ffw weight tensor.");
}

// Transposes a batched `[H, O, I]` weight buffer to `[H, I, O]` for
// BatchedMatmul's RHS layout when needed; returns `raw` unchanged for 2-D or
// already `[H, I, O]` weights.
std::vector<float> MaybeTransposeBatchedWeightsToHio(
    std::vector<float> raw, const BlockFfwWeightOperandSpec& spec) {
  if (!spec.is_batched || spec.oi_transposed) {
    return raw;
  }
  const int H = spec.shape.h;
  const int O = spec.shape.o;
  const int I = spec.shape.i;
  std::vector<float> out(raw.size());
  for (int h = 0; h < H; ++h) {
    for (int i = 0; i < I; ++i) {
      for (int o = 0; o < O; ++o) {
        out[(static_cast<size_t>(h) * I + i) * O + o] =
            raw[(static_cast<size_t>(h) * O + o) * I + i];
      }
    }
  }
  return out;
}

}  // namespace

bool IsBlockFfwQuantizedWeightType(TfLiteType type) {
  return type == kTfLiteInt2 || type == kTfLiteInt4 || type == kTfLiteInt8;
}

absl::StatusOr<ResolvedBlockFfwPlan> ResolveBlockFfwPlan(
    const TfLiteTensor* tensors, size_t tensors_size,
    const TfLiteNode& tflite_node) {
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
  if (tensors == nullptr) {
    return absl::InvalidArgumentError("Null tensors array in TfLiteContext.");
  }
  for (int i = 0; i < num_inputs; ++i) {
    if (!IsValidTensorId(tflite_node.inputs->data[i], tensors_size)) {
      return absl::InvalidArgumentError("Invalid input tensor index.");
    }
  }
  if (!IsValidTensorId(tflite_node.outputs->data[0], tensors_size)) {
    return absl::InvalidArgumentError("Invalid output tensor index.");
  }

  ResolvedBlockFfwPlan plan;
  ABSL_ASSIGN_OR_RETURN(plan.attrs, ParseFlexBufferAttributes(tflite_node));

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
  plan.tokens = x.dims->data[1];
  plan.model_dim = x.dims->data[2];
  if (plan.model_dim % 4 != 0) {
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
  plan.k1 = (num_inputs - 3) / 2;
  if (plan.k1 <= 0 || plan.model_dim % plan.k1 != 0) {
    return absl::InvalidArgumentError(
        "Model dimension is not divisible by K1.");
  }
  plan.i1 = plan.model_dim / plan.k1;
  plan.o1 = first_w.dims->data[0];
  if (plan.o1 <= 0 || first_w.dims->data[1] != plan.i1) {
    return absl::InvalidArgumentError("Mismatched 2-D U1 weight shape.");
  }
  plan.u1_weights.reserve(plan.k1);
  for (int k = 0; k < plan.k1; ++k) {
    const TfLiteTensor& wk = tensors[tflite_node.inputs->data[1 + k]];
    if (wk.dims == nullptr || wk.dims->size != 2 ||
        wk.dims->data[0] != plan.o1 || wk.dims->data[1] != plan.i1) {
      return absl::InvalidArgumentError(
          "All U1 split weights must have shape [O1, I1].");
    }
    ABSL_ASSIGN_OR_RETURN(
        BlockFfwWeightOperandSpec spec,
        ValidateAndBuildWeightSpec(tensors, tensors_size, tflite_node, 1 + k,
                                   OHWI(plan.o1, 1, 1, plan.i1),
                                   /*is_batched=*/false,
                                   /*oi_transposed=*/false));
    plan.u1_weights.push_back(std::move(spec));
  }

  const int u2_idx = 1 + plan.k1;
  const TfLiteTensor& u2_tensor = tensors[tflite_node.inputs->data[u2_idx]];
  if (u2_tensor.dims == nullptr || u2_tensor.dims->size != 3) {
    return absl::InvalidArgumentError("Expected 3-D U2 weight tensor.");
  }
  plan.k2 = u2_tensor.dims->data[0];
  if (plan.k2 <= 0 || plan.o1 % plan.k2 != 0) {
    return absl::InvalidArgumentError("U1 output is not divisible by K2.");
  }
  plan.s = plan.o1 / plan.k2;
  if (plan.s > std::numeric_limits<int>::max() / plan.k1) {
    return absl::InvalidArgumentError("U2 input channels overflow int32.");
  }
  const int i2 = plan.k1 * plan.s;
  ABSL_ASSIGN_OR_RETURN(
      const Resolved3dWeightLayout u2_layout,
      Resolve3dWeightLayout(tensors, tensors_size, u2_tensor, plan.k2, i2,
                            /*default_oi_transposed_when_square=*/true));
  plan.o2 = u2_layout.out_channels;
  ABSL_ASSIGN_OR_RETURN(
      plan.u2_weight,
      ValidateAndBuildWeightSpec(tensors, tensors_size, tflite_node, u2_idx,
                                 OHWI(plan.o2, plan.k2, 1, i2),
                                 /*is_batched=*/true, u2_layout.oi_transposed));

  const int d1_idx = 2 + plan.k1;
  const TfLiteTensor& d1_tensor = tensors[tflite_node.inputs->data[d1_idx]];
  ABSL_ASSIGN_OR_RETURN(
      const Resolved3dWeightLayout d1_layout,
      Resolve3dWeightLayout(tensors, tensors_size, d1_tensor, plan.k2, plan.o2,
                            /*default_oi_transposed_when_square=*/true));
  if (d1_layout.out_channels != i2) {
    return absl::InvalidArgumentError(
        "D1 output channels must equal K1 * S.");
  }
  ABSL_ASSIGN_OR_RETURN(
      plan.d1_weight,
      ValidateAndBuildWeightSpec(
          tensors, tensors_size, tflite_node, d1_idx,
          OHWI(d1_layout.out_channels, plan.k2, 1, plan.o2),
          /*is_batched=*/true, d1_layout.oi_transposed));

  const int i4 = plan.k2 * plan.s;
  plan.o4 = plan.model_dim / plan.k1;
  plan.d2_weights.reserve(plan.k1);
  for (int k = 0; k < plan.k1; ++k) {
    const int d2_idx = 3 + plan.k1 + k;
    const TfLiteTensor& wk = tensors[tflite_node.inputs->data[d2_idx]];
    if (wk.dims == nullptr || wk.dims->size != 2 ||
        wk.dims->data[0] != plan.o4 || wk.dims->data[1] != i4) {
      return absl::InvalidArgumentError(
          "All D2 split weights must have shape [O4, I4].");
    }
    ABSL_ASSIGN_OR_RETURN(
        BlockFfwWeightOperandSpec spec,
        ValidateAndBuildWeightSpec(tensors, tensors_size, tflite_node, d2_idx,
                                   OHWI(plan.o4, 1, 1, i4),
                                   /*is_batched=*/false,
                                   /*oi_transposed=*/false));
    plan.d2_weights.push_back(std::move(spec));
  }

  if (plan.i1 % 4 != 0 || plan.s % 4 != 0 || plan.o2 % 4 != 0 ||
      plan.o4 % 4 != 0) {
    return absl::UnimplementedError(
        "odml.blocked_ffw requires block channel extents I1, S, O2, and O4 "
        "to be multiples of 4.");
  }
  return plan;
}

std::vector<float> ExtractBlockFfwHostFallback(
    const TfLiteTensor* tensors, size_t tensors_size,
    const BlockFfwWeightOperandSpec& spec) {
  ABSL_CHECK(tensors != nullptr);
  ABSL_CHECK_GE(spec.tensor_id, 0);
  ABSL_CHECK_LT(static_cast<size_t>(spec.tensor_id), tensors_size);
  const TfLiteTensor& tensor = tensors[spec.tensor_id];
  ABSL_CHECK(tensor.data.raw_const != nullptr)
      << "odml.blocked_ffw host fallback requires non-null weight host data.";
  const size_t total = spec.shape.DimensionsProduct();
  if (tensor.quantization.type == kTfLiteBlockwiseQuantizationV2) {
    const absl::StatusOr<BlockwiseV2Params> params =
        GetBlockwiseV2Params(tensors, tensors_size, tensor);
    ABSL_CHECK_OK(params.status());
    std::vector<float> raw(total);
    DequantizeBlockwiseV2(tensor, *params, raw.data());
    return MaybeTransposeBatchedWeightsToHio(std::move(raw), spec);
  }

  // Float32 / Float16 or 2-D affine quantized tensor.
  const absl::StatusOr<size_t> required_bytes =
      ComputeRequiredWeightBytes(tensor.type, spec.shape);
  ABSL_CHECK_OK(required_bytes.status());
  ABSL_CHECK_GE(tensor.bytes, *required_bytes);
  if (tensor.type == kTfLiteFloat32) {
    const float* src = tensor.data.f;
    std::vector<float> raw(src, src + total);
    return MaybeTransposeBatchedWeightsToHio(std::move(raw), spec);
  }
  std::vector<float> raw(total);
  if (tensor.type == kTfLiteFloat16) {
    const uint16_t* src =
        reinterpret_cast<const uint16_t*>(tensor.data.raw_const);
    for (size_t i = 0; i < total; ++i) {
      raw[i] = fp16_ieee_to_fp32_value(src[i]);
    }
    return MaybeTransposeBatchedWeightsToHio(std::move(raw), spec);
  }
  if (tensor.quantization.type == kTfLiteAffineQuantization) {
    const auto* q = static_cast<const TfLiteAffineQuantization*>(
        tensor.quantization.params);
    ABSL_CHECK(q != nullptr && q->scale != nullptr);
    const int8_t* quantized = tensor.data.int8;
    std::vector<int8_t> unpacked;
    if (tensor.type != kTfLiteInt8) {
      ABSL_CHECK(tensor.type == kTfLiteInt4 || tensor.type == kTfLiteInt2);
      ABSL_CHECK_LE(total,
                    static_cast<size_t>(std::numeric_limits<int>::max()));
      unpacked.resize(total);
      tflite::tensor_utils::UnpackPackedIntToInt8(
          tensor.data.int8, static_cast<int>(total),
          /*bit_width=*/tensor.type == kTfLiteInt2 ? 2 : 4, unpacked.data());
      quantized = unpacked.data();
    }
    const int O = spec.shape.o;
    const int I = spec.shape.i;
    for (int o = 0; o < O; ++o) {
      const float sc =
          q->scale->size == 1 ? q->scale->data[0] : q->scale->data[o];
      const float zp =
          (q->zero_point != nullptr && q->zero_point->size > 0)
              ? static_cast<float>(q->zero_point->size == 1
                                       ? q->zero_point->data[0]
                                       : q->zero_point->data[o])
              : 0.0f;
      for (int i = 0; i < I; ++i) {
        const size_t idx = static_cast<size_t>(o) * I + i;
        raw[idx] = sc * (static_cast<float>(quantized[idx]) - zp);
      }
    }
    return MaybeTransposeBatchedWeightsToHio(std::move(raw), spec);
  }
  ABSL_LOG(FATAL) << "Unsupported host fallback quantization type "
                  << tensor.quantization.type << " for weight type "
                  << TfLiteTypeGetName(tensor.type);
}

}  // namespace litert::ml_drift
