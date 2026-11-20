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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_BLOCK_FFW_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_BLOCK_FFW_PARSER_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace litert::ml_drift {

// StableHLO composite names for 4-stage block-structured feed-forward layers.
inline constexpr char kBlockedFfwCompositeName[] = "odml.blocked_ffw";
inline constexpr char kBlockFfwCompositeName[] = "odml.block_ffw";

// Internal GraphFloat32 / IrModel node operation types emitted by the
// block_ffw parsers.
inline constexpr char kBlockFfwUnpackPartType[] = "odml.block_ffw_unpack_part";
inline constexpr char kBlockFfwPackPartsType[] = "odml.block_ffw_pack_parts";
inline constexpr char kBlockFfwSiluType[] = "odml.block_ffw_silu";

struct BlockFfwUnpackPartAttributes {
  int part = 0;
  int num_parts = 0;
};

struct BlockFfwPackPartsAttributes {
  int num_parts = 0;
};

struct BlockFfwStageQuantScales {
  std::optional<float> in_scale;
  std::optional<float> out_scale;
};

struct BlockFfwAttributes {
  BlockFfwStageQuantScales u1;
  BlockFfwStageQuantScales u2;
  BlockFfwStageQuantScales d1;
  BlockFfwStageQuantScales d2;
};

struct BlockFfwWeightOperandSpec {
  int node_input_index = -1;
  int tensor_id = -1;
  TfLiteType type = kTfLiteNoType;
  TfLiteQuantizationType quant_type = kTfLiteNoQuantization;
  ::ml_drift::OHWI shape;
  ::ml_drift::OHWI scale_shape;
  bool has_zero_point = false;
  bool oi_transposed = true;
};

struct ResolvedBlockFfwPlan {
  int tokens = 0;
  int model_dim = 0;
  int k1 = 0;
  int k2 = 0;
  int i1 = 0;
  int o1 = 0;
  int s = 0;
  int o2 = 0;
  int o4 = 0;
  BlockFfwAttributes attrs;
  std::vector<BlockFfwWeightOperandSpec> u1_weights;
  BlockFfwWeightOperandSpec u2_weight;
  BlockFfwWeightOperandSpec d1_weight;
  std::vector<BlockFfwWeightOperandSpec> d2_weights;
};

// Returns true if `type` is a supported quantized weight dtype
// (INT2/INT4/INT8).
bool IsBlockFfwQuantizedWeightType(TfLiteType type);

// Validates and resolves shapes, layouts, and quantization metadata for an
// `odml.blocked_ffw` composite node.
absl::Status ResolveBlockFfwPlan(const TfLiteTensor* tensors,
                                 size_t tensors_size,
                                 const TfLiteNode& tflite_node,
                                 ResolvedBlockFfwPlan* plan);

// Extracts a host-side float32 weight buffer for non-shared weights (either
// float or host-dequantized quantized weights). For 2-D weights the output is
// row-major `[O, I]`; for 3-D weights the output is `[H, I, O]` matching
// BatchedMatmul's RHS layout `BHWC(1, H, I, O)`.
std::vector<float> ExtractBlockFfwHostFallback(
    const TfLiteTensor* tensors, size_t tensors_size,
    const BlockFfwWeightOperandSpec& spec);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_BLOCK_FFW_PARSER_H_
