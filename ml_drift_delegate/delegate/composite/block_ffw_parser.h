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

#include <any>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

// StableHLO composite names for 4-stage block-structured feed-forward layers.
inline constexpr absl::string_view kBlockedFfwCompositeName =
    "odml.blocked_ffw";
inline constexpr absl::string_view kBlockFfwCompositeName = "odml.block_ffw";

// Internal GraphFloat32 / IrModel node operation types emitted by the
// block_ffw parsers.
inline constexpr absl::string_view kBlockFfwUnpackPartType =
    "odml.block_ffw_unpack_part";
inline constexpr absl::string_view kBlockFfwPackPartsType =
    "odml.block_ffw_pack_parts";
inline constexpr absl::string_view kBlockFfwSiluType = "odml.block_ffw_silu";

// Attributes for an internal `kBlockFfwUnpackPartType` node that slices one
// channel partition (`part` in `[0, num_parts)`) from an activation tensor.
struct BlockFfwUnpackPartAttributes {
  // Zero-based index of the partition extracted by this node.
  int part = 0;
  // Total number of equal-sized channel partitions (K1).
  int num_parts = 0;
};

// Attributes for an internal `kBlockFfwPackPartsType` node that interleaves
// `num_parts` edge-stage branch outputs along channels.
struct BlockFfwPackPartsAttributes {
  // Number of input branch tensors to pack (K1).
  int num_parts = 0;
};

// Optional static zero-point-0 signed INT8 (`[-128, 127]`) per-tensor
// activation quantization scales for a single linear stage. `std::nullopt`
// means float activations without quantize-and-dequantize.
struct BlockFfwStageQuantScales {
  std::optional<float> in_scale;
  std::optional<float> out_scale;
};

// Parsed FlexBuffer activation quantization scales for the four sequential
// linear stages (`u1`, `u2`, `d1`, `d2`) of an `odml.blocked_ffw` composite.
struct BlockFfwAttributes {
  BlockFfwStageQuantScales u1;
  BlockFfwStageQuantScales u2;
  BlockFfwStageQuantScales d1;
  BlockFfwStageQuantScales d2;
};

// Validated tensor metadata and canonical layout for one weight operand of an
// `odml.blocked_ffw` composite node.
struct BlockFfwWeightOperandSpec {
  // Positional index in `tflite_node.inputs`.
  int node_input_index = -1;
  // TFLite subgraph tensor index in `TfLiteContext::tensors`.
  int tensor_id = -1;
  TfLiteType type = kTfLiteNoType;
  TfLiteQuantizationType quant_type = kTfLiteNoQuantization;
  // Canonical weight shape `(O, H, 1, I)` where `H == 1` for 2-D edge weights
  // and `H == K2` for 3-D middle weights.
  ::ml_drift::OHWI shape;
  // Canonical quantization scale shape in OHWI layout.
  ::ml_drift::OHWI scale_shape;
  // Whether quantization parameters include an explicit zero-point tensor.
  bool has_zero_point = false;
  // True for 3-D batched middle weights (U2/D1); false for 2-D edge weights
  // (U1/D2). Distinguishes 2-D `[O, I]` from 3-D `[1, I, O]` / `[1, O, I]` when
  // `K2 == 1`.
  bool is_batched = false;
  // Only applies when `is_batched` is true: true if 3-D weights are stored as
  // `[H, I, O]`, false if stored as `[H, O, I]` and need transposing for
  // BatchedMatmul host fallbacks.
  bool oi_transposed = false;
};

// Resolved block dimensions, activation attributes, and validated weight specs
// for an `odml.blocked_ffw` composite node.
struct ResolvedBlockFfwPlan {
  // Sequence length (`T`) and hidden width (`D`) of `[1, T, D]` I/O tensors.
  int tokens = 0;
  int model_dim = 0;
  // Block-factorization counts: `K1` edge partitions and `K2` middle groups.
  int k1 = 0;
  int k2 = 0;
  // Per-partition edge-input (`I1 = D / K1`) and U1 output (`O1 = K2 * S`).
  int i1 = 0;
  int o1 = 0;
  // Channels per `(K1, K2)` sub-block (`S = O1 / K2`).
  int s = 0;
  // Per-group hidden width produced by U2 (`O2`) and D2 output width (`O4`).
  int o2 = 0;
  int o4 = 0;
  BlockFfwAttributes attrs;
  // `K1` 2-D weights for stage 1 (U1) and stage 4 (D2).
  std::vector<BlockFfwWeightOperandSpec> u1_weights;
  // 3-D batched weights for middle stages U2 and D1.
  BlockFfwWeightOperandSpec u2_weight;
  BlockFfwWeightOperandSpec d1_weight;
  std::vector<BlockFfwWeightOperandSpec> d2_weights;
};

// Returns true if `type` is a supported quantized weight dtype
// (INT2/INT4/INT8).
bool IsBlockFfwQuantizedWeightType(TfLiteType type);

// Validates and resolves shapes, layouts, and quantization metadata for an
// `odml.blocked_ffw` composite node.
absl::StatusOr<ResolvedBlockFfwPlan> ResolveBlockFfwPlan(
    const TfLiteTensor* tensors, size_t tensors_size,
    const TfLiteNode& tflite_node);

// Extracts a host-side float32 weight buffer for non-shared weights (either
// float or host-dequantized quantized weights). For 2-D weights the output is
// row-major `[O, I]`; for 3-D weights the output is `[H, I, O]` matching
// BatchedMatmul's RHS layout `BHWC(1, H, I, O)`.
std::vector<float> ExtractBlockFfwHostFallback(
    const TfLiteTensor* tensors, size_t tensors_size,
    const BlockFfwWeightOperandSpec& spec);

// Operation type and attributes for a shared quantized FullyConnected op
// (INT2/INT4/INT8) emitted by both the GraphFloat32 and IrModel parsers.
struct BlockFfwQuantizedFcOpSpec {
  std::string op_type;
  std::any attributes;
};

// Builds the ML Drift quantized FullyConnected op type and attributes for a
// shared INT2/INT4/INT8 weight operand.
BlockFfwQuantizedFcOpSpec BuildBlockFfwQuantizedFcOpSpec(
    const BlockFfwWeightOperandSpec& spec);

// GraphFloat32 operation parser for `odml.blocked_ffw` / `odml.block_ffw`
// StableHLO composite ops.
class BlockFfwOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final;
  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_BLOCK_FFW_PARSER_H_
