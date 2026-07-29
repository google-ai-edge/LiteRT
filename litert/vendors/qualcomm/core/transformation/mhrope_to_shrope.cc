// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/mhrope_to_shrope.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pack_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/unpack_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {

// Pattern indices (relative to start_index):
//   0: StridedSlice (half-rotation: first half of D)
//   1: StridedSlice (half-rotation: second half of D, possibly negated)
//   2: Concat  → produces rotate_half(x), shape [B,S,H,D]
//   3: ElementWiseBinary (Mul) — x * cos,          input0=rope_input, input1=cos
//   4: ElementWiseBinary (Mul) — rotate(x) * sin,  input0=concat_out, input1=sin
//   5: ElementWiseBinary (Add) — x*cos + rotate(x)*sin
//   6: Convert (quantize)
//   7: Transpose [0,2,1,3]: [B,S,H,D] → [B,H,S,D]
constexpr size_t kSlice0Idx = 0;
constexpr size_t kSlice1Idx = 1;
constexpr size_t kConcatIdx = 2;
constexpr size_t kMulCosIdx = 3;  // x * cos
constexpr size_t kMulSinIdx = 4;  // rotate(x) * sin
constexpr size_t kAddIdx = 5;
constexpr size_t kConvertIdx = 6;
constexpr size_t kTransposeIdx = 7;

// The head axis in [B, S, H, D].
constexpr size_t kHeadAxis = 2;
// Slice range layout: each axis has 3 entries (begin, end, stride).
constexpr size_t kRangeNumCols = 3;

// Build a 3-D range tensor for a [B,S,D] StridedSlice by dropping the H-axis
// row from the original 4-D range tensor.
const TensorWrapper& BuildRanges3D(TensorPool& tensor_pool,
                                   const TensorWrapper& mh_ranges) {
  auto ranges_opt = mh_ranges.GetTensorData<int32_t>();
  const auto& rd = ranges_opt.value();
  std::vector<int32_t> sh_ranges;
  sh_ranges.reserve(3 * kRangeNumCols);
  for (size_t axis = 0; axis < 4; ++axis) {
    if (axis == kHeadAxis) continue;
    sh_ranges.push_back(rd[axis * kRangeNumCols + 0]);
    sh_ranges.push_back(rd[axis * kRangeNumCols + 1]);
    sh_ranges.push_back(rd[axis * kRangeNumCols + 2]);
  }
  return tensor_pool.CreateStaticTensor(
      mh_ranges.GetDataType(), mh_ranges.GetQuantParams(),
      {3, static_cast<uint32_t>(kRangeNumCols)},
      static_cast<uint32_t>(sh_ranges.size() * sizeof(int32_t)),
      sh_ranges.data());
}

// Unpack a 4-D embedding tensor along axis 2 (H) into per-head 3-D tensors.
// If the tensor's axis-2 size is 1, returns a single 3-D tensor reused for
// all heads (QNN broadcasts [1,S,D] → [B,S,D]).
std::vector<ConstTensorWrapperRef> UnpackEmbedding(TensorPool& tensor_pool,
                                                   std::vector<OpWrapper>& new_ops,
                                                   const TensorWrapper& embed,
                                                   uint32_t num_heads) {
  if (embed.GetRank() != 4) {
    // Already rank < 4; reuse for all heads.
    std::vector<ConstTensorWrapperRef> result;
    result.reserve(num_heads);
    for (uint32_t h = 0; h < num_heads; ++h) {
      result.emplace_back(embed);
    }
    return result;
  }
  uint32_t embed_h = embed.GetDimension(static_cast<size_t>(kHeadAxis));
  // Drop the H axis to get the output dims.
  auto out_dims = embed.GetDimensions();
  out_dims.erase(out_dims.begin() + kHeadAxis);

  if (embed_h == num_heads) {
    // Full per-head embedding: unpack H outputs.
    std::vector<ConstTensorWrapperRef> outputs;
    outputs.reserve(num_heads);
    for (uint32_t h = 0; h < num_heads; ++h) {
      outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(embed, out_dims));
    }
    new_ops.emplace_back(
        CreateUnpackOp(embed, outputs, static_cast<uint32_t>(kHeadAxis)));
    return outputs;
  } else {
    // H==1 or other broadcast case: squeeze to rank-3 via a single Unpack.
    std::vector<ConstTensorWrapperRef> one_out;
    one_out.emplace_back(tensor_pool.CloneNativeTensorFrom(embed, out_dims));
    new_ops.emplace_back(
        CreateUnpackOp(embed, one_out, static_cast<uint32_t>(kHeadAxis)));
    // Return the same rank-3 tensor for every head.
    std::vector<ConstTensorWrapperRef> result;
    result.reserve(num_heads);
    for (uint32_t h = 0; h < num_heads; ++h) {
      result.emplace_back(one_out[0]);
    }
    return result;
  }
}

// Build a single-head RoPE block operating on rank-3 tensors [B,S,D]:
//   input_h  (unpacked head slice)
//   → Slice0 / Slice1 → Concat (rotate_half)
//   → Mul(cos) using input_h * cos_h
//   → Mul(sin) using rotated_h * sin_h
//   → Add
// Returns the Add output tensor.
const TensorWrapper& BuildSingleSHRoPE(std::vector<OpWrapper>& new_ops,
                                       TensorPool& tensor_pool,
                                       const TensorWrapper& input_h,
                                       const TensorWrapper& cos_h,
                                       const TensorWrapper& sin_h,
                                       const OpWrapper& mh_slice0,
                                       const OpWrapper& mh_slice1,
                                       const OpWrapper& mh_concat,
                                       const OpWrapper& mh_mul_cos,
                                       const OpWrapper& mh_mul_sin,
                                       const OpWrapper& mh_add) {
  // Build 3-D range tensors (drop H axis row from the original 4-D ranges).
  const auto& ranges0 = BuildRanges3D(
      tensor_pool, mh_slice0.GetTensorParam(0).GetTensor());
  const auto& ranges1 = BuildRanges3D(
      tensor_pool, mh_slice1.GetTensorParam(0).GetTensor());

  // Slice 0: [B,S,D] → [B,S,D/2]
  auto s0_out_dims = mh_slice0.GetOutputTensor(0).GetDimensions();
  s0_out_dims.erase(s0_out_dims.begin() + kHeadAxis);
  const auto& s0_out =
      tensor_pool.CloneNativeTensorFrom(mh_slice0.GetOutputTensor(0), s0_out_dims);
  new_ops.emplace_back(CreateSliceOp(input_h, s0_out, ranges0));

  // Slice 1: [B,S,D] → [B,S,D/2]
  auto s1_out_dims = mh_slice1.GetOutputTensor(0).GetDimensions();
  s1_out_dims.erase(s1_out_dims.begin() + kHeadAxis);
  const auto& s1_out =
      tensor_pool.CloneNativeTensorFrom(mh_slice1.GetOutputTensor(0), s1_out_dims);
  new_ops.emplace_back(CreateSliceOp(input_h, s1_out, ranges1));

  // Concat: [B,S,D/2] ++ [B,S,D/2] → [B,S,D] (axis = rank-1 = 2)
  // Original concat has slice1 at input[0] and slice0 at input[1]; preserve order.
  auto cat_out_dims = s1_out_dims;
  cat_out_dims.back() += s0_out_dims.back();
  const auto& cat_out =
      tensor_pool.CloneNativeTensorFrom(mh_concat.GetOutputTensor(0), cat_out_dims);
  new_ops.emplace_back(
      CreateConcatenationOp({s1_out, s0_out}, cat_out, 2));

  // Mul cos: input_h * cos_h  → [B,S,D]
  auto mul_cos_out_dims = cat_out_dims;
  const auto& mul_cos_out =
      tensor_pool.CloneNativeTensorFrom(mh_mul_cos.GetOutputTensor(0), mul_cos_out_dims);
  new_ops.emplace_back(
      CreateElementWiseMulOp(input_h, cos_h, mul_cos_out));

  // Mul sin: cat_out * sin_h  → [B,S,D]
  auto mul_sin_out_dims = cat_out_dims;
  const auto& mul_sin_out =
      tensor_pool.CloneNativeTensorFrom(mh_mul_sin.GetOutputTensor(0), mul_sin_out_dims);
  new_ops.emplace_back(
      CreateElementWiseMulOp(cat_out, sin_h, mul_sin_out));

  // Add: mul_cos_out + mul_sin_out  → [B,S,D]
  auto add_out_dims = cat_out_dims;
  const auto& add_out =
      tensor_pool.CloneNativeTensorFrom(mh_add.GetOutputTensor(0), add_out_dims);
  new_ops.emplace_back(
      CreateElementWiseAddOp(mul_cos_out, mul_sin_out, add_out));

  return add_out;
}

}  // namespace

size_t MHRoPEToSHRoPE(std::function<bool(OpWrapper&)> validate_op_config,
                      std::vector<OpWrapper>& ops, size_t start_index,
                      TensorPool& tensor_pool, size_t pattern_size) {
  const auto& slice0_op  = ops[start_index + kSlice0Idx];
  const auto& slice1_op  = ops[start_index + kSlice1Idx];
  const auto& concat_op  = ops[start_index + kConcatIdx];
  const auto& mul_cos_op = ops[start_index + kMulCosIdx];
  const auto& mul_sin_op = ops[start_index + kMulSinIdx];
  const auto& add_op     = ops[start_index + kAddIdx];
  const auto& convert_op = ops[start_index + kConvertIdx];
  const auto& transpose_op = ops[start_index + kTransposeIdx];

  // ---- Connectivity checks ----
  // Both slices take the same rope_input.
  const auto& rope_input = slice0_op.GetInputTensor(0);
  if (slice1_op.GetInputTensor(0) != rope_input) return 1;

  // Concat takes slice0 and slice1 outputs.
  if (slice0_op.GetOutputTensor(0) != concat_op.GetInputTensor(1)) return 1;
  if (slice1_op.GetOutputTensor(0) != concat_op.GetInputTensor(0)) return 1;

  // Standard RoPE: mul_cos takes rope_input as input 0, mul_sin takes
  // concat output as input 0.
  if (mul_cos_op.GetInputTensor(0) != rope_input) return 1;
  if (concat_op.GetOutputTensor(0) != mul_sin_op.GetInputTensor(0)) return 1;

  // Add takes mul_cos and mul_sin outputs.
  if (mul_cos_op.GetOutputTensor(0) != add_op.GetInputTensor(0)) return 1;
  if (mul_sin_op.GetOutputTensor(0) != add_op.GetInputTensor(1)) return 1;

  // Convert and Transpose are sequential after Add.
  if (add_op.GetOutputTensor(0)     != convert_op.GetOutputTensor(0) &&
      add_op.GetOutputTensor(0)     != convert_op.GetInputTensor(0)) return 1;
  // Allow convert to be a no-op passthrough check: just verify chain.
  if (add_op.GetOutputTensor(0)     != convert_op.GetInputTensor(0)) return 1;
  if (convert_op.GetOutputTensor(0) != transpose_op.GetInputTensor(0)) return 1;

  // Slice range params must be static (needed to build 3-D versions).
  if (!slice0_op.GetTensorParam(0).GetTensor().IsTensorStatic()) return 1;
  if (!slice1_op.GetTensorParam(0).GetTensor().IsTensorStatic()) return 1;

  // Input must be rank 4: [B, S, H, D].
  if (rope_input.GetRank() != 4) return 1;
  const uint32_t num_heads = rope_input.GetDimension(kHeadAxis);
  if (num_heads == 0) return 1;

  QNN_LOG_INFO("[G2G] Transforming MH-RoPE to SH-RoPE (eliminating Transpose)");

  std::vector<OpWrapper> new_ops;

  // ---- 1. Unpack rope_input along axis=2 (H): [B,S,H,D] → H × [B,S,D] ----
  auto unpack_dim = rope_input.GetDimensions();
  unpack_dim.erase(unpack_dim.begin() + kHeadAxis);
  std::vector<ConstTensorWrapperRef> unpack_outs;
  unpack_outs.reserve(num_heads);
  for (uint32_t h = 0; h < num_heads; ++h) {
    unpack_outs.emplace_back(
        tensor_pool.CloneNativeTensorFrom(rope_input, unpack_dim));
  }
  new_ops.emplace_back(
      CreateUnpackOp(rope_input, unpack_outs, static_cast<uint32_t>(kHeadAxis)));

  // ---- 2. Unpack cos/sin embedding tensors (input 1 of each Mul) ----
  const auto& cos_embed = mul_cos_op.GetInputTensor(1);
  const auto& sin_embed = mul_sin_op.GetInputTensor(1);
  auto cos_per_head = UnpackEmbedding(tensor_pool, new_ops, cos_embed, num_heads);
  auto sin_per_head = UnpackEmbedding(tensor_pool, new_ops, sin_embed, num_heads);

  // ---- 3. Per-head RoPE blocks ----
  std::vector<ConstTensorWrapperRef> pack_inputs;
  pack_inputs.reserve(num_heads);
  for (uint32_t h = 0; h < num_heads; ++h) {
    const auto& rope_h_out =
        BuildSingleSHRoPE(new_ops, tensor_pool,
                          unpack_outs[h].get(),
                          cos_per_head[h].get(),
                          sin_per_head[h].get(),
                          slice0_op, slice1_op, concat_op,
                          mul_cos_op, mul_sin_op, add_op);
    pack_inputs.emplace_back(rope_h_out);
  }

  // ---- 4. Pack(axis=1): H × [B,S,D] → [B,H,S,D] ----
  auto pack_out_dims = unpack_dim;  // [B, S, D]
  pack_out_dims.insert(pack_out_dims.begin() + 1, num_heads);  // [B, H, S, D]
  const auto& pack_out = tensor_pool.CloneNativeTensorFrom(
      add_op.GetOutputTensor(0), pack_out_dims);
  new_ops.emplace_back(CreatePackOp(pack_inputs, pack_out, 1));

  // ---- 5. Convert (same quantization as original): pack_out → transpose_output ----
  // The original Convert output feeds into Transpose; after transformation the
  // Convert output IS the final tensor (same as old transpose output).
  const auto& final_out = transpose_op.GetOutputTensor(0);
  new_ops.emplace_back(CreateConvertOp(pack_out, final_out));

  // ---- Validate new subgraph ----
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](OpWrapper& op) -> bool {
                    return validate_op_config(op);
                  });
  if (!is_valid) {
    QNN_LOG_WARNING(
        "[G2G] MHRoPEToSHRoPE: validation failed. Rolling back to original.");
    return 1;
  }

  // Rename to avoid QNN JSON dump collisions.
  for (size_t i = 0; i < new_ops.size(); ++i) {
    new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_rope_", i));
  }

  // Replace the matched pattern with the new subgraph.
  const size_t step_size = new_ops.size();
  ops.insert(ops.begin() + start_index + pattern_size,
             std::make_move_iterator(new_ops.begin()),
             std::make_move_iterator(new_ops.end()));
  ops.erase(ops.begin() + start_index,
            ops.begin() + start_index + pattern_size);

  QNN_LOG_INFO("[G2G] Done transforming MH-RoPE to SH-RoPE!");
  return step_size;
}

}  // namespace qnn
