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

#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_kernel.h"

#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"

namespace litert::ml_drift {
namespace {

using TensorHandle = ::ml_drift::GpuModelBuilder::TensorHandle;

// Index of param[0] (start position p0) in the int32 param tensor.
constexpr int kStartParamIndex = 0;

// Ring slot j holds position j until the ring has wrapped, so while p0 < W
// only the slots below p0 hold tokens. The past logits and the P.V product
// are bounded by p0 at runtime (ConvRuntimeCheckDesc), which skips the cache
// entirely for the first prefill chunk (p0 = 0). Depending on the kernel
// chosen, those bounds round p0 up to 4 or to 32 channels, so the softmax
// visits the 32-rounded bound `past_end` (in slices) and gives every empty
// slot (>= `past_filled`) a probability of exactly 0, whatever the mask says.
// Past `past_end`, it writes 0 to the one aligned block of slices that the
// bounded loops may still read (see CreateMaskedSoftmaxOp).
std::string PastEndSliceCode(const std::string& start,
                             const std::string& past_slices) {
  return "  int past_end = max(" +
         ::ml_drift::ConvRuntimeCheckDesc().GetRuntimeEndSlice(start,
                                                               past_slices) +
         ", 0);\n  int past_filled = min(max(" + start + ", 0), " +
         past_slices + " * 4);\n";
}

// Code that reads slice `s` of `SRC` into the float4 `v`. With `apply_mask`,
// it also soft-caps it and applies mask slice `S`.
std::string ReadMaskedLogitsCode(const std::string& src, bool has_softcap,
                                 bool bool_mask, bool apply_mask = true) {
  std::string code = absl::StrReplaceAll(R"(
    float4 v = args.SRC.Read<float>(X, Y, s, 0);
)",
                                         {{"SRC", src}});
  if (apply_mask && has_softcap) {
    code += R"(
    v = ucl::Init<float4>(args.softcap) * tanh(v * ucl::Init<float4>(args.inv_softcap));
)";
  }
  if (apply_mask && bool_mask) {
    // -10000 matches the fill value of the ML Drift SDPA kernels and of the
    // composite's reference decomposition.
    code += R"(
    bool4 m = args.mask.Read<bool>(t, 0, S, 0);
    v.x = m.x ? v.x : -10000.0f;
    v.y = m.y ? v.y : -10000.0f;
    v.z = m.z ? v.z : -10000.0f;
    v.w = m.w ? v.w : -10000.0f;
)";
  } else if (apply_mask) {
    code += R"(
    v += args.mask.Read<float>(t, 0, S, 0);
)";
  }
  if (src == "logits_past") {
    code += R"(
    int c = s * 4;
    v.x = c + 0 < past_filled ? v.x : -3.0e38f;
    v.y = c + 1 < past_filled ? v.y : -3.0e38f;
    v.z = c + 2 < past_filled ? v.z : -3.0e38f;
    v.w = c + 3 < past_filled ? v.w : -3.0e38f;
)";
  }
  if (src == "logits_new") {
    // Channels past T in the last slice do not exist; exp() maps them to 0.
    code += R"(
    int c = s * 4;
    v.y = c + 1 < args.new_len ? v.y : -3.0e38f;
    v.z = c + 2 < args.new_len ? v.z : -3.0e38f;
    v.w = c + 3 < args.new_len ? v.w : -3.0e38f;
)";
  }
  return code;
}

// A GPUOperation with a fixed work group size, for kernels that reduce across
// the work items of a work group.
class FixedWorkGroupOp : public ::ml_drift::GPUOperation {
 public:
  explicit FixedWorkGroupOp(const ::ml_drift::int3& work_group_size) {
    work_group_size_ = work_group_size;
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  FixedWorkGroupOp(FixedWorkGroupOp&&) = default;
  FixedWorkGroupOp& operator=(FixedWorkGroupOp&&) = default;
  FixedWorkGroupOp(const FixedWorkGroupOp&) = delete;
  FixedWorkGroupOp& operator=(const FixedWorkGroupOp&) = delete;
};

// Soft-caps and masks [logits_past | logits_new] with the [1, 1, T, W + T]
// mask and computes the softmax over all W + T keys. Query rows are packed
// g-major, so row r uses mask row r % T.
//
// Every row is handled by up to 2 work items of one work group: pass 1 reduces
// the row to its max and sum of exponentials (online softmax, F32), pass 2
// re-reads the logits and writes the probabilities. Only the cache slots below
// the runtime bound of PastEndSliceCode() are visited.
//
// The new-token block is produced in one of two ways:
//  * `dst_new` holds probabilities [1, H, R, T] (written in pass 2), or
//  * with `defer_new`, `logits_new` must already be soft-capped and masked,
//    and `dst_new` receives the row statistics [1, H, R, 4] = (1 / sum, max)
//    instead. The consumer applies exp(v - max) / sum while it reads the
//    logits (`src_exp` of the ML Drift matmuls), so the T new-token columns
//    are read once here and never written.
std::unique_ptr<::ml_drift::GPUOperation> CreateMaskedSoftmaxOp(
    const ::ml_drift::TensorDescriptor& logits_past,
    const ::ml_drift::TensorDescriptor& logits_new,
    const ::ml_drift::TensorDescriptor& mask,
    const ::ml_drift::TensorDescriptor& param,
    const ::ml_drift::TensorDescriptor& probs_past,
    const ::ml_drift::TensorDescriptor& dst_new, int cache_size, int new_len,
    const FusedSdpaCacheUpdateAttributes& attr, bool defer_new) {
  const int past_slices = cache_size / 4;
  const int new_slices = (new_len + 3) / 4;
  // Tensors are contiguous along rows within a slice, so a work group spans
  // many rows and splits each row over only a few work items (every work item
  // gets at least 4 slices of its row).
  int threads_per_row = 1;
  while (threads_per_row < 2 &&
         threads_per_row * 8 <= past_slices + new_slices) {
    threads_per_row *= 2;
  }
  const int rows_per_group = 128 / threads_per_row;
  FixedWorkGroupOp op(::ml_drift::int3(rows_per_group, 1, threads_per_row));
  op.tensor_to_grid_ = ::ml_drift::TensorToGrid::kWBToX_HDToY_ZIs1;
  op.AddSrcTensor("logits_past", logits_past);
  op.AddSrcTensor("logits_new", logits_new);
  op.AddSrcTensor("mask", mask);
  op.AddSrcTensor("params", param);
  op.AddDstTensor("probs_past", probs_past);
  op.AddDstTensor(defer_new ? "stats" : "probs_new", dst_new);
  op.args_.AddInt("past_slices", past_slices);
  op.args_.AddInt("new_slices", new_slices);
  op.args_.AddInt("new_len", new_len);
  op.args_.AddInt("zero_slices",
                  ::ml_drift::ConvRuntimeCheckDesc().GetSlicesAlignment());

  const bool has_softcap = attr.softcap.has_value() && *attr.softcap > 0.0f;
  if (has_softcap) {
    op.args_.AddFloat("softcap", *attr.softcap);
    op.args_.AddFloat("inv_softcap", 1.0f / *attr.softcap);
  }
  const bool bool_mask = mask.GetDataType() == ::ml_drift::DataType::kBool;
  const std::string read_past =
      ReadMaskedLogitsCode("logits_past", has_softcap, bool_mask);
  const std::string read_new = ReadMaskedLogitsCode(
      "logits_new", has_softcap, bool_mask, /*apply_mask=*/!defer_new);

  // The trailing batch coordinate is ignored by tensors without a batch axis.
  // Work items of rows past the tensor bounds still take part in the barriers.
  std::string code = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int lz = ucl::GetLocalId<2>();
  int lxy = ucl::GetLocalId<1>() * ROWS_PER_GROUP + ucl::GetLocalId<0>();
  __local float2 loc_mem[THREADS_PER_ROW][ROWS_PER_GROUP];
  bool is_active = X < args.probs_past.Width() && Y < args.probs_past.Height();
  int t = X % args.new_len;
)" + PastEndSliceCode("args.params.Read(0, 0, 0, 0).x", "args.past_slices") +
                     R"(
  float maximum = -3.0e38f;
  float sum = 0.0f;
  if (is_active) {
    for (int s = lz; s < past_end; s += THREADS_PER_ROW) {
      int S = s;
)" + read_past + R"(
      float new_max = max(max(max(v.x, v.y), max(v.z, v.w)), maximum);
      sum = sum * exp(maximum - new_max) +
            dot(exp(v - ucl::Init<float4>(new_max)), ucl::Init<float4>(1.0f));
      maximum = new_max;
    }
    for (int s = lz; s < args.new_slices; s += THREADS_PER_ROW) {
      int S = args.past_slices + s;
)" + read_new + R"(
      float new_max = max(max(max(v.x, v.y), max(v.z, v.w)), maximum);
      sum = sum * exp(maximum - new_max) +
            dot(exp(v - ucl::Init<float4>(new_max)), ucl::Init<float4>(1.0f));
      maximum = new_max;
    }
  }
  float2 partial;
  partial.x = sum;
  partial.y = maximum;
  loc_mem[lz][lxy] = partial;
  ucl::SyncThreads<WorkGroup, Local>();
  if (lz == 0) {
    for (int i = 1; i < THREADS_PER_ROW; ++i) {
      float2 other = loc_mem[i][lxy];
      float new_max = max(partial.y, other.y);
      partial.x = partial.x * exp(partial.y - new_max) +
                  other.x * exp(other.y - new_max);
      partial.y = new_max;
    }
    loc_mem[0][lxy] = partial;
  }
  ucl::SyncThreads<WorkGroup, Local>();
  if (!is_active) {
    return;
  }
  partial = loc_mem[0][lxy];
  float4 row_max = ucl::Init<float4>(partial.y);
  float4 inv_sum = ucl::Init<float4>(1.0f / partial.x);
  for (int s = lz; s < past_end; s += THREADS_PER_ROW) {
    int S = s;
)" + read_past + R"(
    float4 p = exp(v - row_max) * inv_sum;
    args.probs_past::type prob_val =
        ucl::Convert<args.probs_past::type>(p);
    args.probs_past.Write(prob_val, X, Y, s);
  }
  // The bounded P.V product steps over its input in blocks of slices that
  // divide the bound alignment, and its do-while loop reads one block even
  // for a bound of 0. So only the first aligned block past the bound needs an
  // explicit probability of 0; later slices are never read.
  int zero_end = min(past_end + args.zero_slices, args.past_slices);
  for (int s = past_end + lz; s < zero_end; s += THREADS_PER_ROW) {
    args.probs_past::type zero_prob =
        ucl::Init<args.probs_past::type>(0.0f);
    args.probs_past.Write(zero_prob, X, Y, s);
  }
)";
  if (defer_new) {
    code += R"(
  if (lz == 0) {
    float4 stats = ucl::Init<float4>(1.0f / partial.x, partial.y, 0.0f, 0.0f);
    args.stats::type stats_val = ucl::Convert<args.stats::type>(stats);
    args.stats.Write(stats_val, X, Y, 0);
  }
}
)";
  } else {
    code += R"(
  for (int s = lz; s < args.new_slices; s += THREADS_PER_ROW) {
    int S = args.past_slices + s;
)" + read_new +
            R"(
    float4 p = exp(v - row_max) * inv_sum;
    args.probs_new::type prob_val =
        ucl::Convert<args.probs_new::type>(p);
    args.probs_new.Write(prob_val, X, Y, s);
  }
}
)";
  }
  op.code_ = absl::StrReplaceAll(
      code, {{"ROWS_PER_GROUP", std::to_string(rows_per_group)},
             {"THREADS_PER_ROW", std::to_string(threads_per_row)}});
  return std::make_unique<FixedWorkGroupOp>(std::move(op));
}

// Applies the new-token columns [W, W + T) of the [1, 1, T, W + T] mask to
// `logits_new` [1, H, R, T]: `mask ? logits : -10000` for a bool mask and
// `logits + mask` for an additive one. Row r reads mask row r % T. Being
// elementwise, it is linked into the epilogue of the QK^T matmul.
TensorHandle MaskNewLogits(::ml_drift::GpuModelBuilder* model_builder,
                           const TensorHandle& logits_new,
                           const TensorHandle& mask, int cache_size,
                           int new_len) {
  ::ml_drift::ElementwiseDescriptor op_desc;
  // The mask is read explicitly (no `in2_value`), at a slice offset of W / 4.
  // The trailing batch coordinate is ignored by tensors without a batch axis.
  const std::string read_mask = absl::StrReplaceAll(
      "args.src_tensor_1.Read<TYPE>(X_COORD % NEW_LEN, 0, S_COORD + "
      "PAST_SLICES, 0)",
      {{"NEW_LEN", std::to_string(new_len)},
       {"PAST_SLICES", std::to_string(cache_size / 4)}});
  if (mask.tensor_desc.GetDataType() == ::ml_drift::DataType::kBool) {
    op_desc.args.AddFloat("mask_value", -10000.0f,
                          logits_new.tensor_desc.GetDataType());
    op_desc.code =
        "  bool4 m = " + absl::StrReplaceAll(read_mask, {{"TYPE", "bool"}}) +
        R"(;
  out_value.x = m.x ? in_value.x : args.mask_value;
  out_value.y = m.y ? in_value.y : args.mask_value;
  out_value.z = m.z ? in_value.z : args.mask_value;
  out_value.w = m.w ? in_value.w : args.mask_value;
)";
  } else {
    const std::string vec_type =
        ::ml_drift::ToUclDataType(logits_new.tensor_desc.GetDataType(), 4);
    op_desc.code = absl::StrCat(
        "  float4 m = ", absl::StrReplaceAll(read_mask, {{"TYPE", "float"}}),
        ";\n  out_value = ucl::Convert<", vec_type,
        ">(ucl::Convert<float4>(in_value) + m);\n");
  }
  TensorHandle dst = model_builder->AddTensor(logits_new.tensor_desc);
  ::ml_drift::OperationDef definition;
  definition.src_tensors.push_back(logits_new.tensor_desc);
  definition.src_tensors.push_back(mask.tensor_desc);
  definition.dst_tensors.push_back(dst.tensor_desc);
  auto op = std::make_unique<::ml_drift::GPUOperation>(
      ::ml_drift::CreateGpuOperation(definition, std::move(op_desc)));
  model_builder->AddGpuOperation({logits_new, mask}, {dst}, std::move(op),
                                 "fused_sdpa_mask_new");
  return dst;
}

// A GPUOperation with a fixed grid of work items.
class FixedGridOp : public ::ml_drift::GPUOperation {
 public:
  explicit FixedGridOp(const ::ml_drift::int3& grid_size)
      : fixed_grid_size_(grid_size) {}

  ::ml_drift::int3 GetGridSize() const override { return fixed_grid_size_; }

  FixedGridOp(FixedGridOp&&) = default;
  FixedGridOp& operator=(FixedGridOp&&) = default;
  FixedGridOp(const FixedGridOp&) = delete;
  FixedGridOp& operator=(const FixedGridOp&) = delete;

 private:
  ::ml_drift::int3 fixed_grid_size_;
};

TensorHandle DequantizeInt8Tensor(::ml_drift::GpuModelBuilder* model_builder,
                                  const TensorHandle& src, float scale,
                                  ::ml_drift::DataType compute_dtype) {
  TensorHandle dst = model_builder->Cast(src, ::ml_drift::DataType::kFloat32);
  dst = model_builder->Multiplication(dst, scale);
  if (compute_dtype != ::ml_drift::DataType::kFloat32) {
    dst = model_builder->Cast(dst, compute_dtype);
  }
  return dst;
}

// Rounds `src` onto the symmetric INT8 or INT4 cache grid `[-q_min, q_max] *
// scale` so new tokens are attended identically to how the cache stores them.
TensorHandle FakeQuantizeToCacheGrid(::ml_drift::GpuModelBuilder* model_builder,
                                     const TensorHandle& src, float scale,
                                     int quant_bits) {
  const ::ml_drift::BHWC shape = src.tensor_desc.GetBHWCShape();
  const int c_slices = (shape.c + 3) / 4;
  TensorHandle dst = model_builder->AddTensor(src.tensor_desc);
  FixedGridOp op(::ml_drift::int3(shape.w, shape.h, c_slices));
  op.AddSrcTensor("src", src.tensor_desc);
  op.AddDstTensor("dst", dst.tensor_desc);
  op.args_.AddInt("w", shape.w);
  op.args_.AddInt("h", shape.h);
  op.args_.AddInt("c_slices", c_slices);
  const float q_min = (quant_bits == 4) ? -8.0f : -128.0f;
  const float q_max = (quant_bits == 4) ? 7.0f : 127.0f;
  op.args_.AddFloat("scale", scale);
  op.args_.AddFloat("inv_scale", 1.0f / scale);
  op.args_.AddFloat("q_min", q_min);
  op.args_.AddFloat("q_max", q_max);
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.w || Y >= args.h || S >= args.c_slices) {
    return;
  }
  float4 v = ucl::Convert<float4>(args.src.Read(X, Y, S, 0));
  float4 q = min(ucl::Init<float4>(args.q_max),
                 max(ucl::Init<float4>(args.q_min),
                     round(v * ucl::Init<float4>(args.inv_scale))));
  args.dst::type out_val =
      ucl::Convert<args.dst::type>(q * ucl::Init<float4>(args.scale));
  args.dst.Write(out_val, X, Y, S);
}
)";
  model_builder->AddGpuOperation({src}, {dst},
                                 std::make_unique<FixedGridOp>(std::move(op)),
                                 "fused_sdpa_fake_quant_new");
  return dst;
}

// Unpacks a packed INT4 ring KV cache (`[1, H, W, D / 2]` and `[1, H, D / 2,
// W]` in kUint8) into transient INT8 weight buffers (`nibble + 120 = signed +
// 128`) up to the 32-aligned filled ring prefix (with a minimum of 1 aligned
// block when p0 == 0 so bounded matmul loops read valid zeros).
std::unique_ptr<::ml_drift::GPUOperation> CreateUnpackRingInt4KvCacheOp(
    const ::ml_drift::TensorDescriptor& key_cache,
    const ::ml_drift::TensorDescriptor& value_cache,
    const ::ml_drift::TensorDescriptor& param,
    const ::ml_drift::TensorDescriptor& unpacked_k,
    const ::ml_drift::TensorDescriptor& unpacked_v, int heads, int cache_size,
    int head_dim) {
  const int half_slices = head_dim / 8;
  FixedGridOp op(::ml_drift::int3(cache_size, heads, half_slices));
  op.AddSrcTensor("src_k", key_cache);
  op.AddSrcTensor("src_v", value_cache);
  op.AddSrcTensor("params", param);
  op.AddDstTensor("dst_k", unpacked_k);
  op.AddDstTensor("dst_v", unpacked_v);
  op.args_.AddInt("cache_size", cache_size);
  op.args_.AddInt("heads", heads);
  op.args_.AddInt("half_slices", half_slices);
  op.args_.AddInt("align_tokens",
                  ::ml_drift::ConvRuntimeCheckDesc::kChannelsAlignment);
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int Z = ucl::GetGlobalId<2>();
  if (X >= args.cache_size || Y >= args.heads || Z >= args.half_slices) {
    return;
  }
  int p0 = args.params.Read(0, 0, 0, 0).x;
  int filled = min(max(p0, 0), args.cache_size);
  int aligned =
      ((filled + args.align_tokens - 1) / args.align_tokens) * args.align_tokens;
  int end_tokens = min(args.cache_size, max(args.align_tokens, aligned));
  if (X >= end_tokens) {
    return;
  }
  int c_slices = args.cache_size / 4;
  int x_slice = X / 4;
  int x_rem = X % 4;
  int h_slices_in = args.half_slices;
  int h_slices_out = args.half_slices * 2;
  {
    int k_in_idx = ((Y * h_slices_in + Z) * c_slices + x_slice) * 4 + x_rem;
    uint4 u_k = ucl::Convert<uint4>(args.src_k.Read(k_in_idx));
    uchar4 k_lo =
        ucl::Convert<uchar4>((u_k & ucl::Init<uint4>(15u)) + ucl::Init<uint4>(120u));
    uchar4 k_hi = ucl::Convert<uchar4>(
        ((u_k / ucl::Init<uint4>(16u)) & ucl::Init<uint4>(15u)) +
        ucl::Init<uint4>(120u));
    int k_out0 = ((Y * h_slices_out + Z * 2) * c_slices + x_slice) * 4 + x_rem;
    int k_out1 =
        ((Y * h_slices_out + (Z * 2 + 1)) * c_slices + x_slice) * 4 + x_rem;
    args.dst_k.WriteLinear(k_lo, k_out0);
    args.dst_k.WriteLinear(k_hi, k_out1);
  }
  {
    int v_in_idx = ((Y * c_slices + x_slice) * h_slices_in + Z) * 4 + x_rem;
    uint4 u_v = ucl::Convert<uint4>(args.src_v.Read(v_in_idx));
    uchar4 v_lo =
        ucl::Convert<uchar4>((u_v & ucl::Init<uint4>(15u)) + ucl::Init<uint4>(120u));
    uchar4 v_hi = ucl::Convert<uchar4>(
        ((u_v / ucl::Init<uint4>(16u)) & ucl::Init<uint4>(15u)) +
        ucl::Init<uint4>(120u));
    int v_out0 = ((Y * c_slices + x_slice) * h_slices_out + Z * 2) * 4 + x_rem;
    int v_out1 =
        ((Y * c_slices + x_slice) * h_slices_out + (Z * 2 + 1)) * 4 + x_rem;
    args.dst_v.WriteLinear(v_lo, v_out0);
    args.dst_v.WriteLinear(v_hi, v_out1);
  }
}
)";
  return std::make_unique<FixedGridOp>(std::move(op));
}

// Writes the last min(valid, W) valid new tokens into the ring buffer, where
// valid = clamp(param[1] - param[0], 0, T) and token X goes to slot
// (param[0] + X) % W. The caches use the packed layouts of AddValuesToCache:
//   key cache   - kOSpatialIOGroupO4I4, O = cache size, I = head dim.
//   value cache - kOSpatialIOGroupI4O4, O = head dim, I = cache size.
// Slots that are not written keep their contents, which relies on the updated
// caches being bound in place to the input caches (as for odml.cache_update).
//
// `attention` is not read. It makes the write data-dependent on the attention
// output so that it can only run after every read of the pre-write caches.
std::unique_ptr<::ml_drift::GPUOperation> CreateRingCacheWriteOp(
    const ::ml_drift::TensorDescriptor& key_new,
    const ::ml_drift::TensorDescriptor& value_new,
    const ::ml_drift::TensorDescriptor& param,
    const ::ml_drift::TensorDescriptor& attention,
    const ::ml_drift::TensorDescriptor& key_cache_out,
    const ::ml_drift::TensorDescriptor& value_cache_out, int heads,
    int cache_size, int head_dim, int new_len, int kv_quant_bits,
    float scale_k, float scale_v) {
  const int head_dim_slices =
      (kv_quant_bits == 4) ? (head_dim / 8) : (head_dim / 4);
  FixedGridOp op(::ml_drift::int3(new_len, heads, head_dim_slices));
  op.AddSrcTensor("key_new", key_new);
  op.AddSrcTensor("value_new", value_new);
  op.AddSrcTensor("params", param);
  op.AddSrcTensor("attention", attention);
  op.AddDstTensor("key_cache_out", key_cache_out);
  op.AddDstTensor("value_cache_out", value_cache_out);
  op.args_.AddInt("new_len", new_len);
  op.args_.AddInt("heads", heads);
  op.args_.AddInt("head_dim_slices", head_dim_slices);
  op.args_.AddInt("cache_size", cache_size);
  op.args_.AddInt("cache_slices", cache_size / 4);
  if (kv_quant_bits > 0) {
    op.args_.AddFloat("inv_scale_k", 1.0f / scale_k);
    op.args_.AddFloat("inv_scale_v", 1.0f / scale_v);
  }

  std::string code = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.new_len || Y >= args.heads || S >= args.head_dim_slices) {
    return;
  }
  // The trailing batch coordinate is ignored by tensors without a batch axis.
  int4 p = args.params.Read(0, 0, 0, 0);
  int start = p.x;
  if (start < 0) {
    return;
  }
  int valid = min(max(p.y - start, 0), args.new_len);
  int first = max(0, valid - args.cache_size);
  if (X < first || X >= valid) {
    return;
  }
  int slot = (start + X) % args.cache_size;
  int slot_slice = slot / 4;
  int slot_lane = slot % 4;
  int k_index =
      ((Y * args.head_dim_slices + S) * args.cache_slices + slot_slice) * 4 +
      slot_lane;
  int v_index =
      ((Y * args.cache_slices + slot_slice) * args.head_dim_slices + S) * 4 +
      slot_lane;
)";

  if (kv_quant_bits == 4) {
    code += R"(
  float4 k0 = ucl::Convert<float4>(args.key_new.Read(X, Y, S * 2, 0));
  float4 k1 = ucl::Convert<float4>(args.key_new.Read(X, Y, S * 2 + 1, 0));
  uint4 uk0 = ucl::Convert<uint4>(min(
      ucl::Init<float4>(15.0f),
      max(ucl::Init<float4>(0.0f),
          round(k0 * ucl::Init<float4>(args.inv_scale_k)) +
              ucl::Init<float4>(8.0f))));
  uint4 uk1 = ucl::Convert<uint4>(min(
      ucl::Init<float4>(15.0f),
      max(ucl::Init<float4>(0.0f),
          round(k1 * ucl::Init<float4>(args.inv_scale_k)) +
              ucl::Init<float4>(8.0f))));
  uchar4 k_packed = ucl::Convert<uchar4>(uk0 + uk1 * ucl::Init<uint4>(16u));
  args.key_cache_out.WriteLinear(k_packed, k_index);

  float v0;
  float v1;
  float v2;
  float v3;
  float v4;
  float v5;
  float v6;
  float v7;
  args.value_new.ReadPerChannel<float>(v0, S * 8 + 0, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v1, S * 8 + 1, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v2, S * 8 + 2, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v3, S * 8 + 3, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v4, S * 8 + 4, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v5, S * 8 + 5, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v6, S * 8 + 6, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v7, S * 8 + 7, Y, X, 0);
  float4 vv0 = ucl::Init<float4>(v0, v1, v2, v3);
  float4 vv1 = ucl::Init<float4>(v4, v5, v6, v7);
  uint4 uv0 = ucl::Convert<uint4>(min(
      ucl::Init<float4>(15.0f),
      max(ucl::Init<float4>(0.0f),
          round(vv0 * ucl::Init<float4>(args.inv_scale_v)) +
              ucl::Init<float4>(8.0f))));
  uint4 uv1 = ucl::Convert<uint4>(min(
      ucl::Init<float4>(15.0f),
      max(ucl::Init<float4>(0.0f),
          round(vv1 * ucl::Init<float4>(args.inv_scale_v)) +
              ucl::Init<float4>(8.0f))));
  uchar4 v_packed = ucl::Convert<uchar4>(uv0 + uv1 * ucl::Init<uint4>(16u));
  args.value_cache_out.WriteLinear(v_packed, v_index);
}
)";
  } else if (kv_quant_bits == 8) {
    code += R"(
  float4 k_f = ucl::Convert<float4>(args.key_new.Read(X, Y, S, 0));
  uchar4 k_q = ucl::Convert<uchar4>(min(
      ucl::Init<float4>(255.0f),
      max(ucl::Init<float4>(0.0f),
          round(k_f * ucl::Init<float4>(args.inv_scale_k)) +
              ucl::Init<float4>(128.0f))));
  args.key_cache_out.WriteLinear(k_q, k_index);

  float v0;
  float v1;
  float v2;
  float v3;
  args.value_new.ReadPerChannel<float>(v0, S * 4 + 0, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v1, S * 4 + 1, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v2, S * 4 + 2, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v3, S * 4 + 3, Y, X, 0);
  float4 v_f = ucl::Init<float4>(v0, v1, v2, v3);
  uchar4 v_q = ucl::Convert<uchar4>(min(
      ucl::Init<float4>(255.0f),
      max(ucl::Init<float4>(0.0f),
          round(v_f * ucl::Init<float4>(args.inv_scale_v)) +
              ucl::Init<float4>(128.0f))));
  args.value_cache_out.WriteLinear(v_q, v_index);
}
)";
  } else {
    code += R"(
  args.key_new::type k_val = args.key_new.Read(X, Y, S, 0);
  args.key_cache_out.WriteLinear(
      ucl::Convert<args.key_cache_out::type>(k_val), k_index);

  float v0;
  float v1;
  float v2;
  float v3;
  args.value_new.ReadPerChannel<float>(v0, S * 4 + 0, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v1, S * 4 + 1, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v2, S * 4 + 2, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v3, S * 4 + 3, Y, X, 0);
  float4 v_val = ucl::Init<float4>(v0, v1, v2, v3);
  args.value_cache_out.WriteLinear(
      ucl::Convert<args.value_cache_out::type>(v_val), v_index);
}
)";
  }
  op.code_ = std::move(code);
  return std::make_unique<FixedGridOp>(std::move(op));
}

// Copies `src` [1, H, N, C] into a linear buffer as the weights OHWI(N, H, 1,
// C) in the kOSpatialIOGroupO4I4 layout of the key cache, [H][C/4][N/4][4][4]
// with 4 consecutive C values innermost, so that a FullyConnected reads it as
// external weights (src . weights^T per head). When `quant_scale` is set,
// rounds each element onto the cache quantization grid in the same pass.
absl::StatusOr<::ml_drift::GpuModelBuilder::Weights> PackAsWeights(
    ::ml_drift::GpuModelBuilder* model_builder, const TensorHandle& src,
    ::ml_drift::DataType data_type,
    std::optional<float> quant_scale = std::nullopt, int quant_bits = 0) {
  const ::ml_drift::BHWC shape = src.tensor_desc.GetBHWCShape();
  const ::ml_drift::OHWI weights_shape(shape.w, shape.h, 1, shape.c);
  TensorHandle packed =
      model_builder->AddTensor(::ml_drift::CreateBhwcTensorDescriptor(
          data_type, ::ml_drift::TensorStorageType::kBuffer, shape));
  FixedGridOp op(::ml_drift::int3(shape.w, shape.h, shape.c / 4));
  op.AddSrcTensor("src", src.tensor_desc);
  op.AddDstTensor("dst", packed.tensor_desc);
  op.args_.AddInt("n", shape.w);
  op.args_.AddInt("heads", shape.h);
  op.args_.AddInt("c_slices", shape.c / 4);
  std::string code = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.n || Y >= args.heads || S >= args.c_slices) {
    return;
  }
  int index = ((Y * args.c_slices + S) * (args.n / 4) + X / 4) * 4 + X % 4;
)";
  if (quant_scale.has_value() && quant_bits > 0) {
    const float q_min = (quant_bits == 4) ? -8.0f : -128.0f;
    const float q_max = (quant_bits == 4) ? 7.0f : 127.0f;
    op.args_.AddFloat("scale", *quant_scale);
    op.args_.AddFloat("inv_scale", 1.0f / *quant_scale);
    op.args_.AddFloat("q_min", q_min);
    op.args_.AddFloat("q_max", q_max);
    code += R"(
  float4 v = ucl::Convert<float4>(args.src.Read(X, Y, S, 0));
  float4 q = min(ucl::Init<float4>(args.q_max),
                 max(ucl::Init<float4>(args.q_min),
                     round(v * ucl::Init<float4>(args.inv_scale))));
  args.dst::type out_val =
      ucl::Convert<args.dst::type>(q * ucl::Init<float4>(args.scale));
  args.dst.WriteLinear(out_val, index);
}
)";
  } else {
    code += R"(
  args.src::type v = args.src.Read(X, Y, S, 0);
  args.dst.WriteLinear(ucl::Convert<args.dst::type>(v), index);
}
)";
  }
  op.code_ = std::move(code);
  model_builder->AddGpuOperation({src}, {packed},
                                 std::make_unique<FixedGridOp>(std::move(op)),
                                 "fused_sdpa_pack_weights");
  ::ml_drift::WeightsDescription desc =
      ::ml_drift::GetFullyConnectedWeightsDesc(data_type, weights_shape);
  desc.layout = ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;
  return ::ml_drift::CreateExternalWeights(packed, desc, weights_shape);
}

}  // namespace

absl::Status BuildSdpaCacheUpdateGpuGraph(
    const std::vector<uint32_t>& input_ids,
    const std::vector<uint32_t>& output_ids,
    const FusedSdpaCacheUpdateAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() != 7) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update expects 7 inputs.");
  }
  if (output_ids.size() != 1 && output_ids.size() != 3) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update expects 1 or 3 outputs.");
  }
  ABSL_ASSIGN_OR_RETURN(TensorHandle query,
                        model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle key_cache,
                        model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle value_cache,
                        model_builder->GetTensor(input_ids[2]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle key_new,
                        model_builder->GetTensor(input_ids[3]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle value_new,
                        model_builder->GetTensor(input_ids[4]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle mask,
                        model_builder->GetTensor(input_ids[5]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle param,
                        model_builder->GetTensor(input_ids[6]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle attention_ref,
                        model_builder->GetTensor(output_ids[0]));

  const ::ml_drift::DataType output_dtype =
      attention_ref.tensor_desc.GetDataType();
  if (!::ml_drift::IsFloatType(output_dtype)) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update expects a float attention output.");
  }

  const bool quant_k =
      key_cache.tensor_desc.GetDataType() == ::ml_drift::DataType::kUint8;
  const bool quant_v =
      value_cache.tensor_desc.GetDataType() == ::ml_drift::DataType::kUint8;
  if (quant_k != quant_v) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update expects both or none of the KV caches "
        "to be quantized.");
  }
  const bool quant_kv = quant_k && quant_v;
  if (!quant_kv &&
      (!::ml_drift::IsFloatType(key_cache.tensor_desc.GetDataType()) ||
       key_cache.tensor_desc.GetDataType() !=
           value_cache.tensor_desc.GetDataType())) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update expects matching float KV caches when "
        "not quantized.");
  }
  const int kv_quant_bits =
      quant_kv ? (attr.kv_quant_bits == 0 ? 8 : attr.kv_quant_bits)
               : attr.kv_quant_bits;
  if ((!quant_kv && kv_quant_bits != 0) ||
      (quant_kv && kv_quant_bits != 4 && kv_quant_bits != 8)) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update has invalid kv_quant_bits.");
  }
  if (quant_kv &&
      (!attr.scale_k.has_value() || *attr.scale_k <= 0.0f ||
       !attr.scale_v.has_value() || *attr.scale_v <= 0.0f)) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update expects positive scale_k and scale_v "
        "for quantized KV caches.");
  }

  ::ml_drift::DataType data_type = output_dtype;
  if (!quant_kv) {
    data_type = key_cache.tensor_desc.GetDataType();
  } else if (::ml_drift::IsFloatType(query.tensor_desc.GetDataType())) {
    data_type = query.tensor_desc.GetDataType();
  } else if (::ml_drift::IsFloatType(key_new.tensor_desc.GetDataType())) {
    data_type = key_new.tensor_desc.GetDataType();
  } else if (::ml_drift::IsFloatType(value_new.tensor_desc.GetDataType())) {
    data_type = value_new.tensor_desc.GetDataType();
  }

  auto normalize_input = [&](TensorHandle& handle,
                             const std::optional<float>& scale,
                             const char* name) -> absl::Status {
    if (handle.tensor_desc.GetDataType() == ::ml_drift::DataType::kInt8) {
      if (!scale.has_value() || *scale <= 0.0f) {
        return absl::InvalidArgumentError(absl::StrCat(
            "odml.fused_sdpa_cache_update expects a positive scale for INT8 ",
            name, "."));
      }
      handle = DequantizeInt8Tensor(model_builder, handle, *scale, data_type);
      return absl::OkStatus();
    }
    if (scale.has_value()) {
      return absl::InvalidArgumentError(
          absl::StrCat("odml.fused_sdpa_cache_update received INT8 scale for "
                       "non-INT8 ",
                       name, "."));
    }
    if (!::ml_drift::IsFloatType(handle.tensor_desc.GetDataType())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "odml.fused_sdpa_cache_update expects float or INT8 ", name, "."));
    }
    if (handle.tensor_desc.GetDataType() != data_type) {
      handle = model_builder->Cast(handle, data_type);
    }
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(normalize_input(query, attr.scale_q, "query"));
  ABSL_RETURN_IF_ERROR(
      normalize_input(key_new, attr.scale_key_new, "key_new"));
  ABSL_RETURN_IF_ERROR(
      normalize_input(value_new, attr.scale_value_new, "value_new"));

  const ::ml_drift::BHWC q_shape = query.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC kc_shape = key_cache.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC vc_shape = value_cache.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC kn_shape = key_new.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC vn_shape = value_new.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC mask_shape = mask.tensor_desc.GetBHWCShape();
  const int heads = kc_shape.h;
  const int cache_size = kc_shape.w;
  const int head_dim = q_shape.c;
  const int new_len = kn_shape.w;
  const int rows = q_shape.w;
  const int required_head_align = (kv_quant_bits == 4) ? 8 : 4;
  if (cache_size <= 0 || cache_size % 4 != 0 || head_dim <= 0 ||
      head_dim % required_head_align != 0) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update requires cache size divisible by 4 and "
        "head dim divisible by 4 (or 8 for INT4 KV caches).");
  }
  const int expected_packed_head_dim =
      (kv_quant_bits == 4) ? (head_dim / 2) : head_dim;
  if (q_shape != ::ml_drift::BHWC(1, heads, rows, head_dim) ||
      kc_shape !=
          ::ml_drift::BHWC(1, heads, cache_size, expected_packed_head_dim) ||
      vc_shape !=
          ::ml_drift::BHWC(1, heads, expected_packed_head_dim, cache_size) ||
      kn_shape != ::ml_drift::BHWC(1, heads, new_len, head_dim) ||
      vn_shape != ::ml_drift::BHWC(1, heads, head_dim, new_len) ||
      mask_shape != ::ml_drift::BHWC(1, 1, new_len, cache_size + new_len) ||
      new_len <= 0 || rows % new_len != 0) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update has inconsistent input shapes.");
  }
  // The caches are read as external FullyConnected weights and written with
  // WriteLinear in their packed layouts, both of which need linear buffers.
  if (key_cache.tensor_desc.GetStorageType() !=
          ::ml_drift::TensorStorageType::kBuffer ||
      value_cache.tensor_desc.GetStorageType() !=
          ::ml_drift::TensorStorageType::kBuffer) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update requires BUFFER storage for the KV "
        "cache.");
  }

  ::ml_drift::BatchedMatMulAttributes bmm_attr;
  bmm_attr.transpose_left = false;
  bmm_attr.transpose_right = true;

  TensorHandle k_cache_for_fc = key_cache;
  TensorHandle v_cache_for_fc = value_cache;
  if (kv_quant_bits == 4) {
    TensorHandle unpacked_k = model_builder->AddTensor(
        1, heads, cache_size, head_dim, ::ml_drift::TensorStorageType::kBuffer,
        ::ml_drift::DataType::kUint8);
    TensorHandle unpacked_v = model_builder->AddTensor(
        1, heads, head_dim, cache_size, ::ml_drift::TensorStorageType::kBuffer,
        ::ml_drift::DataType::kUint8);
    model_builder->AddGpuOperation(
        {key_cache, value_cache, param}, {unpacked_k, unpacked_v},
        CreateUnpackRingInt4KvCacheOp(
            key_cache.tensor_desc, value_cache.tensor_desc, param.tensor_desc,
            unpacked_k.tensor_desc, unpacked_v.tensor_desc, heads, cache_size,
            head_dim),
        "fused_sdpa_unpack_int4_cache");
    k_cache_for_fc = unpacked_k;
    v_cache_for_fc = unpacked_v;
  }

  TensorHandle scale_k_handle;
  TensorHandle scale_v_handle;
  if (quant_kv) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::kFloat32>
        scale_k_cpu, scale_v_cpu;
    scale_k_cpu.shape = ::ml_drift::Linear(cache_size);
    scale_k_cpu.data.assign(cache_size, *attr.scale_k);
    scale_v_cpu.shape = ::ml_drift::Linear(head_dim);
    scale_v_cpu.data.assign(head_dim, *attr.scale_v);
    scale_k_handle = model_builder->AddConstantTensor(scale_k_cpu, data_type);
    scale_v_handle = model_builder->AddConstantTensor(scale_v_cpu, data_type);
  }

  // 1. Logits against the pre-write cache: [1, H, R, D] x [W, D]^T per head,
  //    only for the slots below the runtime bound (see PastEndSliceCode).
  ::ml_drift::ConvRuntimeCheckDesc past_logits_check;
  past_logits_check.dst_end_ch_index = kStartParamIndex;
  const ::ml_drift::OHWI k_weights_shape(cache_size, heads, 1, head_dim);
  TensorHandle logits_past;
  if (quant_kv) {
    ::ml_drift::WeightsDescription k_weights_desc =
        ::ml_drift::GetFullyConnectedWeightsDesc(::ml_drift::DataType::kUint8,
                                                 k_weights_shape);
    k_weights_desc.layout = ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;
    const ::ml_drift::GpuModelBuilder::Weights k_weights =
        ::ml_drift::CreateExternalWeights(
            k_cache_for_fc, k_weights_desc, k_weights_shape,
            ::ml_drift::OHWI(cache_size, 1, 1, 1), &scale_k_handle);
    logits_past = model_builder->FullyConnectedInt8ExternalWeights(
        query, k_weights, /*biases=*/nullptr, /*src_exp=*/nullptr,
        past_logits_check, &param);
  } else {
    ::ml_drift::WeightsDescription k_weights_desc =
        ::ml_drift::GetFullyConnectedWeightsDesc(data_type, k_weights_shape);
    k_weights_desc.layout = ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;
    const ::ml_drift::GpuModelBuilder::Weights k_weights =
        ::ml_drift::CreateExternalWeights(k_cache_for_fc, k_weights_desc,
                                          k_weights_shape);
    ABSL_ASSIGN_OR_RETURN(logits_past,
                          model_builder->FullyConnectedExternalWeights(
                              query, k_weights, /*biases=*/nullptr,
                              /*src_exp=*/nullptr, past_logits_check, &param));
  }

  // 2. Logits against the new tokens: [1, H, R, D] x [1, H, T, D]^T.
  //
  // For multi-token chunks the new keys and values are packed as weights (see
  // PackAsWeights) for the same FullyConnected kernels as the cache, and the
  // new-token block is not materialized as probabilities: its logits are
  // soft-capped and masked in the epilogue of the QK^T matmul, the softmax op
  // only reduces them to the row max / sum, and the P_new . V_new matmul
  // applies exp(v - max) / sum on load. The single-token (decode) graph keeps
  // the materialized probabilities and the BatchedMatMuls.
  const bool defer_new = new_len > 1 && new_len % 4 == 0;
  TensorHandle logits_new;
  ::ml_drift::GpuModelBuilder::Weights v_new_weights;
  TensorHandle k_new_attn = key_new;
  TensorHandle v_new_attn = value_new;
  if (defer_new) {
    ABSL_ASSIGN_OR_RETURN(
        const ::ml_drift::GpuModelBuilder::Weights k_new_weights,
        PackAsWeights(model_builder, key_new, data_type, attr.scale_k,
                      kv_quant_bits));
    ABSL_ASSIGN_OR_RETURN(
        v_new_weights, PackAsWeights(model_builder, value_new, data_type,
                                     attr.scale_v, kv_quant_bits));
    ABSL_ASSIGN_OR_RETURN(
        logits_new,
        model_builder->FullyConnectedExternalWeights(query, k_new_weights));
    if (attr.softcap.has_value() && *attr.softcap > 0.0f) {
      const float cap = *attr.softcap;
      logits_new = model_builder->Multiplication(logits_new, 1.0f / cap);
      logits_new = model_builder->Elementwise(logits_new,
                                              ::ml_drift::OperationType::kTanh);
      logits_new = model_builder->Multiplication(logits_new, cap);
    }
    logits_new =
        MaskNewLogits(model_builder, logits_new, mask, cache_size, new_len);
  } else {
    if (quant_kv) {
      k_new_attn = FakeQuantizeToCacheGrid(model_builder, key_new,
                                           *attr.scale_k, kv_quant_bits);
      v_new_attn = FakeQuantizeToCacheGrid(model_builder, value_new,
                                           *attr.scale_v, kv_quant_bits);
    }
    ABSL_ASSIGN_OR_RETURN(
        logits_new, model_builder->BatchedMatMul(query, k_new_attn, bmm_attr));
  }

  // 3. Soft-capped, masked softmax over all W + T keys: the cache block as
  //    probabilities, the new-token block as probabilities or statistics.
  TensorHandle probs_past = model_builder->AddTensor(
      ::ml_drift::BHWC(1, heads, rows, cache_size), data_type);
  TensorHandle dst_new = model_builder->AddTensor(
      ::ml_drift::BHWC(1, heads, rows, defer_new ? 4 : new_len), data_type);
  model_builder->AddGpuOperation(
      {logits_past, logits_new, mask, param}, {probs_past, dst_new},
      CreateMaskedSoftmaxOp(logits_past.tensor_desc, logits_new.tensor_desc,
                            mask.tensor_desc, param.tensor_desc,
                            probs_past.tensor_desc, dst_new.tensor_desc,
                            cache_size, new_len, attr, defer_new),
      "fused_sdpa_masked_softmax");

  // 4. out = P_new . V_new + P_past . V_cache. The new-token product is
  //    computed first so that the Add can be linked into the FullyConnected.
  TensorHandle out_new;
  if (defer_new) {
    ABSL_ASSIGN_OR_RETURN(out_new,
                          model_builder->FullyConnectedExternalWeights(
                              logits_new, v_new_weights, /*biases=*/nullptr,
                              /*src_exp=*/&dst_new));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        out_new, model_builder->BatchedMatMul(dst_new, v_new_attn, bmm_attr));
  }
  const ::ml_drift::OHWI v_weights_shape(head_dim, heads, 1, cache_size);
  ::ml_drift::ConvRuntimeCheckDesc past_values_check;
  past_values_check.src_end_ch_index = kStartParamIndex;
  TensorHandle out_past;
  if (quant_kv) {
    ::ml_drift::WeightsDescription v_weights_desc =
        ::ml_drift::GetFullyConnectedWeightsDesc(::ml_drift::DataType::kUint8,
                                                 v_weights_shape);
    v_weights_desc.layout = ::ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
    const ::ml_drift::GpuModelBuilder::Weights v_weights =
        ::ml_drift::CreateExternalWeights(
            v_cache_for_fc, v_weights_desc, v_weights_shape,
            ::ml_drift::OHWI(head_dim, 1, 1, 1), &scale_v_handle);
    out_past = model_builder->FullyConnectedInt8ExternalWeights(
        probs_past, v_weights, /*biases=*/nullptr, /*src_exp=*/nullptr,
        past_values_check, &param);
  } else {
    ::ml_drift::WeightsDescription v_weights_desc =
        ::ml_drift::GetFullyConnectedWeightsDesc(data_type, v_weights_shape);
    v_weights_desc.layout = ::ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
    const ::ml_drift::GpuModelBuilder::Weights v_weights =
        ::ml_drift::CreateExternalWeights(v_cache_for_fc, v_weights_desc,
                                          v_weights_shape);
    ABSL_ASSIGN_OR_RETURN(out_past,
                          model_builder->FullyConnectedExternalWeights(
                              probs_past, v_weights, /*biases=*/nullptr,
                              /*src_exp=*/nullptr, past_values_check, &param));
  }
  TensorHandle attention = model_builder->Add(out_past, out_new);

  const ::ml_drift::BHWC out_shape = attention_ref.tensor_desc.GetBHWCShape();
  if (attention.tensor_desc.GetBHWCShape() != out_shape) {
    attention = model_builder->Reshape(attention, out_shape);
  }
  // Dequantized INT8 inputs inside a mixed-precision partition are FLOAT32 on
  // the GPU, whereas the attention output tensor can be FLOAT16.
  if (attention.tensor_desc.GetDataType() != output_dtype) {
    attention = model_builder->Cast(attention, output_dtype);
  }
  ABSL_RETURN_IF_ERROR(
      model_builder->UpdateOutputTensor(attention, output_ids[0]));
  // The output count decides whether the ring buffer is written.
  if (output_ids.size() == 1) {
    return absl::OkStatus();
  }

  // 5. Ring buffer write, strictly after the attention above.
  ABSL_ASSIGN_OR_RETURN(TensorHandle key_cache_ref,
                        model_builder->GetTensor(output_ids[1]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle value_cache_ref,
                        model_builder->GetTensor(output_ids[2]));
  if (key_cache_ref.tensor_desc.GetStorageType() !=
          ::ml_drift::TensorStorageType::kBuffer ||
      value_cache_ref.tensor_desc.GetStorageType() !=
          ::ml_drift::TensorStorageType::kBuffer) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update requires BUFFER storage for the updated "
        "KV cache.");
  }
  if (key_cache_ref.tensor_desc.GetBHWCShape() != kc_shape ||
      value_cache_ref.tensor_desc.GetBHWCShape() != vc_shape ||
      key_cache_ref.tensor_desc.GetDataType() !=
          key_cache.tensor_desc.GetDataType() ||
      value_cache_ref.tensor_desc.GetDataType() !=
          value_cache.tensor_desc.GetDataType()) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update cache output descriptors must match "
        "inputs.");
  }
  ABSL_ASSIGN_OR_RETURN(TensorHandle attention_out,
                        model_builder->GetTensor(output_ids[0]));
  TensorHandle key_cache_out =
      model_builder->AddTensor(key_cache_ref.tensor_desc);
  TensorHandle value_cache_out =
      model_builder->AddTensor(value_cache_ref.tensor_desc);
  model_builder->AddGpuOperation(
      {key_new, value_new, param, attention_out},
      {key_cache_out, value_cache_out},
      CreateRingCacheWriteOp(
          key_new.tensor_desc, value_new.tensor_desc, param.tensor_desc,
          attention_out.tensor_desc, key_cache_out.tensor_desc,
          value_cache_out.tensor_desc, heads, cache_size, head_dim, new_len,
          kv_quant_bits, attr.scale_k.value_or(1.0f),
          attr.scale_v.value_or(1.0f)),
      "fused_sdpa_ring_cache_write");
  return model_builder->UpdateOutputTensors({key_cache_out, value_cache_out},
                                            {output_ids[1], output_ids[2]});
}

absl::Status CreateFusedSdpaCacheUpdateFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* attr =
      std::any_cast<FusedSdpaCacheUpdateAttributes>(&node.operation.attributes);
  if (attr == nullptr) {
    return absl::InvalidArgumentError(
        "Missing odml.fused_sdpa_cache_update attributes.");
  }
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* output : outputs) output_ids.push_back(output->id);
  return BuildSdpaCacheUpdateGpuGraph(input_ids, output_ids, *attr,
                                      model_builder);
}

absl::Status CreateFusedSdpaCacheUpdateFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* attr = std::any_cast<FusedSdpaCacheUpdateAttributes>(&node.attr);
  if (attr == nullptr) {
    return absl::InvalidArgumentError(
        "Missing odml.fused_sdpa_cache_update attributes.");
  }
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* output : outputs) output_ids.push_back(output->id);
  return BuildSdpaCacheUpdateGpuGraph(input_ids, output_ids, *attr,
                                      model_builder);
}

}  // namespace litert::ml_drift
