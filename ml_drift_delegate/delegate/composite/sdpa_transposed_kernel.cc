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

#include "ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.h"

#include <algorithm>
#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

namespace litert::ml_drift {

namespace {

// Number of SIMD groups (warps/threadgroups) cooperating on Flash-Decode per
// head.
// TODO(b/553029558): Tune kNumSimdGroups dynamically based on gpu_info.
constexpr int kNumSimdGroups = 16;

// TODO(b/552147487): Benchmark the kernel on Nividia GPUs.
class FusedFlashDecodeSdpaOp : public ::ml_drift::GPUOperation {
 public:
  explicit FusedFlashDecodeSdpaOp(int slices_per_head = 32)
      : slices_per_head_(slices_per_head) {}

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(
        src_[0]->Width(), src_[0]->Height() * slices_per_head_, kNumSimdGroups);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {::ml_drift::int3(1, slices_per_head_, kNumSimdGroups)};
  }
  FusedFlashDecodeSdpaOp(FusedFlashDecodeSdpaOp&&) = default;
  FusedFlashDecodeSdpaOp& operator=(FusedFlashDecodeSdpaOp&&) = default;
  FusedFlashDecodeSdpaOp(const FusedFlashDecodeSdpaOp&) = delete;
  FusedFlashDecodeSdpaOp& operator=(const FusedFlashDecodeSdpaOp&) = delete;

 private:
  int slices_per_head_ = 32;
};

std::unique_ptr<::ml_drift::GPUOperation> CreateFusedFlashDecodeSdpa(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::TensorDescriptor& q_desc,
    const ::ml_drift::TensorDescriptor& k_desc,
    const ::ml_drift::TensorDescriptor& v_desc,
    const ::ml_drift::TensorDescriptor* mask_desc,
    const ::ml_drift::TensorDescriptor* param_desc,
    const ::ml_drift::TensorDescriptor& dst_desc,
    const SdpaTransposedAttributes& attr, bool is_flattened_dst = false) {
  // Each float4/half4 vector slice consists of 4 channels.
  // slices represents the number of vector slices per head (e.g. 128 / 4 = 32).
  int slices = q_desc.GetBHWCShape().c / 4;
  FusedFlashDecodeSdpaOp custom_op(slices);
  // V-cache memory layout: [num_heads, num_chunks, slices, 4].
  // Each chunk across the head dimension has stride `slices * 4`.
  int v_stride_s = slices * 4;
  int v_stride_2s = slices * 8;
  int v_stride_3s = slices * 12;
  int v_stride_4s = slices * 16;
  int k_o_slices = (k_desc.GetBHWCShape().w + 3) / 4;
  int k_stride_head = slices * k_o_slices * 4;
  int k_stride_slice = k_o_slices * 4;
  int v_stride_head = k_o_slices * v_stride_s;
  const int q_heads = q_desc.GetBHWCShape().h;
  const int kv_heads = k_desc.GetBHWCShape().h;
  const int gqa_ratio =
      (kv_heads > 0 && q_heads >= kv_heads && (q_heads % kv_heads == 0))
          ? (q_heads / kv_heads)
          : 1;

  custom_op.work_group_size_ = ::ml_drift::int3(1, 32, kNumSimdGroups);
  custom_op.args_.AddInt("cache_size", k_desc.GetBHWCShape().w);
  custom_op.args_.AddInt("slices", slices);

  custom_op.AddSrcTensor("q", q_desc);
  custom_op.AddSrcTensor("k", k_desc);
  custom_op.AddSrcTensor("v", v_desc);

  bool has_mask = (mask_desc != nullptr);
  if (has_mask) {
    bool is_bool_mask =
        (mask_desc->GetDataType() == ::ml_drift::DataType::BOOL);
    custom_op.args_.AddInt("is_bool_mask", is_bool_mask ? 1 : 0);
    custom_op.AddSrcTensor("mask", *mask_desc);
  }

  bool has_param = (param_desc != nullptr &&
                    attr.runtime_check.src_end_ch_index.has_value());
  if (has_param) {
    custom_op.args_.AddInt("src_end_ch_index",
                          *attr.runtime_check.src_end_ch_index);
    custom_op.AddSrcTensor("params", *param_desc);
  }

  bool has_softcap = (attr.softcap.has_value() && *attr.softcap > 0.0f);
  if (has_softcap) {
    custom_op.args_.AddFloat("softcap", *attr.softcap);
  }

  custom_op.AddDstTensor("dst", dst_desc);

  // Fused Flash-Decode SDPA kernel for single-token generation (seq_len == 1).
  //
  // Execution model:
  // - 16 SIMD groups per threadgroup (32 threads per SIMD group = 512 threads).
  //   Launch shape: (1, 32, 16).
  // - Grid: X = sequence index (1), Y = query head index.
  // - Within each SIMD group, the 32 lanes compute channel-parallel dot
  // products
  //   (dot(q_slice, k) over head_dim / 4 = 32 slices).
  // - Across the 16 SIMD groups, the KV cache sequence length is partitioned
  // into
  //   chunks of 16 keys (4 vector loads of 4 keys). Each SIMD group maintains a
  //   local online softmax (m_prev = running max, l_prev = running exp sum,
  //   v_acc = accumulator).
  // - Cross-SIMD reduction: SIMD group 0 performs a tree reduction across s_m,
  // s_l,
  //   and s_acc stored in threadgroup memory to produce the final normalized
  //   output.
  std::string op_code = absl::StrCat(R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGroupId<1>();
  int simd_id = ucl::GetLocalId<2>();
  int tid = ucl::GetLocalId<1>();

  threadgroup float s_m[16];
  threadgroup float s_l[16];
  threadgroup float s_w[16];
  threadgroup half4 s_acc[)",
                                      slices, R"(][16];

  if (simd_id == 0 && tid < 16) {
    s_m[tid] = -10000.0f;
    s_l[tid] = 0.0f;
  }

  int active_tokens = args.cache_size;
)");

  if (has_param) {
    op_code += R"(
  int param_slice = args.src_end_ch_index / 4;
  int param_comp = args.src_end_ch_index % 4;
  float4 p_vec = ucl::Convert<float4>(args.params.Read(0, 0, param_slice, 0));
  float p_raw = (param_comp == 0) ? p_vec.x : ((param_comp == 1) ? p_vec.y : ((param_comp == 2) ? p_vec.z : p_vec.w));
  int param_val = (int)p_raw;
  if (param_val > 0 && param_val <= args.cache_size) {
    active_tokens = param_val;
  }
)";
    if (!has_mask && attr.is_causal) {
      op_code += R"(
  float4 p_start_vec = ucl::Convert<float4>(args.params.Read(0, 0, 0, 0));
  int start_val = (int)p_start_vec.x;
  int q_start = (start_val > 0 && start_val < active_tokens)
                    ? start_val
                    : max(0, active_tokens - args.q.Width());
  active_tokens = min(active_tokens, q_start + X + 1);
)";
    }
  }

  absl::StrAppend(&op_code, R"(
  int total_chunks = (active_tokens + 3) / 4;
  int chunks_per_simd = (total_chunks + 15) / 16;
  int chunk_start = simd_id * chunks_per_simd;
  int chunk_end = min(total_chunks, chunk_start + chunks_per_simd);
  int safe_chunk_end = max(chunk_start, min(chunk_end, (active_tokens / 16) * 4));

  // Note: This Flash-Decode SDPA kernel is optimized for float16 / half precision.
  half4 q_slice = (tid < args.slices) ? ucl::Convert<half4>(args.q.Read(X, Y, tid)) : half4(0.0h);
  half m_prev = -10000.0h;
  half l_prev = 0.0h;
  // Note: half4 output accumulator for maximum register efficiency on mobile GPUs.
  half4 out_acc = half4(0.0h);
  half inv_ln2 = 1.4426950408889634h;

  int kv_head = )",
                  (gqa_ratio > 1 ? absl::StrCat("Y / ", gqa_ratio) : "Y"), R"(;
  int k_base_head = kv_head * )",
                  k_stride_head, R"( + tid * )", k_stride_slice, R"(;
  int v_base_head = kv_head * )",
                  v_stride_head, R"( + tid * 4;

  int chunk = chunk_start;
  int k_idx = k_base_head + chunk * 4;
  int v_idx = v_base_head + chunk * )",
                  v_stride_s, R"(;

  for (; chunk + 3 < safe_chunk_end; chunk += 4) {
    half4 k0 = ucl::Convert<half4>(args.k.Read(k_idx + 0));
    half4 k1 = ucl::Convert<half4>(args.k.Read(k_idx + 1));
    half4 k2 = ucl::Convert<half4>(args.k.Read(k_idx + 2));
    half4 k3 = ucl::Convert<half4>(args.k.Read(k_idx + 3));
    half4 d0 = simd_sum(half4(dot(q_slice, k0), dot(q_slice, k1), dot(q_slice, k2), dot(q_slice, k3)));

    half4 k4 = ucl::Convert<half4>(args.k.Read(k_idx + 4));
    half4 k5 = ucl::Convert<half4>(args.k.Read(k_idx + 5));
    half4 k6 = ucl::Convert<half4>(args.k.Read(k_idx + 6));
    half4 k7 = ucl::Convert<half4>(args.k.Read(k_idx + 7));
    half4 d1 = simd_sum(half4(dot(q_slice, k4), dot(q_slice, k5), dot(q_slice, k6), dot(q_slice, k7)));

    half4 k8 = ucl::Convert<half4>(args.k.Read(k_idx + 8));
    half4 k9 = ucl::Convert<half4>(args.k.Read(k_idx + 9));
    half4 k10 = ucl::Convert<half4>(args.k.Read(k_idx + 10));
    half4 k11 = ucl::Convert<half4>(args.k.Read(k_idx + 11));
    half4 d2 = simd_sum(half4(dot(q_slice, k8), dot(q_slice, k9), dot(q_slice, k10), dot(q_slice, k11)));

    half4 k12 = ucl::Convert<half4>(args.k.Read(k_idx + 12));
    half4 k13 = ucl::Convert<half4>(args.k.Read(k_idx + 13));
    half4 k14 = ucl::Convert<half4>(args.k.Read(k_idx + 14));
    half4 k15 = ucl::Convert<half4>(args.k.Read(k_idx + 15));
    half4 d3 = simd_sum(half4(dot(q_slice, k12), dot(q_slice, k13), dot(q_slice, k14), dot(q_slice, k15)));
)");

  if (has_softcap) {
    op_code += R"(
    d0 = (half4)args.softcap * tanh(d0 / (half4)args.softcap);
    d1 = (half4)args.softcap * tanh(d1 / (half4)args.softcap);
    d2 = (half4)args.softcap * tanh(d2 / (half4)args.softcap);
    d3 = (half4)args.softcap * tanh(d3 / (half4)args.softcap);
)";
  }

  absl::StrAppend(&op_code, R"(
    // Online softmax tracking:
    // m_prev: running maximum of dot-product scores (for numerical stability).
    // l_prev: running sum of exponentiated scores (normalization denominator).
    // alpha:  rescaling factor (exp2(m_prev - m_new)) for previous accumulators when a new max is found.
    // p0..p3: unnormalized attention probabilities (exp2(d - m_new)) for keys in this chunk.
    half4 m_c01 = max(max(d0, d1), max(d2, d3));
    half m_chunk = max(max(m_c01.x, m_c01.y), max(m_c01.z, m_c01.w));
    half m_new = max(m_prev, m_chunk);
    half alpha = exp2((m_prev - m_new) * inv_ln2);

    half4 p0 = exp2((d0 - (half4)m_new) * (half4)inv_ln2);
    half4 p1 = exp2((d1 - (half4)m_new) * (half4)inv_ln2);
    half4 p2 = exp2((d2 - (half4)m_new) * (half4)inv_ln2);
    half4 p3 = exp2((d3 - (half4)m_new) * (half4)inv_ln2);

    half4 p_sum01 = (p0 + p1) + (p2 + p3);
    half p_sum = (p_sum01.x + p_sum01.y) + (p_sum01.z + p_sum01.w);
    l_prev = fma(l_prev, alpha, p_sum);
    m_prev = m_new;

    half4 v0 = ucl::Convert<half4>(args.v.Read(v_idx + 0));
    half4 v1 = ucl::Convert<half4>(args.v.Read(v_idx + 1));
    half4 v2 = ucl::Convert<half4>(args.v.Read(v_idx + 2));
    half4 v3 = ucl::Convert<half4>(args.v.Read(v_idx + 3));
    half4 v_acc0 = fma((half4)p0.x, v0, fma((half4)p0.y, v1, fma((half4)p0.z, v2, (half4)p0.w * v3)));

    int v1_base = v_idx + )",
                  v_stride_s, R"(;
    half4 v4 = ucl::Convert<half4>(args.v.Read(v1_base + 0));
    half4 v5 = ucl::Convert<half4>(args.v.Read(v1_base + 1));
    half4 v6 = ucl::Convert<half4>(args.v.Read(v1_base + 2));
    half4 v7 = ucl::Convert<half4>(args.v.Read(v1_base + 3));
    half4 v_acc1 = fma((half4)p1.x, v4, fma((half4)p1.y, v5, fma((half4)p1.z, v6, (half4)p1.w * v7)));

    int v2_base = v_idx + )",
                  v_stride_2s, R"(;
    half4 v8 = ucl::Convert<half4>(args.v.Read(v2_base + 0));
    half4 v9 = ucl::Convert<half4>(args.v.Read(v2_base + 1));
    half4 v10 = ucl::Convert<half4>(args.v.Read(v2_base + 2));
    half4 v11 = ucl::Convert<half4>(args.v.Read(v2_base + 3));
    half4 v_acc2 = fma((half4)p2.x, v8, fma((half4)p2.y, v9, fma((half4)p2.z, v10, (half4)p2.w * v11)));

    int v3_base = v_idx + )",
                  v_stride_3s, R"(;
    half4 v12 = ucl::Convert<half4>(args.v.Read(v3_base + 0));
    half4 v13 = ucl::Convert<half4>(args.v.Read(v3_base + 1));
    half4 v14 = ucl::Convert<half4>(args.v.Read(v3_base + 2));
    half4 v15 = ucl::Convert<half4>(args.v.Read(v3_base + 3));
    half4 v_acc3 = fma((half4)p3.x, v12, fma((half4)p3.y, v13, fma((half4)p3.z, v14, (half4)p3.w * v15)));

    out_acc = fma(out_acc, (half4)alpha, (v_acc0 + v_acc1) + (v_acc2 + v_acc3));

    k_idx += 16;
    v_idx += )",
                  v_stride_4s, R"(;
  }
)");

  absl::StrAppend(&op_code, R"(
  for (; chunk < chunk_end; ++chunk) {
    half4 k0 = ucl::Convert<half4>(args.k.Read(k_idx + 0));
    half4 k1 = ucl::Convert<half4>(args.k.Read(k_idx + 1));
    half4 k2 = ucl::Convert<half4>(args.k.Read(k_idx + 2));
    half4 k3 = ucl::Convert<half4>(args.k.Read(k_idx + 3));

    half4 d = simd_sum(half4(dot(q_slice, k0), dot(q_slice, k1), dot(q_slice, k2), dot(q_slice, k3)));
)");

  if (has_softcap) {
    op_code += R"(
    d = (half4)args.softcap * tanh(d / (half4)args.softcap);
)";
  }

  if (has_mask) {
    op_code += R"(
    // Attention mask and cache boundary check: clamp masked-out or past-active-tokens
    // key scores to -10000.0h (-inf) so they contribute 0.0h to the softmax.
    half4 m_vec = ucl::Convert<half4>(args.mask.Read(X, 0, chunk));
    if (args.is_bool_mask) {
      if (m_vec.x < 0.5h || (chunk * 4 + 0) >= active_tokens) d.x = -10000.0h;
      if (m_vec.y < 0.5h || (chunk * 4 + 1) >= active_tokens) d.y = -10000.0h;
      if (m_vec.z < 0.5h || (chunk * 4 + 2) >= active_tokens) d.z = -10000.0h;
      if (m_vec.w < 0.5h || (chunk * 4 + 3) >= active_tokens) d.w = -10000.0h;
    } else {
      d += m_vec;
      if ((chunk * 4 + 0) >= active_tokens) d.x = -10000.0h;
      if ((chunk * 4 + 1) >= active_tokens) d.y = -10000.0h;
      if ((chunk * 4 + 2) >= active_tokens) d.z = -10000.0h;
      if ((chunk * 4 + 3) >= active_tokens) d.w = -10000.0h;
    }
)";
  } else {
    op_code += R"(
    // Cache boundary check: clamp keys beyond active_tokens to -10000.0h (-inf).
    if ((chunk * 4 + 0) >= active_tokens) d.x = -10000.0h;
    if ((chunk * 4 + 1) >= active_tokens) d.y = -10000.0h;
    if ((chunk * 4 + 2) >= active_tokens) d.z = -10000.0h;
    if ((chunk * 4 + 3) >= active_tokens) d.w = -10000.0h;
)";
  }

  absl::StrAppend(&op_code, R"(
    half m_new = max(m_prev, max(max(d.x, d.y), max(d.z, d.w)));
    half alpha = exp2((m_prev - m_new) * inv_ln2);
    half4 p = exp2((d - (half4)m_new) * (half4)inv_ln2);
    l_prev = fma(l_prev, alpha, (p.x + p.y) + (p.z + p.w));
    m_prev = m_new;

    half4 v0 = ucl::Convert<half4>(args.v.Read(v_idx + 0));
    half4 v1 = ucl::Convert<half4>(args.v.Read(v_idx + 1));
    half4 v2 = ucl::Convert<half4>(args.v.Read(v_idx + 2));
    half4 v3 = ucl::Convert<half4>(args.v.Read(v_idx + 3));
    out_acc = fma(out_acc, (half4)alpha,
                  fma((half4)p.x, v0,
                  fma((half4)p.y, v1,
                  fma((half4)p.z, v2,
                      (half4)p.w * v3))));
    k_idx += 4;
    v_idx += )",
                  v_stride_s, R"(;
  }

  if (tid == 0) {
    s_m[simd_id] = (float)m_prev;
    s_l[simd_id] = (float)l_prev;
  }
  s_acc[tid][simd_id] = out_acc;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    float m_val = (tid < 16) ? s_m[tid] : -10000.0f;
    float global_m = simd_max(m_val);

    float sc = (tid < 16 && s_l[tid] > 0.0f) ? exp2((s_m[tid] - global_m) * 1.44269504f) : 0.0f;
    float l_term = sc * ((tid < 16) ? s_l[tid] : 0.0f);
    float l_total = simd_sum(l_term);
    float inv_l = 1.0f / (l_total + 1e-10f);

    if (tid < 16) {
      s_w[tid] = sc * inv_l;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    half4 acc0 = fma((half4)s_w[0], s_acc[tid][0], (half4)s_w[1] * s_acc[tid][1]);
    half4 acc1 = fma((half4)s_w[2], s_acc[tid][2], (half4)s_w[3] * s_acc[tid][3]);
    half4 acc2 = fma((half4)s_w[4], s_acc[tid][4], (half4)s_w[5] * s_acc[tid][5]);
    half4 acc3 = fma((half4)s_w[6], s_acc[tid][6], (half4)s_w[7] * s_acc[tid][7]);
    half4 acc4 = fma((half4)s_w[8], s_acc[tid][8], (half4)s_w[9] * s_acc[tid][9]);
    half4 acc5 = fma((half4)s_w[10], s_acc[tid][10], (half4)s_w[11] * s_acc[tid][11]);
    half4 acc6 = fma((half4)s_w[12], s_acc[tid][12], (half4)s_w[13] * s_acc[tid][13]);
    half4 acc7 = fma((half4)s_w[14], s_acc[tid][14], (half4)s_w[15] * s_acc[tid][15]);

    half4 sum0 = (acc0 + acc1) + (acc2 + acc3);
    half4 sum1 = (acc4 + acc5) + (acc6 + acc7);
    half4 final_acc = sum0 + sum1;
)",
                  is_flattened_dst ? absl::StrFormat(R"(
    if (tid < args.slices) {
      int out_slice = Y * %d + tid;
      args.dst.Write(ucl::Convert<args.dst::type>(final_acc), X, 0, out_slice);
    }
)",
                                                     slices)
                                   : R"(
    if (tid < args.slices) {
      args.dst.Write(ucl::Convert<args.dst::type>(final_acc), X, Y, tid);
    }
)",
                  R"(
  }
}
)");

  custom_op.code_ = std::move(op_code);
  return std::make_unique<FusedFlashDecodeSdpaOp>(std::move(custom_op));
}

// Prefill tiling: Each threadgroup runs 4 SIMD groups (128 threads)
// and owns BQ=32 query rows
// (4 warps x 8 rows/warp) across GQA sibling query heads sharing the same KV
// head, stepping by BK=16 keys per iteration.
// Threadgroup memory is Q_smem (32x132 halfs = 8,448 B) + KV_smem (2,560 halfs
// = 5,120 B) = 13,568 B (< 16 KB), allowing 2 full threadgroups (8 warps = 256
// threads) per Apple GPU core while keeping live registers <= 48 per thread
// (zero register spills).
constexpr int kPrefillThreadsPerTg = 128;
constexpr int kPrefillTotalRows = 32;
constexpr int kPrefillKeyBlock = 16;

class FusedFlashAttentionPrefillOp : public ::ml_drift::GPUOperation {
 public:
  FusedFlashAttentionPrefillOp() = default;
  FusedFlashAttentionPrefillOp(int q_per_head, int heads_per_tg)
      : q_per_head_(q_per_head), heads_per_tg_(heads_per_tg) {}

  ::ml_drift::int3 GetGridSize() const override {
    const int num_q_tiles = (dst_[0]->Width() + q_per_head_ - 1) / q_per_head_;
    const int num_head_groups =
        (dst_[0]->Height() + heads_per_tg_ - 1) / heads_per_tg_;
    return ::ml_drift::int3(num_q_tiles,
                            num_head_groups * kPrefillThreadsPerTg, 1);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {::ml_drift::int3(1, kPrefillThreadsPerTg, 1)};
  }

  FusedFlashAttentionPrefillOp(FusedFlashAttentionPrefillOp&&) = default;
  FusedFlashAttentionPrefillOp& operator=(FusedFlashAttentionPrefillOp&&) =
      default;
  FusedFlashAttentionPrefillOp(const FusedFlashAttentionPrefillOp&) = delete;
  FusedFlashAttentionPrefillOp& operator=(const FusedFlashAttentionPrefillOp&) =
      delete;

 private:
  int q_per_head_ = kPrefillTotalRows;
  int heads_per_tg_ = 1;
};

std::unique_ptr<::ml_drift::GPUOperation> CreateFusedFlashAttentionPrefill(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::TensorDescriptor& q_desc,
    const ::ml_drift::TensorDescriptor& k_desc,
    const ::ml_drift::TensorDescriptor& v_desc,
    const ::ml_drift::TensorDescriptor* mask_desc,
    const ::ml_drift::TensorDescriptor* param_desc,
    const ::ml_drift::TensorDescriptor& dst_desc,
    const SdpaTransposedAttributes& attr) {
  int slices = dst_desc.GetBHWCShape().c / 4;
  int v_stride_s = slices * 4;
  int k_o_slices = (k_desc.GetBHWCShape().w + 3) / 4;
  int k_stride_head = slices * k_o_slices * 4;
  int k_stride_slice = k_o_slices * 4;
  int v_stride_head = k_o_slices * v_stride_s;
  const int q_heads = q_desc.GetBHWCShape().h;
  const int kv_heads = k_desc.GetBHWCShape().h;
  const int gqa_ratio =
      (kv_heads > 0 && q_heads >= kv_heads && (q_heads % kv_heads == 0))
          ? (q_heads / kv_heads)
          : 1;

  // Step A: Group GQA sibling heads and query tokens into the 4 warps (32 query
  // rows) of the threadgroup so all warps share a single cooperative K/V load.
  int heads_per_tg = 1;
  if (gqa_ratio % 4 == 0) {
    heads_per_tg = 4;
  } else if (gqa_ratio % 2 == 0) {
    heads_per_tg = 2;
  }
  const int q_per_head = kPrefillTotalRows / heads_per_tg;

  FusedFlashAttentionPrefillOp custom_op(q_per_head, heads_per_tg);
  custom_op.work_group_size_ = ::ml_drift::int3(1, kPrefillThreadsPerTg, 1);
  custom_op.args_.AddInt("cache_size", k_desc.GetBHWCShape().w);
  custom_op.args_.AddInt("slices", slices);

  custom_op.AddSrcTensor("q", q_desc);
  custom_op.AddSrcTensor("k", k_desc);
  custom_op.AddSrcTensor("v", v_desc);

  bool has_mask = (mask_desc != nullptr);
  if (has_mask) {
    bool is_bool_mask =
        (mask_desc->GetDataType() == ::ml_drift::DataType::BOOL);
    custom_op.args_.AddInt("is_bool_mask", is_bool_mask ? 1 : 0);
    custom_op.AddSrcTensor("mask", *mask_desc);
  }

  bool has_param = (param_desc != nullptr &&
                    attr.runtime_check.src_end_ch_index.has_value());
  if (has_param) {
    custom_op.args_.AddInt("src_end_ch_index",
                          *attr.runtime_check.src_end_ch_index);
    custom_op.AddSrcTensor("params", *param_desc);
  }

  bool has_softcap = (attr.softcap.has_value() && *attr.softcap > 0.0f);
  if (has_softcap) {
    custom_op.args_.AddFloat("softcap", *attr.softcap);
  }

  custom_op.AddDstTensor("dst", dst_desc);

  const int td = (slices + 1) / 2;  // 8-channel MMA tiles along head_dim
  const int bd_padded = td * 8;
  const int ldq = bd_padded + 4;    // 132 halfs (8,448 B for 32 rows)
  const int ldk = 20;               // 16 keys + 4 padding halfs
  const int ldv = bd_padded + 4;    // 132 halfs
  const int q_smem_size = kPrefillTotalRows * ldq;
  const int kv_smem_size = std::max(bd_padded * ldk, kPrefillKeyBlock * ldv);
  const int k_load_iters = (slices + 7) / 8;

  std::string op_code;
  absl::StrAppend(&op_code, R"(
#include <metal_simdgroup_matrix>

inline float2 mma_f16_f32_8x8(half2 A, half2 B, float2 C) {
  metal::simdgroup_matrix<float, 8, 8> D_mat;
  metal::simdgroup_matrix<half, 8, 8> A_mat;
  metal::simdgroup_matrix<half, 8, 8> B_mat;
  metal::simdgroup_matrix<float, 8, 8> C_mat;
  reinterpret_cast<thread half2&>(A_mat.thread_elements()) = A;
  reinterpret_cast<thread half2&>(B_mat.thread_elements()) = B;
  reinterpret_cast<thread float2&>(C_mat.thread_elements()) = C;
  metal::simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
  return reinterpret_cast<thread float2&>(D_mat.thread_elements());
}

MAIN_FUNCTION($0) {
  int tile_x = ucl::GetGlobalId<0>();
  int tg_y = ucl::GetGroupId<1>();
  int tid = ucl::GetLocalId<1>();
  int sg_id = tid >> 5;
  int lane_id = tid & 31;

  int Y0 = tg_y * )",
                  heads_per_tg, R"(;
  int X0 = tile_x * )",
                  q_per_head, R"(;
  int Y_warp = Y0 + (sg_id % )",
                  heads_per_tg, R"();
  int X_warp = X0 + (sg_id / )",
                  heads_per_tg, R"() * 8;

  int dst_w = args.dst.Width();
  int dst_h = args.dst.Height();
  if (X0 >= dst_w || Y0 >= dst_h) {
    return;
  }

  int active_tokens = args.cache_size;
  int q_start = 0;
)");

  if (has_param) {
    absl::StrAppend(&op_code, R"(
  int param_slice = args.src_end_ch_index / 4;
  int param_comp = args.src_end_ch_index % 4;
  float4 p_vec = ucl::Convert<float4>(args.params.Read(0, 0, param_slice, 0));
  float p_raw = (param_comp == 0) ? p_vec.x : ((param_comp == 1) ? p_vec.y : ((param_comp == 2) ? p_vec.z : p_vec.w));
  int param_val = (int)p_raw;
  if (param_val > 0 && param_val <= args.cache_size) {
    active_tokens = param_val;
  }
  float4 p_start_vec = ucl::Convert<float4>(args.params.Read(0, 0, 0, 0));
  int start_val = (int)p_start_vec.x;
  if (start_val > 0 && start_val < active_tokens) {
    q_start = start_val;
  }
)");
  }

  // When a fixed-width prefill signature (dst_w, e.g. 1024) processes a
  // partial chunk (active_tokens - q_start < dst_w, e.g. 128 tokens) and the
  // BOOL mask is pruned, query columns X >= valid_w are inactive padding.
  // Zero-fill those padded output columns and exit early for padded tiles.
  absl::StrAppend(&op_code, R"(
  int valid_w = min(dst_w, active_tokens - q_start);
  if (X0 >= valid_w) {
    if (lane_id < args.slices && Y_warp < dst_h) {
      for (int r = 0; r < 8; ++r) {
        if (X_warp + r < dst_w) {
          args.dst.Write(ucl::Convert<args.dst::type>(float4(0.0f)),
                         X_warp + r, Y_warp, lane_id);
        }
      }
    }
    return;
  }
)");

  const bool is_causal = attr.is_causal;
  if (is_causal) {
    absl::StrAppend(&op_code, "\n  int max_tokens = min(X0 + ", q_per_head,
                    " + q_start, active_tokens);\n");
  } else {
    absl::StrAppend(&op_code, "\n  int max_tokens = active_tokens;\n");
  }
  absl::StrAppend(&op_code, R"(  float inv_ln2 = 1.4426950408889634f;

  // Threadgroup memory: 8,448 B (Q_smem) + 5,120 B (KV_smem) = 13,568 B.
  threadgroup half Q_smem[)",
                  q_smem_size, R"(];
  threadgroup half KV_smem[)",
                  kv_smem_size, R"(];
  threadgroup half* K_smem = KV_smem;
  threadgroup half* V_smem = KV_smem;

  // Stage scaled Q into Q_smem[sg_id * 8 + r][lane_id * 4].
  for (int r = 0; r < 8; ++r) {
    int q_x = X_warp + r;
    int q_row_off = (sg_id * 8 + r) * )",
                  ldq, R"(;
    if (lane_id < )",
                  slices, R"() {
      bool q_ok = (q_x < valid_w && Y_warp < dst_h);
      half4 q_v = q_ok ? ucl::Convert<half4>(
                             ucl::Convert<float4>(
                                 args.q.Read(q_x, Y_warp, lane_id)) *
                             inv_ln2)
                       : half4(0.0h);
      *reinterpret_cast<threadgroup half4*>(
          &Q_smem[q_row_off + lane_id * 4]) = q_v;
    }
)");
  if (bd_padded > slices * 4) {
    absl::StrAppend(&op_code, "    if (lane_id == 0) {\n");
    for (int c = slices * 4; c < bd_padded; ++c) {
      absl::StrAppend(&op_code, "      Q_smem[q_row_off + ", c, "] = 0.0h;\n");
    }
    absl::StrAppend(&op_code, "    }\n");
  }
  absl::StrAppend(&op_code, R"(  }

  // Apple simdgroup_matrix<T, 8, 8> lane-to-(row, col) fragment mapping.
  int qid = lane_id >> 2;
  int sm = (qid & 4) + ((lane_id >> 1) & 3);
  int sn = ((qid & 2) << 1) + ((lane_id & 1) << 1);
  int q_smem_base = (sg_id * 8 + sm) * )",
                  ldq, R"( + sn;

  int X_row = X_warp + sm;
  int P_row = X_row + q_start;
  bool row_valid = (X_row < valid_w && Y_warp < dst_h);

  float m_prev = -10000.0f;
  float l_prev = 0.0f;
)");
  for (int id = 0; id < td; ++id) {
    absl::StrAppend(&op_code, "  float2 o_frag", id, " = float2(0.0f);\n");
  }

  // KV buffer base addressing (Grouped-Query Attention).
  // K layout (WeightsLayout::kOSpatialIOGroupO4I4):
  //   [kv_heads, slices, k_o_slices, 4_keys, 4_channels] where each half4
  //   holds 4 channels (4*c_slice..4*c_slice+3) for one key at linear index
  //   k_head_base + c_slice * k_stride_slice + key_idx.
  // V layout (WeightsLayout::kOSpatialIOGroupI4O4):
  //   [kv_heads, k_o_slices, 4_keys, slices, 4_channels] where each half4
  //   holds 4 channels (4*c_slice..4*c_slice+3) for key (g*4 + t) at linear
  //   index v_head_base + (g*4 + v_k_sub) * slices + v_c_slice.
  absl::StrAppend(&op_code, "\n  int kv_head = ",
                  (gqa_ratio > 1 ? absl::StrCat("Y0 / ", gqa_ratio) : "Y0"),
                  ";\n  int k_head_base = kv_head * ", k_stride_head,
                  ";\n  int v_head_base = kv_head * ", v_stride_head,
                  ";\n  int k_s = tid & 15;\n  int k_cg = tid >> 4;\n"
                  "  int v_k_sub = tid & 3;\n  int v_c_slice = tid >> 2;\n"
                  "  int v_c_col = v_c_slice * 4;\n\n");

  auto emit_key_block_body = [&](bool is_interior) {
    absl::StrAppend(&op_code,
                    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    // 1. Coalesced K load (16 contiguous half4 keys across k_s = tid & 15,
    // 8 slices per iteration across k_cg = tid >> 4).
    if (is_interior && (slices % 8 == 0)) {
      absl::StrAppend(
          &op_code, "    #pragma unroll\n    for (int r = 0; r < ",
          k_load_iters, "; ++r) {\n      int c_sl = r * 8 + k_cg;\n",
          "      half4 kv = ucl::Convert<half4>(args.k.Read(k_head_base + "
          "c_sl * ",
          k_stride_slice, " + key_base + k_s));\n",
          "      int k_r = c_sl * 4;\n", "      K_smem[(k_r + 0) * ", ldk,
          " + k_s] = kv.x;\n", "      K_smem[(k_r + 1) * ", ldk,
          " + k_s] = kv.y;\n", "      K_smem[(k_r + 2) * ", ldk,
          " + k_s] = kv.z;\n", "      K_smem[(k_r + 3) * ", ldk,
          " + k_s] = kv.w;\n    }\n");
    } else {
      absl::StrAppend(
          &op_code, "    #pragma unroll\n    for (int r = 0; r < ",
          k_load_iters, "; ++r) {\n      int c_sl = r * 8 + k_cg;\n",
          "      bool k_ok = (c_sl < ", slices,
          " && (key_base + k_s) < active_tokens);\n",
          "      half4 kv = k_ok ? "
          "ucl::Convert<half4>(args.k.Read(k_head_base + c_sl * ",
          k_stride_slice, " + key_base + k_s)) : half4(0.0h);\n",
          "      if (c_sl < ", slices, ") {\n",
          "        int k_r = c_sl * 4;\n", "        K_smem[(k_r + 0) * ", ldk,
          " + k_s] = kv.x;\n", "        K_smem[(k_r + 1) * ", ldk,
          " + k_s] = kv.y;\n", "        K_smem[(k_r + 2) * ", ldk,
          " + k_s] = kv.z;\n", "        K_smem[(k_r + 3) * ", ldk,
          " + k_s] = kv.w;\n      }\n    }\n");
    }
    if (bd_padded > slices * 4) {
      absl::StrAppend(&op_code, "    if (tid < 16) {\n");
      for (int c = slices * 4; c < bd_padded; ++c) {
        absl::StrAppend(&op_code, "      K_smem[", c * ldk,
                        " + tid] = 0.0h;\n");
      }
      absl::StrAppend(&op_code, "    }\n");
    }
    absl::StrAppend(
        &op_code, "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n",
        "    // 2. Hardware FP16->FP32 simdgroup_matrix Q * K^T (8x16 per "
        "warp).\n",
        "    float2 s_frag0 = float2(0.0f);\n",
        "    float2 s_frag1 = float2(0.0f);\n",
        "    #pragma unroll\n    for (int dd = 0; dd < ", td, "; ++dd) {\n",
        "      half2 qf = *reinterpret_cast<const threadgroup "
        "half2*>(&Q_smem[q_smem_base + dd * 8]);\n",
        "      int k_off = (dd * 8 + sm) * ", ldk, " + sn;\n",
        "      half2 kf0 = *reinterpret_cast<const threadgroup "
        "half2*>(&K_smem[k_off + 0]);\n",
        "      half2 kf1 = *reinterpret_cast<const threadgroup "
        "half2*>(&K_smem[k_off + 8]);\n",
        "      s_frag0 = mma_f16_f32_8x8(qf, kf0, s_frag0);\n",
        "      s_frag1 = mma_f16_f32_8x8(qf, kf1, s_frag1);\n    }\n");

    // 3. Optional softcapping, mask, and active/causal key bounds.
    if (has_softcap) {
      for (int ik = 0; ik < 2; ++ik) {
        absl::StrAppend(
            &op_code, "    s_frag", ik,
            ".x = (float)args.softcap * tanh((s_frag", ik,
            ".x / inv_ln2) / (float)args.softcap) * inv_ln2;\n", "    s_frag",
            ik, ".y = (float)args.softcap * tanh((s_frag", ik,
            ".y / inv_ln2) / (float)args.softcap) * inv_ln2;\n");
      }
    }

    if (!is_interior) {
      for (int ik = 0; ik < 2; ++ik) {
        absl::StrAppend(&op_code, "    int col", ik, "_0 = key_base + ", ik * 8,
                        " + sn;\n", "    int col", ik, "_1 = col", ik,
                        "_0 + 1;\n");
        if (has_mask) {
          for (int c = 0; c < 2; ++c) {
            absl::StrAppend(
                &op_code, "    bool m_ok", ik, "_", c, " = (row_valid && col",
                ik, "_", c, " < active_tokens);\n", "    int msl", ik, "_", c,
                " = col", ik, "_", c, " >> 2;\n", "    int mcm", ik, "_", c,
                " = col", ik, "_", c, " & 3;\n", "    half4 mk", ik, "_", c,
                " = m_ok", ik, "_", c,
                " ? ucl::Convert<half4>(args.mask.Read(X_row, 0, msl", ik, "_",
                c, ")) : half4(0.0h);\n", "    float mv", ik, "_", c,
                " = (float)((mcm", ik, "_", c, " == 0) ? mk", ik, "_", c,
                ".x : ((mcm", ik, "_", c, " == 1) ? mk", ik, "_", c,
                ".y : ((mcm", ik, "_", c, " == 2) ? mk", ik, "_", c, ".z : mk",
                ik, "_", c, ".w)));\n", "    if (args.is_bool_mask) {\n",
                "      if (mv", ik, "_", c, " < 0.5f) s_frag", ik,
                (c == 0 ? ".x" : ".y"), " = -10000.0f;\n", "    } else {\n",
                "      s_frag", ik, (c == 0 ? ".x" : ".y"), " += mv", ik, "_",
                c, ";\n", "    }\n");
          }
        }
        if (is_causal) {
          absl::StrAppend(&op_code, "    if (col", ik,
                          "_0 >= active_tokens || col", ik,
                          "_0 > P_row) s_frag", ik, ".x = -10000.0f;\n",
                          "    if (col", ik, "_1 >= active_tokens || col", ik,
                          "_1 > P_row) s_frag", ik, ".y = -10000.0f;\n");
        } else {
          absl::StrAppend(&op_code, "    if (col", ik,
                          "_0 >= active_tokens) s_frag", ik,
                          ".x = -10000.0f;\n", "    if (col", ik,
                          "_1 >= active_tokens) s_frag", ik,
                          ".y = -10000.0f;\n");
        }
      }
    }

    // 4. Online softmax reduction across the 4 threads sharing row `sm`
    // (lanes XOR 1 and XOR 8 in Apple's 8x8 simdgroup_matrix layout).
    absl::StrAppend(&op_code, R"(
    float row_max = max(max(s_frag0.x, s_frag0.y), max(s_frag1.x, s_frag1.y));
    row_max = max(row_max, simd_shuffle_xor(row_max, ushort(1)));
    row_max = max(row_max, simd_shuffle_xor(row_max, ushort(8)));
    float m_new = max(m_prev, row_max);
    float alp = exp2(m_prev - m_new);

    float2 p0_f = exp2(s_frag0 - m_new);
    float2 p1_f = exp2(s_frag1 - m_new);

    float row_sum = (p0_f.x + p0_f.y) + (p1_f.x + p1_f.y);
    row_sum += simd_shuffle_xor(row_sum, ushort(1));
    row_sum += simd_shuffle_xor(row_sum, ushort(8));
    l_prev = fma(l_prev, alp, row_sum);
    m_prev = m_new;

    half2 p_frag0 = half2(p0_f);
    half2 p_frag1 = half2(p1_f);
)");
    for (int id = 0; id < td; ++id) {
      absl::StrAppend(&op_code, "    o_frag", id, " *= alp;\n");
    }

    // 5. Coalesced half4 V load into V_smem[16][ldv] (placed after softmax so
    // s_frag0..1 are already dead, keeping register pressure low).
    absl::StrAppend(&op_code,
                    "\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    if (is_interior) {
      if (slices == 32) {
        absl::StrAppend(
            &op_code,
            "    {\n      int v_idx = v_head_base + (key_base / 4) * ",
            v_stride_s,
            " + tid;\n"
            "      #pragma unroll\n      for (int g = 0; g < 4; ++g) {\n"
            "        half4 vv = ucl::Convert<half4>(args.v.Read(v_idx + g * ",
            v_stride_s,
            "));\n"
            "        *reinterpret_cast<threadgroup half4*>(&V_smem[(g * 4 + "
            "v_k_sub) * ",
            ldv, " + v_c_col]) = vv;\n      }\n    }\n");
      } else {
        absl::StrAppend(
            &op_code, "    if (v_c_slice < ", slices, ") {\n",
            "      int v_idx = v_head_base + (key_base / 4) * ", v_stride_s,
            " + tid;\n"
            "      #pragma unroll\n      for (int g = 0; g < 4; ++g) {\n"
            "        half4 vv = ucl::Convert<half4>(args.v.Read(v_idx + g * ",
            v_stride_s,
            "));\n"
            "        *reinterpret_cast<threadgroup half4*>(&V_smem[(g * 4 + "
            "v_k_sub) * ",
            ldv, " + v_c_col]) = vv;\n      }\n    }\n");
      }
    } else {
      absl::StrAppend(
          &op_code, "    if (v_c_slice < ", slices, ") {\n",
          "      #pragma unroll\n      for (int g = 0; g < 4; ++g) {\n",
          "        int gk = key_base + g * 4 + v_k_sub;\n",
          "        half4 vv = (gk < active_tokens) ? "
          "ucl::Convert<half4>(args.v.Read(v_head_base + (key_base / 4 + g) * ",
          v_stride_s, " + tid)) : half4(0.0h);\n",
          "        *reinterpret_cast<threadgroup half4*>(&V_smem[(g * 4 + "
          "v_k_sub) * ",
          ldv, " + v_c_col]) = vv;\n      }\n    }\n");
    }
    if (bd_padded > slices * 4) {
      absl::StrAppend(&op_code, "    if (tid < 16) {\n");
      for (int c = slices * 4; c < bd_padded; ++c) {
        absl::StrAppend(&op_code, "      V_smem[tid * ", ldv, " + ", c,
                        "] = 0.0h;\n");
      }
      absl::StrAppend(&op_code, "    }\n");
    }

    // 6. Wait for V_smem and compute P * V (8x128 per warp).
    absl::StrAppend(&op_code,
                    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    for (int id = 0; id < td; ++id) {
      absl::StrAppend(
          &op_code, "    {\n      int v_col = ", id * 8, " + sn;\n",
          "      half2 vf0 = *reinterpret_cast<const threadgroup "
          "half2*>(&V_smem[(0 + sm) * ",
          ldv, " + v_col]);\n", "      o_frag", id,
          " = mma_f16_f32_8x8(p_frag0, vf0, o_frag", id, ");\n",
          "      half2 vf1 = *reinterpret_cast<const threadgroup "
          "half2*>(&V_smem[(8 + sm) * ",
          ldv, " + v_col]);\n", "      o_frag", id,
          " = mma_f16_f32_8x8(p_frag1, vf1, o_frag", id, ");\n", "    }\n");
    }
  };

  // 6. Main key loop: fast interior loop (when no external mask) + boundary
  // tail.
  absl::StrAppend(&op_code, "  int key_base = 0;\n");
  if (!has_mask) {
    if (is_causal) {
      absl::StrAppend(&op_code,
                      "  int interior_end = min(X0 + q_start + 1, max_tokens)"
                      " - ",
                      kPrefillKeyBlock, ";\n");
    } else {
      absl::StrAppend(&op_code, "  int interior_end = max_tokens - ",
                      kPrefillKeyBlock, ";\n");
    }
    absl::StrAppend(&op_code,
                    "  for (; key_base <= interior_end; key_base += ",
                    kPrefillKeyBlock, ") {\n");
    emit_key_block_body(/*is_interior=*/true);
    absl::StrAppend(&op_code, "  }\n");
  }
  absl::StrAppend(&op_code, "  for (; key_base < max_tokens; key_base += ",
                  kPrefillKeyBlock, ") {\n");
  emit_key_block_body(/*is_interior=*/false);
  absl::StrAppend(&op_code, "  }\n\n");

  // 7. Normalize output and write float4 slices via Q_smem (reusing Q_smem so
  // KV_smem stays at 5,120 B).
  absl::StrAppend(&op_code, R"(  float inv_l = 1.0f / (l_prev + 1e-10f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup half* warp_smem = &Q_smem[sg_id * 8 * )",
                  ldq, "];\n");
  for (int id = 0; id < td; ++id) {
    absl::StrAppend(&op_code,
                    "  *reinterpret_cast<threadgroup half2*>(&warp_smem[sm * ",
                    ldq, " + ", id * 8, " + sn]) = half2(o_frag", id,
                    " * inv_l);\n");
  }
  absl::StrAppend(&op_code, R"(  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane_id < args.slices && Y_warp < dst_h) {
    for (int r = 0; r < 8; ++r) {
      int out_x = X_warp + r;
      if (out_x < valid_w) {
        half4 out_v = *reinterpret_cast<const threadgroup half4*>(
            &warp_smem[r * )",
                  ldq, R"( + lane_id * 4]);
        args.dst.Write(ucl::Convert<args.dst::type>(out_v), out_x, Y_warp,
                       lane_id);
      } else if (out_x < dst_w) {
        args.dst.Write(ucl::Convert<args.dst::type>(float4(0.0f)), out_x,
                       Y_warp, lane_id);
      }
    }
  }
}
)");

  custom_op.code_ = std::move(op_code);
  return std::make_unique<FusedFlashAttentionPrefillOp>(std::move(custom_op));
}

}  // namespace

bool SupportsFusedSdpaKernels(const ::ml_drift::GpuInfo& gpu_info) {
  return gpu_info.IsApple() && gpu_info.IsApiMetal();
}

absl::Status BuildSdpaTransposedGpuGraph(
    const std::vector<uint32_t>& input_ids, uint32_t output_id,
    const SdpaTransposedAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() < 3) {
    return absl::InvalidArgumentError(
        "SDPA transposed expects at least 3 inputs (Q, K, V).");
  }

  ABSL_ASSIGN_OR_RETURN(auto q, model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(auto k, model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(auto v, model_builder->GetTensor(input_ids[2]));

  const ::ml_drift::TensorDescriptor* mask_desc = nullptr;
  ::ml_drift::GpuModelBuilder::TensorHandle mask;
  const ::ml_drift::TensorDescriptor* param_desc = nullptr;
  ::ml_drift::GpuModelBuilder::TensorHandle param_tensor;

  if (input_ids.size() == 4) {
    ABSL_ASSIGN_OR_RETURN(auto mask_or_param,
                          model_builder->GetTensor(input_ids[3]));
    if (mask_or_param.tensor_desc.GetDataType() ==
        ::ml_drift::DataType::INT32) {
      param_tensor = mask_or_param;
      param_desc = &param_tensor.tensor_desc;
    } else {
      mask = mask_or_param;
      mask_desc = &mask.tensor_desc;
    }
  } else if (input_ids.size() > 4) {
    ABSL_ASSIGN_OR_RETURN(mask, model_builder->GetTensor(input_ids[3]));
    mask_desc = &mask.tensor_desc;
    ABSL_ASSIGN_OR_RETURN(param_tensor, model_builder->GetTensor(input_ids[4]));
    param_desc = &param_tensor.tensor_desc;
  }

  const int head_dim = q.tensor_desc.GetBHWCShape().c;
  const bool supports_fused_kernels =
      SupportsFusedSdpaKernels(model_builder->gpu_info());

  // The fused Flash-Attention prefill kernel indexes K and V directly in the
  // packed 4D layout produced by `odml.cache_update`, so it requires
  // `from_cache_update` and BUFFER storage. It is also written against Apple
  // SIMD intrinsics and dispatches a single 32-lane SIMD group per
  // threadgroup, which covers at most 32 channel slices (head_dim <= 128).
  // Everything else falls back to the multi-op graph below.
  const bool is_supported_flash_prefill =
      attr.is_prefill && attr.from_cache_update && head_dim % 4 == 0 &&
      head_dim <= 128 &&
      k.tensor_desc.GetStorageType() == ::ml_drift::TensorStorageType::BUFFER &&
      v.tensor_desc.GetStorageType() == ::ml_drift::TensorStorageType::BUFFER &&
      supports_fused_kernels;

  if (is_supported_flash_prefill) {
    auto dst = model_builder->AddTensor(q.tensor_desc.GetBHWCShape(),
                                        q.tensor_desc.GetDataType());
    auto op = CreateFusedFlashAttentionPrefill(
        model_builder->gpu_info(), q.tensor_desc, k.tensor_desc, v.tensor_desc,
        mask_desc, param_desc, dst.tensor_desc, attr);
    std::vector<::ml_drift::GpuModelBuilder::TensorHandle> src_tensors = {q, k,
                                                                          v};
    if (mask_desc) src_tensors.push_back(mask);
    if (param_desc) src_tensors.push_back(param_tensor);
    model_builder->AddGpuOperation(src_tensors, {dst}, std::move(op),
                                   "flash_prefill_sdpa");
    return model_builder->UpdateOutputTensor(dst, output_id);
  }

  // Fused Flash-Decode is currently optimized for Apple Silicon with
  // head_dim = 128 (slices = 32 matching the 32-thread SIMD wave size).
  // For other head dimensions or non-Metal backends, fall back to the multi-op
  // graph.
  const bool is_supported_flash_decode =
      attr.from_cache_update && !attr.is_prefill && head_dim == 128 &&
      k.tensor_desc.GetStorageType() == ::ml_drift::TensorStorageType::BUFFER &&
      v.tensor_desc.GetStorageType() == ::ml_drift::TensorStorageType::BUFFER &&
      supports_fused_kernels;

  if (is_supported_flash_decode) {
    // Single fused Flash-Decode SDPA op.
    ABSL_ASSIGN_OR_RETURN(auto output_ref, model_builder->GetTensor(output_id));
    const auto output_shape = output_ref.tensor_desc.GetBHWCShape();
    const auto q_shape = q.tensor_desc.GetBHWCShape();
    const bool is_flattened_dst =
        (output_shape.h == 1 && output_shape.c == q_shape.h * q_shape.c);

    const auto dst_shape = is_flattened_dst ? output_shape : q_shape;
    auto dst = model_builder->AddTensor(dst_shape, q.tensor_desc.GetDataType());

    auto op = CreateFusedFlashDecodeSdpa(
        model_builder->gpu_info(), q.tensor_desc, k.tensor_desc, v.tensor_desc,
        mask_desc, param_desc, dst.tensor_desc, attr, is_flattened_dst);

    std::vector<::ml_drift::GpuModelBuilder::TensorHandle> src_tensors = {q, k,
                                                                          v};
    if (mask_desc) src_tensors.push_back(mask);
    if (param_desc) src_tensors.push_back(param_tensor);

    model_builder->AddGpuOperation(src_tensors, {dst}, std::move(op),
                                   "flash_decode_sdpa");
    return model_builder->UpdateOutputTensor(dst, output_id);
  }

  ::ml_drift::GpuModelBuilder::TensorHandle logits;
  if (attr.from_cache_update) {
    ::ml_drift::WeightsDescription bmm1_desc = attr.bmm1_weights.desc;
    bmm1_desc.type = q.tensor_desc.GetDataType();
    const ::ml_drift::GpuModelBuilder::Weights bmm1_external_weights =
        ::ml_drift::CreateExternalWeights(k, bmm1_desc,
                                          attr.bmm1_weights.weights_shape);

    ::ml_drift::ConvRuntimeCheckDesc bmm1_runtime_check;
    if (param_desc) {
      bmm1_runtime_check.dst_end_ch_index = attr.runtime_check.src_end_ch_index;
    }

    ABSL_ASSIGN_OR_RETURN(
        logits,
        model_builder->FullyConnectedExternalWeights(
            q, bmm1_external_weights, /*biases=*/nullptr, /*src_exp=*/nullptr,
            bmm1_runtime_check, param_desc ? &param_tensor : nullptr));
  } else {
    ::ml_drift::BatchedMatMulAttributes bmm1_attr;
    bmm1_attr.transpose_left = false;
    bmm1_attr.transpose_right = true;
    ::ml_drift::ConvRuntimeCheckDesc bmm1_runtime_check;
    if (param_desc) {
      bmm1_runtime_check.dst_end_ch_index = attr.runtime_check.src_end_ch_index;
    }
    ABSL_ASSIGN_OR_RETURN(
        logits, model_builder->BatchedMatMul(
                    q, k, bmm1_attr, /*src_exp=*/nullptr, bmm1_runtime_check,
                    param_desc ? &param_tensor : nullptr));
  }

  if (attr.softcap.has_value() && *attr.softcap > 0.0f) {
    const float cap_val = *attr.softcap;
    logits = model_builder->Multiplication(logits, 1.0f / cap_val);
    logits =
        model_builder->Elementwise(logits, ::ml_drift::OperationType::TANH);
    logits = model_builder->Multiplication(logits, cap_val);
  }

  if (mask_desc != nullptr) {
    if (mask.tensor_desc.GetDataType() == ::ml_drift::DataType::BOOL) {
      ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::BHWC>,
                         ::ml_drift::DataType::FLOAT32>
          fill_tensor;
      fill_tensor.shape = ::ml_drift::BHWC(1, 1, 1, 1);
      // Use a large negative value to simulate -inf. std::limit<float>::min()
      // causes regression.
      fill_tensor.data = {-10000.0f};
      auto neg_val = model_builder->AddConstantTensor(
          fill_tensor, logits.tensor_desc.GetDataType());
      logits = model_builder->SelectV2(mask, logits, neg_val);
    } else {
      logits = model_builder->Add(logits, mask);
    }
  }

  ::ml_drift::SoftmaxRuntimeCheckDesc softmax_runtime_check;
  if (param_desc) {
    softmax_runtime_check.end_ch_index = attr.runtime_check.src_end_ch_index;
  }

  ::ml_drift::GpuModelBuilder::TensorHandle output;
  if (attr.from_cache_update) {
    auto sfmx_partial = model_builder->SoftmaxReduce(
        logits, softmax_runtime_check, param_desc ? &param_tensor : nullptr);

    ::ml_drift::WeightsDescription bmm2_desc = attr.bmm2_weights.desc;
    bmm2_desc.type = logits.tensor_desc.GetDataType();
    const ::ml_drift::GpuModelBuilder::Weights bmm2_external_weights =
        ::ml_drift::CreateExternalWeights(v, bmm2_desc,
                                          attr.bmm2_weights.weights_shape);

    ::ml_drift::ConvRuntimeCheckDesc bmm2_runtime_check;
    if (param_desc) {
      bmm2_runtime_check.src_end_ch_index = attr.runtime_check.src_end_ch_index;
    }

    ABSL_ASSIGN_OR_RETURN(
        output,
        model_builder->FullyConnectedExternalWeights(
            logits, bmm2_external_weights, /*biases=*/nullptr, &sfmx_partial,
            bmm2_runtime_check, param_desc ? &param_tensor : nullptr));
  } else {
    auto probs = model_builder->Softmax(logits, softmax_runtime_check,
                                        param_desc ? &param_tensor : nullptr);

    ::ml_drift::BatchedMatMulAttributes bmm2_attr;
    bmm2_attr.transpose_left = false;
    bmm2_attr.transpose_right = true;
    ::ml_drift::ConvRuntimeCheckDesc bmm2_runtime_check;
    if (param_desc) {
      bmm2_runtime_check.src_end_ch_index = attr.runtime_check.src_end_ch_index;
    }
    ABSL_ASSIGN_OR_RETURN(
        output, model_builder->BatchedMatMul(
                    probs, v, bmm2_attr, /*src_exp=*/nullptr,
                    bmm2_runtime_check, param_desc ? &param_tensor : nullptr));
  }

  ABSL_ASSIGN_OR_RETURN(auto output_ref, model_builder->GetTensor(output_id));
  const auto output_shape = output_ref.tensor_desc.GetBHWCShape();
  if (output.tensor_desc.GetBHWCShape() != output_shape) {
    output = model_builder->Reshape(output, output_shape);
  }
  return model_builder->UpdateOutputTensor(output, output_id);
}

absl::Status CreateSdpaTransposedFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const SdpaTransposedAttributes& attr =
      std::any_cast<const SdpaTransposedAttributes&>(node.operation.attributes);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildSdpaTransposedGpuGraph(input_ids, outputs[0]->id, attr,
                                     model_builder);
}

absl::Status CreateSdpaTransposedFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const SdpaTransposedAttributes& attr =
      std::any_cast<const SdpaTransposedAttributes&>(node.attr);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildSdpaTransposedGpuGraph(input_ids, outputs[0]->id, attr,
                                     model_builder);
}

}  // namespace litert::ml_drift
