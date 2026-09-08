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

  custom_op.work_group_size_ = ::ml_drift::int3(1, slices, kNumSimdGroups);
  custom_op.args_.AddInt("cache_size", k_desc.GetBHWCShape().w);

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
  }

  absl::StrAppend(&op_code, R"(
  int total_chunks = (active_tokens + 3) / 4;
  int chunks_per_simd = (total_chunks + 15) / 16;
  int chunk_start = simd_id * chunks_per_simd;
  int chunk_end = min(total_chunks, chunk_start + chunks_per_simd);
  int safe_chunk_end = max(chunk_start, min(chunk_end, (active_tokens / 16) * 4));

  // Note: This Flash-Decode SDPA kernel is optimized for float16 / half precision.
  half4 q_slice = ucl::Convert<half4>(args.q.Read(X, Y, tid));
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
    int out_slice = Y * %d + tid;
    args.dst.Write(ucl::Convert<args.dst::type>(final_acc), X, 0, out_slice);
)",
                                                     slices)
                                   : R"(
    args.dst.Write(ucl::Convert<args.dst::type>(final_acc), X, Y, tid);
)",
                  R"(
  }
}
)");

  custom_op.code_ = std::move(op_code);
  return std::make_unique<FusedFlashDecodeSdpaOp>(std::move(custom_op));
}

}  // namespace

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

  // Fused Flash-Decode is currently optimized for Apple Silicon with
  // head_dim = 128 (slices = 32 matching the 32-thread SIMD wave size).
  // For prefill, other head dimensions, or non-Apple GPUs, fall back to the
  // multi-op BMM graph.
  const int head_dim = q.tensor_desc.GetBHWCShape().c;
  const bool is_supported_flash_decode = !attr.is_prefill && head_dim == 128 &&
                                         model_builder->gpu_info().IsApple();

  if (!is_supported_flash_decode) {
    ::ml_drift::WeightsDescription bmm1_desc = attr.bmm1_weights.desc;
    bmm1_desc.type = q.tensor_desc.GetDataType();
    const ::ml_drift::GpuModelBuilder::Weights bmm1_external_weights =
        ::ml_drift::CreateExternalWeights(k, bmm1_desc,
                                          attr.bmm1_weights.weights_shape);

    ::ml_drift::ConvRuntimeCheckDesc bmm1_runtime_check = {
        .dst_end_ch_index =
            param_desc ? attr.runtime_check.src_end_ch_index : std::nullopt,
    };

    ABSL_ASSIGN_OR_RETURN(
        auto logits,
        model_builder->FullyConnectedExternalWeights(
            q, bmm1_external_weights, /*biases=*/nullptr, /*src_exp=*/nullptr,
            bmm1_runtime_check, param_desc ? &param_tensor : nullptr));

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

    ::ml_drift::SoftmaxRuntimeCheckDesc softmax_runtime_check = {
        .end_ch_index =
            param_desc ? attr.runtime_check.src_end_ch_index : std::nullopt,
    };
    auto sfmx_partial = model_builder->SoftmaxReduce(
        logits, softmax_runtime_check, param_desc ? &param_tensor : nullptr);

    ::ml_drift::WeightsDescription bmm2_desc = attr.bmm2_weights.desc;
    bmm2_desc.type = logits.tensor_desc.GetDataType();
    const ::ml_drift::GpuModelBuilder::Weights bmm2_external_weights =
        ::ml_drift::CreateExternalWeights(v, bmm2_desc,
                                          attr.bmm2_weights.weights_shape);

    ::ml_drift::ConvRuntimeCheckDesc bmm2_runtime_check = {
        .src_end_ch_index =
            param_desc ? attr.runtime_check.src_end_ch_index : std::nullopt,
    };

    ABSL_ASSIGN_OR_RETURN(
        auto output,
        model_builder->FullyConnectedExternalWeights(
            logits, bmm2_external_weights, /*biases=*/nullptr, &sfmx_partial,
            bmm2_runtime_check, param_desc ? &param_tensor : nullptr));

    return model_builder->UpdateOutputTensor(output, output_id);
  }

  // Single fused SDPA op.
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
