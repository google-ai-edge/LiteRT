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

#include "ml_drift_delegate/delegate/composite/sparse_sdpa_transposed_kernel.h"

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
#include "absl/strings/str_replace.h"  // from @com_google_absl
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
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sparse_sdpa_transposed_parser.h"

namespace litert::ml_drift {
namespace {

using TensorHandle = ::ml_drift::GpuModelBuilder::TensorHandle;

// Pre-computes per-query-row unique 4-token KV micro-block indices into
// `prep_blocks [1, 1, seq_len, (block_topk_slices + 2) * 4]`:
// - Slices `[0, block_topk_slices)` hold sanitized `topk` indices (`-1`
// padded).
// - Slices `block_topk_slices + {0, 1}` hold up to 8 partially visible boundary
//   4-token slices (`any_vis && !all_vis`) deduplicated against `topk`.
// Running this once per `q_pos` removes `num_q_heads * splits` redundant mask
// scans and deduplication loops from the attention kernels.
class SparsePrepareBlocksOp : public ::ml_drift::GPUOperation {
 public:
  SparsePrepareBlocksOp(int seq_len, int cache_len, int block_topk,
                        bool bool_mask)
      : seq_len_(seq_len) {
    work_group_size_ = ::ml_drift::int3(64, 1, 1);
    const int block_topk_slices = (block_topk + 3) / 4;
    const bool cache_topk_smem = (block_topk <= 2048);
    const int smem_topk_len = std::max(4, block_topk_slices * 4);
    args_.AddInt("seq_len", seq_len);
    args_.AddInt("cache_slices", cache_len / 4);
    args_.AddInt("block_topk", block_topk);
    args_.AddInt("block_topk_slices", block_topk_slices);

    const std::string mask_s_code = bool_mask ? R"(
      bool4 m4 = args.mask.Read<bool>(q_pos, 0, s, 0);
      bool v0 = m4.x;
      bool v1 = m4.y;
      bool v2 = m4.z;
      bool v3 = m4.w;
)"
                                              : R"(
      float4 m4 = args.mask.Read<float>(q_pos, 0, s, 0);
      bool v0 = (m4.x == 0.0f);
      bool v1 = (m4.y == 0.0f);
      bool v2 = (m4.z == 0.0f);
      bool v3 = (m4.w == 0.0f);
)";

    const std::string smem_topk_decl =
        cache_topk_smem
            ? absl::StrCat("  __local int smem_topk[", smem_topk_len, "];\n")
            : "";
    const std::string smem_topk_store = cache_topk_smem ? R"(
    smem_topk[base_k + 0] = blk4.x;
    smem_topk[base_k + 1] = blk4.y;
    smem_topk[base_k + 2] = blk4.z;
    smem_topk[base_k + 3] = blk4.w;
)"
                                                        : "";
    const std::string dedup_check = cache_topk_smem ? R"(
      for (int k = tid; k < args.block_topk; k += 64) {
        if (smem_topk[k] == s) {
          found_local = 1;
        }
      }
)"
                                                    : R"(
      for (int bs = tid; bs < args.block_topk_slices; bs += 64) {
        int4 blk4 = args.topk.Read<int>(q_pos, 0, bs, 0);
        int base_k = bs * 4;
        if (base_k + 0 < args.block_topk && blk4.x == s) { found_local = 1; }
        if (base_k + 1 < args.block_topk && blk4.y == s) { found_local = 1; }
        if (base_k + 2 < args.block_topk && blk4.z == s) { found_local = 1; }
        if (base_k + 3 < args.block_topk && blk4.w == s) { found_local = 1; }
      }
)";

    code_ = absl::StrCat(R"(
MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int q_pos = ucl::GetGroupId<1>();
  __local int smem_tail0[64];
  __local int smem_tail1[64];
  __local int smem_tails[8];
)",
                         smem_topk_decl, R"(
  for (int bs = tid; bs < args.block_topk_slices; bs += 64) {
    int4 blk4 = args.topk.Read<int>(q_pos, 0, bs, 0);
    int base_k = bs * 4;
    if (base_k + 0 >= args.block_topk) { blk4.x = -1; }
    if (base_k + 1 >= args.block_topk) { blk4.y = -1; }
    if (base_k + 2 >= args.block_topk) { blk4.z = -1; }
    if (base_k + 3 >= args.block_topk) { blk4.w = -1; }
    args.prep_blocks.Write(blk4, q_pos, 0, bs, 0);
)",
                         smem_topk_store, R"(  }

  int tail0 = -1;
  int tail1 = -1;
  for (int s = tid; s < args.cache_slices; s += 64) {
)",
                         mask_s_code, R"(
    bool any_vis = v0 || v1 || v2 || v3;
    bool all_vis = v0 && v1 && v2 && v3;
    if (any_vis && !all_vis) {
      if (tail0 < 0) {
        tail0 = s;
      } else if (tail1 < 0) {
        tail1 = s;
      }
    }
  }
  smem_tail0[tid] = tail0;
  smem_tail1[tid] = tail1;
  ucl::SyncThreads<WorkGroup, Local>();

  if (tid == 0) {
    for (int i = 0; i < 8; ++i) {
      smem_tails[i] = -1;
    }
    int count = 0;
    for (int t = 0; t < 64; ++t) {
      if (smem_tail0[t] >= 0 && count < 8) {
        smem_tails[count] = smem_tail0[t];
        count += 1;
      }
      if (smem_tail1[t] >= 0 && count < 8) {
        smem_tails[count] = smem_tail1[t];
        count += 1;
      }
    }
  }
  ucl::SyncThreads<WorkGroup, Local>();

  // Fixed-bound loop keeps `SyncThreads` in WGSL-provable uniform control flow.
  for (int ti = 0; ti < 8; ++ti) {
    int s = smem_tails[ti];
    int found_local = 0;
    if (s >= 0) {
)",
                         dedup_check, R"(    }
    smem_tail0[tid] = found_local;
    ucl::SyncThreads<WorkGroup, Local>();
    if (tid == 0) {
      int any_found = 0;
      for (int t = 0; t < 64; ++t) {
        if (smem_tail0[t] != 0) {
          any_found = 1;
        }
      }
      if (any_found != 0) {
        smem_tails[ti] = -1;
      }
    }
    ucl::SyncThreads<WorkGroup, Local>();
  }

  if (tid == 0) {
    int4 b_extra0 = ucl::Init<int4>(smem_tails[0], smem_tails[1],
                                    smem_tails[2], smem_tails[3]);
    int4 b_extra1 = ucl::Init<int4>(smem_tails[4], smem_tails[5],
                                    smem_tails[6], smem_tails[7]);
    args.prep_blocks.Write(b_extra0, q_pos, 0, args.block_topk_slices + 0, 0);
    args.prep_blocks.Write(b_extra1, q_pos, 0, args.block_topk_slices + 1, 0);
  }
}
)");
  }

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(64, seq_len_, 1);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

 private:
  int seq_len_;
};

// Computes sparse GQA attention over `prep_blocks` with:
// 1. GQA sibling query-head grouping (`heads_per_wg` in {1, 2, 4}): loads K and
//    V cache lines once per KV head and updates `heads_per_wg` query heads in
//    registers (`4x` fewer global K/V reads on `gqa_ratio >= 4`).
// 2. Native 32-lane `simd_sum` on Apple Metal (`use_metal_simd`), eliminating
//    all `__local` barriers from the main block loop.
// 3. Strided Split-KV (`splits >= 1`) across `prep_slices` for decode / verify
//    (`seq_len <= 8`), keeping all GPU cores busy even when `num_kv_heads` is
//    small (`2` on Qwen4-2B, `4` on Qwen4-27B).
class SparseSdpaOp : public ::ml_drift::GPUOperation {
 public:
  SparseSdpaOp(const ::ml_drift::GpuInfo& gpu_info, int num_q_heads,
               int num_kv_heads, int heads_per_wg, int splits, int seq_len,
               int head_dim, int cache_len, int prep_slices, float scale,
               bool bool_mask, bool from_cache_update)
      : num_head_groups_(num_q_heads / heads_per_wg),
        seq_len_(seq_len),
        splits_(splits) {
    work_group_size_ = ::ml_drift::int3(32, 1, 1);
    const int head_slices = head_dim / 4;
    const int cache_slices = cache_len / 4;
    const bool has_second_slice = (head_slices > 32);
    const bool use_metal_simd =
        gpu_info.IsApiMetal() && gpu_info.IsWaveSizeEqualTo32();

    args_.AddInt("num_groups", num_q_heads / num_kv_heads);
    args_.AddInt("head_slices", head_slices);
    args_.AddInt("cache_slices", cache_slices);
    args_.AddInt("prep_slices", prep_slices);
    args_.AddFloat("scale", scale);
    if (from_cache_update) {
      const int k_stride_slice = cache_slices * 4;
      const int k_stride_head = head_slices * k_stride_slice;
      const int v_stride_block = head_slices * 4;
      const int v_stride_head = cache_slices * v_stride_block;
      args_.AddInt("k_stride_slice", k_stride_slice);
      args_.AddInt("k_stride_head", k_stride_head);
      args_.AddInt("v_stride_block", v_stride_block);
      args_.AddInt("v_stride_head", v_stride_head);
    }

    const std::string mask_b_code = bool_mask ? R"(
      bool4 m4 = args.mask.Read<bool>(q_pos, 0, b_id, 0);
      bool v0 = m4.x;
      bool v1 = m4.y;
      bool v2 = m4.z;
      bool v3 = m4.w;
)"
                                              : R"(
      float4 m4 = args.mask.Read<float>(q_pos, 0, b_id, 0);
      bool v0 = (m4.x == 0.0f);
      bool v1 = (m4.y == 0.0f);
      bool v2 = (m4.z == 0.0f);
      bool v3 = (m4.w == 0.0f);
)";

    auto emit_k_dot_slice = [&](const std::string& ds_expr,
                                const std::string& q_prefix) {
      std::string out;
      if (from_cache_update) {
        absl::StrAppend(&out,
                        "        int k_base = kv_head * args.k_stride_head + ",
                        ds_expr, " * args.k_stride_slice + base_tok;\n");
      }
      const char* comps[4] = {"x", "y", "z", "w"};
      for (int r = 0; r < 4; ++r) {
        const std::string k_read =
            from_cache_update
                ? absl::StrCat("args.key_cache.Read<float>(k_base + ", r, ")")
                : absl::StrCat("args.key_cache.Read<float>(base_tok + ", r,
                               ", kv_head, ", ds_expr, ", 0)");
        absl::StrAppend(&out, "        if (v", r, ") {\n",
                        "          float4 kv = ", k_read, ";\n");
        for (int h = 0; h < heads_per_wg; ++h) {
          absl::StrAppend(&out, "          partial_qk_", h, ".", comps[r],
                          " += dot(", q_prefix, "_", h, ", kv);\n");
        }
        absl::StrAppend(&out, "        }\n");
      }
      return out;
    };

    // Guard masked/unwritten V lanes before multiplying by 0.0f so unwritten
    // NaN cache slots past `active_tokens` never poison `acc` via `0 * NaN`.
    auto emit_v_acc_slice = [&](const std::string& ds_expr,
                                const std::string& acc_prefix) {
      std::string out;
      if (from_cache_update) {
        absl::StrAppend(&out,
                        "        int v_base = kv_head * args.v_stride_head + "
                        "b_id * args.v_stride_block + ",
                        ds_expr, " * 4;\n");
        const char* comps[4] = {"x", "y", "z", "w"};
        for (int r = 0; r < 4; ++r) {
          absl::StrAppend(
              &out, "        if (v", r, ") {\n",
              "          float4 vv = args.value_cache.Read<float>(v_base + ", r,
              ");\n");
          for (int h = 0; h < heads_per_wg; ++h) {
            absl::StrAppend(&out, "          ", acc_prefix, "_", h,
                            " += probs_", h, ".", comps[r], " * vv;\n");
          }
          absl::StrAppend(&out, "        }\n");
        }
      } else {
        absl::StrAppend(
            &out, "        int d0 = ", ds_expr, " * 4;\n",
            "        float4 u0 = args.value_cache.Read<float>(d0 + 0, kv_head, "
            "b_id, 0);\n",
            "        float4 u1 = args.value_cache.Read<float>(d0 + 1, kv_head, "
            "b_id, 0);\n",
            "        float4 u2 = args.value_cache.Read<float>(d0 + 2, kv_head, "
            "b_id, 0);\n",
            "        float4 u3 = args.value_cache.Read<float>(d0 + 3, kv_head, "
            "b_id, 0);\n",
            "        if (!v0) { u0.x = 0.0f; u1.x = 0.0f; u2.x = 0.0f; u3.x = "
            "0.0f; }\n",
            "        if (!v1) { u0.y = 0.0f; u1.y = 0.0f; u2.y = 0.0f; u3.y = "
            "0.0f; }\n",
            "        if (!v2) { u0.z = 0.0f; u1.z = 0.0f; u2.z = 0.0f; u3.z = "
            "0.0f; }\n",
            "        if (!v3) { u0.w = 0.0f; u1.w = 0.0f; u2.w = 0.0f; u3.w = "
            "0.0f; }\n");
        for (int h = 0; h < heads_per_wg; ++h) {
          absl::StrAppend(&out, "        ", acc_prefix, "_", h,
                          " += ucl::Init<float4>(dot(probs_", h,
                          ", u0), dot(probs_", h, ", u1), dot(probs_", h,
                          ", u2), dot(probs_", h, ", u3));\n");
        }
      }
      return out;
    };

    std::string code;
    if (use_metal_simd) {
      code += "[[max_total_threads_per_threadgroup(32)]]\n";
    }
    code += R"(MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int head_group = ucl::GetGroupId<0>();
)";
    if (splits > 1) {
      absl::StrAppend(&code,
                      "  int q_pos_split = ucl::GetGroupId<1>();\n"
                      "  int q_pos = q_pos_split / ",
                      splits,
                      ";\n"
                      "  int split_id = q_pos_split % ",
                      splits, ";\n");
    } else {
      code +=
          "  int q_pos = ucl::GetGroupId<1>();\n"
          "  int split_id = 0;\n";
    }
    absl::StrAppend(&code, "  int q_head0 = head_group * ", heads_per_wg,
                    ";\n"
                    "  int kv_head = q_head0 / args.num_groups;\n");

    if (!use_metal_simd) {
      absl::StrAppend(&code, "  __local float4 smem_qk[", 2 * 32 * heads_per_wg,
                      "];\n"
                      "  int buf_parity = 0;\n");
    }

    for (int h = 0; h < heads_per_wg; ++h) {
      absl::StrAppend(&code, "  float4 q0_", h, " = ucl::Init<float4>(0.0f);\n",
                      "  float4 acc0_", h, " = ucl::Init<float4>(0.0f);\n",
                      "  float max_", h, " = -3.0e38f;\n", "  float sum_", h,
                      " = 0.0f;\n");
      if (has_second_slice) {
        absl::StrAppend(&code, "  float4 q1_", h,
                        " = ucl::Init<float4>(0.0f);\n", "  float4 acc1_", h,
                        " = ucl::Init<float4>(0.0f);\n");
      }
    }
    code += "  if (tid < args.head_slices) {\n";
    for (int h = 0; h < heads_per_wg; ++h) {
      absl::StrAppend(&code, "    q0_", h,
                      " = args.query.Read<float>(q_pos, q_head0 + ", h,
                      ", tid, 0);\n");
    }
    code += "  }\n";
    if (has_second_slice) {
      code += "  if (tid + 32 < args.head_slices) {\n";
      for (int h = 0; h < heads_per_wg; ++h) {
        absl::StrAppend(&code, "    q1_", h,
                        " = args.query.Read<float>(q_pos, q_head0 + ", h,
                        ", tid + 32, 0);\n");
      }
      code += "  }\n";
    }

    absl::StrAppend(
        &code,
        "\n  for (int bs = split_id; bs < args.prep_slices; bs += ", splits,
        ") {\n"
        "    int4 blk4 = args.prep_blocks.Read<int>(q_pos, 0, bs, "
        "0);\n"
        "    int b_arr[4];\n"
        "    b_arr[0] = blk4.x;\n"
        "    b_arr[1] = blk4.y;\n"
        "    b_arr[2] = blk4.z;\n"
        "    b_arr[3] = blk4.w;\n"
        "    for (int lane = 0; lane < 4; ++lane) {\n"
        "      int b_id = b_arr[lane];\n"
        "      if (b_id < 0 || b_id >= args.cache_slices) {\n"
        "        continue;\n"
        "      }\n",
        mask_b_code,
        "      bool any_vis = v0 || v1 || v2 || v3;\n"
        "      if (!any_vis) {\n"
        "        continue;\n"
        "      }\n"
        "      int base_tok = b_id * 4;\n");
    for (int h = 0; h < heads_per_wg; ++h) {
      absl::StrAppend(&code, "      float4 partial_qk_", h,
                      " = ucl::Init<float4>(0.0f);\n");
    }
    absl::StrAppend(&code, "      if (tid < args.head_slices) {\n",
                    emit_k_dot_slice("tid", "q0"), "      }\n");
    if (has_second_slice) {
      absl::StrAppend(&code, "      if (tid + 32 < args.head_slices) {\n",
                      emit_k_dot_slice("(tid + 32)", "q1"), "      }\n");
    }

    if (use_metal_simd) {
      for (int h = 0; h < heads_per_wg; ++h) {
        absl::StrAppend(&code, "      float4 dot_sum_", h,
                        " = simd_sum(partial_qk_", h, ");\n");
      }
    } else {
      absl::StrAppend(&code, "      int sh_base = buf_parity * ",
                      32 * heads_per_wg,
                      ";\n"
                      "      buf_parity = 1 - buf_parity;\n");
      for (int h = 0; h < heads_per_wg; ++h) {
        absl::StrAppend(&code, "      smem_qk[sh_base + ", h * 32,
                        " + tid] = partial_qk_", h, ";\n");
      }
      absl::StrAppend(
          &code,
          "      ucl::SyncThreads<WorkGroup, Local>();\n"
          "      int rh = tid / 8;\n"
          "      int rl = tid % 8;\n"
          "      if (rh < ",
          heads_per_wg,
          ") {\n"
          "        int r_idx = sh_base + rh * 32 + rl;\n"
          "        smem_qk[r_idx] += smem_qk[r_idx + 8] + smem_qk[r_idx + 16] "
          "+ smem_qk[r_idx + 24];\n"
          "      }\n"
          "      ucl::SyncThreads<WorkGroup, Local>();\n");
      for (int h = 0; h < heads_per_wg; ++h) {
        const int hb = h * 32;
        absl::StrAppend(
            &code, "      float4 dot_sum_", h, " = smem_qk[sh_base + ", hb + 0,
            "] + smem_qk[sh_base + ", hb + 1, "] + smem_qk[sh_base + ", hb + 2,
            "] + smem_qk[sh_base + ", hb + 3, "] + smem_qk[sh_base + ", hb + 4,
            "] + smem_qk[sh_base + ", hb + 5, "] + smem_qk[sh_base + ", hb + 6,
            "] + smem_qk[sh_base + ", hb + 7, "];\n");
      }
    }

    for (int h = 0; h < heads_per_wg; ++h) {
      absl::StrAppend(&code, "      float4 scores_", h,
                      " = ucl::Init<float4>(-3.0e38f);\n"
                      "      if (v0) { scores_",
                      h, ".x = dot_sum_", h,
                      ".x * args.scale; }\n"
                      "      if (v1) { scores_",
                      h, ".y = dot_sum_", h,
                      ".y * args.scale; }\n"
                      "      if (v2) { scores_",
                      h, ".z = dot_sum_", h,
                      ".z * args.scale; }\n"
                      "      if (v3) { scores_",
                      h, ".w = dot_sum_", h,
                      ".w * args.scale; }\n"
                      "      float bmax_",
                      h, " = max(max(scores_", h, ".x, scores_", h,
                      ".y), max(scores_", h, ".z, scores_", h,
                      ".w));\n"
                      "      float nmax_",
                      h, " = max(max_", h, ", bmax_", h,
                      ");\n"
                      "      float alpha_",
                      h, " = exp(max_", h, " - nmax_", h,
                      ");\n"
                      "      float4 probs_",
                      h,
                      " = ucl::Init<float4>(0.0f);\n"
                      "      if (v0) { probs_",
                      h, ".x = exp(scores_", h, ".x - nmax_", h,
                      "); }\n"
                      "      if (v1) { probs_",
                      h, ".y = exp(scores_", h, ".y - nmax_", h,
                      "); }\n"
                      "      if (v2) { probs_",
                      h, ".z = exp(scores_", h, ".z - nmax_", h,
                      "); }\n"
                      "      if (v3) { probs_",
                      h, ".w = exp(scores_", h, ".w - nmax_", h,
                      "); }\n"
                      "      sum_",
                      h, " = sum_", h, " * alpha_", h, " + dot(probs_", h,
                      ", ucl::Init<float4>(1.0f));\n"
                      "      max_",
                      h, " = nmax_", h,
                      ";\n"
                      "      acc0_",
                      h, " = acc0_", h, " * alpha_", h, ";\n");
      if (has_second_slice) {
        absl::StrAppend(&code, "      acc1_", h, " = acc1_", h, " * alpha_", h,
                        ";\n");
      }
    }

    absl::StrAppend(&code, "      if (tid < args.head_slices) {\n",
                    emit_v_acc_slice("tid", "acc0"), "      }\n");
    if (has_second_slice) {
      absl::StrAppend(&code, "      if (tid + 32 < args.head_slices) {\n",
                      emit_v_acc_slice("(tid + 32)", "acc1"), "      }\n");
    }
    code += "    }\n  }\n";

    if (splits > 1) {
      absl::StrAppend(&code, "  int part_x = q_pos * ", splits,
                      " + split_id;\n");
      for (int h = 0; h < heads_per_wg; ++h) {
        absl::StrAppend(&code, "  {\n    int q_head = q_head0 + ", h,
                        ";\n"
                        "    if (tid < args.head_slices) {\n"
                        "      args.part.Write(acc0_",
                        h,
                        ", part_x, q_head, tid, 0);\n"
                        "    }\n");
        if (has_second_slice) {
          absl::StrAppend(&code,
                          "    if (tid + 32 < args.head_slices) {\n"
                          "      args.part.Write(acc1_",
                          h,
                          ", part_x, q_head, tid + 32, 0);\n"
                          "    }\n");
        }
        absl::StrAppend(&code,
                        "    if (tid == 0) {\n"
                        "      float4 ml_out = ucl::Init<float4>(max_",
                        h, ", sum_", h,
                        ", 0.0f, 0.0f);\n"
                        "      args.part.Write(ml_out, part_x, q_head, "
                        "args.head_slices, 0);\n"
                        "    }\n  }\n");
      }
    } else {
      for (int h = 0; h < heads_per_wg; ++h) {
        absl::StrAppend(
            &code, "  {\n    int q_head = q_head0 + ", h,
            ";\n"
            "    float inv_sum = 0.0f;\n"
            "    if (sum_",
            h,
            " > 0.0f) {\n"
            "      inv_sum = 1.0f / sum_",
            h,
            ";\n"
            "    }\n"
            "    if (tid < args.head_slices) {\n"
            "      float4 out0 = acc0_",
            h,
            " * ucl::Init<float4>(inv_sum);\n"
            "      args.dst_tensor::type dst0 = "
            "ucl::Convert<args.dst_tensor::type>(out0);\n"
            "      args.dst_tensor.Write(dst0, q_head, q_pos, tid, 0);\n"
            "    }\n");
        if (has_second_slice) {
          absl::StrAppend(
              &code,
              "    if (tid + 32 < args.head_slices) {\n"
              "      float4 out1 = acc1_",
              h,
              " * ucl::Init<float4>(inv_sum);\n"
              "      args.dst_tensor::type dst1 = "
              "ucl::Convert<args.dst_tensor::type>(out1);\n"
              "      args.dst_tensor.Write(dst1, q_head, q_pos, tid + 32, 0);\n"
              "    }\n");
        }
        code += "  }\n";
      }
    }
    code += "}\n";
    code_ = std::move(code);
    // Upper bound on attended tokens: `prep_slices` slices x 4 block IDs x 4
    // tokens per block, capped by `cache_len`.
    const uint64_t attended_tokens =
        std::min(static_cast<uint64_t>(cache_len),
                 static_cast<uint64_t>(prep_slices) * 16);
    flops_ = static_cast<uint64_t>(num_q_heads) * seq_len * attended_tokens *
             (4ULL * head_dim + 4);
  }

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(32 * num_head_groups_, seq_len_ * splits_, 1);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

 private:
  int num_head_groups_;
  int seq_len_;
  int splits_;
};

// Combines `splits` partial unnormalized attention accumulators and `(max,
// sum)` pairs from `part` into `dst_tensor [1, seq_len, num_q_heads,
// head_dim]`, preserving `dst_tensor` naming so downstream elementwise ops
// (e.g. Qwen4 output gate multiplication) fuse into the combine write via
// `ResolveLinking`.
class SparseFlashDecodeCombineOp : public ::ml_drift::GPUOperation {
 public:
  SparseFlashDecodeCombineOp(int num_q_heads, int seq_len, int head_slices,
                             int splits)
      : num_q_heads_(num_q_heads),
        seq_len_(seq_len),
        wg_size_((head_slices <= 32) ? 32 : 64) {
    work_group_size_ = ::ml_drift::int3(wg_size_, 1, 1);
    args_.AddInt("head_slices", head_slices);

    std::string code = R"(
MAIN_FUNCTION($0) {
  int ds = ucl::GetLocalId<0>();
  int q_head = ucl::GetGroupId<1>();
  int q_pos = ucl::GetGroupId<2>();
  __local float ml_m[SPLITS];
  __local float ml_l[SPLITS];

  int part_base = q_pos * SPLITS;
  for (int sp = ds; sp < SPLITS; sp += WG_SIZE) {
    float4 ml = args.part.Read<float>(part_base + sp, q_head, args.head_slices, 0);
    ml_m[sp] = ml.x;
    ml_l[sp] = ml.y;
  }
  ucl::SyncThreads<WorkGroup, Local>();

  if (ds >= args.head_slices) {
    return;
  }

  float m_max = -3.0e38f;
  for (int sp = 0; sp < SPLITS; ++sp) {
    if (ml_l[sp] > 0.0f) {
      m_max = max(m_max, ml_m[sp]);
    }
  }
  float l_sum = 0.0f;
  float4 acc = ucl::Init<float4>(0.0f);
  for (int sp = 0; sp < SPLITS; ++sp) {
    float li = ml_l[sp];
    if (li > 0.0f) {
      float w = exp(ml_m[sp] - m_max);
      l_sum += w * li;
      acc += w * args.part.Read<float>(part_base + sp, q_head, ds, 0);
    }
  }

  float inv_l = 0.0f;
  if (l_sum > 0.0f) {
    inv_l = 1.0f / l_sum;
  }
  float4 out_val = acc * ucl::Init<float4>(inv_l);
  args.dst_tensor::type dst_val = ucl::Convert<args.dst_tensor::type>(out_val);
  args.dst_tensor.Write(dst_val, q_head, q_pos, ds, 0);
}
)";
    absl::StrReplaceAll(
        {{"SPLITS", absl::StrCat(splits)}, {"WG_SIZE", absl::StrCat(wg_size_)}},
        &code);
    code_ = std::move(code);
  }

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(wg_size_, num_q_heads_, seq_len_);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

 private:
  int num_q_heads_;
  int seq_len_;
  int wg_size_;
};

}  // namespace

absl::Status BuildSparseSdpaTransposedGpuGraph(
    const std::vector<uint32_t>& input_ids,
    const std::vector<uint32_t>& output_ids,
    const SparseSdpaTransposedAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() != 5 || output_ids.size() != 1) {
    return absl::InvalidArgumentError(
        "odml.sparse_sdpa_transposed expects 5 inputs and 1 output.");
  }
  if (attr.compress_ratio != 4 || attr.block_topk <= 0 || attr.k_ts_idx != 2 ||
      attr.v_ts_idx != 3) {
    return absl::UnimplementedError(
        "odml.sparse_sdpa_transposed GPU kernel requires compress_ratio == 4, "
        "block_topk > 0, k_ts_idx == 2, and v_ts_idx == 3.");
  }
  ABSL_ASSIGN_OR_RETURN(TensorHandle query,
                        model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle key_cache,
                        model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle value_cache,
                        model_builder->GetTensor(input_ids[2]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle topk,
                        model_builder->GetTensor(input_ids[3]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle mask,
                        model_builder->GetTensor(input_ids[4]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle dst,
                        model_builder->GetTensor(output_ids[0]));

  const ::ml_drift::BHWC q_shape = query.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC kc_shape = key_cache.tensor_desc.GetBHWCShape();
  if (q_shape.c <= 0 || q_shape.c > 256 || q_shape.c % 4 != 0 ||
      kc_shape.w <= 0 || kc_shape.w % 4 != 0 || kc_shape.h <= 0 ||
      q_shape.h < kc_shape.h || q_shape.h % kc_shape.h != 0) {
    return absl::UnimplementedError(
        "odml.sparse_sdpa_transposed GPU kernel has unsupported Q/KV shapes.");
  }

  const int num_q_heads = q_shape.h;
  const int num_kv_heads = kc_shape.h;
  const int seq_len = q_shape.w;
  const int head_dim = q_shape.c;
  const int head_slices = head_dim / 4;
  const int cache_len = kc_shape.w;
  const int gqa_ratio = num_q_heads / num_kv_heads;
  const int block_topk_slices = (attr.block_topk + 3) / 4;
  const int prep_slices = block_topk_slices + 2;
  const bool bool_mask =
      mask.tensor_desc.GetDataType() == ::ml_drift::DataType::kBool;

  ::ml_drift::TensorDescriptor prep_desc =
      ::ml_drift::CreateBhwcTensorDescriptor(
          ::ml_drift::DataType::kInt32, ::ml_drift::TensorStorageType::kBuffer,
          ::ml_drift::BHWC(1, 1, seq_len, prep_slices * 4));
  TensorHandle prep_blocks = model_builder->AddTensor(prep_desc);
  auto prep_op = std::make_unique<SparsePrepareBlocksOp>(
      seq_len, cache_len, attr.block_topk, bool_mask);
  prep_op->AddSrcTensor("topk", topk.tensor_desc);
  prep_op->AddSrcTensor("mask", mask.tensor_desc);
  prep_op->AddDstTensor("prep_blocks", prep_blocks.tensor_desc);
  model_builder->AddGpuOperation({topk, mask}, {prep_blocks},
                                 std::move(prep_op), "sparse_prepare_blocks");

  // When `from_cache_update` is true, `odml.cache_update` wrote K and V via
  // `WriteLinear(..., idx)`, addressing `(idx)` on BUFFER / IMAGE_BUFFER and
  // `(idx, 0)` on TEXTURE_2D. Binding K/V with `Layout::kLinear` makes 1-arg
  // `Read<float>(idx)` emit `GetPhysicalCoordsLinear` across all storages.
  ::ml_drift::TensorDescriptor kc_src_desc = key_cache.tensor_desc;
  ::ml_drift::TensorDescriptor vc_src_desc = value_cache.tensor_desc;
  if (attr.from_cache_update) {
    kc_src_desc.SetLayout(::ml_drift::Layout::kLinear);
    vc_src_desc.SetLayout(::ml_drift::Layout::kLinear);
  }

  const ::ml_drift::GpuInfo& gpu_info = model_builder->gpu_info();
  const bool limit_regs =
      head_slices > 32 && (gpu_info.IsApiOpenGl() ||
                           (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()));
  int heads_per_wg = 1;
  if (!limit_regs && gqa_ratio % 4 == 0) {
    heads_per_wg = 4;
  } else if (gqa_ratio % 2 == 0) {
    heads_per_wg = 2;
  }
  const int num_head_groups = num_q_heads / heads_per_wg;

  int splits = 1;
  if (seq_len <= 8 && prep_slices >= 4) {
    const int max_useful_splits = std::max(1, prep_slices / 2);
    const int target_splits = std::max(1, 128 / (num_head_groups * seq_len));
    splits = std::min({64, max_useful_splits, target_splits});
  }

  auto sdpa_op = std::make_unique<SparseSdpaOp>(
      gpu_info, num_q_heads, num_kv_heads, heads_per_wg, splits, seq_len,
      head_dim, cache_len, prep_slices, attr.scale, bool_mask,
      attr.from_cache_update);
  sdpa_op->AddSrcTensor("query", query.tensor_desc);
  sdpa_op->AddSrcTensor("key_cache", kc_src_desc);
  sdpa_op->AddSrcTensor("value_cache", vc_src_desc);
  sdpa_op->AddSrcTensor("prep_blocks", prep_blocks.tensor_desc);
  sdpa_op->AddSrcTensor("mask", mask.tensor_desc);

  if (splits > 1) {
    ::ml_drift::TensorDescriptor part_desc =
        ::ml_drift::CreateBhwcTensorDescriptor(
            ::ml_drift::DataType::kFloat32,
            ::ml_drift::TensorStorageType::kBuffer,
            ::ml_drift::BHWC(1, num_q_heads, seq_len * splits,
                             (head_slices + 1) * 4));
    TensorHandle part = model_builder->AddTensor(part_desc);
    sdpa_op->AddDstTensor("part", part.tensor_desc);
    model_builder->AddGpuOperation(
        {query, key_cache, value_cache, prep_blocks, mask}, {part},
        std::move(sdpa_op), "sparse_sdpa_split");

    auto combine_op = std::make_unique<SparseFlashDecodeCombineOp>(
        num_q_heads, seq_len, head_slices, splits);
    combine_op->AddSrcTensor("part", part.tensor_desc);
    combine_op->AddDstTensor("dst_tensor", dst.tensor_desc);
    model_builder->AddGpuOperation({part}, {dst}, std::move(combine_op),
                                   kSparseSdpaTransposedType);
  } else {
    sdpa_op->AddDstTensor("dst_tensor", dst.tensor_desc);
    model_builder->AddGpuOperation(
        {query, key_cache, value_cache, prep_blocks, mask}, {dst},
        std::move(sdpa_op), kSparseSdpaTransposedType);
  }
  return absl::OkStatus();
}

absl::Status CreateSparseSdpaTransposedFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  if (!node.operation.attributes.has_value()) {
    return absl::InvalidArgumentError(
        "Missing odml.sparse_sdpa_transposed attributes.");
  }
  const auto* attr = std::any_cast<SparseSdpaTransposedAttributes>(
      &node.operation.attributes);
  if (attr == nullptr) {
    return absl::InvalidArgumentError(
        "Invalid odml.sparse_sdpa_transposed attributes.");
  }
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* v : inputs) input_ids.push_back(v->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* v : outputs) output_ids.push_back(v->id);
  return BuildSparseSdpaTransposedGpuGraph(input_ids, output_ids, *attr,
                                           model_builder);
}

absl::Status CreateSparseSdpaTransposedFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (!node.attr.has_value()) {
    return absl::InvalidArgumentError(
        "Missing odml.sparse_sdpa_transposed attributes.");
  }
  const auto* attr =
      std::any_cast<SparseSdpaTransposedAttributes>(&node.attr);
  if (attr == nullptr) {
    return absl::InvalidArgumentError(
        "Invalid odml.sparse_sdpa_transposed attributes.");
  }
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* t : inputs) input_ids.push_back(t->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* t : outputs) output_ids.push_back(t->id);
  return BuildSparseSdpaTransposedGpuGraph(input_ids, output_ids, *attr,
                                           model_builder);
}

}  // namespace litert::ml_drift
