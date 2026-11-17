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

#include "ml_drift_delegate/delegate/composite/qsa_indexer_update_kernel.h"

#include <algorithm>
#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
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
#include "ml_drift_delegate/delegate/composite/qsa_indexer_update_parser.h"

namespace litert::ml_drift {
namespace {

using TensorHandle = ::ml_drift::GpuModelBuilder::TensorHandle;

// Builds the shader expression reading slice `ds` (`[ds * 4 .. ds * 4 + 3]`) of
// 1D `k_norm_weight` across TFLite 1D (`BHWC(D, 1, 1, 1)`) and unit-test
// (`BHWC(1, 1, D, 1)` or `BHWC(1, 1, 1, D)`) shapes.
absl::StatusOr<std::string> KNormWeightSliceExpr(
    const ::ml_drift::TensorDescriptor& desc, int idx_dim) {
  const ::ml_drift::BHWC shape = desc.GetBHWCShape();
  if (shape.b == idx_dim && shape.h == 1 && shape.w == 1 && shape.c == 1) {
    return "ucl::Init<float4>("
           "args.k_norm_weight.Read<float>(0, 0, 0, ds * 4 + 0).x, "
           "args.k_norm_weight.Read<float>(0, 0, 0, ds * 4 + 1).x, "
           "args.k_norm_weight.Read<float>(0, 0, 0, ds * 4 + 2).x, "
           "args.k_norm_weight.Read<float>(0, 0, 0, ds * 4 + 3).x)";
  }
  if (shape.b == 1 && shape.h == 1 && shape.w == idx_dim && shape.c == 1) {
    return "ucl::Init<float4>("
           "args.k_norm_weight.Read<float>(ds * 4 + 0, 0, 0, 0).x, "
           "args.k_norm_weight.Read<float>(ds * 4 + 1, 0, 0, 0).x, "
           "args.k_norm_weight.Read<float>(ds * 4 + 2, 0, 0, 0).x, "
           "args.k_norm_weight.Read<float>(ds * 4 + 3, 0, 0, 0).x)";
  }
  if (shape.b == 1 && shape.h == 1 && shape.w == 1 && shape.c == idx_dim) {
    return "args.k_norm_weight.Read<float>(0, 0, ds, 0)";
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "odml.qsa_indexer_update: unsupported k_norm_weight BHWC shape (",
      shape.b, ", ", shape.h, ", ", shape.w, ", ", shape.c, ")."));
}

// Reads and updates `new_ring_k` [1, R, D_idx] and `new_block_k_pool`
// [1, max_blocks, D_idx] in-place on GPU (`kGpuOptimizedInplace`), launching
// only `cand_blocks = min(max_blocks, seq_len / R + 2)` workgroups covering
// `[start_blk, end_blk)` (`2` workgroups on decode `S = 1`). Workgroup
// `local_b == 0` (`b = start_blk`) owns `new_ring_k` via `smem_ring`, keeping
// all `SyncThreads` barriers at top level for WebGPU WGSL static uniformity.
class QsaRingAndPoolUpdateOp : public ::ml_drift::GPUOperation {
 public:
  QsaRingAndPoolUpdateOp(int seq_len, int max_blocks, int idx_dim,
                         int rotary_dim, int compress_ratio, int mask_kv_len,
                         float rms_norm_eps, bool bool_mask,
                         const std::string& k_norm_expr)
      : cand_blocks_(
            std::max(1, std::min(max_blocks, seq_len / compress_ratio + 2))) {
    work_group_size_ = ::ml_drift::int3(32, 1, 1);
    args_.AddInt("seq_len", seq_len);
    args_.AddInt("max_blocks", max_blocks);
    args_.AddInt("idx_dim", idx_dim);
    args_.AddInt("idx_slices", idx_dim / 4);
    args_.AddInt("rotary_slices", rotary_dim / 4);
    args_.AddInt("half_rotary_slices", (rotary_dim / 2) / 4);
    args_.AddInt("compress_ratio", compress_ratio);
    args_.AddInt("mask_kv_len", mask_kv_len);
    args_.AddFloat("inv_compress_ratio",
                   1.0f / static_cast<float>(compress_ratio));
    args_.AddFloat("inv_idx_dim", 1.0f / static_cast<float>(idx_dim));
    args_.AddFloat("rms_norm_eps", rms_norm_eps);

    const std::string mask_diag_check = bool_mask ? R"(
      bool4 m4 = args.mask.Read<bool>(t, 0, m_slice, 0);
      bool tok_vis = m4.x;
      if (m_lane == 1) {
        tok_vis = m4.y;
      } else if (m_lane == 2) {
        tok_vis = m4.z;
      } else if (m_lane == 3) {
        tok_vis = m4.w;
      }
)" : R"(
      float4 m4 = args.mask.Read<float>(t, 0, m_slice, 0);
      float m_val = m4.x;
      if (m_lane == 1) {
        m_val = m4.y;
      } else if (m_lane == 2) {
        m_val = m4.z;
      } else if (m_lane == 3) {
        m_val = m4.w;
      }
      bool tok_vis = m_val > -1.0f;
)";

    std::string code = R"(
MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int local_b = ucl::GetGroupId<0>();
  __local float4 smem_ring[256];
  __local float4 smem_pooled[64];
  __local float smem_sq[32];
  __local int smem_valid[32];

  int pos0 = max(args.start_pos.Read<int>(0, 0, 0, 0).x, 0);
  int start_blk = pos0 / args.compress_ratio;
  int b = start_blk + local_b;

  int my_valid = 0;
  for (int t = tid; t < args.seq_len; t += 32) {
    int g = pos0 + t;
    if (g < args.mask_kv_len) {
      int m_slice = g / 4;
      int m_lane = g % 4;
)" + mask_diag_check + R"(
      if (tok_vis) {
        my_valid = max(my_valid, t + 1);
      }
    }
  }
  smem_valid[tid] = my_valid;

  // Only `local_b == 0` (`b == start_blk`) can ever need `tok_pos < pos0` or
  // read/write `args.new_ring_k`: snapshot incoming `ring_k` before mutation.
  if (local_b == 0) {
    for (int r = 0; r < args.compress_ratio; ++r) {
      for (int ds = tid; ds < args.idx_slices; ds += 32) {
        smem_ring[r * 64 + ds] = args.new_ring_k.Read<float>(r, 0, ds, 0);
      }
    }
  }
  ucl::SyncThreads<WorkGroup, Local>();

  if (tid < 8) {
    int v01 = max(smem_valid[tid], smem_valid[tid + 8]);
    int v23 = max(smem_valid[tid + 16], smem_valid[tid + 24]);
    smem_valid[tid] = max(v01, v23);
  }
  ucl::SyncThreads<WorkGroup, Local>();

  int valid_len = max(
      max(max(smem_valid[0], smem_valid[1]), max(smem_valid[2], smem_valid[3])),
      max(max(smem_valid[4], smem_valid[5]), max(smem_valid[6], smem_valid[7])));
  int end_pos = pos0 + valid_len;
  int end_blk = end_pos / args.compress_ratio;

  if (local_b == 0 && valid_len > 0) {
    int ring_blk = end_blk;
    int ring_base = ring_blk * args.compress_ratio;
    for (int r = 0; r < args.compress_ratio; ++r) {
      int tok_pos = ring_base + r;
      for (int ds = tid; ds < args.idx_slices; ds += 32) {
        float4 val = ucl::Init<float4>(0.0f);
        if (tok_pos >= pos0 && tok_pos < end_pos) {
          val = args.k_idx_new.Read<float>(tok_pos - pos0, 0, ds, 0);
        } else if (tok_pos < pos0 && ring_blk == start_blk) {
          val = smem_ring[r * 64 + ds];
        }
        args.new_ring_k::type out_r = ucl::Convert<args.new_ring_k::type>(val);
        args.new_ring_k.Write(out_r, r, 0, ds, 0);
      }
    }
  }

  bool completed_now =
      (b >= start_blk) && (b < end_blk) && (b < args.max_blocks);
  int blk_base = b * args.compress_ratio;
  float my_sq = 0.0f;
  if (completed_now) {
    // Mean-pool `compress_ratio` keys across `idx_slices` in parallel.
    for (int ds = tid; ds < args.idx_slices; ds += 32) {
      float4 acc = ucl::Init<float4>(0.0f);
      for (int r = 0; r < args.compress_ratio; ++r) {
        int tok_pos = blk_base + r;
        float4 kv;
        if (tok_pos >= pos0) {
          kv = args.k_idx_new.Read<float>(tok_pos - pos0, 0, ds, 0);
        } else {
          kv = smem_ring[r * 64 + ds];
        }
        acc += kv;
      }
      float4 p = acc * ucl::Init<float4>(args.inv_compress_ratio);
      smem_pooled[ds] = p;
      my_sq += dot(p, p);
    }
  }
  smem_sq[tid] = my_sq;
  ucl::SyncThreads<WorkGroup, Local>();

  if (tid < 8) {
    smem_sq[tid] += smem_sq[tid + 8] + smem_sq[tid + 16] + smem_sq[tid + 24];
  }
  ucl::SyncThreads<WorkGroup, Local>();

  if (completed_now) {
    float sq_sum = smem_sq[0] + smem_sq[1] + smem_sq[2] + smem_sq[3] +
                   smem_sq[4] + smem_sq[5] + smem_sq[6] + smem_sq[7];
    float inv_rms = rsqrt(sq_sum * args.inv_idx_dim + args.rms_norm_eps);
    for (int ds = tid; ds < args.idx_slices; ds += 32) {
      float4 norm_w = K_NORM_WEIGHT_EXPR;
      smem_pooled[ds] = smem_pooled[ds] * ucl::Init<float4>(inv_rms) * norm_w;
    }
  }
  ucl::SyncThreads<WorkGroup, Local>();

  if (completed_now) {
    for (int ds = tid; ds < args.idx_slices; ds += 32) {
      float4 out_val = smem_pooled[ds];
      if (ds < args.rotary_slices) {
        float4 c = args.block_cos.Read<float>(b, 0, ds, 0);
        float4 s = args.block_sin.Read<float>(b, 0, ds, 0);
        int pair_ds = ds - args.half_rotary_slices;
        if (ds < args.half_rotary_slices) {
          pair_ds = ds + args.half_rotary_slices;
        }
        float4 partner = smem_pooled[pair_ds];
        float4 rot_half = partner;
        if (ds < args.half_rotary_slices) {
          rot_half = -partner;
        }
        out_val = out_val * c + rot_half * s;
      }
      args.new_block_k_pool::type out_p =
          ucl::Convert<args.new_block_k_pool::type>(out_val);
      args.new_block_k_pool.Write(out_p, b, 0, ds, 0);
    }
  }
}
)";
    absl::StrReplaceAll({{"K_NORM_WEIGHT_EXPR", k_norm_expr}}, &code);
    code_ = std::move(code);
  }

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(32 * cand_blocks_, 1, 1);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

 private:
  int cand_blocks_;
};

// Scores `q_idx` against `new_block_k_pool` into `scores_tmp [1, S, max_blocks,
// 4]`. Bounds active blocks to `end_blk_bound = clamp((pos0 + seq_len) / R, 0,
// max_blocks)`, skips dot-products completely when `end_blk_bound <=
// block_topk` (writing `0.0f` for complete blocks and `-1.0e4f` otherwise), and
// preloads `q_idx` into `__local` memory with a single coalesced pass over
// `block_k_pool` when `end_blk_bound > block_topk`.
class QsaScoreOp : public ::ml_drift::GPUOperation {
 public:
  QsaScoreOp(int seq_len, int index_n_heads, int idx_dim, int max_blocks,
             int compress_ratio, int block_topk, float scale, bool bool_mask)
      : seq_len_(seq_len), max_blocks_(max_blocks) {
    work_group_size_ = ::ml_drift::int3(64, 1, 1);
    const int idx_slices = idx_dim / 4;
    const int q_vec_cnt = index_n_heads * idx_slices;
    const bool use_smem_q = (q_vec_cnt > 0 && q_vec_cnt <= 512);
    args_.AddInt("seq_len", seq_len);
    args_.AddInt("index_n_heads", index_n_heads);
    args_.AddInt("idx_slices", idx_slices);
    args_.AddInt("q_vec_cnt", q_vec_cnt);
    args_.AddInt("max_blocks", max_blocks);
    args_.AddInt("compress_ratio", compress_ratio);
    args_.AddInt("block_topk", block_topk);
    args_.AddFloat("scale", scale);

    const std::string mask_complete = bool_mask ? R"(
  bool4 m4 = args.mask.Read<bool>(q_pos, 0, b, 0);
  bool complete = m4.x && m4.y && m4.z && m4.w;
)" : R"(
  float4 m4 = args.mask.Read<float>(q_pos, 0, b, 0);
  bool complete = (m4.x == 0.0f) && (m4.y == 0.0f) &&
                  (m4.z == 0.0f) && (m4.w == 0.0f);
)";

    std::string dot_body;
    if (use_smem_q && index_n_heads <= 16) {
      for (int h = 0; h < index_n_heads; ++h) {
        absl::StrAppend(&dot_body, "      float dot_", h, " = 0.0f;\n");
      }
      absl::StrAppend(&dot_body,
                      "      for (int ds = 0; ds < args.idx_slices; ++ds) {\n"
                      "        float4 kv = args.block_k_pool.Read<float>(b, "
                      "0, ds, 0);\n");
      for (int h = 0; h < index_n_heads; ++h) {
        absl::StrAppend(&dot_body, "        dot_", h, " += dot(smem_q[",
                        h * idx_slices, " + ds], kv);\n");
      }
      absl::StrAppend(&dot_body, "      }\n      float sum_h = 0.0f;\n");
      for (int h = 0; h < index_n_heads; ++h) {
        absl::StrAppend(&dot_body, "      sum_h += max(dot_", h, ", 0.0f);\n");
      }
      absl::StrAppend(&dot_body, "      score = sum_h * args.scale;\n");
    } else if (use_smem_q) {
      dot_body = R"(
      float sum_h = 0.0f;
      int h = 0;
      for (; h + 3 < args.index_n_heads; h += 4) {
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        int base0 = (h + 0) * args.idx_slices;
        int base1 = (h + 1) * args.idx_slices;
        int base2 = (h + 2) * args.idx_slices;
        int base3 = (h + 3) * args.idx_slices;
        for (int ds = 0; ds < args.idx_slices; ++ds) {
          float4 kv = args.block_k_pool.Read<float>(b, 0, ds, 0);
          dot0 += dot(smem_q[base0 + ds], kv);
          dot1 += dot(smem_q[base1 + ds], kv);
          dot2 += dot(smem_q[base2 + ds], kv);
          dot3 += dot(smem_q[base3 + ds], kv);
        }
        sum_h += max(dot0, 0.0f) + max(dot1, 0.0f) +
                 max(dot2, 0.0f) + max(dot3, 0.0f);
      }
      for (; h < args.index_n_heads; ++h) {
        float dot_qk = 0.0f;
        int base_h = h * args.idx_slices;
        for (int ds = 0; ds < args.idx_slices; ++ds) {
          float4 kv = args.block_k_pool.Read<float>(b, 0, ds, 0);
          dot_qk += dot(smem_q[base_h + ds], kv);
        }
        sum_h += max(dot_qk, 0.0f);
      }
      score = sum_h * args.scale;
)";
    } else {
      dot_body = R"(
      float sum_h = 0.0f;
      int h = 0;
      for (; h + 3 < args.index_n_heads; h += 4) {
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        for (int ds = 0; ds < args.idx_slices; ++ds) {
          float4 kv = args.block_k_pool.Read<float>(b, 0, ds, 0);
          dot0 += dot(args.q_idx.Read<float>(h + 0, q_pos, ds, 0), kv);
          dot1 += dot(args.q_idx.Read<float>(h + 1, q_pos, ds, 0), kv);
          dot2 += dot(args.q_idx.Read<float>(h + 2, q_pos, ds, 0), kv);
          dot3 += dot(args.q_idx.Read<float>(h + 3, q_pos, ds, 0), kv);
        }
        sum_h += max(dot0, 0.0f) + max(dot1, 0.0f) +
                 max(dot2, 0.0f) + max(dot3, 0.0f);
      }
      for (; h < args.index_n_heads; ++h) {
        float dot_qk = 0.0f;
        for (int ds = 0; ds < args.idx_slices; ++ds) {
          float4 qv = args.q_idx.Read<float>(h, q_pos, ds, 0);
          float4 kv = args.block_k_pool.Read<float>(b, 0, ds, 0);
          dot_qk += dot(qv, kv);
        }
        sum_h += max(dot_qk, 0.0f);
      }
      score = sum_h * args.scale;
)";
    }

    std::string smem_preload;
    if (use_smem_q) {
      smem_preload =
          absl::StrCat("  __local float4 smem_q[", std::max(1, q_vec_cnt), R"(];
  int tid = ucl::GetLocalId<0>();
  int group_b0 = ucl::GetGroupId<0>() * 64;
  if (q_pos < args.seq_len && group_b0 < end_blk_bound &&
      end_blk_bound > args.block_topk) {
    for (int idx = tid; idx < args.q_vec_cnt; idx += 64) {
      int h = idx / args.idx_slices;
      int ds = idx % args.idx_slices;
      smem_q[idx] = args.q_idx.Read<float>(h, q_pos, ds, 0);
    }
  }
  ucl::SyncThreads<WorkGroup, Local>();
)");
    }

    code_ = absl::StrCat(R"(
MAIN_FUNCTION($0) {
  int b = ucl::GetGlobalId<0>();
  int q_pos = ucl::GetGlobalId<1>();
  int pos0 = max(args.start_pos.Read<int>(0, 0, 0, 0).x, 0);
  int end_blk_bound =
      clamp((pos0 + args.seq_len) / args.compress_ratio, 0, args.max_blocks);
)",
                         smem_preload, R"(
  if (b >= end_blk_bound || q_pos >= args.seq_len) {
    return;
  }
)",
                         mask_complete, R"(
  float score = -1.0e4f;
  if (complete) {
    if (end_blk_bound <= args.block_topk) {
      score = 0.0f;
    } else {
)",
                         dot_body, R"(    }
  }
  float4 score_out = ucl::Init<float4>(0.0f);
  score_out.x = score;
  args.scores_tmp.Write(score_out, b, q_pos, 0, 0);
}
)");
  }

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(max_blocks_, seq_len_, 1);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

 private:
  int seq_len_;
  int max_blocks_;
};

// Selects top-K complete blocks per query row `q_pos` using a 256-thread (or
// 64-thread on OpenGL ES devices with <256 max invocations) workgroup with
// exact IEEE-754 order-preserving `int` threshold binary search, `__local`
// ordinal caching (`smem_buf[2560]`), 2-barrier `R x R` reductions, and
// atomic-free deterministic prefix compaction (`block_topk <= 2048`).
class QsaSelectTopkOp : public ::ml_drift::GPUOperation {
 public:
  QsaSelectTopkOp(const ::ml_drift::GpuInfo& gpu_info, int seq_len,
                  int max_blocks, int compress_ratio, int block_topk)
      : seq_len_(seq_len) {
    const int max_wg_x = gpu_info.GetMaxWorkGroupSizeForX();
    const bool allow_256 =
        (max_wg_x == 0) ? !gpu_info.IsApiOpenGl() : (max_wg_x >= 256);
    wg_size_ = allow_256 ? 256 : 64;
    const int wg_radix = allow_256 ? 16 : 8;
    const int cached_ords = std::min(max_blocks, 2560);
    const int smem_buf_len =
        std::max({4, cached_ords, ((block_topk + 3) / 4) * 4});

    work_group_size_ = ::ml_drift::int3(wg_size_, 1, 1);
    args_.AddInt("seq_len", seq_len);
    args_.AddInt("max_blocks", max_blocks);
    args_.AddInt("compress_ratio", compress_ratio);
    args_.AddInt("block_topk", block_topk);
    args_.AddInt("block_topk_slices", (block_topk + 3) / 4);

    std::string code = R"(
MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int q_pos = ucl::GetGroupId<1>();
  __local int smem_buf[SMEM_BUF_LEN];
  __local int smem_cnt[WG_SIZE];
  __local int smem_eq[WG_SIZE];
  __local int smem_lo[WG_SIZE];
  __local int smem_hi[WG_SIZE];
  __local int smem_red_cnt[WG_RADIX];
  __local int smem_red_lo[WG_RADIX];
  __local int smem_red_hi[WG_RADIX];

  int pos0 = max(args.start_pos.Read<int>(0, 0, 0, 0).x, 0);
  int end_blk_bound =
      clamp((pos0 + args.seq_len) / args.compress_ratio, 0, args.max_blocks);
  int cached_end = min(end_blk_bound, CACHED_ORDS);

  int local_cnt = 0;
  int local_lo = 2147483647;
  int local_hi = -2147483647;
  for (int b = tid; b < end_blk_bound; b += WG_SIZE) {
    float s = args.scores_tmp.Read<float>(b, q_pos, 0, 0).x;
    int ord = -2147483647;
    if (s > -9999.0f) {
      if (s >= 0.0f) {
        float sp = s + 0.0f;
        ord = ucl::Reinterpret<float, int>(sp);
      } else {
        float sn = -s;
        ord = -ucl::Reinterpret<float, int>(sn);
      }
      local_cnt += 1;
      local_lo = min(local_lo, ord);
      local_hi = max(local_hi, ord);
    }
    if (b < CACHED_ORDS) {
      smem_buf[b] = ord;
    }
  }
  smem_cnt[tid] = local_cnt;
  smem_lo[tid] = local_lo;
  smem_hi[tid] = local_hi;
  ucl::SyncThreads<WorkGroup, Local>();

  if (tid < WG_RADIX) {
    int base_r = tid * WG_RADIX;
    int c_acc = 0;
    int lo_acc = 2147483647;
    int hi_acc = -2147483647;
    for (int i = 0; i < WG_RADIX; ++i) {
      c_acc += smem_cnt[base_r + i];
      lo_acc = min(lo_acc, smem_lo[base_r + i]);
      hi_acc = max(hi_acc, smem_hi[base_r + i]);
    }
    smem_red_cnt[tid] = c_acc;
    smem_red_lo[tid] = lo_acc;
    smem_red_hi[tid] = hi_acc;
  }
  ucl::SyncThreads<WorkGroup, Local>();

  int valid_cnt = 0;
  int lo_ord = 2147483647;
  int hi_ord = -2147483647;
  for (int i = 0; i < WG_RADIX; ++i) {
    valid_cnt += smem_red_cnt[i];
    lo_ord = min(lo_ord, smem_red_lo[i]);
    hi_ord = max(hi_ord, smem_red_hi[i]);
  }
  int target_k = min(valid_cnt, args.block_topk);

  // `end_blk_bound` is derived solely from read-only storage `start_pos` and
  // uniform `args`, which is WGSL-uniform; `SyncThreads` stays outside any
  // `__local`-gated branch (`need_step`).
  if (end_blk_bound > args.block_topk) {
    for (int iter = 0; iter < 32; ++iter) {
      bool need_step = (valid_cnt > args.block_topk) && (lo_ord < hi_ord);
      int mid_ord = 0;
      int gt_local = 0;
      if (need_step) {
        if ((lo_ord >= 0 && hi_ord >= 0) || (lo_ord < 0 && hi_ord < 0)) {
          mid_ord = lo_ord + (hi_ord - lo_ord) / 2;
        } else {
          int sum_lh = lo_ord + hi_ord;
          if (sum_lh >= 0) {
            mid_ord = sum_lh / 2;
          } else {
            mid_ord = (sum_lh - 1) / 2;
          }
        }
        for (int b = tid; b < cached_end; b += WG_SIZE) {
          int ord = smem_buf[b];
          if (ord > -2000000000 && ord > mid_ord) {
            gt_local += 1;
          }
        }
        for (int b = CACHED_ORDS + tid; b < end_blk_bound; b += WG_SIZE) {
          float s = args.scores_tmp.Read<float>(b, q_pos, 0, 0).x;
          if (s > -9999.0f) {
            int ord = 0;
            if (s >= 0.0f) {
              float sp = s + 0.0f;
              ord = ucl::Reinterpret<float, int>(sp);
            } else {
              float sn = -s;
              ord = -ucl::Reinterpret<float, int>(sn);
            }
            if (ord > mid_ord) {
              gt_local += 1;
            }
          }
        }
      }
      smem_cnt[tid] = gt_local;
      ucl::SyncThreads<WorkGroup, Local>();
      if (tid < WG_RADIX) {
        int base_r = tid * WG_RADIX;
        int c_acc = 0;
        for (int i = 0; i < WG_RADIX; ++i) {
          c_acc += smem_cnt[base_r + i];
        }
        smem_red_cnt[tid] = c_acc;
      }
      ucl::SyncThreads<WorkGroup, Local>();
      if (need_step) {
        int total_gt_iter = 0;
        for (int i = 0; i < WG_RADIX; ++i) {
          total_gt_iter += smem_red_cnt[i];
        }
        if (total_gt_iter >= target_k) {
          lo_ord = mid_ord + 1;
        } else {
          hi_ord = mid_ord;
        }
      }
    }
  }

  int thresh_ord = lo_ord;
  int chunk_sz = (end_blk_bound + WG_SIZE - 1) / WG_SIZE;
  int b_start = tid * chunk_sz;
  int b_end = min(end_blk_bound, b_start + chunk_sz);

  int my_gt = 0;
  int my_eq = 0;
  if (valid_cnt > 0) {
    for (int b = b_start; b < b_end; ++b) {
      float s = args.scores_tmp.Read<float>(b, q_pos, 0, 0).x;
      if (s > -9999.0f) {
        int ord = 0;
        if (s >= 0.0f) {
          float sp = s + 0.0f;
          ord = ucl::Reinterpret<float, int>(sp);
        } else {
          float sn = -s;
          ord = -ucl::Reinterpret<float, int>(sn);
        }
        if (ord > thresh_ord) {
          my_gt += 1;
        } else if (ord == thresh_ord) {
          my_eq += 1;
        }
      }
    }
  }
  smem_cnt[tid] = my_gt;
  smem_eq[tid] = my_eq;
  int out_slots = args.block_topk_slices * 4;
  for (int i = tid; i < out_slots; i += WG_SIZE) {
    smem_buf[i] = -1;
  }
  ucl::SyncThreads<WorkGroup, Local>();

  if (tid == 0) {
    int total_gt = 0;
    for (int t = 0; t < WG_SIZE; ++t) {
      total_gt += smem_cnt[t];
    }
    int tie_rem = max(target_k - total_gt, 0);
    int wr_offset = 0;
    for (int t = 0; t < WG_SIZE; ++t) {
      int gt_t = smem_cnt[t];
      int eq_t = min(tie_rem, smem_eq[t]);
      tie_rem -= eq_t;
      smem_cnt[t] = wr_offset;
      smem_eq[t] = eq_t;
      wr_offset += gt_t + eq_t;
    }
  }
  ucl::SyncThreads<WorkGroup, Local>();

  if (valid_cnt > 0) {
    int wr = smem_cnt[tid];
    int eq_left = smem_eq[tid];
    for (int b = b_start; b < b_end; ++b) {
      float s = args.scores_tmp.Read<float>(b, q_pos, 0, 0).x;
      if (s > -9999.0f) {
        int ord = 0;
        if (s >= 0.0f) {
          float sp = s + 0.0f;
          ord = ucl::Reinterpret<float, int>(sp);
        } else {
          float sn = -s;
          ord = -ucl::Reinterpret<float, int>(sn);
        }
        if (ord > thresh_ord) {
          smem_buf[wr] = b;
          wr += 1;
        } else if (ord == thresh_ord && eq_left > 0) {
          smem_buf[wr] = b;
          wr += 1;
          eq_left -= 1;
        }
      }
    }
  }
  ucl::SyncThreads<WorkGroup, Local>();

  for (int bs = tid; bs < args.block_topk_slices; bs += WG_SIZE) {
    int base_o = bs * 4;
    int4 out_vec = ucl::Init<int4>(smem_buf[base_o + 0], smem_buf[base_o + 1],
                                   smem_buf[base_o + 2], smem_buf[base_o + 3]);
    args.topk.Write(out_vec, q_pos, 0, bs, 0);
  }
}
)";
    absl::StrReplaceAll({{"SMEM_BUF_LEN", absl::StrCat(smem_buf_len)},
                         {"CACHED_ORDS", absl::StrCat(cached_ords)},
                         {"WG_SIZE", absl::StrCat(wg_size_)},
                         {"WG_RADIX", absl::StrCat(wg_radix)}},
                        &code);
    code_ = std::move(code);
  }

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(wg_size_, seq_len_, 1);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

 private:
  int seq_len_;
  int wg_size_;
};

}  // namespace

absl::Status BuildQsaIndexerUpdateGpuGraph(
    const std::vector<uint32_t>& input_ids,
    const std::vector<uint32_t>& output_ids,
    const QsaIndexerUpdateAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() != 7 || output_ids.size() != 3) {
    return absl::InvalidArgumentError(
        "odml.qsa_indexer_update expects 7 GPU inputs and 3 outputs.");
  }
  if (attr.compress_ratio != 4 || attr.block_topk <= 0 ||
      attr.block_topk > 2048) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update GPU kernel requires compress_ratio == 4 and "
        "block_topk in (0, 2048].");
  }

  ABSL_ASSIGN_OR_RETURN(TensorHandle q_idx,
                        model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle k_idx_new,
                        model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle k_norm_weight,
                        model_builder->GetTensor(input_ids[2]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle block_cos,
                        model_builder->GetTensor(input_ids[3]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle block_sin,
                        model_builder->GetTensor(input_ids[4]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle start_pos,
                        model_builder->GetTensor(input_ids[5]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle mask,
                        model_builder->GetTensor(input_ids[6]));

  ABSL_ASSIGN_OR_RETURN(TensorHandle topk_out,
                        model_builder->GetTensor(output_ids[0]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle new_ring_k,
                        model_builder->GetTensor(output_ids[1]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle new_block_k_pool,
                        model_builder->GetTensor(output_ids[2]));

  if (new_ring_k.tensor_desc.GetStorageType() !=
          ::ml_drift::TensorStorageType::kBuffer ||
      new_block_k_pool.tensor_desc.GetStorageType() !=
          ::ml_drift::TensorStorageType::kBuffer) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update requires BUFFER storage for new_ring_k and "
        "new_block_k_pool.");
  }

  const ::ml_drift::BHWC q_shape = q_idx.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC pool_shape =
      new_block_k_pool.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC cos_shape = block_cos.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC mask_shape = mask.tensor_desc.GetBHWCShape();
  const int seq_len = q_shape.h;
  const int index_n_heads = q_shape.w;
  const int idx_dim = q_shape.c;
  const int max_blocks = pool_shape.w;
  const int rotary_dim = cos_shape.c;
  const int mask_kv_len = mask_shape.c;
  if (idx_dim <= 0 || idx_dim > 256 || idx_dim % 4 != 0 || rotary_dim <= 0 ||
      rotary_dim > idx_dim || rotary_dim % 8 != 0) {
    return absl::UnimplementedError(
        "odml.qsa_indexer_update GPU kernel requires idx_dim in (0, 256] "
        "divisible by 4 and rotary_dim divisible by 8.");
  }

  ABSL_ASSIGN_OR_RETURN(
      const std::string k_norm_expr,
      KNormWeightSliceExpr(k_norm_weight.tensor_desc, idx_dim));

  const bool bool_mask =
      mask.tensor_desc.GetDataType() == ::ml_drift::DataType::kBool;
  auto update_op = std::make_unique<QsaRingAndPoolUpdateOp>(
      seq_len, max_blocks, idx_dim, rotary_dim, attr.compress_ratio,
      mask_kv_len, attr.rms_norm_eps, bool_mask, k_norm_expr);
  update_op->AddSrcTensor("k_idx_new", k_idx_new.tensor_desc);
  update_op->AddSrcTensor("k_norm_weight", k_norm_weight.tensor_desc);
  update_op->AddSrcTensor("block_cos", block_cos.tensor_desc);
  update_op->AddSrcTensor("block_sin", block_sin.tensor_desc);
  update_op->AddSrcTensor("start_pos", start_pos.tensor_desc);
  update_op->AddSrcTensor("mask", mask.tensor_desc);
  update_op->AddDstTensor("new_ring_k", new_ring_k.tensor_desc);
  update_op->AddDstTensor("new_block_k_pool", new_block_k_pool.tensor_desc);
  model_builder->AddGpuOperation(
      {k_idx_new, k_norm_weight, block_cos, block_sin, start_pos, mask},
      {new_ring_k, new_block_k_pool}, std::move(update_op),
      "qsa_ring_and_pool_update");

  ::ml_drift::TensorDescriptor scores_desc =
      ::ml_drift::CreateBhwcTensorDescriptor(
          ::ml_drift::DataType::kFloat32,
          ::ml_drift::TensorStorageType::kBuffer,
          ::ml_drift::BHWC(1, seq_len, max_blocks, 4));
  TensorHandle scores_tmp = model_builder->AddTensor(scores_desc);
  auto score_op = std::make_unique<QsaScoreOp>(
      seq_len, index_n_heads, idx_dim, max_blocks, attr.compress_ratio,
      attr.block_topk, attr.scale, bool_mask);
  score_op->AddSrcTensor("q_idx", q_idx.tensor_desc);
  score_op->AddSrcTensor("block_k_pool", new_block_k_pool.tensor_desc);
  score_op->AddSrcTensor("start_pos", start_pos.tensor_desc);
  score_op->AddSrcTensor("mask", mask.tensor_desc);
  score_op->AddDstTensor("scores_tmp", scores_tmp.tensor_desc);
  model_builder->AddGpuOperation({q_idx, new_block_k_pool, start_pos, mask},
                                 {scores_tmp}, std::move(score_op),
                                 "qsa_score");

  auto topk_op = std::make_unique<QsaSelectTopkOp>(
      model_builder->gpu_info(), seq_len, max_blocks, attr.compress_ratio,
      attr.block_topk);
  topk_op->AddSrcTensor("scores_tmp", scores_tmp.tensor_desc);
  topk_op->AddSrcTensor("start_pos", start_pos.tensor_desc);
  topk_op->AddDstTensor("topk", topk_out.tensor_desc);
  model_builder->AddGpuOperation({scores_tmp, start_pos}, {topk_out},
                                 std::move(topk_op), "qsa_select_topk");
  return absl::OkStatus();
}

absl::Status CreateQsaIndexerUpdateFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  if (!node.operation.attributes.has_value()) {
    return absl::InvalidArgumentError(
        "Missing odml.qsa_indexer_update attributes.");
  }
  const auto* attr =
      std::any_cast<QsaIndexerUpdateAttributes>(&node.operation.attributes);
  if (attr == nullptr) {
    return absl::InvalidArgumentError(
        "Invalid odml.qsa_indexer_update attributes.");
  }
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* v : inputs) input_ids.push_back(v->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* v : outputs) output_ids.push_back(v->id);
  return BuildQsaIndexerUpdateGpuGraph(input_ids, output_ids, *attr,
                                       model_builder);
}

absl::Status CreateQsaIndexerUpdateFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (!node.attr.has_value()) {
    return absl::InvalidArgumentError(
        "Missing odml.qsa_indexer_update attributes.");
  }
  const auto* attr = std::any_cast<QsaIndexerUpdateAttributes>(&node.attr);
  if (attr == nullptr) {
    return absl::InvalidArgumentError(
        "Invalid odml.qsa_indexer_update attributes.");
  }
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* t : inputs) input_ids.push_back(t->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* t : outputs) output_ids.push_back(t->id);
  return BuildQsaIndexerUpdateGpuGraph(input_ids, output_ids, *attr,
                                       model_builder);
}

}  // namespace litert::ml_drift
