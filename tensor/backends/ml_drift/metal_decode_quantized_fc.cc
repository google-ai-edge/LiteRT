/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/ml_drift/metal_decode_quantized_fc.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/task/util.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift

namespace litert::tensor {
namespace {

int GetEnvInt(const char* name) {
  const char* value = std::getenv(name);
  int parsed = 0;
  if (value == nullptr || !absl::SimpleAtoi(value, &parsed)) return 0;
  return parsed;
}

bool MetalDecodeFcEnabled(const ml_drift::GpuInfo& gpu_info) {
  const char* env = std::getenv("LRT_METAL_DECODE_FC");
  if (env != nullptr && env[0] != '\0') {
    const absl::string_view sv(env);
    if (sv == "0" || sv == "false") return false;
    if (sv == "1" || sv == "true") return true;
  }
  return gpu_info.IsApple() && gpu_info.IsApiMetal();
}

int NextPowerOfTwo(int v) {
  int p = 1;
  while (p < v && p < 1024) p *= 2;
  return p;
}

bool IsApple9OrOlderBionic(const ml_drift::GpuInfo& gpu_info) {
  return gpu_info.IsApple() && gpu_info.apple_info.IsBionic() &&
         gpu_info.apple_info.IsFamilyOrLower(
             ml_drift::AppleInfo::Family::kApple9);
}

// Selects how many adjacent output slices (`dst_s`) each thread computes from a
// single vectorized `uint`/`uint2`/`uint4` weight load. INT4 (`uint2` = 8B) and
// INT8 (`uint4` = 16B) already saturate 128B cache lines at `outs = 1` without
// doubling register pressure; only sub-word INT2 (`uint` = 4B) widens to 2 or 4
// slices on wide layers (`outs = 2` on A17 Pro / Apple9 and older mobile to
// keep `lm_head` register occupancy high).
int SelectOutsPerThread(const ml_drift::GpuInfo& gpu_info,
                        ml_drift::DataType storage_type, int dst_slices,
                        int cu_count) {
  const int override_outs = GetEnvInt("LRT_METAL_DECODE_FC_OUTS");
  const int max_outs = storage_type == ml_drift::DataType::kUint2 ? 4 : 1;
  if (override_outs == 1 || override_outs == 2 || override_outs == 4) {
    const int clamped = std::min(override_outs, max_outs);
    if (dst_slices % clamped == 0) return clamped;
  }
  if (storage_type != ml_drift::DataType::kUint2) return 1;
  if (!IsApple9OrOlderBionic(gpu_info) && dst_slices >= 1024 * 16 &&
      dst_slices % 4 == 0) {
    return 4;
  }
  if (dst_slices % 2 == 0 && dst_slices >= 2 * 32 * cu_count * 2) {
    return 2;
  }
  return 1;
}

// Unified Apple Metal workgroup sizing matching ML Drift's proven
// `task_size_per_cu` split-K occupancy curve (`wg_total = 256` on `cu >= 16`,
// `128` on `cu < 16`), rebalanced against `NextPowerOfTwo(max_y_size)` without
// ever dropping below a full 32-lane SIMD group.
ml_drift::int3 SelectWorkGroupSize(const ml_drift::GpuInfo& gpu_info,
                                   ml_drift::DataType storage_type,
                                   int dst_slices, int x_tasks, int max_y_size,
                                   int outs_per_thread) {
  const int cu_count = std::max(1, gpu_info.GetComputeUnitsCount());
  const int max_wg_threads =
      outs_per_thread >= 4 ? 128 : (cu_count >= 16 ? 256 : 128);
  // Allow `wg_y` up to `NextPowerOfTwo(max_y_size)` so e.g. `src_groups = 48`
  // completes in 1 iteration (`wg_y = 64`) instead of 2 imbalanced iterations.
  const int max_y_pow2 = NextPowerOfTwo(std::max(1, max_y_size));
  const int y_limit = std::max(1, std::min({max_y_pow2, 128, max_wg_threads}));

  const int env_x = GetEnvInt("LRT_METAL_DECODE_FC_WG_X");
  const int env_y = GetEnvInt("LRT_METAL_DECODE_FC_WG_Y");
  if (env_x >= 1 && env_y >= 1 && env_x * env_y <= 256 &&
      (env_y & (env_y - 1)) == 0 && env_y <= y_limit) {
    return ml_drift::int3(env_x, env_y, 1);
  }

  // Very wide outputs (e.g. `lm_head`): zero-barrier `(128|256, 1, 1)`.
  if (dst_slices >= 1024 * 16) {
    return ml_drift::int3(max_wg_threads, 1, 1);
  }

  const double tasks_per_cu = static_cast<double>(x_tasks) / cu_count;
  const int effective_bits =
      ml_drift::SizeInBitsOf(storage_type) * outs_per_thread;
  const double mult = effective_bits <= 8 ? (16.0 / effective_bits) : 1.0;
  int wg_y = 1;
  if (tasks_per_cu <= 2.0 * mult) {
    wg_y = 128;
  } else if (tasks_per_cu <= 4.0 * mult) {
    wg_y = 64;
  } else if (tasks_per_cu <= 8.0 * mult) {
    wg_y = 32;
  } else if (tasks_per_cu <= 16.0 * mult) {
    wg_y = 16;
  } else if (tasks_per_cu <= 32.0 * mult) {
    wg_y = 8;
  } else if (tasks_per_cu <= 64.0 * mult) {
    wg_y = 4;
  } else if (tasks_per_cu <= 128.0 * mult) {
    wg_y = 2;
  }
  // On mobile Apple GPUs (`cu < 16`), keep `wg_x >= 16` (128B full cache line
  // for INT4/INT2 `uint2` loads) and single-barrier `wg_y <= 8` whenever we
  // still have >= `4 * cu_count` workgroups (`16` SIMD groups/CU).
  if (cu_count < 16 && wg_y > 8 && x_tasks >= 16 * 4 * cu_count) {
    wg_y = 8;
  }
  wg_y = std::min(wg_y, max_wg_threads);
  int wg_x = max_wg_threads / wg_y;
  if (wg_y > y_limit) {
    wg_y = y_limit;
    while (wg_x * 2 <= x_tasks && wg_x * wg_y * 2 <= max_wg_threads &&
           (wg_x * wg_y < 32 || x_tasks >= wg_x * 2 * cu_count)) {
      wg_x *= 2;
    }
  }
  while (
      (wg_x > x_tasks || (x_tasks / wg_x < cu_count && wg_y * 2 <= y_limit)) &&
      wg_x > 1 && wg_y * 2 <= max_wg_threads) {
    wg_x /= 2;
    wg_y *= 2;
  }
  return ml_drift::int3(wg_x, wg_y, 1);
}

// INT2 unpack mode:
// - `0` (default): FP-pipe `floor` without inner-loop `* 4.0f` (1 convert,
//   0 integer ops; exact 4x factor folded into hoisted output scale).
// - `1`: integer bit-mask + 4x `uchar4` converts (`LRT_METAL_DECODE_FC_INT2=1`
//   for A/B testing).
bool UseInt2BitmaskUnpack() {
  return GetEnvInt("LRT_METAL_DECODE_FC_INT2") == 1;
}

// Maximum K-slices accumulated in FP16 before flushing scale/bias into `r`.
// Bounding each chunk to 8 slices (32 ch) for INT4/INT8 and 32 slices (128 ch)
// for INT2 keeps peak unscaled sums <= `480 * |v|` (INT4), `384 * |v|` (INT2),
// and `256 * |v|` (INT8 with 1/16 prescale), preventing FP16 overflow and
// catastrophic cancellation at arbitrary `K` and large activation magnitudes.
int ChunkSlices(ml_drift::DataType storage_type) {
  return storage_type == ml_drift::DataType::kUint2 ? 32 : 8;
}

// Power-of-2 prescaling applied to `v` inside the K-loop when accumulating in
// FP16 so unscaled quantized sums stay well inside `half` range (max 65504)
// while preserving exact FP16 mantissa bits (exponent shift only).
int InputPrescaleShift(ml_drift::DataType src_dtype,
                       ml_drift::DataType storage_type) {
  if (src_dtype != ml_drift::DataType::kFloat16) return 0;
  // INT8 (`[-128, 127]` over 32 channels/chunk) uses 1/16 prescale so even
  // uniform `|v| = 100` stays at `25600 < 65504`; INT4 and INT2 are bounded
  // by chunk size (`<= 480 * |v|` / `<= 384 * |v|`) with zero prescaling.
  return storage_type == ml_drift::DataType::kUint8 ? 4 : 0;
}

void AppendUnpackAccumulate(ml_drift::DataType storage_type, bool int2_bitmask,
                            const std::string& w_expr, int out_idx,
                            const std::string& acc_prefix, std::string* code) {
  if (storage_type == ml_drift::DataType::kUint2) {
    if (int2_bitmask) {
      // Integer bit-mask path: each byte `i` of `uint w` stores 4 x 2-bit
      // weights `(o0, o1, o2, o3)` for input channel `i`. Accumulate each
      // channel immediately to keep live temporaries to 1 `half4`.
      absl::StrAppend(
          code, "      {\n", "        uint w_u = ", w_expr, ";\n", "        ",
          acc_prefix, out_idx,
          ".x += dot(vs, ucl::Convert<Type>("
          "ucl::Reinterpret<uint, uchar4>(w_u & 0x03030303u)));\n",
          "        ", acc_prefix, out_idx,
          ".y += dot(vs, ucl::Convert<Type>("
          "ucl::Reinterpret<uint, uchar4>((w_u >> 2u) & 0x03030303u)));\n",
          "        ", acc_prefix, out_idx,
          ".z += dot(vs, ucl::Convert<Type>("
          "ucl::Reinterpret<uint, uchar4>((w_u >> 4u) & 0x03030303u)));\n",
          "        ", acc_prefix, out_idx,
          ".w += dot(vs, ucl::Convert<Type>("
          "ucl::Reinterpret<uint, uchar4>((w_u >> 6u) & 0x03030303u)));\n",
          "      }\n");
    } else {
      // FP-pipe `floor` unpack with only 1 `uchar4` convert, 0 integer ops,
      // and NO `* 4.0f` inside the loop. Accumulating each output lane
      // immediately keeps only 2 `half4` temporaries (`t`, `f`) live.
      absl::StrAppend(code, "      {\n",
                      "        Type t = ucl::Convert<Type>("
                      "ucl::Reinterpret<uint, uchar4>(",
                      w_expr, ")) * ucl::Init<SType>(0.25f);\n",
                      "        Type f = floor(t);\n", "        ", acc_prefix,
                      out_idx, ".x += dot(vs, t - f);\n",
                      "        t = f * ucl::Init<SType>(0.25f);\n",
                      "        f = floor(t);\n", "        ", acc_prefix,
                      out_idx, ".y += dot(vs, t - f);\n",
                      "        t = f * ucl::Init<SType>(0.25f);\n",
                      "        f = floor(t);\n", "        ", acc_prefix,
                      out_idx, ".z += dot(vs, t - f);\n", "        ",
                      acc_prefix, out_idx, ".w += dot(vs, f);\n", "      }\n");
    }
  } else if (storage_type == ml_drift::DataType::kUint4) {
    // Apple FP-pipe `floor` unpack with only 2 `uchar4` converts, 0 integer
    // ops, and NO `* 16.0f` inside the loop. Processing `w_p.x` before `w_p.y`
    // and accumulating sequentially keeps live temporaries to 3 `half4`s and
    // emits clean FMA chains.
    absl::StrAppend(
        code, "      {\n", "        uint2 w_p = ", w_expr, ";\n",
        "        Type wt = ucl::Convert<Type>("
        "ucl::Reinterpret<uint, uchar4>(w_p.x)) * ucl::Init<SType>(0.0625f);\n",
        "        Type hi = floor(wt);\n", "        Type lo = wt - hi;\n",
        "        ", acc_prefix, "ev", out_idx, " += vs.x * lo.xy;\n",
        "        ", acc_prefix, "ev", out_idx, " += vs.y * lo.zw;\n",
        "        ", acc_prefix, "od", out_idx, " += vs.x * hi.xy;\n",
        "        ", acc_prefix, "od", out_idx, " += vs.y * hi.zw;\n",
        "        wt = ucl::Convert<Type>("
        "ucl::Reinterpret<uint, uchar4>(w_p.y)) * ucl::Init<SType>(0.0625f);\n",
        "        hi = floor(wt);\n", "        lo = wt - hi;\n", "        ",
        acc_prefix, "ev", out_idx, " += vs.z * lo.xy;\n", "        ",
        acc_prefix, "ev", out_idx, " += vs.w * lo.zw;\n", "        ",
        acc_prefix, "od", out_idx, " += vs.z * hi.xy;\n", "        ",
        acc_prefix, "od", out_idx, " += vs.w * hi.zw;\n", "      }\n");
  } else {
    // `uint4 w`: XORing with `0x80808080u` converts stored unsigned `q + 128`
    // back to signed `char4` `q in [-128, 127]` in a single instruction,
    // making symmetric INT8 accumulation zero-mean and eliminating the bias
    // correction term altogether.
    absl::StrAppend(code, "      {\n", "        uint4 w_q = ", w_expr, ";\n",
                    "        ", acc_prefix, out_idx,
                    " += vs.x * ucl::Convert<Type>("
                    "ucl::Reinterpret<uint, char4>(w_q.x ^ 0x80808080u));\n",
                    "        ", acc_prefix, out_idx,
                    " += vs.y * ucl::Convert<Type>("
                    "ucl::Reinterpret<uint, char4>(w_q.y ^ 0x80808080u));\n",
                    "        ", acc_prefix, out_idx,
                    " += vs.z * ucl::Convert<Type>("
                    "ucl::Reinterpret<uint, char4>(w_q.z ^ 0x80808080u));\n",
                    "        ", acc_prefix, out_idx,
                    " += vs.w * ucl::Convert<Type>("
                    "ucl::Reinterpret<uint, char4>(w_q.w ^ 0x80808080u));\n",
                    "      }\n");
  }
}

void AppendApplyScaleBias(ml_drift::DataType storage_type, bool int2_bitmask,
                          bool scale_is_fp32, int outs_per_thread,
                          int prescale_shift, bool is_blockwise,
                          bool has_zero_point, const std::string& acc_prefix,
                          const std::string& x_sum_name,
                          const std::string& dst_r_prefix,
                          const std::string& assign_op, std::string* code) {
  // INT8 (`^ 0x80808080u`) already removed its `128` storage offset during
  // unpack; INT4 and INT2 still hold unsigned `[0, 15]` / `[0, 3]` weights.
  const bool split_even_odd = storage_type == ml_drift::DataType::kUint4;
  const int range_shift =
      storage_type == ml_drift::DataType::kUint4
          ? 8
          : (storage_type == ml_drift::DataType::kUint2 ? 2 : 0);
  const bool needs_bias = range_shift != 0 || has_zero_point;
  const char* rescale_lit = prescale_shift == 4 ? "16.0f" : "4.0f";
  for (int u = 0; u < outs_per_thread; ++u) {
    const std::string dst_s_expr =
        u == 0 ? "base_dst_s" : absl::StrCat("(base_dst_s + ", u, ")");
    const std::string read_args =
        is_blockwise ? absl::StrCat(dst_s_expr, ", 0, g") : dst_s_expr;
    absl::StrAppend(code, "    {\n");
    if (prescale_shift > 0 && scale_is_fp32) {
      // Scale subnormal FP32 scales (`|s| < 6.1e-5`, e.g. Gemma 4 PLE INT8)
      // into normal FP16 range BEFORE narrowing to `half4`.
      absl::StrAppend(code,
                      "      Type ws = ucl::Convert<Type>("
                      "args.weights_scale.Read(",
                      read_args, ") * ", rescale_lit, ");\n");
    } else {
      absl::StrAppend(code,
                      "      Type ws = ucl::Convert<Type>("
                      "args.weights_scale.Read(",
                      read_args, "));\n");
      if (prescale_shift > 0) {
        absl::StrAppend(code, "      ws *= ucl::Init<SType>(", rescale_lit,
                        ");\n");
      }
    }
    std::string ws_acc_expr = "ws";
    if (storage_type == ml_drift::DataType::kUint4) {
      absl::StrAppend(
          code,
          "      Type ws_acc = ws * ucl::Init<Type>(16.0f, 1.0f, 16.0f, "
          "1.0f);\n");
      ws_acc_expr = "ws_acc";
    } else if (storage_type == ml_drift::DataType::kUint2 && !int2_bitmask) {
      absl::StrAppend(
          code,
          "      Type ws_acc = ws * ucl::Init<Type>(4.0f, 4.0f, 4.0f, "
          "1.0f);\n");
      ws_acc_expr = "ws_acc";
    }
    const std::string acc_vec =
        split_even_odd
            ? absl::StrCat("ucl::Init<Type>(", acc_prefix, "ev", u, ".x, ",
                           acc_prefix, "od", u, ".x, ", acc_prefix, "ev", u,
                           ".y, ", acc_prefix, "od", u, ".y)")
            : absl::StrCat(acc_prefix, u);
    if (!needs_bias) {
      absl::StrAppend(code, "      ", dst_r_prefix, u, " ", assign_op, " ",
                      acc_vec, " * ", ws_acc_expr, ";\n");
    } else {
      if (has_zero_point) {
        absl::StrAppend(code,
                        "      Type wzp = ucl::Convert<Type>("
                        "args.weights_zero_point.Read(",
                        read_args, "));\n");
        if (range_shift != 0) {
          absl::StrAppend(code, "      Type wb = -ws * (ucl::Init<Type>(",
                          range_shift, ".0f) + wzp);\n");
        } else {
          absl::StrAppend(code, "      Type wb = -ws * wzp;\n");
        }
      } else {
        absl::StrAppend(code, "      Type wb = -ws * ucl::Init<Type>(",
                        range_shift, ".0f);\n");
      }
      absl::StrAppend(code, "      ", dst_r_prefix, u, " ", assign_op, " ",
                      acc_vec, " * ", ws_acc_expr, " + ", x_sum_name,
                      " * wb;\n");
    }
    absl::StrAppend(code, "    }\n");
  }
}

class MetalDecodeQuantizedFc : public ml_drift::GPUOperation {
 public:
  MetalDecodeQuantizedFc() = default;
  MetalDecodeQuantizedFc(MetalDecodeQuantizedFc&&) = default;
  MetalDecodeQuantizedFc& operator=(MetalDecodeQuantizedFc&&) = default;
  MetalDecodeQuantizedFc(const MetalDecodeQuantizedFc&) = delete;
  MetalDecodeQuantizedFc& operator=(const MetalDecodeQuantizedFc&) = delete;

  std::vector<ml_drift::int3> GetPossibleKernelWorkGroups(
      ml_drift::TuningType tuning_type, const ml_drift::GpuInfo& gpu_info,
      const ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  ml_drift::int3 GetGridSize() const override {
    return ml_drift::int3(x_tasks_, work_group_size_.y, 1);
  }

  static std::unique_ptr<MetalDecodeQuantizedFc> Create(
      const ml_drift::GpuInfo& gpu_info, const ml_drift::TensorDescriptor& src,
      const ml_drift::TensorDescriptor& dst,
      const ml_drift::GpuModelBuilder::Weights& weights,
      const ml_drift::TensorDescriptor& scale_desc,
      const ml_drift::TensorDescriptor* zero_point_desc,
      const ml_drift::TensorDescriptor* bias) {
    auto op = std::make_unique<MetalDecodeQuantizedFc>();
    const ml_drift::DataType storage_type = weights.desc.type;
    const int dst_slices = ml_drift::DivideRoundUp(weights.shape.o, 4);
    const int src_slices = ml_drift::DivideRoundUp(weights.shape.i, 4);
    const bool is_blockwise = weights.scale_zp_shape.i > 1;
    const int src_group_size =
        is_blockwise ? src_slices / weights.scale_zp_shape.i : src_slices;
    const bool has_zero_point = zero_point_desc != nullptr;
    const bool has_bias = bias != nullptr;
    const int cu_count = std::max(1, gpu_info.GetComputeUnitsCount());
    const int outs_per_thread =
        SelectOutsPerThread(gpu_info, storage_type, dst_slices, cu_count);
    const int x_tasks = dst_slices / outs_per_thread;
    const int max_y_size = is_blockwise ? weights.scale_zp_shape.i : src_slices;

    op->x_tasks_ = x_tasks;
    op->work_group_size_ =
        SelectWorkGroupSize(gpu_info, storage_type, dst_slices, x_tasks,
                            max_y_size, outs_per_thread);

    const bool emit_wide_sync =
        gpu_info.IsApple() && gpu_info.IsApiMetal() &&
        (gpu_info.apple_info.IsMSeries() || IsApple9OrOlderBionic(gpu_info)) &&
        dst_slices >= 1024 * 16 && x_tasks % op->work_group_size_.x == 0 &&
        op->work_group_size_.y == 1;

    op->AddSrcTensor("src_tensor", src);
    op->AddDstTensor("dst_tensor", dst);
    op->args_.AddInt("x_tasks", x_tasks);
    if (is_blockwise) {
      op->args_.AddInt("src_groups", weights.scale_zp_shape.i);
    }

    ml_drift::BufferDescriptor w_buf;
    w_buf.element_type = ml_drift::DataType::kUint32;
    if (storage_type == ml_drift::DataType::kUint2) {
      w_buf.element_size = outs_per_thread;
    } else if (storage_type == ml_drift::DataType::kUint4) {
      w_buf.element_size = 2 * outs_per_thread;
    } else {
      w_buf.element_size = 4;
    }
    op->AddSrcBuffer("weights", w_buf);
    op->AddSrcTensor("weights_scale", scale_desc);
    if (zero_point_desc != nullptr) {
      op->AddSrcTensor("weights_zero_point", *zero_point_desc);
    }
    if (has_bias) {
      op->AddSrcTensor("biases", *bias);
    }

    const bool scale_is_fp32 =
        scale_desc.GetDataType() == ml_drift::DataType::kFloat32;
    op->code_ = GenerateCode(
        src.GetDataType(), storage_type, UseInt2BitmaskUnpack(), scale_is_fp32,
        outs_per_thread, is_blockwise, src_group_size, has_zero_point, has_bias,
        emit_wide_sync, op->work_group_size_);
    op->flops_ =
        static_cast<uint64_t>(weights.shape.o) * weights.shape.i * 2ULL;
    return op;
  }

 private:
  static std::string GenerateCode(ml_drift::DataType src_dtype,
                                  ml_drift::DataType storage_type,
                                  bool int2_bitmask, bool scale_is_fp32,
                                  int outs_per_thread, bool is_blockwise,
                                  int src_group_size, bool has_zero_point,
                                  bool has_bias, bool emit_wide_sync,
                                  const ml_drift::int3& wg) {
    const bool wg_reduction = wg.y > 1;
    const bool split_even_odd = storage_type == ml_drift::DataType::kUint4;
    const int prescale_shift = InputPrescaleShift(src_dtype, storage_type);
    const int chunk_slices = ChunkSlices(storage_type);
    const bool track_x_sum = has_zero_point ||
                             storage_type == ml_drift::DataType::kUint2 ||
                             storage_type == ml_drift::DataType::kUint4;
    const std::string zero_val = ml_drift::GetZeroValue(src_dtype);

    std::string c;
    c += "MAIN_FUNCTION($0) {\n";
    c += "  int base_task = ucl::GetGlobalId<0>();\n";
    if (wg_reduction) {
      c += "  if (ucl::GetGroupId<0>() * WG_SIZE_X >= args.x_tasks) return;\n";
      c += "  int tid_x = ucl::GetLocalId<0>();\n";
      c += "  int tid_y = ucl::GetLocalId<1>();\n";
    } else {
      c += "  if (base_task >= args.x_tasks) return;\n";
    }
    absl::StrAppend(&c, "  int base_dst_s = base_task * ", outs_per_thread,
                    ";\n");
    for (int u = 0; u < outs_per_thread; ++u) {
      absl::StrAppend(&c, "  Type r", u, " = ucl::Init<Type>(", zero_val,
                      ");\n");
    }

    if (wg_reduction) {
      c += "  if (base_task < args.x_tasks) {\n";
    }

    auto emit_acc_decls = [&](const std::string& indent,
                              const std::string& acc_prefix) {
      for (int u = 0; u < outs_per_thread; ++u) {
        if (split_even_odd) {
          absl::StrAppend(&c, indent, "Type2 ", acc_prefix, "ev", u,
                          " = ucl::Init<Type2>(", zero_val, ");\n");
          absl::StrAppend(&c, indent, "Type2 ", acc_prefix, "od", u,
                          " = ucl::Init<Type2>(", zero_val, ");\n");
        } else {
          absl::StrAppend(&c, indent, "Type ", acc_prefix, u,
                          " = ucl::Init<Type>(", zero_val, ");\n");
        }
      }
    };

    auto emit_weight_step = [&](const std::string& acc_prefix,
                                const std::string& x_sum_name,
                                const std::string& linear_idx_expr) {
      c += "      Type v = args.src_tensor.Read(0, 0, src_s, 0);\n";
      if (prescale_shift > 0) {
        const char* inv_lit = prescale_shift == 4 ? "0.0625f" : "0.25f";
        absl::StrAppend(&c, "      Type vs = v * ucl::Init<SType>(", inv_lit,
                        ");\n");
      } else {
        c += "      Type vs = v;\n";
      }
      if (track_x_sum) {
        absl::StrAppend(&c, "      ", x_sum_name,
                        " += (vs.x + vs.y) + (vs.z + vs.w);\n");
      }
      if (storage_type == ml_drift::DataType::kUint2) {
        if (outs_per_thread == 4) {
          absl::StrAppend(&c, "      uint4 w_vec = args.weights.Read(",
                          linear_idx_expr, ");\n");
          AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec.x", 0,
                                 acc_prefix, &c);
          AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec.y", 1,
                                 acc_prefix, &c);
          AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec.z", 2,
                                 acc_prefix, &c);
          AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec.w", 3,
                                 acc_prefix, &c);
        } else if (outs_per_thread == 2) {
          absl::StrAppend(&c, "      uint2 w_vec = args.weights.Read(",
                          linear_idx_expr, ");\n");
          AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec.x", 0,
                                 acc_prefix, &c);
          AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec.y", 1,
                                 acc_prefix, &c);
        } else {
          absl::StrAppend(&c, "      uint w_vec = args.weights.Read(",
                          linear_idx_expr, ");\n");
          AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec", 0,
                                 acc_prefix, &c);
        }
      } else if (storage_type == ml_drift::DataType::kUint4) {
        if (outs_per_thread == 2) {
          absl::StrAppend(&c, "      uint4 w_vec = args.weights.Read(",
                          linear_idx_expr, ");\n");
          AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec.xy", 0,
                                 acc_prefix, &c);
          AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec.zw", 1,
                                 acc_prefix, &c);
        } else {
          absl::StrAppend(&c, "      uint2 w_vec = args.weights.Read(",
                          linear_idx_expr, ");\n");
          AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec", 0,
                                 acc_prefix, &c);
        }
      } else {
        absl::StrAppend(&c, "      uint4 w_vec = args.weights.Read(",
                        linear_idx_expr, ");\n");
        AppendUnpackAccumulate(storage_type, int2_bitmask, "w_vec", 0,
                               acc_prefix, &c);
      }
      if (emit_wide_sync) {
        // Keep SIMD groups in lockstep across wide `lm_head` rows to prevent
        // SLC/DRAM thrashing (mirrors `fully_connected.cc`).
        c += "      ucl::SyncThreads<WorkGroup, None>();\n";
      }
    };

    const std::string y_start = wg_reduction ? "tid_y" : "0";
    const std::string y_step = wg_reduction ? "WG_SIZE_Y" : "1";
    if (is_blockwise) {
      absl::StrAppend(&c, "  for (int g = ", y_start,
                      "; g < args.src_groups; g += ", y_step, ") {\n");
      absl::StrAppend(&c, "    int base_src_s = g * ", src_group_size, ";\n");
      c += "    int linear_idx = base_src_s * args.x_tasks + base_task;\n";
      if (src_group_size <= chunk_slices) {
        emit_acc_decls("    ", "bacc");
        if (track_x_sum) {
          absl::StrAppend(&c, "    SType bx_sum = ", zero_val, ";\n");
        }
        absl::StrAppend(&c, "    for (int k = 0; k < ", src_group_size,
                        "; ++k) {\n");
        c += "      int src_s = base_src_s + k;\n";
        emit_weight_step("bacc", "bx_sum", "linear_idx");
        c += "      linear_idx += args.x_tasks;\n";
        c += "    }\n";
        AppendApplyScaleBias(storage_type, int2_bitmask, scale_is_fp32,
                             outs_per_thread, prescale_shift,
                             /*is_blockwise=*/true, has_zero_point, "bacc",
                             "bx_sum", "r", "+=", &c);
      } else {
        absl::StrAppend(&c, "    for (int k0 = 0; k0 < ", src_group_size,
                        "; k0 += ", chunk_slices, ") {\n");
        emit_acc_decls("      ", "bacc");
        if (track_x_sum) {
          absl::StrAppend(&c, "      SType bx_sum = ", zero_val, ";\n");
        }
        absl::StrAppend(&c, "      int k_end = min(k0 + ", chunk_slices, ", ",
                        src_group_size, ");\n");
        c += "      for (int k = k0; k < k_end; ++k) {\n";
        c += "        int src_s = base_src_s + k;\n";
        emit_weight_step("bacc", "bx_sum", "linear_idx");
        c += "        linear_idx += args.x_tasks;\n";
        c += "      }\n";
        AppendApplyScaleBias(storage_type, int2_bitmask, scale_is_fp32,
                             outs_per_thread, prescale_shift,
                             /*is_blockwise=*/true, has_zero_point, "bacc",
                             "bx_sum", "r", "+=", &c);
        c += "    }\n";
      }
      c += "  }\n";
    } else {
      c += "  int src_slices = args.src_tensor.Slices();\n";
      absl::StrAppend(&c, "  int linear_idx = ", y_start,
                      " * args.x_tasks + base_task;\n");
      absl::StrAppend(&c, "  int linear_step = ", y_step, " * args.x_tasks;\n");
      absl::StrAppend(&c, "  int chunk_step = ", chunk_slices, " * ", y_step,
                      ";\n");
      // For `outs_per_thread == 1` (all INT4/INT8 layers), hoist per-channel
      // scale/bias once before the chunk loop to avoid redundant scale loads.
      const bool hoist_single_scale = outs_per_thread == 1;
      if (hoist_single_scale) {
        const int range_shift =
            storage_type == ml_drift::DataType::kUint4
                ? 8
                : (storage_type == ml_drift::DataType::kUint2 ? 2 : 0);
        const bool needs_bias = range_shift != 0 || has_zero_point;
        const char* rescale_lit = prescale_shift == 4 ? "16.0f" : "4.0f";
        if (prescale_shift > 0 && scale_is_fp32) {
          absl::StrAppend(&c,
                          "  Type ws_base = ucl::Convert<Type>("
                          "args.weights_scale.Read(base_dst_s) * ",
                          rescale_lit, ");\n");
        } else {
          c += "  Type ws_base = ucl::Convert<Type>("
               "args.weights_scale.Read(base_dst_s));\n";
          if (prescale_shift > 0) {
            absl::StrAppend(&c, "  ws_base *= ucl::Init<SType>(", rescale_lit,
                            ");\n");
          }
        }
        if (storage_type == ml_drift::DataType::kUint4) {
          c += "  Type ws_acc = ws_base * "
               "ucl::Init<Type>(16.0f, 1.0f, 16.0f, 1.0f);\n";
        } else if (storage_type == ml_drift::DataType::kUint2 &&
                   !int2_bitmask) {
          c += "  Type ws_acc = ws_base * "
               "ucl::Init<Type>(4.0f, 4.0f, 4.0f, 1.0f);\n";
        } else {
          c += "  Type ws_acc = ws_base;\n";
        }
        if (needs_bias) {
          if (has_zero_point) {
            c += "  Type wzp = ucl::Convert<Type>("
                 "args.weights_zero_point.Read(base_dst_s));\n";
            if (range_shift != 0) {
              absl::StrAppend(&c,
                              "  Type wb_hoist = -ws_base * ("
                              "ucl::Init<Type>(",
                              range_shift, ".0f) + wzp);\n");
            } else {
              c += "  Type wb_hoist = -ws_base * wzp;\n";
            }
          } else {
            absl::StrAppend(&c,
                            "  Type wb_hoist = -ws_base * "
                            "ucl::Init<Type>(",
                            range_shift, ".0f);\n");
          }
        }
      }
      absl::StrAppend(&c, "  for (int src_base = ", y_start,
                      "; src_base < src_slices; src_base += chunk_step) {\n");
      emit_acc_decls("    ", "acc");
      if (track_x_sum) {
        absl::StrAppend(&c, "    SType x_sum = ", zero_val, ";\n");
      }
      c += "    int src_end = min(src_base + chunk_step, src_slices);\n";
      absl::StrAppend(&c,
                      "    for (int src_s = src_base; src_s < src_end; "
                      "src_s += ",
                      y_step, ") {\n");
      emit_weight_step("acc", "x_sum", "linear_idx");
      c += "      linear_idx += linear_step;\n";
      c += "    }\n";
      if (hoist_single_scale) {
        const std::string acc_vec =
            split_even_odd
                ? "ucl::Init<Type>(accev0.x, accod0.x, accev0.y, accod0.y)"
                : "acc0";
        const bool needs_bias = has_zero_point ||
                                storage_type == ml_drift::DataType::kUint4 ||
                                storage_type == ml_drift::DataType::kUint2;
        if (needs_bias) {
          absl::StrAppend(&c, "    r0 += ", acc_vec,
                          " * ws_acc + x_sum * wb_hoist;\n");
        } else {
          absl::StrAppend(&c, "    r0 += ", acc_vec, " * ws_acc;\n");
        }
      } else {
        AppendApplyScaleBias(storage_type, int2_bitmask, scale_is_fp32,
                             outs_per_thread, prescale_shift,
                             /*is_blockwise=*/false, has_zero_point, "acc",
                             "x_sum", "r", "+=", &c);
      }
      c += "  }\n";
    }

    if (wg_reduction) {
      c += "  }\n";
      const int red_elems = outs_per_thread * wg.x * wg.y;
      absl::StrAppend(&c, "  __local Type shared_red[", red_elems, "];\n");
      const int plane_stride = wg.x * wg.y;
      for (int u = 0; u < outs_per_thread; ++u) {
        absl::StrAppend(&c, "  shared_red[", u * plane_stride,
                        " + tid_y * WG_SIZE_X + tid_x] = r", u, ";\n");
      }
      if (wg.y <= 8) {
        c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
        c += "  if (tid_y != 0 || base_task >= args.x_tasks) return;\n";
        for (int u = 0; u < outs_per_thread; ++u) {
          for (int y = 1; y < wg.y; ++y) {
            absl::StrAppend(&c, "  r", u, " += shared_red[",
                            u * plane_stride + y * wg.x, " + tid_x];\n");
          }
        }
      } else {
        c += "  for (int stride = WG_SIZE_Y / 2; stride > 0; stride /= 2) {\n";
        c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
        c += "    if (tid_y < stride) {\n";
        for (int u = 0; u < outs_per_thread; ++u) {
          absl::StrAppend(&c, "      r", u, " += shared_red[", u * plane_stride,
                          " + (tid_y + stride) * WG_SIZE_X + tid_x];\n");
          absl::StrAppend(&c, "      shared_red[", u * plane_stride,
                          " + tid_y * WG_SIZE_X + tid_x] = r", u, ";\n");
        }
        c += "    }\n";
        c += "  }\n";
        c += "  if (tid_y != 0 || base_task >= args.x_tasks) return;\n";
      }
    }

    for (int u = 0; u < outs_per_thread; ++u) {
      const std::string dst_s_expr =
          u == 0 ? "base_dst_s" : absl::StrCat("(base_dst_s + ", u, ")");
      c += "  {\n";
      if (has_bias) {
        absl::StrAppend(&c, "    r", u,
                        " += ucl::Convert<Type>(args.biases.Read(", dst_s_expr,
                        "));\n");
      }
      absl::StrAppend(&c,
                      "    args.dst_tensor::type res_value = "
                      "ucl::Convert<args.dst_tensor::type>(r",
                      u, ");\n");
      absl::StrAppend(&c, "    args.dst_tensor.Write(res_value, 0, 0, ",
                      dst_s_expr, ", 0);\n");
      c += "  }\n";
    }
    c += "}\n";

    absl::StrReplaceAll({{"SType", ml_drift::ToUclDataType(src_dtype, 1)},
                         {"Type2", ml_drift::ToUclDataType(src_dtype, 2)},
                         {"Type", ml_drift::ToUclDataType(src_dtype, 4)},
                         {"WG_SIZE_X", std::to_string(wg.x)},
                         {"WG_SIZE_Y", std::to_string(wg.y)}},
                        &c);
    return c;
  }

  int x_tasks_ = 1;
};

}  // namespace

std::optional<ml_drift::GpuModelBuilder::TensorHandle>
TryAddMetalDecodeQuantizedFullyConnected(
    ml_drift::GpuModelBuilder* builder,
    const ml_drift::GpuModelBuilder::TensorHandle& src,
    const ml_drift::GpuModelBuilder::Weights& weights,
    const ml_drift::GpuModelBuilder::TensorHandle* biases,
    absl::string_view op_name) {
  if (builder == nullptr || !MetalDecodeFcEnabled(builder->gpu_info())) {
    return std::nullopt;
  }
  const ml_drift::BHWC src_shape = src.tensor_desc.GetBHWCShape();
  if (src_shape.b != 1 || src_shape.h != 1 || src_shape.w != 1) {
    return std::nullopt;
  }
  if (weights.shape.h != 1 || weights.shape.w != 1 || weights.shape.o <= 0 ||
      weights.shape.i <= 0 || src_shape.c != weights.shape.i) {
    return std::nullopt;
  }
  if (weights.desc.layout != ml_drift::WeightsLayout::kOSpatialIOGroupI4O4 ||
      weights.weights.tensor_desc.GetStorageType() !=
          ml_drift::TensorStorageType::kBuffer) {
    return std::nullopt;
  }
  const int dst_slices = ml_drift::DivideRoundUp(weights.shape.o, 4);
  if (weights.desc.GetOutputGroupSize() != dst_slices) return std::nullopt;
  if (weights.desc.type != ml_drift::DataType::kUint2 &&
      weights.desc.type != ml_drift::DataType::kUint4 &&
      weights.desc.type != ml_drift::DataType::kUint8) {
    return std::nullopt;
  }
  const ml_drift::DataType src_dtype = src.tensor_desc.GetDataType();
  if (src_dtype != ml_drift::DataType::kFloat16 &&
      src_dtype != ml_drift::DataType::kFloat32) {
    return std::nullopt;
  }
  if (builder->GetConvPrecision(src_dtype) ==
      ml_drift::CalculationsPrecision::kF32F16) {
    return std::nullopt;
  }
  if (!weights.scale.has_value() || weights.scale_zp_shape.h != 1 ||
      weights.scale_zp_shape.w != 1 || weights.scale_zp_shape.o <= 1) {
    return std::nullopt;
  }
  const int src_slices = ml_drift::DivideRoundUp(weights.shape.i, 4);
  if (weights.scale_zp_shape.i > 1 &&
      src_slices % weights.scale_zp_shape.i != 0) {
    return std::nullopt;
  }

  ml_drift::BHWC dst_shape = src_shape;
  dst_shape.c = weights.shape.o;
  ml_drift::GpuModelBuilder::TensorHandle dst =
      builder->AddTensor(dst_shape, src_dtype);
  const ml_drift::TensorDescriptor* bias_desc =
      biases != nullptr ? &biases->tensor_desc : nullptr;
  const ml_drift::TensorDescriptor* zero_point_desc =
      weights.zero_point.has_value() ? &weights.zero_point->tensor_desc
                                     : nullptr;

  auto op = MetalDecodeQuantizedFc::Create(
      builder->gpu_info(), src.tensor_desc, dst.tensor_desc, weights,
      weights.scale->tensor_desc, zero_point_desc, bias_desc);
  std::vector<ml_drift::GpuModelBuilder::ValueId> input_ids = {
      src.id, weights.weights.id, weights.scale->id};
  if (weights.zero_point.has_value()) {
    input_ids.push_back(weights.zero_point->id);
  }
  if (biases != nullptr) {
    input_ids.push_back(biases->id);
  }
  builder->AddGpuOperation(input_ids, {dst.id}, std::move(op),
                           absl::StrCat(op_name, "_metal_decode"));
  return dst;
}

}  // namespace litert::tensor
