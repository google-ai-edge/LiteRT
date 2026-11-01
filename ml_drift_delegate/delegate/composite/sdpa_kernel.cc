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

#include "ml_drift_delegate/delegate/composite/sdpa_kernel.h"

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
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/kernels/special/conv_softmax_conv.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/compiler_options.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

// Work-group and tiling parameters for `FusedFlashSdpaOp`.
struct FlashSdpaConfig {
  // Work items per work group (`s_splits * b_r`).
  int threads = 0;
  // Total channel slices (`head_dim / 4`).
  int slices = 0;
  // Channel slices per thread (`<= 16` in FP16, `<= 8` in FP32).
  int block_s = 0;
  // Number of threads cooperating across the channel slices (`1`, `2`, or `4`).
  int s_splits = 0;
  // Query rows per work group (`threads / s_splits`).
  int b_r = 0;
  // Keys cooperatively staged per interior tile (`16` or `4`).
  int b_c = 16;
  // Whether `Q`, `K`, `V`, and `O_acc` use `half4` (`true`) or `float4`
  // (`false`).
  bool use_fp16 = false;
  // Whether pre-scaled `Q` is staged in `__local` SRAM (`true`) or held in
  // registers (`false`).
  bool q_in_local = false;
  // Device local memory budget in bytes (32 KB is the minimum guaranteed
  // CL_DEVICE_LOCAL_MEM_SIZE / threadgroup memory across Adreno, Mali, and
  // Apple GPUs; WebGPU queries max_compute_workgroup_storage_size or defaults
  // to 16 KB).
  int local_memory_bytes = 32 * 1024;
};

// Computes the total per-work-group `__local` SRAM footprint in bytes
// (`k_local`, `v_local`, optional `q_local`, and optional `red_local`) for the
// given tiling parameters so `GetFlashSdpaConfig` can verify and downscale
// `b_c` / `b_r` to fit within the device's local memory budget.
int FlashSdpaLocalMemoryBytes(int slices, int block_s, int s_splits, int b_r,
                              int b_c, bool use_fp16, bool q_in_local) {
  const int threads = s_splits * b_r;
  const int vec_bytes = use_fp16 ? 8 : 16;
  const int kv_local_size = slices * b_c + (s_splits - 1);
  // `k_local[kv_local_size]` + `v_local[kv_local_size]` of `vec_type`.
  int bytes = 2 * kv_local_size * vec_bytes;
  if (q_in_local) {
    // `q_local[block_s * threads]` of `vec_type`.
    bytes += block_s * threads * vec_bytes;
  }
  if (s_splits > 1) {
    // `red_local` ping-pongs two `threads`-element buffers when `b_c > 4` to
    // avoid a second work-group barrier per 4-key substep.
    const int red_buffers = (b_c > 4) ? 2 : 1;
    bytes += red_buffers * threads * 16;
  }
  return bytes;
}

// Selects work-group and tiling parameters for `FusedFlashSdpaOp`.
// In ML Drift (`ReserveGraphTensors`), `q_desc.GetDataType()` is set to
// `DataType::kFloat16` when the model precision is
// `CalculationsPrecision::kF16` or `kF32F16`, and `DataType::kFloat32` when
// `kF32`. When `use_f32_accum` is false (i.e.,
// `model_builder->GetConvPrecision(...) == kF16`), `use_fp16` is enabled on
// FP16-capable OpenCL/Metal devices so `Q`, `K`, `V`, and `O_acc` use `half4`;
// when `kF32` or `kF32F16` (`use_f32_accum == true`) or WebGPU, `float4` is
// used.
FlashSdpaConfig GetFlashSdpaConfig(const ::ml_drift::GpuInfo& gpu_info,
                                   const ::ml_drift::TensorDescriptor& q_desc,
                                   bool use_f32_accum = false) {
  FlashSdpaConfig config;
  if (gpu_info.IsApiWebGpu()) {
    config.local_memory_bytes =
        gpu_info.webgpu_info.max_compute_workgroup_storage_size > 0
            ? gpu_info.webgpu_info.max_compute_workgroup_storage_size
            : 16 * 1024;
  }
  config.use_fp16 = (q_desc.GetDataType() == ::ml_drift::DataType::kFloat16) &&
                    !use_f32_accum && gpu_info.SupportsFP16() &&
                    !gpu_info.IsApiWebGpu();

  const int head_dim = q_desc.GetBHWCShape().c;
  if (head_dim <= 0 || head_dim % 4 != 0 || head_dim > 128) {
    return config;
  }
  const int slices = head_dim / 4;

  // Keep per-thread `acc` register footprint `<= 32` 32-bit registers to
  // prevent register spilling across the `K`/`V` loop:
  // - FP16 (`half4`, 2 32-bit registers per slice): `max_block_s = 16` slices
  //   (`H <= 64` uses `s_splits = 1`; `64 < H <= 128` uses `s_splits = 2`).
  // - FP32 (`float4`, 4 32-bit registers per slice): `max_block_s = 8` slices
  //   (`H <= 32` uses `s_splits = 1`; `32 < H <= 64` uses `s_splits = 2`;
  //   `64 < H <= 128` uses `s_splits = 4`), except on Adreno Gen 8 (`8xx`) in
  //   FP32 where `max_block_s = 4` (`s_splits = 4, b_r = 16, threads = 64` for
  //   `H = 64`) keeps `q_0..q_3` (16 regs) + `acc_0..acc_3` (16 regs) = 32
  //   32-bit registers in GPRs (`q_in_local = false`) and `__local` SRAM at
  //   `10.1 KB` (`<= 16 KB`), avoiding both 48-register single-wave spilling
  //   and 64-lane `q_local` bank conflicts.
  // - On Mali Valhall (16-lane warp, 64 32-bit registers per thread), stage `Q`
  //   in `__local` SRAM (`q_in_local = true`) so `Q` consumes 0 general-purpose
  //   registers while preserving `b_r = 32` in FP32 and `b_r = 64` in FP16.
  const bool is_adreno8xx_fp32 = gpu_info.IsAdreno() &&
                                 gpu_info.adreno_info.IsAdreno8xx() &&
                                 !config.use_fp16;
  const int max_block_s = config.use_fp16 ? 16 : (is_adreno8xx_fp32 ? 4 : 8);
  const int s_splits =
      (slices <= max_block_s) ? 1 : ((slices <= 2 * max_block_s) ? 2 : 4);
  const int block_s = (slices + s_splits - 1) / s_splits;
  const bool q_in_local =
      gpu_info.IsMali() || (is_adreno8xx_fp32 && block_s > 4);

  int b_r =
      (s_splits == 4 && !is_adreno8xx_fp32) ? 32 : std::max(1, 64 / s_splits);
  const int max_work_group_size = gpu_info.GetMaxWorkGroupTotalSize();
  if (max_work_group_size > 0) {
    while (b_r > 1 && s_splits * b_r > max_work_group_size) {
      b_r /= 2;
    }
    if (s_splits * b_r > max_work_group_size) {
      return config;
    }
  }

  int b_c = 16;
  if (FlashSdpaLocalMemoryBytes(slices, block_s, s_splits, b_r, b_c,
                                config.use_fp16,
                                q_in_local) > config.local_memory_bytes) {
    if (s_splits == 4 && b_r > 16 &&
        FlashSdpaLocalMemoryBytes(slices, block_s, s_splits, /*b_r=*/16, b_c,
                                  config.use_fp16,
                                  q_in_local) <= config.local_memory_bytes) {
      b_r = 16;
    } else {
      b_c = 4;
    }
  }
  while (b_r > 1 && FlashSdpaLocalMemoryBytes(
                        slices, block_s, s_splits, b_r, b_c, config.use_fp16,
                        q_in_local) > config.local_memory_bytes) {
    b_r /= 2;
  }
  if (FlashSdpaLocalMemoryBytes(slices, block_s, s_splits, b_r, b_c,
                                config.use_fp16,
                                q_in_local) > config.local_memory_bytes) {
    return config;
  }

  config.threads = s_splits * b_r;
  config.slices = slices;
  config.block_s = block_s;
  config.s_splits = s_splits;
  config.b_r = b_r;
  config.b_c = b_c;
  config.q_in_local = q_in_local;
  return config;
}

class FusedFlashSdpaOp : public ::ml_drift::GPUOperation {
 public:
  // Uses a 1D work group of `(threads, 1, 1)` (`threads = s_splits * b_r`, up
  // to 128 work items) so all threads in a work group share the same `(B,
  // q_head)` and `kv_head`, cooperatively staging `b_c` keys of `K` and `V`
  // into `__local` SRAM (`b_r`x — up to `64x` — global read reuse and
  // bank-conflict-free SRAM reads). Within the work group, `tid =
  // GetLocalId<0>()` is split into
  // `q_grp = tid / s_splits` (`b_r` query rows) and `s_grp = tid % s_splits`
  // (`s_splits` channel-slice partitions).
  FusedFlashSdpaOp(int threads, int b_r) : b_r_(b_r) {
    work_group_size_ = ::ml_drift::int3(threads, 1, 1);
  }

  // In ML Drift, `GetGridSize()` returns total global work items `(global_x,
  // global_y, global_z)`, which divided by `work_group_size_ = (threads, 1, 1)`
  // launches `(1, num_q_tiles, total_heads)` work groups:
  // - X (`threads`): 1 work group of `threads` work items per `(q_tile,
  //   head_batch)` pair.
  // - Y (`num_q_tiles = ceil(T / b_r_)`): tiles the query sequence `T`
  //   (`dst_[0]->Width()`) into blocks of `b_r_` rows (`GetGroupId<1>()`).
  // - Z (`total_heads = N_q * B`): flattens query heads (`dst_[0]->Height()`)
  //   and batch (`dst_[0]->Batch()`) onto `GetGroupId<2>()` so every work group
  //   processes a single `(B, q_head)` pair and shares one `kv_head`.
  ::ml_drift::int3 GetGridSize() const override {
    const int num_q_tiles = (dst_[0]->Width() + b_r_ - 1) / b_r_;
    const int total_heads = dst_[0]->Height() * dst_[0]->Batch();
    return ::ml_drift::int3(work_group_size_.x, num_q_tiles, total_heads);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  FusedFlashSdpaOp(FusedFlashSdpaOp&&) = default;
  FusedFlashSdpaOp& operator=(FusedFlashSdpaOp&&) = default;
  FusedFlashSdpaOp(const FusedFlashSdpaOp&) = delete;
  FusedFlashSdpaOp& operator=(const FusedFlashSdpaOp&) = delete;

 private:
  int b_r_ = 0;
};

// Generates the fused Flash Attention shader operating on `[B, N, T, H]`
// (`BHWC`: `Batch=B`, `Height=N_q/N_kv`, `Width=T/S`, `Channels=H`) tensors:
// - All `threads` work items in the work group cooperatively stage `b_c` (16)
//   keys of `K` and `V` into `__local` SRAM (`k_local` and `v_local`), reducing
//   global memory `K`/`V` reads and 3D address calculations by up to 64x.
// - Each thread then processes the staged keys in 4-key sub-steps (`sub += 4`)
//   using bank-conflict-free SRAM reads from `k_local` and `v_local` (uniform
//   broadcast when `s_splits == 1`, or group-skewed `+ s_grp` indexing when
//   `s_splits > 1`), vectorized `half4` / `float4` dot products and `V`
//   accumulation, and normalized online softmax (`exp2`) in registers.
//
// Final rendered shader example (OpenCL, H=16, slices=4, block_s=4,
// s_splits=1, b_r=64, b_c=16, threads=64, use_fp16=true, q_in_local=false,
// gqa_ratio=1, has_batch=false, has_mask=false):
// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// ```opencl
// __attribute__((reqd_work_group_size(64, 1, 1)))
// MAIN_FUNCTION($0) {
//   int tid = ucl::GetLocalId<0>();
//   int q_tile = ucl::GetGroupId<1>();
//   int head_batch = ucl::GetGroupId<2>();
//   int q_head = head_batch;
//   int kv_head = q_head;
//   int s_grp = 0;
//   int q_grp = tid;
//   int s_base = s_grp * 4;
//   int q_idx = q_tile * 64 + q_grp;
//   int qr = min(q_idx, args.q.Width() - 1);
//   float q_scale = args.scale * 1.4426950408889634f;
//   __local half4 k_local[64];
//   __local half4 v_local[64];
//   half4 q_0 = ucl::Convert<half4>(ucl::Convert<float4>(args.q.Read(qr, q_head, s_base + 0)) * q_scale);
//   half4 q_1 = ucl::Convert<half4>(ucl::Convert<float4>(args.q.Read(qr, q_head, s_base + 1)) * q_scale);
//   half4 q_2 = ucl::Convert<half4>(ucl::Convert<float4>(args.q.Read(qr, q_head, s_base + 2)) * q_scale);
//   half4 q_3 = ucl::Convert<half4>(ucl::Convert<float4>(args.q.Read(qr, q_head, s_base + 3)) * q_scale);
//   half4 acc_0 = ucl::Init<half4>(0.0f);
//   half4 acc_1 = ucl::Init<half4>(0.0f);
//   half4 acc_2 = ucl::Init<half4>(0.0f);
//   half4 acc_3 = ucl::Init<half4>(0.0f);
//   float m_i = -1.0e30f;
//   float l_i = 0.0f;
//   int key_base = 0;
//   for (; key_base + 16 <= args.k.Width(); key_base += 16) {
//     for (int idx = tid; idx < 64; idx += 64) {
//       int k_off = idx % 16;
//       int sl = idx / 16;
//       k_local[idx] = ucl::Convert<half4>(args.k.Read(key_base + k_off, kv_head, sl));
//       v_local[idx] = ucl::Convert<half4>(args.v.Read(key_base + k_off, kv_head, sl));
//     }
//     ucl::SyncThreads<WorkGroup, Local>();
//     for (int sub = 0; sub < 16; sub += 4) {
//       float4 d = ucl::Init<float4>(0.0f);
//       {
//         int kv_idx = (s_base + 0) * 16 + sub;
//         half4 q_val = q_0;
//         d.x += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 0]));
//         d.y += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 1]));
//         d.z += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 2]));
//         d.w += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 3]));
//       }
//       {
//         int kv_idx = (s_base + 1) * 16 + sub;
//         half4 q_val = q_1;
//         d.x += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 0]));
//         d.y += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 1]));
//         d.z += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 2]));
//         d.w += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 3]));
//       }
//       {
//         int kv_idx = (s_base + 2) * 16 + sub;
//         half4 q_val = q_2;
//         d.x += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 0]));
//         d.y += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 1]));
//         d.z += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 2]));
//         d.w += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 3]));
//       }
//       {
//         int kv_idx = (s_base + 3) * 16 + sub;
//         half4 q_val = q_3;
//         d.x += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 0]));
//         d.y += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 1]));
//         d.z += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 2]));
//         d.w += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 3]));
//       }
//       float row_max = max(max(d.x, d.y), max(d.z, d.w));
//       float m_new = max(m_i, row_max);
//       float alpha = exp2(m_i - m_new);
//       float4 p = exp2(d - m_new);
//       float p_sum = (p.x + p.y) + (p.z + p.w);
//       float l_new = l_i * alpha + p_sum;
//       float inv_l = l_new > 0.0f ? (1.0f / l_new) : 0.0f;
//       float acc_scale = (l_i * alpha) * inv_l;
//       p *= inv_l;
//       l_i = l_new;
//       m_i = m_new;
//       half4 p_v = ucl::Convert<half4>(p);
//       half4 acc_scale_v = ucl::Init<half4>(acc_scale);
//       {
//         int kv_idx = (s_base + 0) * 16 + sub;
//         acc_0 = acc_0 * acc_scale_v + p_v.x * v_local[kv_idx + 0];
//         acc_0 += p_v.y * v_local[kv_idx + 1];
//         acc_0 += p_v.z * v_local[kv_idx + 2];
//         acc_0 += p_v.w * v_local[kv_idx + 3];
//       }
//       {
//         int kv_idx = (s_base + 1) * 16 + sub;
//         acc_1 = acc_1 * acc_scale_v + p_v.x * v_local[kv_idx + 0];
//         acc_1 += p_v.y * v_local[kv_idx + 1];
//         acc_1 += p_v.z * v_local[kv_idx + 2];
//         acc_1 += p_v.w * v_local[kv_idx + 3];
//       }
//       {
//         int kv_idx = (s_base + 2) * 16 + sub;
//         acc_2 = acc_2 * acc_scale_v + p_v.x * v_local[kv_idx + 0];
//         acc_2 += p_v.y * v_local[kv_idx + 1];
//         acc_2 += p_v.z * v_local[kv_idx + 2];
//         acc_2 += p_v.w * v_local[kv_idx + 3];
//       }
//       {
//         int kv_idx = (s_base + 3) * 16 + sub;
//         acc_3 = acc_3 * acc_scale_v + p_v.x * v_local[kv_idx + 0];
//         acc_3 += p_v.y * v_local[kv_idx + 1];
//         acc_3 += p_v.z * v_local[kv_idx + 2];
//         acc_3 += p_v.w * v_local[kv_idx + 3];
//       }
//     }
//     ucl::SyncThreads<WorkGroup, Local>();
//   }
//   for (; key_base < args.k.Width(); key_base += 4) {
//     for (int idx = tid; idx < 16; idx += 64) {
//       int k_off = idx % 4;
//       int sl = idx / 4;
//       int k_idx = key_base + k_off;
//       k_local[idx] = (k_idx < args.k.Width()) ? ucl::Convert<half4>(args.k.Read(k_idx, kv_head, sl)) : ucl::Init<half4>(0.0f);
//       v_local[idx] = (k_idx < args.v.Width()) ? ucl::Convert<half4>(args.v.Read(k_idx, kv_head, sl)) : ucl::Init<half4>(0.0f);
//     }
//     ucl::SyncThreads<WorkGroup, Local>();
//     float4 d = ucl::Init<float4>(0.0f);
//     {
//       int kv_idx = (s_base + 0) * 4;
//       half4 q_val = q_0;
//       d.x += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 0]));
//       d.y += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 1]));
//       d.z += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 2]));
//       d.w += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 3]));
//     }
//     {
//       int kv_idx = (s_base + 1) * 4;
//       half4 q_val = q_1;
//       d.x += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 0]));
//       d.y += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 1]));
//       d.z += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 2]));
//       d.w += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 3]));
//     }
//     {
//       int kv_idx = (s_base + 2) * 4;
//       half4 q_val = q_2;
//       d.x += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 0]));
//       d.y += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 1]));
//       d.z += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 2]));
//       d.w += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 3]));
//     }
//     {
//       int kv_idx = (s_base + 3) * 4;
//       half4 q_val = q_3;
//       d.x += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 0]));
//       d.y += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 1]));
//       d.z += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 2]));
//       d.w += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 3]));
//     }
//     if ((key_base) + 0 >= args.k.Width()) d.x = -1.0e30f;
//     if ((key_base) + 1 >= args.k.Width()) d.y = -1.0e30f;
//     if ((key_base) + 2 >= args.k.Width()) d.z = -1.0e30f;
//     if ((key_base) + 3 >= args.k.Width()) d.w = -1.0e30f;
//     float row_max = max(max(d.x, d.y), max(d.z, d.w));
//     float m_new = max(m_i, row_max);
//     float alpha = exp2(m_i - m_new);
//     float4 p;
//     p.x = d.x > -1.0e29f ? exp2(d.x - m_new) : 0.0f;
//     p.y = d.y > -1.0e29f ? exp2(d.y - m_new) : 0.0f;
//     p.z = d.z > -1.0e29f ? exp2(d.z - m_new) : 0.0f;
//     p.w = d.w > -1.0e29f ? exp2(d.w - m_new) : 0.0f;
//     float p_sum = (p.x + p.y) + (p.z + p.w);
//     float l_new = l_i * alpha + p_sum;
//     float inv_l = l_new > 0.0f ? (1.0f / l_new) : 0.0f;
//     float acc_scale = (l_i * alpha) * inv_l;
//     p *= inv_l;
//     l_i = l_new;
//     m_i = m_new;
//     half4 p_v = ucl::Convert<half4>(p);
//     half4 acc_scale_v = ucl::Init<half4>(acc_scale);
//     {
//       int kv_idx = (s_base + 0) * 4;
//       acc_0 = acc_0 * acc_scale_v + p_v.x * v_local[kv_idx + 0];
//       acc_0 += p_v.y * v_local[kv_idx + 1];
//       acc_0 += p_v.z * v_local[kv_idx + 2];
//       acc_0 += p_v.w * v_local[kv_idx + 3];
//     }
//     {
//       int kv_idx = (s_base + 1) * 4;
//       acc_1 = acc_1 * acc_scale_v + p_v.x * v_local[kv_idx + 0];
//       acc_1 += p_v.y * v_local[kv_idx + 1];
//       acc_1 += p_v.z * v_local[kv_idx + 2];
//       acc_1 += p_v.w * v_local[kv_idx + 3];
//     }
//     {
//       int kv_idx = (s_base + 2) * 4;
//       acc_2 = acc_2 * acc_scale_v + p_v.x * v_local[kv_idx + 0];
//       acc_2 += p_v.y * v_local[kv_idx + 1];
//       acc_2 += p_v.z * v_local[kv_idx + 2];
//       acc_2 += p_v.w * v_local[kv_idx + 3];
//     }
//     {
//       int kv_idx = (s_base + 3) * 4;
//       acc_3 = acc_3 * acc_scale_v + p_v.x * v_local[kv_idx + 0];
//       acc_3 += p_v.y * v_local[kv_idx + 1];
//       acc_3 += p_v.z * v_local[kv_idx + 2];
//       acc_3 += p_v.w * v_local[kv_idx + 3];
//     }
//     ucl::SyncThreads<WorkGroup, Local>();
//   }
//
//   if (q_idx < args.dst.Width()) {
//     {
//       args.dst::type res = ucl::Convert<args.dst::type>(acc_0);
//       args.dst.Write(res, q_idx, q_head, s_base + 0);
//     }
//     {
//       args.dst::type res = ucl::Convert<args.dst::type>(acc_1);
//       args.dst.Write(res, q_idx, q_head, s_base + 1);
//     }
//     {
//       args.dst::type res = ucl::Convert<args.dst::type>(acc_2);
//       args.dst.Write(res, q_idx, q_head, s_base + 2);
//     }
//     {
//       args.dst::type res = ucl::Convert<args.dst::type>(acc_3);
//       args.dst.Write(res, q_idx, q_head, s_base + 3);
//     }
//   }
// }
// ```
// NOLINTEND(whitespace/line_length)
// clang-format on
std::string GenerateFlashSdpaCode(const ::ml_drift::GpuInfo& gpu_info,
                                  const FlashSdpaConfig& config, int gqa_ratio,
                                  bool has_batch, bool has_mask,
                                  bool is_bool_mask, bool mask_has_batch) {
  const int threads = config.threads;
  const int slices = config.slices;
  const int block_s = config.block_s;
  const int s_splits = config.s_splits;
  const int b_r = config.b_r;
  const int b_c = config.b_c;
  const bool q_in_local = config.q_in_local;
  const int floor_s = slices / s_splits;
  const int rem_s = slices % s_splits;
  const int first_tail_s = (rem_s == 0) ? block_s : floor_s;
  const int kv_local_size = slices * b_c + (s_splits - 1);
  const absl::string_view vec_type = config.use_fp16 ? "half4" : "float4";

  auto repeat_s = [block_s, first_tail_s](absl::string_view pattern,
                                          absl::string_view tail_pattern) {
    std::string out;
    for (int s = 0; s < block_s; ++s) {
      const absl::string_view pat =
          (s >= first_tail_s) ? tail_pattern : pattern;
      absl::StrAppend(&out,
                      absl::StrReplaceAll(pat, {{"$s", absl::StrCat(s)}}));
    }
    return out;
  };

  std::string c;
  if (gpu_info.IsApiOpenCl()) {
    c = absl::StrCat("__attribute__((reqd_work_group_size(", threads,
                     ", 1, 1)))\n");
    if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno8xx() &&
        !gpu_info.opencl_info.IsCLVK() && !config.use_fp16) {
      absl::StrAppend(&c, "__attribute__((qcom_wave_pair_mode(1)))\n");
    }
  }
  absl::StrAppend(&c, R"(MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int q_tile = ucl::GetGroupId<1>();
  int head_batch = ucl::GetGroupId<2>();
)");

  if (has_batch) {
    absl::StrAppend(&c, R"(  int q_head = head_batch % args.q.Height();
  int B = head_batch / args.q.Height();
  args.q.SetBatchRef(B);
  args.k.SetBatchRef(B);
  args.v.SetBatchRef(B);
  args.dst.SetBatchRef(B);
)");
    if (has_mask && mask_has_batch) {
      absl::StrAppend(&c, R"(  int mask_b = min(B, args.mask.Batch() - 1);
  args.mask.SetBatchRef(mask_b);
)");
    }
  } else {
    absl::StrAppend(&c, "  int q_head = head_batch;\n");
    if (has_mask && mask_has_batch) {
      absl::StrAppend(&c, "  args.mask.SetBatchRef(0);\n");
    }
  }

  const std::string s_base_expr =
      (rem_s > 0 && rem_s < s_splits - 1)
          ? absl::StrCat("s_grp * ", floor_s, " + min(s_grp, ", rem_s, ")")
          : absl::StrCat("s_grp * ", block_s);
  absl::StrAppend(
      &c, "  int kv_head = ",
      (gqa_ratio > 1 ? absl::StrCat("q_head / ", gqa_ratio) : "q_head"), ";\n",
      "  int s_grp = ", (s_splits > 1 ? absl::StrCat("tid % ", s_splits) : "0"),
      ";\n", "  int q_grp = ",
      (s_splits > 1 ? absl::StrCat("tid / ", s_splits) : "tid"), ";\n",
      "  int s_base = ", s_base_expr, ";\n", "  int q_idx = q_tile * ", b_r,
      " + q_grp;\n", "  int qr = min(q_idx, args.q.Width() - 1);\n",
      "  float q_scale = args.scale * 1.4426950408889634f;\n", "  __local ",
      vec_type, " k_local[", kv_local_size, "];\n", "  __local ", vec_type,
      " v_local[", kv_local_size, "];\n");

  if (q_in_local) {
    absl::StrAppend(
        &c, "  __local ", vec_type, " q_local[", block_s * threads, "];\n",
        repeat_s(
            absl::StrCat("  q_local[$s * ", threads, " + tid] = ucl::Convert<",
                         vec_type,
                         ">(ucl::Convert<float4>(args.q.Read(qr, q_head, "
                         "s_base + $s)) * q_scale);\n"),
            absl::StrCat(
                "  q_local[$s * ", threads, " + tid] = (s_grp < ", rem_s,
                ") ? ucl::Convert<", vec_type,
                ">(ucl::Convert<float4>(args.q.Read(qr, q_head, s_base + $s)) "
                "* q_scale) : ucl::Init<",
                vec_type, ">(0.0f);\n")));
  } else {
    absl::StrAppend(
        &c,
        repeat_s(absl::StrCat("  ", vec_type, " q_$s = ucl::Convert<", vec_type,
                              ">(ucl::Convert<float4>(args.q.Read(qr, q_head, "
                              "s_base + $s)) * q_scale);\n"),
                 absl::StrCat("  ", vec_type, " q_$s = (s_grp < ", rem_s,
                              ") ? ucl::Convert<", vec_type,
                              ">(ucl::Convert<float4>(args.q.Read(qr, q_head, "
                              "s_base + $s)) * q_scale) : ucl::Init<",
                              vec_type, ">(0.0f);\n")));
  }

  absl::StrAppend(&c,
                  repeat_s(absl::StrCat("  ", vec_type, " acc_$s = ucl::Init<",
                                        vec_type, ">(0.0f);\n"),
                           absl::StrCat("  ", vec_type, " acc_$s = ucl::Init<",
                                        vec_type, ">(0.0f);\n")),
                  "  float m_i = -1.0e30f;\n  float l_i = 0.0f;\n");

  if (s_splits > 1) {
    const int red_buffers = (b_c > 4) ? 2 : 1;
    absl::StrAppend(&c, "  __local float4 red_local[", red_buffers * threads,
                    "];\n\n");
  }

  if (has_mask) {
    absl::StrAppend(&c, "  int mask_y = min(q_head, args.mask.Height() - 1);\n",
                    "  int mx = min(qr, args.mask.Width() - 1);\n");
  }

  const std::string q_expr =
      q_in_local ? absl::StrCat("q_local[$s * ", threads, " + tid]") : "q_$s";
  const std::string kv_grp_skew = (s_splits > 1) ? " + s_grp" : "";

  auto emit_substep = [&](bool is_interior, int kv_stride,
                          absl::string_view cur_key_expr,
                          absl::string_view kv_sub_offset) {
    absl::StrAppend(&c, "      float4 d = ucl::Init<float4>(0.0f);\n");
    const std::string k_block = absl::StrCat(
        "      {\n"
        "        int kv_idx = (s_base + $s) * ",
        kv_stride, kv_sub_offset, kv_grp_skew,
        ";\n"
        "        ",
        vec_type, " q_val = ", q_expr,
        ";\n"
        "        d.x += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 0]));\n"
        "        d.y += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 1]));\n"
        "        d.z += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 2]));\n"
        "        d.w += ucl::Convert<float>(dot(q_val, k_local[kv_idx + 3]));\n"
        "      }\n");
    const std::string k_tail_block = absl::StrCat(
        "      if (s_grp < ", rem_s, ") {\n", k_block, "      }\n");
    absl::StrAppend(&c, repeat_s(k_block, k_tail_block));

    if (s_splits > 1) {
      // Index `red_local` linearly by `tid` (`q_grp * s_splits + s_grp`) so
      // consecutive lanes write consecutive `float4` entries without bank
      // conflicts, and ping-pong across two `threads`-element buffers in the
      // interior `b_c > 4` loop so no second barrier is needed after reading.
      if (is_interior && b_c > 4) {
        absl::StrAppend(&c, "      int red_base = ((sub >> 2) & 1) * ", threads,
                        ";\n", "      red_local[red_base + tid] = d;\n",
                        "      ucl::SyncThreads<WorkGroup, Local>();\n",
                        "      int red_row = red_base + q_grp * ", s_splits,
                        ";\n", "      d = red_local[red_row]");
        for (int sp = 1; sp < s_splits; ++sp) {
          absl::StrAppend(&c, " + red_local[red_row + ", sp, "]");
        }
        absl::StrAppend(&c, ";\n");
      } else {
        absl::StrAppend(&c, "      red_local[tid] = d;\n",
                        "      ucl::SyncThreads<WorkGroup, Local>();\n",
                        "      d = red_local[q_grp * ", s_splits, "]");
        for (int sp = 1; sp < s_splits; ++sp) {
          absl::StrAppend(&c, " + red_local[q_grp * ", s_splits, " + ", sp,
                          "]");
        }
        absl::StrAppend(&c, ";\n");
      }
    }

    if (has_mask) {
      absl::StrAppend(&c, "      int mask_slice = (", cur_key_expr, ") / 4;\n");
      if (is_bool_mask) {
        absl::StrAppend(&c, R"(      {
        bool4 mv = args.mask.Read<bool>(mx, mask_y, mask_slice);
        d.x = mv.x ? d.x : -1.0e30f;
        d.y = mv.y ? d.y : -1.0e30f;
        d.z = mv.z ? d.z : -1.0e30f;
        d.w = mv.w ? d.w : -1.0e30f;
      }
)");
      } else {
        absl::StrAppend(&c, R"(      {
        float4 mv =
            ucl::Convert<float4>(args.mask.Read(mx, mask_y, mask_slice));
        if (mv.x <= -1.0e29f) {
          d.x = -1.0e30f;
        } else {
          d.x = max(d.x + mv.x * 1.4426950408889634f, -1.0e30f);
        }
        if (mv.y <= -1.0e29f) {
          d.y = -1.0e30f;
        } else {
          d.y = max(d.y + mv.y * 1.4426950408889634f, -1.0e30f);
        }
        if (mv.z <= -1.0e29f) {
          d.z = -1.0e30f;
        } else {
          d.z = max(d.z + mv.z * 1.4426950408889634f, -1.0e30f);
        }
        if (mv.w <= -1.0e29f) {
          d.w = -1.0e30f;
        } else {
          d.w = max(d.w + mv.w * 1.4426950408889634f, -1.0e30f);
        }
      }
)");
      }
    }

    if (!is_interior) {
      absl::StrAppend(
          &c, "      if ((", cur_key_expr,
          ") + 0 >= args.k.Width()) d.x = -1.0e30f;\n", "      if ((",
          cur_key_expr, ") + 1 >= args.k.Width()) d.y = -1.0e30f;\n",
          "      if ((", cur_key_expr,
          ") + 2 >= args.k.Width()) d.z = -1.0e30f;\n", "      if ((",
          cur_key_expr, ") + 3 >= args.k.Width()) d.w = -1.0e30f;\n");
    }

    // When all keys seen so far are masked (`l_new == 0.0f`), `inv_l = 0.0f`
    // keeps `p` and `acc` at `0.0f` instead of producing `NaN`.
    if (is_interior && !has_mask) {
      absl::StrAppend(
          &c, R"(      float row_max = max(max(d.x, d.y), max(d.z, d.w));
      float m_new = max(m_i, row_max);
      float alpha = exp2(m_i - m_new);
      float4 p = exp2(d - m_new);
      float p_sum = (p.x + p.y) + (p.z + p.w);
      float l_new = l_i * alpha + p_sum;
      float inv_l = l_new > 0.0f ? (1.0f / l_new) : 0.0f;
      float acc_scale = (l_i * alpha) * inv_l;
      p *= inv_l;
      l_i = l_new;
      m_i = m_new;
)");
    } else {
      absl::StrAppend(
          &c, R"(      float row_max = max(max(d.x, d.y), max(d.z, d.w));
      float m_new = max(m_i, row_max);
      float alpha = exp2(m_i - m_new);
      float4 p;
      p.x = d.x > -1.0e29f ? exp2(d.x - m_new) : 0.0f;
      p.y = d.y > -1.0e29f ? exp2(d.y - m_new) : 0.0f;
      p.z = d.z > -1.0e29f ? exp2(d.z - m_new) : 0.0f;
      p.w = d.w > -1.0e29f ? exp2(d.w - m_new) : 0.0f;
      float p_sum = (p.x + p.y) + (p.z + p.w);
      float l_new = l_i * alpha + p_sum;
      float inv_l = l_new > 0.0f ? (1.0f / l_new) : 0.0f;
      float acc_scale = (l_i * alpha) * inv_l;
      p *= inv_l;
      l_i = l_new;
      m_i = m_new;
)");
    }

    absl::StrAppend(&c, "      ", vec_type, " p_v = ucl::Convert<", vec_type,
                    ">(p);\n", "      ", vec_type, " acc_scale_v = ucl::Init<",
                    vec_type, ">(acc_scale);\n");

    const std::string v_block = absl::StrCat(
        "      {\n"
        "        int kv_idx = (s_base + $s) * ",
        kv_stride, kv_sub_offset, kv_grp_skew,
        ";\n"
        "        acc_$s = acc_$s * acc_scale_v + p_v.x * v_local[kv_idx + 0];\n"
        "        acc_$s += p_v.y * v_local[kv_idx + 1];\n"
        "        acc_$s += p_v.z * v_local[kv_idx + 2];\n"
        "        acc_$s += p_v.w * v_local[kv_idx + 3];\n"
        "      }\n");
    const std::string v_tail_block = absl::StrCat(
        "      if (s_grp < ", rem_s, ") {\n", v_block, "      }\n");
    absl::StrAppend(&c, repeat_s(v_block, v_tail_block));
  };

  const std::string kv_dst_expr = [&]() -> std::string {
    if (s_splits <= 1) {
      return "idx";
    }
    if (rem_s > 0 && rem_s < s_splits - 1) {
      const int split_sl = rem_s * block_s;
      return absl::StrCat("idx + min(sl / ", block_s, ", ", rem_s,
                          ") + max(0, sl - ", split_sl, ") / ", floor_s);
    }
    return absl::StrCat("idx + sl / ", block_s);
  }();

  absl::StrAppend(&c, "  int key_base = 0;\n", "  for (; key_base + ", b_c,
                  " <= args.k.Width(); key_base += ", b_c, ") {\n",
                  "    for (int idx = tid; idx < ", slices * b_c,
                  "; idx += ", threads, ") {\n", "      int k_off = idx % ",
                  b_c, ";\n", "      int sl = idx / ", b_c, ";\n",
                  "      k_local[", kv_dst_expr, "] = ucl::Convert<", vec_type,
                  ">(args.k.Read(key_base + k_off, kv_head, sl));\n",
                  "      v_local[", kv_dst_expr, "] = ucl::Convert<", vec_type,
                  ">(args.v.Read(key_base + k_off, kv_head, sl));\n", "    }\n",
                  "    ucl::SyncThreads<WorkGroup, Local>();\n",
                  "    for (int sub = 0; sub < ", b_c, "; sub += 4) {\n");
  emit_substep(/*is_interior=*/true, b_c, "key_base + sub", " + sub");
  absl::StrAppend(&c, "    }\n", "    ucl::SyncThreads<WorkGroup, Local>();\n",
                  "  }\n");

  absl::StrAppend(
      &c, "  for (; key_base < args.k.Width(); key_base += 4) {\n",
      "    for (int idx = tid; idx < ", slices * 4, "; idx += ", threads,
      ") {\n", "      int k_off = idx % 4;\n", "      int sl = idx / 4;\n",
      "      int k_idx = key_base + k_off;\n", "      k_local[", kv_dst_expr,
      "] = (k_idx < args.k.Width()) ? ucl::Convert<", vec_type,
      ">(args.k.Read(k_idx, kv_head, sl)) : ucl::Init<", vec_type, ">(0.0f);\n",
      "      v_local[", kv_dst_expr,
      "] = (k_idx < args.v.Width()) ? ucl::Convert<", vec_type,
      ">(args.v.Read(k_idx, kv_head, sl)) : ucl::Init<", vec_type, ">(0.0f);\n",
      "    }\n", "    ucl::SyncThreads<WorkGroup, Local>();\n");
  emit_substep(/*is_interior=*/false, 4, "key_base", "");
  absl::StrAppend(
      &c, "    ucl::SyncThreads<WorkGroup, Local>();\n", "  }\n\n",
      "  if (q_idx < args.dst.Width()) {\n",
      repeat_s(
          "    {\n"
          "      args.dst::type res = ucl::Convert<args.dst::type>(acc_$s);\n"
          "      args.dst.Write(res, q_idx, q_head, s_base + $s);\n"
          "    }\n",
          absl::StrCat(
              "    if (s_grp < ", rem_s,
              ") {\n"
              "      args.dst::type res = "
              "ucl::Convert<args.dst::type>(acc_$s);\n"
              "      args.dst.Write(res, q_idx, q_head, s_base + $s);\n"
              "    }\n")),
      "  }\n}\n");
  return c;
}

std::unique_ptr<::ml_drift::GPUOperation> CreateFusedFlashSdpa(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::TensorDescriptor& q_desc,
    const ::ml_drift::TensorDescriptor& k_desc,
    const ::ml_drift::TensorDescriptor& v_desc,
    const ::ml_drift::TensorDescriptor* mask_desc,
    const ::ml_drift::TensorDescriptor& dst_desc,
    const ::ml_drift::ScaledDotProductAttentionAttributes& attr,
    const FlashSdpaConfig& config) {
  const ::ml_drift::BHWC q_shape = q_desc.GetBHWCShape();
  const ::ml_drift::BHWC k_shape = k_desc.GetBHWCShape();
  const int gqa_ratio = q_shape.h / k_shape.h;
  const float scale = attr.scale.value_or(1.0f);

  FusedFlashSdpaOp op(config.threads, config.b_r);
  op.args_.AddFloat("scale", scale);
  op.AddSrcTensor("q", q_desc);
  op.AddSrcTensor("k", k_desc);
  op.AddSrcTensor("v", v_desc);

  const bool has_mask = (mask_desc != nullptr);
  const bool is_bool_mask =
      has_mask && (mask_desc->GetDataType() == ::ml_drift::DataType::kBool);
  const bool mask_has_batch =
      has_mask && mask_desc->HasAxis(::ml_drift::Axis::kBatch);
  if (has_mask) {
    op.AddSrcTensor("mask", *mask_desc);
  }
  op.AddDstTensor("dst", dst_desc);

  const bool has_batch = q_desc.HasAxis(::ml_drift::Axis::kBatch);
  if (gpu_info.IsApiOpenCl() && gpu_info.IsMali()) {
    op.compiler_options_.push_back(
        ::ml_drift::CompilerOptions::kClRegisterAllocation64);
  }
  op.code_ = GenerateFlashSdpaCode(gpu_info, config, gqa_ratio, has_batch,
                                   has_mask, is_bool_mask, mask_has_batch);
  return std::make_unique<FusedFlashSdpaOp>(std::move(op));
}

}  // namespace

// Validates whether the tensor shapes, data types, MHA/GQA head counts, mask
// broadcasting dimensions, and device work-group / `__local` memory limits are
// supported by `FusedFlashSdpaOp` (see `sdpa_kernel.h` for full contract).
bool IsSupportedFlashSdpa(const ::ml_drift::GpuInfo& gpu_info,
                          const ::ml_drift::TensorDescriptor& q_desc,
                          const ::ml_drift::TensorDescriptor& k_desc,
                          const ::ml_drift::TensorDescriptor& v_desc,
                          const ::ml_drift::TensorDescriptor* mask_desc,
                          const ::ml_drift::TensorDescriptor& dst_desc,
                          bool use_f32_accum) {
  if (!gpu_info.IsApiOpenCl() && !gpu_info.IsApiMetal() &&
      !gpu_info.IsApiWebGpu()) {
    return false;
  }
  const ::ml_drift::BHWC q_shape = q_desc.GetBHWCShape();
  const ::ml_drift::BHWC k_shape = k_desc.GetBHWCShape();
  const ::ml_drift::BHWC v_shape = v_desc.GetBHWCShape();
  const ::ml_drift::BHWC dst_shape = dst_desc.GetBHWCShape();

  // Require `T > 1` (`q_shape.h > 1`) so single-token decode (`T == 1`) falls
  // back to ML Drift's decode-parallel path instead of launching a 64-thread
  // work-group with only 1 active query row.
  if (q_shape.b <= 0 || q_shape.h <= 1 || k_shape.h <= 0) {
    return false;
  }
  auto is_float_type = [](::ml_drift::DataType dt) {
    return dt == ::ml_drift::DataType::kFloat16 ||
           dt == ::ml_drift::DataType::kFloat32;
  };
  if (!is_float_type(q_desc.GetDataType()) ||
      !is_float_type(k_desc.GetDataType()) ||
      !is_float_type(v_desc.GetDataType()) ||
      !is_float_type(dst_desc.GetDataType())) {
    return false;
  }

  // Batch size `B` and head dimension `H` (`c`) must match across Q, K, V, Dst.
  if (q_shape.b != k_shape.b || q_shape.b != v_shape.b ||
      q_shape.b != dst_shape.b) {
    return false;
  }
  if (q_shape.c != k_shape.c || q_shape.c != v_shape.c ||
      q_shape.c != dst_shape.c) {
    return false;
  }
  // `Dst` must match `Q` in query length `T` (`h`) and query heads `N_q` (`w`),
  // and `V` must match `K` in key length `S` (`h`) and KV heads `N_kv` (`w`).
  if (q_shape.h != dst_shape.h || q_shape.w != dst_shape.w) {
    return false;
  }
  if (k_shape.h != v_shape.h || k_shape.w != v_shape.w) {
    return false;
  }
  // Head dimension `H` must be a multiple of 4 (vectorized 4-channel slices)
  // and at most 128 (`<= 32` slices; `<= 16` `half4` or `<= 8` `float4` slices
  // per thread with `s_splits` in `{1, 2, 4}`).
  if (q_shape.c <= 0 || q_shape.c % 4 != 0 || q_shape.c > 128) {
    return false;
  }
  // Query heads `N_q` (`q_shape.w`) must be a positive multiple of KV heads
  // `N_kv` (`k_shape.w`) for MHA (`N_q == N_kv`) or GQA (`N_q % N_kv == 0`).
  if (k_shape.w <= 0 || q_shape.w < k_shape.w || q_shape.w % k_shape.w != 0) {
    return false;
  }
  // Defer unmasked MHA with `T >= 1024` and small head dimension (`H <= 40`) to
  // ML Drift's specialized `ConvSoftmaxConv` kernel when supported.
  if (mask_desc == nullptr && q_shape.w == k_shape.w && q_shape.h >= 1024) {
    const auto precision =
        (q_desc.GetDataType() == ::ml_drift::DataType::kFloat16)
            ? (use_f32_accum ? ::ml_drift::CalculationsPrecision::kF32F16
                             : ::ml_drift::CalculationsPrecision::kF16)
            : ::ml_drift::CalculationsPrecision::kF32;
    if (::ml_drift::IsConvSoftmaxConvSupported(gpu_info, precision, q_shape.c,
                                               k_shape.h, dst_shape.c)) {
      return false;
    }
  }
  // Optional mask has shape `[B_m, N_m, T_m, S]` (`BHWC`: `b=B_m`, `h=N_m`,
  // `w=T_m`, `c=S`), covering all `S` keys and broadcasting across batch,
  // heads, and query tokens.
  if (mask_desc != nullptr) {
    const ::ml_drift::DataType mask_dt = mask_desc->GetDataType();
    if (mask_dt != ::ml_drift::DataType::kBool && !is_float_type(mask_dt)) {
      return false;
    }
    const ::ml_drift::BHWC mask_shape = mask_desc->GetBHWCShape();
    if ((mask_shape.b != 1 && mask_shape.b != q_shape.b) ||
        (mask_shape.h != 1 && mask_shape.h != q_shape.w) ||
        (mask_shape.w != 1 && mask_shape.w != q_shape.h) ||
        mask_shape.c != k_shape.h) {
      return false;
    }
  }
  const FlashSdpaConfig config =
      GetFlashSdpaConfig(gpu_info, q_desc, use_f32_accum);
  return config.threads > 0 && config.b_r > 0 && config.b_c > 0;
}

absl::Status BuildSdpaGpuGraph(
    const std::vector<uint32_t>& input_ids, uint32_t output_id,
    const ::ml_drift::ScaledDotProductAttentionAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() < 3 || input_ids.size() > 4) {
    return absl::UnimplementedError(
        "SDPA expects 3 or 4 inputs (Q, K, V, [mask]).");
  }
  ABSL_ASSIGN_OR_RETURN(auto q, model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(auto k, model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(auto v, model_builder->GetTensor(input_ids[2]));
  ABSL_ASSIGN_OR_RETURN(auto output_ref, model_builder->GetTensor(output_id));

  const ::ml_drift::TensorDescriptor* mask_desc = nullptr;
  ::ml_drift::GpuModelBuilder::TensorHandle mask;
  if (input_ids.size() == 4) {
    ABSL_ASSIGN_OR_RETURN(mask, model_builder->GetTensor(input_ids[3]));
    mask_desc = &mask.tensor_desc;
  }

  const bool use_f32_accum =
      model_builder->GetConvPrecision(q.tensor_desc.GetDataType()) !=
      ::ml_drift::CalculationsPrecision::kF16;
  if (!IsSupportedFlashSdpa(model_builder->gpu_info(), q.tensor_desc,
                            k.tensor_desc, v.tensor_desc, mask_desc,
                            output_ref.tensor_desc, use_f32_accum)) {
    return absl::UnimplementedError(
        "Fused Flash SDPA does not support this shape or device.");
  }

  const FlashSdpaConfig config = GetFlashSdpaConfig(
      model_builder->gpu_info(), q.tensor_desc, use_f32_accum);

  // Transpose Q, K, V from `[B, T, N, H]` to `[B, N, T, H]` so that the
  // sequence dimension `T` is placed on `Width` (`x`), which is unit-stride
  // contiguous in `PhysicalLayout1D::kDCHWBC4`.
  auto q_t = model_builder->Transpose(q, ::ml_drift::BHWC(0, 2, 1, 3));
  auto k_t = model_builder->Transpose(k, ::ml_drift::BHWC(0, 2, 1, 3));
  auto v_t = model_builder->Transpose(v, ::ml_drift::BHWC(0, 2, 1, 3));

  const ::ml_drift::BHWC dst_shape = output_ref.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC dst_t_shape(dst_shape.b, dst_shape.w, dst_shape.h,
                                     dst_shape.c);
  auto dst_t = model_builder->AddTensor(dst_t_shape,
                                        output_ref.tensor_desc.GetDataType());

  auto op = CreateFusedFlashSdpa(model_builder->gpu_info(), q_t.tensor_desc,
                                 k_t.tensor_desc, v_t.tensor_desc, mask_desc,
                                 dst_t.tensor_desc, attr, config);
  std::vector<::ml_drift::GpuModelBuilder::TensorHandle> src_tensors = {
      q_t, k_t, v_t};
  if (mask_desc != nullptr) {
    src_tensors.push_back(mask);
  }
  model_builder->AddGpuOperation(src_tensors, {dst_t}, std::move(op),
                                 "flash_sdpa");
  auto out = model_builder->Transpose(dst_t, ::ml_drift::BHWC(0, 2, 1, 3));
  return model_builder->UpdateOutputTensor(out, output_id);
}

absl::Status CreateSdpaFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                const std::vector<::ml_drift::Value*>& outputs,
                                const ::ml_drift::Node& node,
                                ::ml_drift::GpuModelBuilder* model_builder) {
  if (outputs.size() != 1) {
    return absl::UnimplementedError("SDPA expects 1 output.");
  }
  ::ml_drift::ScaledDotProductAttentionAttributes attr;
  if (node.operation.attributes.has_value()) {
    if (const auto* sdpa_attr =
            std::any_cast<::ml_drift::ScaledDotProductAttentionAttributes>(
                &node.operation.attributes)) {
      attr = *sdpa_attr;
    }
  }
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) {
    input_ids.push_back(input->id);
  }
  return BuildSdpaGpuGraph(input_ids, outputs[0]->id, attr, model_builder);
}

absl::Status CreateSdpaFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& op,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (outputs.size() != 1) {
    return absl::UnimplementedError("SDPA expects 1 output.");
  }
  ::ml_drift::ScaledDotProductAttentionAttributes attr;
  if (op.attr.has_value()) {
    if (const auto* sdpa_attr =
            std::any_cast<::ml_drift::ScaledDotProductAttentionAttributes>(
                &op.attr)) {
      attr = *sdpa_attr;
    }
  }
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) {
    input_ids.push_back(input->id);
  }
  return BuildSdpaGpuGraph(input_ids, outputs[0]->id, attr, model_builder);
}

}  // namespace litert::ml_drift
