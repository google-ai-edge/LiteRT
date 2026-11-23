// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/qsa_indexer_update_kernel.h"
#include "ml_drift_delegate/delegate/composite/qsa_indexer_update_parser.h"
#include "ml_drift_delegate/delegate/composite/sparse_sdpa_transposed_kernel.h"
#include "ml_drift_delegate/delegate/composite/sparse_sdpa_transposed_parser.h"

namespace litert::ml_drift {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

std::vector<float> MakePatternData(int n, int seed) {
  std::vector<float> out(n);
  for (int i = 0; i < n; ++i) {
    out[i] = 0.25f * std::sin(0.37f * static_cast<float>(i + seed));
  }
  return out;
}

std::vector<float> ReferenceSparseGqa(
    const std::vector<float>& q, const std::vector<float>& k_cache,
    const std::vector<float>& v_cache, const std::vector<int>& selected_blocks,
    const std::vector<float>& mask, int num_q_heads, int num_kv_heads,
    int seq_len, int head_dim, int cache_len, int block_topk, float scale) {
  const int num_groups = num_q_heads / num_kv_heads;
  std::vector<float> out(seq_len * num_q_heads * head_dim, 0.0f);
  for (int qp = 0; qp < seq_len; ++qp) {
    std::vector<bool> keep_tok(cache_len, false);
    for (int k = 0; k < block_topk; ++k) {
      int b = selected_blocks[qp * block_topk + k];
      if (b >= 0 && b < cache_len / 4) {
        for (int r = 0; r < 4; ++r) {
          keep_tok[b * 4 + r] = true;
        }
      }
    }
    for (int b = 0; b < cache_len / 4; ++b) {
      int vis_cnt = 0;
      for (int r = 0; r < 4; ++r) {
        if (mask[qp * cache_len + b * 4 + r] == 0.0f) ++vis_cnt;
      }
      if (vis_cnt > 0 && vis_cnt < 4) {
        for (int r = 0; r < 4; ++r) {
          if (mask[qp * cache_len + b * 4 + r] == 0.0f) {
            keep_tok[b * 4 + r] = true;
          }
        }
      }
    }

    for (int qh = 0; qh < num_q_heads; ++qh) {
      const int kvh = qh / num_groups;
      std::vector<float> logits(cache_len,
                                -std::numeric_limits<float>::infinity());
      float max_l = -std::numeric_limits<float>::infinity();
      for (int t = 0; t < cache_len; ++t) {
        if (!keep_tok[t] || mask[qp * cache_len + t] != 0.0f) continue;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          float qv = q[(qh * seq_len + qp) * head_dim + d];
          float kv = k_cache[(kvh * cache_len + t) * head_dim + d];
          dot += qv * kv;
        }
        logits[t] = dot * scale;
        max_l = std::max(max_l, logits[t]);
      }
      if (!std::isfinite(max_l)) {
        continue;
      }
      float sum = 0.0f;
      std::vector<float> probs(cache_len, 0.0f);
      for (int t = 0; t < cache_len; ++t) {
        if (std::isfinite(logits[t])) {
          probs[t] = std::exp(logits[t] - max_l);
          sum += probs[t];
        }
      }
      for (int d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (int t = 0; t < cache_len; ++t) {
          if (probs[t] > 0.0f) {
            float vv = v_cache[(kvh * head_dim + d) * cache_len + t];
            acc += (probs[t] / sum) * vv;
          }
        }
        out[(qp * num_q_heads + qh) * head_dim + d] = acc;
      }
    }
  }
  return out;
}

std::vector<::ml_drift::GpuInfo> MakeAllFiveGpuApiInfos() {
  std::vector<::ml_drift::GpuInfo> infos;

  ::ml_drift::GpuInfo metal;
  metal.gpu_api = ::ml_drift::GpuApi::kMetal;
  infos.push_back(metal);

  ::ml_drift::GpuInfo metal_simd;
  metal_simd.gpu_api = ::ml_drift::GpuApi::kMetal;
  metal_simd.supported_wave_sizes = {32};
  infos.push_back(metal_simd);

  ::ml_drift::GpuInfo opencl;
  opencl.gpu_api = ::ml_drift::GpuApi::kOpenCl;
  opencl.opencl_info.cl_version = ::ml_drift::OpenClVersion::kCl2_0;
  opencl.opencl_info.supports_fp16 = true;
  infos.push_back(opencl);

  ::ml_drift::GpuInfo vulkan;
  vulkan.gpu_api = ::ml_drift::GpuApi::kVulkan;
  vulkan.vulkan_info.extensions = {"VK_KHR_shader_float16_int8"};
  infos.push_back(vulkan);

  ::ml_drift::GpuInfo opengl;
  opengl.gpu_api = ::ml_drift::GpuApi::kOpenGl;
  infos.push_back(opengl);

  ::ml_drift::GpuInfo webgpu;
  webgpu.gpu_api = ::ml_drift::GpuApi::kWebGpu;
  webgpu.webgpu_info.supports_fp16 = true;
  infos.push_back(webgpu);

  return infos;
}

TEST(QsaCompositesCodegenTest, AssemblesAcrossAllFiveGpuApisF32AndF16) {
  for (const ::ml_drift::GpuInfo& gpu_info : MakeAllFiveGpuApiInfos()) {
    for (const auto storage : {::ml_drift::TensorStorageType::kBuffer,
                               ::ml_drift::TensorStorageType::kTexture2D}) {
      for (auto [precision, dtype] :
           std::vector<std::pair<::ml_drift::CalculationsPrecision,
                                 ::ml_drift::DataType>>{
               {::ml_drift::CalculationsPrecision::kF32,
                ::ml_drift::DataType::kFloat32},
               {::ml_drift::CalculationsPrecision::kF16,
                ::ml_drift::DataType::kFloat16}}) {
        for (bool bool_mask : {false, true}) {
          for (bool from_cache_update : {false, true}) {
            for (int seq_len : {1, 16}) {
              ::ml_drift::GpuModelBuilder builder(gpu_info, /*hints=*/{},
                                                  precision, storage);
              auto q_idx = builder.AddTensor(
                  ::ml_drift::BHWC(1, seq_len, 8, 128), dtype);
              auto k_idx_new = builder.AddTensor(
                  ::ml_drift::BHWC(1, 1, seq_len, 128), dtype);
              // 1D TFLite shape BHWC(D_idx, 1, 1, 1).
              auto k_norm_w =
                  builder.AddTensor(::ml_drift::BHWC(128, 1, 1, 1), dtype);
              auto b_cos =
                  builder.AddTensor(::ml_drift::BHWC(1, 1, 8192, 64), dtype);
              auto b_sin =
                  builder.AddTensor(::ml_drift::BHWC(1, 1, 8192, 64), dtype);
              auto start_pos = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 1),
                                                 ::ml_drift::DataType::kInt32);
              auto mask = builder.AddTensor(
                  ::ml_drift::BHWC(1, 1, seq_len, 32768),
                  bool_mask ? ::ml_drift::DataType::kBool : dtype);
              auto topk = builder.AddTensor(::ml_drift::BHWC(1, 1, seq_len, 16),
                                            ::ml_drift::DataType::kInt32);
              // External in-place state buffers (`new_ring_k`,
              // `new_block_k_pool`) always use BUFFER storage (`max_blocks =
              // 8192` also exercises `scores_tmp` under default `kTexture2D`).
              ::ml_drift::TensorDescriptor ring_desc(
                  dtype, ::ml_drift::TensorStorageType::kBuffer,
                  ::ml_drift::Layout::kBHWC);
              ring_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 4, 128));
              auto new_ring = builder.AddTensor(ring_desc);
              ::ml_drift::TensorDescriptor pool_desc(
                  dtype, ::ml_drift::TensorStorageType::kBuffer,
                  ::ml_drift::Layout::kBHWC);
              pool_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 8192, 128));
              auto new_pool = builder.AddTensor(pool_desc);

              auto q_main = builder.AddTensor(
                  ::ml_drift::BHWC(1, 8, seq_len, 256), dtype);
              auto kc =
                  builder.AddTensor(::ml_drift::BHWC(1, 2, 32768, 256), dtype);
              auto vc =
                  builder.AddTensor(::ml_drift::BHWC(1, 2, 256, 32768), dtype);
              auto attn_out = builder.AddTensor(
                  ::ml_drift::BHWC(1, seq_len, 8, 256), dtype);
              auto gate = builder.AddTensor(
                  ::ml_drift::BHWC(1, seq_len, 8, 256), dtype);

              QsaIndexerUpdateAttributes idx_attr{
                  .compress_ratio = 4,
                  .block_topk = 16,
                  .rms_norm_eps = 1e-6f,
                  .scale = 0.088388f,
              };
              ASSERT_TRUE(BuildQsaIndexerUpdateGpuGraph(
                              {q_idx.id, k_idx_new.id, k_norm_w.id, b_cos.id,
                               b_sin.id, start_pos.id, mask.id},
                              {topk.id, new_ring.id, new_pool.id}, idx_attr,
                              &builder)
                              .ok());

              SparseSdpaTransposedAttributes sdpa_attr{
                  .compress_ratio = 4,
                  .block_topk = 16,
                  .scale = 0.0625f,
                  .k_ts_idx = 2,
                  .v_ts_idx = 3,
                  .from_cache_update = from_cache_update,
              };
              ASSERT_TRUE(BuildSparseSdpaTransposedGpuGraph(
                              {q_main.id, kc.id, vc.id, topk.id, mask.id},
                              {attn_out.id}, sdpa_attr, &builder)
                              .ok());
              // Exercise post-op elementwise fusion (`ResolveLinking`) onto
              // both Split-KV combine (`seq_len = 1`) and single-pass
              // (`seq_len = 16`).
              auto gated_out = builder.Multiplication(attn_out, gate);

              ::ml_drift::GpuModel gpu_model;
              const absl::Status status = builder.GetGpuModel(
                  std::vector<std::pair<uint32_t, uint32_t>>{{q_idx.id, 0},
                                                             {k_idx_new.id, 1},
                                                             {k_norm_w.id, 2},
                                                             {b_cos.id, 3},
                                                             {b_sin.id, 4},
                                                             {start_pos.id, 5},
                                                             {mask.id, 6},
                                                             {q_main.id, 7},
                                                             {kc.id, 8},
                                                             {vc.id, 9},
                                                             {gate.id, 10}},
                  std::vector<std::pair<uint32_t, uint32_t>>{
                      {gated_out.id, 0}, {new_ring.id, 1}, {new_pool.id, 2}},
                  &gpu_model);
              ASSERT_TRUE(status.ok()) << status;
            }
          }
        }
      }
    }
  }
}

TEST(QsaCompositesCodegenTest,
     RejectsNonBufferStorageOnInPlaceIndexerOutputs) {
  const ::ml_drift::GpuInfo gpu_info = MakeAllFiveGpuApiInfos()[0];
  ::ml_drift::GpuModelBuilder builder(
      gpu_info, /*hints=*/{}, ::ml_drift::CalculationsPrecision::kF32,
      ::ml_drift::TensorStorageType::kTexture2D);
  auto q_idx = builder.AddTensor(::ml_drift::BHWC(1, 1, 4, 16),
                                 ::ml_drift::DataType::kFloat32);
  auto k_idx_new = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 16),
                                     ::ml_drift::DataType::kFloat32);
  auto k_norm_w = builder.AddTensor(::ml_drift::BHWC(16, 1, 1, 1),
                                    ::ml_drift::DataType::kFloat32);
  auto b_cos = builder.AddTensor(::ml_drift::BHWC(1, 1, 8, 8),
                                 ::ml_drift::DataType::kFloat32);
  auto b_sin = builder.AddTensor(::ml_drift::BHWC(1, 1, 8, 8),
                                 ::ml_drift::DataType::kFloat32);
  auto start_pos = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 1),
                                     ::ml_drift::DataType::kInt32);
  auto mask = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 32),
                                ::ml_drift::DataType::kFloat32);
  auto topk = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 2),
                                ::ml_drift::DataType::kInt32);
  // Explicitly force `kTexture2D` on `new_ring` (`AddTensor(BHWC)` falls back
  // to `kBuffer` when `image2d_max_width` is unset on `gpu_info`).
  ::ml_drift::TensorDescriptor ring_tex_desc(
      ::ml_drift::DataType::kFloat32, ::ml_drift::TensorStorageType::kTexture2D,
      ::ml_drift::Layout::kBHWC);
  ring_tex_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 4, 16));
  auto new_ring = builder.AddTensor(ring_tex_desc);
  auto new_pool = builder.AddTensor(::ml_drift::BHWC(1, 1, 8, 16),
                                    ::ml_drift::DataType::kFloat32);

  QsaIndexerUpdateAttributes attr{
      .compress_ratio = 4,
      .block_topk = 2,
      .rms_norm_eps = 1e-6f,
      .scale = 0.25f,
  };
  EXPECT_TRUE(absl::IsUnimplemented(BuildQsaIndexerUpdateGpuGraph(
      {q_idx.id, k_idx_new.id, k_norm_w.id, b_cos.id, b_sin.id, start_pos.id,
       mask.id},
      {topk.id, new_ring.id, new_pool.id}, attr, &builder)));
}

class QsaCompositesKernelTest : public ::testing::Test {
 public:
  void SetUp() override {
    if (!exec_env) {
      GTEST_SKIP() << "TestExecutionEnvironment not initialized.";
    }
    if (!exec_env->IsStorageSupported(::ml_drift::TensorStorageType::kBuffer,
                                      ::ml_drift::DataType::kFloat32)) {
      GTEST_SKIP() << "Float32 BUFFER storage not supported.";
    }
  }
};

TEST_F(QsaCompositesKernelTest, SparseSdpaTransposedLargeHeadDimMatchesCpu) {
  // D = 160 (> 128, so `tid + 32 < head_slices` is exercised), NaN in masked V.
  const int H_q = 4, H_kv = 2, S = 1, D = 160, cache_len = 16, block_topk = 2;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto storage = ::ml_drift::TensorStorageType::kBuffer;

  ::ml_drift::GpuModelBuilder builder(exec_env->GetGpuInfo(), /*hints=*/{},
                                      ::ml_drift::CalculationsPrecision::kF32,
                                      storage);
  auto q = builder.AddTensor(::ml_drift::BHWC(1, H_q, S, D),
                             ::ml_drift::DataType::kFloat32);
  auto kc = builder.AddTensor(::ml_drift::BHWC(1, H_kv, cache_len, D),
                              ::ml_drift::DataType::kFloat32);
  auto vc = builder.AddTensor(::ml_drift::BHWC(1, H_kv, D, cache_len),
                              ::ml_drift::DataType::kFloat32);
  auto mask = builder.AddTensor(::ml_drift::BHWC(1, 1, S, cache_len),
                                ::ml_drift::DataType::kFloat32);
  auto out = builder.AddTensor(::ml_drift::BHWC(1, S, H_q, D),
                               ::ml_drift::DataType::kFloat32);

  std::vector<int> selected_blocks = {0, 2};
  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                     ::ml_drift::DataType::kInt32>
      topk_cpu;
  topk_cpu.shape = ::ml_drift::BHWC(1, 1, S, block_topk);
  topk_cpu.data = selected_blocks;
  ::ml_drift::TensorDescriptor topk_desc(::ml_drift::DataType::kInt32, storage,
                                         ::ml_drift::Layout::kBHWC);
  topk_desc.UploadData(topk_cpu);
  auto topk = builder.AddConstantTensor(std::move(topk_desc));

  SparseSdpaTransposedAttributes attr{
      .compress_ratio = 4,
      .block_topk = block_topk,
      .scale = scale,
      .k_ts_idx = 2,
      .v_ts_idx = 3,
      .from_cache_update = false,
  };
  ASSERT_TRUE(BuildSparseSdpaTransposedGpuGraph(
                  {q.id, kc.id, vc.id, topk.id, mask.id}, {out.id}, attr,
                  &builder)
                  .ok());
  ::ml_drift::GpuModel gpu_model;
  ASSERT_TRUE(
      builder
          .GetGpuModel(
              std::vector<std::pair<uint32_t, uint32_t>>{
                  {q.id, 0}, {kc.id, 1}, {vc.id, 2}, {mask.id, 3}},
              std::vector<std::pair<uint32_t, uint32_t>>{{out.id, 0}},
              &gpu_model)
          .ok());

  std::vector<float> q_data = MakePatternData(H_q * S * D, 1);
  std::vector<float> kc_data = MakePatternData(H_kv * cache_len * D, 7);
  std::vector<float> vc_data = MakePatternData(H_kv * D * cache_len, 13);
  // Poison unwritten KV slots 13..15 with NaN to ensure 0 * NaN never occurs.
  for (int kvh = 0; kvh < H_kv; ++kvh) {
    for (int d = 0; d < D; ++d) {
      for (int t = 13; t < cache_len; ++t) {
        vc_data[(kvh * D + d) * cache_len + t] =
            std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  std::vector<float> mask_data(S * cache_len, -10000.0f);
  for (int t = 0; t <= 12; ++t) mask_data[t] = 0.0f;

  std::vector<float> expected = ReferenceSparseGqa(
      q_data, kc_data, vc_data, selected_blocks, mask_data, H_q, H_kv, S, D,
      cache_len, block_topk, scale);

  auto make_tf32 = [](::ml_drift::BHWC shape, std::vector<float> data) {
    ::ml_drift::TensorFloat32 t;
    t.shape = shape;
    t.data = std::move(data);
    return t;
  };
  std::vector<::ml_drift::TensorFloat32> src_cpu = {
      make_tf32(::ml_drift::BHWC(1, H_q, S, D), q_data),
      make_tf32(::ml_drift::BHWC(1, H_kv, cache_len, D), kc_data),
      make_tf32(::ml_drift::BHWC(1, H_kv, D, cache_len), vc_data),
      make_tf32(::ml_drift::BHWC(1, 1, S, cache_len), mask_data),
  };
  ::ml_drift::TensorFloat32 out_cpu =
      make_tf32(::ml_drift::BHWC(1, S, H_q, D), {});
  std::vector<::ml_drift::TensorFloat32*> dst_cpu = {&out_cpu};
  const absl::Status exec_status =
      exec_env->ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model);
  ASSERT_TRUE(exec_status.ok()) << exec_status;
  EXPECT_THAT(out_cpu.data, Pointwise(FloatNear(1e-4f), expected));
}

TEST_F(QsaCompositesKernelTest,
       SparseSdpaTransposedPackedFromCacheUpdateBoolMaskAndFp16) {
  const int H_q = 4, H_kv = 2, S = 2, D = 16, cache_len = 16, block_topk = 2;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto storage = ::ml_drift::TensorStorageType::kBuffer;
  const bool fp16_ok = exec_env->IsStorageSupported(
      storage, ::ml_drift::DataType::kFloat16);
  const auto dtype = fp16_ok ? ::ml_drift::DataType::kFloat16
                             : ::ml_drift::DataType::kFloat32;
  const auto precision = fp16_ok ? ::ml_drift::CalculationsPrecision::kF16
                                 : ::ml_drift::CalculationsPrecision::kF32;

  ::ml_drift::GpuModelBuilder builder(exec_env->GetGpuInfo(), /*hints=*/{},
                                      precision, storage);
  auto q = builder.AddTensor(::ml_drift::BHWC(1, H_q, S, D), dtype);
  // Linear BUFFER descriptors matching `odml.cache_update` packed outputs.
  ::ml_drift::TensorDescriptor kc_desc(dtype, storage,
                                       ::ml_drift::Layout::kBHWC);
  kc_desc.SetBHWCShape(::ml_drift::BHWC(1, H_kv, cache_len, D));
  auto kc = builder.AddTensor(kc_desc);
  ::ml_drift::TensorDescriptor vc_desc(dtype, storage,
                                       ::ml_drift::Layout::kBHWC);
  vc_desc.SetBHWCShape(::ml_drift::BHWC(1, H_kv, D, cache_len));
  auto vc = builder.AddTensor(vc_desc);
  auto out = builder.AddTensor(::ml_drift::BHWC(1, S, H_q, D), dtype);

  std::vector<int> selected_blocks = {0, 1, 0, 2};
  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                     ::ml_drift::DataType::kInt32>
      topk_cpu;
  topk_cpu.shape = ::ml_drift::BHWC(1, 1, S, block_topk);
  topk_cpu.data = selected_blocks;
  ::ml_drift::TensorDescriptor topk_desc(::ml_drift::DataType::kInt32, storage,
                                         ::ml_drift::Layout::kBHWC);
  topk_desc.UploadData(topk_cpu);
  auto topk = builder.AddConstantTensor(std::move(topk_desc));

  std::vector<uint8_t> bool_mask_vec(S * cache_len, 0);
  std::vector<float> float_mask_ref(S * cache_len, -10000.0f);
  for (int t = 0; t <= 12; ++t) {
    bool_mask_vec[0 * cache_len + t] = 1;
    float_mask_ref[0 * cache_len + t] = 0.0f;
  }
  for (int t = 0; t <= 13; ++t) {
    bool_mask_vec[1 * cache_len + t] = 1;
    float_mask_ref[1 * cache_len + t] = 0.0f;
  }
  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                     ::ml_drift::DataType::kBool>
      mask_cpu;
  mask_cpu.shape = ::ml_drift::BHWC(1, 1, S, cache_len);
  mask_cpu.data = bool_mask_vec;
  ::ml_drift::TensorDescriptor mask_desc(::ml_drift::DataType::kBool, storage,
                                         ::ml_drift::Layout::kBHWC);
  mask_desc.UploadData(mask_cpu);
  auto mask = builder.AddConstantTensor(std::move(mask_desc));

  SparseSdpaTransposedAttributes attr{
      .compress_ratio = 4,
      .block_topk = block_topk,
      .scale = scale,
      .k_ts_idx = 2,
      .v_ts_idx = 3,
      .from_cache_update = true,
  };
  ASSERT_TRUE(BuildSparseSdpaTransposedGpuGraph(
                  {q.id, kc.id, vc.id, topk.id, mask.id}, {out.id}, attr,
                  &builder)
                  .ok());
  ::ml_drift::GpuModel gpu_model;
  ASSERT_TRUE(
      builder
          .GetGpuModel(std::vector<std::pair<uint32_t, uint32_t>>{
                           {q.id, 0}, {kc.id, 1}, {vc.id, 2}},
                       std::vector<std::pair<uint32_t, uint32_t>>{{out.id, 0}},
                       &gpu_model)
          .ok());

  std::vector<float> q_data = MakePatternData(H_q * S * D, 4);
  std::vector<float> kc_logical = MakePatternData(H_kv * cache_len * D, 8);
  std::vector<float> vc_logical = MakePatternData(H_kv * D * cache_len, 12);
  // UploadData on BHWC(1, H_kv, cache_len, D) permutes slices in channel-major
  // order; for a linear `Read(idx)` test we want GPU buffer[vec_idx] to equal
  // `PackKeyCacheForCacheUpdate`, so pass BHWC(1, 1, 1, total) or map through
  // `RearrangeK` / `RearrangeV` from `fused_sdpa_cache_update_kernel_test.cc`.
  std::vector<float> kc_packed(kc_logical.size(), 0.0f);
  for (int bk = 0; bk < H_kv; ++bk) {
    for (int s = 0; s < cache_len; ++s) {
      for (int h = 0; h < D; ++h) {
        int orig_idx = (bk * cache_len + s) * D + h;
        int linear_hb = bk * (D / 4) + (h / 4);
        int bk_src = linear_hb % H_kv;
        int h_src = (linear_hb / H_kv) * 4 + (h % 4);
        kc_packed[(bk_src * cache_len + s) * D + h_src] = kc_logical[orig_idx];
      }
    }
  }
  std::vector<float> vc_packed(vc_logical.size(), 0.0f);
  for (int bk = 0; bk < H_kv; ++bk) {
    for (int h = 0; h < D; ++h) {
      for (int s = 0; s < cache_len; ++s) {
        int orig_idx = (bk * D + h) * cache_len + s;
        int rhs = ((bk * (cache_len / 4) + (s / 4)) * (D / 4) + (h / 4)) * 4 +
                  (s % 4);
        int h_src = rhs % D;
        int temp = rhs / D;
        int bk_src = temp % H_kv;
        int s_src = (temp / H_kv) * 4 + (h % 4);
        vc_packed[(bk_src * D + h_src) * cache_len + s_src] =
            vc_logical[orig_idx];
      }
    }
  }

  std::vector<float> expected = ReferenceSparseGqa(
      q_data, kc_logical, vc_logical, selected_blocks, float_mask_ref, H_q,
      H_kv, S, D, cache_len, block_topk, scale);

  auto make_tf32 = [](::ml_drift::BHWC shape, std::vector<float> data) {
    ::ml_drift::TensorFloat32 t;
    t.shape = shape;
    t.data = std::move(data);
    return t;
  };
  std::vector<::ml_drift::TensorFloat32> src_cpu = {
      make_tf32(::ml_drift::BHWC(1, H_q, S, D), q_data),
      make_tf32(::ml_drift::BHWC(1, H_kv, cache_len, D), kc_packed),
      make_tf32(::ml_drift::BHWC(1, H_kv, D, cache_len), vc_packed),
  };
  ::ml_drift::TensorFloat32 out_cpu =
      make_tf32(::ml_drift::BHWC(1, S, H_q, D), {});
  std::vector<::ml_drift::TensorFloat32*> dst_cpu = {&out_cpu};
  const absl::Status exec_status =
      exec_env->ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model);
  ASSERT_TRUE(exec_status.ok()) << exec_status;
  const float tol = fp16_ok ? 2e-3f : 1e-4f;
  EXPECT_THAT(out_cpu.data, Pointwise(FloatNear(tol), expected));
}

TEST_F(QsaCompositesKernelTest,
       EndToEndIndexer1DNormWeightRotaryAndInPlaceStateAliasing) {
  const int H_idx = 5, D_idx = 16, R = 4, max_blocks = 8, cache_len = 32;
  const int rotary_dim = 8, block_topk = 2;
  const int H_q = 4, H_kv = 2, S = 1, D = 16;
  const float idx_scale = 1.0f / std::sqrt(static_cast<float>(D_idx));
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto storage = ::ml_drift::TensorStorageType::kBuffer;

  ::ml_drift::GpuModelBuilder builder(exec_env->GetGpuInfo(), /*hints=*/{},
                                      ::ml_drift::CalculationsPrecision::kF32,
                                      storage);
  auto q_idx = builder.AddTensor(::ml_drift::BHWC(1, S, H_idx, D_idx),
                                 ::ml_drift::DataType::kFloat32);
  auto k_idx_new = builder.AddTensor(::ml_drift::BHWC(1, 1, S, D_idx),
                                     ::ml_drift::DataType::kFloat32);
  auto ring_init = builder.AddTensor(::ml_drift::BHWC(1, 1, R, D_idx),
                                     ::ml_drift::DataType::kFloat32);
  auto pool_init =
      builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, D_idx),
                        ::ml_drift::DataType::kFloat32);
  // Seed `ring_k` and `block_k_pool` before `BuildQsaIndexerUpdateGpuGraph`
  // reads and mutates them in-place on GPU (`kGpuOptimizedInplace`).
  auto ring_k = builder.AddTensor(::ml_drift::BHWC(1, 1, R, D_idx),
                                  ::ml_drift::DataType::kFloat32);
  auto block_k_pool =
      builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, D_idx),
                        ::ml_drift::DataType::kFloat32);
  builder.Copy(ring_init, ring_k);
  builder.Copy(pool_init, block_k_pool);
  // TFLite 1D shape BHWC(D_idx, 1, 1, 1).
  auto k_norm_w = builder.AddTensor(::ml_drift::BHWC(D_idx, 1, 1, 1),
                                    ::ml_drift::DataType::kFloat32);
  auto b_cos = builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim),
                                 ::ml_drift::DataType::kFloat32);
  auto b_sin = builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim),
                                 ::ml_drift::DataType::kFloat32);
  auto mask = builder.AddTensor(::ml_drift::BHWC(1, 1, S, cache_len),
                                ::ml_drift::DataType::kFloat32);
  auto q_main = builder.AddTensor(::ml_drift::BHWC(1, H_q, S, D),
                                  ::ml_drift::DataType::kFloat32);
  auto kc = builder.AddTensor(::ml_drift::BHWC(1, H_kv, cache_len, D),
                              ::ml_drift::DataType::kFloat32);
  auto vc = builder.AddTensor(::ml_drift::BHWC(1, H_kv, D, cache_len),
                              ::ml_drift::DataType::kFloat32);

  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                     ::ml_drift::DataType::kInt32>
      start_pos_cpu;
  start_pos_cpu.shape = ::ml_drift::BHWC(1, 1, 1, 1);
  // Completes block 4 (tokens 16..19), 5 blocks valid.
  start_pos_cpu.data = {19};
  ::ml_drift::TensorDescriptor start_pos_desc(::ml_drift::DataType::kInt32,
                                              storage,
                                              ::ml_drift::Layout::kBHWC);
  start_pos_desc.UploadData(start_pos_cpu);
  auto start_pos = builder.AddConstantTensor(std::move(start_pos_desc));

  auto topk = builder.AddTensor(::ml_drift::BHWC(1, 1, S, block_topk),
                                ::ml_drift::DataType::kInt32);
  auto attn_out = builder.AddTensor(::ml_drift::BHWC(1, S, H_q, D),
                                    ::ml_drift::DataType::kFloat32);

  QsaIndexerUpdateAttributes idx_attr{
      .compress_ratio = R,
      .block_topk = block_topk,
      .rms_norm_eps = 1e-6f,
      .scale = idx_scale,
  };
  ASSERT_TRUE(
      BuildQsaIndexerUpdateGpuGraph(
          {q_idx.id, k_idx_new.id, k_norm_w.id, b_cos.id, b_sin.id,
           start_pos.id, mask.id},
          {topk.id, ring_k.id, block_k_pool.id}, idx_attr, &builder)
          .ok());

  SparseSdpaTransposedAttributes sdpa_attr{
      .compress_ratio = R,
      .block_topk = block_topk,
      .scale = attn_scale,
      .k_ts_idx = 2,
      .v_ts_idx = 3,
  };
  ASSERT_TRUE(
      BuildSparseSdpaTransposedGpuGraph(
          {q_main.id, kc.id, vc.id, topk.id, mask.id}, {attn_out.id}, sdpa_attr,
          &builder)
          .ok());

  ::ml_drift::GpuModel gpu_model;
  ASSERT_TRUE(
      builder
          .GetGpuModel(
              std::vector<std::pair<uint32_t, uint32_t>>{
                  {q_idx.id, 0},
                  {k_idx_new.id, 1},
                  {ring_init.id, 2},
                  {pool_init.id, 3},
                  {k_norm_w.id, 4},
                  {b_cos.id, 5},
                  {b_sin.id, 6},
                  {mask.id, 7},
                  {q_main.id, 8},
                  {kc.id, 9},
                  {vc.id, 10}},
              std::vector<std::pair<uint32_t, uint32_t>>{
                  {attn_out.id, 0}, {ring_k.id, 1}, {block_k_pool.id, 2}},
              &gpu_model)
          .ok());

  auto make_tf32 = [](::ml_drift::BHWC shape, std::vector<float> data) {
    ::ml_drift::TensorFloat32 t;
    t.shape = shape;
    t.data = std::move(data);
    return t;
  };
  std::vector<float> mask_data(S * cache_len, -10000.0f);
  for (int t = 0; t <= 19; ++t) mask_data[t] = 0.0f;

  std::vector<float> q_idx_data = MakePatternData(S * H_idx * D_idx, 2);
  std::vector<float> k_idx_new_data = MakePatternData(S * D_idx, 5);
  std::vector<float> ring_k_data = MakePatternData(R * D_idx, 9);
  std::vector<float> pool_data = MakePatternData(max_blocks * D_idx, 11);
  // Make blocks 1 and 3 identical to exercise deterministic tie-breaking too.
  for (int d = 0; d < D_idx; ++d) {
    pool_data[3 * D_idx + d] = pool_data[1 * D_idx + d];
  }
  std::vector<float> k_norm_data = MakePatternData(D_idx, 31);
  for (float& w : k_norm_data) w = 0.8f + w;
  std::vector<float> b_cos_data(max_blocks * rotary_dim);
  std::vector<float> b_sin_data(max_blocks * rotary_dim);
  for (int i = 0; i < max_blocks * rotary_dim; ++i) {
    const float angle = 0.13f * static_cast<float>(i + 1);
    b_cos_data[i] = std::cos(angle);
    b_sin_data[i] = std::sin(angle);
  }
  std::vector<float> q_main_data = MakePatternData(H_q * S * D, 3);
  std::vector<float> kc_data = MakePatternData(H_kv * cache_len * D, 17);
  std::vector<float> vc_data = MakePatternData(H_kv * D * cache_len, 23);

  std::vector<float> expected_ring(R * D_idx, 0.0f);
  std::vector<float> expected_pool = pool_data;
  {
    std::vector<float> pooled(D_idx, 0.0f);
    float sq_sum = 0.0f;
    for (int d = 0; d < D_idx; ++d) {
      float sum_r = k_idx_new_data[d];
      for (int r = 0; r < R - 1; ++r) sum_r += ring_k_data[r * D_idx + d];
      pooled[d] = sum_r / static_cast<float>(R);
      sq_sum += pooled[d] * pooled[d];
    }
    float inv_rms =
        1.0f / std::sqrt(sq_sum / static_cast<float>(D_idx) + 1e-6f);
    for (int d = 0; d < D_idx; ++d) {
      pooled[d] = pooled[d] * inv_rms * k_norm_data[d];
    }
    const int half_rot = rotary_dim / 2;
    for (int d = 0; d < D_idx; ++d) {
      float out_v = pooled[d];
      if (d < rotary_dim) {
        float c = b_cos_data[4 * rotary_dim + d];
        float s = b_sin_data[4 * rotary_dim + d];
        float partner =
            (d < half_rot) ? -pooled[d + half_rot] : pooled[d - half_rot];
        out_v = out_v * c + partner * s;
      }
      expected_pool[4 * D_idx + d] = out_v;
    }
  }
  std::vector<std::pair<float, int>> block_scores;
  for (int b = 0; b < 5; ++b) {
    float score = 0.0f;
    for (int h = 0; h < H_idx; ++h) {
      float dot = 0.0f;
      for (int d = 0; d < D_idx; ++d) {
        dot += q_idx_data[h * D_idx + d] * expected_pool[b * D_idx + d];
      }
      score += std::max(dot, 0.0f);
    }
    block_scores.push_back({score * idx_scale, b});
  }
  std::sort(block_scores.begin(), block_scores.end(),
            [](const auto& a, const auto& b) {
              if (a.first != b.first) return a.first > b.first;
              return a.second < b.second;
            });
  std::vector<int> selected_blocks = {block_scores[0].second,
                                      block_scores[1].second};
  std::vector<float> expected_attn = ReferenceSparseGqa(
      q_main_data, kc_data, vc_data, selected_blocks, mask_data, H_q, H_kv, S,
      D, cache_len, block_topk, attn_scale);

  std::vector<::ml_drift::TensorFloat32> src_cpu = {
      make_tf32(::ml_drift::BHWC(1, S, H_idx, D_idx), q_idx_data),
      make_tf32(::ml_drift::BHWC(1, 1, S, D_idx), k_idx_new_data),
      make_tf32(::ml_drift::BHWC(1, 1, R, D_idx), ring_k_data),
      make_tf32(::ml_drift::BHWC(1, 1, max_blocks, D_idx), pool_data),
      make_tf32(::ml_drift::BHWC(D_idx, 1, 1, 1), k_norm_data),
      make_tf32(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim), b_cos_data),
      make_tf32(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim), b_sin_data),
      make_tf32(::ml_drift::BHWC(1, 1, S, cache_len), mask_data),
      make_tf32(::ml_drift::BHWC(1, H_q, S, D), q_main_data),
      make_tf32(::ml_drift::BHWC(1, H_kv, cache_len, D), kc_data),
      make_tf32(::ml_drift::BHWC(1, H_kv, D, cache_len), vc_data),
  };
  ::ml_drift::TensorFloat32 out_attn =
      make_tf32(::ml_drift::BHWC(1, S, H_q, D), {});
  ::ml_drift::TensorFloat32 out_ring =
      make_tf32(::ml_drift::BHWC(1, 1, R, D_idx), {});
  ::ml_drift::TensorFloat32 out_pool =
      make_tf32(::ml_drift::BHWC(1, 1, max_blocks, D_idx), {});
  std::vector<::ml_drift::TensorFloat32*> dst_cpu = {
      &out_attn, &out_ring, &out_pool};

  const absl::Status exec_status =
      exec_env->ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model);
  ASSERT_TRUE(exec_status.ok()) << exec_status;
  EXPECT_THAT(out_ring.data, Pointwise(FloatNear(1e-5f), expected_ring));
  EXPECT_THAT(out_pool.data, Pointwise(FloatNear(1e-4f), expected_pool));
  EXPECT_THAT(out_attn.data, Pointwise(FloatNear(1e-4f), expected_attn));
}

TEST_F(QsaCompositesKernelTest, PaddedPrefillDoesNotCorruptRingOrBlockPool) {
  // Prefill S = 8 at pos0 = 0 with only 5 valid tokens (0..4) and 3 padded
  // tokens (5..7): block 0 completes, ring_k[0] = k_idx_new[4], ring_k[1..3]=0,
  // and block 1 in block_k_pool MUST remain untouched despite S = 8!
  const int S = 8, H_idx = 2, D_idx = 8, R = 4, max_blocks = 4, cache_len = 16;
  const int rotary_dim = 8, block_topk = 2;
  const auto storage = ::ml_drift::TensorStorageType::kBuffer;

  ::ml_drift::GpuModelBuilder builder(exec_env->GetGpuInfo(), /*hints=*/{},
                                      ::ml_drift::CalculationsPrecision::kF32,
                                      storage);
  auto q_idx = builder.AddTensor(::ml_drift::BHWC(1, S, H_idx, D_idx),
                                 ::ml_drift::DataType::kFloat32);
  auto k_idx_new = builder.AddTensor(::ml_drift::BHWC(1, 1, S, D_idx),
                                     ::ml_drift::DataType::kFloat32);
  auto ring_init = builder.AddTensor(::ml_drift::BHWC(1, 1, R, D_idx),
                                     ::ml_drift::DataType::kFloat32);
  auto pool_init =
      builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, D_idx),
                        ::ml_drift::DataType::kFloat32);
  auto k_norm_w = builder.AddTensor(::ml_drift::BHWC(D_idx, 1, 1, 1),
                                    ::ml_drift::DataType::kFloat32);
  auto b_cos = builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim),
                                 ::ml_drift::DataType::kFloat32);
  auto b_sin = builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim),
                                 ::ml_drift::DataType::kFloat32);
  auto mask = builder.AddTensor(::ml_drift::BHWC(1, 1, S, cache_len),
                                ::ml_drift::DataType::kFloat32);

  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                     ::ml_drift::DataType::kInt32>
      start_pos_cpu;
  start_pos_cpu.shape = ::ml_drift::BHWC(1, 1, 1, 1);
  start_pos_cpu.data = {0};
  ::ml_drift::TensorDescriptor start_pos_desc(::ml_drift::DataType::kInt32,
                                              storage,
                                              ::ml_drift::Layout::kBHWC);
  start_pos_desc.UploadData(start_pos_cpu);
  auto start_pos = builder.AddConstantTensor(std::move(start_pos_desc));

  auto topk = builder.AddTensor(::ml_drift::BHWC(1, 1, S, block_topk),
                                ::ml_drift::DataType::kInt32);
  auto new_ring = builder.AddTensor(::ml_drift::BHWC(1, 1, R, D_idx),
                                    ::ml_drift::DataType::kFloat32);
  auto new_pool = builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, D_idx),
                                    ::ml_drift::DataType::kFloat32);
  builder.Copy(ring_init, new_ring);
  builder.Copy(pool_init, new_pool);

  QsaIndexerUpdateAttributes idx_attr{
      .compress_ratio = R,
      .block_topk = block_topk,
      .rms_norm_eps = 1e-6f,
      .scale = 0.25f,
  };
  ASSERT_TRUE(
      BuildQsaIndexerUpdateGpuGraph(
          {q_idx.id, k_idx_new.id, k_norm_w.id, b_cos.id, b_sin.id,
           start_pos.id, mask.id},
          {topk.id, new_ring.id, new_pool.id}, idx_attr, &builder)
          .ok());

  ::ml_drift::GpuModel gpu_model;
  ASSERT_TRUE(
      builder
          .GetGpuModel(
              std::vector<std::pair<uint32_t, uint32_t>>{
                  {q_idx.id, 0},
                  {k_idx_new.id, 1},
                  {ring_init.id, 2},
                  {pool_init.id, 3},
                  {k_norm_w.id, 4},
                  {b_cos.id, 5},
                  {b_sin.id, 6},
                  {mask.id, 7}},
              std::vector<std::pair<uint32_t, uint32_t>>{
                  {new_ring.id, 0}, {new_pool.id, 1}},
              &gpu_model)
          .ok());

  std::vector<float> q_idx_data = MakePatternData(S * H_idx * D_idx, 1);
  std::vector<float> k_idx_new_data = MakePatternData(S * D_idx, 2);
  // Put huge values in right-padded tokens 5..7 so any leak into block 1 or
  // ring_k[1..3] fails immediately.
  for (int t = 5; t < S; ++t) {
    for (int d = 0; d < D_idx; ++d) {
      k_idx_new_data[t * D_idx + d] = 999.0f;
    }
  }
  std::vector<float> ring_k_data(R * D_idx, -7.0f);
  std::vector<float> pool_data = MakePatternData(max_blocks * D_idx, 9);
  std::vector<float> k_norm_data(D_idx, 1.0f);
  std::vector<float> b_cos_data(max_blocks * rotary_dim, 1.0f);
  std::vector<float> b_sin_data(max_blocks * rotary_dim, 0.0f);

  std::vector<float> mask_data(S * cache_len, -10000.0f);
  for (int t = 0; t < 5; ++t) {
    for (int kv = 0; kv <= t; ++kv) {
      mask_data[t * cache_len + kv] = 0.0f;
    }
  }

  std::vector<float> expected_ring(R * D_idx, 0.0f);
  for (int d = 0; d < D_idx; ++d) {
    expected_ring[0 * D_idx + d] = k_idx_new_data[4 * D_idx + d];
  }
  std::vector<float> expected_pool = pool_data;
  {
    std::vector<float> pooled(D_idx, 0.0f);
    float sq_sum = 0.0f;
    for (int d = 0; d < D_idx; ++d) {
      float sum_r = 0.0f;
      for (int r = 0; r < 4; ++r) sum_r += k_idx_new_data[r * D_idx + d];
      pooled[d] = sum_r / 4.0f;
      sq_sum += pooled[d] * pooled[d];
    }
    float inv_rms =
        1.0f / std::sqrt(sq_sum / static_cast<float>(D_idx) + 1e-6f);
    for (int d = 0; d < D_idx; ++d) {
      expected_pool[0 * D_idx + d] = pooled[d] * inv_rms;
    }
  }

  auto make_tf32 = [](::ml_drift::BHWC shape, std::vector<float> data) {
    ::ml_drift::TensorFloat32 t;
    t.shape = shape;
    t.data = std::move(data);
    return t;
  };
  std::vector<::ml_drift::TensorFloat32> src_cpu = {
      make_tf32(::ml_drift::BHWC(1, S, H_idx, D_idx), q_idx_data),
      make_tf32(::ml_drift::BHWC(1, 1, S, D_idx), k_idx_new_data),
      make_tf32(::ml_drift::BHWC(1, 1, R, D_idx), ring_k_data),
      make_tf32(::ml_drift::BHWC(1, 1, max_blocks, D_idx), pool_data),
      make_tf32(::ml_drift::BHWC(D_idx, 1, 1, 1), k_norm_data),
      make_tf32(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim), b_cos_data),
      make_tf32(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim), b_sin_data),
      make_tf32(::ml_drift::BHWC(1, 1, S, cache_len), mask_data),
  };
  ::ml_drift::TensorFloat32 out_ring =
      make_tf32(::ml_drift::BHWC(1, 1, R, D_idx), {});
  ::ml_drift::TensorFloat32 out_pool =
      make_tf32(::ml_drift::BHWC(1, 1, max_blocks, D_idx), {});
  std::vector<::ml_drift::TensorFloat32*> dst_cpu = {&out_ring, &out_pool};

  const absl::Status exec_status =
      exec_env->ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model);
  ASSERT_TRUE(exec_status.ok()) << exec_status;
  EXPECT_THAT(out_ring.data, Pointwise(FloatNear(1e-5f), expected_ring));
  EXPECT_THAT(out_pool.data, Pointwise(FloatNear(1e-4f), expected_pool));
}

TEST_F(QsaCompositesKernelTest,
       SparseSdpaSplitKvGqa4VerifyAndPrefillMatchesCpu) {
  // Exercises `gqa_ratio = 4` (`heads_per_wg = 4`), `D = 256`, `block_topk =
  // 16` (`prep_slices = 6` -> `splits = 3` for `S in {1, 5}` and `splits = 1`
  // for `S = 16`), boundary blocks, NaN in masked V, and fused post-op gate.
  const int H_q = 8, H_kv = 2, D = 256, cache_len = 128, block_topk = 16;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto storage = ::ml_drift::TensorStorageType::kBuffer;

  for (int S : {1, 5, 16}) {
    ::ml_drift::GpuModelBuilder builder(exec_env->GetGpuInfo(), /*hints=*/{},
                                        ::ml_drift::CalculationsPrecision::kF32,
                                        storage);
    auto q = builder.AddTensor(::ml_drift::BHWC(1, H_q, S, D),
                               ::ml_drift::DataType::kFloat32);
    auto kc = builder.AddTensor(::ml_drift::BHWC(1, H_kv, cache_len, D),
                                ::ml_drift::DataType::kFloat32);
    auto vc = builder.AddTensor(::ml_drift::BHWC(1, H_kv, D, cache_len),
                                ::ml_drift::DataType::kFloat32);
    auto mask = builder.AddTensor(::ml_drift::BHWC(1, 1, S, cache_len),
                                  ::ml_drift::DataType::kFloat32);
    auto attn_out = builder.AddTensor(::ml_drift::BHWC(1, S, H_q, D),
                                      ::ml_drift::DataType::kFloat32);
    auto gate = builder.AddTensor(::ml_drift::BHWC(1, S, H_q, D),
                                  ::ml_drift::DataType::kFloat32);

    std::vector<int> selected_blocks(S * block_topk, -1);
    for (int qp = 0; qp < S; ++qp) {
      for (int k = 0; k < block_topk; ++k) {
        selected_blocks[qp * block_topk + k] =
            (k == 15 && qp == 0) ? 24 : (k + (qp % 6));
      }
    }
    ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                       ::ml_drift::DataType::kInt32>
        topk_cpu;
    topk_cpu.shape = ::ml_drift::BHWC(1, 1, S, block_topk);
    topk_cpu.data = selected_blocks;
    ::ml_drift::TensorDescriptor topk_desc(::ml_drift::DataType::kInt32,
                                           storage, ::ml_drift::Layout::kBHWC);
    topk_desc.UploadData(topk_cpu);
    auto topk = builder.AddConstantTensor(std::move(topk_desc));

    SparseSdpaTransposedAttributes attr{
        .compress_ratio = 4,
        .block_topk = block_topk,
        .scale = scale,
        .k_ts_idx = 2,
        .v_ts_idx = 3,
        .from_cache_update = false,
    };
    ASSERT_TRUE(BuildSparseSdpaTransposedGpuGraph(
                    {q.id, kc.id, vc.id, topk.id, mask.id}, {attn_out.id}, attr,
                    &builder)
                    .ok());
    auto gated_out = builder.Multiplication(attn_out, gate);

    ::ml_drift::GpuModel gpu_model;
    ASSERT_TRUE(
        builder
            .GetGpuModel(
                std::vector<std::pair<uint32_t, uint32_t>>{{q.id, 0},
                                                           {kc.id, 1},
                                                           {vc.id, 2},
                                                           {mask.id, 3},
                                                           {gate.id, 4}},
                std::vector<std::pair<uint32_t, uint32_t>>{{gated_out.id, 0}},
                &gpu_model)
            .ok());

    std::vector<float> q_data = MakePatternData(H_q * S * D, 1 + S);
    std::vector<float> kc_data = MakePatternData(H_kv * cache_len * D, 7);
    std::vector<float> vc_data = MakePatternData(H_kv * D * cache_len, 13);
    std::vector<float> gate_data = MakePatternData(S * H_q * D, 19);
    for (int kvh = 0; kvh < H_kv; ++kvh) {
      for (int d = 0; d < D; ++d) {
        for (int t = 115; t < cache_len; ++t) {
          vc_data[(kvh * D + d) * cache_len + t] =
              std::numeric_limits<float>::quiet_NaN();
        }
      }
    }
    std::vector<float> mask_data(S * cache_len, -10000.0f);
    for (int qp = 0; qp < S; ++qp) {
      const int vis_end = 98 + qp;
      for (int t = 0; t <= vis_end; ++t) {
        mask_data[qp * cache_len + t] = 0.0f;
      }
    }

    std::vector<float> expected =
        ReferenceSparseGqa(q_data, kc_data, vc_data, selected_blocks, mask_data,
                           H_q, H_kv, S, D, cache_len, block_topk, scale);
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] *= gate_data[i];
    }

    auto make_tf32 = [](::ml_drift::BHWC shape, std::vector<float> data) {
      ::ml_drift::TensorFloat32 t;
      t.shape = shape;
      t.data = std::move(data);
      return t;
    };
    std::vector<::ml_drift::TensorFloat32> src_cpu = {
        make_tf32(::ml_drift::BHWC(1, H_q, S, D), q_data),
        make_tf32(::ml_drift::BHWC(1, H_kv, cache_len, D), kc_data),
        make_tf32(::ml_drift::BHWC(1, H_kv, D, cache_len), vc_data),
        make_tf32(::ml_drift::BHWC(1, 1, S, cache_len), mask_data),
        make_tf32(::ml_drift::BHWC(1, S, H_q, D), gate_data),
    };
    ::ml_drift::TensorFloat32 out_cpu =
        make_tf32(::ml_drift::BHWC(1, S, H_q, D), {});
    std::vector<::ml_drift::TensorFloat32*> dst_cpu = {&out_cpu};
    const absl::Status exec_status =
        exec_env->ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model);
    ASSERT_TRUE(exec_status.ok()) << exec_status;
    EXPECT_THAT(out_cpu.data, Pointwise(FloatNear(1e-4f), expected));
  }
}

TEST_F(QsaCompositesKernelTest,
       IndexerShortContextFastPathAndLargePoolTopkMatchesCpu) {
  // Exercises both `end_blk_bound <= block_topk` (`start_pos_val = 18` -> 4
  // complete blocks <= `block_topk = 8`) and `end_blk_bound > block_topk`
  // (`start_pos_val = 142` -> 35 complete blocks > `block_topk = 8`) with
  // boolean mask, `index_n_heads = 8`, `D_idx = 32`, `max_blocks = 64`.
  const int H_idx = 8, D_idx = 32, R = 4, max_blocks = 64, cache_len = 256;
  const int rotary_dim = 16, block_topk = 8;
  const int H_q = 8, H_kv = 2, S = 1, D = 64;
  const float idx_scale = 1.0f / std::sqrt(static_cast<float>(D_idx));
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto storage = ::ml_drift::TensorStorageType::kBuffer;

  for (int start_pos_val : {18, 142}) {
    ::ml_drift::GpuModelBuilder builder(exec_env->GetGpuInfo(), /*hints=*/{},
                                        ::ml_drift::CalculationsPrecision::kF32,
                                        storage);
    auto q_idx = builder.AddTensor(::ml_drift::BHWC(1, S, H_idx, D_idx),
                                   ::ml_drift::DataType::kFloat32);
    auto k_idx_new = builder.AddTensor(::ml_drift::BHWC(1, 1, S, D_idx),
                                       ::ml_drift::DataType::kFloat32);
    auto ring_init = builder.AddTensor(::ml_drift::BHWC(1, 1, R, D_idx),
                                       ::ml_drift::DataType::kFloat32);
    auto pool_init =
        builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, D_idx),
                          ::ml_drift::DataType::kFloat32);
    auto ring_k = builder.AddTensor(::ml_drift::BHWC(1, 1, R, D_idx),
                                    ::ml_drift::DataType::kFloat32);
    auto block_k_pool =
        builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, D_idx),
                          ::ml_drift::DataType::kFloat32);
    builder.Copy(ring_init, ring_k);
    builder.Copy(pool_init, block_k_pool);
    auto k_norm_w = builder.AddTensor(::ml_drift::BHWC(D_idx, 1, 1, 1),
                                      ::ml_drift::DataType::kFloat32);
    auto b_cos =
        builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim),
                          ::ml_drift::DataType::kFloat32);
    auto b_sin =
        builder.AddTensor(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim),
                          ::ml_drift::DataType::kFloat32);
    auto q_main = builder.AddTensor(::ml_drift::BHWC(1, H_q, S, D),
                                    ::ml_drift::DataType::kFloat32);
    auto kc = builder.AddTensor(::ml_drift::BHWC(1, H_kv, cache_len, D),
                                ::ml_drift::DataType::kFloat32);
    auto vc = builder.AddTensor(::ml_drift::BHWC(1, H_kv, D, cache_len),
                                ::ml_drift::DataType::kFloat32);

    ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                       ::ml_drift::DataType::kInt32>
        start_pos_cpu;
    start_pos_cpu.shape = ::ml_drift::BHWC(1, 1, 1, 1);
    start_pos_cpu.data = {start_pos_val};
    ::ml_drift::TensorDescriptor start_pos_desc(
        ::ml_drift::DataType::kInt32, storage, ::ml_drift::Layout::kBHWC);
    start_pos_desc.UploadData(start_pos_cpu);
    auto start_pos = builder.AddConstantTensor(std::move(start_pos_desc));

    std::vector<uint8_t> bool_mask_vec(cache_len, 0);
    std::vector<float> float_mask_ref(cache_len, -10000.0f);
    for (int t = 0; t <= start_pos_val; ++t) {
      bool_mask_vec[t] = 1;
      float_mask_ref[t] = 0.0f;
    }
    ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                       ::ml_drift::DataType::kBool>
        mask_cpu;
    mask_cpu.shape = ::ml_drift::BHWC(1, 1, S, cache_len);
    mask_cpu.data = bool_mask_vec;
    ::ml_drift::TensorDescriptor mask_desc(::ml_drift::DataType::kBool, storage,
                                           ::ml_drift::Layout::kBHWC);
    mask_desc.UploadData(mask_cpu);
    auto mask = builder.AddConstantTensor(std::move(mask_desc));

    auto topk = builder.AddTensor(::ml_drift::BHWC(1, 1, S, block_topk),
                                  ::ml_drift::DataType::kInt32);
    auto attn_out = builder.AddTensor(::ml_drift::BHWC(1, S, H_q, D),
                                      ::ml_drift::DataType::kFloat32);

    QsaIndexerUpdateAttributes idx_attr{
        .compress_ratio = R,
        .block_topk = block_topk,
        .rms_norm_eps = 1e-6f,
        .scale = idx_scale,
    };
    ASSERT_TRUE(BuildQsaIndexerUpdateGpuGraph(
                    {q_idx.id, k_idx_new.id, k_norm_w.id, b_cos.id, b_sin.id,
                     start_pos.id, mask.id},
                    {topk.id, ring_k.id, block_k_pool.id}, idx_attr, &builder)
                    .ok());

    SparseSdpaTransposedAttributes sdpa_attr{
        .compress_ratio = R,
        .block_topk = block_topk,
        .scale = attn_scale,
        .k_ts_idx = 2,
        .v_ts_idx = 3,
    };
    ASSERT_TRUE(BuildSparseSdpaTransposedGpuGraph(
                    {q_main.id, kc.id, vc.id, topk.id, mask.id}, {attn_out.id},
                    sdpa_attr, &builder)
                    .ok());

    ::ml_drift::GpuModel gpu_model;
    ASSERT_TRUE(
        builder
            .GetGpuModel(
                std::vector<std::pair<uint32_t, uint32_t>>{{q_idx.id, 0},
                                                           {k_idx_new.id, 1},
                                                           {ring_init.id, 2},
                                                           {pool_init.id, 3},
                                                           {k_norm_w.id, 4},
                                                           {b_cos.id, 5},
                                                           {b_sin.id, 6},
                                                           {q_main.id, 7},
                                                           {kc.id, 8},
                                                           {vc.id, 9}},
                std::vector<std::pair<uint32_t, uint32_t>>{{attn_out.id, 0}},
                &gpu_model)
            .ok());

    std::vector<float> q_idx_data = MakePatternData(S * H_idx * D_idx, 2);
    std::vector<float> k_idx_new_data = MakePatternData(S * D_idx, 5);
    std::vector<float> ring_k_data = MakePatternData(R * D_idx, 9);
    std::vector<float> pool_data = MakePatternData(max_blocks * D_idx, 11);
    std::vector<float> k_norm_data(D_idx, 1.0f);
    std::vector<float> b_cos_data(max_blocks * rotary_dim, 1.0f);
    std::vector<float> b_sin_data(max_blocks * rotary_dim, 0.0f);
    std::vector<float> q_main_data = MakePatternData(H_q * S * D, 3);
    std::vector<float> kc_data = MakePatternData(H_kv * cache_len * D, 17);
    std::vector<float> vc_data = MakePatternData(H_kv * D * cache_len, 23);

    const int complete_blocks = (start_pos_val + 1) / R;
    std::vector<std::pair<float, int>> block_scores;
    for (int b = 0; b < complete_blocks; ++b) {
      float score = 0.0f;
      for (int h = 0; h < H_idx; ++h) {
        float dot = 0.0f;
        for (int d = 0; d < D_idx; ++d) {
          dot += q_idx_data[h * D_idx + d] * pool_data[b * D_idx + d];
        }
        score += std::max(dot, 0.0f);
      }
      block_scores.push_back({score * idx_scale, b});
    }
    std::sort(block_scores.begin(), block_scores.end(),
              [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
              });
    std::vector<int> selected_blocks(block_topk, -1);
    for (int k = 0;
         k < std::min(block_topk, static_cast<int>(block_scores.size())); ++k) {
      selected_blocks[k] = block_scores[k].second;
    }
    std::vector<float> expected_attn = ReferenceSparseGqa(
        q_main_data, kc_data, vc_data, selected_blocks, float_mask_ref, H_q,
        H_kv, S, D, cache_len, block_topk, attn_scale);

    auto make_tf32 = [](::ml_drift::BHWC shape, std::vector<float> data) {
      ::ml_drift::TensorFloat32 t;
      t.shape = shape;
      t.data = std::move(data);
      return t;
    };
    std::vector<::ml_drift::TensorFloat32> src_cpu = {
        make_tf32(::ml_drift::BHWC(1, S, H_idx, D_idx), q_idx_data),
        make_tf32(::ml_drift::BHWC(1, 1, S, D_idx), k_idx_new_data),
        make_tf32(::ml_drift::BHWC(1, 1, R, D_idx), ring_k_data),
        make_tf32(::ml_drift::BHWC(1, 1, max_blocks, D_idx), pool_data),
        make_tf32(::ml_drift::BHWC(D_idx, 1, 1, 1), k_norm_data),
        make_tf32(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim), b_cos_data),
        make_tf32(::ml_drift::BHWC(1, 1, max_blocks, rotary_dim), b_sin_data),
        make_tf32(::ml_drift::BHWC(1, H_q, S, D), q_main_data),
        make_tf32(::ml_drift::BHWC(1, H_kv, cache_len, D), kc_data),
        make_tf32(::ml_drift::BHWC(1, H_kv, D, cache_len), vc_data),
    };
    ::ml_drift::TensorFloat32 out_attn =
        make_tf32(::ml_drift::BHWC(1, S, H_q, D), {});
    std::vector<::ml_drift::TensorFloat32*> dst_cpu = {&out_attn};

    const absl::Status exec_status =
        exec_env->ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model);
    ASSERT_TRUE(exec_status.ok()) << exec_status;
    EXPECT_THAT(out_attn.data, Pointwise(FloatNear(1e-4f), expected_attn));
  }
}

}  // namespace
}  // namespace litert::ml_drift
