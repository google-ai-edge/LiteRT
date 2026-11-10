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

#include "ml_drift_delegate/delegate/composite/sdpa_transposed_support.h"

#include <utility>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

::ml_drift::GpuInfo OpenClGpuInfo(::ml_drift::GpuVendor vendor) {
  ::ml_drift::GpuInfo gpu_info;
  gpu_info.vendor = vendor;
  gpu_info.gpu_api = ::ml_drift::GpuApi::kOpenCl;
  gpu_info.opencl_info.max_work_group_size_x = 1024;
  gpu_info.opencl_info.max_work_group_size_y = 1024;
  gpu_info.opencl_info.max_work_group_size_z = 1024;
  gpu_info.opencl_info.max_work_group_total_size = 1024;
  return gpu_info;
}

// Single-token decode query and K cache of a model with 8 query heads sharing
// 2 K/V heads (GQA group 4), e.g. Gemma 4 E4B.
constexpr int kQueryHeads = 8;
constexpr int kKvHeads = 2;

::ml_drift::BHWC QueryShape(int head_dim) {
  return ::ml_drift::BHWC(1, kQueryHeads, 1, head_dim);
}

::ml_drift::BHWC CacheShape(int cache_size, int head_dim) {
  return ::ml_drift::BHWC(1, kKvHeads, cache_size, head_dim);
}

TEST(SdpaTransposedSupportTest, AdrenoOpenClProcessesAtMostTwoHeadsPerGroup) {
  // With 4 heads per work group the Adreno 830 compiler limits the kernel to
  // 192 work items, fewer than the 256 it is launched with (b/578633071).
  const ::ml_drift::GpuInfo gpu_info =
      OpenClGpuInfo(::ml_drift::GpuVendor::kQualcomm);
  EXPECT_EQ(GetFlashDecodeTuning(gpu_info).max_heads_per_work_group, 2);

  for (const auto& [cache_size, head_dim] :
       {std::pair{512, 256}, std::pair{4096, 512}}) {
    const ::ml_drift::BHWC q_shape = QueryShape(head_dim);
    const ::ml_drift::BHWC k_shape = CacheShape(cache_size, head_dim);
    const FlashDecodeConfig config =
        GetFlashDecodeConfig(gpu_info, q_shape, k_shape);
    EXPECT_FALSE(config.use_wave_simd);
    EXPECT_EQ(config.heads, 2) << "head_dim=" << head_dim;
    EXPECT_EQ(config.tuning.threads, 256) << "head_dim=" << head_dim;
    EXPECT_TRUE(IsSupportedFlashSdpaShape(gpu_info, q_shape, k_shape))
        << "head_dim=" << head_dim;
  }
}

TEST(SdpaTransposedSupportTest, OtherOpenClGpusProcessTheWholeGqaGroup) {
  const ::ml_drift::GpuInfo gpu_info =
      OpenClGpuInfo(::ml_drift::GpuVendor::kMali);
  EXPECT_EQ(GetFlashDecodeTuning(gpu_info).max_heads_per_work_group, 8);

  const ::ml_drift::BHWC q_shape = QueryShape(/*head_dim=*/256);
  const ::ml_drift::BHWC k_shape = CacheShape(/*cache_size=*/512, 256);
  const FlashDecodeConfig config =
      GetFlashDecodeConfig(gpu_info, q_shape, k_shape);
  EXPECT_EQ(config.heads, kQueryHeads / kKvHeads);
  EXPECT_EQ(config.tuning.threads, 256);
}

TEST(SdpaTransposedSupportTest, ThreadsAreClampedToTheDeviceLimit) {
  ::ml_drift::GpuInfo gpu_info = OpenClGpuInfo(::ml_drift::GpuVendor::kMali);
  gpu_info.opencl_info.max_work_group_total_size = 128;
  EXPECT_EQ(GetFlashDecodeTuning(gpu_info).threads, 128);
}

::ml_drift::GpuInfo AppleMetalGpuInfo() {
  ::ml_drift::GpuInfo gpu_info;
  gpu_info.vendor = ::ml_drift::GpuVendor::kApple;
  gpu_info.gpu_api = ::ml_drift::GpuApi::kMetal;
  gpu_info.supported_wave_sizes = {32};
  return gpu_info;
}

TEST(SdpaTransposedSupportTest, AppleMetalUsesTheSimdWorkGroupKernel) {
  const ::ml_drift::GpuInfo gpu_info = AppleMetalGpuInfo();
  const FlashDecodeTuning tuning = GetFlashDecodeTuning(gpu_info);
  EXPECT_TRUE(tuning.simd_work_group);
  EXPECT_EQ(tuning.max_heads_per_work_group, 4);
  EXPECT_EQ(tuning.lanes_per_key, 4);

  // Gemma 4 26B-A4B global attention: 16 query heads share 2 K/V heads of head
  // dim 512 in a 32K cache. 4 heads per work group and 10 splits give 40 work
  // groups.
  const ::ml_drift::BHWC q_shape(1, 16, 1, 512);
  const ::ml_drift::BHWC k_shape(1, 2, 32768, 512);
  const FlashDecodeConfig config =
      GetFlashDecodeConfig(gpu_info, q_shape, k_shape);
  EXPECT_FALSE(config.use_wave_simd);
  EXPECT_TRUE(config.tuning.simd_work_group);
  EXPECT_EQ(config.heads, 4);
  EXPECT_EQ(config.splits, 10);
  EXPECT_TRUE(IsSupportedFlashSdpaShape(gpu_info, q_shape, k_shape));

  // Head dim 128 keeps the wave-SIMD kernel.
  EXPECT_TRUE(GetFlashDecodeConfig(gpu_info, ::ml_drift::BHWC(1, 16, 1, 128),
                                   ::ml_drift::BHWC(1, 8, 1280, 128))
                  .use_wave_simd);
}

TEST(SdpaTransposedSupportTest, SimdWorkGroupSplitsKeysAcrossDividingLanes) {
  const ::ml_drift::GpuInfo gpu_info = AppleMetalGpuInfo();
  for (const auto& [head_dim, lanes] :
       {std::pair{4, 1}, std::pair{8, 2}, std::pair{16, 4}, std::pair{24, 2},
        std::pair{256, 4}}) {
    const FlashDecodeConfig config = GetFlashDecodeConfig(
        gpu_info, QueryShape(head_dim), CacheShape(1024, head_dim));
    EXPECT_EQ(config.tuning.lanes_per_key, lanes) << "head_dim=" << head_dim;
  }
}

TEST(SdpaTransposedSupportTest, SimdWorkGroupLocalMemoryBytes) {
  FlashDecodeTuning tuning = GetFlashDecodeTuning(AppleMetalGpuInfo());
  ASSERT_EQ(tuning.threads, 256);
  // 4 heads of head dim 512: float4 query slices (4 * 128 * 16 B), the scores
  // of 256 keys (4 * 256 * 4 B) and 2 * 4 * 8 SIMD-group values. The output
  // accumulators reuse the score buffer.
  EXPECT_TRUE(SimdWorkGroupAccumulatorsReuseScores(4));
  EXPECT_EQ(FlashDecodeLocalMemoryBytes(tuning, /*heads=*/4, /*slices=*/128),
            8192 + 4096 + 256);
  // 2 heads of head dim 256 need separate float4 accumulators (256 * 16 B).
  EXPECT_FALSE(SimdWorkGroupAccumulatorsReuseScores(2));
  EXPECT_EQ(FlashDecodeLocalMemoryBytes(tuning, /*heads=*/2, /*slices=*/64),
            2048 + 2048 + 128 + 4096);
  // The portable kernel needs 25,088 B for the first config.
  tuning.simd_work_group = false;
  EXPECT_EQ(FlashDecodeLocalMemoryBytes(tuning, /*heads=*/4, /*slices=*/128),
            25088);
}

TEST(SdpaTransposedSupportTest,
     MetalWave32UsesWaveSimdForHeadDims64To256AndNonDividingUpTo512) {
  ::ml_drift::GpuInfo gpu_info = AppleMetalGpuInfo();
  gpu_info.apple_info = ::ml_drift::AppleInfo("apple m4 pro");
  gpu_info.metal_info.language_version =
      ::ml_drift::MetalLanguageVersion::kMetal3_0;
  gpu_info.metal_info.max_work_group_size_x = 1024;
  gpu_info.metal_info.max_work_group_size_y = 1024;
  gpu_info.metal_info.max_work_group_size_z = 1024;

  for (int head_dim : {64, 96, 128, 192, 256, 384}) {
    const ::ml_drift::BHWC q_shape = QueryShape(head_dim);
    const ::ml_drift::BHWC k_shape = CacheShape(/*cache_size=*/1024, head_dim);
    const FlashDecodeConfig config =
        GetFlashDecodeConfig(gpu_info, q_shape, k_shape);
    EXPECT_TRUE(config.use_wave_simd) << "head_dim=" << head_dim;
    EXPECT_EQ(config.splits, 1) << "head_dim=" << head_dim;
    EXPECT_TRUE(IsSupportedFlashSdpaShape(gpu_info, q_shape, k_shape))
        << "head_dim=" << head_dim;
  }

  // Head dim 512 (128 slices, divides 256 threads) stays on the 4-head SIMD
  // work-group kernel.
  const FlashDecodeConfig hd512 = GetFlashDecodeConfig(
      gpu_info, QueryShape(/*head_dim=*/512), CacheShape(1024, 512));
  EXPECT_FALSE(hd512.use_wave_simd);
  EXPECT_TRUE(hd512.tuning.simd_work_group);
}

}  // namespace
}  // namespace litert::ml_drift
