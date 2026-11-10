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

}  // namespace
}  // namespace litert::ml_drift
