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

#include "ml_drift_delegate/delegate/gpu_backend_opencl_litert.h"

#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift

namespace litert::ml_drift {

class GpuBackendOpenClLitertTest : public ::testing::Test {
 protected:
  static uint64_t ComputeMaxFlopsPerKernelBatch(
      const ::ml_drift::GpuInfo& gpu_info,
      ::ml_drift::CalculationsPrecision precision) {
    return GpuBackendOpenClLitert::ComputeMaxFlopsPerKernelBatch(gpu_info,
                                                                 precision);
  }

  static std::vector<int> PartitionNodesByFlops(
      absl::Span<const uint64_t> node_flops, uint64_t max_flops_per_batch) {
    return GpuBackendOpenClLitert::PartitionNodesByFlops(node_flops,
                                                         max_flops_per_batch);
  }
};

namespace {

constexpr uint64_t kGFlop = 1000ULL * 1000 * 1000;

::ml_drift::GpuInfo MakeOpenClGpuInfo(const std::string& description,
                                      int opencl_compute_units = 0) {
  ::ml_drift::GpuInfo gpu_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      description, ::ml_drift::GpuApi::kOpenCl, &gpu_info);
  gpu_info.opencl_info.compute_units_count = opencl_compute_units;
  return gpu_info;
}

TEST_F(GpuBackendOpenClLitertTest, MaxFlopsBudgetOnlyEnabledOnAdreno8xx) {
  EXPECT_EQ(
      ComputeMaxFlopsPerKernelBatch(
          MakeOpenClGpuInfo("Adreno (TM) 750", /*opencl_compute_units=*/6),
          ::ml_drift::CalculationsPrecision::kF16),
      0);
  EXPECT_EQ(
      ComputeMaxFlopsPerKernelBatch(
          MakeOpenClGpuInfo("Adreno (TM) 750", /*opencl_compute_units=*/6),
          ::ml_drift::CalculationsPrecision::kF32),
      0);
  EXPECT_EQ(
      ComputeMaxFlopsPerKernelBatch(
          MakeOpenClGpuInfo("Adreno (TM) 650", /*opencl_compute_units=*/3),
          ::ml_drift::CalculationsPrecision::kF32),
      0);
  EXPECT_EQ(ComputeMaxFlopsPerKernelBatch(
                MakeOpenClGpuInfo("Mali-G715", /*opencl_compute_units=*/10),
                ::ml_drift::CalculationsPrecision::kF16),
            0);
  EXPECT_EQ(
      ComputeMaxFlopsPerKernelBatch(
          MakeOpenClGpuInfo("PowerVR Rogue GM9446", /*opencl_compute_units=*/4),
          ::ml_drift::CalculationsPrecision::kF32),
      0);
  EXPECT_EQ(ComputeMaxFlopsPerKernelBatch(
                MakeOpenClGpuInfo("NVIDIA GeForce RTX 4090",
                                  /*opencl_compute_units=*/128),
                ::ml_drift::CalculationsPrecision::kF32),
            0);
}

TEST_F(GpuBackendOpenClLitertTest, MaxFlopsBudgetScalesByPrecisionAndCUs) {
  const auto adreno840 =
      MakeOpenClGpuInfo("Adreno (TM) 840", /*opencl_compute_units=*/12);
  EXPECT_EQ(ComputeMaxFlopsPerKernelBatch(
                adreno840, ::ml_drift::CalculationsPrecision::kF16),
            600ULL * kGFlop);
  EXPECT_EQ(ComputeMaxFlopsPerKernelBatch(
                adreno840, ::ml_drift::CalculationsPrecision::kF32),
            240ULL * kGFlop);
  EXPECT_EQ(ComputeMaxFlopsPerKernelBatch(
                adreno840, ::ml_drift::CalculationsPrecision::kF32F16),
            240ULL * kGFlop);

  // Fallback to AdrenoInfo table when OpenCL CU count is unset (0).
  const auto adreno840_fallback = MakeOpenClGpuInfo("Adreno (TM) 840", 0);
  EXPECT_EQ(ComputeMaxFlopsPerKernelBatch(
                adreno840_fallback, ::ml_drift::CalculationsPrecision::kF16),
            600ULL * kGFlop);

  const auto adreno825 = MakeOpenClGpuInfo("Adreno (TM) 825", 0);
  EXPECT_EQ(ComputeMaxFlopsPerKernelBatch(
                adreno825, ::ml_drift::CalculationsPrecision::kF16),
            400ULL * kGFlop);

  const auto adreno810 = MakeOpenClGpuInfo("Adreno (TM) 810", 0);
  EXPECT_EQ(ComputeMaxFlopsPerKernelBatch(
                adreno810, ::ml_drift::CalculationsPrecision::kF16),
            100ULL * kGFlop);
  EXPECT_EQ(ComputeMaxFlopsPerKernelBatch(
                adreno810, ::ml_drift::CalculationsPrecision::kF32),
            40ULL * kGFlop);
}

TEST_F(GpuBackendOpenClLitertTest, PartitionEmptyOrDisabledReturnsEmpty) {
  EXPECT_THAT(PartitionNodesByFlops({}, 600ULL * kGFlop), ::testing::IsEmpty());
  EXPECT_THAT(PartitionNodesByFlops({500ULL * kGFlop, 500ULL * kGFlop}, 0),
              ::testing::IsEmpty());
}

TEST_F(GpuBackendOpenClLitertTest, ManySmallDecodeNodesStaySingleBatch) {
  // Simulate a 2,500-node single-token LLM decode graph (~10 GFLOPs total).
  const std::vector<uint64_t> decode_flops(2500, 4ULL * 1000 * 1000);
  EXPECT_THAT(PartitionNodesByFlops(decode_flops, 600ULL * kGFlop),
              ::testing::IsEmpty());
  EXPECT_THAT(PartitionNodesByFlops(decode_flops, 240ULL * kGFlop),
              ::testing::IsEmpty());
}

TEST_F(GpuBackendOpenClLitertTest, ExactFitAndBoundaryFlush) {
  EXPECT_THAT(PartitionNodesByFlops({300ULL * kGFlop, 300ULL * kGFlop},
                                    600ULL * kGFlop),
              ::testing::IsEmpty());
  // Once a batch reaches budget, even a following zero-FLOP node triggers a
  // new batch so un-annotated ops are not accumulated onto a full batch.
  EXPECT_THAT(PartitionNodesByFlops({300ULL * kGFlop, 300ULL * kGFlop, 0},
                                    600ULL * kGFlop),
              ::testing::ElementsAre(2, 1));
}

TEST_F(GpuBackendOpenClLitertTest, SingleHeavyNodeCannotBeSplitFurther) {
  EXPECT_THAT(PartitionNodesByFlops({800ULL * kGFlop}, 600ULL * kGFlop),
              ::testing::IsEmpty());
}

TEST_F(GpuBackendOpenClLitertTest, IsolatesSingleOverBudgetOp) {
  // Heavy SDPA op (317 GFLOPs) under FP32 budget (240 GFLOPs) gets flushed
  // right before and right after it.
  EXPECT_THAT(PartitionNodesByFlops(
                  {50ULL * kGFlop, 317ULL * kGFlop, 0, 50ULL * kGFlop},
                  240ULL * kGFlop),
              ::testing::ElementsAre(1, 1, 2));
}

TEST_F(GpuBackendOpenClLitertTest, PartitionsVision1120ScaleGraph) {
  // 16 layers x (84 linear/conv/norm nodes ~142 GFLOPs + 1 SDPA 317 GFLOPs).
  constexpr int kLayers = 16;
  constexpr int kNodesPerLayer = 85;
  std::vector<uint64_t> vision_flops;
  vision_flops.reserve(kLayers * kNodesPerLayer);
  for (int layer = 0; layer < kLayers; ++layer) {
    for (int i = 0; i < kNodesPerLayer - 1; ++i) {
      vision_flops.push_back(142ULL * kGFlop / (kNodesPerLayer - 1));
    }
    vision_flops.push_back(317ULL * kGFlop);
  }

  const auto fp16_batches =
      PartitionNodesByFlops(vision_flops, 600ULL * kGFlop);
  EXPECT_EQ(std::accumulate(fp16_batches.begin(), fp16_batches.end(), 0),
            static_cast<int>(vision_flops.size()));
  EXPECT_GE(fp16_batches.size(), 12);

  const auto fp32_batches =
      PartitionNodesByFlops(vision_flops, 240ULL * kGFlop);
  EXPECT_EQ(std::accumulate(fp32_batches.begin(), fp32_batches.end(), 0),
            static_cast<int>(vision_flops.size()));
  EXPECT_GE(fp32_batches.size(), 30);

  // Legacy-cache case: even if SDPA ops reported 0 FLOPs, linear/conv alone
  // (~2.27 TFLOPs) still splits into multiple safe batches.
  for (int layer = 0; layer < kLayers; ++layer) {
    vision_flops[layer * kNodesPerLayer + (kNodesPerLayer - 1)] = 0;
  }
  const auto legacy_fp16_batches =
      PartitionNodesByFlops(vision_flops, 600ULL * kGFlop);
  EXPECT_EQ(std::accumulate(legacy_fp16_batches.begin(),
                            legacy_fp16_batches.end(), 0),
            static_cast<int>(vision_flops.size()));
  EXPECT_GE(legacy_fp16_batches.size(), 4);
}

TEST_F(GpuBackendOpenClLitertTest, SaturatesWithoutOverflow) {
  const uint64_t kMax = std::numeric_limits<uint64_t>::max();
  EXPECT_THAT(PartitionNodesByFlops({kMax - 10, 20, 5}, kMax),
              ::testing::ElementsAre(1, 2));
}

}  // namespace
}  // namespace litert::ml_drift
