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

#include "ml_drift_delegate/delegate/delegate_utils.h"

#include <string>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/precision.h"

namespace litert::ml_drift {
namespace {

::ml_drift::GpuInfo MakeOpenGlGpuInfo(const std::string& renderer) {
  ::ml_drift::GpuInfo gpu_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      renderer, ::ml_drift::GpuApi::kOpenGl, &gpu_info);
  return gpu_info;
}

struct DefaultPrecisionCase {
  std::string test_name;
  std::string renderer;
  ::ml_drift::CalculationsPrecision expected_precision;
};

class ResolveOpenGlDefaultPrecisionTest
    : public ::testing::TestWithParam<DefaultPrecisionCase> {};

TEST_P(ResolveOpenGlDefaultPrecisionTest, SelectsExpectedDefaultPrecision) {
  const DefaultPrecisionCase& tc = GetParam();
  const ::ml_drift::GpuInfo gpu_info = MakeOpenGlGpuInfo(tc.renderer);
  const MlDriftDelegateOptions options;

  EXPECT_EQ(ResolveOpenGlCalculationPrecision(gpu_info, options),
            tc.expected_precision);
}

INSTANTIATE_TEST_SUITE_P(
    OpenGlDevices, ResolveOpenGlDefaultPrecisionTest,
    ::testing::Values(
        DefaultPrecisionCase{
            .test_name = "Adreno750UsesFp16",
            .renderer = "Adreno (TM) 750",
            .expected_precision = ::ml_drift::CalculationsPrecision::kF16,
        },
        DefaultPrecisionCase{
            .test_name = "Adreno308UsesFp32Fp16",
            .renderer = "Adreno (TM) 308",
            .expected_precision = ::ml_drift::CalculationsPrecision::kF32F16,
        },
        DefaultPrecisionCase{
            .test_name = "Adreno306UsesFp32Fp16",
            .renderer = "Adreno (TM) 306",
            .expected_precision = ::ml_drift::CalculationsPrecision::kF32F16,
        },
        DefaultPrecisionCase{
            .test_name = "MaliG78UsesFp16",
            .renderer = "Mali-G78",
            .expected_precision = ::ml_drift::CalculationsPrecision::kF16,
        },
        DefaultPrecisionCase{
            .test_name = "PowerVrRogueGm9446UsesFp32Fp16",
            .renderer = "PowerVR Rogue GM9446",
            .expected_precision = ::ml_drift::CalculationsPrecision::kF32F16,
        },
        DefaultPrecisionCase{
            .test_name = "PowerVrBSeriesUsesFp32Fp16",
            .renderer = "PowerVR B-Series BXM-8-256",
            .expected_precision = ::ml_drift::CalculationsPrecision::kF32F16,
        },
        DefaultPrecisionCase{
            .test_name = "PowerVrRogueGe8320UsesFp32",
            .renderer = "PowerVR Rogue GE8320",
            .expected_precision = ::ml_drift::CalculationsPrecision::kF32,
        },
        DefaultPrecisionCase{
            .test_name = "PowerVrRogueG6430UsesFp32",
            .renderer = "PowerVR Rogue G6430",
            .expected_precision = ::ml_drift::CalculationsPrecision::kF32,
        }),
    [](const ::testing::TestParamInfo<DefaultPrecisionCase>& info) {
      return info.param.test_name;
    });

TEST(ResolveOpenGlCalculationPrecisionTest, ExplicitFp16OverridesRtzFallback) {
  const ::ml_drift::GpuInfo gpu_info = MakeOpenGlGpuInfo("Adreno (TM) 308");
  MlDriftDelegateOptions options;
  options.precision = MlDriftDelegatePrecision::kFp16;

  EXPECT_EQ(ResolveOpenGlCalculationPrecision(gpu_info, options),
            ::ml_drift::CalculationsPrecision::kF16);
}

TEST(ResolveOpenGlCalculationPrecisionTest, ExplicitFp32ForcesFp32) {
  const ::ml_drift::GpuInfo gpu_info = MakeOpenGlGpuInfo("Adreno (TM) 750");
  MlDriftDelegateOptions options;
  options.precision = MlDriftDelegatePrecision::kFp32;

  EXPECT_EQ(ResolveOpenGlCalculationPrecision(gpu_info, options),
            ::ml_drift::CalculationsPrecision::kF32);
}

TEST(ResolveOpenGlCalculationPrecisionTest,
     UseF32AccumForFp16ForcesFp32Fp16OnRtnGpu) {
  const ::ml_drift::GpuInfo gpu_info = MakeOpenGlGpuInfo("Adreno (TM) 750");
  MlDriftDelegateOptions options;
  options.precision = MlDriftDelegatePrecision::kFp16;
  options.use_f32_accum_for_fp16 = true;

  EXPECT_EQ(ResolveOpenGlCalculationPrecision(gpu_info, options),
            ::ml_drift::CalculationsPrecision::kF32F16);
}

}  // namespace
}  // namespace litert::ml_drift
