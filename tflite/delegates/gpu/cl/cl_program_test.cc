/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tflite/delegates/gpu/cl/cl_program.h"

#include <gtest/gtest.h>

#include <string>

#include "tflite/delegates/gpu/common/gpu_info.h"
#include "tflite/delegates/gpu/common/task/compiler_options.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

GpuInfo AdrenoGpuInfo(const std::string& platform_name) {
  GpuInfo info;
  GetGpuInfoFromDeviceDescription("Adreno (TM) 722 Qualcomm OpenCL C 1.2",
                                  GpuApi::kOpenCl, &info);
  info.opencl_info.platform_name = platform_name;
  return info;
}

TEST(CompilerOptionsToString, AdrenoOptionsOnQualcommPlatform) {
  const GpuInfo info = AdrenoGpuInfo("QUALCOMM Snapdragon(TM)");
  EXPECT_EQ(CompilerOptionsToString(info, {CompilerOptions::kAdrenoFullSimd}),
            "-qcom-accelerate-16-bit=true ");
  EXPECT_EQ(CompilerOptionsToString(info, {CompilerOptions::kAdrenoMoreWaves}),
            "-qcom-accelerate-16-bit=false ");
}

TEST(CompilerOptionsToString, NoAdrenoOptionsOnRusticl) {
  const GpuInfo info = AdrenoGpuInfo("rusticl");
  EXPECT_EQ(CompilerOptionsToString(info, {CompilerOptions::kAdrenoFullSimd,
                                           CompilerOptions::kAdrenoMoreWaves}),
            "  ");
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
