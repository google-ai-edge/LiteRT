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

#include "litert/vendors/nvidia/dispatch/greedy_sampler_kernel.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "driver_types.h"

namespace {

int CpuArgMax(const std::vector<float>& values) {
  int max_id = 0;
  float max_value = values[0];
  for (int i = 1; i < values.size(); ++i) {
    if (values[i] > max_value) {
      max_value = values[i];
      max_id = i;
    }
  }
  return max_id;
}

int RunGpuArgMax(const std::vector<float>& values) {
  float* device_values = nullptr;
  void* workspace = nullptr;
  int32_t* device_result = nullptr;
  const size_t workspace_bytes =
      LiteRtNvidiaF32ArgMaxWorkspaceBytes(values.size());
  if (cudaMalloc(reinterpret_cast<void**>(&device_values),
                 values.size() * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&workspace, workspace_bytes) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(int32_t)) !=
          cudaSuccess ||
      cudaMemcpy(device_values, values.data(), values.size() * sizeof(float),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    ADD_FAILURE() << "Failed to allocate or copy CUDA argmax inputs";
    cudaFree(device_values);
    cudaFree(workspace);
    cudaFree(device_result);
    return -1;
  }
  const cudaError_t launch_status = LiteRtNvidiaLaunchF32ArgMax(
      device_values, values.size(), workspace, workspace_bytes, device_result,
      /*stream=*/nullptr);
  int32_t result = -1;
  const cudaError_t copy_status =
      launch_status == cudaSuccess
          ? cudaMemcpy(&result, device_result, sizeof(int32_t),
                       cudaMemcpyDeviceToHost)
          : launch_status;
  if (launch_status != cudaSuccess || copy_status != cudaSuccess) {
    ADD_FAILURE() << "CUDA argmax failed: "
                  << cudaGetErrorString(launch_status != cudaSuccess
                                            ? launch_status
                                            : copy_status);
  }
  cudaFree(device_values);
  cudaFree(workspace);
  cudaFree(device_result);
  return result;
}

TEST(GreedySamplerKernelTest, F32ArgMaxMatchesCpuAcrossReductionBoundaries) {
  std::mt19937 generator(12345);
  std::uniform_real_distribution<float> distribution(-100.0f, 100.0f);
  for (const size_t size :
       {size_t{1}, size_t{255}, size_t{256}, size_t{257}, size_t{262144}}) {
    std::vector<float> values(size);
    for (float& value : values) {
      value = distribution(generator);
    }
    EXPECT_EQ(RunGpuArgMax(values), CpuArgMax(values)) << "size=" << size;
  }
}

TEST(GreedySamplerKernelTest, F32ArgMaxUsesFirstMaximum) {
  std::vector<float> values(1025, -4.0f);
  values[7] = 9.0f;
  values[1000] = 9.0f;
  EXPECT_EQ(RunGpuArgMax(values), 7);
}

TEST(GreedySamplerKernelTest, F32ArgMaxMatchesCpuNanAndInfinityBehavior) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  for (const std::vector<float>& values : {
           std::vector<float>{nan, 100.0f, infinity},
           std::vector<float>{-5.0f, nan, 3.0f, infinity, infinity},
           std::vector<float>{-infinity, -10.0f, -3.0f},
       }) {
    EXPECT_EQ(RunGpuArgMax(values), CpuArgMax(values));
  }
}

TEST(GreedySamplerKernelTest, F32ArgMaxRejectsInvalidWorkspace) {
  float* device_value = nullptr;
  int32_t* device_result = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_value), sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(
      cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(int32_t)),
      cudaSuccess);
  EXPECT_EQ(LiteRtNvidiaLaunchF32ArgMax(
                device_value, /*count=*/1, /*workspace=*/nullptr,
                /*workspace_bytes=*/0, device_result, /*stream=*/nullptr),
            cudaErrorInvalidValue);
  cudaFree(device_value);
  cudaFree(device_result);
}

}  // namespace
