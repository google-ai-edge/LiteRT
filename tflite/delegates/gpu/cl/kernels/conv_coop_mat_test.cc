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

#include <gtest/gtest.h>

#include "tflite/delegates/gpu/cl/kernels/cl_test.h"
#include "tflite/delegates/gpu/common/tasks/conv_coop_mat_test_util.h"

namespace tflite {
namespace gpu {
namespace cl {

TEST_F(OpenCLOperationTest, ConvCoopMat1x1) {
  const auto status = ConvCoopMat1x1Test(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ConvCoopMatSmallChannelCounts) {
  const auto status = ConvCoopMatSmallChannelCountsTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ConvCoopMatKnownWeights) {
  const auto status = ConvCoopMatKnownWeightsTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ConvCoopMat1x1OutputChannelsTail) {
  const auto status = ConvCoopMat1x1OutputChannelsTailTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ConvCoopMat3x3Padding) {
  const auto status = ConvCoopMat3x3PaddingTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ConvCoopMat3x3OutputChannelsTail) {
  const auto status = ConvCoopMat3x3OutputChannelsTailTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ConvCoopMatStrideDilation) {
  const auto status = ConvCoopMatStrideDilationTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
