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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_COOP_MAT_TEST_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_COOP_MAT_TEST_UTIL_H_

#include "tflite/delegates/gpu/common/status.h"
#include "tflite/delegates/gpu/common/task/testing_util.h"

namespace tflite {
namespace gpu {

absl::Status ConvCoopMat1x1Test(TestExecutionEnvironment* env);
absl::Status ConvCoopMatSmallChannelCountsTest(TestExecutionEnvironment* env);
absl::Status ConvCoopMatKnownWeightsTest(TestExecutionEnvironment* env);
absl::Status ConvCoopMat1x1OutputChannelsTailTest(TestExecutionEnvironment* env);
absl::Status ConvCoopMat3x3PaddingTest(TestExecutionEnvironment* env);
absl::Status ConvCoopMat3x3OutputChannelsTailTest(TestExecutionEnvironment* env);
absl::Status ConvCoopMatStrideDilationTest(TestExecutionEnvironment* env);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_COOP_MAT_TEST_UTIL_H_
