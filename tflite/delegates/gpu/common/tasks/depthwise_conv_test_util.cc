/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tflite/delegates/gpu/common/tasks/depthwise_conv_test_util.h"

#include <memory>
#include <vector>

#include "tflite/delegates/gpu/common/operations.h"
#include "tflite/delegates/gpu/common/status.h"
#include "tflite/delegates/gpu/common/task/testing_util.h"
#include "tflite/delegates/gpu/common/tasks/depthwise_conv.h"

namespace tflite {
namespace gpu {

absl::Status DepthwiseConvSimpleWeightsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights.shape = OHWI(1, 3, 1, 2);
  attr.weights.data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.0f, 0.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      DepthwiseConv operation =
          CreateDepthwiseConvolution2D(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<DepthwiseConv>(std::move(operation)),
          BHWC(1, 2, 2, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({4.0f, 6.0f, 8.0f, 10.0f, 4.0f, 6.0f, 8.0f, 10.0f},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status DepthwiseConvNoMultiplierTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights.shape = OHWI(1, 3, 1, 2);
  attr.weights.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      DepthwiseConv operation =
          CreateDepthwiseConvolution2D(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<DepthwiseConv>(std::move(operation)),
          BHWC(1, 2, 2, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({16.5f, 27.5f, 28.5f, 43.5f, 8.5f, 15.5f, 12.5f, 23.5f},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status DepthwiseConvMultiplier2Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights.shape = OHWI(2, 3, 1, 2);
  attr.weights.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                       6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  attr.bias.shape = Linear(4);
  attr.bias.data = {0.5f, -0.5f, 1.0f, -1.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      DepthwiseConv operation =
          CreateDepthwiseConvolution2D(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<DepthwiseConv>(std::move(operation)),
          BHWC(1, 2, 2, 4), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {16.5f, 39.5f, 29.0f, 63.0f, 28.5f, 75.5f, 45.0f, 103.0f, 8.5f, 31.5f,
           17.0f, 51.0f, 12.5f, 59.5f, 25.0f, 83.0f},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
