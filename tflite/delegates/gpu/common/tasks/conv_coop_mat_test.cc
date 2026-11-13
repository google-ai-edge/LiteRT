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

#include "tflite/delegates/gpu/common/tasks/conv_coop_mat.h"

#include <cstring>
#include <string>

#include <gtest/gtest.h>
#include "tflite/delegates/gpu/common/model_hints.h"
#include "tflite/delegates/gpu/common/selectors/convolution_selector.h"
#include "tflite/delegates/gpu/common/task/arguments.h"
#include "tflite/delegates/gpu/common/task/buffer_desc.h"
#include "tflite/delegates/gpu/common/task/gpu_tensor.h"
#include "tflite/delegates/gpu/common/task/tensor_desc.h"
#include "tflite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {

GpuInfo CooperativeGpu() {
  GpuInfo info;
  info.gpu_api = GpuApi::kOpenCl;
  info.vendor = GpuVendor::kAMD;
  info.opencl_info.extensions = {"cl_khr_cooperative_matrix"};
  return info;
}

OperationDef Definition() {
  OperationDef definition;
  definition.precision = CalculationsPrecision::F32;
  definition.src_tensors.push_back(
      {DataType::FLOAT32, TensorStorageType::BUFFER, Layout::HWC});
  definition.dst_tensors = definition.src_tensors;
  return definition;
}

Convolution2DAttributes Attributes() {
  Convolution2DAttributes attr;
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.weights.shape = OHWI(65, 1, 1, 64);
  attr.weights.data.resize(65 * 64);
  for (int i = 0; i < 65 * 64; ++i) {
    attr.weights.data[i] = static_cast<float>(i);
  }
  attr.bias.shape = Linear(65);
  attr.bias.data.resize(65, 2.0f);
  return attr;
}

class ShapeTensor : public GpuSpatialTensor {
 public:
  explicit ShapeTensor(const BHWC& shape) : shape(shape) {}
  int Width() const override { return shape.w; }
  int Height() const override { return shape.h; }
  int Depth() const override { return 1; }
  int Channels() const override { return shape.c; }
  int Slices() const override { return (shape.c + 3) / 4; }
  int Batch() const override { return shape.b; }
  TensorDescriptor GetDescriptor() const override {
    return {DataType::FLOAT32, TensorStorageType::BUFFER, Layout::HWC};
  }
  BHWC shape;
};

TEST(ConvCoopMatTest, EligibilityRejectsUnsupportedCases) {
  auto info = CooperativeGpu();
  auto definition = Definition();
  auto attr = Attributes();
  const BHWC output(1, 32, 32, 65);
  EXPECT_TRUE(IsConvCoopMatSupported(info, definition, attr, output));
  EXPECT_FALSE(IsConvCoopMatSupported(
      info, definition, attr, BHWC(2, 32, 32, 65)));
  attr.groups = 2;
  EXPECT_FALSE(IsConvCoopMatSupported(info, definition, attr, output));
  attr.groups = 1;
  attr.weights.shape.i = 63;
  EXPECT_FALSE(IsConvCoopMatSupported(info, definition, attr, output));
  attr.weights.shape.i = 64;
  definition.precision = CalculationsPrecision::F32_F16;
  EXPECT_FALSE(IsConvCoopMatSupported(info, definition, attr, output));
  definition = Definition();
  info.opencl_info.extensions.clear();
  EXPECT_FALSE(IsConvCoopMatSupported(info, definition, attr, output));
  info = CooperativeGpu();
  info.gpu_api = GpuApi::kMetal;
  EXPECT_FALSE(IsConvCoopMatSupported(info, definition, attr, output));
}

TEST(ConvCoopMatTest, SelectorChoosesCooperativeKernelOrFallback) {
  auto info = CooperativeGpu();
  const auto definition = Definition();
  const auto attr = Attributes();
  const BHWC output(1, 32, 32, 65);
  EXPECT_TRUE(IsGoodTaskSizeForConvCoopMat(attr, output));
  auto operation = SelectConvolution(
      attr, output, info, definition, ModelHints());
  EXPECT_NE(operation->code_.find("coop_mat_mulAdd"), std::string::npos);
  EXPECT_FALSE(IsGoodTaskSizeForConvCoopMat(attr, BHWC(1, 2, 2, 65)));
  operation = SelectConvolution(
      attr, BHWC(1, 2, 2, 65), info, definition, ModelHints());
  EXPECT_EQ(operation->code_.find("coop_mat_mulAdd"), std::string::npos);
  info.opencl_info.extensions.clear();
  operation = SelectConvolution(attr, output, info, definition, ModelHints());
  EXPECT_EQ(operation->code_.find("coop_mat_mulAdd"), std::string::npos);
}

TEST(ConvCoopMatTest, BindsTensorShapesAndRecalculatesGrid) {
  auto operation = CreateConvCoopMat(Definition(), Attributes());
  ShapeTensor src(BHWC(1, 32, 32, 64));
  ShapeTensor dst(BHWC(1, 32, 32, 65));
  operation.SetSrc(&src);
  operation.SetDst(&dst);
  ASSERT_TRUE(operation.BindArguments(&operation.args_).ok());
  const auto& values = operation.args_.GetIntValues();
  EXPECT_EQ(values.at("num_input_channel_groups").value, 4);
  EXPECT_EQ(values.at("num_output_channel_groups").value, 5);
  EXPECT_EQ(values.at("output_channels").value, 65);
  EXPECT_EQ(values.at("stride_x").value, 1);
  auto grid = operation.GetGridSize();
  EXPECT_EQ(grid.x, 32);
  EXPECT_EQ(grid.y, 128);
  EXPECT_EQ(grid.z, 1);
  dst.shape = BHWC(1, 3, 9, 65);
  ASSERT_TRUE(operation.BindArguments(&operation.args_).ok());
  grid = operation.GetGridSize();
  EXPECT_EQ(grid.x, 16);
  EXPECT_EQ(grid.y, 16);
  EXPECT_EQ(grid.z, 1);
}

TEST(ConvCoopMatTest, PacksWeightsAndPadsOutputChannels) {
  const auto attr = Attributes();
  auto operation = CreateConvCoopMat(Definition(), attr);
  const auto& objects = operation.args_.GetObjects();
  const auto& weights =
      static_cast<const BufferDescriptor&>(*objects.at("weights"));
  ASSERT_EQ(weights.data.size(), 80 * 64 * sizeof(float));
  // Decode each tile independently and check both real and padded channels.
  for (int output_group = 0; output_group < 5; ++output_group) {
    for (int input_group = 0; input_group < 4; ++input_group) {
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          const int o = output_group * 16 + row;
          const int i = input_group * 16 + col;
          const int offset =
              ((output_group * 4 + input_group) * 256 + row * 16 + col);
          float value;
          std::memcpy(&value, weights.data.data() + offset * sizeof(float),
                      sizeof(float));
          EXPECT_EQ(value, o < 65 ? attr.weights.data[o * 64 + i] : 0.0f);
        }
      }
    }
  }
  const auto& bias =
      static_cast<const BufferDescriptor&>(*objects.at("biases"));
  ASSERT_EQ(bias.data.size(), 68 * sizeof(float));
  float padded_bias;
  std::memcpy(&padded_bias, bias.data.data() + 65 * sizeof(float), sizeof(float));
  EXPECT_EQ(padded_bias, 0.0f);
}

TEST(ConvCoopMatTest, PacksHalfWeightsAndBias) {
  auto definition = Definition();
  definition.precision = CalculationsPrecision::F16;
  definition.src_tensors[0] =
      {DataType::FLOAT16, TensorStorageType::BUFFER, Layout::HWC};
  definition.dst_tensors = definition.src_tensors;
  auto attr = Attributes();
  for (auto& value : attr.weights.data) {
    value = 1.5f;
  }
  EXPECT_TRUE(IsConvCoopMatSupported(
      CooperativeGpu(), definition, attr, BHWC(1, 32, 32, 65)));
  auto operation = CreateConvCoopMat(definition, attr);
  const auto& objects = operation.args_.GetObjects();
  const auto& weights =
      static_cast<const BufferDescriptor&>(*objects.at("weights"));
  ASSERT_EQ(weights.data.size(), 80 * 64 * sizeof(half));
  half weight;
  std::memcpy(&weight, weights.data.data(), sizeof(half));
  EXPECT_EQ(static_cast<float>(weight), 1.5f);
  std::memcpy(&weight, weights.data.data() + (80 * 64 - 1) * sizeof(half),
              sizeof(half));
  EXPECT_EQ(static_cast<float>(weight), 0.0f);
  const auto& bias =
      static_cast<const BufferDescriptor&>(*objects.at("biases"));
  ASSERT_EQ(bias.data.size(), 68 * sizeof(half));
  half bias_value;
  std::memcpy(&bias_value, bias.data.data(), sizeof(half));
  EXPECT_EQ(static_cast<float>(bias_value), 2.0f);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
