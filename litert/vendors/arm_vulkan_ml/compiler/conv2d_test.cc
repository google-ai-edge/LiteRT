// Copyright 2026 The LiteRT Authors. All Rights Reserved.
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

#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include <string>
#include <gtest/gtest.h>
#include "TosaForSpirvCodegen.hpp"
namespace litert::arm_vulkan_ml {
namespace {
Conv2dSpec Fixture() {
  Conv2dSpec s{};
  s.input_shape = {1, 8, 8, 2};
  s.weight_shape = {3, 3, 3, 2};
  s.output_shape = {1, 3, 3, 3};
  s.padding = {0, 0, 0, 0};
  s.stride = {2, 2};
  s.dilation = {1, 1};
  s.weights.assign(54, 2);
  s.bias = {10, -20, 30};
  s.input_scale = .125;
  s.output_scale = .25;
  s.input_zero_point = -14;
  s.output_zero_point = -128;
  s.weight_scales = {.25, .125, 1e-12};
  s.bias_scales = {.03125, .015625, 1.25e-13};
  s.activation = Activation::kRelu;
  return s;
}
TEST(Conv2d, TrailingSlicePreservesValidOutput) {
  std::string reason;
  auto plan = AnalyzeConv2d(Fixture(), reason);
  ASSERT_TRUE(plan) << reason;
  EXPECT_EQ(plan->convolution_input_shape,
            (std::array<uint32_t, 4>{1, 7, 7, 2}));
  EXPECT_EQ(plan->scales[2].shift, 62);
  auto module = tfsc::CreateModule();
  tfsc::tosa::Graph graph(module, "main");
  auto input = graph.AddInput(tfsc::tosa::Tensor(tfsc::tosa::DataType::int8_t,
                                                {1, 8, 8, 2}), 0);
  std::vector<GraphConstant> constants;
  auto result = EmitConv2d(graph, input, *plan, constants);
  ASSERT_FALSE(tfsc::IsError(result)) << tfsc::GetErrorMsg(result);
  graph.AddOutput(result, 1);
  graph.FinalizeGraph();
  EXPECT_FALSE(tfsc::WriteToBinary(module).empty());
  ASSERT_GE(constants.size(), 4);
  EXPECT_EQ(constants[0].bytes.size(), 54);
  EXPECT_EQ(constants[1].bytes.size(), 12);
}
TEST(Conv2d, GroupedChannelsAreSlicedAndConcatenated) {
  auto spec=Fixture();spec.input_shape[3]=4;
  spec.output_shape[3]=6;spec.weight_shape[0]=6;
  spec.weights.assign(108,2);spec.bias.assign(6,0);
  spec.weight_scales.assign(6,.25);spec.bias_scales.assign(6,.03125);
  std::string reason;auto plan=AnalyzeConv2d(spec,reason);ASSERT_TRUE(plan)<<reason;
  auto module=tfsc::CreateModule();tfsc::tosa::Graph graph(module,"main");
  auto input=graph.AddInput(tfsc::tosa::Tensor(tfsc::tosa::DataType::int8_t,{1,8,8,4}),0);
  std::vector<GraphConstant> constants;auto output=EmitConv2d(graph,input,*plan,constants);
  ASSERT_TRUE(output);ASSERT_FALSE(tfsc::IsError(output))<<tfsc::GetErrorMsg(output);
  graph.AddOutput(output,1);graph.FinalizeGraph();EXPECT_FALSE(tfsc::WriteToBinary(module).empty());
  spec.input_shape[3]=3;EXPECT_FALSE(AnalyzeConv2d(spec,reason));
  spec.input_shape[3]=4;spec.weight_shape[0]=5;spec.output_shape[3]=5;
  EXPECT_FALSE(AnalyzeConv2d(spec,reason));
}
TEST(Conv2d, LargeBiasChannelDoesNotChangeOtherChannelsRounding) {
  auto spec=Fixture();spec.bias[2]=1073741824;
  std::string reason;auto plan=AnalyzeConv2d(spec,reason);ASSERT_TRUE(plan)<<reason;
  EXPECT_FALSE(UsesStagedRounding(*plan));
  auto module=tfsc::CreateModule();tfsc::tosa::Graph graph(module,"main");
  auto input=graph.AddInput(tfsc::tosa::Tensor(tfsc::tosa::DataType::int8_t,{1,8,8,2}),0);
  std::vector<GraphConstant> constants;auto output=EmitConv2d(graph,input,*plan,constants);
  ASSERT_TRUE(output);ASSERT_FALSE(tfsc::IsError(output))<<tfsc::GetErrorMsg(output);
  graph.AddOutput(output,1);graph.FinalizeGraph();EXPECT_FALSE(tfsc::WriteToBinary(module).empty());
}
TEST(Conv2d, RejectsUnsafeOrInconsistentMetadata) {
  std::string reason;
  auto s = Fixture();
  s.bias_scales[1] *= 2;
  EXPECT_FALSE(AnalyzeConv2d(s, reason));
  EXPECT_NE(reason.find("bias scale"), std::string::npos);
  s = Fixture(); s.bias[0] = INT32_MAX;
  EXPECT_FALSE(AnalyzeConv2d(s, reason));
  EXPECT_NE(reason.find("range safety"), std::string::npos);
  s = Fixture(); s.output_shape[1] = 4;
  EXPECT_FALSE(AnalyzeConv2d(s, reason));
  s = Fixture(); s.input_shape[0] = 0;
  EXPECT_FALSE(AnalyzeConv2d(s, reason));
  s = Fixture(); s.weights.pop_back();
  EXPECT_FALSE(AnalyzeConv2d(s, reason));
  s = Fixture(); s.stride[0] = 0;
  EXPECT_FALSE(AnalyzeConv2d(s, reason));
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
