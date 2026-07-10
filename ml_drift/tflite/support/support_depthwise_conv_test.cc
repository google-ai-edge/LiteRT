// Copyright 2025 Google LLC.
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

#include <array>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"

namespace litert::ml_drift::ir {

// GetSupportedNodes is module-private (support.cc) and not public (support.h)
extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::TestWithParam;

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 5, 5, 3};
constexpr std::array<int, 4> kDefaultWeightsDims = {1, 3, 3, 6};
constexpr std::array<int, 1> kDefaultBiasDims = {6};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 3, 3, 6};
constexpr TfLiteDepthwiseConvParams kDefaultConvParams = {
    .padding = kTfLitePaddingValid,
    .stride_width = 1,
    .stride_height = 1,
    .depth_multiplier = 2,
    .activation = kTfLiteActNone,
    .dilation_width_factor = 1,
    .dilation_height_factor = 1,
};

class DepthwiseConv2dSupportTest : public TestWithParam<int> {
 protected:
  void SetUp() override {
    builder_.AddTensor(kDefaultDtype, kDefaultInputDims);
    builder_.AddTensor(kDefaultDtype, kDefaultWeightsDims);
    builder_.AddConstTensor(kDefaultDtype, kDefaultBiasDims);
    builder_.AddTensor(kDefaultDtype, kDefaultOutputDims);
    builder_.SetOp(kTfLiteBuiltinDepthwiseConv2d, GetParam(),
                   &kDefaultConvParams, /*inputs=*/{0, 1, 2},
                   /*outputs=*/{3});
  }

  StubContextBuilder builder_;
};

TEST_P(DepthwiseConv2dSupportTest, SupportedVersions) {
  TfLiteContext* context = builder_.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(Versions, DepthwiseConv2dSupportTest,
                         ::testing::Range(1, 7));  // Versions 1 through 6

class DepthwiseConv2dUnsupportedTest : public TestWithParam<int> {
 protected:
  void SetUp() override {
    builder_.AddTensor(kDefaultDtype, kDefaultInputDims);
    builder_.AddTensor(kDefaultDtype, kDefaultWeightsDims);
    builder_.AddTensor(kDefaultDtype, kDefaultOutputDims);
    builder_.SetOp(kTfLiteBuiltinDepthwiseConv2d, GetParam(),
                   &kDefaultConvParams, /*inputs=*/{0, 1}, /*outputs=*/{2});
  }

  StubContextBuilder builder_;
};

TEST_P(DepthwiseConv2dUnsupportedTest, UnsupportedVersions) {
  TfLiteContext* context = builder_.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(Versions, DepthwiseConv2dUnsupportedTest,
                         ::testing::Values(0, 7));

class DepthwiseConv2dConfigTest : public ::testing::Test {};

TEST_F(DepthwiseConv2dConfigTest, ValidWithoutBias) {
  StubContextBuilder builder;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &kDefaultConvParams,
                /*inputs=*/{0, 1}, /*outputs=*/{2});
  TfLiteContext* context = builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(DepthwiseConv2dConfigTest, RejectsWrongNumberOfInputs) {
  StubContextBuilder builder;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &kDefaultConvParams,
                /*inputs=*/{0}, /*outputs=*/{1});
  TfLiteContext* context = builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DepthwiseConv2dConfigTest, RejectsWrongNumberOfOutputs) {
  StubContextBuilder builder;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &kDefaultConvParams,
                /*inputs=*/{0, 1}, /*outputs=*/{2, 2});
  TfLiteContext* context = builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DepthwiseConv2dConfigTest, Rejects3DInput) {
  StubContextBuilder builder;
  builder.AddTensor(kDefaultDtype, {1, 5, 3});
  builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &kDefaultConvParams,
                /*inputs=*/{0, 1}, /*outputs=*/{2});
  TfLiteContext* context = builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DepthwiseConv2dConfigTest, RejectsChannelMultiplierMismatch) {
  StubContextBuilder builder;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, {1, 3, 3, 5});  // Should be 6
  builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &kDefaultConvParams,
                /*inputs=*/{0, 1}, /*outputs=*/{2});
  TfLiteContext* context = builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DepthwiseConv2dConfigTest, RejectsIncorrectOutputShape) {
  StubContextBuilder builder;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  builder.AddTensor(kDefaultDtype, {1, 9, 9, 6});  // Should be {1, 3, 3, 6}
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &kDefaultConvParams,
                /*inputs=*/{0, 1}, /*outputs=*/{2});
  TfLiteContext* context = builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DepthwiseConv2dConfigTest, RejectsNonConstantBias) {
  StubContextBuilder builder;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  builder.AddTensor(kDefaultDtype, kDefaultBiasDims);  // Non-constant
  builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &kDefaultConvParams,
                /*inputs=*/{0, 1, 2}, /*outputs=*/{3});
  TfLiteContext* context = builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DepthwiseConv2dConfigTest, SupportsConstantBias) {
  StubContextBuilder builder;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  builder.AddConstTensor(kDefaultDtype, kDefaultBiasDims);
  builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &kDefaultConvParams,
                /*inputs=*/{0, 1, 2}, /*outputs=*/{3});
  TfLiteContext* context = builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(DepthwiseConv2dConfigTest, SupportsSamePadding) {
  StubContextBuilder builder;
  TfLiteDepthwiseConvParams params = kDefaultConvParams;
  params.padding = kTfLitePaddingSame;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  builder.AddTensor(kDefaultDtype, {1, 5, 5, 6});
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &params,
                /*inputs=*/{0, 1}, /*outputs=*/{2});
  TfLiteContext* context = builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(DepthwiseConv2dConfigTest, SupportsStride) {
  StubContextBuilder builder;
  TfLiteDepthwiseConvParams params = kDefaultConvParams;
  params.stride_height = 2;
  params.stride_width = 2;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  builder.AddTensor(kDefaultDtype, {1, 2, 2, 6});  // (5-3)/2 + 1 = 2
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &params,
                /*inputs=*/{0, 1}, /*outputs=*/{2});
  TfLiteContext* context = builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

class ActivationTest : public testing::Test {};
TEST_F(ActivationTest, SupportsRelu) {
  StubContextBuilder builder;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  constexpr TfLiteDepthwiseConvParams params = {
      .padding = kTfLitePaddingValid,
      .stride_width = 1,
      .stride_height = 1,
      .depth_multiplier = 2,
      .activation = kTfLiteActRelu,
      .dilation_width_factor = 1,
      .dilation_height_factor = 1,
  };
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &params,
                /*inputs=*/{0, 1}, /*outputs=*/{2});
  TfLiteContext* context = builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(ActivationTest, RejectsActivationTooLarge) {
  StubContextBuilder builder;
  builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  constexpr TfLiteDepthwiseConvParams params = {
      .padding = kTfLitePaddingValid,
      .stride_width = 1,
      .stride_height = 1,
      .depth_multiplier = 2,
      .activation = static_cast<TfLiteFusedActivation>(7),
      .dilation_width_factor = 1,
      .dilation_height_factor = 1,
  };
  builder.SetOp(kTfLiteBuiltinDepthwiseConv2d, 1, &params,
                /*inputs=*/{0, 1}, /*outputs=*/{2});
  TfLiteContext* context = builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
