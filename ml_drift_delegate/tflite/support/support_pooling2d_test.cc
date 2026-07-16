// Copyright 2026 The ML Drift Authors.
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

#include <string>
#include <tuple>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

using ::testing::Combine;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

static constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;

inline std::vector<int> DefaultDims() { return {1, 4, 4, 1}; }

TfLitePoolParams CreateDefaultPoolParams(
    TfLiteFusedActivation activation = kTfLiteActNone) {
  return {kTfLitePaddingSame, 2, 2, 2, 2, activation};
}

class Pooling2dSupportTest
    : public TestWithParam<
          std::tuple<TfLiteBuiltinOperator, TfLiteType, std::vector<int>>> {};

TEST_P(Pooling2dSupportTest, SupportsStandardConfigs) {
  const auto& [op_code, dtype, dims] = GetParam();
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(dtype, dims);
  const int output = context_builder.AddTensor(dtype, dims);
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOp(op_code, /*version=*/1, &params, {input}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    Pooling2dTests, Pooling2dSupportTest,
    Combine(Values(kTfLiteBuiltinAveragePool2d, kTfLiteBuiltinMaxPool2d),
            Values(kTfLiteFloat32, kTfLiteFloat16, kTfLiteBFloat16),
            Values(std::vector<int>{1, 4, 4, 1}, std::vector<int>{1, 4, 4},
                   std::vector<int>{4, 4, 4})));

TEST(Pooling2dSupportTest, RejectsUnsupportedDtype) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, DefaultDims());
  const int output = context_builder.AddTensor(kTfLiteInt32, DefaultDims());
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOp(kTfLiteBuiltinMaxPool2d, /*version=*/1, &params,
                        {input}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, RejectsInvalidInputRank) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {4, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOp(kTfLiteBuiltinMaxPool2d, /*version=*/1, &params,
                        {input}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, RejectsInvalidOutputRank) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output = context_builder.AddTensor(kDefaultDtype, {4, 4});
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOp(kTfLiteBuiltinMaxPool2d, /*version=*/1, &params,
                        {input}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, RejectsInvalidSecondOutputRank) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output1 = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  const int output2 = context_builder.AddTensor(kDefaultDtype, {4, 4});
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOpCustom("MaxPoolingWithArgmax2D", /*version=*/1, &params,
                              {input}, {output1, output2});
  context_builder.SetOpCustomInitialData(&params, sizeof(params));
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, Supports2OutputsWithCustomInitialData) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  const int indices = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOpCustom("MaxPoolingWithArgmax2D", /*version=*/1, &params,
                              {input}, {output, indices});
  context_builder.SetOpCustomInitialData(&params, sizeof(params));
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), ElementsAre(0));
}

TEST(Pooling2dSupportTest, Supports2OutputsWithFusedActivation) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  const int indices = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  TfLitePoolParams params = CreateDefaultPoolParams(kTfLiteActRelu);
  context_builder.SetOpCustom("MaxPoolingWithArgmax2D", /*version=*/1, &params,
                              {input}, {output, indices});
  context_builder.SetOpCustomInitialData(&params, sizeof(params));
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), ElementsAre(0));
}

TEST(Pooling2dSupportTest, RejectsInvalidFilterSize) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  TfLitePoolParams params = CreateDefaultPoolParams();
  params.filter_height = -1;
  context_builder.SetOp(kTfLiteBuiltinAveragePool2d, /*version=*/1, &params,
                        {input}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, RejectsInvalidStrideSize) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  TfLitePoolParams params = CreateDefaultPoolParams();
  params.stride_height = 0;
  context_builder.SetOp(kTfLiteBuiltinAveragePool2d, /*version=*/1, &params,
                        {input}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, Rejects2OutputsBuiltinMaxPool) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  const int indices = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOp(kTfLiteBuiltinMaxPool2d, /*version=*/1, &params,
                        {input}, {output, indices});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, Rejects2OutputsBuiltinAvgPool) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  const int indices = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOp(kTfLiteBuiltinAveragePool2d, /*version=*/1, &params,
                        {input}, {output, indices});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, RejectsUnsupportedVersion) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOp(kTfLiteBuiltinMaxPool2d, /*version=*/3, &params,
                        {input}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, RejectsConstantInput) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddConstTensor(kDefaultDtype, DefaultDims());
  const int output = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOp(kTfLiteBuiltinMaxPool2d, /*version=*/1, &params,
                        {input}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, RejectsUnsupportedOutputDtype) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output = context_builder.AddTensor(kTfLiteInt32, DefaultDims());
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOp(kTfLiteBuiltinMaxPool2d, /*version=*/1, &params,
                        {input}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}

TEST(Pooling2dSupportTest, RejectsUnsupportedSecondOutputDtype) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, DefaultDims());
  const int output1 = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  const int output2 = context_builder.AddTensor(kTfLiteInt32, {1, 2, 2, 1});
  TfLitePoolParams params = CreateDefaultPoolParams();
  context_builder.SetOpCustom("MaxPoolingWithArgmax2D", /*version=*/1, &params,
                              {input}, {output1, output2});
  context_builder.SetOpCustomInitialData(&params, sizeof(params));
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, {}), IsEmpty());
}
}  // namespace
}  // namespace litert::ml_drift::ir
