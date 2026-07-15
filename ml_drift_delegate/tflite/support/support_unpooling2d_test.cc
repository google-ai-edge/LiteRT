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
#include <cstdint>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"

// These tests indirectly verify IsNodeSupported through GetOpsToReplace,
// which in turn uses GetSupportedNodes to leverage existing matchers.
//
// Note that the functionality of tflite::delegates::GraphPartitionHelper is
// intentionally NOT tested, as that's an implementation detail and that should
// be covered by its own unit tests.

namespace litert::ml_drift::ir {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

std::vector<uint8_t> CreateUnpooling2dBuffer() {
  TfLitePoolParams params;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 2;
  params.stride_width = 2;
  params.padding = kTfLitePaddingSame;
  uint8_t* buffer_t = reinterpret_cast<uint8_t*>(&params);
  return std::vector<uint8_t>(buffer_t, buffer_t + sizeof(TfLitePoolParams));
}

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 4, 4, 1};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 5, 5, 1};

// Tests for unpooling 2d ops for different number of I/O tensors.
TEST(NumInputOutputTests, Supports2Inputs1Output) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int indices =
      context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(NumInputOutputTests, Rejects1Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects3Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int indices =
      context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int extra = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices, extra},
                              {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int indices =
      context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);

  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects2Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int indices =
      context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);

  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output2 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices},
                              {output1, output2});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for unpooling 2d op x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<TfLiteType>;

TEST_P(SupportedDtypeTest, SupportedDtypes) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int indices =
      context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedAxis) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int indices =
      context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int indices =
      context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int indices =
      context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kTfLiteNoType, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    cumsumOps, SupportedDtypeTest,
    ValuesIn<TfLiteType>({
        kTfLiteBFloat16,
        kTfLiteFloat16,
        kTfLiteFloat32,
    }),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return TfLiteTypeGetName(info.param);
    });

// Test that we can reject constant input
TEST(ConstantTestSuite, RejectsConstInput) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int indices =
      context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the number of dimensions.
TEST(DimsTest, Rejects6dInput) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddTensor(kDefaultDtype, {1, 1, 4, 4, 1, 1});
  const int indices =
      context_builder.AddTensor(kTfLiteInt32, {1, 1, 4, 4, 1, 1});
  const int output =
      context_builder.AddTensor(kDefaultDtype, {1, 1, 5, 5, 1, 1});

  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects2dInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {4, 4});
  const int indices = context_builder.AddTensor(kTfLiteInt32, {4, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {5, 5});

  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, RejectsDiffDims) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 4, 4, 1});
  const int indices = context_builder.AddTensor(kTfLiteInt32, {1, 4, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 5, 5, 1});

  const std::vector<uint8_t> buffer = CreateUnpooling2dBuffer();
  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the params.
TEST(ParamsTest, RejectsNullParams) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int indices =
      context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  context_builder.SetOpCustom("custom_call.MaxUnpooling2D", /*version=*/1,
                              /*params=*/nullptr, {input, indices}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
