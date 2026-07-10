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
#include <tuple>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
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

using ::testing::Combine;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

std::vector<uint8_t> CreateTransposeConvBuffer() {
  TfLiteTransposeConvParams params;
  params.stride_height = 2;
  params.stride_width = 2;
  params.padding = kTfLitePaddingSame;
  uint8_t* buffer_t = reinterpret_cast<uint8_t*>(&params);
  return std::vector<uint8_t>(buffer_t,
                              buffer_t + sizeof(TfLiteTransposeConvParams));
}

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 4, 4, 1};
constexpr std::array<int, 4> kDefaultWeightsDims = {1, 4, 4, 1};
constexpr std::array<int, 4> kDefaultBiasDims = {1, 4, 4, 1};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 5, 5, 1};

// Tests for transpose conv ops for different number of I/O tensors.
TEST(NumInputOutputTests, Supports2Inputs1Output) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(NumInputOutputTests, Supports3Inputs1Output) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int bias = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights, bias},
                              {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(NumInputOutputTests, Rejects1Input) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects4Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int bias = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int extra = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights, bias, extra},
                              {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects2Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);

  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output2 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights},
                              {output1, output2});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for convolutions ops x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<
    std::tuple</*input_dtype=*/TfLiteType,
               /*weights_dtype=*/TfLiteType, /*bias_dtype=*/TfLiteType,
               /*output_dtype=*/TfLiteType>>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  const auto [idtype, wdtype, bdtype, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(idtype, kDefaultInputDims);
  const int weights = context_builder.AddTensor(wdtype, kDefaultWeightsDims);
  const int bias = context_builder.AddTensor(bdtype, kDefaultBiasDims);
  const int output = context_builder.AddTensor(odtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights, bias},
                              {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, SupportsSupportedDtypesWithoutOptionalBias) {
  const auto [idtype, wdtype, _, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(idtype, kDefaultInputDims);
  const int weights = context_builder.AddTensor(wdtype, kDefaultWeightsDims);
  const int output = context_builder.AddTensor(odtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput) {
  const auto [_, wdtype, bdtype, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int weights = context_builder.AddTensor(wdtype, kDefaultWeightsDims);
  const int bias = context_builder.AddTensor(bdtype, kDefaultBiasDims);
  const int output = context_builder.AddTensor(odtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights, bias},
                              {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedWeights) {
  const auto [idtype, _, bdtype, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(idtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kTfLiteNoType, kDefaultWeightsDims);
  const int bias = context_builder.AddTensor(bdtype, kDefaultBiasDims);
  const int output = context_builder.AddTensor(odtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights, bias},
                              {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedBias) {
  const auto [idtype, wdtype, _, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(idtype, kDefaultInputDims);
  const int weights = context_builder.AddTensor(wdtype, kDefaultWeightsDims);
  const int bias = context_builder.AddTensor(kTfLiteNoType, kDefaultBiasDims);
  const int output = context_builder.AddTensor(odtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights, bias},
                              {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  const auto [idtype, wdtype, bdtype, _] = GetParam();
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(idtype, kDefaultInputDims);
  const int weights = context_builder.AddTensor(wdtype, kDefaultWeightsDims);
  const int bias = context_builder.AddTensor(bdtype, kDefaultBiasDims);
  const int output =
      context_builder.AddTensor(kTfLiteNoType, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights, bias},
                              {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    TransposeConvOps, SupportedDtypeTest,
    Combine(  // input_dtype
        ValuesIn<TfLiteType>({
            // clang-format off
                // go/keep-sorted start numeric=yes
                kTfLiteBFloat16,
                kTfLiteFloat16,
                kTfLiteFloat32,
                kTfLiteInt8,
                kTfLiteUInt8,
                // go/keep-sorted end
            // clang-format on
        }),
        // weights_dtype
        ValuesIn<TfLiteType>({
            // clang-format off
                // go/keep-sorted start numeric=yes
                kTfLiteFloat16,
                kTfLiteFloat32,
                kTfLiteInt8,
                kTfLiteUInt8,
                // go/keep-sorted end
            // clang-format on
        }),
        // bias_dtype
        ValuesIn<TfLiteType>({
            // clang-format off
                // go/keep-sorted start numeric=yes
                kTfLiteFloat16,
                kTfLiteFloat32,
                // go/keep-sorted end
            // clang-format on
        }),
        // output_dtype
        ValuesIn<TfLiteType>({
            // clang-format off
                // go/keep-sorted start numeric=yes
                kTfLiteBFloat16,
                kTfLiteFloat16,
                kTfLiteFloat32,
                kTfLiteInt8,
                kTfLiteUInt8,
                // go/keep-sorted end
            // clang-format on
        })),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return absl::StrCat(TfLiteTypeGetName(std::get<0>(info.param)), "_",
                          TfLiteTypeGetName(std::get<1>(info.param)), "_",
                          TfLiteTypeGetName(std::get<2>(info.param)), "_",
                          TfLiteTypeGetName(std::get<3>(info.param)));
    });

// Test that we can check for constant input
TEST(ConstantTestSuite, SupportConstInputRuntimeWeights) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int output =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(ConstantTestSuite, SupportRuntimeInputConstWeights) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultWeightsDims);
  const int output =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(ConstantTestSuite, RejectsConstInput) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultWeightsDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(ConstantTestSuite, RejectsConstInputWithOptionalBias) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultWeightsDims);
  const int bias =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultBiasDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights, bias},
                              {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the number of dimensions.
TEST(DimsTest, Rejects5dInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 4, 4, 1, 1});
  const int weights =
      context_builder.AddTensor(kTfLiteInt32, kDefaultWeightsDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects1dInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {4});
  const int weights =
      context_builder.AddTensor(kTfLiteInt32, kDefaultWeightsDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects5dWeights) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights = context_builder.AddTensor(kTfLiteInt32, {1, 4, 4, 1, 1});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects1dWeights) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights = context_builder.AddTensor(kTfLiteInt32, {4});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects5dBias) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kTfLiteInt32, kDefaultWeightsDims);
  const int bias = context_builder.AddTensor(kDefaultDtype, {1, 4, 4, 1, 1});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights, bias},
                              {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects1dBias) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kTfLiteInt32, kDefaultWeightsDims);
  const int bias = context_builder.AddTensor(kDefaultDtype, {4});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights, bias},
                              {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects5dOutput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kTfLiteInt32, kDefaultWeightsDims);
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 4, 4, 1, 1});

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects1dOutput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kTfLiteInt32, kDefaultWeightsDims);
  const int output = context_builder.AddTensor(kDefaultDtype, {4});

  const std::vector<uint8_t> buffer = CreateTransposeConvBuffer();
  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the params.
TEST(ParamsTest, SupportsNullParams) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  context_builder.SetOpCustom("Convolution2DTransposeBias", /*version=*/1,
                              /*params=*/nullptr, {input, weights}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

}  // namespace
}  // namespace litert::ml_drift::ir
