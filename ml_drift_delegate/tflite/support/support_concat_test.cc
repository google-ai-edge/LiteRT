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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"

// These tests indirectly verify IsNodeSupported through GetOpsToReplace,
// which in turn uses GetSupportedNodes to leverage existing matchers.
//
// Note that the functionality of tflite::delegates::GraphPartitionHelper is
// intentionally NOT tested, as that's an implementation detail and that should
// be covered by its own unit tests.

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

namespace litert::ml_drift::ir {

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 2, 3, 4};
constexpr std::array<int, 5> kDefaultOutputDims = {1, 2, 3, 1, 8};
constexpr TfLiteConcatenationParams kDefaultParams = {
    /*axis=*/-1, /*activation=*/kTfLiteActNone};

struct VersionTestCase {
  int version = 0;
};

// Test suite for concat logical ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, GetParam().version,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    ConcatOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {1},  // min
        {6},  // max
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

// Test suite for concat logical ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, GetParam().version,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    ConcatOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {0},  // min-1
        {7},  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

class NumInputOutputTest : public testing::Test {};

// Tests for concat ops for different number of I/O tensors.
TEST_F(NumInputOutputTest, Supports2Inputs1Output) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTest, SupportsDiffDims) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTest, Rejects0Inputs) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams, /*inputs=*/{},
                        /*outputs=*/{a});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects1Input) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams, /*inputs=*/{a},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects2Outputs) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int d = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c, d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for concat logical ops x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<TfLiteType>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int b = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int c = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput0) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int b = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int c = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput1) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int b = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int c = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int b = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int c = context_builder.AddTensor(kTfLiteNoType, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    ConcatOps, SupportedDtypeTest,
    ValuesIn<TfLiteType>({
        kTfLiteFloat32,
        kTfLiteInt32,
        kTfLiteUInt8,
        kTfLiteInt16,
        kTfLiteInt8,
        kTfLiteFloat16,
        kTfLiteUInt32,
        kTfLiteUInt16,
        kTfLiteBFloat16,
        kTfLiteBool,
    }),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return TfLiteTypeGetName(info.param);
    });

// Test that we can reject all constant inputs
class ConstantTestSuite : public testing::Test {};
TEST_F(ConstantTestSuite, RejectsAllConstantTensors) {
  StubContextBuilder context_builder;
  const int a =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int b =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test that we can parse params correctly.
class ParamsTest : public testing::Test {};
TEST_F(ParamsTest, RejectsUnsupportedParams) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the number of dimensions.
class DimsTest : public testing::Test {};
TEST_F(DimsTest, Rejects6dInput) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5, 6});
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Rejects6dOutput) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5, 6});
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsDiffInputDims) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsDiffOutputDims) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 3});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 1});
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

class ShapeTest : public testing::Test {};
TEST_F(ShapeTest, RejectsInputAxisMismatch) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 3, 3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 8});
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ShapeTest, RejectsInputOuputAxisMismatch) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 3, 3, 8});
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ShapeTest, RejectsAxisSumMismatch) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 9});
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

class ActivationTest : public testing::Test {};
TEST_F(ActivationTest, SupportsRelu) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  TfLiteConcatenationParams params = kDefaultParams;
  params.activation = kTfLiteActRelu;
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(ActivationTest, RejectsActivationTooLarge) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  TfLiteConcatenationParams params = kDefaultParams;
  params.activation = static_cast<TfLiteFusedActivation>(7);
  context_builder.SetOp(kTfLiteBuiltinConcatenation, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
