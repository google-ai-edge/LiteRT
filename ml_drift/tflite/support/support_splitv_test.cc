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
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
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

// GetSupportedNodes is module-private (support.cc) and not public (support.h),
// prioritizing encapsulation over test convenience.
extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 2, 3, 8};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 2, 3, 4};
constexpr TfLiteSplitParams kDefaultParams = {/*num_splits=*/2};
int kDefaultAxis = -1;

struct VersionTestCase {
  int version = 0;
};

// Test suite for split logical ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, GetParam().version,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    splitOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {1},  // only supported version
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

// Test suite for split logical ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, GetParam().version,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    splitOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {0},  // min-1
        {2},  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

class NumInputOutputTest : public testing::Test {};

// Tests for split ops for different number of I/O tensors.
TEST_F(NumInputOutputTest, Supports3Inputs2Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTest, Rejects2Input) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects4Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int input2 =
      context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis, input2},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for split op x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<TfLiteType>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  const int output1 = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedAxis) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteFloat32, &kDefaultAxis);
  const int output0 = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  const int output1 = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  const int output1 = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kTfLiteNoType, kDefaultOutputDims);
  const int output1 = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    splitOps, SupportedDtypeTest,
    ValuesIn<TfLiteType>({
        kTfLiteFloat32,
        kTfLiteFloat16,
        kTfLiteBFloat16,
    }),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return TfLiteTypeGetName(info.param);
    });

// Test that we can reject constant input
class ConstantTestSuite : public testing::Test {};
TEST_F(ConstantTestSuite, RejectsConstInputTensor) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ConstantTestSuite, RejectsNonConstAxisTensor) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis = context_builder.AddScalarTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ConstantTestSuite, RejectsNonConstNumSplitsTensor) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddTensor(kTfLiteInt32, {1});
  const int axis = context_builder.AddScalarTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test that we can parse params correctly.
class ParamsTest : public testing::Test {};
TEST_F(ParamsTest, RejectsUnsupportedParams) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ParamsTest, RejectsNumSplitsZero) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  TfLiteSplitParams params = {
      .num_splits = 0,
  };
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ParamsTest, RejectsNumSplitsTooBig) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  TfLiteSplitParams params = {
      .num_splits = 3,
  };
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the number of dimensions.
class DimsTest : public testing::Test {};

TEST_F(DimsTest, Rejects2dAxis) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis = context_builder.AddConstTensor(kTfLiteInt32, {1, 1});
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Rejects5dInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 8, 1});
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsDiffOutputDims) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int output1 = context_builder.AddTensor(kDefaultDtype, {2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsDiffInputOutputDims) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 = context_builder.AddTensor(kDefaultDtype, {2, 3, 4});
  const int output1 = context_builder.AddTensor(kDefaultDtype, {2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

class ShapeTest : public testing::Test {};
TEST_F(ShapeTest, RejectsOutputAxisMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 8});
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int output1 = context_builder.AddTensor(kDefaultDtype, {1, 3, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ShapeTest, RejectsInputOuputAxisMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 8});
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 = context_builder.AddTensor(kDefaultDtype, {1, 3, 3, 4});
  const int output1 = context_builder.AddTensor(kDefaultDtype, {1, 3, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ShapeTest, RejectsAxisSumMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 9});
  const int num_splits = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &kDefaultAxis);
  const int output0 = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int output1 = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinSplitV, /*version=*/1,
                        /*params=*/&kDefaultParams,
                        /*inputs=*/{input, num_splits, axis},
                        /*outputs=*/{output0, output1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
