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

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 2, 3, 4};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 2, 3, 4};

struct VersionTestCase {
  int version = 0;
};

// Test suite for clamp ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int max = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int min = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, GetParam().version,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    ClampOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {1},  // only supported version
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

// Test suite for clamp x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int max = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int min = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, GetParam().version,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    ClampOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {0},  // min-1
        {2},  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

class NumInputOutputTest : public testing::Test {};

// Tests for clamp ops for different number of I/O tensors.
TEST_F(NumInputOutputTest, Supports3Inputs1Output) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int max = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int min = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTest, Rejects2Input) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int max = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects4Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int max = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int min = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int extra = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min, extra},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int max = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int min = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects2Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int max = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int min = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output2 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output, output2});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for clamp op x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<TfLiteType>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int max = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int min = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int max = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int min = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedMax) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int max = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int min = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedMin) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int max = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int min = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int max = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int min = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kTfLiteNoType, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    ClampOps, SupportedDtypeTest,
    ValuesIn<TfLiteType>({
        kTfLiteBFloat16,
        kTfLiteFloat16,
        kTfLiteFloat32,
        kTfLiteInt8,
        kTfLiteInt16,
        kTfLiteInt32,
        kTfLiteUInt8,
        kTfLiteUInt16,
        kTfLiteUInt32,
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
  const int max = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int min = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the number of dimensions.
class DimsTest : public testing::Test {};
TEST_F(DimsTest, RejectsMismatchInputOutput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kDefaultDtype, {1, 2, 3, 4});
  const int max = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int min = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 5});
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsMaxDimMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kDefaultDtype, {1, 2, 3, 4});
  const int max = context_builder.AddTensor(kDefaultDtype, {1, 2});
  const int min = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsMaxVectorMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kDefaultDtype, {1, 2, 3, 4});
  const int max = context_builder.AddTensor(kDefaultDtype, {2});
  const int min = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, SupportScalarMax) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int max = context_builder.AddTensor(kDefaultDtype, {1});
  const int min = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(DimsTest, RejectsMinDimMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kDefaultDtype, {1, 2, 3, 4});
  const int max = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int min = context_builder.AddTensor(kDefaultDtype, {1, 2});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsMinVectorMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kDefaultDtype, {1, 2, 3, 4});
  const int max = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int min = context_builder.AddTensor(kDefaultDtype, {2});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, SupportScalarMin) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int max = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int min = context_builder.AddTensor(kDefaultDtype, {1});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloClamp, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, max, min},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

}  // namespace
}  // namespace litert::ml_drift::ir
