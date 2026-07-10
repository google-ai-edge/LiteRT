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

// GetSupportedNodes is module-private (support.cc) and not public (support.h),
// prioritizing encapsulation over test convenience.
extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 5> kDefaultInputDims = {1, 2, 3, 4, 5};
constexpr std::array<int, 5> kDefaultBroadcastValues = {0, 1, 2, 3, 4};
constexpr std::array<int, 5> kDefaultOutputDims = {1, 2, 3, 4, 5};

struct VersionTestCase {
  int version = 0;
};

// Test suite for broadcast_in_dim logical ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim,
                        GetParam().version,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastInDimOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {1},  // only supported version
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

// Test suite for broadcast_in_dim ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim,
                        GetParam().version,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastInDimOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {0},  // min-1
        {2},  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

class NumInputOutputTest : public testing::Test {};

// Tests for broadcast_in_dim ops for different number of I/O tensors.
TEST_F(NumInputOutputTest, Supports2Inputs1Output) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTest, Rejects1Input) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr, /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects3Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int input2 =
      context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{axis, input, input2},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for broadcast_in_dim op x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<TfLiteType>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedAxis) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteFloat32, kDefaultBroadcastValues);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output =
      context_builder.AddTensor(kTfLiteNoType, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastInDimOps, SupportedDtypeTest,
    ValuesIn<TfLiteType>({
        kTfLiteBFloat16,
        kTfLiteBool,
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
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ConstantTestSuite, RejectsNonConstAxisTensor) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  int axis_data = 3;
  const int axis = context_builder.AddScalarTensor(kTfLiteInt32, &axis_data);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the number of dimensions.
class DimsTest : public testing::Test {};
TEST_F(DimsTest, Rejects0dAxis) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis = context_builder.AddConstTensor(kTfLiteInt32, {});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Rejects2dAxis) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis = context_builder.AddConstTensor(kTfLiteInt32, {1, 1});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsAxisSizeMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis = context_builder.AddConstTensor(kTfLiteInt32, {1});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Rejects0dInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {});
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Rejects6dInput) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 8, 1, 1});
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsDiffInputOutputDims) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output = context_builder.AddTensor(kDefaultDtype, {2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

class ShapeTest : public testing::Test {};
TEST_F(ShapeTest, RejectsInputOutputShapeMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 5});
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

class BroadcastInDimDimensionTest : public testing::Test {};
TEST_F(BroadcastInDimDimensionTest, RejectsNegAxis) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {-1, 0, 1, 2, 3});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(BroadcastInDimDimensionTest, RejectsTooLargeAxis) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axis =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 1, 2, 3, 5});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(BroadcastInDimDimensionTest, RejectsBadMapping) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 2});
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5});
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(BroadcastInDimDimensionTest, SupportsDimOne) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 1, 1});
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(BroadcastInDimDimensionTest, RejectsDimMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 2});
  const int axis = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, kDefaultBroadcastValues);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(BroadcastInDimDimensionTest, RejectsBadTile) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1});
  const int axis = context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0});
  // Second dimension must be 1 b/c it isn't mapped to by the input.
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2});
  context_builder.SetOp(kTfLiteBuiltinStablehloBroadcastInDim, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, axis},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
