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
#include <tuple>
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
#include "tflite/schema/schema_generated.h"

// These tests indirectly verify IsNodeSupported through GetOpsToReplace,
// which in turn uses GetSupportedNodes to leverage existing matchers.
//
// Note that the functionality of tflite::delegates::GraphPartitionHelper is
// intentionally NOT tested, as that's an implementation detail and that should
// be covered by its own unit tests.

using ::testing::Combine;
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
constexpr std::array<int, 4> kDefaultDims = {1, 2, 3, 4};

struct OpParams {
  TfLiteAddParams add{};
  TfLiteSubParams sub{};
  TfLiteMulParams mul{};
  TfLiteDivParams div{};

  void* Get(TfLiteBuiltinOperator op) {
    switch (op) {
      case kTfLiteBuiltinAdd:
        return &add;
      case kTfLiteBuiltinSub:
        return &sub;
      case kTfLiteBuiltinMul:
        return &mul;
      case kTfLiteBuiltinDiv:
        return &div;
      default:
        return nullptr;
    }
  }
};

struct VersionTestCase {
  TfLiteBuiltinOperator op = kTfLiteBuiltinAdd;
  int version = 0;
};

// Test suite for binary arithmetic ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  OpParams params;
  const TfLiteBuiltinOperator op = GetParam().op;
  context_builder.SetOp(op, GetParam().version, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    BinaryArithmeticOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinAdd,                1},  // min
        {kTfLiteBuiltinAdd,                6},  // max
        {kTfLiteBuiltinAtan2,              1},  // min
        {kTfLiteBuiltinAtan2,              2},  // max
        {kTfLiteBuiltinDiv,                1},  // min
        {kTfLiteBuiltinDiv,                2},  // max
        {kTfLiteBuiltinFloorDiv,           1},  // min
        {kTfLiteBuiltinFloorDiv,           3},  // max
        {kTfLiteBuiltinFloorMod,           1},  // min
        {kTfLiteBuiltinFloorMod,           2},  // max
        {kTfLiteBuiltinMaximum,            1},  // min
        {kTfLiteBuiltinMaximum,            4},  // max
        {kTfLiteBuiltinMinimum,            1},  // min
        {kTfLiteBuiltinMinimum,            4},  // max
        {kTfLiteBuiltinMul,                1},  // min
        {kTfLiteBuiltinMul,                8},  // max
        {kTfLiteBuiltinPow,                1},  // min
        {kTfLiteBuiltinPow,                2},  // max
        {kTfLiteBuiltinRightShift,         1},  // only version
        {kTfLiteBuiltinStablehloShiftLeft, 1},  // only version
        {kTfLiteBuiltinSquaredDifference,  1},  // min
        {kTfLiteBuiltinSquaredDifference,  2},  // max
        {kTfLiteBuiltinStablehloRemainder, 1},  // only version
        {kTfLiteBuiltinSub,                1},  // min
        {kTfLiteBuiltinSub,                3},  // max
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });
// clang-format on

// Test suite for binary arithmetic ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(GetParam().op, GetParam().version, /*params=*/nullptr,
                        /*inputs=*/{a}, /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    BinaryArithmeticOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinAdd,                0},  // min-1
        {kTfLiteBuiltinAdd,                7},  // max+1
        {kTfLiteBuiltinAtan2,              0},  // min-1
        {kTfLiteBuiltinAtan2,              3},  // max+1
        {kTfLiteBuiltinDiv,                0},  // min-1
        {kTfLiteBuiltinDiv,                3},  // max+1
        {kTfLiteBuiltinFloorDiv,           0},  // min-1
        {kTfLiteBuiltinFloorDiv,           4},  // max+1
        {kTfLiteBuiltinFloorMod,           0},  // min-1
        {kTfLiteBuiltinFloorMod,           3},  // max+1
        {kTfLiteBuiltinMaximum,            0},  // min-1
        {kTfLiteBuiltinMaximum,            5},  // max+1
        {kTfLiteBuiltinMinimum,            0},  // min-1
        {kTfLiteBuiltinMinimum,            5},  // max+1
        {kTfLiteBuiltinMul,                0},  // min-1
        {kTfLiteBuiltinMul,                9},  // max+1
        {kTfLiteBuiltinPow,                0},  // min-1
        {kTfLiteBuiltinPow,                3},  // max+1
        {kTfLiteBuiltinRightShift,         0},  // min-1
        {kTfLiteBuiltinRightShift,         2},  // max+1
        {kTfLiteBuiltinStablehloShiftLeft, 0},  // min-1
        {kTfLiteBuiltinStablehloShiftLeft, 2},  // max+1
        {kTfLiteBuiltinSquaredDifference,  0},  // min-1
        {kTfLiteBuiltinSquaredDifference,  3},  // max+1
        {kTfLiteBuiltinStablehloRemainder, 0},  // min-1
        {kTfLiteBuiltinStablehloRemainder, 2},  // max+1
        {kTfLiteBuiltinSub,                0},  // min-1
        {kTfLiteBuiltinSub,                4},  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });
// clang-format on

// Test suite for binary arithmetic ops for broadcast support.
using BroadcastTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(BroadcastTest, Supports1dTo2d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{2});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects1dTo2d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, Supports1dTo3d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{3});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects1dTo3d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, Supports1dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{4});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects1dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, Supports2dTo3d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{2, 3});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects2dTo3d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{8, 9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, Supports2dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects2dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{8, 9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, Supports3dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{2, 3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects3dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{7, 8, 9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, RejectsInputDimsMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{6, 7, 8, 9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, RejectsInputOutputDimsMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{6, 7, 8, 9});
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    BinaryArithmeticOps, BroadcastTest,
    ValuesIn<TfLiteBuiltinOperator>({
        kTfLiteBuiltinAdd,
        kTfLiteBuiltinAtan2,
        kTfLiteBuiltinDiv,
        kTfLiteBuiltinFloorDiv,
        kTfLiteBuiltinFloorMod,
        kTfLiteBuiltinMaximum,
        kTfLiteBuiltinMinimum,
        kTfLiteBuiltinMul,
        kTfLiteBuiltinPow,
        kTfLiteBuiltinRightShift,
        kTfLiteBuiltinStablehloShiftLeft,
        kTfLiteBuiltinSquaredDifference,
        kTfLiteBuiltinStablehloRemainder,
        kTfLiteBuiltinSub,
    }),
    [](const TestParamInfo<BroadcastTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });
// clang-format on

// Test suite for binary arithmetic ops for different number of I/O tensors.
using NumInputOutputTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(NumInputOutputTest, Supports2Inputs1Output) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Rejects0Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, params.Get(op), /*inputs=*/{},
                        /*outputs=*/{a});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects1Input) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, params.Get(op), /*inputs=*/{a},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects3Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int d = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b, c},
                        /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects0Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects2Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int d = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c, d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    BinaryArithmeticOps, NumInputOutputTest,
    ValuesIn<TfLiteBuiltinOperator>({
        kTfLiteBuiltinAdd,
        kTfLiteBuiltinAtan2,
        kTfLiteBuiltinDiv,
        kTfLiteBuiltinFloorDiv,
        kTfLiteBuiltinFloorMod,
        kTfLiteBuiltinMaximum,
        kTfLiteBuiltinMinimum,
        kTfLiteBuiltinMul,
        kTfLiteBuiltinPow,
        kTfLiteBuiltinRightShift,
        kTfLiteBuiltinStablehloShiftLeft,
        kTfLiteBuiltinSquaredDifference,
        kTfLiteBuiltinStablehloRemainder,
        kTfLiteBuiltinSub,
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for binary arithmetic ops x supported subject dtypes.
using SupportedDtypeTest =
    TestWithParam<std::tuple<TfLiteBuiltinOperator, TfLiteType>>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  const auto [op, dtype] = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(dtype, kDefaultDims);
  const int b = context_builder.AddTensor(dtype, kDefaultDims);
  const int c = context_builder.AddTensor(dtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput0) {
  const auto [op, dtype] = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kTfLiteNoType, kDefaultDims);
  const int b = context_builder.AddTensor(dtype, kDefaultDims);
  const int c = context_builder.AddTensor(dtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput1) {
  const auto [op, dtype] = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(dtype, kDefaultDims);
  const int b = context_builder.AddTensor(kTfLiteNoType, kDefaultDims);
  const int c = context_builder.AddTensor(dtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  const auto [op, dtype] = GetParam();
  OpParams params;
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(dtype, kDefaultDims);
  const int b = context_builder.AddTensor(dtype, kDefaultDims);
  const int c = context_builder.AddTensor(kTfLiteNoType, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, params.Get(op),
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    BinaryArithmeticOps, SupportedDtypeTest,
    Combine(ValuesIn<TfLiteBuiltinOperator>({
                kTfLiteBuiltinAdd,
                kTfLiteBuiltinAtan2,
                kTfLiteBuiltinDiv,
                kTfLiteBuiltinFloorDiv,
                kTfLiteBuiltinFloorMod,
                kTfLiteBuiltinMaximum,
                kTfLiteBuiltinMinimum,
                kTfLiteBuiltinMul,
                kTfLiteBuiltinPow,
                kTfLiteBuiltinRightShift,
                kTfLiteBuiltinStablehloShiftLeft,
                kTfLiteBuiltinSquaredDifference,
                kTfLiteBuiltinStablehloRemainder,
                kTfLiteBuiltinSub,
            }),
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
            })),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return absl::StrCat(
          ::tflite::EnumNamesBuiltinOperator()[std::get<0>(info.param)], "_",
          TfLiteTypeGetName(std::get<1>(info.param)));
    });

}  // namespace
}  // namespace litert::ml_drift::ir
