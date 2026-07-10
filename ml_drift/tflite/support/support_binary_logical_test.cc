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
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
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

struct VersionTestCase {
  TfLiteBuiltinOperator op = kTfLiteBuiltinAdd;
  int version = 0;
};

// Test suite for binary logical ops x supported version.
using SupportedVersionTest  = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(GetParam().op, GetParam().version, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    BinaryLogicalOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinBitwiseXor, 1},    // min
        {kTfLiteBuiltinBitwiseXor, 2},    // max
        {kTfLiteBuiltinEqual, 1},         // min
        {kTfLiteBuiltinEqual, 2},         // max
        {kTfLiteBuiltinGreater, 1},       // min
        {kTfLiteBuiltinGreater, 2},       // max
        {kTfLiteBuiltinGreaterEqual, 1},  // min
        {kTfLiteBuiltinGreaterEqual, 2},  // max
        {kTfLiteBuiltinLess, 1},          // min
        {kTfLiteBuiltinLess, 2},          // max
        {kTfLiteBuiltinLessEqual, 1},     // min
        {kTfLiteBuiltinLessEqual, 2},     // max
        {kTfLiteBuiltinLogicalAnd, 1},    // min
        {kTfLiteBuiltinLogicalAnd, 2},    // max
        {kTfLiteBuiltinLogicalOr, 1},     // min
        {kTfLiteBuiltinLogicalOr, 2},     // max
        {kTfLiteBuiltinNotEqual, 1},      // min
        {kTfLiteBuiltinNotEqual, 2},      // max
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });

// Test suite for binary logical ops x unsupported version.
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

INSTANTIATE_TEST_SUITE_P(
    BinaryLogicalOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinBitwiseXor, 0},    // min-1
        {kTfLiteBuiltinBitwiseXor, 3},    // max+1
        {kTfLiteBuiltinEqual, 0},         // min-1
        {kTfLiteBuiltinEqual, 3},         // max+1
        {kTfLiteBuiltinGreater, 0},       // min-1
        {kTfLiteBuiltinGreater, 3},       // max+1
        {kTfLiteBuiltinGreaterEqual, 0},  // min-1
        {kTfLiteBuiltinGreaterEqual, 3},  // max+1
        {kTfLiteBuiltinLess, 0},          // min-1
        {kTfLiteBuiltinLess, 3},          // max+1
        {kTfLiteBuiltinLessEqual, 0},     // min-1
        {kTfLiteBuiltinLessEqual, 3},     // max+1
        {kTfLiteBuiltinLogicalAnd, 0},    // min-1
        {kTfLiteBuiltinLogicalAnd, 3},    // max+1
        {kTfLiteBuiltinLogicalOr, 0},     // min-1
        {kTfLiteBuiltinLogicalOr, 3},     // max+1
        {kTfLiteBuiltinNotEqual, 0},      // min-1
        {kTfLiteBuiltinNotEqual, 3},      // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });

// Test suite for binary logical ops for broadcast support.
using BroadcastTest  = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(BroadcastTest, Supports1dTo2d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{2});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects1dTo2d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, Supports1dTo3d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{3});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects1dTo3d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, Supports1dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{4});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects1dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, Supports2dTo3d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{2, 3});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects2dTo3d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{8, 9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, Supports2dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects2dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{8, 9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, Supports3dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{2, 3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(BroadcastTest, Rejects3dTo4d) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{7, 8, 9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, RejectsInputDimsMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{6, 7, 8, 9});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(BroadcastTest, RejectsInputOutputDimsMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, /*dims=*/{1, 2, 3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, /*dims=*/{6, 7, 8, 9});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    BinaryLogicalOps, BroadcastTest,
    ValuesIn<TfLiteBuiltinOperator>({
        kTfLiteBuiltinBitwiseXor,
        kTfLiteBuiltinEqual,
        kTfLiteBuiltinGreater,
        kTfLiteBuiltinGreaterEqual,
        kTfLiteBuiltinLess,
        kTfLiteBuiltinLessEqual,
        kTfLiteBuiltinLogicalAnd,
        kTfLiteBuiltinLogicalOr,
        kTfLiteBuiltinNotEqual,
    }),
    [](const TestParamInfo<BroadcastTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for binary logical ops for different number of I/O tensors.
using NumInputOutputTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(NumInputOutputTest, Supports2Inputs1Output) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Rejects0Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr, /*inputs=*/{},
                        /*outputs=*/{a});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects1Input) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr, /*inputs=*/{a},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects3Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int d = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects0Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects2Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int d = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c, d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    BinaryLogicalOps, NumInputOutputTest,
    ValuesIn<TfLiteBuiltinOperator>({
        kTfLiteBuiltinBitwiseXor,
        kTfLiteBuiltinEqual,
        kTfLiteBuiltinGreater,
        kTfLiteBuiltinGreaterEqual,
        kTfLiteBuiltinLess,
        kTfLiteBuiltinLessEqual,
        kTfLiteBuiltinLogicalAnd,
        kTfLiteBuiltinLogicalOr,
        kTfLiteBuiltinNotEqual,
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for binary logical ops x supported subject dtypes.
using SupportedDtypeTest =
    TestWithParam<std::tuple<TfLiteBuiltinOperator, TfLiteType>>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  const auto [op, dtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(dtype, kDefaultDims);
  const int b = context_builder.AddTensor(dtype, kDefaultDims);
  const int c = context_builder.AddTensor(dtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput0) {
  const auto [op, dtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kTfLiteNoType, kDefaultDims);
  const int b = context_builder.AddTensor(dtype, kDefaultDims);
  const int c = context_builder.AddTensor(dtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput1) {
  const auto [op, dtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(dtype, kDefaultDims);
  const int b = context_builder.AddTensor(kTfLiteNoType, kDefaultDims);
  const int c = context_builder.AddTensor(dtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  const auto [op, dtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(dtype, kDefaultDims);
  const int b = context_builder.AddTensor(dtype, kDefaultDims);
  const int c = context_builder.AddTensor(kTfLiteNoType, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    BinaryLogicalOps, SupportedDtypeTest,
    Combine(ValuesIn<TfLiteBuiltinOperator>({
                kTfLiteBuiltinBitwiseXor,
                kTfLiteBuiltinEqual,
                kTfLiteBuiltinGreater,
                kTfLiteBuiltinGreaterEqual,
                kTfLiteBuiltinLess,
                kTfLiteBuiltinLessEqual,
                kTfLiteBuiltinLogicalAnd,
                kTfLiteBuiltinLogicalOr,
                kTfLiteBuiltinNotEqual,
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
                kTfLiteBool,
            })),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return absl::StrCat(
          ::tflite::EnumNamesBuiltinOperator()[std::get<0>(info.param)], "_",
          TfLiteTypeGetName(std::get<1>(info.param)));
    });

}  // namespace
}  // namespace litert::ml_drift::ir
