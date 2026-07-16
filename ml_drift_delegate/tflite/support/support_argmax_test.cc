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
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/schema/schema_generated.h"

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
constexpr TfLiteType kDefaultDimIndexDtype = kTfLiteInt32;
constexpr TfLiteType kDefaultUnsupportedDimIndexDtype = kTfLiteFloat32;
// kDefaultDimIndex has to be a non-const value, as the test cases will
// have TfLiteTensor::data.data (non-const pointer) point to it.
int kDefaultDimIndex = 3;
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 2, 3, 4};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 2, 3, 1};

// Test suite for argmax ops for different number of I/O tensors.
using NumInputOutputTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(NumInputOutputTest, Supports2Input1Output) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kDefaultDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Rejects0Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kDefaultDimIndex);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects3Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kDefaultDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1});
  const int d = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects0Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kDefaultDimIndex);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects2Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kDefaultDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c, d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    ArgMaxOps, NumInputOutputTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinArgMax,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for argmax ops x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(SupportedDtypeTest, RejectsUnsupportedDimensionTensor) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddScalarConstTensor(
      kDefaultUnsupportedDimIndexDtype, &kDefaultDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    ArgMaxOps, SupportedDtypeTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinArgMax,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for argmax ops with different dims.
using DimsTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(DimsTest, Supports4dInput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kDefaultDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 1});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, Rejects5dInput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 1, 2, 3, 4});
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kDefaultDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, RejectsNonScalarDimTensor) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b =
      context_builder.AddTensor(kDefaultDimIndexDtype, {1, 1, 2, 3, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Supports4dOutput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kDefaultDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 1});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, Rejects5dOutput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kDefaultDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 1, 2, 3, 1});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    ArgMaxOps, DimsTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinArgMax,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for argmax ops with different shapes.
using ShapesTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(ShapesTest, SupportsNegativeDimIndex) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  int kNegativeDimIndex = -1;
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kNegativeDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(ShapesTest, RejectsInputDimsDimIndexMismatch) {
  int kInvalidDimIndex = kDefaultInputDims.size();
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddScalarConstTensor(kDefaultDimIndexDtype,
                                                     &kInvalidDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(ShapesTest, SupportReshapeInputToMatchOutputShape) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3});
  int kDimIndex = 2;
  const int b =
      context_builder.AddScalarConstTensor(kDefaultDimIndexDtype, &kDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 1, 2, 1});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(ShapesTest, RejectsInputShapeOutputShapeMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3});
  int kDimIndex = 2;
  const int b =
      context_builder.AddScalarConstTensor(kDefaultDimIndexDtype, &kDimIndex);
  const int c = context_builder.AddTensor(kDefaultDtype, {2, 1, 2, 1});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    ArgMaxOps, ShapesTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinArgMax,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

}  // namespace
}  // namespace litert::ml_drift::ir
