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
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/schema/schema_generated.h"

// These tests indirectly verify IsNodeSupported through GetOpsToReplace,
// which in turn uses GetSupportedNodes to leverage existing matchers.
//
// Note that the functionality of tflite::delegates::GraphPartitionHelper is
// intentionally NOT tested, as that's an implementation detail and that should
// be covered by its own unit tests.

namespace litert::ml_drift::ir {

// GetSupportedNodes is module-private (support.cc) and not public (support.h),
// prioritizing encapsulation over test convenience.
extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

using ::testing::Combine;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 3> kDefaultFirstInputDims = {2, 3, 4};
constexpr std::array<int, 3> kDefaultSecondInputDims = {2, 4, 5};
constexpr std::array<int, 3> kDefaultOutputDims = {2, 3, 5};

// Test suite for batch matmul ops for different number of I/O tensors.
using NumInputOutputTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(NumInputOutputTest, Supports2RuntimeInputs1Output) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a =
      context_builder.AddTensor(kDefaultDtype, kDefaultFirstInputDims);
  const int b =
      context_builder.AddTensor(kDefaultDtype, kDefaultSecondInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Supports1RuntimeInput1ConstantInput1Output) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a =
      context_builder.AddTensor(kDefaultDtype, {2, 3, 4});
  const int b =
      context_builder.AddConstTensor(kDefaultDtype, {4, 5});
  const int c = context_builder.AddTensor(kDefaultDtype, {2, 3, 5});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Rejects2ConstantInputs1Output) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultFirstInputDims);
  const int b =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultSecondInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects0Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a =
      context_builder.AddTensor(kDefaultDtype, kDefaultFirstInputDims);
  const int b =
      context_builder.AddTensor(kDefaultDtype, kDefaultSecondInputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects3Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a =
      context_builder.AddTensor(kDefaultDtype, kDefaultFirstInputDims);
  const int b =
      context_builder.AddTensor(kDefaultDtype, kDefaultSecondInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1});
  const int d = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects0Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a =
      context_builder.AddTensor(kDefaultDtype, kDefaultFirstInputDims);
  const int b =
      context_builder.AddTensor(kDefaultDtype, kDefaultSecondInputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects2Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a =
      context_builder.AddTensor(kDefaultDtype, kDefaultFirstInputDims);
  const int b =
      context_builder.AddTensor(kDefaultDtype, kDefaultSecondInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c, d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    BatchMatMulOps, NumInputOutputTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinBatchMatmul,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for batch matmul ops x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<
    std::tuple<TfLiteBuiltinOperator, /*first_input_dtype=*/TfLiteType,
               /*second_input_dtype=*/TfLiteType, /*output_dtype=*/TfLiteType>>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  const auto [op, i0dtype, i1dtype, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(i0dtype, kDefaultFirstInputDims);
  const int b = context_builder.AddTensor(i1dtype, kDefaultSecondInputDims);
  const int c = context_builder.AddTensor(odtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput) {
  const auto [op, _, i1dtype, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int a =
      context_builder.AddTensor(kTfLiteNoType, kDefaultFirstInputDims);
  const int b = context_builder.AddTensor(i1dtype, kDefaultSecondInputDims);
  const int c = context_builder.AddTensor(odtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  const auto [op, i0dtype, i1dtype, _] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(i0dtype, kDefaultFirstInputDims);
  const int b = context_builder.AddTensor(i1dtype, kDefaultSecondInputDims);
  const int c = context_builder.AddTensor(kTfLiteNoType, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    BatchMatMulOps, SupportedDtypeTest,
    Combine(ValuesIn<TfLiteBuiltinOperator>({
                // clang-format off
                // go/keep-sorted start
                kTfLiteBuiltinBatchMatmul,
                // go/keep-sorted end
                // clang-format on
            }),
            // first_input_dtype
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
            // second_input_dtype
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
      return absl::StrCat(
          ::tflite::EnumNamesBuiltinOperator()[std::get<0>(info.param)], "_",
          TfLiteTypeGetName(std::get<1>(info.param)), "_",
          TfLiteTypeGetName(std::get<2>(info.param)), "_",
          TfLiteTypeGetName(std::get<3>(info.param)));
    });

// Test suite for batch matmul ops with different dims.
using DimsTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(DimsTest, Supports4dInput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 4, 5});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 5});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, Supports5dInput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 1, 2, 4, 5});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 1, 2, 3, 5});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, Rejects1dInput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1});
  const int b = context_builder.AddTensor(kDefaultDtype, {1});
  const int c = context_builder.AddTensor(kDefaultDtype, {1});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Supports4dOutput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 4, 5});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 5});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, Supports5dOutput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 4, 5});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 1, 2, 3, 5});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    BatchMatmulOps, DimsTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinBatchMatmul,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for argmax ops with different shapes.
using ShapesTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(ShapesTest, RejectsInputDimsDimIndexMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {2, 3, 10});
  const int b = context_builder.AddTensor(kDefaultDtype, {2, 20, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(ShapesTest, RejectsInputShapeOutputShapeMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {2, 4, 5});
  const int c = context_builder.AddTensor(kDefaultDtype, {2, 10, 5});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(ShapesTest, SupportsAdjY) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {2, 5, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {2, 3, 5});
  TfLiteBatchMatMulParams params = {.adj_y = true};
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(ShapesTest, SupportsBroadcastingBatchDims) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {2, 1, 1, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 1, 2, 4, 5});
  const int c = context_builder.AddTensor(kDefaultDtype, {2, 1, 2, 3, 5});
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    BatchMatmulOps, ShapesTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinBatchMatmul,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

}  // namespace
}  // namespace litert::ml_drift::ir
