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
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
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
constexpr std::array<int, 4> kDefaultInputDims = {4, 1, 1, 4};
constexpr std::array<int, 4> kDefaultWeightsDims = {4, 1, 1, 4};
constexpr std::array<int, 1> kDefaultBiasDims = {4};
constexpr std::array<int, 4> kDefaultOutputDims = {4, 1, 1, 4};
constexpr TfLiteFullyConnectedParams kDefaultFullyConnectedParams = {
    .weights_format = kTfLiteFullyConnectedWeightsFormatDefault,
};

struct VersionTestCase {
  TfLiteBuiltinOperator op;
  int version = 0;
};

// Test suite for fully connected ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(GetParam().op, GetParam().version,
                        &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    FCOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinFullyConnected,          1},  // min
        {kTfLiteBuiltinFullyConnected,          12},  // max
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });
// clang-format on

// Test suite for fully connected ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(GetParam().op, GetParam().version,
                        &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    FCOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinFullyConnected,          0},  // min-1
        {kTfLiteBuiltinFullyConnected,          13},  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });
// clang-format on

// Test suite for fc ops for different number of I/O tensors.
using NumInputOutputTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(NumInputOutputTest, Supports2Input1Output) {  // without bias
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Supports3Input1Output) {  // with bias
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int d = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Rejects1Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a}, /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects4Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultBiasDims);
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1});
  const int e = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c, d}, /*outputs=*/{e});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects0Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects2Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c, d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, SupportsOptionalBias) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, kTfLiteOptionalTensor},
                        /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, RejectsInvalidInputId) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, -2}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    FCOps, NumInputOutputTest,
    ValuesIn<TfLiteBuiltinOperator>({
        kTfLiteBuiltinFullyConnected,
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for fully connected ops x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<
    std::tuple<TfLiteBuiltinOperator, /*input_dtype=*/TfLiteType,
               /*weights_dtype=*/TfLiteType, /*bias_dtype=*/TfLiteType,
               /*output_dtype=*/TfLiteType>>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  const auto [op, idtype, wdtype, bdtype, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(idtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(wdtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(bdtype, kDefaultBiasDims);
  const int d = context_builder.AddTensor(odtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, SupportsSupportedDtypesWithoutOptionalBias) {
  const auto [op, idtype, wdtype, _, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(idtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(wdtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(odtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput) {
  const auto [op, _, wdtype, bdtype, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int b = context_builder.AddTensor(wdtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(bdtype, kDefaultBiasDims);
  const int d = context_builder.AddTensor(odtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedWeights) {
  const auto [op, idtype, _, bdtype, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(idtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kTfLiteNoType, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(bdtype, kDefaultBiasDims);
  const int d = context_builder.AddTensor(odtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedBias) {
  const auto [op, idtype, wdtype, _, odtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(idtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(wdtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kTfLiteNoType, kDefaultBiasDims);
  const int d = context_builder.AddTensor(odtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  const auto [op, idtype, wdtype, bdtype, _] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(idtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(wdtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(bdtype, kDefaultBiasDims);
  const int d = context_builder.AddTensor(kTfLiteNoType, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    FCOps, SupportedDtypeTest,
    Combine(ValuesIn<TfLiteBuiltinOperator>({
                // clang-format off
                // go/keep-sorted start
                kTfLiteBuiltinFullyConnected,
                // go/keep-sorted end
                // clang-format on
            }),
            // input_dtype
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
                kTfLiteInt4,
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
      return absl::StrCat(
          ::tflite::EnumNamesBuiltinOperator()[std::get<0>(info.param)], "_",
          TfLiteTypeGetName(std::get<1>(info.param)), "_",
          TfLiteTypeGetName(std::get<2>(info.param)), "_",
          TfLiteTypeGetName(std::get<3>(info.param)), "_",
          TfLiteTypeGetName(std::get<4>(info.param)));
    });

// Test suite for fc ops with different dims.
using DimsTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(DimsTest, Supports4dInput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, Rejects5dInput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, RejectsQuantizedF32Weights) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b =
      context_builder.AddQuantizedTensor(kTfLiteFloat32, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, RejectsQuantizedBad4DWeights) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  // Only support quantized 4D weights with 1 in the second and third
  // dimensions.
  const int b = context_builder.AddQuantizedTensor(kTfLiteInt8, {4, 1, 2, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, SupportsQuantizedGood4DWeights) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddQuantizedTensor(kTfLiteInt8, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, SupportsQuantized2DWeights) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddQuantizedTensor(kTfLiteInt8, {4, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, RejectsFloatWeightsSizeMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 1});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, SupportsFloatWeightsSizeMatch) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 2, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, Rejects5dWeights) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Rejects2dBias) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 4});
  const int d = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Supports4dOutput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, Rejects5dOutput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 4, 1, 1, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    FCOps, DimsTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinFullyConnected,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test that we can parse params correctly.
class ParamsTest : public testing::Test {};
TEST_F(ParamsTest, RejectsNullptrParams) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinFullyConnected, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ParamsTest, RejectsBadParams) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const TfLiteFullyConnectedParams bad_params = {
      .weights_format = kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8,
  };
  context_builder.SetOp(kTfLiteBuiltinFullyConnected, /*version=*/1,
                        /*params=*/&bad_params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

class ConstantTestSuite : public testing::Test {};
TEST_F(ConstantTestSuite, RejectsAllConstantInputs) {
  StubContextBuilder context_builder;
  const int a =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int b =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinFullyConnected, /*version=*/1,
                        /*params=*/&kDefaultFullyConnectedParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

class ActivationTest : public testing::Test {};
TEST_F(ActivationTest, SupportsRelu) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  constexpr TfLiteFullyConnectedParams params = {
      .activation = kTfLiteActRelu,
      .weights_format = kTfLiteFullyConnectedWeightsFormatDefault,
  };
  context_builder.SetOp(kTfLiteBuiltinFullyConnected, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(ActivationTest, RejectsActivationTooLarge) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  constexpr TfLiteFullyConnectedParams params = {
      .activation = static_cast<TfLiteFusedActivation>(7),
      .weights_format = kTfLiteFullyConnectedWeightsFormatDefault,
  };
  context_builder.SetOp(kTfLiteBuiltinFullyConnected, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
