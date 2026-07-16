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
constexpr std::array<int, 4> kDefaultInputDims = {1, 2, 3, 4};
constexpr std::array<int, 4> kDefaultWeightsDims = {4, 1, 1, 4};
constexpr std::array<int, 1> kDefaultBiasDims = {4};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 2, 3, 4};
constexpr TfLiteConvParams kDefaultConvParams = {
    .padding = kTfLitePaddingValid,
    .stride_width = 1,
    .stride_height = 1,
    .activation = kTfLiteActNone,
    .dilation_width_factor = 1,
    .dilation_height_factor = 1,
};

struct VersionTestCase {
  TfLiteBuiltinOperator op = kTfLiteBuiltinAdd;
  int version = 0;
};

// Test suite for convolution ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(GetParam().op, GetParam().version, &kDefaultConvParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    ConvOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinConv2d,          1},  // min
        {kTfLiteBuiltinConv2d,          6},  // max
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });
// clang-format on

// Test suite for unary arithmetic ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(GetParam().op, GetParam().version, &kDefaultConvParams,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    ConvOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinConv2d,          0},  // min-1
        {kTfLiteBuiltinConv2d,          7},  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });
// clang-format on

// Test suite for convolution ops for different number of I/O tensors.
using NumInputOutputTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(NumInputOutputTest, Supports2Input1Output) {  // without bias
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Rejects0Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b}, /*outputs=*/{});
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, -2}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    ConvOps, NumInputOutputTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinConv2d,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for convolutions ops x supported subject dtypes.
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
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
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    ConvOps, SupportedDtypeTest,
    Combine(ValuesIn<TfLiteBuiltinOperator>({
                // clang-format off
                // go/keep-sorted start
                kTfLiteBuiltinConv2d,
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

// Test suite for convolution ops with different dims.
using DimsTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(DimsTest, Rejects3dInput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 6, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Supports4dInput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, Rejects5dInput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Rejects3dWeights) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Rejects5dWeights) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Rejects2dBias) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Rejects3dOutput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Supports4dOutput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, Rejects5dOutput) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, RejectsInputChannelWeightsChannelMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 5});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, RejectsInputChannelBiasChannelMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {5});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, RejectsWeightsChannelOutputChannelsMismatch) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {5, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4});
  const int d = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(op, /*version=*/1, &kDefaultConvParams,
                        /*inputs=*/{a, b, c}, /*outputs=*/{d});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, SupportsStrideGt1) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 7, 7, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, {8, 3, 3, 3});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 3, 3, 8});
  constexpr TfLiteConvParams params = {
      .padding = kTfLitePaddingValid,
      .stride_width = 2,
      .stride_height = 2,
      .activation = kTfLiteActNone,
      .dilation_width_factor = 1,
      .dilation_height_factor = 1,
  };
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, SupportsSamePadding) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 5, 5, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, {8, 3, 3, 3});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 5, 5, 8});
  constexpr TfLiteConvParams params = {
      .padding = kTfLitePaddingSame,
      .stride_width = 1,
      .stride_height = 1,
      .activation = kTfLiteActNone,
      .dilation_width_factor = 1,
      .dilation_height_factor = 1,
  };
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, SupportsDilationGt1) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 7, 7, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, {8, 3, 3, 3});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 3, 3, 8});
  constexpr TfLiteConvParams params = {
      .padding = kTfLitePaddingValid,
      .stride_width = 1,
      .stride_height = 1,
      .activation = kTfLiteActNone,
      .dilation_width_factor = 2,
      .dilation_height_factor = 2,
  };
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, SupportsDilationWithSamePadding) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 5, 5, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, {8, 3, 3, 3});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 5, 5, 8});
  constexpr TfLiteConvParams params = {
      .padding = kTfLitePaddingSame,
      .stride_width = 1,
      .stride_height = 1,
      .activation = kTfLiteActNone,
      .dilation_width_factor = 2,
      .dilation_height_factor = 2,
  };
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, SupportsDilationWithStride) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 10, 10, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, {8, 3, 3, 3});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 3, 3, 8});
  constexpr TfLiteConvParams params = {
      .padding = kTfLitePaddingValid,
      .stride_width = 2,
      .stride_height = 2,
      .activation = kTfLiteActNone,
      .dilation_width_factor = 2,
      .dilation_height_factor = 2,
  };
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, SupportsAsymmetricDilation) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 10, 10, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, {8, 3, 3, 3});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 6, 4, 8});
  constexpr TfLiteConvParams params = {
      .padding = kTfLitePaddingValid,
      .stride_width = 1,
      .stride_height = 1,
      .activation = kTfLiteActNone,
      .dilation_width_factor = 3,
      .dilation_height_factor = 2,
  };
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(DimsTest, SupportsAsymmetricStrideAndDilation) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 9, 11, 3});
  const int b = context_builder.AddTensor(kDefaultDtype, {8, 3, 4, 3});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 4, 5, 8});
  constexpr TfLiteConvParams params = {
      .padding = kTfLitePaddingValid,
      .stride_width = 1,
      .stride_height = 2,
      .activation = kTfLiteActNone,
      .dilation_width_factor = 2,
      .dilation_height_factor = 1,
  };
  context_builder.SetOp(op, /*version=*/1, &params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    ConvOps, DimsTest,
    ValuesIn<TfLiteBuiltinOperator>({
        // clang-format off
        // go/keep-sorted start
        kTfLiteBuiltinConv2d,
        // go/keep-sorted end
        // clang-format on
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

class ActivationTest : public testing::Test {};
TEST_F(ActivationTest, SupportsRelu) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  constexpr TfLiteConvParams params = {
      .padding = kTfLitePaddingValid,
      .stride_width = 1,
      .stride_height = 1,
      .activation = kTfLiteActRelu,
      .dilation_width_factor = 1,
      .dilation_height_factor = 1,
  };
  context_builder.SetOp(kTfLiteBuiltinConv2d, /*version=*/1, &params,
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
  constexpr TfLiteConvParams params = {
      .padding = kTfLitePaddingValid,
      .stride_width = 1,
      .stride_height = 1,
      .activation = static_cast<TfLiteFusedActivation>(7),
      .dilation_width_factor = 1,
      .dilation_height_factor = 1,
  };
  context_builder.SetOp(kTfLiteBuiltinConv2d, /*version=*/1, &params,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
