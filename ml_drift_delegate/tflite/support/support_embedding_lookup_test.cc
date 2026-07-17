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
#include "tflite/core/c/builtin_op_data.h"

// These tests indirectly verify IsNodeSupported through GetOpsToReplace,
// which in turn uses GetSupportedNodes to leverage existing matchers.
//
// Note that the functionality of tflite::delegates::GraphPartitionHelper is
// intentionally NOT tested, as that's an implementation detail and that should
// be covered by its own unit tests.

namespace litert::ml_drift::ir {

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

using ::testing::Combine;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {4, 1, 1, 4};
constexpr std::array<int, 4> kDefaultWeightsDims = {4, 1, 1, 4};
constexpr std::array<int, 4> kDefaultOutputDims = {4, 1, 1, 4};

struct VersionTestCase {
  int version = 0;
};

// Test suite for embedding lookup ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, GetParam().version,
                        /*params=*/nullptr,
                        /*inputs=*/{src, weights}, /*outputs=*/{dst});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    embLookupOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {1},  // only supported version
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });
// clang-format on

// Test suite for embedding lookup ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, GetParam().version,
                        /*params=*/nullptr,
                        /*inputs=*/{src, weights}, /*outputs=*/{dst});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    embLookupOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {0},  // min-1
        {2},  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

// Test suite for embedding lookup ops for different number of I/O tensors.
class NumInputOutputTest : public testing::Test {};
TEST_F(NumInputOutputTest, Supports2Input1Output) {  // without bias
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{src, weights}, /*outputs=*/{dst});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTest, Rejects3Input1Output) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int extra = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{src, weights, extra}, /*outputs=*/{dst});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects1Inputs1Output) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{src}, /*outputs=*/{dst});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{src, weights}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects2Outputs) {
  StubContextBuilder context_builder;
    const int src = context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int weights =
      context_builder.AddTensor(kDefaultDtype, kDefaultWeightsDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int extra =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{src, weights}, /*outputs=*/{dst, extra});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for embedding lookup ops x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<
    std::tuple</*input_dtype=*/TfLiteType,
               /*weights_dtype=*/TfLiteType,
               /*output_dtype=*/TfLiteType>>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  const auto [src_dtype, weights_dtype, output_dtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(src_dtype, kDefaultInputDims);
  int b;
  if (weights_dtype == kTfLiteInt4 || weights_dtype == kTfLiteInt8) {
    b = context_builder.AddQuantizedTensor(weights_dtype, kDefaultWeightsDims);
  } else {
    b = context_builder.AddTensor(weights_dtype, kDefaultWeightsDims);
  }
  const int c = context_builder.AddTensor(output_dtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput) {
  const auto [unused, weights_dtype, output_dtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  int b;
  if (weights_dtype == kTfLiteInt4 || weights_dtype == kTfLiteInt8) {
    b = context_builder.AddQuantizedTensor(weights_dtype, kDefaultWeightsDims);
  } else {
    b = context_builder.AddTensor(weights_dtype, kDefaultWeightsDims);
  }
  const int c = context_builder.AddTensor(output_dtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedWeights) {
  const auto [src_dtype, unused, output_dtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(src_dtype, kDefaultInputDims);
  const int b = context_builder.AddTensor(kTfLiteNoType, kDefaultWeightsDims);
  const int c = context_builder.AddTensor(output_dtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  const auto [src_dtype, weights_dtype, unused] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(src_dtype, kDefaultInputDims);
  int b;
  if (weights_dtype == kTfLiteInt4 || weights_dtype == kTfLiteInt8) {
    b = context_builder.AddQuantizedTensor(weights_dtype, kDefaultWeightsDims);
  } else {
    b = context_builder.AddTensor(weights_dtype, kDefaultWeightsDims);
  }
  const int c = context_builder.AddTensor(kTfLiteNoType, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    EmbeddingLookupOps, SupportedDtypeTest,
    Combine(  // input_dtype
            ValuesIn<TfLiteType>({
                // clang-format off
                // go/keep-sorted start numeric=yes
                kTfLiteFloat16,
                kTfLiteFloat32,
                kTfLiteInt32,
                // go/keep-sorted end
            // clang-format on
        }),
        // weights_dtype
        ValuesIn<TfLiteType>({
            // clang-format off
                // go/keep-sorted start numeric=yes
                kTfLiteFloat32,
                kTfLiteInt4,
                kTfLiteInt8,
                // go/keep-sorted end
            // clang-format on
        }),
        // output_dtype
        ValuesIn<TfLiteType>({
            // clang-format off
                // go/keep-sorted start numeric=yes
                kTfLiteFloat16,
                kTfLiteFloat32,
                // go/keep-sorted end
            // clang-format on
        })),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return absl::StrCat(TfLiteTypeGetName(std::get<0>(info.param)), "_",
                          TfLiteTypeGetName(std::get<1>(info.param)), "_",
                          TfLiteTypeGetName(std::get<2>(info.param)));
    });

// Test suite for embedding lookup ops with different dims.
class DimsTest : public testing::Test {};
TEST_F(DimsTest, Supports4dInput) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(DimsTest, Rejects5dInput) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Rejects0dInput) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsQuantizedBad4DWeights) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  // Only support quantized 4D weights with 1 in the second and third
  // dimensions.
  const int b = context_builder.AddQuantizedTensor(kTfLiteInt8, {4, 1, 2, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, SupportsQuantizedGood4DWeights) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddQuantizedTensor(kTfLiteInt8, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(DimsTest, SupportsQuantized2DWeights) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddQuantizedTensor(kTfLiteInt8, {4, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(DimsTest, RejectsQuantized3DWeights) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddQuantizedTensor(kTfLiteInt8, {4, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Rejects5dWeights) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Rejects0dOutput) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Supports4dOutput) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(DimsTest, Rejects5dOutput) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {4, 1, 1, 4});
  const int c = context_builder.AddTensor(kDefaultDtype, {1, 4, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinEmbeddingLookup, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
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
                        /*params=*//*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
