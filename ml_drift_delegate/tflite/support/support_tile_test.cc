// Copyright 2026 Google LLC.
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
#include <cstdint>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"

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

struct VersionTestCase {
  int version = 0;
};

using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const std::array<int, 4> input_dims = {1, 2, 3, 4};
  const std::array<int, 4> output_dims = {1, 4, 3, 8};
  const std::vector<int32_t> multiples = {1, 2, 1, 2};

  const int a = context_builder.AddTensor(kDefaultDtype, input_dims);
  const int m = context_builder.AddConst1dTensor(
      kTfLiteInt32, absl::MakeConstSpan(multiples));
  const int b = context_builder.AddTensor(kDefaultDtype, output_dims);

  context_builder.SetOp(kTfLiteBuiltinTile, GetParam().version,
                        /*params=*/nullptr,
                        /*inputs=*/{a, m}, /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    tileOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {1},
        {2},
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const std::array<int, 4> input_dims = {1, 2, 3, 4};
  const std::array<int, 4> output_dims = {1, 4, 3, 8};
  const std::vector<int32_t> multiples = {1, 2, 1, 2};

  const int a = context_builder.AddTensor(kDefaultDtype, input_dims);
  const int m = context_builder.AddConst1dTensor(
      kTfLiteInt32, absl::MakeConstSpan(multiples));
  const int b = context_builder.AddTensor(kDefaultDtype, output_dims);

  context_builder.SetOp(kTfLiteBuiltinTile, GetParam().version,
                        /*params=*/nullptr,
                        /*inputs=*/{a, m}, /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    tileOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {0},
        {3},
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

class NumInputOutputTest : public testing::Test {};

TEST_F(NumInputOutputTest, Rejects1Input) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 4, 3, 8});
  context_builder.SetOp(kTfLiteBuiltinTile, /*version=*/1,
                        /*params=*/nullptr, /*inputs=*/{a},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTest, Rejects2Outputs) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const std::vector<int32_t> multiples = {1, 2, 1, 2};
  const int m = context_builder.AddConst1dTensor(
      kTfLiteInt32, absl::MakeConstSpan(multiples));
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 4, 3, 8});
  context_builder.SetOp(kTfLiteBuiltinTile, /*version=*/1,
                        /*params=*/nullptr, /*inputs=*/{a, m},
                        /*outputs=*/{b, b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

class ConstantTileTest : public testing::Test {};

TEST_F(ConstantTileTest, RejectsConstantInput) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddConstTensor(kDefaultDtype, {1, 2, 3, 4});
  const std::vector<int32_t> multiples = {1, 2, 1, 2};
  const int m = context_builder.AddConst1dTensor(
      kTfLiteInt32, absl::MakeConstSpan(multiples));
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 4, 3, 8});
  context_builder.SetOp(kTfLiteBuiltinTile, /*version=*/1,
                        /*params=*/nullptr, /*inputs=*/{a, m},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ConstantTileTest, RejectsNonConstantMultiples) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int m = context_builder.AddTensor(kTfLiteInt32, {4});
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 4, 3, 8});
  context_builder.SetOp(kTfLiteBuiltinTile, /*version=*/1,
                        /*params=*/nullptr, /*inputs=*/{a, m},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

class ShapeValidationTest : public testing::Test {};

TEST_F(ShapeValidationTest, RejectsMultiplesLengthMismatch) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {2, 3});
  const std::vector<int32_t> multiples = {1, 2, 1};  // Rank mismatch
  const int m = context_builder.AddConst1dTensor(
      kTfLiteInt32, absl::MakeConstSpan(multiples));
  const int b = context_builder.AddTensor(kDefaultDtype, {2, 6});
  context_builder.SetOp(kTfLiteBuiltinTile, /*version=*/1,
                        /*params=*/nullptr, /*inputs=*/{a, m},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ShapeValidationTest, RejectsOutputShapeMismatch) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {2, 3});
  const std::vector<int32_t> multiples = {1, 2};
  const int m = context_builder.AddConst1dTensor(
      kTfLiteInt32, absl::MakeConstSpan(multiples));
  const int b =
      context_builder.AddTensor(kDefaultDtype, {2, 7});  // Should be {2, 6}
  context_builder.SetOp(kTfLiteBuiltinTile, /*version=*/1,
                        /*params=*/nullptr, /*inputs=*/{a, m},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ShapeValidationTest, Supports5d) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5});
  const std::vector<int32_t> multiples = {1, 1, 1, 1, 1};
  const int m = context_builder.AddConst1dTensor(
      kTfLiteInt32, absl::MakeConstSpan(multiples));
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5});
  context_builder.SetOp(kTfLiteBuiltinTile, /*version=*/1,
                        /*params=*/nullptr, /*inputs=*/{a, m},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(ShapeValidationTest, RejectsUnsupportedDims) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5, 6});
  const std::vector<int32_t> multiples = {1, 1, 1, 1, 1, 1};
  const int m = context_builder.AddConst1dTensor(
      kTfLiteInt32, absl::MakeConstSpan(multiples));
  const int b = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5, 6});
  context_builder.SetOp(kTfLiteBuiltinTile, /*version=*/1,
                        /*params=*/nullptr, /*inputs=*/{a, m},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
