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
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr std::array<int, 4> kDefaultInputDims = {1, 2, 3, 4};

class TransposeOpTest : public testing::Test {};

TEST_F(TransposeOpTest, RejectsWrongNumberOfInputs) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  context_builder.SetOp(kTfLiteBuiltinTranspose, 1, nullptr, {input}, {out});

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(TransposeOpTest, RejectsNonConstantPerm) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int perm =
      context_builder.AddTensor(kTfLiteInt32, {4});  // Non-constant
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  context_builder.SetOp(kTfLiteBuiltinTranspose, 1, nullptr, {input, perm},
                        {out});

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(TransposeOpTest, RejectsWrongVersion) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int perm = context_builder.AddConstTensor(kTfLiteInt32, {4});
  const int out = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 2, 3});
  context_builder.SetOp(kTfLiteBuiltinTranspose, 10, nullptr, {input, perm},
                        {out});  // Version 10

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(TransposeOpTest, SupportsVersion9) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int perm = context_builder.AddConstTensor(kTfLiteInt32, {4});
  const int out = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 2, 3});
  context_builder.SetOp(kTfLiteBuiltinTranspose, 9, nullptr, {input, perm},
                        {out});  // Version 9

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(TransposeOpTest, RejectsInvalidPermSize) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int perm = context_builder.AddConstTensor(kTfLiteInt32, {1});  // Size 1
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  context_builder.SetOp(kTfLiteBuiltinTranspose, 1, nullptr, {input, perm},
                        {out});

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(TransposeOpTest, Supports5DTranspose) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 2, 3, 4, 5});
  const int perm = context_builder.AddConstTensor(kTfLiteInt32, {5});
  const int out = context_builder.AddTensor(kTfLiteFloat32, {1, 5, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinTranspose, 1, nullptr, {input, perm},
                        {out});

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

class TransposeOpDataTypeTest : public testing::TestWithParam<TfLiteType> {};

TEST_P(TransposeOpDataTypeTest, SupportsValidTranspose) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int perm = context_builder.AddConstTensor(kTfLiteInt32, {4});
  const int out = context_builder.AddTensor(GetParam(), {1, 4, 2, 3});
  context_builder.SetOp(kTfLiteBuiltinTranspose, 1, nullptr, {input, perm},
                        {out});

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(DataType, TransposeOpDataTypeTest,
                         ::testing::Values(  // clang-format off
                             // go/keep-sorted start numeric=yes
                             kTfLiteBool,
                             kTfLiteFloat16,
                             kTfLiteFloat32,
                             kTfLiteInt8,
                             kTfLiteInt16,
                             kTfLiteInt32,
                             kTfLiteUInt8,
                             kTfLiteUInt16,
                             kTfLiteUInt32
                             // go/keep-sorted end
                             // clang-format on
                             ));

}  // namespace
}  // namespace litert::ml_drift::ir
