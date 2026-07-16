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

namespace litert::ml_drift::ir {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr std::array<int, 4> kDefaultCondDims = {1, 1, 1, 8};
constexpr std::array<int, 4> kDefaultInputDims = {1, 1, 1, 8};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 1, 1, 8};

class SelectOpTest : public testing::Test {};

class SelectOpParameterizedTest
    : public SelectOpTest,
      public ::testing::WithParamInterface<TfLiteType> {};

TEST_P(SelectOpParameterizedTest, SupportsType) {
  TfLiteType data_type = GetParam();
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteBool, kDefaultCondDims);
  const int in0 = context_builder.AddTensor(data_type, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(data_type, kDefaultInputDims);
  const int out = context_builder.AddTensor(data_type, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 1, nullptr, {cond, in0, in1},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(SelectOpSupport, SelectOpParameterizedTest,
                         ::testing::Values(kTfLiteFloat16, kTfLiteFloat32,
                                           kTfLiteInt8, kTfLiteInt16,
                                           kTfLiteInt32, kTfLiteUInt8,
                                           kTfLiteUInt16, kTfLiteUInt32,
                                           kTfLiteBool, kTfLiteBFloat16));

TEST_F(SelectOpTest, RejectsWrongNumberOfInputs) {
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteBool, kDefaultCondDims);
  const int in0 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 1, nullptr, {cond, in0}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SelectOpTest, RejectsWrongNumberOfOutputs) {
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteBool, kDefaultCondDims);
  const int in0 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 1, nullptr, {cond, in0, in1},
                        {});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SelectOpTest, RejectsUnsupportedCondDType) {
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteInt32, kDefaultCondDims);
  const int in0 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 1, nullptr, {cond, in0, in1},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SelectOpTest, RejectsUnsupportedInputDType) {
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteBool, kDefaultCondDims);
  const int in0 =
      context_builder.AddTensor(kTfLiteComplex64, kDefaultInputDims);
  const int in1 =
      context_builder.AddTensor(kTfLiteComplex64, kDefaultInputDims);
  const int out =
      context_builder.AddTensor(kTfLiteComplex64, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 1, nullptr, {cond, in0, in1},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SelectOpTest, RejectsWrongVersion) {
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteBool, kDefaultCondDims);
  const int in0 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 2, nullptr, {cond, in0, in1},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SelectOpTest, RejectsMismatchedInputOutputShapes) {
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteBool, kDefaultCondDims);
  const int in0 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kTfLiteFloat32, {1, 1, 2, 8});
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 1, nullptr, {cond, in0, in1},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SelectOpTest, SupportsFp32Cond) {
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteFloat32, kDefaultCondDims);
  const int in0 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int in1 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 1, nullptr, {cond, in0, in1},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(SelectOpTest, SupportsScalarif) {
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteBool, kDefaultCondDims);
  const int in0 = context_builder.AddTensor(kTfLiteFloat32, {});  // Scalar
  const int in1 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 1, nullptr, {cond, in0, in1},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(SelectOpTest, SupportsScalarifAndElse) {
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteBool, kDefaultCondDims);
  const int in0 = context_builder.AddTensor(kTfLiteFloat32, {});  // Scalar
  const int in1 = context_builder.AddTensor(kTfLiteFloat32, {});  // Scalar
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 1, nullptr, {cond, in0, in1},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(SelectOpTest, SupportsSingleElementif) {
  StubContextBuilder context_builder;
  const int cond = context_builder.AddTensor(kTfLiteBool, kDefaultCondDims);
  const int in0 = context_builder.AddTensor(kTfLiteFloat32, {1});
  const int in1 = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSelectV2, 1, nullptr, {cond, in0, in1},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

}  // namespace
}  // namespace litert::ml_drift::ir
