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
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 8, 8, 16};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 8, 8, 16};
constexpr std::array<int, 1> kDefaultLinearAlphaDims = {16};
constexpr std::array<int, 3> kDefaultHwcAlphaDims = {8, 8, 16};

class PReLUOpTest : public testing::Test {};

TEST_F(PReLUOpTest, SupportsFp32LinearAlpha) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int alpha =
      context_builder.AddConstTensor(kTfLiteFloat32, kDefaultLinearAlphaDims);
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in, alpha}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(PReLUOpTest, SupportsFp16LinearAlpha) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kTfLiteFloat16, kDefaultInputDims);
  const int alpha =
      context_builder.AddConstTensor(kTfLiteFloat16, kDefaultLinearAlphaDims);
  const int out = context_builder.AddTensor(kTfLiteFloat16, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in, alpha}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(PReLUOpTest, SupportsHwcAlpha) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int alpha =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultHwcAlphaDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in, alpha}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(PReLUOpTest, RejectsWrongNumberOfInputs) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(PReLUOpTest, RejectsWrongNumberOfOutputs) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int alpha =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultLinearAlphaDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in, alpha}, {});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(PReLUOpTest, RejectsUnsupportedDType) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int alpha =
      context_builder.AddConstTensor(kTfLiteInt32, kDefaultLinearAlphaDims);
  const int out = context_builder.AddTensor(kTfLiteInt32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in, alpha}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(PReLUOpTest, RejectsConstInput) {
  StubContextBuilder context_builder;
  const int in =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int alpha =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultLinearAlphaDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in, alpha}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(PReLUOpTest, RejectsNonConstAlpha) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int alpha =
      context_builder.AddTensor(kDefaultDtype, kDefaultLinearAlphaDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in, alpha}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(PReLUOpTest, RejectsWrongVersion) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int alpha =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultLinearAlphaDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 2, nullptr, {in, alpha}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(PReLUOpTest, RejectsMismatchedInputOutputShapes) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int alpha =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultLinearAlphaDims);
  const int out = context_builder.AddTensor(kDefaultDtype, {1, 8, 8, 32});
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in, alpha}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(PReLUOpTest, RejectsLinearAlphaMismatchedChannels) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int alpha = context_builder.AddConstTensor(kDefaultDtype, {32});
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in, alpha}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(PReLUOpTest, RejectsHwcAlphaMismatchedShape) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int alpha = context_builder.AddConstTensor(kDefaultDtype, {4, 4, 16});
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinPrelu, 1, nullptr, {in, alpha}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
