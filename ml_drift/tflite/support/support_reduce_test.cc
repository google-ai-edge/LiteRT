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
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

// These tests indirectly verify IsNodeSupported through GetOpsToReplace,
// which in turn uses GetSupportedNodes to leverage existing matchers.
//
// Note that the functionality of tflite::delegates::GraphPartitionHelper is
// intentionally NOT tested, as that's an implementation detail and that should
// be covered by its own unit tests.

namespace litert::ml_drift::ir {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;

// GetSupportedNodes is module-private (support.cc) and not public (support.h),
// prioritizing encapsulation over test convenience.
extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 2, 3, 4};
constexpr std::array<int, 1> kAxesDims = {1};
constexpr std::array<int, 3> kDefaultOutputDims = {1, 3, 4};

class ReduceOpTest : public testing::Test {
 protected:
  void SetUp() override { params_.keep_dims = false; }
  TfLiteReducerParams params_ = {};
};

TEST_F(ReduceOpTest, SupportsFp32Mean) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kTfLiteFloat32, kDefaultInputDims);
  const int axes = context_builder.AddConstTensor(kTfLiteInt32, kAxesDims);
  const int out = context_builder.AddTensor(kTfLiteFloat32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinMean, 2, &params_, {in, axes}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(ReduceOpTest, SupportsFp16Mean) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kTfLiteFloat16, kDefaultInputDims);
  const int axes = context_builder.AddConstTensor(kTfLiteInt32, kAxesDims);
  const int out = context_builder.AddTensor(kTfLiteFloat16, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinMean, 2, &params_, {in, axes}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(ReduceOpTest, SupportsInt32Mean) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kTfLiteInt32, kDefaultInputDims);
  const int axes = context_builder.AddConstTensor(kTfLiteInt32, kAxesDims);
  const int out = context_builder.AddTensor(kTfLiteInt32, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinMean, 2, &params_, {in, axes}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(ReduceOpTest, RejectsWrongNumberOfInputs) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinMean, 2, &params_, {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ReduceOpTest, RejectsWrongNumberOfOutputs) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axes = context_builder.AddConstTensor(kTfLiteInt32, kAxesDims);
  context_builder.SetOp(kTfLiteBuiltinMean, 2, &params_, {in, axes}, {});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ReduceOpTest, RejectsUnsupportedInputDType) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kTfLiteBool, kDefaultInputDims);
  const int axes = context_builder.AddConstTensor(kTfLiteInt32, kAxesDims);
  const int out = context_builder.AddTensor(kTfLiteBool, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinMean, 2, &params_, {in, axes}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ReduceOpTest, RejectsConstInput) {
  StubContextBuilder context_builder;
  const int in =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int axes = context_builder.AddConstTensor(kTfLiteInt32, kAxesDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinMean, 2, &params_, {in, axes}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ReduceOpTest, RejectsNonConstAxes) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axes = context_builder.AddTensor(kTfLiteInt32, kAxesDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinMean, 2, &params_, {in, axes}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ReduceOpTest, RejectsMissingParams) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int axes = context_builder.AddConstTensor(kTfLiteInt32, kAxesDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinMean, 2, nullptr, {in, axes}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ReduceOpTest, RejectsInvalidAxisValue) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  int axis_value = 4;
  const int axes =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &axis_value);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinMean, 2, &params_, {in, axes}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
