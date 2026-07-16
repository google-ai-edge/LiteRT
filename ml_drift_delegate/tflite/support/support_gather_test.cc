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
constexpr std::array<int, 4> kInputDims = {1, 2, 3, 4};
constexpr std::array<int, 1> kIndicesDims = {2};
constexpr std::array<int, 4> kOutputDims = {1, 2, 2, 4};

class GatherOpTest : public ::testing::Test {
 protected:
  TfLiteGatherParams params_ = {.axis = 0, .batch_dims = 0};
};

class GatherOpSupportsDtypeTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  TfLiteGatherParams params_ = {.axis = 0, .batch_dims = 0};
};

TEST_P(GatherOpSupportsDtypeTest, SupportsDtype) {
  TfLiteType dtype = GetParam();
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(dtype, kInputDims);
  const int indices =
      context_builder.AddConstTensor(kTfLiteInt32, kIndicesDims);
  const int out = context_builder.AddTensor(dtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinGather, /*version=*/1, &params_,
                        {in, indices}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(GatherOpTest, GatherOpSupportsDtypeTest,
                         ::testing::Values(kTfLiteBFloat16, kTfLiteBool,
                                           kTfLiteFloat16, kTfLiteFloat32,
                                           kTfLiteInt8, kTfLiteInt16,
                                           kTfLiteInt32, kTfLiteUInt8,
                                           kTfLiteUInt16, kTfLiteUInt32));

TEST_F(GatherOpTest, SupportsRuntimeIndices) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int indices = context_builder.AddTensor(kTfLiteInt32, kIndicesDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinGather, /*version=*/1, &params_,
                        {in, indices}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(GatherOpTest, RejectsWrongNumberOfInputs) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinGather, /*version=*/1, &params_, {in},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(GatherOpTest, RejectsWrongNumberOfOutputs) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int indices =
      context_builder.AddConstTensor(kTfLiteInt32, kIndicesDims);
  context_builder.SetOp(kTfLiteBuiltinGather, /*version=*/1, &params_,
                        {in, indices}, {});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(GatherOpTest, RejectsInvalidOutputDtype) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int indices =
      context_builder.AddConstTensor(kTfLiteInt32, kIndicesDims);
  const int out = context_builder.AddTensor(kTfLiteInt64, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinGather, /*version=*/1, &params_,
                        {in, indices}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(GatherOpTest, RejectsUnsupportedIndicesDtype) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int indices =
      context_builder.AddConstTensor(kTfLiteInt16, kIndicesDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinGather, /*version=*/1, &params_,
                        {in, indices}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(GatherOpTest, RejectsMissingParams) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int indices = context_builder.AddTensor(kTfLiteInt32, kIndicesDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinGather, /*version=*/1, /*params=*/nullptr,
                        {in, indices}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(GatherOpTest, Rejects5DInput) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5});
  const int indices = context_builder.AddTensor(kTfLiteInt32, kIndicesDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinGather, /*version=*/1, &params_,
                        {in, indices}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
