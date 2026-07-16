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
constexpr std::array<int, 4> kInputDims = {1, 4, 4, 1};
constexpr std::array<int, 4> kOutputDims = {1, 2, 2, 4};

class SpaceToDepthOpTest : public ::testing::Test {};

class SpaceToDepthOpSupportsDtypeTest
    : public ::testing::TestWithParam<TfLiteType> {};

TEST_P(SpaceToDepthOpSupportsDtypeTest, SupportsDtype) {
  TfLiteType dtype = GetParam();
  StubContextBuilder context_builder;
  TfLiteSpaceToDepthParams params = {.block_size = 2};
  const int in = context_builder.AddTensor(dtype, kInputDims);
  const int out = context_builder.AddTensor(dtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSpaceToDepth, /*version=*/1, &params,
                        {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(SpaceToDepthOpTest, SpaceToDepthOpSupportsDtypeTest,
                         ::testing::Values(kTfLiteBFloat16, kTfLiteBool,
                                           kTfLiteFloat16, kTfLiteFloat32,
                                           kTfLiteInt8, kTfLiteInt16,
                                           kTfLiteInt32, kTfLiteUInt8,
                                           kTfLiteUInt16, kTfLiteUInt32));

TEST_F(SpaceToDepthOpTest, RejectsWrongNumberOfInputs) {
  StubContextBuilder context_builder;
  TfLiteSpaceToDepthParams params = {.block_size = 2};
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int in2 = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSpaceToDepth, /*version=*/1, &params,
                        {in, in2}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SpaceToDepthOpTest, RejectsWrongNumberOfOutputs) {
  StubContextBuilder context_builder;
  TfLiteSpaceToDepthParams params = {.block_size = 2};
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  const int out2 = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSpaceToDepth, /*version=*/1, &params,
                        {in}, {out, out2});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SpaceToDepthOpTest, RejectsUnsupportedInputDtype) {
  StubContextBuilder context_builder;
  TfLiteSpaceToDepthParams params = {.block_size = 2};
  const int in = context_builder.AddTensor(kTfLiteInt64, kInputDims);
  const int out = context_builder.AddTensor(kTfLiteInt64, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSpaceToDepth, /*version=*/1, &params,
                        {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SpaceToDepthOpTest, RejectsUnsupportedOutputDtype) {
  StubContextBuilder context_builder;
  TfLiteSpaceToDepthParams params = {.block_size = 2};
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int out = context_builder.AddTensor(kTfLiteInt64, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSpaceToDepth, /*version=*/1, &params,
                        {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SpaceToDepthOpTest, RejectsConstInput) {
  StubContextBuilder context_builder;
  TfLiteSpaceToDepthParams params = {.block_size = 2};
  const int in = context_builder.AddConstTensor(kDefaultDtype, kInputDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSpaceToDepth, /*version=*/1, &params,
                        {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SpaceToDepthOpTest, RejectsBlockSize1) {
  StubContextBuilder context_builder;
  TfLiteSpaceToDepthParams params = {.block_size = 1};
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSpaceToDepth, /*version=*/1, &params,
                        {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SpaceToDepthOpTest, RejectsBlockSize0) {
  StubContextBuilder context_builder;
  TfLiteSpaceToDepthParams params = {.block_size = 0};
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSpaceToDepth, /*version=*/1, &params,
                        {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(SpaceToDepthOpTest, RejectsMissingParams) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int out = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinSpaceToDepth, /*version=*/1,
                        /*params=*/nullptr, {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
