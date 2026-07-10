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
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
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
constexpr std::array<int, 4> kInputDims = {1, 2, 3, 4};
constexpr std::array<int, 1> kKDims = {1};
constexpr std::array<int, 4> kOutputDims = {1, 2, 3, 2};  // if K=2

class TopKOpTest : public ::testing::Test {};

class TopKOpSupportsDtypeTest : public ::testing::TestWithParam<TfLiteType> {};

TEST_P(TopKOpSupportsDtypeTest, SupportsDtype) {
  TfLiteType dtype = GetParam();
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(dtype, kInputDims);
  const int k = context_builder.AddConstTensor(kTfLiteInt32, kKDims);
  const int out_vals = context_builder.AddTensor(dtype, kOutputDims);
  const int out_indices = context_builder.AddTensor(kTfLiteInt32, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinTopkV2, /*version=*/1, /*params=*/nullptr,
                        {in, k}, {out_vals, out_indices});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(TopKOpTest, TopKOpSupportsDtypeTest,
                         ::testing::Values(kTfLiteFloat16, kTfLiteFloat32));

class TopKOpRejectsDtypeTest : public ::testing::TestWithParam<TfLiteType> {};

TEST_P(TopKOpRejectsDtypeTest, RejectsDtype) {
  TfLiteType dtype = GetParam();
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(dtype, kInputDims);
  const int k = context_builder.AddConstTensor(kTfLiteInt32, kKDims);
  const int out_vals = context_builder.AddTensor(dtype, kOutputDims);
  const int out_indices = context_builder.AddTensor(kTfLiteInt32, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinTopkV2, /*version=*/1, /*params=*/nullptr,
                        {in, k}, {out_vals, out_indices});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(TopKOpTest, TopKOpRejectsDtypeTest,
                         ::testing::Values(kTfLiteInt8, kTfLiteUInt8,
                                           kTfLiteInt16, kTfLiteInt32));

TEST_F(TopKOpTest, RejectsUnsupportedVersion) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int k = context_builder.AddConstTensor(kTfLiteInt32, kKDims);
  const int out_vals = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  const int out_indices = context_builder.AddTensor(kTfLiteInt32, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinTopkV2, /*version=*/3, /*params=*/nullptr,
                        {in, k}, {out_vals, out_indices});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(TopKOpTest, RejectsRuntimeK) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int k = context_builder.AddTensor(kTfLiteInt32, kKDims);
  const int out_vals = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  const int out_indices = context_builder.AddTensor(kTfLiteInt32, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinTopkV2, /*version=*/1, /*params=*/nullptr,
                        {in, k}, {out_vals, out_indices});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(TopKOpTest, RejectsNonScalarK) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kDefaultDtype, kInputDims);
  const int k = context_builder.AddConstTensor(kTfLiteInt32, {2});
  const int out_vals = context_builder.AddTensor(kDefaultDtype, kOutputDims);
  const int out_indices = context_builder.AddTensor(kTfLiteInt32, kOutputDims);
  context_builder.SetOp(kTfLiteBuiltinTopkV2, /*version=*/1, /*params=*/nullptr,
                        {in, k}, {out_vals, out_indices});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
