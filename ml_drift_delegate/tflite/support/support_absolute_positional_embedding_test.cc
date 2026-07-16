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
constexpr std::array<int, 4> kDefaultInput0Dims = {1, 1, 4, 2};
constexpr std::array<int, 4> kDefaultInput1Dims = {1, 1, 4, 8};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 1, 4, 2};

class AbsPosEmbedOpTest : public testing::Test {};

TEST_F(AbsPosEmbedOpTest, SupportsFp16) {
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kTfLiteFloat16, kDefaultInput0Dims);
  const int in1 = context_builder.AddTensor(kTfLiteFloat16, kDefaultInput1Dims);
  const int out = context_builder.AddTensor(kTfLiteFloat16, kDefaultOutputDims);
  context_builder.SetOpCustom("custom_call.absolute_positional_embedding",
                              /*version=*/1, /*params=*/nullptr, {in0, in1},
                              {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(AbsPosEmbedOpTest, RejectsWrongNumberOfInputs) {
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInput0Dims);
  const int out = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOpCustom("custom_call.absolute_positional_embedding",
                              /*version=*/1, /*params=*/nullptr, {in0}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(AbsPosEmbedOpTest, RejectsWrongNumberOfOutputs) {
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, kDefaultInput0Dims);
  const int in1 = context_builder.AddTensor(kDefaultDtype, kDefaultInput1Dims);
  context_builder.SetOpCustom("custom_call.absolute_positional_embedding",
                              /*version=*/1, /*params=*/nullptr, {in0, in1},
                              {});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(AbsPosEmbedOpTest, RejectsUnsupportedDType) {
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kTfLiteBool, kDefaultInput0Dims);
  const int in1 = context_builder.AddTensor(kTfLiteBool, kDefaultInput1Dims);
  const int out = context_builder.AddTensor(kTfLiteBool, kDefaultOutputDims);
  context_builder.SetOpCustom("custom_call.absolute_positional_embedding",
                              /*version=*/1, /*params=*/nullptr, {in0, in1},
                              {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(AbsPosEmbedOpTest, RejectsTooManyDimensions) {
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1, 1});
  const int in1 = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1, 1});
  const int out = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1, 1});
  context_builder.SetOpCustom("custom_call.absolute_positional_embedding",
                              /*version=*/1, /*params=*/nullptr, {in0, in1},
                              {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(AbsPosEmbedOpTest, RejectsMismatchedWidth) {
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kDefaultDtype, {1, 1, 5, 2});
  const int in1 = context_builder.AddTensor(kDefaultDtype, {1, 1, 4, 8});
  const int out = context_builder.AddTensor(kDefaultDtype, {1, 1, 5, 2});
  context_builder.SetOpCustom("custom_call.absolute_positional_embedding",
                              /*version=*/1, /*params=*/nullptr, {in0, in1},
                              {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
