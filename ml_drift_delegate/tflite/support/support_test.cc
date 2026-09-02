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

#include "ml_drift_delegate/tflite/support/support.h"

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(OptionToggleTest, HasBoolTensorToggle) {
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kTfLiteBool, {1, 1, 1, 8});
  const int in1 = context_builder.AddTensor(kTfLiteBool, {1, 1, 1, 8});
  const int out = context_builder.AddTensor(kTfLiteBool, {1, 1, 1, 8});
  context_builder.SetOp(kTfLiteBuiltinLogicalAnd, 2, nullptr, {in0, in1},
                        {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);

  IrModelBuilderOptions options;
  options.allow_bool_tensors = true;
  EXPECT_THAT(GetSupportedNodes(context, options), ElementsAre(0));

  options.allow_bool_tensors = false;
  EXPECT_THAT(GetSupportedNodes(context, options), IsEmpty());
}

TEST(OptionToggleTest, HasQuantTensorToggle) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddQuantizedTensor(kTfLiteInt8, {1, 1, 1, 8});
  const int out = context_builder.AddTensor(kTfLiteFloat32, {1, 1, 1, 8});
  context_builder.SetOp(kTfLiteBuiltinDequantize, 1, nullptr, {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);

  IrModelBuilderOptions options;
  options.allow_quant_ops = true;
  EXPECT_THAT(GetSupportedNodes(context, options), ElementsAre(0));

  options.allow_quant_ops = false;
  EXPECT_THAT(GetSupportedNodes(context, options), IsEmpty());
}

TEST(OptionToggleTest, HasQuantTensorOutputQuantized) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kTfLiteFloat32, {1, 1, 1, 8});
  const int out = context_builder.AddQuantizedTensor(kTfLiteInt8, {1, 1, 1, 8});
  context_builder.SetOp(kTfLiteBuiltinConcatenation, 1, nullptr, {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);

  IrModelBuilderOptions options;
  options.allow_quant_ops = false;
  EXPECT_THAT(GetSupportedNodes(context, options), IsEmpty());
}

TEST(OptionToggleTest, HasQuantTensorBuiltinQuantize) {
  StubContextBuilder context_builder;
  const int in = context_builder.AddTensor(kTfLiteFloat32, {1, 1, 1, 8});
  const int out = context_builder.AddTensor(kTfLiteInt8, {1, 1, 1, 8});
  context_builder.SetOp(kTfLiteBuiltinQuantize, 1, nullptr, {in}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);

  IrModelBuilderOptions options;
  options.allow_quant_ops = false;
  EXPECT_THAT(GetSupportedNodes(context, options), IsEmpty());
}

TEST(EmptyTensorTest, RejectEmptyTensorWithZeroDimension) {
  constexpr TfLiteAddParams kAddParams = {};
  {
    StubContextBuilder context_builder;
    const int in0 = context_builder.AddTensor(kTfLiteFloat32, {1, 1, 1, 8});
    const int in1 = context_builder.AddTensor(kTfLiteFloat32, {1, 1, 1, 8});
    const int out = context_builder.AddTensor(kTfLiteFloat32, {1, 1, 1, 8});
    context_builder.SetOp(kTfLiteBuiltinAdd, 1, &kAddParams, {in0, in1}, {out});
    TfLiteContext* context = context_builder.Build();
    ASSERT_NE(context, nullptr);

    IrModelBuilderOptions options;
    EXPECT_THAT(GetSupportedNodes(context, options), ElementsAre(0));
  }
  {
    StubContextBuilder context_builder;
    const int in0 = context_builder.AddTensor(kTfLiteFloat32, {0, 1, 1, 8});
    const int in1 = context_builder.AddTensor(kTfLiteFloat32, {0, 1, 1, 8});
    const int out = context_builder.AddTensor(kTfLiteFloat32, {0, 1, 1, 8});
    context_builder.SetOp(kTfLiteBuiltinAdd, 1, &kAddParams, {in0, in1}, {out});
    TfLiteContext* context = context_builder.Build();
    ASSERT_NE(context, nullptr);

    IrModelBuilderOptions options;
    EXPECT_THAT(GetSupportedNodes(context, options), IsEmpty());
  }
}

TEST(EmptyTensorTest, AcceptNodeWithScalarConstTensor) {
  constexpr int kDefaultAxis = 0;
  constexpr TfLiteSplitParams kSplitParams = {/*num_splits=*/2};
  StubContextBuilder context_builder;
  int axis_val = kDefaultAxis;
  const int axis =
      context_builder.AddScalarConstTensor(kTfLiteInt32, &axis_val);
  const int in = context_builder.AddTensor(kTfLiteFloat32, {2, 4});
  const int out0 = context_builder.AddTensor(kTfLiteFloat32, {1, 4});
  const int out1 = context_builder.AddTensor(kTfLiteFloat32, {1, 4});
  context_builder.SetOp(kTfLiteBuiltinSplit, 1, &kSplitParams, {axis, in},
                        {out0, out1});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);

  IrModelBuilderOptions options;
  EXPECT_THAT(GetSupportedNodes(context, options), ElementsAre(0));
}

TEST(EmptyTensorTest, AcceptNodeWithScalarInput) {
  constexpr TfLiteAddParams kAddParams = {};
  StubContextBuilder context_builder;
  const int in0 = context_builder.AddTensor(kTfLiteFloat32, {});
  const int in1 = context_builder.AddTensor(kTfLiteFloat32, {});
  const int out = context_builder.AddTensor(kTfLiteFloat32, {});
  context_builder.SetOp(kTfLiteBuiltinAdd, 1, &kAddParams, {in0, in1}, {out});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);

  IrModelBuilderOptions options;
  EXPECT_THAT(GetSupportedNodes(context, options), ElementsAre(0));
}

}  // namespace
}  // namespace litert::ml_drift::ir
