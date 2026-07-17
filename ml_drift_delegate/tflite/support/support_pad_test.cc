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
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

constexpr IrModelBuilderOptions kDefaultOptions = {};

//
// PAD op
//
TEST(PadSupportTest, Pad) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 2, 2, 1});
  const int paddings = context_builder.AddConstTensor(kTfLiteInt32, {4, 2});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 1});
  context_builder.SetOp(kTfLiteBuiltinPad, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, paddings},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(PadSupportTest, PadV2) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 2, 2, 1});
  const int paddings = context_builder.AddConstTensor(kTfLiteInt32, {4, 2});
  float const_val = 0.0f;
  const int const_val_tensor =
      context_builder.AddScalarConstTensor(kTfLiteFloat32, &const_val);
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 1});
  context_builder.SetOp(kTfLiteBuiltinPadv2, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, paddings, const_val_tensor},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(PadSupportTest, PadMirror) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 2, 2, 1});
  const int paddings = context_builder.AddConstTensor(kTfLiteInt32, {4, 2});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 1});
  TfLiteMirrorPaddingParams params = {
      .mode = kTfLiteMirrorPaddingReflect,
  };
  context_builder.SetOp(kTfLiteBuiltinMirrorPad, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, paddings},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(PadSupportTest, PadV2NoConstantValue) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 2, 2, 1});
  const int paddings = context_builder.AddConstTensor(kTfLiteInt32, {4, 2});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 1});
  context_builder.SetOp(kTfLiteBuiltinPadv2, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, paddings},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(PadSupportTest, NonConstPadding) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 2, 2, 1});
  const int paddings = context_builder.AddTensor(kTfLiteInt32, {4, 2});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 1});
  context_builder.SetOp(kTfLiteBuiltinPad, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input, paddings},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
