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
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"

using ::testing::ElementsAre;
using ::testing::IsEmpty;

namespace litert::ml_drift::ir {

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};

TEST(DynamicUpdateSliceSupportTest, SupportsValid4D) {
  StubContextBuilder context_builder;
  const int operand = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int update = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});
  const int start_indices = context_builder.AddTensor(kTfLiteInt32, {4});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});

  context_builder.SetOp(kTfLiteBuiltinDynamicUpdateSlice, /*version=*/1,
                        /*builtin_data=*/nullptr,
                        /*inputs=*/{operand, update, start_indices},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(DynamicUpdateSliceSupportTest, RejectsUnsupportedVersion) {
  StubContextBuilder context_builder;
  const int operand = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int update = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});
  const int start_indices = context_builder.AddTensor(kTfLiteInt32, {4});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});

  context_builder.SetOp(kTfLiteBuiltinDynamicUpdateSlice, /*version=*/2,
                        /*builtin_data=*/nullptr,
                        /*inputs=*/{operand, update, start_indices},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DynamicUpdateSliceSupportTest, RejectsConstStartIndices) {
  StubContextBuilder context_builder;
  const int operand = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int update = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});
  const int start_indices =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 2, 2, 0});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});

  context_builder.SetOp(kTfLiteBuiltinDynamicUpdateSlice, /*version=*/1,
                        /*builtin_data=*/nullptr,
                        /*inputs=*/{operand, update, start_indices},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DynamicUpdateSliceSupportTest, RejectsUpdateShapeGreaterThanOperand) {
  StubContextBuilder context_builder;
  const int operand = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});
  const int update = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int start_indices = context_builder.AddTensor(kTfLiteInt32, {4});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinDynamicUpdateSlice, /*version=*/1,
                        /*builtin_data=*/nullptr,
                        /*inputs=*/{operand, update, start_indices},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
