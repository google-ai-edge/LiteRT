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

TEST(SliceSupportTest, SupportsValidSlice) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 2, 2, 0});
  const int size =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 4, 4, 3});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinSlice, /*version=*/1, nullptr,
                        /*inputs=*/{input, begin, size},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(SliceSupportTest, RejectsMismatchedBeginSizeShapes) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 2, 2, 0});
  const int size =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 4, 4});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinSlice, /*version=*/1, nullptr,
                        /*inputs=*/{input, begin, size},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(SliceSupportTest, RejectsMismatchedBeginInputShapes) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 2, 2});
  const int size =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 4, 4});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinSlice, /*version=*/1, nullptr,
                        /*inputs=*/{input, begin, size},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(SliceSupportTest, RejectsOutOfBoundsSlice) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 2, 2, 0});
  // Size is too large for the second dimension (8), begin is 2, so 2+7=9 > 8.
  const int size =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 7, 4, 3});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinSlice, /*version=*/1, nullptr,
                        /*inputs=*/{input, begin, size},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(SliceSupportTest, RejectsWrongVersion) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 2, 2, 0});
  const int size =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 4, 4, 3});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinSlice, /*version=*/9, nullptr,
                        /*inputs=*/{input, begin, size},
                        /*outputs=*/{output});  // version 9
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(SliceSupportTest, SupportsVersion8) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 2, 2, 0});
  const int size =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 4, 4, 3});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinSlice, /*version=*/8, nullptr,
                        /*inputs=*/{input, begin, size},
                        /*outputs=*/{output});  // version 8
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

}  // namespace
}  // namespace litert::ml_drift::ir
