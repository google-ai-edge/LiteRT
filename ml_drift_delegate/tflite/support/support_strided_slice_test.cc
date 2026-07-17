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
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"

using ::testing::ElementsAre;
using ::testing::IsEmpty;

namespace litert::ml_drift::ir {

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};

TEST(StridedSliceSupportTest, SupportsValid4D) {
  TfLiteStridedSliceParams params{};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 2, 2, 0});
  const int end =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 6, 6, 3});
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 1, 1, 1});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/1, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(StridedSliceSupportTest, SupportsMasking) {
  TfLiteStridedSliceParams params = {.begin_mask = 1, .end_mask = 8};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, {99, 2, 2, 0});  // 99 is ignored
  const int end = context_builder.AddConst1dTensor<int>(
      kTfLiteInt32, {1, 6, 6, 99});  // 99 is ignored
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 1, 1, 1});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/1, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(StridedSliceSupportTest, SupportsNegativeIndices) {
  TfLiteStridedSliceParams params{};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, -6, -6, 0});
  const int end =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, -2, -2, 3});
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 1, 1, 1});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/1, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(StridedSliceSupportTest, SupportsNegativeIndicesAndMask) {
  TfLiteStridedSliceParams params = {.begin_mask = 2, .end_mask = 4};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, -6, -6, 0});
  const int end =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, -2, -2, 3});
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 1, 1, 1});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 6, 6, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/1, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(StridedSliceSupportTest, RejectsEllipsisMask) {
  TfLiteStridedSliceParams params = {.ellipsis_mask = 1};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 0, 0, 0});
  const int end =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 8, 8, 3});
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 1, 1, 1});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/1, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(StridedSliceSupportTest, RejectsUnsupportedVersion) {
  TfLiteStridedSliceParams params{};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 0, 0, 0});
  const int end =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 8, 8, 3});
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 1, 1, 1});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/3, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(StridedSliceSupportTest, RejectsShrinkMask) {
  TfLiteStridedSliceParams params = {.shrink_axis_mask = 1};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 0, 0, 0});
  const int end =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 8, 8, 3});
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 1, 1, 1});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {8, 8, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/1, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(StridedSliceSupportTest, RejectsZeroStride) {
  TfLiteStridedSliceParams params{};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 0, 0, 0});
  const int end =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 8, 8, 3});
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 0, 1, 1});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/1, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(StridedSliceSupportTest, RejectsMismatchedOutputShape) {
  TfLiteStridedSliceParams params{};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 2, 2, 0});
  const int end =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 6, 6, 3});
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 1, 1, 1});
  // Correct output shape should be {1, 4, 4, 3}, this is wrong.
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 5, 5, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/1, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(StridedSliceSupportTest, RejectsMismatchedBeginEndStridesShapes) {
  TfLiteStridedSliceParams params{};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {0, 2, 2, 0});
  const int end = context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 6});
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 1, 1, 1});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/1, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(StridedSliceSupportTest, RejectsNonConstantBegin) {
  TfLiteStridedSliceParams params{};
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteFloat32, {1, 8, 8, 3});
  const int begin = context_builder.AddTensor(kTfLiteInt32, {4});
  const int end =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 6, 6, 3});
  const int strides =
      context_builder.AddConst1dTensor<int>(kTfLiteInt32, {1, 1, 1, 1});
  const int output = context_builder.AddTensor(kTfLiteFloat32, {1, 4, 4, 3});

  context_builder.SetOp(kTfLiteBuiltinStridedSlice, /*version=*/1, &params,
                        /*inputs=*/{input, begin, end, strides},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_NE(context, nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
