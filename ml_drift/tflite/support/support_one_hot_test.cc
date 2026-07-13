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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api_types.h"
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
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr std::array<int, 4> kDefaultDims = {1, 1, 1, 4};

TEST(VersionTests, SupportedVersion) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

using UnsupportedVersionTest = TestWithParam<int>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, GetParam(),
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    OneHotOps, UnsupportedVersionTest,
    ValuesIn<int>({
        0,  // min-1
        2,  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param);
    });

// Tests for one hot with different number of I/O tensors.
TEST(NumInputOutputTests, Supports4Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(NumInputOutputTests, Rejects3Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects5Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int extra = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off, extra},
                        {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects2Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output1 = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int output2 = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off},
                        {output1, output2});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test dtypes
TEST(SupportDtypesTest, RejectsBadInputDtype) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteNoType, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(SupportDtypesTest, RejectsBadOnDtype) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteNoType, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(SupportDtypesTest, RejectsBadOffDtype) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteNoType, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(SupportDtypesTest, RejectsBadOutputDtype) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteNoType, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test that we can check for constant input
TEST(ConstantTestSuite, RejectsConstInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the number of dimensions.
TEST(DimsTest, Rejects5dInput) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddConstTensor(kTfLiteInt32, {1, 1, 1, 1, 4});
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects0dInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kTfLiteInt32, {});
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects5dOutput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, {1, 1, 1, 1, 4});
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects0dOutput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, {});
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects2dOn) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConstTensor(kTfLiteFloat32, {1, 1});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects2dOff) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConstTensor(kTfLiteFloat32, {1, 1});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects0dOn) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConstTensor(kTfLiteFloat32, {});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects0dOff) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConstTensor(kTfLiteFloat32, {});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, RejectsOnMultipleValues) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConstTensor(kTfLiteFloat32, {2});
  const int off = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {0});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, RejectsOffMultipleValues) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddConstTensor(kTfLiteInt32, kDefaultDims);
  const int depth = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  const int on = context_builder.AddConst1dTensor<int>(kTfLiteFloat32, {1});
  const int off = context_builder.AddConstTensor(kTfLiteFloat32, {2});
  const int output = context_builder.AddTensor(kTfLiteInt32, kDefaultDims);
  context_builder.SetOp(kTfLiteBuiltinOneHot, /*version=*/1,
                        /*params=*/nullptr, {input, depth, on, off}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
