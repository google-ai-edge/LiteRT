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

#include <algorithm>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;

struct UnpackTestCase {
  int version;
  std::vector<int> input_dims;
  int axis;
  int num;
  std::vector<int> output_dims;
};

using UnpackSupportTest = TestWithParam<UnpackTestCase>;

TEST_P(UnpackSupportTest, SupportsValidUnpack) {
  const auto& param = GetParam();
  StubContextBuilder context_builder;

  const int input = context_builder.AddTensor(kDefaultDtype, param.input_dims);
  std::vector<int> output_ids;
  for (int i = 0; i < param.num; ++i) {
    output_ids.push_back(
        context_builder.AddTensor(kDefaultDtype, param.output_dims));
  }

  TfLiteUnpackParams op_params = {.num = param.num, .axis = param.axis};

  context_builder.SetOp(kTfLiteBuiltinUnpack, param.version, &op_params,
                        {input}, output_ids);

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    UnpackSuccess, UnpackSupportTest,
    ValuesIn<UnpackTestCase>({
        {1, {2, 3}, 0, 2, {3}},                    // 2D, axis 0
        {1, {2, 3}, 1, 3, {2}},                    // 2D, axis 1
        {1, {2, 3}, -1, 3, {2}},                   // Negative axis
        {2, {1, 2, 3, 4}, 2, 3, {1, 2, 4}},        // 4D, axis 2, V2
        {1, {1, 2, 3, 4, 5}, 4, 5, {1, 2, 3, 4}},  // 5D, axis 4
    }),
    [](const TestParamInfo<UnpackSupportTest::ParamType>& info) {
      std::string name =
          absl::StrCat("V", info.param.version, "_Rank",
                       info.param.input_dims.size(), "_Axis", info.param.axis);
      std::replace(name.begin(), name.end(), '-', 'n');
      return name;
    });

class UnpackValidationTest : public ::testing::Test {
 protected:
  StubContextBuilder context_builder_;
};

TEST_F(UnpackValidationTest, RejectsUnsupportedVersion) {
  const int a = context_builder_.AddTensor(kDefaultDtype, {2, 3});
  const int b0 = context_builder_.AddTensor(kDefaultDtype, {3});
  const int b1 = context_builder_.AddTensor(kDefaultDtype, {3});
  TfLiteUnpackParams params = {.num = 2, .axis = 0};
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/3, &params, {a},
                         {b0, b1});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(UnpackValidationTest, RejectsMissingParams) {
  const int a = context_builder_.AddTensor(kDefaultDtype, {2, 3});
  const int b0 = context_builder_.AddTensor(kDefaultDtype, {3});
  const int b1 = context_builder_.AddTensor(kDefaultDtype, {3});
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/1, nullptr, {a},
                         {b0, b1});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(UnpackValidationTest, RejectsWrongInputCount) {
  const int b0 = context_builder_.AddTensor(kDefaultDtype, {3});
  const int b1 = context_builder_.AddTensor(kDefaultDtype, {3});
  TfLiteUnpackParams params = {.num = 2, .axis = 0};
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/1, &params, {},
                         {b0, b1});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(UnpackValidationTest, RejectsWrongOutputCount) {
  const int a = context_builder_.AddTensor(kDefaultDtype, {2, 3});
  const int b0 = context_builder_.AddTensor(kDefaultDtype, {3});
  TfLiteUnpackParams params = {.num = 2, .axis = 0};
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/1, &params, {a},
                         {b0});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(UnpackValidationTest, RejectsConstantInput) {
  const int a = context_builder_.AddConstTensor(kDefaultDtype, {2, 3});
  const int b0 = context_builder_.AddTensor(kDefaultDtype, {3});
  const int b1 = context_builder_.AddTensor(kDefaultDtype, {3});
  TfLiteUnpackParams params = {.num = 2, .axis = 0};
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/1, &params, {a},
                         {b0, b1});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(UnpackValidationTest, RejectsUnsupportedDtype) {
  const int a = context_builder_.AddTensor(kTfLiteInt64, {2, 3});
  const int b0 = context_builder_.AddTensor(kTfLiteInt64, {3});
  const int b1 = context_builder_.AddTensor(kTfLiteInt64, {3});
  TfLiteUnpackParams params = {.num = 2, .axis = 0};
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/1, &params, {a},
                         {b0, b1});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(UnpackValidationTest, RejectsUnsupportedRank) {
  const int a = context_builder_.AddTensor(kDefaultDtype, {1, 1, 1, 1, 1, 2});
  const int b0 = context_builder_.AddTensor(kDefaultDtype, {1, 1, 1, 1, 1});
  const int b1 = context_builder_.AddTensor(kDefaultDtype, {1, 1, 1, 1, 1});
  TfLiteUnpackParams params = {.num = 2, .axis = 5};
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/1, &params, {a},
                         {b0, b1});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(UnpackValidationTest, RejectsInvalidAxis) {
  const int a = context_builder_.AddTensor(kDefaultDtype, {2, 3});
  const int b0 = context_builder_.AddTensor(kDefaultDtype, {3});
  const int b1 = context_builder_.AddTensor(kDefaultDtype, {3});
  TfLiteUnpackParams params = {.num = 2, .axis = 2};
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/1, &params, {a},
                         {b0, b1});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(UnpackValidationTest, RejectsNumMismatchWithAxisDim) {
  const int a = context_builder_.AddTensor(kDefaultDtype, {2, 3});
  const int b0 = context_builder_.AddTensor(kDefaultDtype, {3});
  const int b1 = context_builder_.AddTensor(kDefaultDtype, {3});
  const int b2 = context_builder_.AddTensor(kDefaultDtype, {3});
  TfLiteUnpackParams params = {.num = 3, .axis = 0};  // Dim is 2, but num is 3
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/1, &params, {a},
                         {b0, b1, b2});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(UnpackValidationTest, RejectsOutputRankMismatch) {
  const int a = context_builder_.AddTensor(kDefaultDtype, {2, 3});
  const int b0 =
      context_builder_.AddTensor(kDefaultDtype, {2, 3});  // Should be rank 1
  const int b1 = context_builder_.AddTensor(kDefaultDtype, {2, 3});
  TfLiteUnpackParams params = {.num = 2, .axis = 0};
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/1, &params, {a},
                         {b0, b1});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(UnpackValidationTest, RejectsOutputShapeMismatch) {
  const int a = context_builder_.AddTensor(kDefaultDtype, {2, 3});
  const int b0 =
      context_builder_.AddTensor(kDefaultDtype, {2});  // Should be {3}
  const int b1 = context_builder_.AddTensor(kDefaultDtype, {2});
  TfLiteUnpackParams params = {.num = 2, .axis = 0};
  context_builder_.SetOp(kTfLiteBuiltinUnpack, /*version=*/1, &params, {a},
                         {b0, b1});
  TfLiteContext* context = context_builder_.Build();
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir
