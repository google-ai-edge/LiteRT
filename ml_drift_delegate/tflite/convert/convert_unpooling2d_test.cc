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

#include <any>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertUnpooling2dTest
    : public ::testing::TestWithParam<
          std::tuple<TfLitePadding, TfLiteFusedActivation>> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
    std::tie(padding_, activation_) = GetParam();
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
  TfLitePadding padding_;
  TfLiteFusedActivation activation_;
};

TEST_P(ConvertUnpooling2dTest, Parameterized) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinCustom, /*version=*/1);
  model.SetCustomName("custom_call.MaxUnpooling2D");

  model.AddInput(kTfLiteFloat32, {1, 2, 2, 4});   // input
  model.AddInput(kTfLiteInt32, {1, 2, 2, 4});     // indices
  model.AddOutput(kTfLiteFloat32, {1, 4, 4, 4});  // output

  TfLitePoolParams* params =
      reinterpret_cast<TfLitePoolParams*>(calloc(1, sizeof(TfLitePoolParams)));
  params->padding = padding_;
  params->stride_height = 2;
  params->stride_width = 2;
  params->filter_height = 2;
  params->filter_width = 2;
  params->activation = activation_;
  model.SetCustomData(params, sizeof(TfLitePoolParams));

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // Expect 1 MAX_UNPOOLING_2D op [+ 1 Activation op if activation != None]
  const int expected_ops = (activation_ == kTfLiteActNone) ? 1 : 2;
  ASSERT_EQ(ir_model->ops().size(), expected_ops);

  const ::ml_drift::ir::IrOp* unpool_op = ir_model->op(0);
  EXPECT_EQ(unpool_op->name, "max_unpooling");
  const auto* attr =
      std::any_cast<::ml_drift::MaxUnpooling2DAttributes>(&unpool_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->kernel.h, 2);
  EXPECT_EQ(attr->kernel.w, 2);
  EXPECT_EQ(attr->strides.h, 2);
  EXPECT_EQ(attr->strides.w, 2);
  EXPECT_EQ(unpool_op->inputs.size(), 2);
  EXPECT_EQ(unpool_op->outputs.size(), 1);
}

INSTANTIATE_TEST_SUITE_P(
    ConvertUnpooling2dTest, ConvertUnpooling2dTest,
    ::testing::Combine(::testing::Values(kTfLitePaddingSame,
                                         kTfLitePaddingValid),
                       ::testing::Values(kTfLiteActNone, kTfLiteActRelu)),
    [](const ::testing::TestParamInfo<ConvertUnpooling2dTest::ParamType>&
           info) {
      std::string name;
      name +=
          (std::get<0>(info.param) == kTfLitePaddingSame ? "same" : "valid");
      name += (std::get<1>(info.param) == kTfLiteActNone ? "_noact" : "_relu");
      return name;
    });

}  // namespace
}  // namespace litert::ml_drift::ir
