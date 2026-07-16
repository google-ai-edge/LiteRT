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

#include <any>
#include <cstdlib>
#include <memory>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertReluTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertReluTest, Relu) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinRelu, /*version=*/1);
  model.AddInput(GetParam(), {1, 2, 3, 4});
  model.AddOutput(GetParam(), {1, 2, 3, 4});

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* relu_op = ir_model->op(0);
  EXPECT_EQ(relu_op->name, "relu");
  ASSERT_EQ(relu_op->inputs.size(), 1);
  ASSERT_EQ(relu_op->outputs.size(), 1);
  const ::ml_drift::ReLUAttributes* attr =
      std::any_cast<::ml_drift::ReLUAttributes>(&relu_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->activation_min, 0);
  EXPECT_EQ(attr->activation_max, 0);
  EXPECT_EQ(attr->alpha, 0);

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_P(ConvertReluTest, Relu6) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinRelu6, /*version=*/1);
  model.AddInput(GetParam(), {1, 2, 3, 4});
  model.AddOutput(GetParam(), {1, 2, 3, 4});

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* relu_op = ir_model->op(0);
  EXPECT_EQ(relu_op->name, "relu");
  const ::ml_drift::ReLUAttributes* attr =
      std::any_cast<::ml_drift::ReLUAttributes>(&relu_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->activation_min, 0);
  EXPECT_EQ(attr->activation_max, 6);
  EXPECT_EQ(attr->alpha, 0);

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_P(ConvertReluTest, LeakyRelu) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinLeakyRelu, /*version=*/1);
  model.AddInput(GetParam(), {1, 2, 3, 4});
  model.AddOutput(GetParam(), {1, 2, 3, 4});
  TfLiteLeakyReluParams* params =
      reinterpret_cast<TfLiteLeakyReluParams*>(
          malloc(sizeof(TfLiteLeakyReluParams)));
  params->alpha = 0.1f;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* relu_op = ir_model->op(0);
  EXPECT_EQ(relu_op->name, "relu");
  const ::ml_drift::ReLUAttributes* attr =
      std::any_cast<::ml_drift::ReLUAttributes>(&relu_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->activation_min, 0);
  EXPECT_EQ(attr->activation_max, 0);
  EXPECT_NEAR(attr->alpha, 0.1f, 1e-6);

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

INSTANTIATE_TEST_SUITE_P(ConvertReluTest, ConvertReluTest,
                         ::testing::Values(kTfLiteFloat32, kTfLiteFloat16));

}  // namespace
}  // namespace litert::ml_drift::ir
