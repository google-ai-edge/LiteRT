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
#include <memory>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertBitcastTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        CreateStubDelegate(), DeleteStubDelegate);
  }

  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_ = {
      nullptr, DeleteStubDelegate};
};

TEST_F(ConvertBitcastTest, MaintainPrecisionSize) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBitcast);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddOutput(kTfLiteInt32, {1, 2, 3, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "bitcast");
  EXPECT_EQ(op->inputs.size(), 1);
  EXPECT_EQ(op->outputs.size(), 1);
}

TEST_F(ConvertBitcastTest, DecreasePrecisionSize) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBitcast);
  // Example: 2x2x4 (8-bit) -> 2x2 (32-bit)
  model.AddInput(kTfLiteInt8, {2, 2, 4});
  model.AddOutput(kTfLiteInt32, {2, 2});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 2);
  const ::ml_drift::ir::IrOp* bitcast_op = ir_model->op(0);
  EXPECT_EQ(bitcast_op->name, "bitcast");

  const ::ml_drift::ir::IrOp* reshape_op = ir_model->op(1);
  EXPECT_EQ(reshape_op->name, "reshape");

  const auto* attr =
      std::any_cast<::ml_drift::ReshapeAttributes>(&reshape_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->new_shape, ::ml_drift::BHWC(2, 1, 1, 2));
}

TEST_F(ConvertBitcastTest, IncreasePrecisionSize) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBitcast);
  // Example: 2x2 (32-bit) -> 2x2x4 (8-bit)
  model.AddInput(kTfLiteInt32, {2, 2});
  model.AddOutput(kTfLiteInt8, {2, 2, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 2);
  const ::ml_drift::ir::IrOp* reshape_op = ir_model->op(0);
  EXPECT_EQ(reshape_op->name, "reshape");

  const auto* attr =
      std::any_cast<::ml_drift::ReshapeAttributes>(&reshape_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->new_shape, ::ml_drift::BHWC(2, 1, 2, 1));

  const ::ml_drift::ir::IrOp* bitcast_op = ir_model->op(1);
  EXPECT_EQ(bitcast_op->name, "bitcast");
}

TEST_F(ConvertBitcastTest, DecreasePrecisionSize_NoReshape) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBitcast);
  // Example: 2x1x4 (8-bit) -> 2x1 (32-bit)
  // src mapped to 1x2x1x4, dst mapped to 1x2x1x1. Interim: 1x2x1x1. Match!
  model.AddInput(kTfLiteInt8, {2, 1, 4});
  model.AddOutput(kTfLiteInt32, {2, 1});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "bitcast");
}

TEST_F(ConvertBitcastTest, IncreasePrecisionSize_NoReshape) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBitcast);
  // Example: 2x1 (32-bit) -> 2x1x4 (8-bit)
  // src mapped to 1x2x1x1, dst mapped to 1x2x1x4. Interim: 1x2x1x1. Match!
  model.AddInput(kTfLiteInt32, {2, 1});
  model.AddOutput(kTfLiteInt8, {2, 1, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "bitcast");
}

}  // namespace
}  // namespace litert::ml_drift::ir
