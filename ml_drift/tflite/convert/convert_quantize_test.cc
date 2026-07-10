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
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertQuantizeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        CreateStubDelegate(), DeleteStubDelegate);
  }

  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_ = {
      nullptr, DeleteStubDelegate};
};

TEST_F(ConvertQuantizeTest, BasicInt8) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinQuantize);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});

  model.AddOutput(kTfLiteInt8, {1, 2, 3, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "quantize_and_dequantize");

  const auto* attr =
      std::any_cast<::ml_drift::QuantizeAndDequantizeAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_FLOAT_EQ(attr->scale, 1.0f);
  EXPECT_FLOAT_EQ(attr->min, 1.0f * (-128.0f - 0.0f));
  EXPECT_FLOAT_EQ(attr->max, 1.0f * (127.0f - 0.0f));
}

TEST_F(ConvertQuantizeTest, BasicInt4) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinQuantize);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});

  model.AddOutput(kTfLiteInt4, {1, 2, 3, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "quantize_and_dequantize");

  const auto* attr =
      std::any_cast<::ml_drift::QuantizeAndDequantizeAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_FLOAT_EQ(attr->scale, 1.0f);
  EXPECT_FLOAT_EQ(attr->min, 1.0f * (-8.0f - 0.0f));
  EXPECT_FLOAT_EQ(attr->max, 1.0f * (7.0f - 0.0f));
}

TEST_F(ConvertQuantizeTest, BasicInt2) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinQuantize);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});

  model.AddOutput(kTfLiteInt2, {1, 2, 3, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "quantize_and_dequantize");

  const auto* attr =
      std::any_cast<::ml_drift::QuantizeAndDequantizeAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_FLOAT_EQ(attr->scale, 1.0f);
  EXPECT_FLOAT_EQ(attr->min, 1.0f * (-2.0f - 0.0f));
  EXPECT_FLOAT_EQ(attr->max, 1.0f * (1.0f - 0.0f));
}

TEST_F(ConvertQuantizeTest, BasicUInt8) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinQuantize);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});

  model.AddOutput(kTfLiteUInt8, {1, 2, 3, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "quantize_and_dequantize");

  const auto* attr =
      std::any_cast<::ml_drift::QuantizeAndDequantizeAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_FLOAT_EQ(attr->scale, 1.0f);
  EXPECT_FLOAT_EQ(attr->min, 1.0f * (0.0f - 0.0f));
  EXPECT_FLOAT_EQ(attr->max, 1.0f * (255.0f - 0.0f));
}

TEST_F(ConvertQuantizeTest, BasicUInt4) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinQuantize);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});

  model.AddOutput(kTfLiteUInt4, {1, 2, 3, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "quantize_and_dequantize");

  const auto* attr =
      std::any_cast<::ml_drift::QuantizeAndDequantizeAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_FLOAT_EQ(attr->scale, 1.0f);
  EXPECT_FLOAT_EQ(attr->min, 1.0f * (0.0f - 0.0f));
  EXPECT_FLOAT_EQ(attr->max, 1.0f * (15.0f - 0.0f));
}

}  // namespace
}  // namespace litert::ml_drift::ir
