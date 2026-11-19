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

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertScatterNdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertScatterNdTest, ConvertsToOneOpWithTwoRuntimeInputs) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinScatterNd, /*version=*/1);

  model.AddInput(kTfLiteInt32, {3, 3});          // indices
  model.AddInput(kTfLiteFloat32, {3, 5});           // updates

  const std::vector<int32_t> shape = {4, 2, 3, 5};
  std::vector<uint8_t> shape_data(shape.size() * sizeof(int32_t));
  std::memcpy(shape_data.data(), shape.data(), shape_data.size());
  model.AddConstInput(kTfLiteInt32, {4}, shape_data);  // shape

  model.AddOutput(kTfLiteFloat32, {4, 2, 3, 5});  // output

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "scatter_nd");
  // The constant shape input is not an input of the IR op.
  EXPECT_EQ(op->inputs.size(), 2);
  EXPECT_EQ(op->outputs.size(), 1);
}

// A constant updates input is produced by a CONSTANT op, which has to precede
// the scatter_nd op.
TEST_F(ConvertScatterNdTest, ConvertsConstantUpdatesToConstantOp) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinScatterNd, /*version=*/1);

  model.AddInput(kTfLiteInt32, {3, 3});  // indices

  const std::vector<float> updates(3 * 5, 1.0f);
  std::vector<uint8_t> updates_data(updates.size() * sizeof(float));
  std::memcpy(updates_data.data(), updates.data(), updates_data.size());
  model.AddConstInput(kTfLiteFloat32, {3, 5}, updates_data);  // updates

  const std::vector<int32_t> shape = {4, 2, 3, 5};
  std::vector<uint8_t> shape_data(shape.size() * sizeof(int32_t));
  std::memcpy(shape_data.data(), shape.data(), shape_data.size());
  model.AddConstInput(kTfLiteInt32, {4}, shape_data);  // shape

  model.AddOutput(kTfLiteFloat32, {4, 2, 3, 5});  // output

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  // Only the updates are materialized; the constant shape is dropped.
  ASSERT_EQ(ir_model->ops().size(), 2);

  const ::ml_drift::ir::IrOp* const_op = ir_model->op(0);
  const ::ml_drift::ir::IrOp* scatter_op = ir_model->op(1);
  EXPECT_EQ(const_op->name,
            ::ml_drift::ToString(::ml_drift::OperationType::kConstant));
  EXPECT_EQ(scatter_op->name, "scatter_nd");

  ASSERT_EQ(const_op->outputs.size(), 1);
  ASSERT_EQ(scatter_op->inputs.size(), 2);
  // The scatter_nd consumes the constant as its updates input.
  EXPECT_EQ(scatter_op->inputs[1], const_op->outputs[0]);
  EXPECT_NE(scatter_op->inputs[0], const_op->outputs[0]);
}

}  // namespace
}  // namespace litert::ml_drift::ir
