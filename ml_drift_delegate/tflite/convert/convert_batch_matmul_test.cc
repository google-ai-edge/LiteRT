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
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::SizeIs;

class ConvertBatchMatMulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        CreateStubDelegate(), DeleteStubDelegate);
  }

  const ::ml_drift::ir::IrModel* GetIrModelFromBuilder(
      SingleOpInterpreterBuilder& model_builder) {
    interpreter_ = model_builder.Build();
    if (!interpreter_) return nullptr;
    if (interpreter_->ModifyGraphWithDelegate(delegate_.get()) != kTfLiteOk) {
      return nullptr;
    }
    return GetIrModel(delegate_.get());
  }

  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_ = {
      nullptr, [](TfLiteDelegate*) {}};
  std::unique_ptr<::tflite::Interpreter> interpreter_;
};

TEST_F(ConvertBatchMatMulTest, ConvertToFullyConnected) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {1, 4});  // Input 0

  // Input 1 is constant and 2D: shape {4, 2}
  std::vector<float> weights_data(4 * 2, 1.0f);
  std::vector<uint8_t> weights_bytes(weights_data.size() * sizeof(float));
  std::memcpy(weights_bytes.data(), weights_data.data(), weights_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {4, 2}, weights_bytes);

  model.AddOutput(kTfLiteFloat32, {1, 2});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, ToString(::ml_drift::OperationType::FULLY_CONNECTED));
  EXPECT_THAT(op->inputs,
              SizeIs(1));  // FullyConnected expects 1 input in this system

  const ::ml_drift::FullyConnectedAttributes* attr =
      std::any_cast<::ml_drift::FullyConnectedAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->weights.shape.o, 2);
  EXPECT_EQ(attr->weights.shape.i, 4);
}

TEST_F(ConvertBatchMatMulTest, ConvertToBatchedMatMul) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {1, 2, 3});  // Input 0
  model.AddInput(kTfLiteFloat32, {1, 3, 4});  // Input 1
  model.AddOutput(kTfLiteFloat32, {1, 2, 4});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, ToString(::ml_drift::OperationType::BATCHED_MATMUL));
  EXPECT_THAT(op->inputs, SizeIs(2));
}

TEST_F(ConvertBatchMatMulTest, ConvertToBatchedMatMul5D) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {1, 64, 2, 49, 32});  // Input 0
  model.AddInput(kTfLiteFloat32, {1, 64, 2, 32, 49});  // Input 1
  model.AddOutput(kTfLiteFloat32, {1, 64, 2, 49, 49});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // We expect 4 operations: Reshape (left), Reshape (right), BatchedMatMul,
  // Reshape (output)
  ASSERT_THAT(ir_model->ops(), SizeIs(4));

  EXPECT_EQ(ir_model->op(0)->name,
            ToString(::ml_drift::OperationType::RESHAPE));
  EXPECT_EQ(ir_model->op(1)->name,
            ToString(::ml_drift::OperationType::RESHAPE));
  EXPECT_EQ(ir_model->op(2)->name,
            ToString(::ml_drift::OperationType::BATCHED_MATMUL));
  EXPECT_EQ(ir_model->op(3)->name,
            ToString(::ml_drift::OperationType::RESHAPE));

  // Check shapes after reshape
  const ::ml_drift::ir::IrOp* reshape_left_op = ir_model->op(0);
  ASSERT_THAT(reshape_left_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* reshaped_left_tensor =
      ir_model->tensor(reshape_left_op->outputs[0]);
  EXPECT_EQ(reshaped_left_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 128, 49, 1, 32));

  const ::ml_drift::ir::IrOp* reshape_right_op = ir_model->op(1);
  ASSERT_THAT(reshape_right_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* reshaped_right_tensor =
      ir_model->tensor(reshape_right_op->outputs[0]);
  EXPECT_EQ(reshaped_right_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 128, 32, 1, 49));

  const ::ml_drift::ir::IrOp* bmm_op = ir_model->op(2);
  ASSERT_THAT(bmm_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* bmm_output_tensor =
      ir_model->tensor(bmm_op->outputs[0]);
  EXPECT_EQ(bmm_output_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 128, 49, 1, 49));

  const ::ml_drift::ir::IrOp* reshape_result_op = ir_model->op(3);
  ASSERT_THAT(reshape_result_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* final_output_tensor =
      ir_model->tensor(reshape_result_op->outputs[0]);
  EXPECT_EQ(final_output_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 64, 2, 49, 49));
}

TEST_F(ConvertBatchMatMulTest, ConvertToBatchedMatMul4DReshape) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {2, 4, 128, 32});  // Input 0
  model.AddInput(kTfLiteFloat32, {2, 4, 32, 64});   // Input 1
  model.AddOutput(kTfLiteFloat32, {2, 4, 128, 64});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // We expect 4 operations: Reshape (left), Reshape (right), BatchedMatMul,
  // Reshape (output)
  ASSERT_THAT(ir_model->ops(), SizeIs(4));

  EXPECT_EQ(ir_model->op(0)->name,
            ToString(::ml_drift::OperationType::RESHAPE));
  EXPECT_EQ(ir_model->op(1)->name,
            ToString(::ml_drift::OperationType::RESHAPE));
  EXPECT_EQ(ir_model->op(2)->name,
            ToString(::ml_drift::OperationType::BATCHED_MATMUL));
  EXPECT_EQ(ir_model->op(3)->name,
            ToString(::ml_drift::OperationType::RESHAPE));

  // Check shapes after reshape
  const ::ml_drift::ir::IrOp* reshape_left_op = ir_model->op(0);
  ASSERT_THAT(reshape_left_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* reshaped_left_tensor =
      ir_model->tensor(reshape_left_op->outputs[0]);
  EXPECT_EQ(reshaped_left_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 8, 128, 1, 32));

  const ::ml_drift::ir::IrOp* reshape_right_op = ir_model->op(1);
  ASSERT_THAT(reshape_right_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* reshaped_right_tensor =
      ir_model->tensor(reshape_right_op->outputs[0]);
  EXPECT_EQ(reshaped_right_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 8, 32, 1, 64));

  const ::ml_drift::ir::IrOp* bmm_op = ir_model->op(2);
  ASSERT_THAT(bmm_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* bmm_output_tensor =
      ir_model->tensor(bmm_op->outputs[0]);
  EXPECT_EQ(bmm_output_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 8, 128, 1, 64));

  const ::ml_drift::ir::IrOp* reshape_result_op = ir_model->op(3);
  ASSERT_THAT(reshape_result_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* final_output_tensor =
      ir_model->tensor(reshape_result_op->outputs[0]);
  EXPECT_EQ(final_output_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(2, 4, 128, 1, 64));
}

}  // namespace
}  // namespace litert::ml_drift::ir
