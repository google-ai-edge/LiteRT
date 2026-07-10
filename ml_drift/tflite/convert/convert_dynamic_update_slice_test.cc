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

#include <memory>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertDynamicUpdateSliceTest
    : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        CreateStubDelegate(), DeleteStubDelegate);
    ASSERT_TRUE(delegate_);
  }

  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_ = {
      nullptr, [](TfLiteDelegate*) {}};
};

TEST_P(ConvertDynamicUpdateSliceTest, DynamicUpdateSlice4D) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinDynamicUpdateSlice,
                                   /*version=*/1);
  model.AddInput(GetParam(), {1, 4, 4, 3});  // operand
  model.AddInput(GetParam(), {1, 2, 2, 3});  // update
  model.AddInput(kTfLiteInt32, {4});         // start_indices
  model.AddOutput(GetParam(), {1, 4, 4, 3});

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(ir_model->ops()[0]->id);
  EXPECT_EQ(op->name, "dynamic_update_slice");
  EXPECT_EQ(op->inputs.size(), 3);
  EXPECT_EQ(ir_model->tensor(op->inputs[0])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, 4, 4, 3));
  EXPECT_EQ(ir_model->tensor(op->inputs[1])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, 2, 2, 3));
  EXPECT_EQ(ir_model->tensor(op->inputs[2])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(4, 1, 1, 1));
}

TEST_P(ConvertDynamicUpdateSliceTest, DynamicUpdateSlice3D) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinDynamicUpdateSlice,
                                   /*version=*/1);
  model.AddInput(GetParam(), {1, 4, 3});  // operand
  model.AddInput(GetParam(), {1, 2, 3});  // update
  model.AddInput(kTfLiteInt32, {3});      // start_indices
  model.AddOutput(GetParam(), {1, 4, 3});

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  // Expect 1 operation because {1, 4, 3} maps to BHWC(1, 1, 4, 3) in both
  // ExtractTensorShape (left-align) and GetRightAlignedBHWC (right-align).
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* dus_op = ir_model->op(ir_model->ops()[0]->id);

  EXPECT_EQ(dus_op->name, "dynamic_update_slice");

  // Verify shapes fed into DYNAMIC_UPDATE_SLICE
  EXPECT_EQ(ir_model->tensor(dus_op->inputs[0])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, 1, 4, 3));
  EXPECT_EQ(ir_model->tensor(dus_op->inputs[1])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, 1, 2, 3));
  // start_indices remains standard 1D map: {3} -> BHWC(3, 1, 1, 1)
  EXPECT_EQ(ir_model->tensor(dus_op->inputs[2])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(3, 1, 1, 1));
}

TEST_P(ConvertDynamicUpdateSliceTest, DynamicUpdateSlice2D) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinDynamicUpdateSlice,
                                   /*version=*/1);
  model.AddInput(GetParam(), {1, 4});  // operand
  model.AddInput(GetParam(), {1, 2});  // update
  model.AddInput(kTfLiteInt32, {2});   // start_indices
  model.AddOutput(GetParam(), {1, 4});

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  // Expect 1 operation because {1, 4} maps to BHWC(1, 1, 1, 4) in both
  // ExtractTensorShape (left-align) and GetRightAlignedBHWC (right-align).
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* dus_op = ir_model->op(ir_model->ops()[0]->id);

  EXPECT_EQ(dus_op->name, "dynamic_update_slice");

  // Verify shapes fed into DYNAMIC_UPDATE_SLICE
  EXPECT_EQ(ir_model->tensor(dus_op->inputs[0])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, 1, 1, 4));
  EXPECT_EQ(ir_model->tensor(dus_op->inputs[1])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, 1, 1, 2));
  // start_indices remains standard 1D map: {2} -> BHWC(2, 1, 1, 1)
  EXPECT_EQ(ir_model->tensor(dus_op->inputs[2])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(2, 1, 1, 1));
}

INSTANTIATE_TEST_SUITE_P(ConvertDynamicUpdateSliceTest,
                         ConvertDynamicUpdateSliceTest,
                         ::testing::Values(kTfLiteFloat32, kTfLiteInt32));

}  // namespace
}  // namespace litert::ml_drift::ir
