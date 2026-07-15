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
#include <string>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertSelectTest : public ::testing::Test {
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

class ConvertSelectParameterizedTest
    : public ConvertSelectTest,
      public ::testing::WithParamInterface<TfLiteType> {};

TEST_P(ConvertSelectParameterizedTest, SelectV2Basic) {
  TfLiteType data_type = GetParam();
  SingleOpInterpreterBuilder model(kTfLiteBuiltinSelectV2);
  model.AddInput(kTfLiteBool, {1, 2, 3, 4});  // Cond
  model.AddInput(data_type, {1, 2, 3, 4});    // If
  model.AddInput(data_type, {1, 2, 3, 4});    // Else
  model.AddOutput(data_type, {1, 2, 3, 4});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "select_v2");
  EXPECT_EQ(op->inputs.size(), 3);
  EXPECT_EQ(op->outputs.size(), 1);
}

TEST_P(ConvertSelectParameterizedTest, SelectBasic) {
  TfLiteType data_type = GetParam();
  SingleOpInterpreterBuilder model(kTfLiteBuiltinSelect);
  model.AddInput(kTfLiteBool, {1, 2, 3, 4});  // Cond
  model.AddInput(data_type, {1, 2, 3, 4});    // If
  model.AddInput(data_type, {1, 2, 3, 4});    // Else
  model.AddOutput(data_type, {1, 2, 3, 4});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "select_v2");
  EXPECT_EQ(op->inputs.size(), 3);
}

INSTANTIATE_TEST_SUITE_P(ConvertSelect, ConvertSelectParameterizedTest,
                         ::testing::Values(kTfLiteInt8, kTfLiteInt16,
                                           kTfLiteInt32, kTfLiteUInt8,
                                           kTfLiteBool, kTfLiteBFloat16,
                                           kTfLiteFloat32, kTfLiteFloat16));

TEST_F(ConvertSelectTest, SelectV2WithConstants) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinSelectV2);
  model.AddInput(kTfLiteBool, {1, 2, 3, 4});  // Cond

  std::vector<float> then_data(24, 1.0f);
  std::vector<uint8_t> then_bytes(then_data.size() * sizeof(float));
  std::memcpy(then_bytes.data(), then_data.data(), then_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {1, 2, 3, 4}, then_bytes);  // If

  std::vector<float> else_data(24, 2.0f);
  std::vector<uint8_t> else_bytes(else_data.size() * sizeof(float));
  std::memcpy(else_bytes.data(), else_data.data(), else_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {1, 2, 3, 4}, else_bytes);  // Else

  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // Expect 3 ops: 2 constant ops and the select op!
  ASSERT_EQ(ir_model->ops().size(), 3);

  // Find the select op
  const ::ml_drift::ir::IrOp* select_op = nullptr;
  for (const auto& op : ir_model->ops()) {
    if (op->name == "select_v2") {
      select_op = op.get();
      break;
    }
  }
  ASSERT_TRUE(select_op);
  EXPECT_EQ(select_op->inputs.size(), 3);
}

}  // namespace
}  // namespace litert::ml_drift::ir
