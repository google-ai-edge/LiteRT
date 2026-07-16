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

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertRoPETest : public ::testing::Test {};

TEST_F(ConvertRoPETest, TwoInputs) {
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      CreateStubDelegate(), DeleteStubDelegate);
  SingleOpInterpreterBuilder model(kTfLiteBuiltinCustom);
  model.SetCustomName("custom_call.rotary_positional_embedding");

  model.AddInput(kTfLiteFloat32, {1, 2, 2, 4});
  model.AddInput(kTfLiteFloat32, {1, 2, 2, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 2, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const auto* op = ir_model->op(0);
  EXPECT_EQ(op->name, "rope");
  EXPECT_EQ(op->inputs.size(), 2);
  EXPECT_EQ(op->outputs.size(), 1);
}

TEST_F(ConvertRoPETest, ThreeInputs) {
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      CreateStubDelegate(), DeleteStubDelegate);
  SingleOpInterpreterBuilder model(kTfLiteBuiltinCustom);
  model.SetCustomName("custom_call.rotary_positional_embedding");

  model.AddInput(kTfLiteFloat32, {1, 2, 2, 4});
  model.AddInput(kTfLiteFloat32, {1, 2, 2, 4});
  model.AddInput(kTfLiteFloat32, {1, 2, 2, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 2, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 2, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const auto* op = ir_model->op(0);
  EXPECT_EQ(op->name, "rope");
  EXPECT_EQ(op->inputs.size(), 3);
  EXPECT_EQ(op->outputs.size(), 2);
}

}  // namespace
}  // namespace litert::ml_drift::ir
