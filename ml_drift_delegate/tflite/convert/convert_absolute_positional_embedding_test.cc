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
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertAbsolutePositionalEmbeddingTest
    : public ::testing::TestWithParam<TfLiteType> {};

TEST_P(ConvertAbsolutePositionalEmbeddingTest, Simple) {
  const TfLiteType dtype = GetParam();
  SingleOpInterpreterBuilder model(kTfLiteBuiltinCustom);
  model.SetCustomName("custom_call.absolute_positional_embedding");

  const int batch = 1;
  const int height = 1;
  const int width = 10;
  const int channels = 16;

  // Use 4D tensors for both to satisfy support library width requirement.
  model.AddInput(dtype, {batch, height, width, channels});  // src
  model.AddInput(dtype, {batch, height, width, channels});  // position
  model.AddOutput(dtype, {batch, height, width, channels});  // output

  TfLiteDelegate* delegate = CreateStubDelegate();
  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate);
  ASSERT_TRUE(ir_model);
  EXPECT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "positional_embedding");
  EXPECT_EQ(ir_model->inputs().size(), 2);
  EXPECT_EQ(ir_model->outputs().size(), 1);

  DeleteStubDelegate(delegate);
}

INSTANTIATE_TEST_SUITE_P(
    AllTypes, ConvertAbsolutePositionalEmbeddingTest,
    ::testing::Values(kTfLiteFloat16, kTfLiteFloat32, kTfLiteInt8, kTfLiteInt16,
                      kTfLiteInt32, kTfLiteUInt8, kTfLiteUInt16,
                      kTfLiteUInt32));

}  // namespace
}  // namespace litert::ml_drift::ir
