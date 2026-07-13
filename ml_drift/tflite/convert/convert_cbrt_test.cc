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
#include <variant>

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

TEST(ConvertCbrtTest, CbrtToPow) {
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      CreateStubDelegate(), DeleteStubDelegate);
  SingleOpInterpreterBuilder model(kTfLiteBuiltinStablehloCbrt);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "pow");
  EXPECT_EQ(op->inputs.size(), 1);
  EXPECT_EQ(op->outputs.size(), 1);

  const auto* attr =
      std::any_cast<::ml_drift::ElementwiseAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* scalar_val = std::get_if<::ml_drift::ScalarValue>(&attr->param);
  ASSERT_TRUE(scalar_val);
  const float* p = std::get_if<float>(scalar_val);
  ASSERT_TRUE(p);
  EXPECT_FLOAT_EQ(*p, 1.0f / 3.0f);
  EXPECT_FALSE(attr->runtime_tensor_is_second);
}

}  // namespace
}  // namespace litert::ml_drift::ir
