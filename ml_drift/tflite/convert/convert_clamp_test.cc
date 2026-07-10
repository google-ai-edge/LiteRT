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

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertClampTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertClampTest, ClampOperation) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinStablehloClamp, /*version=*/1);
  model.AddInput(GetParam(), {1, 2, 3, 4});   // min
  model.AddInput(GetParam(), {1, 2, 3, 4});   // operand
  model.AddInput(GetParam(), {1, 2, 3, 4});   // max
  model.AddOutput(GetParam(), {1, 2, 3, 4});  // output

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // We expect TWO ops: MAXIMUM and MINIMUM
  ASSERT_THAT(ir_model->ops(), ::testing::SizeIs(2));

  const ::ml_drift::ir::IrOp* max_op = ir_model->op(0);
  EXPECT_EQ(max_op->name, "maximum");

  const ::ml_drift::ir::IrOp* min_op = ir_model->op(1);
  EXPECT_EQ(min_op->name, "minimum");
}

INSTANTIATE_TEST_SUITE_P(ConvertClampTest, ConvertClampTest,
                         ::testing::Values(
                             // clang-format off
                             // go/keep-sorted start numeric=yes
                             kTfLiteBFloat16,
                             kTfLiteFloat16,
                             kTfLiteFloat32,
                             kTfLiteInt8,
                             kTfLiteInt16,
                             kTfLiteInt32,
                             kTfLiteUInt8,
                             kTfLiteUInt16,
                             kTfLiteUInt32
                             // go/keep-sorted end
                             // clang-format on
                             ));

}  // namespace
}  // namespace litert::ml_drift::ir
