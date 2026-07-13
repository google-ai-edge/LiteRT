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
#include <cstdlib>
#include <memory>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::ml_drift::Axis;
using ::ml_drift::SoftmaxAttributes;
using ::ml_drift::ir::IrModel;
using ::ml_drift::ir::IrOp;
using ::testing::NotNull;
using ::testing::SizeIs;

class ConvertSoftmaxTest : public testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_THAT(delegate_, NotNull());
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_ = nullptr;
};

TEST_P(ConvertSoftmaxTest, SoftmaxOperation) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinSoftmax, /*version=*/1);

  model.AddInput(GetParam(), {1, 2, 3, 4});   // input
  model.AddOutput(GetParam(), {1, 2, 3, 4});  // output

  TfLiteSoftmaxParams* params = reinterpret_cast<TfLiteSoftmaxParams*>(
      calloc(1, sizeof(TfLiteSoftmaxParams)));
  params->beta = 1.0f;
  model.SetParameters(params);

  std::unique_ptr<tflite::Interpreter> interpreter = model.Build();
  ASSERT_THAT(interpreter, NotNull());
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_THAT(ir_model, NotNull());
  ASSERT_THAT(ir_model->ops(), SizeIs(1));

  const IrOp* softmax_op = ir_model->op(0);
  ASSERT_THAT(softmax_op, NotNull());
  EXPECT_EQ(softmax_op->name, "softmax");

  const SoftmaxAttributes* attr =
      std::any_cast<SoftmaxAttributes>(&softmax_op->attr);
  ASSERT_THAT(attr, NotNull());
  EXPECT_EQ(attr->axis, Axis::CHANNELS);
}

TEST(ConvertSoftmaxTestNonParam, SoftmaxOperationWithCapping) {
  IrModelBuilderOptions options = {.enable_infinite_float_capping = true};
  TfLiteDelegate* delegate = CreateStubDelegate(options);
  ASSERT_THAT(delegate, NotNull());

  SingleOpInterpreterBuilder model(kTfLiteBuiltinSoftmax, /*version=*/1);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});   // input
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});  // output

  TfLiteSoftmaxParams* params = reinterpret_cast<TfLiteSoftmaxParams*>(
      calloc(1, sizeof(TfLiteSoftmaxParams)));
  params->beta = 1.0f;
  model.SetParameters(params);

  std::unique_ptr<tflite::Interpreter> interpreter = model.Build();
  ASSERT_THAT(interpreter, NotNull());
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  const IrModel* ir_model = GetIrModel(delegate);
  ASSERT_THAT(ir_model, NotNull());
  ASSERT_THAT(ir_model->ops(), SizeIs(3));

  const IrOp* max_op = ir_model->op(0);
  ASSERT_THAT(max_op, NotNull());
  EXPECT_EQ(max_op->name, "maximum");

  const IrOp* min_op = ir_model->op(1);
  ASSERT_THAT(min_op, NotNull());
  EXPECT_EQ(min_op->name, "minimum");

  const IrOp* softmax_op = ir_model->op(2);
  ASSERT_THAT(softmax_op, NotNull());
  EXPECT_EQ(softmax_op->name, "softmax");

  DeleteStubDelegate(delegate);
}

INSTANTIATE_TEST_SUITE_P(ConvertSoftmaxTest, ConvertSoftmaxTest,
                         testing::Values(kTfLiteFloat16, kTfLiteFloat32));

}  // namespace
}  // namespace litert::ml_drift::ir
