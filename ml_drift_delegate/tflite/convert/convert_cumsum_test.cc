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
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertCumsumTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertCumsumTest, CumsumOperation4D) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinCumsum, /*version=*/1);

  model.AddInput(GetParam(), {1, 2, 3, 4});  // input

  std::vector<uint8_t> axis_data(sizeof(int32_t));
  int32_t axis = 3;  // Channel axis for 4D
  std::memcpy(axis_data.data(), &axis, sizeof(int32_t));
  model.AddConstInput(kTfLiteInt32, {1}, axis_data);  // axis

  model.AddOutput(GetParam(), {1, 2, 3, 4});  // output

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* cumsum_op = ir_model->op(0);
  EXPECT_EQ(cumsum_op->name, "cumsum");

  const ::ml_drift::CumsumAttributes* attr =
      std::any_cast<::ml_drift::CumsumAttributes>(&cumsum_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->axis, ::ml_drift::Axis::CHANNELS);
}

TEST_P(ConvertCumsumTest, CumsumOperationNegativeAxis) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinCumsum, /*version=*/1);

  model.AddInput(GetParam(), {1, 2, 3, 4});  // input

  std::vector<uint8_t> axis_data(sizeof(int32_t));
  int32_t axis = -1;  // Innermost axis
  std::memcpy(axis_data.data(), &axis, sizeof(int32_t));
  model.AddConstInput(kTfLiteInt32, {1}, axis_data);  // axis

  model.AddOutput(GetParam(), {1, 2, 3, 4});  // output

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* cumsum_op = ir_model->op(0);
  EXPECT_EQ(cumsum_op->name, "cumsum");

  const ::ml_drift::CumsumAttributes* attr =
      std::any_cast<::ml_drift::CumsumAttributes>(&cumsum_op->attr);
  ASSERT_TRUE(attr);
  // Rank 4, axis -1 resolves to 3 (CHANNELS)
  EXPECT_EQ(attr->axis, ::ml_drift::Axis::CHANNELS);
}

TEST_P(ConvertCumsumTest, CumsumOperation2D) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinCumsum, /*version=*/1);

  model.AddInput(GetParam(), {2, 3});  // input

  std::vector<uint8_t> axis_data(sizeof(int32_t));
  int32_t axis = 1;  // Inner axis for 2D
  std::memcpy(axis_data.data(), &axis, sizeof(int32_t));
  model.AddConstInput(kTfLiteInt32, {1}, axis_data);  // axis

  model.AddOutput(GetParam(), {2, 3});  // output

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* cumsum_op = ir_model->op(0);
  EXPECT_EQ(cumsum_op->name, "cumsum");

  const ::ml_drift::CumsumAttributes* attr =
      std::any_cast<::ml_drift::CumsumAttributes>(&cumsum_op->attr);
  ASSERT_TRUE(attr);
  // Rank 2, axis 1 maps to WIDTH in ExtractAxisFromIndex
  EXPECT_EQ(attr->axis, ::ml_drift::Axis::CHANNELS);
}

TEST_P(ConvertCumsumTest, CumsumOperation3D) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinCumsum, /*version=*/1);

  model.AddInput(GetParam(), {2, 3, 4});  // input

  std::vector<uint8_t> axis_data(sizeof(int32_t));
  int32_t axis = 2;  // Inner axis for 3D
  std::memcpy(axis_data.data(), &axis, sizeof(int32_t));
  model.AddConstInput(kTfLiteInt32, {1}, axis_data);  // axis

  model.AddOutput(GetParam(), {2, 3, 4});  // output

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* cumsum_op = ir_model->op(0);
  EXPECT_EQ(cumsum_op->name, "cumsum");

  const ::ml_drift::CumsumAttributes* attr =
      std::any_cast<::ml_drift::CumsumAttributes>(&cumsum_op->attr);
  ASSERT_TRUE(attr);
  // Rank 3, axis 2 maps to CHANNELS in ExtractAxisFromIndex
  EXPECT_EQ(attr->axis, ::ml_drift::Axis::CHANNELS);
}

INSTANTIATE_TEST_SUITE_P(ConvertCumsumTest, ConvertCumsumTest,
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
