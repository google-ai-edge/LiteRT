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
#include <cstdlib>
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
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertSplitTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertSplitTest, SplitAlongChannel) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinSplit, /*version=*/1);
  // Split has inputs: axis, input
  std::vector<uint8_t> axis_data(sizeof(int32_t));
  int32_t axis = 3;  // Channel axis for 4D
  std::memcpy(axis_data.data(), &axis, sizeof(int32_t));
  model.AddConstInput(kTfLiteInt32, {1}, axis_data);  // axis
  model.AddInput(GetParam(), {1, 2, 3, 4});           // input
  model.AddOutput(GetParam(), {1, 2, 3, 2});          // output 0
  model.AddOutput(GetParam(), {1, 2, 3, 2});          // output 1

  TfLiteSplitParams* params = static_cast<TfLiteSplitParams*>(
      calloc(1, sizeof(TfLiteSplitParams)));
  params->num_splits = 2;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* split_op = ir_model->op(0);
  EXPECT_EQ(split_op->name, "split");
  EXPECT_EQ(split_op->inputs.size(), 1);  // only input tensor
  EXPECT_EQ(split_op->outputs.size(), 2);

  const ::ml_drift::SplitAttributes* attr =
      std::any_cast<::ml_drift::SplitAttributes>(&split_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->axis, ::ml_drift::Axis::CHANNELS);
}

TEST_P(ConvertSplitTest, SplitNumSplitsOne) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinSplit, /*version=*/1);
  std::vector<uint8_t> axis_data(sizeof(int32_t));
  int32_t axis = 3;
  std::memcpy(axis_data.data(), &axis, sizeof(int32_t));
  model.AddConstInput(kTfLiteInt32, {1}, axis_data);  // axis
  model.AddInput(GetParam(), {1, 2, 3, 4});           // input
  model.AddOutput(GetParam(), {1, 2, 3, 4});          // output

  TfLiteSplitParams* params = static_cast<TfLiteSplitParams*>(
      calloc(1, sizeof(TfLiteSplitParams)));
  params->num_splits = 1;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* reshape_op = ir_model->op(0);
  EXPECT_EQ(reshape_op->name, "reshape");

  ASSERT_EQ(reshape_op->inputs.size(), 1);
  EXPECT_TRUE(ir_model->IsGraphInput(reshape_op->inputs[0]));

  const ::ml_drift::ReshapeAttributes* attr =
      std::any_cast<::ml_drift::ReshapeAttributes>(&reshape_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->new_shape, ::ml_drift::BHWC(1, 2, 3, 4));
}

class ConvertSplitVTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertSplitVTest, SplitVAlongChannel) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinSplitV, /*version=*/1);
  // SplitV has inputs: input, size_splits, axis
  model.AddInput(GetParam(), {1, 2, 3, 4});  // input

  std::vector<uint8_t> size_splits_data(2 * sizeof(int32_t));
  int32_t sizes[2] = {2, 2};
  std::memcpy(size_splits_data.data(), sizes, 2 * sizeof(int32_t));
  model.AddConstInput(kTfLiteInt32, {2}, size_splits_data);  // size_splits

  std::vector<uint8_t> axis_data(sizeof(int32_t));
  int32_t axis = 3;
  std::memcpy(axis_data.data(), &axis, sizeof(int32_t));
  model.AddConstInput(kTfLiteInt32, {1}, axis_data);  // axis

  model.AddOutput(GetParam(), {1, 2, 3, 2});  // output 0
  model.AddOutput(GetParam(), {1, 2, 3, 2});  // output 1

  TfLiteSplitVParams* params = static_cast<TfLiteSplitVParams*>(
      calloc(1, sizeof(TfLiteSplitVParams)));
  params->num_splits = 2;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  const ::ml_drift::ir::IrOp* split_op = nullptr;
  for (const auto& op : ir_model->ops()) {
    if (op->name == "split") {
      split_op = op.get();
      break;
    }
  }
  ASSERT_TRUE(split_op);
  EXPECT_EQ(split_op->inputs.size(), 1);
  EXPECT_EQ(split_op->outputs.size(), 2);

  const ::ml_drift::SplitAttributes* attr =
      std::any_cast<::ml_drift::SplitAttributes>(&split_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->axis, ::ml_drift::Axis::CHANNELS);
}

INSTANTIATE_TEST_SUITE_P(ConvertSplitTest, ConvertSplitTest,
                         ::testing::Values(kTfLiteFloat32, kTfLiteFloat16));

INSTANTIATE_TEST_SUITE_P(ConvertSplitVTest, ConvertSplitVTest,
                         ::testing::Values(kTfLiteFloat32, kTfLiteFloat16));

}  // namespace
}  // namespace litert::ml_drift::ir
