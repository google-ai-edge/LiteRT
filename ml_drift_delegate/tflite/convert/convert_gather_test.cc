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
#include <tuple>
#include <vector>

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

class ConvertGatherTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(ConvertGatherTest, Basic) {
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      CreateStubDelegate(), DeleteStubDelegate);
  SingleOpInterpreterBuilder model(kTfLiteBuiltinGather);

  const bool is_input_constant = std::get<0>(GetParam());
  const bool is_indices_constant = std::get<1>(GetParam());

  if (is_input_constant) {
    std::vector<float> input_data(12, 1.0f);
    std::vector<uint8_t> input_bytes(input_data.size() * sizeof(float));
    std::memcpy(input_bytes.data(), input_data.data(), input_bytes.size());
    model.AddConstInput(kTfLiteFloat32, {1, 3, 4}, input_bytes);
  } else {
    model.AddInput(kTfLiteFloat32, {1, 3, 4});
  }

  std::vector<int> indices_shape_arr = {2, 1};
  std::vector<int32_t> indices_data = {0, 1};

  if (is_indices_constant) {
    std::vector<uint8_t> indices_bytes(indices_data.size() * sizeof(int32_t));
    std::memcpy(indices_bytes.data(), indices_data.data(),
                indices_bytes.size());
    model.AddConstInput(kTfLiteInt32, indices_shape_arr, indices_bytes);
  } else {
    model.AddInput(kTfLiteInt32, indices_shape_arr);
  }

  model.AddOutput(kTfLiteFloat32, {1, 2, 1, 4});

  TfLiteGatherParams* gather_params = reinterpret_cast<TfLiteGatherParams*>(
      calloc(1, sizeof(TfLiteGatherParams)));
  gather_params->axis = 1;  // H
  gather_params->batch_dims = 0;
  model.SetParameters(gather_params);

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "gather");
  EXPECT_EQ(op->inputs.size(), 2);
  EXPECT_EQ(op->outputs.size(), 1);

  const auto* attr = std::any_cast<::ml_drift::GatherAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->axis, ::ml_drift::Axis::WIDTH);
}

TEST_P(ConvertGatherTest, Indices1D_WithReshapeOp) {
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      CreateStubDelegate(), DeleteStubDelegate);
  SingleOpInterpreterBuilder model(kTfLiteBuiltinGather);

  const bool is_input_constant = std::get<0>(GetParam());
  const bool is_indices_constant = std::get<1>(GetParam());

  if (is_input_constant) {
    std::vector<float> input_data(12, 1.0f);
    std::vector<uint8_t> input_bytes(input_data.size() * sizeof(float));
    std::memcpy(input_bytes.data(), input_data.data(), input_bytes.size());
    model.AddConstInput(kTfLiteFloat32, {1, 3, 4}, input_bytes);
  } else {
    model.AddInput(kTfLiteFloat32, {1, 3, 4});
  }

  std::vector<int> indices_shape_arr = {2};
  std::vector<int32_t> indices_data = {0, 1};

  if (is_indices_constant) {
    std::vector<uint8_t> indices_bytes(indices_data.size() * sizeof(int32_t));
    std::memcpy(indices_bytes.data(), indices_data.data(),
                indices_bytes.size());
    model.AddConstInput(kTfLiteInt32, indices_shape_arr, indices_bytes);
  } else {
    model.AddInput(kTfLiteInt32, indices_shape_arr);
  }

  model.AddOutput(kTfLiteFloat32, {1, 2, 4});

  TfLiteGatherParams* gather_params = reinterpret_cast<TfLiteGatherParams*>(
      calloc(1, sizeof(TfLiteGatherParams)));
  gather_params->axis = 1;  // H
  gather_params->batch_dims = 0;
  model.SetParameters(gather_params);

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  int expected_ops = 1;
  if (!is_indices_constant) {
    expected_ops++;  // Reshape op
  }

  ASSERT_EQ(ir_model->ops().size(), expected_ops);

  if (!is_indices_constant) {
    EXPECT_EQ(ir_model->op(0)->name, "reshape");
  }
  const ::ml_drift::ir::IrOp* op = ir_model->op(is_indices_constant ? 0 : 1);
  EXPECT_EQ(op->name, "gather");
  EXPECT_EQ(op->inputs.size(), 2);
  EXPECT_EQ(op->outputs.size(), 1);

  const auto* attr = std::any_cast<::ml_drift::GatherAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->axis, ::ml_drift::Axis::WIDTH);
}

INSTANTIATE_TEST_SUITE_P(ConvertGatherTest, ConvertGatherTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
}  // namespace litert::ml_drift::ir
