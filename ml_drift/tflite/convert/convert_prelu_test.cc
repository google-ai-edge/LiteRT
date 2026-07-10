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
#include <variant>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

TEST(ConvertPreluTest, LinearAlpha) {
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      CreateStubDelegate(), DeleteStubDelegate);
  SingleOpInterpreterBuilder model(kTfLiteBuiltinPrelu);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  std::vector<float> alpha_data = {0.1f, 0.2f, 0.3f, 0.4f};
  std::vector<uint8_t> alpha_bytes(alpha_data.size() * sizeof(float));
  std::memcpy(alpha_bytes.data(), alpha_data.data(), alpha_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {4}, alpha_bytes);
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "prelu");

  const auto* attr = std::any_cast<::ml_drift::PReLUAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* linear_alpha = std::get_if<
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>>(
      &attr->alpha);
  ASSERT_TRUE(linear_alpha);
  EXPECT_EQ(linear_alpha->shape.v, 4);
}

TEST(ConvertPreluTest, HwcAlpha) {
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      CreateStubDelegate(), DeleteStubDelegate);
  SingleOpInterpreterBuilder model(kTfLiteBuiltinPrelu);
  model.AddInput(kTfLiteFloat32, {1, 2, 2, 2});
  std::vector<float> alpha_data(8, 0.1f);
  std::vector<uint8_t> alpha_bytes(alpha_data.size() * sizeof(float));
  std::memcpy(alpha_bytes.data(), alpha_data.data(), alpha_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {1, 2, 2, 2}, alpha_bytes);
  model.AddOutput(kTfLiteFloat32, {1, 2, 2, 2});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "prelu");

  const auto* attr = std::any_cast<::ml_drift::PReLUAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* hwc_alpha = std::get_if<
      ::ml_drift::Tensor<::ml_drift::HWC, ::ml_drift::DataType::FLOAT32>>(
      &attr->alpha);
  ASSERT_TRUE(hwc_alpha);
  EXPECT_EQ(hwc_alpha->shape.h, 2);
  EXPECT_EQ(hwc_alpha->shape.w, 2);
  EXPECT_EQ(hwc_alpha->shape.c, 2);
}

}  // namespace
}  // namespace litert::ml_drift::ir
