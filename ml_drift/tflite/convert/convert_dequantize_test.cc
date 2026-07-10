// Copyright 2026 The ML Drift Authors.
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
#include <memory>
#include <vector>

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

class ConvertDequantizeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        CreateStubDelegate(), DeleteStubDelegate);
  }

  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_ = {
      nullptr, DeleteStubDelegate};
};

struct DequantizeParam {
  TfLiteType type;
  float min;
  float max;
};

class ConvertDequantizeParamTest
    : public ConvertDequantizeTest,
      public ::testing::WithParamInterface<DequantizeParam> {};

TEST_P(ConvertDequantizeParamTest, Basic) {
  const DequantizeParam& param = GetParam();
  SingleOpInterpreterBuilder model(kTfLiteBuiltinDequantize);
  model.AddInput(param.type, {1, 2, 3, 4});

  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "quantize_and_dequantize");

  const auto* attr =
      std::any_cast<::ml_drift::QuantizeAndDequantizeAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_FLOAT_EQ(attr->scale, 1.0f);
  EXPECT_FLOAT_EQ(attr->min, 1.0f * (param.min - 0.0f));
  EXPECT_FLOAT_EQ(attr->max, 1.0f * (param.max - 0.0f));
}

INSTANTIATE_TEST_SUITE_P(
    SupportedTypes, ConvertDequantizeParamTest,
    ::testing::Values(DequantizeParam{kTfLiteUInt8, 0.0f, 255.0f},
                      DequantizeParam{kTfLiteInt8, -128.0f, 127.0f},
                      DequantizeParam{kTfLiteUInt4, 0.0f, 15.0f},
                      DequantizeParam{kTfLiteInt4, -8.0f, 7.0f},
                      DequantizeParam{kTfLiteInt2, -2.0f, 1.0f}));

TEST_F(ConvertDequantizeTest, BasicConstantInput) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinDequantize);
  std::vector<uint8_t> data = {1, 2, 3, 4};
  model.AddConstInput(kTfLiteInt8, {1, 1, 2, 2}, data);

  model.AddOutput(kTfLiteFloat32, {1, 1, 2, 2});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "const");

  const auto* attr =
      std::any_cast<::ml_drift::ConstTensorAttributes>(&op->attr);
  ASSERT_TRUE(attr);
}

}  // namespace
}  // namespace litert::ml_drift::ir
