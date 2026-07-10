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
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertTransposeConvTest
    : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertTransposeConvTest, TransposeConvVariations) {
  const bool is_builtin = std::get<0>(GetParam());
  const bool is_dynamic_weights = std::get<1>(GetParam());
  const bool has_bias = std::get<2>(GetParam());

  int builtin_code =
      is_builtin ? kTfLiteBuiltinTransposeConv : kTfLiteBuiltinCustom;
  SingleOpInterpreterBuilder model(builtin_code, /*version=*/1);
  if (!is_builtin) {
    model.SetCustomName("Convolution2DTransposeBias");
  }

  std::vector<int32_t> output_shape = {1, 4, 4, 1};
  std::vector<uint8_t> os_bytes(output_shape.size() * sizeof(int32_t));
  std::memcpy(os_bytes.data(), output_shape.data(), os_bytes.size());

  std::vector<float> weights_data(16, 1.0f);
  std::vector<uint8_t> w_bytes(weights_data.size() * sizeof(float));
  std::memcpy(w_bytes.data(), weights_data.data(), w_bytes.size());

  std::vector<float> bias_data(1, 0.5f);
  std::vector<uint8_t> b_bytes(bias_data.size() * sizeof(float));
  std::memcpy(b_bytes.data(), bias_data.data(), b_bytes.size());

  if (is_builtin) {
    model.AddConstInput(kTfLiteInt32, {4},
                        os_bytes);  // output_shape (tensor 0)
  }

  if (is_dynamic_weights) {
    model.AddInput(
        kTfLiteFloat32,
        {1, 4, 4, 1});  // weights (tensor 1 for builtin, tensor 0 for custom)
  } else {
    model.AddConstInput(kTfLiteFloat32, {1, 4, 4, 1}, w_bytes);  // weights
  }

  model.AddInput(
      kTfLiteFloat32,
      {1, 4, 4, 1});  // input (tensor 2 for builtin, tensor 1 for custom)

  if (has_bias) {
    model.AddConstInput(
        kTfLiteFloat32, {1, 1},
        b_bytes);  // bias (tensor 3 for builtin, tensor 2 for custom)
  }

  model.AddOutput(kTfLiteFloat32, {1, 4, 4, 1});

  TfLiteTransposeConvParams* params =
      reinterpret_cast<TfLiteTransposeConvParams*>(
          calloc(1, sizeof(TfLiteTransposeConvParams)));
  params->padding = kTfLitePaddingSame;
  params->stride_width = 1;
  params->stride_height = 1;

  if (is_builtin) {
    model.SetParameters(params);
  } else {
    model.SetCustomData(params, sizeof(TfLiteTransposeConvParams));
  }

  std::unique_ptr<::tflite::Interpreter> interpreter;

  if (is_builtin) {
    if (has_bias) {
      interpreter = model.Build({0, 1, 2, 3});
    } else {
      interpreter = model.Build({0, 1, 2});
    }
  } else {
    // For custom, the expected input order for ML Drift is {input, weights,
    // bias}
    if (has_bias) {
      interpreter = model.Build({1, 0, 2});  // input(1), weights(0), bias(2)
    } else {
      interpreter = model.Build({1, 0});  // input(1), weights(0)
    }
  }

  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  int expected_inputs = 1;  // The main input
  if (is_dynamic_weights) expected_inputs++;
  EXPECT_EQ(ir_model->inputs().size(), expected_inputs);
  EXPECT_EQ(ir_model->outputs().size(), 1);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* transpose_conv_op = ir_model->op(0);
  EXPECT_EQ(transpose_conv_op->name, "convolution_transposed");

  const ::ml_drift::ConvolutionTransposedAttributes* attr =
      std::any_cast<::ml_drift::ConvolutionTransposedAttributes>(
          &transpose_conv_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->stride.w, 1);
  EXPECT_EQ(attr->stride.h, 1);

  if (is_dynamic_weights) {
    EXPECT_TRUE(attr->weights.data.empty());
  } else {
    EXPECT_FALSE(attr->weights.data.empty());
  }

  if (has_bias) {
    EXPECT_FALSE(attr->bias.data.empty());
  } else {
    EXPECT_TRUE(attr->bias.data.empty());
  }

  // Sanity check inference.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  if (is_dynamic_weights) {
    // Need to populate dynamic weights before invoke
    // For builtin, weights are tensor 1. For custom, weights are tensor 0.
    TfLiteTensor* weights_tensor = interpreter->tensor(is_builtin ? 1 : 0);
    std::memcpy(weights_tensor->data.f, weights_data.data(),
                weights_data.size() * sizeof(float));
  }

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

INSTANTIATE_TEST_SUITE_P(TransposeConvTests, ConvertTransposeConvTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
}  // namespace litert::ml_drift::ir
