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
#include <memory>
#include <string>
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
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertEmbeddingLookupTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
    dtype_ = GetParam();
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
  TfLiteType dtype_;
};

TEST_P(ConvertEmbeddingLookupTest, Parameterized) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinEmbeddingLookup,
                                   /*version=*/1);
  const int lookups = 3;
  const int embedding_dim = 4;
  const int vocab_size = 5;

  model.AddInput(kTfLiteInt32, {lookups});                    // indices
  model.AddInput(dtype_, {vocab_size, embedding_dim});        // weights
  model.AddOutput(kTfLiteFloat32, {lookups, embedding_dim});  // output

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);

  // Set weights as constant
  TfLiteTensor* weights = interpreter->tensor(interpreter->inputs()[1]);
  weights->allocation_type = kTfLiteMmapRo;
  std::vector<uint8_t> weights_data(weights->bytes, 1);
  weights->data.raw = reinterpret_cast<char*>(weights_data.data());

  // Add mock quantization params for quantized types
  if (dtype_ == kTfLiteInt2 || dtype_ == kTfLiteInt4 || dtype_ == kTfLiteInt8) {
    if (weights->quantization.type == kTfLiteAffineQuantization &&
        weights->quantization.params) {
      auto* q = reinterpret_cast<TfLiteAffineQuantization*>(
          weights->quantization.params);
      q->scale->data[0] = 1.0f;
      q->zero_point->data[0] = 0;
    } else {
      TfLiteAffineQuantization* quant_params =
          reinterpret_cast<TfLiteAffineQuantization*>(
              malloc(sizeof(TfLiteAffineQuantization)));
      quant_params->scale = TfLiteFloatArrayCreate(1);
      quant_params->scale->data[0] = 1.0f;
      quant_params->zero_point = TfLiteIntArrayCreate(1);
      quant_params->zero_point->data[0] = 0;
      quant_params->quantized_dimension = 0;
      weights->quantization.type = kTfLiteAffineQuantization;
      weights->quantization.params = quant_params;
    }
  }

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  EXPECT_EQ(ir_model->inputs().size(),
            1);  // 1 input (indices), weights are in attr
  EXPECT_EQ(ir_model->outputs().size(), 1);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "embedding_lookup");

  const auto* attr =
      std::any_cast<::ml_drift::EmbeddingLookupAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->original_weights_shape,
            ::ml_drift::OHWI(vocab_size, 1, 1, embedding_dim));

  if (dtype_ == kTfLiteInt2) {
    EXPECT_EQ(attr->weights_type,
              ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt2);
    EXPECT_TRUE(
        (std::holds_alternative<
            ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::UINT8>>(
            attr->weights)));
  } else if (dtype_ == kTfLiteInt4) {
    EXPECT_EQ(attr->weights_type,
              ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt4);
    EXPECT_TRUE(
        (std::holds_alternative<
            ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::UINT8>>(
            attr->weights)));
  } else if (dtype_ == kTfLiteInt8) {
    EXPECT_EQ(attr->weights_type,
              ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt8);
    EXPECT_TRUE(
        (std::holds_alternative<
            ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>>(
            attr->weights)));
  } else if (dtype_ == kTfLiteFloat32) {
    EXPECT_EQ(attr->weights_type,
              ::ml_drift::EmbeddingLookupAttributes::WeightsType::kFloat32);
    EXPECT_TRUE(
        (std::holds_alternative<::ml_drift::Tensor<
             ::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>(attr->weights)));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ConvertEmbeddingLookupTest, ConvertEmbeddingLookupTest,
    ::testing::Values(kTfLiteFloat32, kTfLiteInt8, kTfLiteInt4, kTfLiteInt2),
    [](const ::testing::TestParamInfo<ConvertEmbeddingLookupTest::ParamType>&
           info) {
      switch (info.param) {
        case kTfLiteFloat32:
          return "float32";
        case kTfLiteInt2:
          return "int2";
        case kTfLiteInt4:
          return "int4";
        case kTfLiteInt8:
          return "int8";
        default:
          return "unknown";
      }
    });

}  // namespace
}  // namespace litert::ml_drift::ir
