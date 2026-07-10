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
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
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

class ConvertLayerNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertLayerNormTest, BasicLayerNorm) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinStablehloComposite,
                                   /*version=*/1);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});  // input

  std::vector<float> scale_data(4, 1.0f);
  std::vector<uint8_t> scale_bytes(scale_data.size() * sizeof(float));
  std::memcpy(scale_bytes.data(), scale_data.data(), scale_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {4}, scale_bytes);  // scale

  std::vector<float> bias_data(4, 0.0f);
  std::vector<uint8_t> bias_bytes(bias_data.size() * sizeof(float));
  std::memcpy(bias_bytes.data(), bias_data.data(), bias_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {4}, bias_bytes);  // bias

  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteStablehloCompositeParams* params =
      reinterpret_cast<TfLiteStablehloCompositeParams*>(
          calloc(1, sizeof(TfLiteStablehloCompositeParams)));
  params->name = "odml.group_norm";  // LayerNorm uses this name in support.cc

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Float("epsilon", 1e-5f);
    // Notice we do NOT include "_TENSOR_V1_reduction_axes" here, this tells
    // support.cc and ir_model_builder that it's LayerNorm
  });
  fbb.Finish();
  const std::vector<uint8_t>& buffer = fbb.GetBuffer();

  std::vector<uint8_t> attr_buffer(buffer.begin(), buffer.end());
  params->attributes = attr_buffer.data();
  params->attributes_size = attr_buffer.size();

  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  EXPECT_EQ(ir_model->inputs().size(), 1);
  EXPECT_EQ(ir_model->outputs().size(), 1);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* layer_norm_op = ir_model->op(0);
  EXPECT_EQ(layer_norm_op->name, "layer_norm");

  const ::ml_drift::LayerNormAttributes* attr =
      std::any_cast<::ml_drift::LayerNormAttributes>(&layer_norm_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_FLOAT_EQ(attr->epsilon, 1e-5f);
  EXPECT_TRUE(attr->scale.has_value());
  EXPECT_TRUE(attr->bias.has_value());

  // Sanity check inference.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

}  // namespace
}  // namespace litert::ml_drift::ir
