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
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertRmsNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertRmsNormTest, CompositeRmsNorm) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinStablehloComposite,
                                   /*version=*/1);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});  // input

  std::vector<float> scale_data(4, 1.0f);
  std::vector<uint8_t> scale_bytes(scale_data.size() * sizeof(float));
  std::memcpy(scale_bytes.data(), scale_data.data(), scale_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {4}, scale_bytes);  // scale

  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteStablehloCompositeParams* params =
      reinterpret_cast<TfLiteStablehloCompositeParams*>(
          calloc(1, sizeof(TfLiteStablehloCompositeParams)));
  params->name = "odml.rms_norm";

  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Float("epsilon", 1e-5f); });
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

  const ::ml_drift::ir::IrOp* rms_norm_op = ir_model->op(0);
  EXPECT_EQ(rms_norm_op->name, "rms_norm");

  const ::ml_drift::RmsNormAttributes* attr =
      std::any_cast<::ml_drift::RmsNormAttributes>(&rms_norm_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_FLOAT_EQ(attr->epsilon, 1e-5f);
  EXPECT_TRUE(attr->scale.has_value());

  // Sanity check inference.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

}  // namespace
}  // namespace litert::ml_drift::ir
