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
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertOneHotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertOneHotTest, BasicOneHot) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinOneHot, /*version=*/1);
  model.AddInput(kTfLiteInt32, {1, 2, 3});  // input indices

  std::vector<int32_t> depth = {4};
  std::vector<uint8_t> depth_bytes(depth.size() * sizeof(int32_t));
  std::memcpy(depth_bytes.data(), depth.data(), depth_bytes.size());
  model.AddConstInput(kTfLiteInt32, {1}, depth_bytes);  // depth

  float on_value = 1.0f;
  std::vector<uint8_t> on_bytes(sizeof(float));
  std::memcpy(on_bytes.data(), &on_value, sizeof(float));
  model.AddConstInput(kTfLiteFloat32, {1}, on_bytes);  // on_value

  float off_value = 0.0f;
  std::vector<uint8_t> off_bytes(sizeof(float));
  std::memcpy(off_bytes.data(), &off_value, sizeof(float));
  model.AddConstInput(kTfLiteFloat32, {1}, off_bytes);  // off_value

  model.AddOutput(kTfLiteInt32, {1, 2, 3, 4});

  TfLiteOneHotParams* params = reinterpret_cast<TfLiteOneHotParams*>(
      calloc(1, sizeof(TfLiteOneHotParams)));
  params->axis = -1;
  model.SetParameters(params);

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  EXPECT_EQ(ir_model->inputs().size(), 1);
  EXPECT_EQ(ir_model->outputs().size(), 1);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* one_hot_op = ir_model->op(0);
  EXPECT_EQ(one_hot_op->name, "one_hot");

  const ::ml_drift::OneHotAttributes* attr =
      std::any_cast<::ml_drift::OneHotAttributes>(&one_hot_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->on_value, 1.0f);
  EXPECT_EQ(attr->off_value, 0.0f);

  // Sanity check inference.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

}  // namespace
}  // namespace litert::ml_drift::ir
