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
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertPixelShuffleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertPixelShuffleTest, BasicPixelShuffle) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinCustom, /*version=*/1);
  model.SetCustomName("custom_call.pixel_shuffle");

  // TFLite pixel shuffle uses NCHW format
  model.AddInput(kTfLiteFloat32, {1, 9, 4, 4});  // input
  model.AddOutput(kTfLiteFloat32, {1, 1, 12, 12});

  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("num_groups", 3); });
  fbb.Finish();
  const std::vector<uint8_t>& buffer = fbb.GetBuffer();

  void* custom_data = malloc(buffer.size());
  std::memcpy(custom_data, buffer.data(), buffer.size());
  model.SetCustomData(custom_data, buffer.size());

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  EXPECT_EQ(ir_model->inputs().size(), 1);
  EXPECT_EQ(ir_model->outputs().size(), 1);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "depth_to_space");

  const ::ml_drift::SpaceToDepthAttributes* attr =
      std::any_cast<::ml_drift::SpaceToDepthAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->block_size, 3);
}

}  // namespace
}  // namespace litert::ml_drift::ir
