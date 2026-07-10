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

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertTileTest : public ::testing::TestWithParam<TfLiteType> {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_P(ConvertTileTest, BasicTile) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinTile, /*version=*/1);
  model.AddInput(GetParam(), {1, 2, 3, 4});  // input data
  std::vector<int32_t> multiples = {
      1,  // batch
      2,  // height
      1,  // width
      3   // channels
  };
  std::vector<uint8_t> multiples_bytes(multiples.size() * sizeof(int32_t));
  std::memcpy(multiples_bytes.data(), multiples.data(), multiples_bytes.size());
  model.AddConstInput(kTfLiteInt32, {4}, multiples_bytes);  // multiples
  model.AddOutput(GetParam(), {1, 4, 3, 12});

  std::unique_ptr<::tflite::Interpreter> interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  EXPECT_EQ(ir_model->inputs().size(), 1);
  EXPECT_EQ(ir_model->outputs().size(), 1);
  ASSERT_EQ(ir_model->ops().size(), 1);

  const ::ml_drift::ir::IrOp* tile_op = ir_model->op(0);
  EXPECT_EQ(tile_op->name, "tile");

  // Sanity check inference.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
}

INSTANTIATE_TEST_SUITE_P(ConvertTileTestTypes, ConvertTileTest,
                         ::testing::Values(kTfLiteFloat32, kTfLiteFloat16,
                                           kTfLiteInt32, kTfLiteInt8,
                                           kTfLiteUInt8, kTfLiteBool));

}  // namespace
}  // namespace litert::ml_drift::ir
