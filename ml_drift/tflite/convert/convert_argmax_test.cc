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
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

class ConvertArgMaxTest : public ::testing::Test {};

TEST_F(ConvertArgMaxTest, SameRank) {
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      CreateStubDelegate(), DeleteStubDelegate);
  SingleOpInterpreterBuilder model(kTfLiteBuiltinArgMax);

  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});

  std::vector<int32_t> dim_data = {3};
  std::vector<uint8_t> dim_bytes(dim_data.size() * sizeof(int32_t));
  std::memcpy(dim_bytes.data(), dim_data.data(), dim_bytes.size());
  model.AddConstInput(kTfLiteInt32, {1}, dim_bytes);

  model.AddOutput(kTfLiteInt32, {1, 2, 3, 1});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 1);

  const auto* op = ir_model->op(0);
  EXPECT_EQ(op->name, "max_index");
  EXPECT_EQ(op->inputs.size(), 1);
  EXPECT_EQ(op->outputs.size(), 1);

  const auto* attr = std::any_cast<::ml_drift::MaxIndexAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->dim, ::ml_drift::Axis::CHANNELS);
}

TEST_F(ConvertArgMaxTest, WithReshape) {
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      CreateStubDelegate(), DeleteStubDelegate);
  SingleOpInterpreterBuilder model(kTfLiteBuiltinArgMax);

  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});

  std::vector<int32_t> dim_data = {3};
  std::vector<uint8_t> dim_bytes(dim_data.size() * sizeof(int32_t));
  std::memcpy(dim_bytes.data(), dim_data.data(), dim_bytes.size());
  model.AddConstInput(kTfLiteInt32, {1}, dim_bytes);

  model.AddOutput(kTfLiteInt32, {2, 3, 1});

  auto interpreter = model.Build();
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate.get());
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 2);

  const auto* op = ir_model->op(0);
  EXPECT_EQ(op->name, "max_index");
  EXPECT_EQ(op->inputs.size(), 1);
  EXPECT_EQ(op->outputs.size(), 1);

  const auto* attr = std::any_cast<::ml_drift::MaxIndexAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->dim, ::ml_drift::Axis::CHANNELS);

  const auto* reshape = ir_model->op(1);
  EXPECT_EQ(reshape->name, "reshape");
  EXPECT_EQ(reshape->inputs.size(), 1);
  EXPECT_EQ(reshape->outputs.size(), 1);
}

}  // namespace
}  // namespace litert::ml_drift::ir
