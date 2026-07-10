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
#include <optional>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::Eq;
using ::testing::SizeIs;

TfLiteStablehloCompositeParams* CreateSdpaParams(
    std::optional<float> scale = std::nullopt) {
  size_t total_size = sizeof(TfLiteStablehloCompositeParams);
  std::vector<uint8_t> buffer;
  if (scale.has_value()) {
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Float("scale", *scale); });
    fbb.Finish();
    buffer = fbb.GetBuffer();
    total_size += buffer.size();
  }

  void* block = calloc(1, total_size);
  TfLiteStablehloCompositeParams* params =
      reinterpret_cast<TfLiteStablehloCompositeParams*>(block);
  params->name = "odml.scaled_dot_product_attention";

  if (scale.has_value()) {
    uint8_t* attr_data = reinterpret_cast<uint8_t*>(params + 1);
    params->attributes = attr_data;
    params->attributes_size = buffer.size();
    memcpy(attr_data, buffer.data(), buffer.size());
  }

  return params;
}

class ConvertSdpaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = CreateStubDelegate();
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertSdpaTest, Basic) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 4, 8, 16});  // Q
  builder.AddInput(kTfLiteFloat32, {1, 4, 8, 16});  // K
  builder.AddInput(kTfLiteFloat32, {1, 4, 8, 16});  // V
  builder.AddOutput(kTfLiteFloat32, {1, 4, 8, 16});

  TfLiteStablehloCompositeParams* params = CreateSdpaParams(0.5f);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(
      op->name,
      Eq(ToString(::ml_drift::OperationType::SCALED_DOT_PRODUCT_ATTENTION)));
  EXPECT_THAT(op->inputs, SizeIs(3));

  const auto* attr =
      std::any_cast<::ml_drift::ScaledDotProductAttentionAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_TRUE(attr->scale.has_value());
  EXPECT_THAT(*attr->scale, Eq(0.5f));
}

TEST_F(ConvertSdpaTest, WithMask) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 4, 8, 16});  // Q
  builder.AddInput(kTfLiteFloat32, {1, 4, 8, 16});  // K
  builder.AddInput(kTfLiteFloat32, {1, 4, 8, 16});  // V
  builder.AddInput(kTfLiteFloat32, {1, 1, 8, 4});   // Mask (C=4 matches K.H=4)
  builder.AddOutput(kTfLiteFloat32, {1, 4, 8, 16});

  TfLiteStablehloCompositeParams* params = CreateSdpaParams();
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->inputs, SizeIs(4));
}

TEST_F(ConvertSdpaTest, WithConstMask) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 4, 8, 16});  // Q
  builder.AddInput(kTfLiteFloat32, {1, 4, 8, 16});  // K
  builder.AddInput(kTfLiteFloat32, {1, 4, 8, 16});  // V

  std::vector<float> mask_data(8 * 4, 1.0f);
  std::vector<uint8_t> mask_bytes(mask_data.size() * sizeof(float));
  memcpy(mask_bytes.data(), mask_data.data(), mask_bytes.size());
  builder.AddConstInput(kTfLiteFloat32, {1, 1, 8, 4}, mask_bytes);  // Mask

  builder.AddOutput(kTfLiteFloat32, {1, 4, 8, 16});

  TfLiteStablehloCompositeParams* params = CreateSdpaParams();
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // Should have 2 ops: CONSTANT and SCALED_DOT_PRODUCT_ATTENTION
  ASSERT_THAT(ir_model->ops(), SizeIs(2));
  EXPECT_THAT(
      ir_model->ops()[0]->name,
      Eq(ToString(::ml_drift::OperationType::SCALED_DOT_PRODUCT_ATTENTION)));
  EXPECT_THAT(ir_model->ops()[0]->inputs, SizeIs(4));
  EXPECT_THAT(ir_model->ops()[1]->name,
              Eq(ToString(::ml_drift::OperationType::CONSTANT)));
}

}  // namespace
}  // namespace litert::ml_drift::ir
