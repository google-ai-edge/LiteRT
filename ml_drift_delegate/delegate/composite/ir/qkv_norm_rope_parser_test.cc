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

#include "ml_drift_delegate/delegate/composite/ir/qkv_norm_rope_parser.h"

#include <any>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_parser.h"
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::Eq;
using ::testing::SizeIs;

TfLiteStablehloCompositeParams* CreateQkvNormRopeParams(
    int num_heads, int num_kv_heads, int head_dim, float min_timescale,
    float max_timescale, float proportion, float epsilon) {
  size_t total_size = sizeof(TfLiteStablehloCompositeParams);
  std::vector<uint8_t> buffer;

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_heads", num_heads);
    fbb.Int("num_kv_heads", num_kv_heads);
    fbb.Int("head_dim", head_dim);
    fbb.Float("min_timescale", min_timescale);
    fbb.Float("max_timescale", max_timescale);
    fbb.Float("proportion", proportion);
    fbb.Float("epsilon", epsilon);
  });
  fbb.Finish();
  buffer = fbb.GetBuffer();
  total_size += buffer.size();

  void* block = calloc(1, total_size);
  TfLiteStablehloCompositeParams* params =
      reinterpret_cast<TfLiteStablehloCompositeParams*>(block);
  params->name = "odml.qkv_norm_rope";

  uint8_t* attr_data = reinterpret_cast<uint8_t*>(params + 1);
  params->attributes = attr_data;
  params->attributes_size = buffer.size();
  memcpy(attr_data, buffer.data(), buffer.size());

  return params;
}

class ConvertQkvNormRopeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CustomIrOpMap custom_parsers;
    custom_parsers["odml.qkv_norm_rope"] = GetQkvNormRopeParser();
    delegate_ = CreateStubDelegate(/*options=*/{}, std::move(custom_parsers));
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertQkvNormRopeTest, BasicConversion) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 1, 32 * 128});  // qkv
  builder.AddInput(kTfLiteInt32, {1, 1});                 // position
  builder.AddInput(kTfLiteFloat32, {128});                // q_weight
  builder.AddInput(kTfLiteFloat32, {128});                // k_weight
  builder.AddOutput(kTfLiteFloat32, {1, 16, 1, 128});     // q_out
  builder.AddOutput(kTfLiteFloat32, {1, 8, 1, 128});      // k_out
  builder.AddOutput(kTfLiteFloat32, {1, 8, 1, 128});      // v_out

  TfLiteStablehloCompositeParams* params =
      CreateQkvNormRopeParams(16, 8, 128, 1.0f, 10000.0f, 1.0f, 1e-6f);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("qkv_norm_rope"));
  EXPECT_THAT(op->inputs, SizeIs(4));
  EXPECT_THAT(op->outputs, SizeIs(3));

  const auto* attr =
      std::any_cast<::litert::ml_drift::QkvNormRopeAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->num_heads, 16);
  EXPECT_EQ(attr->num_kv_heads, 8);
  EXPECT_EQ(attr->head_dim, 128);
  EXPECT_FLOAT_EQ(attr->min_timescale, 1.0f);
  EXPECT_FLOAT_EQ(attr->max_timescale, 10000.0f);
  EXPECT_FLOAT_EQ(attr->proportion, 1.0f);
  EXPECT_FLOAT_EQ(attr->epsilon, 1e-6f);
}

TEST_F(ConvertQkvNormRopeTest, CustomHeadAttributes) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32,
                   {1, 1, 1, 48 * 128});               // qkv (32 + 8 + 8) * 128
  builder.AddInput(kTfLiteInt32, {1, 1});              // position
  builder.AddInput(kTfLiteFloat32, {128});             // q_weight
  builder.AddInput(kTfLiteFloat32, {128});             // k_weight
  builder.AddOutput(kTfLiteFloat32, {1, 32, 1, 128});  // q_out
  builder.AddOutput(kTfLiteFloat32, {1, 8, 1, 128});   // k_out
  builder.AddOutput(kTfLiteFloat32, {1, 8, 1, 128});   // v_out

  TfLiteStablehloCompositeParams* params =
      CreateQkvNormRopeParams(32, 8, 128, 1.0f, 10000.0f, 1.0f, 1e-6f);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("qkv_norm_rope"));
  EXPECT_THAT(op->inputs, SizeIs(4));
  EXPECT_THAT(op->outputs, SizeIs(3));

  const auto* attr =
      std::any_cast<::litert::ml_drift::QkvNormRopeAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->num_heads, 32);
  EXPECT_EQ(attr->num_kv_heads, 8);
  EXPECT_EQ(attr->head_dim, 128);
  EXPECT_FLOAT_EQ(attr->min_timescale, 1.0f);
  EXPECT_FLOAT_EQ(attr->max_timescale, 10000.0f);
  EXPECT_FLOAT_EQ(attr->proportion, 1.0f);
  EXPECT_FLOAT_EQ(attr->epsilon, 1e-6f);
}

}  // namespace
}  // namespace litert::ml_drift::ir
