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

#include "ml_drift_delegate/delegate/composite/ir/swiglu_parser.h"

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
#include "ml_drift_delegate/delegate/composite/swiglu_parser.h"
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

TfLiteStablehloCompositeParams* CreateSwigluParams(int gate_size = 0) {
  size_t total_size = sizeof(TfLiteStablehloCompositeParams);
  std::vector<uint8_t> buffer;

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    if (gate_size > 0) {
      fbb.Int("gate_size", gate_size);
    }
  });
  fbb.Finish();
  buffer = fbb.GetBuffer();
  total_size += buffer.size();

  void* block = calloc(1, total_size);
  TfLiteStablehloCompositeParams* params =
      reinterpret_cast<TfLiteStablehloCompositeParams*>(block);
  params->name = "odml.swiglu";

  uint8_t* attr_data = reinterpret_cast<uint8_t*>(params + 1);
  params->attributes = attr_data;
  params->attributes_size = buffer.size();
  memcpy(attr_data, buffer.data(), buffer.size());

  return params;
}

class ConvertSwigluTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CustomIrOpMap custom_parsers;
    custom_parsers["odml.swiglu"] = GetSwigluParser();
    delegate_ = CreateStubDelegate(/*options=*/{}, std::move(custom_parsers));
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertSwigluTest, SingleInputFp32) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 1, 64});   // fused gate_up
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 32});  // result

  TfLiteStablehloCompositeParams* params = CreateSwigluParams(32);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("swiglu"));
  EXPECT_THAT(op->inputs, SizeIs(1));
  EXPECT_THAT(op->outputs, SizeIs(1));

  const auto* attr =
      std::any_cast<::litert::ml_drift::SwigluAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->gate_size, 32);
}

TEST_F(ConvertSwigluTest, TwoInputsFp16) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat16, {1, 1, 1, 32});   // gate
  builder.AddInput(kTfLiteFloat16, {1, 1, 1, 32});   // up
  builder.AddOutput(kTfLiteFloat16, {1, 1, 1, 32});  // result

  TfLiteStablehloCompositeParams* params = CreateSwigluParams(0);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("swiglu"));
  EXPECT_THAT(op->inputs, SizeIs(2));
  EXPECT_THAT(op->outputs, SizeIs(1));
}

}  // namespace
}  // namespace litert::ml_drift::ir
