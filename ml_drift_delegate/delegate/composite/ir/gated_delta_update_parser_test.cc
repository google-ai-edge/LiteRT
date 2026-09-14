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

#include "ml_drift_delegate/delegate/composite/ir/gated_delta_update_parser.h"

#include <any>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/gated_delta_update_parser.h"
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::Eq;
using ::testing::SizeIs;

class ConvertGatedDeltaUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CustomIrOpMap custom_parsers;
    custom_parsers["gated_delta_update"] = GetGatedDeltaUpdateParser();
    delegate_ = CreateStubDelegate(/*options=*/{}, std::move(custom_parsers));
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertGatedDeltaUpdateTest, BasicCustomOpConversion) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("gated_delta_update");

  int B = 1, H = 2, N = 4, D_k = 16, D_v = 16;
  builder.AddInput(kTfLiteFloat32, {B, H, N, D_k});     // q
  builder.AddInput(kTfLiteFloat32, {B, H, N, D_k});     // k
  builder.AddInput(kTfLiteFloat32, {B, H, N, D_v});     // v
  builder.AddInput(kTfLiteFloat32, {B, H, N});          // beta
  builder.AddInput(kTfLiteFloat32, {B, H, N});          // g
  builder.AddInput(kTfLiteFloat32, {B, H, D_k, D_v});   // initial_state
  builder.AddOutput(kTfLiteFloat32, {B, H, N, D_v});    // out
  builder.AddOutput(kTfLiteFloat32, {B, H, D_k, D_v});  // final_state

  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("mode", 0); });
  fbb.Finish();
  auto fbb_buf = fbb.GetBuffer();
  void* custom_data = malloc(fbb_buf.size());
  memcpy(custom_data, fbb_buf.data(), fbb_buf.size());
  builder.SetCustomData(custom_data, fbb_buf.size());

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("gated_delta_update"));
  EXPECT_THAT(op->inputs, SizeIs(6));
  EXPECT_THAT(op->outputs, SizeIs(2));

  const auto* attr =
      std::any_cast<::litert::ml_drift::GatedDeltaUpdateAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->mode, 0);
}

}  // namespace
}  // namespace litert::ml_drift::ir
