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

#include "ml_drift_delegate/delegate/composite/ir/runtime_batched_matmul_parser.h"

#include <any>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"
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

TfLiteStablehloCompositeParams* CreateRuntimeBmmParams(
    bool is_global, bool is_src, bool rhs_cache_update,
    std::optional<float> scale = std::nullopt) {
  size_t total_size = sizeof(TfLiteStablehloCompositeParams);
  std::vector<uint8_t> buffer;

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Bool("is_global", is_global);
    fbb.Bool("is_src", is_src);
    fbb.Bool("rhs_cache_update", rhs_cache_update);
    if (scale.has_value()) {
      fbb.Float("scale", *scale);
    }
  });
  fbb.Finish();
  buffer = fbb.GetBuffer();
  total_size += buffer.size();

  void* block = calloc(1, total_size);
  TfLiteStablehloCompositeParams* params =
      reinterpret_cast<TfLiteStablehloCompositeParams*>(block);
  params->name = "odml.runtime_bmm";

  uint8_t* attr_data = reinterpret_cast<uint8_t*>(params + 1);
  params->attributes = attr_data;
  params->attributes_size = buffer.size();
  memcpy(attr_data, buffer.data(), buffer.size());

  return params;
}

class ConvertRuntimeBatchedMatMulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CustomIrOpMap custom_parsers;
    custom_parsers["odml.runtime_bmm"] = GetRuntimeBatchedMatMulParser();
    delegate_ = CreateStubDelegate(/*options=*/{}, std::move(custom_parsers));
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertRuntimeBatchedMatMulTest, BasicFp32) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});  // lhs
  builder.AddInput(
      kTfLiteFloat32,
      {1, 1, 128, 64});  // rhs (transposed right so channels match)
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});       // param tensor
  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 128});  // result

  TfLiteStablehloCompositeParams* params = CreateRuntimeBmmParams(
      /*is_global=*/true, /*is_src=*/true, /*rhs_cache_update=*/false);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("runtime_batched_matmul"));
  EXPECT_THAT(op->inputs, SizeIs(3));
  EXPECT_THAT(op->outputs, SizeIs(1));

  const auto* attr =
      std::any_cast<::litert::ml_drift::RuntimeBatchedMatMulAttributes>(
          &op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_TRUE(attr->transpose_right.has_value());
  EXPECT_TRUE(*attr->transpose_right);
  EXPECT_TRUE(attr->runtime_check.src_end_ch_index.has_value());
  EXPECT_EQ(*attr->runtime_check.src_end_ch_index, 2);
}

TEST_F(ConvertRuntimeBatchedMatMulTest, ReshapesModelBatch) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {2, 2, 4, 64});    // lhs
  builder.AddInput(kTfLiteFloat32, {2, 2, 128, 64});  // rhs
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});       // param tensor
  builder.AddOutput(kTfLiteFloat32, {2, 2, 4, 128});  // result

  TfLiteStablehloCompositeParams* params = CreateRuntimeBmmParams(
      /*is_global=*/true, /*is_src=*/true, /*rhs_cache_update=*/false);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  // 1 BMM + 3 Reshapes = 4 ops total
  ASSERT_THAT(ir_model->ops(), SizeIs(4));
}

TEST_F(ConvertRuntimeBatchedMatMulTest, Int8Basic) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});    // lhs
  builder.AddInput(kTfLiteInt8, {1, 1, 128, 64});     // rhs
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});       // param tensor
  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 128});  // result

  TfLiteStablehloCompositeParams* params = CreateRuntimeBmmParams(
      /*is_global=*/true, /*is_src=*/true, /*rhs_cache_update=*/true, 0.5f);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const auto* pair = interpreter->node_and_registration(0);
  const TfLiteNode* node = &pair->first;
  const TfLiteRegistration* registration = &pair->second;
  auto parser = GetRuntimeBatchedMatMulParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    node, registration);
  EXPECT_TRUE(status.ok()) << status.message();

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("runtime_batched_matmul"));
  EXPECT_THAT(op->inputs, SizeIs(3));
  EXPECT_THAT(op->outputs, SizeIs(1));

  const auto* attr =
      std::any_cast<::litert::ml_drift::RuntimeBatchedMatMulAttributes>(
          &op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_TRUE(attr->external_weights.has_value());
  EXPECT_EQ(attr->external_weights->desc.type, ::ml_drift::DataType::UINT8);
  EXPECT_TRUE(attr->scale.has_value());
}

}  // namespace
}  // namespace litert::ml_drift::ir
