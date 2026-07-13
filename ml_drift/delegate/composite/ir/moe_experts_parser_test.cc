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

#include "third_party/odml/litert/ml_drift/delegate/composite/ir/moe_experts_parser.h"

#include <any>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_testing_utils.h"
#include "third_party/odml/litert/ml_drift/tflite/convert/stub_delegate.h"
#include "third_party/odml/litert/ml_drift/tflite/custom_ir_operation_parser.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::Eq;
using ::testing::SizeIs;

TfLiteCustomAllocation CreateMoeExpertsParams(
    int num_experts, int num_active_experts, int model_dim, int hidden_dim,
    const std::string& weight_type, const std::string& activation = "gelu",
    bool renormalized_top_weights = true) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_experts", num_experts);
    fbb.Int("num_active_experts", num_active_experts);
    fbb.Int("model_dim", model_dim);
    fbb.Int("hidden_dim", hidden_dim);
    fbb.String("weight_type", weight_type);
    fbb.String("activation", activation);
    fbb.Bool("renormalized_top_weights", renormalized_top_weights);
  });
  fbb.Finish();
  auto buffer = fbb.GetBuffer();

  void* block = calloc(1, buffer.size());
  memcpy(block, buffer.data(), buffer.size());

  TfLiteCustomAllocation allocation;
  allocation.data = block;
  allocation.bytes = buffer.size();
  return allocation;
}

class ConvertMoeExpertsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CustomIrOpMap custom_parsers;
    custom_parsers["moe"] = GetMoeExpertsParser();
    delegate_ = CreateStubDelegate(/*options=*/{}, std::move(custom_parsers));
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  TfLiteDelegate* delegate_;
};

TEST_F(ConvertMoeExpertsTest, Fp32Basic) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});  // src
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 2});   // top_weights
  builder.AddInput(kTfLiteInt32, {1, 1, 4, 2});     // top_indices

  std::vector<uint8_t> gate_weight_data(128 * 4 * 1 * 64 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 64},
                        gate_weight_data);  // gate_weight
  std::vector<uint8_t> ff1_weight_data(128 * 4 * 1 * 64 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 64},
                        ff1_weight_data);  // ff1_weight
  std::vector<uint8_t> linear_weight_data(64 * 4 * 1 * 128 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {64, 4, 1, 128},
                        linear_weight_data);  // linear_weight
  std::vector<uint8_t> scale_data(1 * 1 * 1 * 4 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {1, 1, 1, 4},
                        scale_data);  // per_expert_scale

  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});  // output

  TfLiteCustomAllocation custom_alloc =
      CreateMoeExpertsParams(4, 2, 64, 128, "fp32");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const auto* pair = interpreter->node_and_registration(0);
  const TfLiteNode* node = &pair->first;
  const TfLiteRegistration* registration = &pair->second;
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    node, registration);
  EXPECT_TRUE(status.ok()) << status.message();

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("moe_experts"));
  EXPECT_THAT(op->inputs, SizeIs(7));
  EXPECT_THAT(op->outputs, SizeIs(1));

  const auto* attr =
      std::any_cast<::litert::ml_drift::ir::MoeExpertsAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->num_experts, 4);
  EXPECT_EQ(attr->num_active_experts, 2);
  EXPECT_EQ(attr->model_dim, 64);
  EXPECT_EQ(attr->hidden_dim, 128);
  EXPECT_EQ(attr->weight_type,
            ::litert::ml_drift::ir::MoeExpertsAttributes::WeightType::kFp32);
}

TEST_F(ConvertMoeExpertsTest, Int8Basic) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});  // src
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 2});   // top_weights
  builder.AddInput(kTfLiteInt32, {1, 1, 4, 2});     // top_indices

  std::vector<uint8_t> gate_weight_data(128 * 4 * 1 * 64 * sizeof(int8_t), 0);
  builder.AddConstInput(kTfLiteInt8, {128, 4, 1, 64},
                        gate_weight_data);  // gate_weight
  std::vector<uint8_t> gate_scale_data(128 * 4 * 1 * 1 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 1},
                        gate_scale_data);  // gate_scale

  std::vector<uint8_t> ff1_weight_data(128 * 4 * 1 * 64 * sizeof(int8_t), 0);
  builder.AddConstInput(kTfLiteInt8, {128, 4, 1, 64},
                        ff1_weight_data);  // ff1_weight
  std::vector<uint8_t> ff1_scale_data(128 * 4 * 1 * 1 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 1},
                        ff1_scale_data);  // ff1_scale

  std::vector<uint8_t> linear_weight_data(64 * 4 * 1 * 128 * sizeof(int8_t), 0);
  builder.AddConstInput(kTfLiteInt8, {64, 4, 1, 128},
                        linear_weight_data);  // linear_weight
  std::vector<uint8_t> linear_scale_data(64 * 4 * 1 * 1 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {64, 4, 1, 1},
                        linear_scale_data);  // linear_scale

  std::vector<uint8_t> per_expert_scale_data(1 * 1 * 1 * 4 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {1, 1, 1, 4},
                        per_expert_scale_data);  // per_expert_scale

  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});  // output

  TfLiteCustomAllocation custom_alloc =
      CreateMoeExpertsParams(4, 2, 64, 128, "int8");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const auto* pair = interpreter->node_and_registration(0);
  const TfLiteNode* node = &pair->first;
  const TfLiteRegistration* registration = &pair->second;
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    node, registration);
  EXPECT_TRUE(status.ok()) << status.message();

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("moe_experts"));
  EXPECT_THAT(op->inputs, SizeIs(7));  // Int8 scale tensors are stripped!
  EXPECT_THAT(op->outputs, SizeIs(1));

  const auto* attr = std::any_cast<MoeExpertsAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->num_experts, 4);
  EXPECT_EQ(attr->num_active_experts, 2);
  EXPECT_EQ(attr->model_dim, 64);
  EXPECT_EQ(attr->hidden_dim, 128);
  EXPECT_EQ(attr->weight_type, MoeExpertsAttributes::WeightType::kInt8);
  EXPECT_TRUE(attr->ff_gate_scale.has_value());
  EXPECT_TRUE(attr->ff1_scale.has_value());
  EXPECT_TRUE(attr->linear_scale.has_value());
}

TEST_F(ConvertMoeExpertsTest, InferAttributesFromTensors) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});  // src
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 2});   // top_weights
  builder.AddInput(kTfLiteInt32, {1, 1, 4, 2});     // top_indices

  std::vector<uint8_t> gate_weight_data(128 * 4 * 1 * 64 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 64},
                        gate_weight_data);  // gate_weight
  std::vector<uint8_t> ff1_weight_data(128 * 4 * 1 * 64 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 64},
                        ff1_weight_data);  // ff1_weight
  std::vector<uint8_t> linear_weight_data(64 * 4 * 1 * 128 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {64, 4, 1, 128},
                        linear_weight_data);  // linear_weight
  std::vector<uint8_t> scale_data(1 * 1 * 1 * 4 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {1, 1, 1, 4},
                        scale_data);  // per_expert_scale

  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});  // output

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const auto* pair = interpreter->node_and_registration(0);
  const TfLiteNode* node = &pair->first;
  const TfLiteRegistration* registration = &pair->second;
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    node, registration);
  EXPECT_TRUE(status.ok()) << status.message();

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  const auto& op = ir_model->ops()[0];
  const auto* attr = std::any_cast<MoeExpertsAttributes>(&op->attr);
  EXPECT_EQ(attr->num_experts, 4);
  EXPECT_EQ(attr->num_active_experts, 2);
  EXPECT_EQ(attr->model_dim, 64);
  EXPECT_EQ(attr->hidden_dim, 128);
}

TEST_F(ConvertMoeExpertsTest, RejectsInvalidActivation) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 2});
  builder.AddInput(kTfLiteInt32, {1, 1, 4, 2});

  std::vector<uint8_t> gate_weight_data(128 * 4 * 1 * 64 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 64}, gate_weight_data);
  std::vector<uint8_t> ff1_weight_data(128 * 4 * 1 * 64 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 64}, ff1_weight_data);
  std::vector<uint8_t> linear_weight_data(64 * 4 * 1 * 128 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {64, 4, 1, 128}, linear_weight_data);
  std::vector<uint8_t> scale_data(1 * 1 * 1 * 4 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {1, 1, 1, 4}, scale_data);

  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});

  TfLiteCustomAllocation custom_alloc =
      CreateMoeExpertsParams(4, 2, 64, 128, "fp32", "relu");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  const auto* pair = interpreter->node_and_registration(0);
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    &pair->first, &pair->second);
  EXPECT_FALSE(status.ok());
}

TEST_F(ConvertMoeExpertsTest, RejectsAsymmetricQuantization) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});  // src
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 2});   // top_weights
  builder.AddInput(kTfLiteInt32, {1, 1, 4, 2});     // top_indices

  std::vector<uint8_t> gate_weight_data(128 * 4 * 1 * 64 * sizeof(int8_t), 0);
  builder.AddConstInput(kTfLiteInt8, {128, 4, 1, 64},
                        gate_weight_data);  // gate_weight
  std::vector<uint8_t> gate_scale_data(128 * 4 * 1 * 1 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 1},
                        gate_scale_data);  // gate_scale

  std::vector<uint8_t> ff1_weight_data(128 * 4 * 1 * 64 * sizeof(int8_t), 0);
  builder.AddConstInput(kTfLiteInt8, {128, 4, 1, 64},
                        ff1_weight_data);  // ff1_weight
  std::vector<uint8_t> ff1_scale_data(128 * 4 * 1 * 1 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 1},
                        ff1_scale_data);  // ff1_scale

  std::vector<uint8_t> linear_weight_data(64 * 4 * 1 * 128 * sizeof(int8_t), 0);
  builder.AddConstInput(kTfLiteInt8, {64, 4, 1, 128},
                        linear_weight_data);  // linear_weight
  std::vector<uint8_t> linear_scale_data(64 * 4 * 1 * 1 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {64, 4, 1, 1},
                        linear_scale_data);  // linear_scale

  std::vector<uint8_t> per_expert_scale_data(1 * 1 * 1 * 4 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {1, 1, 1, 4},
                        per_expert_scale_data);  // per_expert_scale

  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});  // output

  TfLiteCustomAllocation custom_alloc =
      CreateMoeExpertsParams(4, 2, 64, 128, "int8");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();

  // Set asymmetric quantization for Int8 weights (zero point = 5)
  for (int i : {3, 5, 7}) {
    TfLiteTensor* t = interpreter->tensor(interpreter->inputs()[i]);
    auto* q_params =
        reinterpret_cast<TfLiteAffineQuantization*>(t->quantization.params);
    q_params->zero_point->data[0] = 5;
  }

  const auto* pair = interpreter->node_and_registration(0);
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    &pair->first, &pair->second);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), ::testing::HasSubstr("symmetric int8"));
}

}  // namespace
}  // namespace litert::ml_drift::ir
