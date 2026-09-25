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

#include "ml_drift_delegate/delegate/composite/ir/moe_experts_parser.h"

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

// Dimensions shared by the int8 fixtures below. The gate/ff1 projections read
// `kModelDim` inputs and produce `kHiddenDim` outputs; the linear projection is
// the transpose of that.
constexpr int kNumExperts = 4;
constexpr int kNumActiveExperts = 2;
constexpr int kModelDim = 64;
constexpr int kHiddenDim = 128;
constexpr int kNumTokens = 4;

std::vector<uint8_t> ZeroBytes(int num_elements, size_t element_size) {
  return std::vector<uint8_t>(num_elements * element_size, 0);
}

// Adds the ten inputs of an int8 moe op. The `*_blocks` arguments set the
// innermost extent of each scale tensor: 1 is per-output-channel quantization,
// and larger values split the input axis into that many equally sized blocks.
// Gate and ff1 block along `kModelDim`; linear blocks along `kHiddenDim`.
void AddInt8MoeInputs(SingleOpInterpreterBuilder& builder, int gate_blocks,
                      int ff1_blocks, int linear_blocks) {
  builder.AddInput(kTfLiteFloat32, {1, 1, kNumTokens, kModelDim});  // src
  builder.AddInput(kTfLiteFloat32,
                   {1, 1, kNumTokens, kNumActiveExperts});  // top_weights
  builder.AddInput(kTfLiteInt32,
                   {1, 1, kNumTokens, kNumActiveExperts});  // top_indices

  builder.AddConstInput(
      kTfLiteInt8, {kHiddenDim, kNumExperts, 1, kModelDim},
      ZeroBytes(kHiddenDim * kNumExperts * kModelDim, sizeof(int8_t)));
  builder.AddConstInput(
      kTfLiteFloat32, {kHiddenDim, kNumExperts, 1, gate_blocks},
      ZeroBytes(kHiddenDim * kNumExperts * gate_blocks, sizeof(float)));

  builder.AddConstInput(
      kTfLiteInt8, {kHiddenDim, kNumExperts, 1, kModelDim},
      ZeroBytes(kHiddenDim * kNumExperts * kModelDim, sizeof(int8_t)));
  builder.AddConstInput(
      kTfLiteFloat32, {kHiddenDim, kNumExperts, 1, ff1_blocks},
      ZeroBytes(kHiddenDim * kNumExperts * ff1_blocks, sizeof(float)));

  builder.AddConstInput(
      kTfLiteInt8, {kModelDim, kNumExperts, 1, kHiddenDim},
      ZeroBytes(kModelDim * kNumExperts * kHiddenDim, sizeof(int8_t)));
  builder.AddConstInput(
      kTfLiteFloat32, {kModelDim, kNumExperts, 1, linear_blocks},
      ZeroBytes(kModelDim * kNumExperts * linear_blocks, sizeof(float)));

  builder.AddConstInput(kTfLiteFloat32, {1, 1, 1, kNumExperts},
                        ZeroBytes(kNumExperts, sizeof(float)));

  builder.AddOutput(kTfLiteFloat32, {1, 1, kNumTokens, kModelDim});
}

// Replaces a tensor's affine quantization with V2 blockwise quantization
// pointing at `zero_point_tensor` (use a negative index for the tflite
// optional-tensor convention, which means the weights are symmetric).
void SetBlockwiseQuantization(TfLiteTensor* tensor, int zero_point_tensor,
                              int blocksize) {
  TfLiteQuantizationFree(&tensor->quantization);
  auto* params = reinterpret_cast<TfLiteBlockwiseQuantizationV2*>(
      calloc(1, sizeof(TfLiteBlockwiseQuantizationV2)));
  params->scale = -1;
  params->zero_point = zero_point_tensor;
  params->blocksize = blocksize;
  params->quantized_dimension = 0;
  params->block_shape = TfLiteIntArrayCreate(4);
  params->block_shape->data[0] = 1;
  params->block_shape->data[1] = 1;
  params->block_shape->data[2] = 1;
  params->block_shape->data[3] = blocksize;
  tensor->quantization.type = kTfLiteBlockwiseQuantizationV2;
  tensor->quantization.params = params;
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

TEST_F(ConvertMoeExpertsTest, Int4Basic) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});  // src
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 2});   // top_weights
  builder.AddInput(kTfLiteInt32, {1, 1, 4, 2});     // top_indices

  std::vector<uint8_t> gate_weight_data(128 * 4 * 1 * 64 * sizeof(int8_t), 0);
  builder.AddConstInput(kTfLiteInt4, {128, 4, 1, 64},
                        gate_weight_data);  // gate_weight
  std::vector<uint8_t> gate_scale_data(128 * 4 * 1 * 1 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 1},
                        gate_scale_data);  // gate_scale

  std::vector<uint8_t> ff1_weight_data(128 * 4 * 1 * 64 * sizeof(int8_t), 0);
  builder.AddConstInput(kTfLiteInt4, {128, 4, 1, 64},
                        ff1_weight_data);  // ff1_weight
  std::vector<uint8_t> ff1_scale_data(128 * 4 * 1 * 1 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {128, 4, 1, 1},
                        ff1_scale_data);  // ff1_scale

  std::vector<uint8_t> linear_weight_data(64 * 4 * 1 * 128 * sizeof(int8_t), 0);
  builder.AddConstInput(kTfLiteInt4, {64, 4, 1, 128},
                        linear_weight_data);  // linear_weight
  std::vector<uint8_t> linear_scale_data(64 * 4 * 1 * 1 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {64, 4, 1, 1},
                        linear_scale_data);  // linear_scale

  std::vector<uint8_t> per_expert_scale_data(1 * 1 * 1 * 4 * sizeof(float), 0);
  builder.AddConstInput(kTfLiteFloat32, {1, 1, 1, 4},
                        per_expert_scale_data);  // per_expert_scale

  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});  // output

  TfLiteCustomAllocation custom_alloc =
      CreateMoeExpertsParams(4, 2, 64, 128, "int4");
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
  EXPECT_THAT(op->inputs, SizeIs(7));  // Int4 scale tensors are stripped!
  EXPECT_THAT(op->outputs, SizeIs(1));

  const auto* attr = std::any_cast<MoeExpertsAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->num_experts, 4);
  EXPECT_EQ(attr->num_active_experts, 2);
  EXPECT_EQ(attr->model_dim, 64);
  EXPECT_EQ(attr->hidden_dim, 128);
  EXPECT_EQ(attr->weight_type, MoeExpertsAttributes::WeightType::kInt4);
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

// Blockwise scales are the reason the parser reads the scale shape instead of
// assuming one scale per output channel: here each row of gate/ff1 carries 4
// scales over 64 input channels (block size 16), and linear carries 4 scales
// over 128 input channels (block size 32).
TEST_F(ConvertMoeExpertsTest, Int8BlockwiseScales) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  AddInt8MoeInputs(builder, /*gate_blocks=*/4, /*ff1_blocks=*/4,
                   /*linear_blocks=*/4);

  TfLiteCustomAllocation custom_alloc = CreateMoeExpertsParams(
      kNumExperts, kNumActiveExperts, kModelDim, kHiddenDim, "int8");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const auto* pair = interpreter->node_and_registration(0);
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    &pair->first, &pair->second);
  EXPECT_TRUE(status.ok()) << status.message();

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto* attr =
      std::any_cast<MoeExpertsAttributes>(&ir_model->ops()[0]->attr);
  ASSERT_NE(attr, nullptr);

  // The block count has to survive into the attributes, because that shape is
  // what the kernel hands to the matmul as `scale_zp_shape`.
  ASSERT_TRUE(attr->ff_gate_scale.has_value());
  EXPECT_EQ(attr->ff_gate_scale->shape.o, kHiddenDim);
  EXPECT_EQ(attr->ff_gate_scale->shape.h, kNumExperts);
  EXPECT_EQ(attr->ff_gate_scale->shape.w, 1);
  EXPECT_EQ(attr->ff_gate_scale->shape.i, 4);

  ASSERT_TRUE(attr->ff1_scale.has_value());
  EXPECT_EQ(attr->ff1_scale->shape.i, 4);

  ASSERT_TRUE(attr->linear_scale.has_value());
  EXPECT_EQ(attr->linear_scale->shape.o, kModelDim);
  EXPECT_EQ(attr->linear_scale->shape.i, 4);
}

// Per-output-channel scales are the degenerate one-block case and must keep
// parsing exactly as before.
TEST_F(ConvertMoeExpertsTest, Int8PerChannelScalesAreSingleBlock) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  AddInt8MoeInputs(builder, /*gate_blocks=*/1, /*ff1_blocks=*/1,
                   /*linear_blocks=*/1);

  TfLiteCustomAllocation custom_alloc = CreateMoeExpertsParams(
      kNumExperts, kNumActiveExperts, kModelDim, kHiddenDim, "int8");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);
  const auto* attr =
      std::any_cast<MoeExpertsAttributes>(&ir_model->ops()[0]->attr);
  ASSERT_NE(attr, nullptr);
  ASSERT_TRUE(attr->ff_gate_scale.has_value());
  EXPECT_EQ(attr->ff_gate_scale->shape.i, 1);
}

// A block count that does not divide the input axis would leave the kernel
// with a ragged final block, so it is rejected rather than truncated.
TEST_F(ConvertMoeExpertsTest, RejectsBlockCountNotDividingInputChannels) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  AddInt8MoeInputs(builder, /*gate_blocks=*/5, /*ff1_blocks=*/1,
                   /*linear_blocks=*/1);

  TfLiteCustomAllocation custom_alloc = CreateMoeExpertsParams(
      kNumExperts, kNumActiveExperts, kModelDim, kHiddenDim, "int8");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const auto* pair = interpreter->node_and_registration(0);
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    &pair->first, &pair->second);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("divide the input channel"));
}

// More blocks than input channels is the same failure mode; the block count is
// compared against the projection's own input axis, not the model dim.
TEST_F(ConvertMoeExpertsTest, RejectsMoreBlocksThanInputChannels) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  AddInt8MoeInputs(builder, /*gate_blocks=*/1, /*ff1_blocks=*/1,
                   /*linear_blocks=*/kHiddenDim * 2);

  TfLiteCustomAllocation custom_alloc = CreateMoeExpertsParams(
      kNumExperts, kNumActiveExperts, kModelDim, kHiddenDim, "int8");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const auto* pair = interpreter->node_and_registration(0);
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    &pair->first, &pair->second);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("divide the input channel"));
}

// The outer dimensions of a scale tensor still have to match the projection,
// even though the innermost one is now free.
TEST_F(ConvertMoeExpertsTest, RejectsScaleWithWrongExpertCount) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  builder.AddInput(kTfLiteFloat32, {1, 1, kNumTokens, kModelDim});
  builder.AddInput(kTfLiteFloat32, {1, 1, kNumTokens, kNumActiveExperts});
  builder.AddInput(kTfLiteInt32, {1, 1, kNumTokens, kNumActiveExperts});

  builder.AddConstInput(
      kTfLiteInt8, {kHiddenDim, kNumExperts, 1, kModelDim},
      ZeroBytes(kHiddenDim * kNumExperts * kModelDim, sizeof(int8_t)));
  // One expert too few.
  builder.AddConstInput(
      kTfLiteFloat32, {kHiddenDim, kNumExperts - 1, 1, 1},
      ZeroBytes(kHiddenDim * (kNumExperts - 1), sizeof(float)));
  builder.AddConstInput(
      kTfLiteInt8, {kHiddenDim, kNumExperts, 1, kModelDim},
      ZeroBytes(kHiddenDim * kNumExperts * kModelDim, sizeof(int8_t)));
  builder.AddConstInput(kTfLiteFloat32, {kHiddenDim, kNumExperts, 1, 1},
                        ZeroBytes(kHiddenDim * kNumExperts, sizeof(float)));
  builder.AddConstInput(
      kTfLiteInt8, {kModelDim, kNumExperts, 1, kHiddenDim},
      ZeroBytes(kModelDim * kNumExperts * kHiddenDim, sizeof(int8_t)));
  builder.AddConstInput(kTfLiteFloat32, {kModelDim, kNumExperts, 1, 1},
                        ZeroBytes(kModelDim * kNumExperts, sizeof(float)));
  builder.AddConstInput(kTfLiteFloat32, {1, 1, 1, kNumExperts},
                        ZeroBytes(kNumExperts, sizeof(float)));
  builder.AddOutput(kTfLiteFloat32, {1, 1, kNumTokens, kModelDim});

  TfLiteCustomAllocation custom_alloc = CreateMoeExpertsParams(
      kNumExperts, kNumActiveExperts, kModelDim, kHiddenDim, "int8");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const auto* pair = interpreter->node_and_registration(0);
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    &pair->first, &pair->second);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), ::testing::HasSubstr("ff_gate_scale"));
}

// Blockwise-quantized weights carry their zero point in a separate tensor. A
// negative index is the tflite optional-tensor convention for "no zero point",
// i.e. symmetric, which is what the kernel requires.
TEST_F(ConvertMoeExpertsTest, AcceptsBlockwiseQuantizationWithoutZeroPoint) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  AddInt8MoeInputs(builder, /*gate_blocks=*/4, /*ff1_blocks=*/4,
                   /*linear_blocks=*/4);

  TfLiteCustomAllocation custom_alloc = CreateMoeExpertsParams(
      kNumExperts, kNumActiveExperts, kModelDim, kHiddenDim, "int8");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  for (int i : {3, 5, 7}) {
    SetBlockwiseQuantization(interpreter->tensor(interpreter->inputs()[i]),
                             /*zero_point_tensor=*/-1, /*blocksize=*/16);
  }

  const auto* pair = interpreter->node_and_registration(0);
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    &pair->first, &pair->second);
  EXPECT_TRUE(status.ok()) << status.message();
}

// A zero point tensor that is present but not all zeros means asymmetric
// weights, which the kernel cannot represent.
TEST_F(ConvertMoeExpertsTest, RejectsBlockwiseAsymmetricZeroPoint) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  AddInt8MoeInputs(builder, /*gate_blocks=*/4, /*ff1_blocks=*/4,
                   /*linear_blocks=*/4);

  TfLiteCustomAllocation custom_alloc = CreateMoeExpertsParams(
      kNumExperts, kNumActiveExperts, kModelDim, kHiddenDim, "int8");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  // A constant int8 tensor outside the op's inputs, used only as the zero
  // point of the gate weights.
  int zero_point_index = 0;
  ASSERT_EQ(interpreter->AddTensors(1, &zero_point_index), kTfLiteOk);
  const std::vector<int8_t> zero_point_data(kHiddenDim * kNumExperts * 4, 3);
  ASSERT_EQ(
      interpreter->SetTensorParametersReadOnly(
          zero_point_index, kTfLiteInt8, "zero_point",
          {kHiddenDim, kNumExperts, 1, 4}, {kTfLiteNoQuantization, nullptr},
          reinterpret_cast<const char*>(zero_point_data.data()),
          zero_point_data.size() * sizeof(int8_t)),
      kTfLiteOk);

  SetBlockwiseQuantization(interpreter->tensor(interpreter->inputs()[3]),
                           zero_point_index, /*blocksize=*/16);

  const auto* pair = interpreter->node_and_registration(0);
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    &pair->first, &pair->second);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), ::testing::HasSubstr("symmetric"));
}

// The kernel emits the tanh approximation of gelu, so "gelu_tanh" is the
// accurate label. "gelu" stays accepted for models from older exporters.
TEST_F(ConvertMoeExpertsTest, AcceptsGeluTanhActivation) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinCustom);
  builder.SetCustomName("moe");
  AddInt8MoeInputs(builder, /*gate_blocks=*/1, /*ff1_blocks=*/1,
                   /*linear_blocks=*/1);

  TfLiteCustomAllocation custom_alloc =
      CreateMoeExpertsParams(kNumExperts, kNumActiveExperts, kModelDim,
                             kHiddenDim, "int8", "gelu_tanh");
  builder.SetCustomData(custom_alloc.data, custom_alloc.bytes);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const auto* pair = interpreter->node_and_registration(0);
  auto parser = GetMoeExpertsParser();
  auto status = parser.is_supported(interpreter->primary_subgraph().context(),
                                    &pair->first, &pair->second);
  EXPECT_TRUE(status.ok()) << status.message();
}

}  // namespace
}  // namespace litert::ml_drift::ir
