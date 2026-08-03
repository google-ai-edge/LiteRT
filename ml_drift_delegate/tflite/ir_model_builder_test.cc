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

#include "ml_drift_delegate/tflite/ir_model_builder.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

struct TestDelegateData {
  IrModelBuilderOptions options;
  TensorIndexToExternalBufferIdMap external_buffer_map;
  TensorIndexToBufferIdMap internal_buffer_map;
  SharedConstTensorsMap shared_tensors;
  ::ml_drift::ir::IrModel ir_model;
  absl::Status status;
};

TfLiteStatus TestPrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  auto* test_data = static_cast<TestDelegateData*>(delegate->data_);

  TfLiteIntArray* execution_plan;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    return kTfLiteError;
  }

  TfLiteIntArray* nodes_to_replace = TfLiteIntArrayCreate(execution_plan->size);
  for (int i = 0; i < execution_plan->size; ++i) {
    nodes_to_replace->data[i] = execution_plan->data[i];
  }

  std::vector<int> inputs;
  std::vector<int> outputs;
  for (int i = 0; i < nodes_to_replace->size; ++i) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    context->GetNodeAndRegistration(context, nodes_to_replace->data[i], &node,
                                    &reg);
    for (int j = 0; j < node->inputs->size; ++j) {
      if (node->inputs->data[j] != kTfLiteOptionalTensor) {
        inputs.push_back(node->inputs->data[j]);
      }
    }
    for (int j = 0; j < node->outputs->size; ++j) {
      outputs.push_back(node->outputs->data[j]);
    }
  }

  TfLiteIntArray* inputs_array = TfLiteIntArrayCreate(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs_array->data[i] = inputs[i];
  }

  TfLiteIntArray* outputs_array = TfLiteIntArrayCreate(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs_array->data[i] = outputs[i];
  }

  TfLiteDelegateParams delegate_params = {};
  delegate_params.delegate = delegate;
  delegate_params.nodes_to_replace = nodes_to_replace;
  delegate_params.input_tensors = inputs_array;
  delegate_params.output_tensors = outputs_array;

  ::ml_drift::ir::IrModel* model = BuildIrModel(
      *context, delegate_params, test_data->options, /*custom_parsers=*/nullptr,
      &test_data->shared_tensors, &test_data->internal_buffer_map,
      &test_data->external_buffer_map);
  if (model) {
    test_data->ir_model = std::move(*model);
    delete model;
    test_data->status = absl::OkStatus();
  } else {
    test_data->status = absl::InternalError("Failed to build model");
  }

  TfLiteIntArrayFree(nodes_to_replace);
  TfLiteIntArrayFree(inputs_array);
  TfLiteIntArrayFree(outputs_array);

  return kTfLiteOk;
}

class IrModelBuilderTest : public ::testing::Test {
 protected:
  // Builds a single Conv2D interpreter whose constant weights use the given
  // dtype. int8 weights carry affine quantization (see GetQuantization).
  void BuildInterpreter(TfLiteType weights_type,
                        const std::vector<uint8_t>& weights_data,
                        bool add_bias = false) {
    model_ = std::make_unique<SingleOpInterpreterBuilder>(kTfLiteBuiltinConv2d);
    model_->AddInput(kTfLiteFloat32, {1, 2, 3, 4});
    // AddConstInput stores a pointer to the data rather than copying it, so the
    // buffers must outlive the interpreter. Hold them in fixture members.
    weights_data_ = weights_data;
    model_->AddConstInput(weights_type, {8, 1, 1, 4}, weights_data_);
    if (add_bias) {
      bias_data_.assign(8 * sizeof(float), 0);
      model_->AddConstInput(kTfLiteFloat32, {8}, bias_data_);
    }
    model_->AddOutput(kTfLiteFloat32, {1, 2, 3, 8});

    auto* params = reinterpret_cast<TfLiteConvParams*>(
        calloc(1, sizeof(TfLiteConvParams)));
    params->padding = kTfLitePaddingSame;
    params->stride_height = 1;
    params->stride_width = 1;
    params->activation = kTfLiteActNone;
    params->dilation_height_factor = 1;
    params->dilation_width_factor = 1;
    model_->SetParameters(params);

    interpreter_ = model_->Build();
    ASSERT_NE(interpreter_, nullptr);
  }

  // Builds a single FullyConnected interpreter whose constant weights use the
  // given dtype (int8 weights carry affine quantization) plus a constant fp32
  // bias.
  void BuildFullyConnectedInterpreter(
      TfLiteType weights_type = kTfLiteFloat32,
      std::vector<uint8_t> weights_data =
          std::vector<uint8_t>(8 * 4 * sizeof(float), 0)) {
    model_ = std::make_unique<SingleOpInterpreterBuilder>(
        kTfLiteBuiltinFullyConnected);
    model_->AddInput(kTfLiteFloat32, {1, 1, 1, 4});
    // AddConstInput stores a pointer to the data rather than copying it, so the
    // buffers must outlive the interpreter. Hold them in fixture members.
    weights_data_ = std::move(weights_data);
    model_->AddConstInput(weights_type, {8, 4}, weights_data_);
    bias_data_.assign(8 * sizeof(float), 0);
    model_->AddConstInput(kTfLiteFloat32, {8}, bias_data_);
    model_->AddOutput(kTfLiteFloat32, {1, 8});

    auto* params = reinterpret_cast<TfLiteFullyConnectedParams*>(
        calloc(1, sizeof(TfLiteFullyConnectedParams)));
    params->activation = kTfLiteActNone;
    model_->SetParameters(params);

    interpreter_ = model_->Build();
    ASSERT_NE(interpreter_, nullptr);
  }

  // Builds a single EmbeddingLookup interpreter: int32 lookup indices plus
  // constant weights of the given dtype (vocab=5, embedding_dim=4).
  void BuildEmbeddingLookupInterpreter(
      TfLiteType weights_type, const std::vector<uint8_t>& weights_data) {
    model_ = std::make_unique<SingleOpInterpreterBuilder>(
        kTfLiteBuiltinEmbeddingLookup);
    model_->AddInput(kTfLiteInt32, {3});  // lookup indices
    model_->AddConstInput(weights_type, {5, 4}, weights_data);  // weights
    model_->AddOutput(kTfLiteFloat32, {3, 4});                  // output

    interpreter_ = model_->Build();
    ASSERT_NE(interpreter_, nullptr);
  }

  // Builds a pointwise (1x1) Conv2D over a 1x1 spatial input, which is
  // equivalent to a fully-connected op and is reduced to one.
  void BuildPointwiseConvInterpreter(TfLiteType weights_type,
                                     const std::vector<uint8_t>& weights_data) {
    model_ = std::make_unique<SingleOpInterpreterBuilder>(kTfLiteBuiltinConv2d);
    model_->AddInput(kTfLiteFloat32, {1, 1, 1, 4});
    model_->AddConstInput(weights_type, {8, 1, 1, 4}, weights_data);
    model_->AddOutput(kTfLiteFloat32, {1, 1, 1, 8});

    auto* params = reinterpret_cast<TfLiteConvParams*>(
        calloc(1, sizeof(TfLiteConvParams)));
    params->padding = kTfLitePaddingValid;
    params->stride_height = 1;
    params->stride_width = 1;
    params->activation = kTfLiteActNone;
    params->dilation_height_factor = 1;
    params->dilation_width_factor = 1;
    model_->SetParameters(params);

    interpreter_ = model_->Build();
    ASSERT_NE(interpreter_, nullptr);
  }

  void SetUp() override {
    BuildInterpreter(kTfLiteFloat32,
                     std::vector<uint8_t>(8 * 4 * sizeof(float), 0));

    delegate_.data_ = &test_data_;
    delegate_.Prepare = TestPrepare;
    delegate_.CopyFromBufferHandle = nullptr;
    delegate_.CopyToBufferHandle = nullptr;
    delegate_.FreeBufferHandle = nullptr;
    delegate_.flags = kTfLiteDelegateFlagsNone;
  }

  void VerifySharedWeightsTensor(int weights_tensor_index,
                                 int expected_global_id) {
    ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
    EXPECT_TRUE(test_data_.status.ok());
    EXPECT_EQ(test_data_.shared_tensors.size(), 1);

    bool found_shared_tensor = false;
    for (const auto& tensor : test_data_.ir_model.tensors()) {
      if (tensor->buffer_source.is_shared) {
        found_shared_tensor = true;
        EXPECT_TRUE(test_data_.shared_tensors.contains(tensor->id));
        EXPECT_EQ(test_data_.shared_tensors.at(tensor->id).global_id,
                  expected_global_id);
        EXPECT_EQ(test_data_.shared_tensors.at(tensor->id).tflite_tensor_id,
                  weights_tensor_index);
        // fp32 weights are not affine-quantized, so no forced dequantization.
        EXPECT_FALSE(test_data_.shared_tensors.at(tensor->id).dequant_forced);
      }
    }
    EXPECT_TRUE(found_shared_tensor);

    bool found_conv = false;
    for (const auto& op : test_data_.ir_model.ops()) {
      if (op->name == ToString(::ml_drift::OperationType::CONVOLUTION_2D)) {
        found_conv = true;
        int num_inputs_to_conv = 0;
        for (const auto& tensor : test_data_.ir_model.tensors()) {
          if (tensor->consumers.contains(op->id)) {
            num_inputs_to_conv++;
          }
        }
        // Explicitly verifies the tensor was not fused.
        EXPECT_EQ(num_inputs_to_conv, 2);
      }
    }
    EXPECT_TRUE(found_conv);
  }

  std::unique_ptr<SingleOpInterpreterBuilder> model_;
  // Backing storage for const-tensor data. The interpreter's read-only tensors
  // reference these buffers by pointer, so they must outlive interpreter_.
  // Declared before interpreter_ so they are destroyed after it.
  std::vector<uint8_t> weights_data_;
  std::vector<uint8_t> bias_data_;
  std::unique_ptr<::tflite::Interpreter> interpreter_;
  TestDelegateData test_data_;
  TfLiteDelegate delegate_;
  static constexpr int kWeightsTensorIndex = 1;
};

TEST_F(IrModelBuilderTest, PopulatesSharedConstTensorsMapForExternalBuffer) {
  constexpr int kGlobalId = 42;
  test_data_.external_buffer_map[kWeightsTensorIndex] = kGlobalId;

  VerifySharedWeightsTensor(kWeightsTensorIndex, kGlobalId);
}

TEST_F(IrModelBuilderTest, PopulatesSharedConstTensorsMapForInternalBuffer) {
  constexpr int kGlobalId = 42;
  test_data_.internal_buffer_map[kWeightsTensorIndex] = kGlobalId;

  VerifySharedWeightsTensor(kWeightsTensorIndex, kGlobalId);
}

TEST_F(IrModelBuilderTest, PrefersExternalWhenTensorInBothMaps) {
  constexpr int kInternalGlobalId = 42;
  constexpr int kExternalGlobalId = 99;
  test_data_.internal_buffer_map[kWeightsTensorIndex] = kInternalGlobalId;
  test_data_.external_buffer_map[kWeightsTensorIndex] = kExternalGlobalId;

  // The external map takes precedence (matching GF32's
  // ConstantInputSharingInfo::PreferredId()), so the tensor is treated as an
  // external shared constant using the external global id.
  VerifySharedWeightsTensor(kWeightsTensorIndex, kExternalGlobalId);
}

TEST_F(IrModelBuilderTest, DoesNotShareUnmappedConstant) {
  // Neither buffer map is seeded, so the constant weights tensor must not be
  // treated as a shared constant.
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  EXPECT_TRUE(test_data_.status.ok());
  EXPECT_TRUE(test_data_.shared_tensors.empty());

  for (const auto& tensor : test_data_.ir_model.tensors()) {
    EXPECT_FALSE(tensor->buffer_source.is_shared);
  }
}

TEST_F(IrModelBuilderTest, ForcesDequantForQuantizedSharedWeights) {
  constexpr int kGlobalId = 42;
  // Rebuild the model with int8 (affine-quantized) weights, which Conv2D
  // cannot consume when shared, so dequantization must be forced.
  BuildInterpreter(kTfLiteInt8, std::vector<uint8_t>(8 * 4, 0));
  test_data_.internal_buffer_map[kWeightsTensorIndex] = kGlobalId;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  EXPECT_TRUE(test_data_.status.ok());
  ASSERT_EQ(test_data_.shared_tensors.size(), 1);

  bool found_shared_tensor = false;
  for (const auto& tensor : test_data_.ir_model.tensors()) {
    if (tensor->buffer_source.is_shared) {
      found_shared_tensor = true;
      EXPECT_TRUE(test_data_.shared_tensors.at(tensor->id).dequant_forced);
    }
  }
  EXPECT_TRUE(found_shared_tensor);
}

TEST_F(IrModelBuilderTest, ForcesLinearLayoutForSharedBias) {
  constexpr int kBiasTensorIndex = 2;
  constexpr int kGlobalId = 7;
  // Only the bias is shared; a shared bias must be materialized with LINEAR
  // layout.
  BuildInterpreter(kTfLiteFloat32,
                   std::vector<uint8_t>(8 * 4 * sizeof(float), 0),
                   /*add_bias=*/true);
  test_data_.internal_buffer_map[kBiasTensorIndex] = kGlobalId;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  EXPECT_TRUE(test_data_.status.ok());
  ASSERT_EQ(test_data_.shared_tensors.size(), 1);

  bool found_bias = false;
  for (const auto& tensor : test_data_.ir_model.tensors()) {
    if (tensor->buffer_source.is_shared) {
      const auto& info = test_data_.shared_tensors.at(tensor->id);
      EXPECT_EQ(info.tflite_tensor_id, kBiasTensorIndex);
      found_bias = true;
      ASSERT_TRUE(info.layout.has_value());
      EXPECT_EQ(info.layout.value(), ::ml_drift::Layout::LINEAR);
      // The 1-D bias's 8 channels must live in the channel dim, not batch, so
      // the LINEAR reshape to (1,1,1,c) preserves them.
      EXPECT_EQ(tensor->desc.GetBHWCShape().b, 1);
      EXPECT_EQ(tensor->desc.GetBHWCShape().c, 8);
    }
  }
  EXPECT_TRUE(found_bias);
}

TEST_F(IrModelBuilderTest, ForcesLinearLayoutForSharedFullyConnectedBias) {
  constexpr int kBiasTensorIndex = 2;
  constexpr int kGlobalId = 9;
  // Only the bias is shared; a shared FullyConnected bias must be materialized
  // with LINEAR layout, just like a shared Conv2D bias.
  BuildFullyConnectedInterpreter();
  test_data_.internal_buffer_map[kBiasTensorIndex] = kGlobalId;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  EXPECT_TRUE(test_data_.status.ok());
  ASSERT_EQ(test_data_.shared_tensors.size(), 1);

  bool found_bias = false;
  for (const auto& tensor : test_data_.ir_model.tensors()) {
    if (tensor->buffer_source.is_shared) {
      const auto& info = test_data_.shared_tensors.at(tensor->id);
      EXPECT_EQ(info.tflite_tensor_id, kBiasTensorIndex);
      found_bias = true;
      ASSERT_TRUE(info.layout.has_value());
      EXPECT_EQ(info.layout.value(), ::ml_drift::Layout::LINEAR);
      EXPECT_EQ(tensor->desc.GetBHWCShape().b, 1);
      EXPECT_EQ(tensor->desc.GetBHWCShape().c, 8);
    }
  }
  EXPECT_TRUE(found_bias);
}

TEST_F(IrModelBuilderTest,
       ReducesSharedQuantizedPointwiseConvToFullyConnected) {
  constexpr int kGlobalId = 11;
  // A shared 1x1 int8 conv over a 1x1 input is reduced to a fully-connected op,
  // which (unlike Convolution2D) consumes the quantized weights directly, so
  // dequantization must NOT be forced.
  BuildPointwiseConvInterpreter(kTfLiteInt8, std::vector<uint8_t>(8 * 4, 0));
  test_data_.internal_buffer_map[kWeightsTensorIndex] = kGlobalId;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  EXPECT_TRUE(test_data_.status.ok());
  ASSERT_EQ(test_data_.shared_tensors.size(), 1);

  bool found_fc = false;
  bool found_conv = false;
  const ::ml_drift::ir::IrOp* fc_op = nullptr;
  for (const auto& op : test_data_.ir_model.ops()) {
    if (op->name == ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT8)) {
      found_fc = true;
      fc_op = op.get();
    }
    if (op->name == ToString(::ml_drift::OperationType::CONVOLUTION_2D)) {
      found_conv = true;
    }
  }
  EXPECT_TRUE(found_fc);
  EXPECT_FALSE(found_conv);
  ASSERT_NE(fc_op, nullptr);

  // The reduced op carries proper quantized attributes (weights + per-channel
  // scale shapes), mirroring GraphFloat32's
  // ConfigSharedWeightFullyConnectedNode.
  const auto& fc_attr =
      std::any_cast<const ::ml_drift::FullyConnectedInt8Attributes&>(
          fc_op->attr);
  EXPECT_EQ(fc_attr.weights.shape.o, 8);
  EXPECT_EQ(fc_attr.weights.shape.i, 4);
  EXPECT_EQ(fc_attr.scale.shape.o, 8);
  EXPECT_EQ(fc_attr.scale.shape.i, 1);

  bool found_shared_weights = false;
  for (const auto& tensor : test_data_.ir_model.tensors()) {
    if (tensor->buffer_source.is_shared) {
      found_shared_weights = true;
      // Quantized weights are kept quantized for the fully-connected kernel.
      EXPECT_FALSE(test_data_.shared_tensors.at(tensor->id).dequant_forced);
      // The shared weights are consumed as a runtime input of the FC op.
      EXPECT_TRUE(tensor->consumers.contains(fc_op->id));
    }
  }
  EXPECT_TRUE(found_shared_weights);
}

TEST_F(IrModelBuilderTest, ConfiguresSharedQuantizedFullyConnected) {
  constexpr int kWeightsIndex = 1;
  constexpr int kGlobalId = 13;
  // A shared int8 fully-connected weight is consumed quantized (no forced
  // dequantization) and emitted as FULLY_CONNECTED_INT8 with the weights and
  // per-channel scale shapes (parity with GraphFloat32).
  BuildFullyConnectedInterpreter(kTfLiteInt8, std::vector<uint8_t>(8 * 4, 0));
  test_data_.internal_buffer_map[kWeightsIndex] = kGlobalId;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  EXPECT_TRUE(test_data_.status.ok());
  ASSERT_EQ(test_data_.shared_tensors.size(), 1);

  const ::ml_drift::ir::IrOp* fc_op = nullptr;
  for (const auto& op : test_data_.ir_model.ops()) {
    if (op->name == ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT8)) {
      fc_op = op.get();
    }
  }
  ASSERT_NE(fc_op, nullptr);

  const auto& fc_attr =
      std::any_cast<const ::ml_drift::FullyConnectedInt8Attributes&>(
          fc_op->attr);
  EXPECT_EQ(fc_attr.weights.shape.o, 8);
  EXPECT_EQ(fc_attr.weights.shape.i, 4);
  EXPECT_EQ(fc_attr.scale.shape.o, 8);
  EXPECT_EQ(fc_attr.scale.shape.i, 1);

  bool weights_kept_quantized = false;
  for (const auto& tensor : test_data_.ir_model.tensors()) {
    if (tensor->buffer_source.is_shared) {
      weights_kept_quantized = true;
      EXPECT_FALSE(test_data_.shared_tensors.at(tensor->id).dequant_forced);
      EXPECT_TRUE(tensor->consumers.contains(fc_op->id));
    }
  }
  EXPECT_TRUE(weights_kept_quantized);
}

TEST_F(IrModelBuilderTest, ConfiguresSharedQuantizedEmbeddingLookup) {
  constexpr int kWeightsIndex = 1;
  constexpr int kGlobalId = 31;
  // A shared int8 embedding weight is passed as a runtime input; the op records
  // its type and per-row scale/zero-point shapes (parity with GraphFloat32's
  // shared EMBEDDING_LOOKUP branch), and does not embed the weight data.
  BuildEmbeddingLookupInterpreter(kTfLiteInt8, std::vector<uint8_t>(5 * 4, 0));
  test_data_.internal_buffer_map[kWeightsIndex] = kGlobalId;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  EXPECT_TRUE(test_data_.status.ok());
  ASSERT_EQ(test_data_.shared_tensors.size(), 1);

  const ::ml_drift::ir::IrOp* el_op = nullptr;
  for (const auto& op : test_data_.ir_model.ops()) {
    if (op->name == ToString(::ml_drift::OperationType::EMBEDDING_LOOKUP)) {
      el_op = op.get();
    }
  }
  ASSERT_NE(el_op, nullptr);

  const auto& attr =
      std::any_cast<const ::ml_drift::EmbeddingLookupAttributes&>(el_op->attr);
  EXPECT_EQ(attr.weights_type,
            ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt8);
  EXPECT_EQ(attr.original_weights_shape, ::ml_drift::OHWI(5, 1, 1, 4));
  EXPECT_EQ(attr.scale_zp_shape, ::ml_drift::OHWI(5, 1, 1, 1));

  bool found_shared_weights = false;
  for (const auto& tensor : test_data_.ir_model.tensors()) {
    if (tensor->buffer_source.is_shared) {
      found_shared_weights = true;
      // Shared weights are consumed as a runtime input of the embedding op.
      EXPECT_TRUE(tensor->consumers.contains(el_op->id));
    }
  }
  EXPECT_TRUE(found_shared_weights);
}

TEST_F(IrModelBuilderTest, ConfiguresSharedBlockwiseEmbeddingLookup) {
  constexpr int kWeightsIndex = 1;
  constexpr int kGlobalId = 41;
  constexpr int kBlockSize = 2;
  constexpr int kEmbeddingDim = 4;
  // A shared, blockwise-quantized int4 embedding weight is passed as a runtime
  // input. Blockwise quantization splits the embedding dim into blocks, so the
  // recorded per-block scale/zero-point shape has i = embedding_dim / blocksize
  // (parity with GraphFloat32's shared EMBEDDING_LOOKUP blockwise branch).
  // int4 weights pack two elements per byte: 5*4 = 20 elements -> 10 bytes.
  BuildEmbeddingLookupInterpreter(
      kTfLiteInt4, std::vector<uint8_t>(5 * kEmbeddingDim / 2, 0));

  // Replace the affine quantization applied by AddConstInput with blockwise
  // quantization. Only the blocksize is read on the shared path; the scale and
  // zero-point tensor indices are not dereferenced.
  TfLiteTensor* weights = interpreter_->tensor(kWeightsIndex);
  TfLiteQuantizationFree(&weights->quantization);
  auto* blockwise_params = static_cast<TfLiteBlockwiseQuantization*>(
      malloc(sizeof(TfLiteBlockwiseQuantization)));
  blockwise_params->scale = 0;
  blockwise_params->zero_point = -1;
  blockwise_params->blocksize = kBlockSize;
  weights->quantization.type = kTfLiteBlockwiseQuantization;
  weights->quantization.params = blockwise_params;

  test_data_.internal_buffer_map[kWeightsIndex] = kGlobalId;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  EXPECT_TRUE(test_data_.status.ok());
  ASSERT_EQ(test_data_.shared_tensors.size(), 1);

  const ::ml_drift::ir::IrOp* el_op = nullptr;
  for (const auto& op : test_data_.ir_model.ops()) {
    if (op->name == ToString(::ml_drift::OperationType::EMBEDDING_LOOKUP)) {
      el_op = op.get();
    }
  }
  ASSERT_NE(el_op, nullptr);

  const auto& attr =
      std::any_cast<const ::ml_drift::EmbeddingLookupAttributes&>(el_op->attr);
  EXPECT_EQ(attr.weights_type,
            ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt4);
  EXPECT_EQ(attr.original_weights_shape,
            ::ml_drift::OHWI(5, 1, 1, kEmbeddingDim));
  // Blockwise splits the embedding dim (4) into 4 / blocksize(2) = 2 blocks.
  EXPECT_EQ(attr.scale_zp_shape,
            ::ml_drift::OHWI(5, 1, 1, kEmbeddingDim / kBlockSize));

  bool found_shared_weights = false;
  for (const auto& tensor : test_data_.ir_model.tensors()) {
    if (tensor->buffer_source.is_shared) {
      found_shared_weights = true;
      // Shared weights are consumed as a runtime input of the embedding op.
      EXPECT_TRUE(tensor->consumers.contains(el_op->id));
    }
  }
  EXPECT_TRUE(found_shared_weights);
}

TEST_F(IrModelBuilderTest, ReducesSharedInt4PointwiseConvToFullyConnected) {
  constexpr int kGlobalId = 21;
  // int4 weights pack two elements per byte: 8*1*1*4 = 32 elements -> 16 bytes.
  BuildPointwiseConvInterpreter(kTfLiteInt4,
                                std::vector<uint8_t>(8 * 4 / 2, 0));
  test_data_.internal_buffer_map[kWeightsTensorIndex] = kGlobalId;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  EXPECT_TRUE(test_data_.status.ok());
  ASSERT_EQ(test_data_.shared_tensors.size(), 1);

  const ::ml_drift::ir::IrOp* fc_op = nullptr;
  bool found_conv = false;
  for (const auto& op : test_data_.ir_model.ops()) {
    if (op->name == ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT4)) {
      fc_op = op.get();
    }
    if (op->name == ToString(::ml_drift::OperationType::CONVOLUTION_2D)) {
      found_conv = true;
    }
  }
  EXPECT_FALSE(found_conv);
  ASSERT_NE(fc_op, nullptr);

  const auto& fc_attr =
      std::any_cast<const ::ml_drift::FullyConnectedInt4Attributes&>(
          fc_op->attr);
  const ::ml_drift::OHWI weights_shape =
      std::visit([](const auto& w) { return w.shape; }, fc_attr.weights);
  EXPECT_EQ(weights_shape.o, 8);
  EXPECT_EQ(weights_shape.i, 4);
  EXPECT_EQ(fc_attr.scale.shape.o, 8);

  for (const auto& tensor : test_data_.ir_model.tensors()) {
    if (tensor->buffer_source.is_shared) {
      EXPECT_FALSE(test_data_.shared_tensors.at(tensor->id).dequant_forced);
      EXPECT_TRUE(tensor->consumers.contains(fc_op->id));
    }
  }
}

TEST_F(IrModelBuilderTest, DoesNotReduceSharedInt2PointwiseConv) {
  constexpr int kGlobalId = 22;
  // GraphFloat32's IsConvertibleToFC only reduces int4/int8 shared convs, so an
  // int2 shared pointwise conv stays a convolution (with forced
  // dequantization). int2 weights pack four elements per byte: 8*1*1*4 = 32
  // elements -> 8 bytes.
  BuildPointwiseConvInterpreter(kTfLiteInt2,
                                std::vector<uint8_t>(8 * 4 / 4, 0));
  test_data_.internal_buffer_map[kWeightsTensorIndex] = kGlobalId;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  EXPECT_TRUE(test_data_.status.ok());
  ASSERT_EQ(test_data_.shared_tensors.size(), 1);

  bool found_conv = false;
  bool found_fc = false;
  for (const auto& op : test_data_.ir_model.ops()) {
    if (op->name == ToString(::ml_drift::OperationType::CONVOLUTION_2D)) {
      found_conv = true;
    }
    if (op->name == ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT2) ||
        op->name == ToString(::ml_drift::OperationType::FULLY_CONNECTED)) {
      found_fc = true;
    }
  }
  EXPECT_TRUE(found_conv);
  EXPECT_FALSE(found_fc);

  // The un-reduced convolution can't consume quantized shared weights, so they
  // are forced to dequantize (parity with GraphFloat32).
  bool found_shared_weights = false;
  for (const auto& tensor : test_data_.ir_model.tensors()) {
    if (tensor->buffer_source.is_shared) {
      found_shared_weights = true;
      EXPECT_TRUE(test_data_.shared_tensors.at(tensor->id).dequant_forced);
    }
  }
  EXPECT_TRUE(found_shared_weights);
}

}  // namespace
}  // namespace litert::ml_drift::ir
