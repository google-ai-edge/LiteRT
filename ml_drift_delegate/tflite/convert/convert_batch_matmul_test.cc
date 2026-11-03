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
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::SizeIs;

class ConvertBatchMatMulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        CreateStubDelegate(), DeleteStubDelegate);
  }

  const ::ml_drift::ir::IrModel* GetIrModelFromBuilder(
      SingleOpInterpreterBuilder& model_builder) {
    interpreter_ = model_builder.Build();
    if (!interpreter_) return nullptr;
    if (interpreter_->ModifyGraphWithDelegate(delegate_.get()) != kTfLiteOk) {
      return nullptr;
    }
    return GetIrModel(delegate_.get());
  }

  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_ = {
      nullptr, [](TfLiteDelegate*) {}};
  std::unique_ptr<::tflite::Interpreter> interpreter_;
};

TEST_F(ConvertBatchMatMulTest, ConvertToFullyConnected) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {1, 4});  // Input 0

  // Input 1 is constant and 2D: shape {4, 2}
  std::vector<float> weights_data(4 * 2, 1.0f);
  std::vector<uint8_t> weights_bytes(weights_data.size() * sizeof(float));
  std::memcpy(weights_bytes.data(), weights_data.data(), weights_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {4, 2}, weights_bytes);

  model.AddOutput(kTfLiteFloat32, {1, 2});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, ToString(::ml_drift::OperationType::kFullyConnected));
  EXPECT_THAT(op->inputs,
              SizeIs(1));  // FullyConnected expects 1 input in this system

  const ::ml_drift::FullyConnectedAttributes* attr =
      std::any_cast<::ml_drift::FullyConnectedAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->weights.shape.o, 2);
  EXPECT_EQ(attr->weights.shape.i, 4);
}

TEST_F(ConvertBatchMatMulTest, ConvertToFullyConnectedWithAdjY) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {1, 4});  // Input 0

  // Input 1 is constant and 2D: shape {2, 4} (adj_y=true means N=2, K=4)
  std::vector<float> weights_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<uint8_t> weights_bytes(weights_data.size() * sizeof(float));
  std::memcpy(weights_bytes.data(), weights_data.data(), weights_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {2, 4}, weights_bytes);

  model.AddOutput(kTfLiteFloat32, {1, 2});

  auto* params = reinterpret_cast<TfLiteBatchMatMulParams*>(
      calloc(1, sizeof(TfLiteBatchMatMulParams)));
  params->adj_x = false;
  params->adj_y = true;
  params->asymmetric_quantize_inputs = false;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, ToString(::ml_drift::OperationType::kFullyConnected));
  EXPECT_THAT(op->inputs, SizeIs(1));

  const ::ml_drift::FullyConnectedAttributes* attr =
      std::any_cast<::ml_drift::FullyConnectedAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->weights.shape.o, 2);
  EXPECT_EQ(attr->weights.shape.i, 4);
  for (size_t i = 0; i < weights_data.size(); ++i) {
    EXPECT_EQ(attr->weights.data[i], weights_data[i]);
  }
}

TEST_F(ConvertBatchMatMulTest, ConvertToBatchedMatMul) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {1, 2, 3});  // Input 0
  model.AddInput(kTfLiteFloat32, {1, 3, 4});  // Input 1
  model.AddOutput(kTfLiteFloat32, {1, 2, 4});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, ToString(::ml_drift::OperationType::kBatchedMatmul));
  EXPECT_THAT(op->inputs, SizeIs(2));
}

TEST_F(ConvertBatchMatMulTest, ConvertToBatchedMatMul5D) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {1, 64, 2, 49, 32});  // Input 0
  model.AddInput(kTfLiteFloat32, {1, 64, 2, 32, 49});  // Input 1
  model.AddOutput(kTfLiteFloat32, {1, 64, 2, 49, 49});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // We expect 4 operations: Reshape (left), Reshape (right), BatchedMatMul,
  // Reshape (output)
  ASSERT_THAT(ir_model->ops(), SizeIs(4));

  EXPECT_EQ(ir_model->op(0)->name,
            ToString(::ml_drift::OperationType::kReshape));
  EXPECT_EQ(ir_model->op(1)->name,
            ToString(::ml_drift::OperationType::kReshape));
  EXPECT_EQ(ir_model->op(2)->name,
            ToString(::ml_drift::OperationType::kBatchedMatmul));
  EXPECT_EQ(ir_model->op(3)->name,
            ToString(::ml_drift::OperationType::kReshape));

  // Check shapes after reshape
  const ::ml_drift::ir::IrOp* reshape_left_op = ir_model->op(0);
  ASSERT_THAT(reshape_left_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* reshaped_left_tensor =
      ir_model->tensor(reshape_left_op->outputs[0]);
  EXPECT_EQ(reshaped_left_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 128, 49, 1, 32));

  const ::ml_drift::ir::IrOp* reshape_right_op = ir_model->op(1);
  ASSERT_THAT(reshape_right_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* reshaped_right_tensor =
      ir_model->tensor(reshape_right_op->outputs[0]);
  EXPECT_EQ(reshaped_right_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 128, 32, 1, 49));

  const ::ml_drift::ir::IrOp* bmm_op = ir_model->op(2);
  ASSERT_THAT(bmm_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* bmm_output_tensor =
      ir_model->tensor(bmm_op->outputs[0]);
  EXPECT_EQ(bmm_output_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 128, 49, 1, 49));

  const ::ml_drift::ir::IrOp* reshape_result_op = ir_model->op(3);
  ASSERT_THAT(reshape_result_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* final_output_tensor =
      ir_model->tensor(reshape_result_op->outputs[0]);
  EXPECT_EQ(final_output_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 64, 2, 49, 49));
}

TEST_F(ConvertBatchMatMulTest, ConvertToBatchedMatMul4DReshape) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {2, 4, 128, 32});  // Input 0
  model.AddInput(kTfLiteFloat32, {2, 4, 32, 64});   // Input 1
  model.AddOutput(kTfLiteFloat32, {2, 4, 128, 64});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // We expect 4 operations: Reshape (left), Reshape (right), BatchedMatMul,
  // Reshape (output)
  ASSERT_THAT(ir_model->ops(), SizeIs(4));

  EXPECT_EQ(ir_model->op(0)->name,
            ToString(::ml_drift::OperationType::kReshape));
  EXPECT_EQ(ir_model->op(1)->name,
            ToString(::ml_drift::OperationType::kReshape));
  EXPECT_EQ(ir_model->op(2)->name,
            ToString(::ml_drift::OperationType::kBatchedMatmul));
  EXPECT_EQ(ir_model->op(3)->name,
            ToString(::ml_drift::OperationType::kReshape));

  // Check shapes after reshape
  const ::ml_drift::ir::IrOp* reshape_left_op = ir_model->op(0);
  ASSERT_THAT(reshape_left_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* reshaped_left_tensor =
      ir_model->tensor(reshape_left_op->outputs[0]);
  EXPECT_EQ(reshaped_left_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 8, 128, 1, 32));

  const ::ml_drift::ir::IrOp* reshape_right_op = ir_model->op(1);
  ASSERT_THAT(reshape_right_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* reshaped_right_tensor =
      ir_model->tensor(reshape_right_op->outputs[0]);
  EXPECT_EQ(reshaped_right_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 8, 32, 1, 64));

  const ::ml_drift::ir::IrOp* bmm_op = ir_model->op(2);
  ASSERT_THAT(bmm_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* bmm_output_tensor =
      ir_model->tensor(bmm_op->outputs[0]);
  EXPECT_EQ(bmm_output_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 8, 128, 1, 64));

  const ::ml_drift::ir::IrOp* reshape_result_op = ir_model->op(3);
  ASSERT_THAT(reshape_result_op->outputs, SizeIs(1));
  const ::ml_drift::ir::IrTensor* final_output_tensor =
      ir_model->tensor(reshape_result_op->outputs[0]);
  EXPECT_EQ(final_output_tensor->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(2, 4, 128, 1, 64));
}

TEST_F(ConvertBatchMatMulTest, ConvertToFullyConnectedWithAdjX2D) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {4, 3});  // [K=4, M=3] with adj_x=true

  // Constant 2D RHS [K=4, N=2]:
  // [[1, 2], [3, 4], [5, 6], [7, 8]] -> transposed OHWI(2, 1, 1, 4):
  // [1, 3, 5, 7, 2, 4, 6, 8].
  std::vector<float> weights_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<uint8_t> weights_bytes(weights_data.size() * sizeof(float));
  std::memcpy(weights_bytes.data(), weights_data.data(), weights_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {4, 2}, weights_bytes);
  model.AddOutput(kTfLiteFloat32, {3, 2});

  auto* params = reinterpret_cast<TfLiteBatchMatMulParams*>(
      calloc(1, sizeof(TfLiteBatchMatMulParams)));
  params->adj_x = true;
  params->adj_y = false;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(2));

  const ::ml_drift::ir::IrOp* transpose_op = ir_model->op(0);
  EXPECT_EQ(transpose_op->name,
            ToString(::ml_drift::OperationType::kTranspose));
  const auto* transpose_attr =
      std::any_cast<::ml_drift::TransposeAttributes>(&transpose_op->attr);
  ASSERT_TRUE(transpose_attr);
  EXPECT_EQ(transpose_attr->perm, ::ml_drift::BHWC(3, 1, 2, 0));
  ASSERT_THAT(transpose_op->outputs, SizeIs(1));
  EXPECT_EQ(ir_model->tensor(transpose_op->outputs[0])->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(3, 1, 1, 1, 4));

  const ::ml_drift::ir::IrOp* fc_op = ir_model->op(1);
  EXPECT_EQ(fc_op->name, ToString(::ml_drift::OperationType::kFullyConnected));
  const auto* fc_attr =
      std::any_cast<::ml_drift::FullyConnectedAttributes>(&fc_op->attr);
  ASSERT_TRUE(fc_attr);
  EXPECT_EQ(fc_attr->weights.shape, ::ml_drift::OHWI(2, 1, 1, 4));
  const std::vector<float> expected_weights = {1.0f, 3.0f, 5.0f, 7.0f,
                                               2.0f, 4.0f, 6.0f, 8.0f};
  ASSERT_GE(fc_attr->weights.data.size(), expected_weights.size());
  for (size_t i = 0; i < expected_weights.size(); ++i) {
    EXPECT_EQ(fc_attr->weights.data[i], expected_weights[i]);
  }
}

TEST_F(ConvertBatchMatMulTest, ConvertToFullyConnectedWithAdjXAndAdjY3D) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {2, 4, 3});  // [B=2, K=4, M=3]

  std::vector<float> weights_data(5 * 4);
  for (size_t i = 0; i < weights_data.size(); ++i) {
    weights_data[i] = static_cast<float>(i + 1);
  }
  std::vector<uint8_t> weights_bytes(weights_data.size() * sizeof(float));
  std::memcpy(weights_bytes.data(), weights_data.data(), weights_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {5, 4}, weights_bytes);
  model.AddOutput(kTfLiteFloat32, {2, 3, 5});

  auto* params = reinterpret_cast<TfLiteBatchMatMulParams*>(
      calloc(1, sizeof(TfLiteBatchMatMulParams)));
  params->adj_x = true;
  params->adj_y = true;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(2));

  const ::ml_drift::ir::IrOp* transpose_op = ir_model->op(0);
  EXPECT_EQ(transpose_op->name,
            ToString(::ml_drift::OperationType::kTranspose));
  const auto* transpose_attr =
      std::any_cast<::ml_drift::TransposeAttributes>(&transpose_op->attr);
  ASSERT_TRUE(transpose_attr);
  EXPECT_EQ(transpose_attr->perm, ::ml_drift::BHWC(0, 1, 3, 2));
  ASSERT_THAT(transpose_op->outputs, SizeIs(1));
  EXPECT_EQ(ir_model->tensor(transpose_op->outputs[0])->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(2, 1, 3, 1, 4));

  const ::ml_drift::ir::IrOp* fc_op = ir_model->op(1);
  EXPECT_EQ(fc_op->name, ToString(::ml_drift::OperationType::kFullyConnected));
  const auto* fc_attr =
      std::any_cast<::ml_drift::FullyConnectedAttributes>(&fc_op->attr);
  ASSERT_TRUE(fc_attr);
  EXPECT_EQ(fc_attr->weights.shape, ::ml_drift::OHWI(5, 1, 1, 4));
  ASSERT_GE(fc_attr->weights.data.size(), weights_data.size());
  for (size_t i = 0; i < weights_data.size(); ++i) {
    EXPECT_EQ(fc_attr->weights.data[i], weights_data[i]);
  }
}

TEST_F(ConvertBatchMatMulTest, ConvertToFullyConnected5DWithAdjXAndAdjY) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  // 5D input with trailing M=1 tests 5D detection when d=1 in BHWDC.
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4, 1});

  std::vector<float> weights_data(5 * 4, 0.5f);
  std::vector<uint8_t> weights_bytes(weights_data.size() * sizeof(float));
  std::memcpy(weights_bytes.data(), weights_data.data(), weights_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {5, 4}, weights_bytes);
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 1, 5});

  auto* params = reinterpret_cast<TfLiteBatchMatMulParams*>(
      calloc(1, sizeof(TfLiteBatchMatMulParams)));
  params->adj_x = true;
  params->adj_y = true;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(4));
  EXPECT_EQ(ir_model->op(0)->name,
            ToString(::ml_drift::OperationType::kReshape));
  EXPECT_EQ(ir_model->op(1)->name,
            ToString(::ml_drift::OperationType::kTranspose));
  EXPECT_EQ(ir_model->op(2)->name,
            ToString(::ml_drift::OperationType::kFullyConnected));
  EXPECT_EQ(ir_model->op(3)->name,
            ToString(::ml_drift::OperationType::kReshape));
}

TEST_F(ConvertBatchMatMulTest, ConvertToBatchedMatMulWithAdjXAndAdjY) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {1, 4, 3});  // [B=1, K=4, M=3]
  model.AddInput(kTfLiteFloat32, {1, 2, 4});  // [B=1, N=2, K=4]
  model.AddOutput(kTfLiteFloat32, {1, 3, 2});

  auto* params = reinterpret_cast<TfLiteBatchMatMulParams*>(
      calloc(1, sizeof(TfLiteBatchMatMulParams)));
  params->adj_x = true;
  params->adj_y = true;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(1));

  const ::ml_drift::ir::IrOp* bmm_op = ir_model->op(0);
  EXPECT_EQ(bmm_op->name, ToString(::ml_drift::OperationType::kBatchedMatmul));
  const auto* attr =
      std::any_cast<::ml_drift::BatchedMatMulAttributes>(&bmm_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_TRUE(attr->transpose_left);
  EXPECT_TRUE(attr->transpose_right);
}

TEST_F(ConvertBatchMatMulTest, ConvertToBatchedMatMul3DSingleRowBatchReshape) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
  model.AddInput(kTfLiteFloat32, {2, 1, 4});  // 3D with M=1 and B=2
  model.AddInput(kTfLiteFloat32, {2, 4, 3});
  model.AddOutput(kTfLiteFloat32, {2, 1, 3});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);
  ASSERT_THAT(ir_model->ops(), SizeIs(4));

  const ::ml_drift::ir::IrOp* reshape_left_op = ir_model->op(0);
  ASSERT_THAT(reshape_left_op->outputs, SizeIs(1));
  EXPECT_EQ(ir_model->tensor(reshape_left_op->outputs[0])->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 2, 1, 1, 4));

  const ::ml_drift::ir::IrOp* bmm_op = ir_model->op(2);
  ASSERT_THAT(bmm_op->outputs, SizeIs(1));
  EXPECT_EQ(ir_model->tensor(bmm_op->outputs[0])->desc.GetBHWDCShape(),
            ::ml_drift::BHWDC(1, 2, 1, 1, 3));
}

TEST_F(ConvertBatchMatMulTest,
       QuantizedSyntheticIntermediateTensorsOmitQuantParams) {
  auto quant_delegate =
      std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
          CreateStubDelegate({.allow_quant_ops = true}), DeleteStubDelegate);

  // 1. 5D quantized FullyConnected path with adj_x=true (Reshape + Transpose +
  // FullyConnected + Reshape).
  {
    SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
    model.AddInput(kTfLiteInt8, {1, 2, 3, 4, 2});
    std::vector<float> weights_data(5 * 4, 0.5f);
    std::vector<uint8_t> weights_bytes(weights_data.size() * sizeof(float));
    std::memcpy(weights_bytes.data(), weights_data.data(),
                weights_bytes.size());
    model.AddConstInput(kTfLiteFloat32, {5, 4}, weights_bytes);
    model.AddOutput(kTfLiteInt8, {1, 2, 3, 2, 5});

    auto* params = reinterpret_cast<TfLiteBatchMatMulParams*>(
        calloc(1, sizeof(TfLiteBatchMatMulParams)));
    params->adj_x = true;
    params->adj_y = true;
    model.SetParameters(params);

    auto interp = model.Build();
    ASSERT_NE(interp, nullptr);
    ASSERT_EQ(interp->ModifyGraphWithDelegate(quant_delegate.get()), kTfLiteOk);
    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(quant_delegate.get());
    ASSERT_NE(ir_model, nullptr);
    ASSERT_THAT(ir_model->ops(), SizeIs(4));
    // Boundary I/O tensors retain quant_params, while synthetic intermediate
    // tensors (outputs of ops 0, 1, 2) must omit quant_params.
    EXPECT_TRUE(
        ir_model->tensor(ir_model->inputs()[0])->quant_params.has_value());
    EXPECT_TRUE(
        ir_model->tensor(ir_model->outputs()[0])->quant_params.has_value());
    for (int op_idx : {0, 1, 2}) {
      ASSERT_THAT(ir_model->op(op_idx)->outputs, SizeIs(1));
      EXPECT_FALSE(ir_model->tensor(ir_model->op(op_idx)->outputs[0])
                       ->quant_params.has_value());
    }
  }

  // 2. 4D quantized BatchedMatmul path with b != 1 (Reshape + Reshape +
  // BatchedMatmul + Reshape).
  quant_delegate.reset(CreateStubDelegate({.allow_quant_ops = true}));
  {
    SingleOpInterpreterBuilder model(kTfLiteBuiltinBatchMatmul);
    model.AddInput(kTfLiteInt8, {2, 2, 3, 4});
    model.AddInput(kTfLiteInt8, {2, 2, 4, 5});
    model.AddOutput(kTfLiteInt8, {2, 2, 3, 5});

    auto interp = model.Build();
    ASSERT_NE(interp, nullptr);
    ASSERT_EQ(interp->ModifyGraphWithDelegate(quant_delegate.get()), kTfLiteOk);
    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(quant_delegate.get());
    ASSERT_NE(ir_model, nullptr);
    ASSERT_THAT(ir_model->ops(), SizeIs(4));
    EXPECT_TRUE(
        ir_model->tensor(ir_model->inputs()[0])->quant_params.has_value());
    EXPECT_TRUE(
        ir_model->tensor(ir_model->inputs()[1])->quant_params.has_value());
    EXPECT_TRUE(
        ir_model->tensor(ir_model->outputs()[0])->quant_params.has_value());
    for (int op_idx : {0, 1, 2}) {
      ASSERT_THAT(ir_model->op(op_idx)->outputs, SizeIs(1));
      EXPECT_FALSE(ir_model->tensor(ir_model->op(op_idx)->outputs[0])
                       ->quant_params.has_value());
    }
  }
}

}  // namespace
}  // namespace litert::ml_drift::ir
