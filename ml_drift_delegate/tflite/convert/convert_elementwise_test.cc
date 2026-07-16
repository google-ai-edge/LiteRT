// Copyright 2025 Google LLC.
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

#include "ml_drift_delegate/tflite/convert/convert_elementwise.h"

#include <any>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ElementwiseParam = std::tuple<TfLiteBuiltinOperator, TfLiteType>;

class ConvertElementwiseTest : public ::testing::Test {
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

class UnaryElementwiseTest
    : public ConvertElementwiseTest,
      public ::testing::WithParamInterface<ElementwiseParam> {};

TEST_P(UnaryElementwiseTest, Basic) {
  auto [op_code, data_type] = GetParam();
  SingleOpInterpreterBuilder model(op_code);
  model.AddInput(data_type, {1, 2, 3, 4});
  model.AddOutput(data_type, {1, 2, 3, 4});

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  std::string expected_name =
      ::ml_drift::ToString(GetElementwiseOperationType(op_code));
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  ASSERT_TRUE(op);
  EXPECT_EQ(op->name, expected_name);
  EXPECT_EQ(op->inputs.size(), 1);
  EXPECT_EQ(op->outputs.size(), 1);
}

INSTANTIATE_TEST_SUITE_P(
    UnaryElementwise, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(
            // go/keep-sorted start numeric=yes
            kTfLiteBuiltinAbs, kTfLiteBuiltinCast, kTfLiteBuiltinCeil,
            kTfLiteBuiltinCos, kTfLiteBuiltinExp, kTfLiteBuiltinFloor,
            kTfLiteBuiltinGelu, kTfLiteBuiltinHardSwish, kTfLiteBuiltinLog,
            kTfLiteBuiltinLogicalNot, kTfLiteBuiltinLogistic, kTfLiteBuiltinNeg,
            kTfLiteBuiltinRound, kTfLiteBuiltinRsqrt, kTfLiteBuiltinSin,
            kTfLiteBuiltinSign, kTfLiteBuiltinSqrt, kTfLiteBuiltinSquare,
            kTfLiteBuiltinTanh, kTfLiteBuiltinElu
            // go/keep-sorted end
            ),
        ::testing::Values(
            // go/keep-sorted start numeric=yes
            kTfLiteBFloat16, kTfLiteFloat16, kTfLiteFloat32, kTfLiteInt4,
            kTfLiteInt8, kTfLiteInt16, kTfLiteInt32, kTfLiteUInt8,
            kTfLiteUInt16, kTfLiteUInt32
            // go/keep-sorted end
            )));

class BinaryElementwiseTest
    : public ConvertElementwiseTest,
      public ::testing::WithParamInterface<ElementwiseParam> {};

TEST_P(BinaryElementwiseTest, Basic) {
  auto [op_code, data_type] = GetParam();
  SingleOpInterpreterBuilder model(op_code);
  model.AddInput(data_type, {1, 2, 3, 4});
  model.AddInput(data_type, {1, 2, 3, 4});
  TfLiteType output_type = data_type;
  if (op_code == kTfLiteBuiltinEqual || op_code == kTfLiteBuiltinNotEqual ||
      op_code == kTfLiteBuiltinGreater ||
      op_code == kTfLiteBuiltinGreaterEqual || op_code == kTfLiteBuiltinLess ||
      op_code == kTfLiteBuiltinLessEqual ||
      op_code == kTfLiteBuiltinLogicalAnd ||
      op_code == kTfLiteBuiltinLogicalOr ||
      // TODO: BitwiseXor is not supported yet
      op_code == kTfLiteBuiltinBitwiseXor) {
    output_type = kTfLiteBool;
  }
  model.AddOutput(output_type, {1, 2, 3, 4});

  void* params = nullptr;
  if (op_code == kTfLiteBuiltinAdd) {
    params = calloc(1, sizeof(TfLiteAddParams));
  } else if (op_code == kTfLiteBuiltinSub) {
    params = calloc(1, sizeof(TfLiteSubParams));
  } else if (op_code == kTfLiteBuiltinMul) {
    params = calloc(1, sizeof(TfLiteMulParams));
  } else if (op_code == kTfLiteBuiltinDiv) {
    params = calloc(1, sizeof(TfLiteDivParams));
  }
  if (params) model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  std::string expected_name =
      ::ml_drift::ToString(GetElementwiseOperationType(op_code));
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  ASSERT_TRUE(op);
  EXPECT_EQ(op->name, expected_name);
  EXPECT_EQ(op->inputs.size(), 2);
}

INSTANTIATE_TEST_SUITE_P(
    BinaryElementwise, BinaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(
            // go/keep-sorted start numeric=yes
            kTfLiteBuiltinAdd, kTfLiteBuiltinBitwiseXor, kTfLiteBuiltinAtan2,
            kTfLiteBuiltinDiv, kTfLiteBuiltinEqual, kTfLiteBuiltinFloorDiv,
            kTfLiteBuiltinFloorMod, kTfLiteBuiltinStablehloRemainder,
            kTfLiteBuiltinGreater, kTfLiteBuiltinGreaterEqual,
            kTfLiteBuiltinLess, kTfLiteBuiltinLessEqual,
            kTfLiteBuiltinLogicalAnd, kTfLiteBuiltinLogicalOr,
            kTfLiteBuiltinMaximum, kTfLiteBuiltinMinimum, kTfLiteBuiltinMul,
            kTfLiteBuiltinNotEqual, kTfLiteBuiltinPow,
            kTfLiteBuiltinSquaredDifference, kTfLiteBuiltinRightShift,
            kTfLiteBuiltinStablehloShiftLeft, kTfLiteBuiltinSub
            // go/keep-sorted end
            ),
        ::testing::Values(
            // go/keep-sorted start numeric=yes
            kTfLiteBFloat16, kTfLiteFloat16, kTfLiteFloat32, kTfLiteInt4,
            kTfLiteInt8, kTfLiteInt16, kTfLiteInt32, kTfLiteUInt8,
            kTfLiteUInt16, kTfLiteUInt32
            // go/keep-sorted end
            )));

TEST_F(ConvertElementwiseTest, Broadcast) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddInput(kTfLiteFloat32, {3, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // Broadcast branch creates 2 reshapes and the add op, then another reshape
  // for the output.
  ASSERT_EQ(ir_model->ops().size(), 4);
  EXPECT_EQ(ir_model->op(0)->name, "reshape");
  EXPECT_EQ(ir_model->op(1)->name, "reshape");
  EXPECT_EQ(ir_model->op(2)->name, "add");
  EXPECT_EQ(ir_model->op(3)->name, "reshape");
}

TEST_F(ConvertElementwiseTest, BroadcastWithActivation) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddInput(kTfLiteFloat32, {3, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActRelu;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // Expect 5 ops: 2 input reshapes, the add op, the relu op, and the output
  // reshape.
  ASSERT_EQ(ir_model->ops().size(), 5);
  EXPECT_EQ(ir_model->op(0)->name, "reshape");
  EXPECT_EQ(ir_model->op(1)->name, "reshape");
  EXPECT_EQ(ir_model->op(2)->name, "add");
  EXPECT_EQ(ir_model->op(3)->name, "relu");
  EXPECT_EQ(ir_model->op(4)->name, "reshape");

  // Verify that relu consumes the output of add and reshape consumes the output
  // of relu.
  const ::ml_drift::ir::IrOp* add_op = ir_model->op(2);
  const ::ml_drift::ir::IrOp* relu_op = ir_model->op(3);
  const ::ml_drift::ir::IrOp* reshape_op = ir_model->op(4);

  // Find the tensor connecting add and relu.
  bool add_relu_connected = false;
  for (auto out_id : add_op->outputs) {
    for (auto in_id : relu_op->inputs) {
      if (out_id == in_id) {
        add_relu_connected = true;
        break;
      }
    }
  }
  EXPECT_TRUE(add_relu_connected);

  // Find the tensor connecting relu and reshape.
  bool relu_reshape_connected = false;
  for (auto out_id : relu_op->outputs) {
    for (auto in_id : reshape_op->inputs) {
      if (out_id == in_id) {
        relu_reshape_connected = true;
        break;
      }
    }
  }
  EXPECT_TRUE(relu_reshape_connected);
}

TEST_F(ConvertElementwiseTest, IdenticalInputsAdd) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  // Use the same input tensor twice.
  model.AddInputWithId(0);
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // ADD(A, A) -> MUL(A, 2.0)
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  ASSERT_TRUE(op);
  EXPECT_EQ(op->inputs.size(), 1);
  const ::ml_drift::ElementwiseAttributes* attr =
      std::any_cast<::ml_drift::ElementwiseAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* scalar_val = std::get_if<::ml_drift::ScalarValue>(&attr->param);
  ASSERT_TRUE(scalar_val);
  const float* p = std::get_if<float>(scalar_val);
  ASSERT_TRUE(p);
  EXPECT_EQ(*p, 2.0f);
}

TEST_F(ConvertElementwiseTest, IdenticalInputsMul) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinMul);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  // Use the same input tensor twice.
  model.AddInputWithId(0);
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteMulParams* params =
      reinterpret_cast<TfLiteMulParams*>(calloc(1, sizeof(TfLiteMulParams)));
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // MUL(A, A) -> SQUARE(A)
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  ASSERT_TRUE(op);
  EXPECT_EQ(op->inputs.size(), 1);
}

TEST_F(ConvertElementwiseTest, ConstantInputScalar) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  std::vector<float> const_data = {3.14f};
  std::vector<uint8_t> const_data_bytes(const_data.size() * sizeof(float));
  std::memcpy(const_data_bytes.data(), const_data.data(),
              const_data_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {}, const_data_bytes);
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // Expect 1 op: add (constant is folded into attributes).
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "add");

  const ::ml_drift::ElementwiseAttributes* attr =
      std::any_cast<::ml_drift::ElementwiseAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* scalar_val = std::get_if<::ml_drift::ScalarValue>(&attr->param);
  ASSERT_TRUE(scalar_val);
  const float* p = std::get_if<float>(scalar_val);
  ASSERT_TRUE(p);
  EXPECT_EQ(*p, 3.14f);
  EXPECT_TRUE(attr->runtime_tensor_is_second == false);
}

TEST_F(ConvertElementwiseTest, ConstantInputInt32Scalar) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd);
  model.AddInput(kTfLiteInt32, {1, 2, 3, 4});
  std::vector<int32_t> const_data = {42};
  std::vector<uint8_t> const_data_bytes(const_data.size() * sizeof(int32_t));
  std::memcpy(const_data_bytes.data(), const_data.data(),
              const_data_bytes.size());
  model.AddConstInput(kTfLiteInt32, {}, const_data_bytes);
  model.AddOutput(kTfLiteInt32, {1, 2, 3, 4});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // Expect 1 op: add (constant is folded into attributes).
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "add");

  const ::ml_drift::ElementwiseAttributes* attr =
      std::any_cast<::ml_drift::ElementwiseAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* scalar_val = std::get_if<::ml_drift::ScalarValue>(&attr->param);
  ASSERT_TRUE(scalar_val);
  const int* p = std::get_if<int>(scalar_val);
  ASSERT_TRUE(p);
  EXPECT_EQ(*p, 42);
}

TEST_F(ConvertElementwiseTest, ConstantInputLinear) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  std::vector<float> const_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<uint8_t> const_data_bytes(const_data.size() * sizeof(float));
  std::memcpy(const_data_bytes.data(), const_data.data(),
              const_data_bytes.size());
  model.AddConstInput(kTfLiteFloat32, {4}, const_data_bytes);
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // Expect 1 op: add (constant is folded into attributes).
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "add");

  const ::ml_drift::ElementwiseAttributes* attr =
      std::any_cast<::ml_drift::ElementwiseAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* t = std::get_if<
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>>(
      &attr->param);
  ASSERT_TRUE(t);
  EXPECT_EQ(t->shape.v, 4);
}

TEST_F(ConvertElementwiseTest, ConstantInputBHWC) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  std::vector<float> const_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<uint8_t> const_data_bytes(const_data.size() * sizeof(float));
  std::memcpy(const_data_bytes.data(), const_data.data(),
              const_data_bytes.size());
  // [1, 1, 3, 4] -> non-linear (2 dims != 1), and broadcastable (1->2, 1->1
  // match).
  std::vector<uint8_t> const_data_bytes_long(12 * sizeof(float), 0);
  model.AddConstInput(kTfLiteFloat32, {1, 1, 3, 4}, const_data_bytes_long);
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // Expect 1 op: add (constant is folded into attributes).
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "add");

  const ::ml_drift::ElementwiseAttributes* attr =
      std::any_cast<::ml_drift::ElementwiseAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* t = std::get_if<
      ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>>(
      &attr->param);
  ASSERT_TRUE(t);
  EXPECT_EQ(t->shape, ::ml_drift::BHWC(1, 1, 3, 4));
}

TEST_F(ConvertElementwiseTest, ConstantInputNotConvertible) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd);
  model.AddInput(kTfLiteInt32, {1, 2, 3, 4});
  std::vector<int32_t> const_data = {1, 2, 3, 4};
  std::vector<uint8_t> const_data_bytes(const_data.size() * sizeof(int32_t));
  std::memcpy(const_data_bytes.data(), const_data.data(),
              const_data_bytes.size());
  // Int32 constant with rank 1 is not convertible to f32 or scalar int32 in
  // ParseInputsWithConstTensor.
  model.AddConstInput(kTfLiteInt32, {4}, const_data_bytes);
  model.AddOutput(kTfLiteInt32, {1, 2, 3, 4});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // Expect 2 ops: add and const.
  ASSERT_EQ(ir_model->ops().size(), 2);
  EXPECT_EQ(ir_model->op(0)->name, "add");
  EXPECT_EQ(ir_model->op(1)->name, "const");
}

TEST_F(ConvertElementwiseTest, FusedActivation) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActRelu;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  // Expect 2 ops: add and relu.
  ASSERT_EQ(ir_model->ops().size(), 2);
  EXPECT_EQ(ir_model->op(0)->name, "add");
  EXPECT_EQ(ir_model->op(1)->name, "relu");
}

TEST_F(ConvertElementwiseTest, FusedActivationMul) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinMul);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteMulParams* params =
      reinterpret_cast<TfLiteMulParams*>(calloc(1, sizeof(TfLiteMulParams)));
  params->activation = kTfLiteActRelu;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 2);
  EXPECT_EQ(ir_model->op(0)->name, "mul");
  EXPECT_EQ(ir_model->op(1)->name, "relu");
}

TEST_F(ConvertElementwiseTest, FusedActivationSub) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinSub);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteSubParams* params =
      reinterpret_cast<TfLiteSubParams*>(calloc(1, sizeof(TfLiteSubParams)));
  params->activation = kTfLiteActRelu;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 2);
  EXPECT_EQ(ir_model->op(0)->name, "subtract");
  EXPECT_EQ(ir_model->op(1)->name, "relu");
}

TEST_F(ConvertElementwiseTest, FusedActivationDiv) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinDiv);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteDivParams* params =
      reinterpret_cast<TfLiteDivParams*>(calloc(1, sizeof(TfLiteDivParams)));
  params->activation = kTfLiteActRelu;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  ASSERT_EQ(ir_model->ops().size(), 2);
  EXPECT_EQ(ir_model->op(0)->name, "div");
  EXPECT_EQ(ir_model->op(1)->name, "relu");
}

TEST_F(ConvertElementwiseTest, GeluApprox) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinGelu);
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});

  TfLiteGeluParams* params =
      reinterpret_cast<TfLiteGeluParams*>(calloc(1, sizeof(TfLiteGeluParams)));
  params->approximate = true;
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  EXPECT_EQ(op->name, "gelu_tanh_approx");
}

TEST_F(ConvertElementwiseTest, ConstantInputPosition) {
  {
    // Constant first: Sub(Const, Runtime) -> runtime_tensor_is_second = true
    SingleOpInterpreterBuilder model(kTfLiteBuiltinSub);
    std::vector<float> const_data = {3.14f};
    std::vector<uint8_t> const_data_bytes(sizeof(float));
    std::memcpy(const_data_bytes.data(), const_data.data(), sizeof(float));
    model.AddConstInput(kTfLiteFloat32, {}, const_data_bytes);
    model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
    model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});
    model.SetParameters(calloc(1, sizeof(TfLiteSubParams)));

    const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
    ASSERT_TRUE(ir_model);
    const auto* attr = std::any_cast<::ml_drift::ElementwiseAttributes>(
        &ir_model->op(0)->attr);
    EXPECT_TRUE(attr->runtime_tensor_is_second);
  }
  {
    // Constant second: Sub(Runtime, Const) -> runtime_tensor_is_second = false
    SingleOpInterpreterBuilder model(kTfLiteBuiltinSub);
    model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
    std::vector<float> const_data = {3.14f};
    std::vector<uint8_t> const_data_bytes(sizeof(float));
    std::memcpy(const_data_bytes.data(), const_data.data(), sizeof(float));
    model.AddConstInput(kTfLiteFloat32, {}, const_data_bytes);
    model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});
    model.SetParameters(calloc(1, sizeof(TfLiteSubParams)));

    const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
    ASSERT_TRUE(ir_model);
    const auto* attr = std::any_cast<::ml_drift::ElementwiseAttributes>(
        &ir_model->op(0)->attr);
    EXPECT_FALSE(attr->runtime_tensor_is_second);
  }
}

TEST_F(ConvertElementwiseTest, SwapInputs) {
  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd);
  // Input 0: [1, 1, 1, 4] (smaller)
  // Input 1: [1, 2, 3, 4] (larger)
  model.AddInput(kTfLiteFloat32, {1, 1, 1, 4});
  model.AddInput(kTfLiteFloat32, {1, 2, 3, 4});
  model.AddOutput(kTfLiteFloat32, {1, 2, 3, 4});
  model.SetParameters(calloc(1, sizeof(TfLiteAddParams)));

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);

  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  ASSERT_EQ(op->inputs.size(), 2);
  // After SwapInputs, the larger tensor (Input 1) should be first.
  EXPECT_EQ(ir_model->tensor(op->inputs[0])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, 2, 3, 4));
  EXPECT_EQ(ir_model->tensor(op->inputs[1])->desc.GetBHWCShape(),
            ::ml_drift::BHWC(1, 1, 1, 4));
}

TEST_F(ConvertElementwiseTest, ConstantInput2D) {
  std::vector<float> data = {1, 2,  3,  4,  5,  6,  7,  8,
                             9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint8_t> data_bytes(data.size() * sizeof(float));
  std::memcpy(data_bytes.data(), data.data(), data_bytes.size());

  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd, /*version=*/1);
  model.AddInput(kTfLiteFloat32, {1, 4, 4, 1});
  model.AddConstInput(kTfLiteFloat32, {4, 4}, data_bytes);
  model.AddOutput(kTfLiteFloat32, {1, 4, 4, 1});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  const ::ml_drift::ElementwiseAttributes* attr =
      std::any_cast<::ml_drift::ElementwiseAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* t = std::get_if<
      ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>>(
      &attr->param);
  ASSERT_TRUE(t);
  EXPECT_EQ(t->shape, ::ml_drift::BHWC(1, 1, 4, 4));
}

TEST_F(ConvertElementwiseTest, ConstantInput3D) {
  std::vector<float> data(48, 1.0f);
  std::vector<uint8_t> data_bytes(data.size() * sizeof(float));
  std::memcpy(data_bytes.data(), data.data(), data_bytes.size());

  SingleOpInterpreterBuilder model(kTfLiteBuiltinAdd, /*version=*/1);
  // {1, 4, 4, 3} + {4, 4, 3} is valid broadcasting.
  model.AddInput(kTfLiteFloat32, {1, 4, 4, 3});
  model.AddConstInput(kTfLiteFloat32, {4, 4, 3}, data_bytes);
  model.AddOutput(kTfLiteFloat32, {1, 4, 4, 3});

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  model.SetParameters(params);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModelFromBuilder(model);
  ASSERT_TRUE(ir_model);
  ASSERT_EQ(ir_model->ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = ir_model->op(0);
  const ::ml_drift::ElementwiseAttributes* attr =
      std::any_cast<::ml_drift::ElementwiseAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* t = std::get_if<
      ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>>(
      &attr->param);
  ASSERT_TRUE(t);
  EXPECT_EQ(t->shape, ::ml_drift::BHWC(1, 4, 4, 3));
}

}  // namespace
}  // namespace litert::ml_drift::ir
