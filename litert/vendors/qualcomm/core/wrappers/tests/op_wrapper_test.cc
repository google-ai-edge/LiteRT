// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {

void EXPECT_TENSOR_EQ(Qnn_Tensor_t actual, Qnn_Tensor_t expected) {
  EXPECT_EQ(actual.v2.id, expected.v2.id);
  EXPECT_EQ(actual.v2.type, expected.v2.type);
  EXPECT_EQ(actual.v2.dataFormat, expected.v2.dataFormat);
  EXPECT_EQ(actual.v2.dataType, expected.v2.dataType);
  EXPECT_EQ(actual.v2.quantizeParams.encodingDefinition,
            expected.v2.quantizeParams.encodingDefinition);
  EXPECT_EQ(actual.v2.rank, expected.v2.rank);
  for (size_t i = 0; i < actual.v2.rank; i++) {
    EXPECT_EQ(actual.v2.dimensions[i], expected.v2.dimensions[i]);
  }
  EXPECT_EQ(actual.v2.memType, expected.v2.memType);
  EXPECT_EQ(actual.v2.clientBuf.dataSize, expected.v2.clientBuf.dataSize);
  const auto* actual_data =
      reinterpret_cast<const std::uint8_t*>(actual.v2.clientBuf.data);
  const auto* expected_data =
      reinterpret_cast<const std::uint8_t*>(expected.v2.clientBuf.data);
  for (size_t i = 0; i < actual.v2.clientBuf.dataSize; i++) {
    EXPECT_EQ(actual_data[i], expected_data[i]);
  }
}

TEST(OpWrapperTest, DefaultConstructor) {
  OpWrapper op_wrapper;
  EXPECT_EQ(op_wrapper.IsOpCode(QnnOpCode::kUnknown), true);
  const Qnn_OpConfig_t op_config = op_wrapper.GetOpConfig();
  EXPECT_EQ(op_config.version, QNN_OPCONFIG_VERSION_1);
  EXPECT_STREQ(op_config.v1.name, "");
  EXPECT_STREQ(op_config.v1.packageName, QNN_OP_PACKAGE_NAME_QTI_AISW);
  EXPECT_EQ(op_config.v1.typeName, nullptr);
  EXPECT_EQ(op_config.v1.numOfParams, 0);
  EXPECT_EQ(op_config.v1.params, nullptr);
  EXPECT_EQ(op_config.v1.numOfInputs, 0);
  EXPECT_EQ(op_config.v1.inputTensors, nullptr);
  EXPECT_EQ(op_config.v1.numOfOutputs, 0);
  EXPECT_EQ(op_config.v1.outputTensors, nullptr);
}

TEST(OpWrapperTest, SanityTest) {
  OpWrapper op_wrapper{"name", "OP_TYPE", QnnOpCode::kUnknown};
  EXPECT_EQ(op_wrapper.IsOpCode(QnnOpCode::kUnknown), true);
  const Qnn_OpConfig_t op_config = op_wrapper.GetOpConfig();
  EXPECT_EQ(op_config.version, QNN_OPCONFIG_VERSION_1);

  const Qnn_OpConfigV1_t& op_config_v1 = op_config.v1;
  EXPECT_STREQ(op_config_v1.typeName, "OP_TYPE");
  EXPECT_STREQ(op_config_v1.packageName, QNN_OP_PACKAGE_NAME_QTI_AISW);
  EXPECT_STREQ(op_config_v1.name, "name");
  EXPECT_EQ(op_config_v1.numOfInputs, 0);
  EXPECT_EQ(op_config_v1.numOfOutputs, 0);
  EXPECT_EQ(op_config_v1.numOfParams, 0);
  EXPECT_EQ(op_config_v1.params, nullptr);
  EXPECT_EQ(op_config_v1.inputTensors, nullptr);
  EXPECT_EQ(op_config_v1.outputTensors, nullptr);
}

TEST(OpWrapperTest, MoveCtorSanityTest) {
  OpWrapper op_wrapper{"name", "OP_TYPE", QnnOpCode::kUnknown};
  OpWrapper moved{std::move(op_wrapper)};
  EXPECT_EQ(moved.IsOpCode(QnnOpCode::kUnknown), true);
  const Qnn_OpConfig_t op_config = moved.GetOpConfig();
  EXPECT_EQ(op_config.version, QNN_OPCONFIG_VERSION_1);

  const Qnn_OpConfigV1_t& op_config_v1 = op_config.v1;
  EXPECT_STREQ(op_config_v1.typeName, "OP_TYPE");
  EXPECT_STREQ(op_config_v1.packageName, QNN_OP_PACKAGE_NAME_QTI_AISW);
  EXPECT_STREQ(op_config_v1.name, "name");
  EXPECT_EQ(op_config_v1.numOfInputs, 0);
  EXPECT_EQ(op_config_v1.numOfOutputs, 0);
  EXPECT_EQ(op_config_v1.numOfParams, 0);
  EXPECT_EQ(op_config_v1.params, nullptr);
  EXPECT_EQ(op_config_v1.inputTensors, nullptr);
  EXPECT_EQ(op_config_v1.outputTensors, nullptr);
}

TEST(OpWrapperTest, OpConfigTest) {
  std::vector<std::uint32_t> dummy_dims = {1, 1, 3};
  std::vector<std::uint8_t> data = {1, 2, 3};
  void* data_ptr = reinterpret_cast<void*>(data.data());
  const auto data_size =
      std::accumulate(dummy_dims.begin(), dummy_dims.end(),
                      sizeof(decltype(data)::value_type), std::multiplies<>());

  TensorWrapper tensor_wrapper{"",
                               QNN_TENSOR_TYPE_APP_WRITE,
                               QNN_DATATYPE_UFIXED_POINT_8,
                               QuantizeParamsWrapperVariant(),
                               dummy_dims,
                               static_cast<uint32_t>(data_size),
                               data_ptr,
                               true};

  Qnn_Tensor_t golden_qnn_tensor;
  tensor_wrapper.CloneTo(golden_qnn_tensor);

  std::uint8_t value = 255;
  OpWrapper op_wrapper{"name", "OP_TYPE", QnnOpCode::kUnknown};
  op_wrapper.AddInputTensor(tensor_wrapper);
  op_wrapper.AddOutputTensor(tensor_wrapper);
  op_wrapper.AddScalarParam("uint8_param", value, false);
  op_wrapper.AddTensorParam("tensor_param", tensor_wrapper);

  Qnn_OpConfig_t op_config = op_wrapper.GetOpConfig();
  EXPECT_EQ(op_config.version, QNN_OPCONFIG_VERSION_1);
  EXPECT_STREQ(op_config.v1.typeName, "OP_TYPE");
  EXPECT_STREQ(op_config.v1.packageName, QNN_OP_PACKAGE_NAME_QTI_AISW);
  EXPECT_STREQ(op_config.v1.name, "name");

  Qnn_OpConfigV1_t op_config_v1 = op_config.v1;

  EXPECT_EQ(op_config_v1.numOfInputs, 1);
  EXPECT_EQ(op_config_v1.numOfOutputs, 1);
  EXPECT_EQ(op_config_v1.numOfParams, 2);
  EXPECT_TENSOR_EQ(op_config_v1.inputTensors[0], golden_qnn_tensor);
  EXPECT_TENSOR_EQ(op_config_v1.outputTensors[0], golden_qnn_tensor);
  EXPECT_EQ(op_config_v1.params[0].paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(op_config_v1.params[0].name, "uint8_param");
  EXPECT_EQ(op_config_v1.params[0].scalarParam.dataType, QNN_DATATYPE_UINT_8);
  EXPECT_EQ(op_config_v1.params[0].scalarParam.uint8Value, value);
  EXPECT_EQ(op_config_v1.params[1].paramType, QNN_PARAMTYPE_TENSOR);
  EXPECT_EQ(op_config_v1.params[1].name, "tensor_param");
  EXPECT_TENSOR_EQ(op_config_v1.params[1].tensorParam, golden_qnn_tensor);
}

TEST(OpWrapperTest, MoveConstructorTest) {
  std::vector<std::uint32_t> dummy_dims = {1, 1, 3};
  std::vector<std::uint8_t> data = {1, 2, 3};
  void* data_ptr = reinterpret_cast<void*>(data.data());
  TensorWrapper tensor_wrapper{"",
                               QNN_TENSOR_TYPE_APP_WRITE,
                               QNN_DATATYPE_UFIXED_POINT_8,
                               QuantizeParamsWrapperVariant(),
                               dummy_dims,
                               static_cast<uint32_t>(data.size()),
                               data_ptr,
                               true};
  Qnn_Tensor_t golden_qnn_tensor;
  tensor_wrapper.CloneTo(golden_qnn_tensor);
  std::uint8_t value = 255;
  OpWrapper op_wrapper{"name", "OP_TYPE", QnnOpCode::kUnknown};
  op_wrapper.AddInputTensor(tensor_wrapper);
  op_wrapper.AddOutputTensor(tensor_wrapper);
  op_wrapper.AddScalarParam("uint8_param", value, false);
  op_wrapper.AddTensorParam("tensor_param", tensor_wrapper);
  OpWrapper op_wrapper_move(std::move(op_wrapper));
  EXPECT_EQ(op_wrapper_move.IsOpCode(QnnOpCode::kUnknown), true);
  Qnn_OpConfig_t op_config = op_wrapper_move.GetOpConfig();
  EXPECT_EQ(op_config.version, QNN_OPCONFIG_VERSION_1);
  EXPECT_STREQ(op_config.v1.typeName, "OP_TYPE");
  EXPECT_STREQ(op_config.v1.packageName, QNN_OP_PACKAGE_NAME_QTI_AISW);
  EXPECT_STREQ(op_config.v1.name, "name");
  Qnn_OpConfigV1_t op_config_v1 = op_config.v1;
  EXPECT_EQ(op_config_v1.numOfInputs, 1);
  EXPECT_EQ(op_config_v1.numOfOutputs, 1);
  EXPECT_EQ(op_config_v1.numOfParams, 2);
  EXPECT_TENSOR_EQ(op_config_v1.inputTensors[0], golden_qnn_tensor);
  EXPECT_TENSOR_EQ(op_config_v1.outputTensors[0], golden_qnn_tensor);
  EXPECT_EQ(op_config_v1.params[0].paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(op_config_v1.params[0].name, "uint8_param");
  EXPECT_EQ(op_config_v1.params[0].scalarParam.dataType, QNN_DATATYPE_UINT_8);
  EXPECT_EQ(op_config_v1.params[0].scalarParam.uint8Value, value);
  EXPECT_EQ(op_config_v1.params[1].paramType, QNN_PARAMTYPE_TENSOR);
  EXPECT_EQ(op_config_v1.params[1].name, "tensor_param");
  EXPECT_TENSOR_EQ(op_config_v1.params[1].tensorParam, golden_qnn_tensor);
}

TEST(OpWrapperTest, GetInputOutputTensorTest) {
  TensorWrapper tensor_wrapper_input{};
  TensorWrapper tensor_wrapper_output{};
  OpWrapper op_wrapper{"name", "OP_TYPE", QnnOpCode::kUnknown};
  op_wrapper.AddInputTensor(tensor_wrapper_input);
  op_wrapper.AddOutputTensor(tensor_wrapper_output);
  EXPECT_EQ(op_wrapper.GetInputTensor(0), tensor_wrapper_input);
  EXPECT_EQ(op_wrapper.GetOutputTensor(0), tensor_wrapper_output);
}

TEST(OpWrapperTest, GetScalarParamNameTest) {
  std::uint8_t value = 255;

  OpWrapper op_wrapper{"name", "OP_TYPE", QnnOpCode::kUnknown};
  op_wrapper.AddScalarParam("uint8_param", value, false);
  const auto& param_wrapper = op_wrapper.GetScalarPararm(0);
  EXPECT_STREQ(param_wrapper.GetName().data(), "uint8_param");
}

TEST(OpWrapperTest, GetScalarParamTypesTest) {
  OpWrapper op_wrapper{"name", "OP_TYPE", QnnOpCode::kUnknown};

  // Bool
  bool bool_val = true;
  op_wrapper.AddScalarParam("bool_param", bool_val, false);

  // Int8
  int8_t int8_val = -10;
  op_wrapper.AddScalarParam("int8_param", int8_val, false);

  // UInt16
  uint16_t uint16_val = 60000;
  op_wrapper.AddScalarParam("uint16_param", uint16_val, false);

  // Int16
  int16_t int16_val = -30000;
  op_wrapper.AddScalarParam("int16_param", int16_val, false);

  // UInt32
  uint32_t uint32_val = 123456789;
  op_wrapper.AddScalarParam("uint32_param", uint32_val, false);

  // Int32
  int32_t int32_val = -123456789;
  op_wrapper.AddScalarParam("int32_param", int32_val, false);

  // Float
  float float_val = 3.14f;
  op_wrapper.AddScalarParam("float_param", float_val, false);

  // Verify
  EXPECT_EQ(op_wrapper.GetScalarPararm(0).GetValue<bool>(), bool_val);
  EXPECT_EQ(op_wrapper.GetScalarPararm(1).GetValue<int8_t>(), int8_val);
  EXPECT_EQ(op_wrapper.GetScalarPararm(2).GetValue<uint16_t>(), uint16_val);
  EXPECT_EQ(op_wrapper.GetScalarPararm(3).GetValue<int16_t>(), int16_val);
  EXPECT_EQ(op_wrapper.GetScalarPararm(4).GetValue<uint32_t>(), uint32_val);
  EXPECT_EQ(op_wrapper.GetScalarPararm(5).GetValue<int32_t>(), int32_val);
  EXPECT_FLOAT_EQ(op_wrapper.GetScalarPararm(6).GetValue<float>(), float_val);
}

TEST(OpWrapperTest, SwapOutputsTest) {
  TensorWrapper input_1{};
  TensorWrapper output_1{};
  OpWrapper op_wrapper_1{"name", "OP_TYPE", QnnOpCode::kUnknown};
  op_wrapper_1.AddInputTensor(input_1);
  op_wrapper_1.AddOutputTensor(output_1);

  TensorWrapper input_2{};
  TensorWrapper output_2{};
  OpWrapper op_wrapper_2{"name", "OP_TYPE", QnnOpCode::kUnknown};
  op_wrapper_2.AddInputTensor(input_2);
  op_wrapper_2.AddOutputTensor(output_2);

  EXPECT_EQ(op_wrapper_1.GetOutputTensor(0), output_1);
  op_wrapper_1.SwapOutputs(op_wrapper_2);
  EXPECT_EQ(op_wrapper_1.GetOutputTensor(0), output_2);
  op_wrapper_1.SwapOutputs(op_wrapper_2);
  EXPECT_EQ(op_wrapper_1.GetOutputTensor(0), output_1);
}

TEST(OpWrapperTest, GetName) {
  OpWrapper op{"name", "OP_TYPE", QnnOpCode::kUnknown};
  EXPECT_STREQ(op.GetName().data(), "name");
}

TEST(OpWrapperTest, ChangeOpName) {
  OpWrapper op{"name", "OP_TYPE", QnnOpCode::kUnknown};
  EXPECT_STREQ(op.GetName().data(), "name");
  op.AddPrefixToName("namespace/");
  EXPECT_STREQ(op.GetName().data(), "namespace/name");
  op.AddSuffixToName("_new");
  EXPECT_STREQ(op.GetName().data(), "namespace/name_new");
}

TEST(OpWrapperTest, SetName) {
  OpWrapper op_wrapper;
  op_wrapper.SetName("test_name");
  const Qnn_OpConfig_t op_config = op_wrapper.GetOpConfig();
  EXPECT_STREQ(op_config.v1.name, "test_name");
}

TEST(OpWrapperTest, SetType) {
  OpWrapper op_wrapper;
  op_wrapper.SetType(QNN_OP_CONV_2D, QnnOpCode::kConv2d);
  const Qnn_OpConfig_t op_config = op_wrapper.GetOpConfig();
  EXPECT_TRUE(op_wrapper.IsOpCode(QnnOpCode::kConv2d));
  EXPECT_EQ(op_config.v1.typeName, QNN_OP_CONV_2D);
}

TensorWrapper CreateTensor(const std::vector<uint32_t>& dims,
                           Qnn_TensorType_t type,
                           std::optional<uint8_t> val = std::nullopt) {
  std::vector<uint8_t> data(
      std::accumulate(dims.begin(), dims.end(), 1u, std::multiplies<>()));
  if (val.has_value()) {
    std::fill(data.begin(), data.end(), *val);
  }
  return TensorWrapper("", type, QNN_DATATYPE_UFIXED_POINT_8,
                       QuantizeParamsWrapperVariant(), dims,
                       static_cast<uint32_t>(data.size()), data.data(), true);
}

TEST(OpWrapperEqualityOperatorTest, TypeName) {
  std::vector<uint32_t> dims = {1, 1, 3};
  auto input = CreateTensor(dims, QNN_TENSOR_TYPE_NATIVE);
  auto output = input;
  auto param_tensor = CreateTensor(dims, QNN_TENSOR_TYPE_STATIC, 1);

  qnn::OpWrapper op1{"op", "TYPE_A", QnnOpCode::kUnknown};
  op1.AddInputTensor(input);
  op1.AddOutputTensor(output);
  op1.AddScalarParam<std::uint8_t>("param", 255, false);
  op1.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op2{"op", "TYPE_A", QnnOpCode::kUnknown};
  op2.AddInputTensor(input);
  op2.AddOutputTensor(output);
  op2.AddScalarParam<std::uint8_t>("param", 255, false);
  op2.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op3{"op", "TYPE_B", QnnOpCode::kUnknown};
  op3.AddInputTensor(input);
  op3.AddOutputTensor(output);
  op3.AddScalarParam<std::uint8_t>("param", 255, false);
  op3.AddTensorParam("tensor_param", param_tensor);

  EXPECT_TRUE(op1 == op2);
  EXPECT_FALSE(op1 == op3);
}

TEST(OpWrapperEqualityOperatorTest, OpCode) {
  std::vector<uint32_t> dims = {1, 1, 3};
  auto input = CreateTensor(dims, QNN_TENSOR_TYPE_NATIVE);
  auto output = input;
  auto param_tensor = CreateTensor(dims, QNN_TENSOR_TYPE_STATIC, 1);

  qnn::OpWrapper op1{"op", "OP_TYPE", QnnOpCode::kElementWiseAdd};
  op1.AddInputTensor(input);
  op1.AddOutputTensor(output);
  op1.AddScalarParam<std::uint8_t>("param", 255, false);
  op1.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op2{"op", "OP_TYPE", QnnOpCode::kElementWiseAdd};
  op2.AddInputTensor(input);
  op2.AddOutputTensor(output);
  op2.AddScalarParam<std::uint8_t>("param", 255, false);
  op2.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op3{"op", "OP_TYPE", QnnOpCode::kElementWiseMultiply};
  op3.AddInputTensor(input);
  op3.AddOutputTensor(output);
  op3.AddScalarParam<std::uint8_t>("param", 255, false);
  op3.AddTensorParam("tensor_param", param_tensor);

  EXPECT_TRUE(op1 == op2);
  EXPECT_FALSE(op1 == op3);
}

TEST(OpWrapperEqualityOperatorTest, InputDims) {
  std::vector<uint32_t> dims1 = {1, 1, 3};
  std::vector<uint32_t> dims2 = {1, 1, 4};
  auto input1 = CreateTensor(dims1, QNN_TENSOR_TYPE_NATIVE);
  auto input2 = CreateTensor(dims2, QNN_TENSOR_TYPE_NATIVE);
  auto output = input1;
  auto param_tensor = CreateTensor(dims1, QNN_TENSOR_TYPE_STATIC, 1);

  qnn::OpWrapper op1{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op1.AddInputTensor(input1);
  op1.AddOutputTensor(output);
  op1.AddScalarParam<std::uint8_t>("param", 255, false);
  op1.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op2{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op2.AddInputTensor(input1);
  op2.AddOutputTensor(output);
  op2.AddScalarParam<std::uint8_t>("param", 255, false);
  op2.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op3{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op3.AddInputTensor(input2);
  op3.AddOutputTensor(output);
  op3.AddScalarParam<std::uint8_t>("param", 255, false);
  op3.AddTensorParam("tensor_param", param_tensor);

  EXPECT_TRUE(op1 == op2);
  EXPECT_FALSE(op1 == op3);
}

TEST(OpWrapperEqualityOperatorTest, ScalarParam) {
  std::vector<uint32_t> dims = {1, 1, 3};
  auto input = CreateTensor(dims, QNN_TENSOR_TYPE_NATIVE);
  auto output = input;
  auto param_tensor = CreateTensor(dims, QNN_TENSOR_TYPE_STATIC, 1);

  qnn::OpWrapper op1{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op1.AddInputTensor(input);
  op1.AddOutputTensor(output);
  op1.AddScalarParam<std::uint8_t>("param", 255, false);
  op1.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op2{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op2.AddInputTensor(input);
  op2.AddOutputTensor(output);
  op2.AddScalarParam<std::uint8_t>("param", 255, false);
  op2.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op3{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op3.AddInputTensor(input);
  op3.AddOutputTensor(output);
  op3.AddScalarParam<std::uint8_t>("param", 252, false);
  op3.AddTensorParam("tensor_param", param_tensor);

  EXPECT_TRUE(op1 == op2);
  EXPECT_FALSE(op1 == op3);
}

TEST(OpWrapperEqualityOperatorTest, TensorParam) {
  std::vector<uint32_t> dims = {1, 1, 3};
  auto input = CreateTensor(dims, QNN_TENSOR_TYPE_NATIVE);
  auto output = input;
  auto param_tensor = CreateTensor(dims, QNN_TENSOR_TYPE_STATIC, 1);
  auto param_tensor_diff = CreateTensor(dims, QNN_TENSOR_TYPE_STATIC, 2);

  qnn::OpWrapper op1{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op1.AddInputTensor(input);
  op1.AddOutputTensor(output);
  op1.AddScalarParam<std::uint8_t>("param", 255, false);
  op1.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op2{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op2.AddInputTensor(input);
  op2.AddOutputTensor(output);
  op2.AddScalarParam<std::uint8_t>("param", 255, false);
  op2.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op3{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op3.AddInputTensor(input);
  op3.AddOutputTensor(output);
  op3.AddScalarParam<std::uint8_t>("param", 255, false);
  op3.AddTensorParam("tensor_param", param_tensor_diff);

  EXPECT_TRUE(op1 == op2);
  EXPECT_FALSE(op1 == op3);
}

TEST(OpWrapperEqualityOperatorTest, InputSize) {
  std::vector<uint32_t> dims = {1, 1, 3};
  auto input = CreateTensor(dims, QNN_TENSOR_TYPE_NATIVE);
  auto output = input;
  auto param_tensor = CreateTensor(dims, QNN_TENSOR_TYPE_STATIC, 1);

  qnn::OpWrapper op1{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op1.AddInputTensor(input);
  op1.AddOutputTensor(output);
  op1.AddScalarParam<std::uint8_t>("param", 255, false);
  op1.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op2{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op2.AddInputTensor(input);
  op2.AddOutputTensor(output);
  op2.AddScalarParam<std::uint8_t>("param", 255, false);
  op2.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op3{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op3.AddInputTensor(input);
  op3.AddInputTensor(input);
  op3.AddOutputTensor(output);
  op3.AddScalarParam<std::uint8_t>("param", 255, false);
  op3.AddTensorParam("tensor_param", param_tensor);

  EXPECT_TRUE(op1 == op2);
  EXPECT_FALSE(op1 == op3);
}

TEST(OpWrapperEqualityOperatorTest, InputData) {
  std::vector<uint32_t> dims = {1, 1, 3};
  auto input1 = CreateTensor(dims, QNN_TENSOR_TYPE_APP_WRITE, 1);
  auto input2 = CreateTensor(dims, QNN_TENSOR_TYPE_APP_WRITE, 2);
  auto output = CreateTensor(dims, QNN_TENSOR_TYPE_APP_READ, 2);
  auto param_tensor = CreateTensor(dims, QNN_TENSOR_TYPE_STATIC, 1);

  qnn::OpWrapper op1{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op1.AddInputTensor(input1);
  op1.AddOutputTensor(output);
  op1.AddScalarParam<std::uint8_t>("param", 255, false);
  op1.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op2{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op2.AddInputTensor(input1);
  op2.AddOutputTensor(output);
  op2.AddScalarParam<std::uint8_t>("param", 255, false);
  op2.AddTensorParam("tensor_param", param_tensor);

  qnn::OpWrapper op3{"op", "OP_TYPE", QnnOpCode::kUnknown};
  op3.AddInputTensor(input2);
  op3.AddOutputTensor(output);
  op3.AddScalarParam<std::uint8_t>("param", 255, false);
  op3.AddTensorParam("tensor_param", param_tensor);

  EXPECT_TRUE(op1 == op2);
  EXPECT_FALSE(op1 == op3);
}

}  // namespace
}  // namespace qnn
