// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
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

TEST(OpWrapperTest, SanityTest) {
  OpWrapper op_wrapper{"name", "OP_TYPE", QnnOpCode::kUnknown};
  EXPECT_EQ(op_wrapper.IsOpCode(QnnOpCode::kUnknown), true);
  const Qnn_OpConfig_t& op_config = op_wrapper.GetOpConfig();
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
  const Qnn_OpConfig_t& op_config = moved.GetOpConfig();
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
                               data_ptr};

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
                               data_ptr};
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

}  // namespace
}  // namespace qnn
