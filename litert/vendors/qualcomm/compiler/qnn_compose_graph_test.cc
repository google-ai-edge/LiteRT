// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/compiler/qnn_compose_graph.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers

namespace litert::qnn {
namespace {

const Qnn_Param_t* FindParam(const Qnn_OpConfig_t& op_config,
                             const char* name) {
  for (uint32_t i = 0; i < op_config.v1.numOfParams; ++i) {
    if (std::strcmp(op_config.v1.params[i].name, name) == 0) {
      return &op_config.v1.params[i];
    }
  }
  return nullptr;
}

TEST(AddCustomOpOptionsAsParamsTest, EmptyOptionsAddsNoParams) {
  ::qnn::TensorPool tensor_pool;
  ::qnn::OpWrapper custom_op{"test_op", "TestOp", ::qnn::QnnOpCode::kUnknown};

  const std::vector<uint8_t> custom_options;
  EXPECT_EQ(AddCustomOpOptionsAsParams(custom_options, tensor_pool, custom_op),
            kLiteRtStatusOk);

  const Qnn_OpConfig_t op_config = custom_op.GetOpConfig();
  EXPECT_EQ(op_config.v1.numOfParams, 0);
}

TEST(AddCustomOpOptionsAsParamsTest, InvalidOptionsRejected) {
  ::qnn::TensorPool tensor_pool;
  ::qnn::OpWrapper custom_op{"test_op", "TestOp", ::qnn::QnnOpCode::kUnknown};

  const std::vector<uint8_t> custom_options = {'x', 'y'};
  EXPECT_EQ(AddCustomOpOptionsAsParams(custom_options, tensor_pool, custom_op),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(AddCustomOpOptionsAsParamsTest, NonMapOptionsRejected) {
  flexbuffers::Builder fbb;
  fbb.Int(42);
  fbb.Finish();
  const auto custom_options = fbb.GetBuffer();

  ::qnn::TensorPool tensor_pool;
  ::qnn::OpWrapper custom_op{"test_op", "TestOp", ::qnn::QnnOpCode::kUnknown};

  EXPECT_EQ(AddCustomOpOptionsAsParams(custom_options, tensor_pool, custom_op),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(AddCustomOpOptionsAsParamsTest, AddsScalarParams) {
  flexbuffers::Builder fbb;
  auto map_start = fbb.StartMap();
  fbb.Bool("bool_param", true);
  fbb.Int("int_param", -7);
  fbb.UInt("uint_param", 9);
  fbb.Float("float_param", 1.5f);
  fbb.EndMap(map_start);
  fbb.Finish();
  const auto custom_options = fbb.GetBuffer();

  ::qnn::TensorPool tensor_pool;
  ::qnn::OpWrapper custom_op{"test_op", "TestOp", ::qnn::QnnOpCode::kUnknown};

  EXPECT_EQ(AddCustomOpOptionsAsParams(custom_options, tensor_pool, custom_op),
            kLiteRtStatusOk);

  const Qnn_OpConfig_t op_config = custom_op.GetOpConfig();
  ASSERT_EQ(op_config.v1.numOfParams, 4);

  const Qnn_Param_t* bool_param = FindParam(op_config, "bool_param");
  ASSERT_NE(bool_param, nullptr);
  EXPECT_EQ(bool_param->paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(bool_param->scalarParam.dataType, QNN_DATATYPE_BOOL_8);
  EXPECT_TRUE(bool_param->scalarParam.bool8Value);

  const Qnn_Param_t* int_param = FindParam(op_config, "int_param");
  ASSERT_NE(int_param, nullptr);
  EXPECT_EQ(int_param->paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(int_param->scalarParam.dataType, QNN_DATATYPE_INT_32);
  EXPECT_EQ(int_param->scalarParam.int32Value, -7);

  const Qnn_Param_t* uint_param = FindParam(op_config, "uint_param");
  ASSERT_NE(uint_param, nullptr);
  EXPECT_EQ(uint_param->paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(uint_param->scalarParam.dataType, QNN_DATATYPE_UINT_32);
  EXPECT_EQ(uint_param->scalarParam.uint32Value, 9);

  const Qnn_Param_t* float_param = FindParam(op_config, "float_param");
  ASSERT_NE(float_param, nullptr);
  EXPECT_EQ(float_param->paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(float_param->scalarParam.dataType, QNN_DATATYPE_FLOAT_32);
  EXPECT_FLOAT_EQ(float_param->scalarParam.floatValue, 1.5f);
}

TEST(AddCustomOpOptionsAsParamsTest, AddsVectorParamAsStaticTensor) {
  flexbuffers::Builder fbb;
  auto map_start = fbb.StartMap();
  fbb.Vector("weights", [&fbb]() {
    fbb.Vector([&fbb]() {
      fbb.Int(1);
      fbb.Int(2);
    });
    fbb.Vector([&fbb]() {
      fbb.Int(3);
      fbb.Int(4);
    });
  });
  fbb.EndMap(map_start);
  fbb.Finish();
  const auto custom_options = fbb.GetBuffer();

  ::qnn::TensorPool tensor_pool;
  ::qnn::OpWrapper custom_op{"test_op", "TestOp", ::qnn::QnnOpCode::kUnknown};

  EXPECT_EQ(AddCustomOpOptionsAsParams(custom_options, tensor_pool, custom_op),
            kLiteRtStatusOk);

  const Qnn_OpConfig_t op_config = custom_op.GetOpConfig();
  ASSERT_EQ(op_config.v1.numOfParams, 1);

  const Qnn_Param_t* weights = FindParam(op_config, "weights");
  ASSERT_NE(weights, nullptr);
  ASSERT_EQ(weights->paramType, QNN_PARAMTYPE_TENSOR);

  const Qnn_Tensor_t& tensor = weights->tensorParam;
  EXPECT_EQ(tensor.v2.type, QNN_TENSOR_TYPE_STATIC);
  EXPECT_EQ(tensor.v2.dataType, QNN_DATATYPE_INT_32);
  ASSERT_EQ(tensor.v2.rank, 2);
  EXPECT_EQ(tensor.v2.dimensions[0], 2);
  EXPECT_EQ(tensor.v2.dimensions[1], 2);
  EXPECT_EQ(tensor.v2.clientBuf.dataSize, 4 * sizeof(int32_t));
  ASSERT_NE(tensor.v2.clientBuf.data, nullptr);

  const auto* data = static_cast<const int32_t*>(tensor.v2.clientBuf.data);
  EXPECT_EQ(std::vector<int32_t>(data, data + 4),
            (std::vector<int32_t>{1, 2, 3, 4}));
}

TEST(AddCustomOpOptionsAsParamsTest, RaggedVectorRejected) {
  flexbuffers::Builder fbb;
  auto map_start = fbb.StartMap();
  fbb.Vector("ragged", [&fbb]() {
    fbb.Vector([&fbb]() {
      fbb.Int(1);
      fbb.Int(2);
    });
    fbb.Vector([&fbb]() { fbb.Int(3); });
  });
  fbb.EndMap(map_start);
  fbb.Finish();
  const auto custom_options = fbb.GetBuffer();

  ::qnn::TensorPool tensor_pool;
  ::qnn::OpWrapper custom_op{"test_op", "TestOp", ::qnn::QnnOpCode::kUnknown};

  EXPECT_EQ(AddCustomOpOptionsAsParams(custom_options, tensor_pool, custom_op),
            kLiteRtStatusErrorUnsupported);
}

TEST(AddCustomOpOptionsAsParamsTest, MixedVectorRejected) {
  flexbuffers::Builder fbb;
  auto map_start = fbb.StartMap();
  fbb.Vector("mixed", [&fbb]() {
    fbb.Int(1);
    fbb.Float(2.0f);
  });
  fbb.EndMap(map_start);
  fbb.Finish();
  const auto custom_options = fbb.GetBuffer();

  ::qnn::TensorPool tensor_pool;
  ::qnn::OpWrapper custom_op{"test_op", "TestOp", ::qnn::QnnOpCode::kUnknown};

  EXPECT_EQ(AddCustomOpOptionsAsParams(custom_options, tensor_pool, custom_op),
            kLiteRtStatusErrorUnsupported);
}

}  // namespace
}  // namespace litert::qnn
