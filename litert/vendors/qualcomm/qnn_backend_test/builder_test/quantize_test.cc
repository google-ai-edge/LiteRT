// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {

TEST(QuantizeOpBuilderTest, SelectsBackendSpecificOpsForQuantizedTensors) {
  const std::vector<std::uint32_t> kDims{1};
  const QuantizeParamsWrapperVariant input_quant{
      std::in_place_type<ScaleOffsetQuantizeParamsWrapper>, 1.0f, 0};
  const QuantizeParamsWrapperVariant output_quant{
      std::in_place_type<ScaleOffsetQuantizeParamsWrapper>, 1.0f, 128};
  TensorPool tensor_pool;
  auto& quantized_input = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_8, input_quant, kDims);
  auto& quantized_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_UFIXED_POINT_8, output_quant, kDims);

  const auto lpai_ops =
      BuildQuantizeOp(tensor_pool, {quantized_input}, {quantized_output},
                      BackendType::kLpaiBackend);
  ASSERT_EQ(lpai_ops.size(), 1u);
  EXPECT_TRUE(lpai_ops[0].IsOpCode(QnnOpCode::kConvert));

  const auto htp_ops =
      BuildQuantizeOp(tensor_pool, {quantized_input}, {quantized_output},
                      BackendType::kHtpBackend);
  ASSERT_EQ(htp_ops.size(), 1u);
  EXPECT_TRUE(htp_ops[0].IsOpCode(QnnOpCode::kCast));

  const auto ir_ops =
      BuildQuantizeOp(tensor_pool, {quantized_input}, {quantized_output},
                      BackendType::kIrBackend);
  ASSERT_EQ(ir_ops.size(), 1u);
  EXPECT_TRUE(ir_ops[0].IsOpCode(QnnOpCode::kCast));
}

TEST(QuantizeOpBuilderTest, BuildsQuantizeOpForFloatingPointInput) {
  const std::vector<std::uint32_t> kDims{1};
  const QuantizeParamsWrapperVariant output_quant{
      std::in_place_type<ScaleOffsetQuantizeParamsWrapper>, 1.0f, 128};
  TensorPool tensor_pool;
  auto& float_input =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& quantized_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_UFIXED_POINT_8, output_quant, kDims);

  const auto lpai_ops =
      BuildQuantizeOp(tensor_pool, {float_input}, {quantized_output},
                      BackendType::kLpaiBackend);
  ASSERT_EQ(lpai_ops.size(), 1u);
  EXPECT_TRUE(lpai_ops[0].IsOpCode(QnnOpCode::kQuantize));

  const auto htp_ops = BuildQuantizeOp(
      tensor_pool, {float_input}, {quantized_output}, BackendType::kHtpBackend);
  ASSERT_EQ(htp_ops.size(), 1u);
  EXPECT_TRUE(htp_ops[0].IsOpCode(QnnOpCode::kQuantize));

  const auto ir_ops = BuildQuantizeOp(
      tensor_pool, {float_input}, {quantized_output}, BackendType::kIrBackend);
  ASSERT_EQ(ir_ops.size(), 1u);
  EXPECT_TRUE(ir_ops[0].IsOpCode(QnnOpCode::kQuantize));
}

}  // namespace
}  // namespace qnn
