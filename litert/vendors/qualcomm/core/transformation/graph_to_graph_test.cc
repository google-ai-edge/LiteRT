// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/graph_to_graph.h"


#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {
TEST(MatMulConvertTest, Gemma3Prefill) {
  // G2G Test case: 2 MatMuls with 1 Convert
  //
  // ----- Before -----
  //   In1   In0   In2
  //    |  /     \  |
  //    | /       \ |
  // MatMul0     MatMul1
  //    |           |
  // Convert       Out1
  //    |
  //   Out0
  //
  // ----- After -----
  //   In1   In0   In2
  //    |  /     \  |
  //    | /       \ |
  // MatMul0     MatMul1
  //    |           |
  //   Out0        Out1
  //

  TensorPool tensor_pool;

  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);

  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 512, 256});
  auto& input1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 1280, 256});

  auto& matmul_to_concat = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 1280});

  auto& input2 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 128, 256});
  auto& output1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                 quant_param, {1, 1, 512, 128});

  QuantizeParamsWrapperVariant convert_quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-3f, 0);
  auto& output0 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, convert_quant_param, {1, 1, 512, 1280});

  std::vector<OpWrapper> op_wrappers;

  auto& matmul0 = op_wrappers.emplace_back(
      CreateMatmulOp(input0, input1, matmul_to_concat, false, true));
  auto& matmul1 = op_wrappers.emplace_back(
      CreateMatmulOp(input0, input2, output1, false, true));

  // Convert: matmul_to_concat, Out0
  std::vector<::qnn::TensorWrapperRef> convert_inputs;
  convert_inputs.emplace_back(matmul_to_concat);
  std::vector<::qnn::TensorWrapperRef> convert_outputs;
  convert_outputs.emplace_back(output0);
  auto convert = BuildQuantizeOp(tensor_pool, convert_inputs, convert_outputs);
  std::move(convert.begin(), convert.end(), std::back_inserter(op_wrappers));

  ASSERT_EQ(op_wrappers.size(), 3);

  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMatMulConvert;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });

  ASSERT_EQ(op_wrappers.size(), 2);
  ASSERT_EQ(op_wrappers[0].IsOpCode(QnnOpCode::kMatMul), true);
  ASSERT_EQ(op_wrappers[1].IsOpCode(QnnOpCode::kMatMul), true);
}

TEST(MatMulConvertTest, Gemma3Decode) {
  // G2G Test case: 2 MatMuls with 1 Convert
  //
  // ----- Before -----
  //   In0 In1
  //    |  /
  //    | /
  // MatMul0
  //    |
  // Convert
  //    |
  //   Out0
  //
  // ----- After -----
  //   In0 In1
  //    |  /
  //    | /
  // MatMul0
  //    |
  //   Out0
  //

  TensorPool tensor_pool;

  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);

  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 512, 256});
  auto& input1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 1280, 256});

  auto& matmul_to_concat = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 1280});

  QuantizeParamsWrapperVariant convert_quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-3f, 0);
  auto& output0 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, convert_quant_param, {1, 1, 512, 1280});

  std::vector<OpWrapper> op_wrappers;

  auto& matmul0 = op_wrappers.emplace_back(
      CreateMatmulOp(input0, input1, matmul_to_concat, false, true));

  // Convert: matmul_to_concat, Out0
  std::vector<::qnn::TensorWrapperRef> convert_inputs;
  convert_inputs.emplace_back(matmul_to_concat);
  std::vector<::qnn::TensorWrapperRef> convert_outputs;
  convert_outputs.emplace_back(output0);
  auto convert = BuildQuantizeOp(tensor_pool, convert_inputs, convert_outputs);
  std::move(convert.begin(), convert.end(), std::back_inserter(op_wrappers));

  ASSERT_EQ(op_wrappers.size(), 2);

  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMatMulConvert;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });

  ASSERT_EQ(op_wrappers.size(), 1);
  ASSERT_EQ(op_wrappers[0].IsOpCode(QnnOpCode::kMatMul), true);
}

TEST(MHAOptimization, Gemma3Prefill) {
  // G2G Test case: MHA -> SHA
  //
  // ---------------- Before ---------------------
  //                   In0
  //                    |
  //                   Mul
  //                    |
  //                Transpose0
  //                    |
  //                 Reshape0
  //      kv_cache_k  /   \  kv_slice_k
  //              \  /     \  /
  //           MatMulK0   MatMulK1
  //                 \     /
  //                  Concat
  //                    |
  //                 Reshape1
  //                    |
  //               mask |
  //                  \ |
  //                   Add0
  //                    |
  //                 Reshape2
  //                    |
  //                 Softmax
  //                  /   \
  //                 /     \
  // kv_cache_v  Slice0   Slice1  kv_slice_v
  //      \        /         \        /
  //       \      /           \      /
  //       MatMulV0           MatMulV1
  //              \           /
  //               \         /
  //                \       /
  //                 \     /
  //                   Add1
  //                    |
  //                 Reshape3
  //                    |
  //                Transpose1
  //                    |
  //                 Reshape4
  //                    |
  //                   Out0
  //
  // ---------------- After ---------------------
  //                   In0
  //                    |
  //                Transpose
  //                    |
  //                 Reshape
  //                    |
  //                  Split
  //                    | \\\
  //                    |  \\\
  //                    |   ...
  //                   Mul
  //      kv_cache_k  /   \  kv_slice_k
  //              \  /     \  /
  //              MatMul  MatMul
  //                 \     /
  //                  Concat
  //                    |
  //                   Add
  //                    |
  //                 Softmax
  //                  /   \
  //                 /     \
  // kv_cache_v  Slice0   Slice1  kv_slice_v
  //      \        /         \        /
  //       \      /           \      /
  //        MatMul             MatMul
  //              \           /
  //               \         /
  //                \       /
  //                 \     /
  //                   Add    ...
  //                    |    ///
  //                    |   ///
  //                    Concat
  //                      |
  //                   Reshape
  //                      |
  //                     Out0
  //
  TensorPool tensor_pool;
  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);
  std::vector<OpWrapper> op_wrappers;
  // Mul
  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 128, 4, 256});
  std::array<int16_t, 1> mul_val = {32767};
  auto& mul_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {mul_val.size()},
      mul_val.size() * sizeof(mul_val[0]), mul_val.data());
  auto& mul_output =
      tensor_pool.CloneNativeTensorFrom(input0, {1, 128, 4, 256});
  op_wrappers.emplace_back(
      CreateElementWiseMulOp(input0, mul_const, mul_output));
  // Transpose0
  std::array<std::uint32_t, 4> transpose_val = {0, 2, 1, 3};
  auto& transpose_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, quant_param, {transpose_val.size()},
      transpose_val.size() * sizeof(transpose_val[0]), transpose_val.data());
  auto& transpose0_output =
      tensor_pool.CloneNativeTensorFrom(mul_output, {1, 4, 128, 256});
  auto& transpose0 = op_wrappers.emplace_back(
      CreateTransposeOp(mul_output, transpose0_output, transpose_perm));

  // Reshape0
  auto& reshape0_output =
      tensor_pool.CloneNativeTensorFrom(transpose0_output, {1, 1, 512, 256});
  auto& reshape0 = op_wrappers.emplace_back(
      CreateReshapeOp(transpose0_output, reshape0_output));

  // MatMulK0
  auto& kv_cache_k = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 1280, 256});
  auto& matmulk0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 1280});
  auto& matmulk0 = op_wrappers.emplace_back(CreateMatmulOp(
      reshape0_output, kv_cache_k, matmulk0_output, false, true));

  // MatMulK1
  auto& kv_slice_k = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 128, 256});
  auto& matmulk1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 128});
  auto& matmulk1 = op_wrappers.emplace_back(CreateMatmulOp(
      reshape0_output, kv_slice_k, matmulk1_output, false, true));
  // Concat
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(matmulk0_output, {1, 1, 512, 1408});
  op_wrappers.emplace_back(CreateConcatenationOp(
      {matmulk0_output, matmulk1_output}, concat_output, 3));
  // Reshape1
  auto& reshape1_output =
      tensor_pool.CloneNativeTensorFrom(concat_output, {1, 4, 128, 1408});
  auto& reshape1 =
      op_wrappers.emplace_back(CreateReshapeOp(concat_output, reshape1_output));
  // Add
  auto& add0_output = tensor_pool.CloneNativeTensorFrom(reshape1_output);
  auto& mask = tensor_pool.CloneNativeTensorFrom(reshape1_output);
  op_wrappers.emplace_back(
      CreateElementWiseAddOp(reshape1_output, mask, add0_output));
  // Reshape2
  auto& reshape2_output =
      tensor_pool.CloneNativeTensorFrom(add0_output, {1, 1, 512, 1408});
  auto& reshape2 =
      op_wrappers.emplace_back(CreateReshapeOp(add0_output, reshape2_output));
  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(reshape2_output);
  auto& softmax = op_wrappers.emplace_back(
      CreateSoftmaxOp(reshape2_output, softmax_output, 1.0f));
  // Slice0
  const std::array<int32_t, 12> slice0_ranges_data{0, 1,   1, 0, 1,    1,
                                                   0, 512, 1, 0, 1280, 1};
  auto& slice0_rangs = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {4, 3},
      sizeof(int32_t) * slice0_ranges_data.size(), slice0_ranges_data.data());
  auto& slice0_output =
      tensor_pool.CloneNativeTensorFrom(reshape2_output, {1, 1, 512, 1280});
  auto& slice0 = op_wrappers.emplace_back(
      CreateSliceOp(softmax_output, slice0_output, slice0_rangs));
  // Slice1
  const std::array<int32_t, 12> slice1_ranges_data{0, 1,   1, 0,    1,    1,
                                                   0, 512, 1, 1280, 1408, 1};
  auto& slice1_rangs = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {4, 3},
      sizeof(int32_t) * slice1_ranges_data.size(), slice1_ranges_data.data());
  auto& slice1_output =
      tensor_pool.CloneNativeTensorFrom(reshape2_output, {1, 1, 512, 128});
  auto& slice1 = op_wrappers.emplace_back(
      CreateSliceOp(softmax_output, slice1_output, slice1_rangs));
  // MatMulV0
  auto& kv_cache_v = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 256, 1280});
  auto& matmulv0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 256});
  auto& matmulv0 = op_wrappers.emplace_back(
      CreateMatmulOp(slice0_output, kv_cache_v, matmulv0_output, false, true));
  // MatMulV1
  auto& kv_slice_v = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 256, 128});
  auto& matmulv1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 256});
  auto& matmulv1 = op_wrappers.emplace_back(
      CreateMatmulOp(slice1_output, kv_slice_v, matmulv1_output, false, true));
  // Add1
  auto& add1_output = tensor_pool.CloneNativeTensorFrom(matmulv0_output);
  op_wrappers.emplace_back(
      CreateElementWiseAddOp(matmulv0_output, matmulv1_output, add1_output));
  // Reshape3
  auto& reshape3_output =
      tensor_pool.CloneNativeTensorFrom(add1_output, {1, 4, 128, 256});
  auto& reshape3 =
      op_wrappers.emplace_back(CreateReshapeOp(add1_output, reshape3_output));
  // Transpose
  auto& transpose1_output =
      tensor_pool.CloneNativeTensorFrom(reshape3_output, {1, 128, 4, 256});
  auto& transpose1 = op_wrappers.emplace_back(
      CreateTransposeOp(reshape3_output, transpose1_output, transpose_perm));
  // Reshape4
  auto& reshape4_output =
      tensor_pool.CloneNativeTensorFrom(transpose1_output, {1, 128, 1024});
  auto& reshape4 = op_wrappers.emplace_back(
      CreateReshapeOp(transpose1_output, reshape4_output));

  ASSERT_EQ(op_wrappers.size(), 18);

  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMHAOptPrefill;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });

  ASSERT_EQ(op_wrappers.size(), 49);

  ASSERT_EQ(op_wrappers[0].IsOpCode(QnnOpCode::kTranspose), true);
  ASSERT_EQ(op_wrappers[1].IsOpCode(QnnOpCode::kReshape), true);
  ASSERT_EQ(op_wrappers[2].IsOpCode(QnnOpCode::kSplit), true);
  const size_t sha_size = 11;
  const size_t num_head = 4;
  for (int i = 0; i < num_head; ++i) {
    ASSERT_EQ(op_wrappers[3 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseBinary);
    ASSERT_TRUE(IsElementWiseMultiply(op_wrappers[3 + sha_size * i]));
    ASSERT_EQ(op_wrappers[4 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[5 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[6 + sha_size * i].GetOpCode(), QnnOpCode::kConcat);
    ASSERT_EQ(op_wrappers[7 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseBinary);
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[7 + sha_size * i]));
    ASSERT_EQ(op_wrappers[8 + sha_size * i].GetOpCode(), QnnOpCode::kSoftmax);
    ASSERT_EQ(op_wrappers[9 + sha_size * i].GetOpCode(),
              QnnOpCode::kStridedSlice);
    ASSERT_EQ(op_wrappers[10 + sha_size * i].GetOpCode(),
              QnnOpCode::kStridedSlice);
    ASSERT_EQ(op_wrappers[11 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[12 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[13 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseBinary);
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[13 + sha_size * i]));
  }
  ASSERT_EQ(op_wrappers[47].GetOpCode(), QnnOpCode::kConcat);
  ASSERT_EQ(op_wrappers[48].GetOpCode(), QnnOpCode::kReshape);
}

TEST(MHAOptimization, Gemma3Decode) {
  // G2G Test case: MHA -> SHA
  //
  // ---------------- Before ---------------------
  //                   In0
  //                    |
  //                   Mul
  //      kv_cache_k  /   \  kv_slice_k
  //              \  /     \  /
  //           MatMulK0   MatMulK1
  //                 \     /
  //                  Concat
  //                    |
  //                 Reshape0
  //                    |
  //               mask |
  //                  \ |
  //                   Add0
  //                    |
  //                 Reshape1
  //                    |
  //                 Softmax
  //                  /   \
  //                 /     \
  // kv_cache_v  Slice0   Slice1  kv_slice_v
  //      \        /         \        /
  //       \      /           \      /
  //       MatMulV0           MatMulV1
  //              \           /
  //               \         /
  //                \       /
  //                 \     /
  //                   Add1
  //                    |
  //                 Reshape2
  //                    |
  //                   Out0
  //
  // ---------------- After ---------------------
  //                   In0
  //                    |
  //                  Split
  //                    | \\\
  //                    |  \\\
  //                    |   ...
  //                   Mul
  //      kv_cache_k  /   \  kv_slice_k
  //              \  /     \  /
  //              MatMul  MatMul
  //                 \     /
  //                  Concat
  //                    |
  //                   Add
  //                    |
  //                 Softmax
  //                  /   \
  //                 /     \
  // kv_cache_v  Slice0   Slice1  kv_slice_v
  //      \        /         \        /
  //       \      /           \      /
  //        MatMul             MatMul
  //              \           /
  //               \         /
  //                \       /
  //                 \     /
  //                   Add    ...
  //                    |    ///
  //                    |   ///
  //                    Concat
  //                      |
  //                   Reshape
  //                      |
  //                     Out0
  //
  TensorPool tensor_pool;
  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);
  std::vector<OpWrapper> op_wrappers;
  // Mul
  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 4, 256});
  std::array<int16_t, 1> mul_val = {32767};
  auto& mul_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {mul_val.size()},
      mul_val.size() * sizeof(mul_val[0]), mul_val.data());
  auto& mul_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 4, 256});
  auto& mul = op_wrappers.emplace_back(
      CreateElementWiseMulOp(input0, mul_const, mul_output));

  // MatMulK0
  auto& kv_cache_k = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 1280, 256});
  auto& matmulk0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 4, 1280});
  auto& matmulk0 = op_wrappers.emplace_back(
      CreateMatmulOp(mul_output, kv_cache_k, matmulk0_output, false, true));
  // MatMulK1
  auto& kv_slice_k = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 1, 256});
  auto& matmulk1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 4, 1});
  auto& matmulk1 = op_wrappers.emplace_back(
      CreateMatmulOp(mul_output, kv_slice_k, matmulk1_output, false, true));
  // Concat
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(matmulk0_output, {1, 1, 4, 1281});
  op_wrappers.emplace_back(CreateConcatenationOp(
      {matmulk0_output, matmulk1_output}, concat_output, 3));
  // Reshape0
  auto& reshape0_output =
      tensor_pool.CloneNativeTensorFrom(concat_output, {1, 4, 1, 1281});
  auto& reshape0 =
      op_wrappers.emplace_back(CreateReshapeOp(concat_output, reshape0_output));
  // Add
  auto& add0_output = tensor_pool.CloneNativeTensorFrom(reshape0_output);
  auto& mask = tensor_pool.CloneNativeTensorFrom(reshape0_output);
  op_wrappers.emplace_back(
      CreateElementWiseAddOp(reshape0_output, mask, add0_output));
  // Reshape1
  auto& reshape1_output =
      tensor_pool.CloneNativeTensorFrom(add0_output, {1, 1, 4, 1281});
  auto& reshape1 =
      op_wrappers.emplace_back(CreateReshapeOp(add0_output, reshape1_output));
  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(reshape1_output);
  auto& softmax = op_wrappers.emplace_back(
      CreateSoftmaxOp(reshape1_output, softmax_output, 1.0f));
  // Slice0
  const std::array<int32_t, 12> slice0_ranges_data{0, 1, 1, 0, 1,    1,
                                                   0, 4, 1, 0, 1280, 1};
  auto& slice0_ranges = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {4, 3},
      sizeof(int32_t) * slice0_ranges_data.size(), slice0_ranges_data.data());
  auto& slice0_output =
      tensor_pool.CloneNativeTensorFrom(reshape1_output, {1, 1, 4, 1280});
  op_wrappers.emplace_back(
      CreateSliceOp(softmax_output, slice0_output, slice0_ranges));
  // Slice1
  const std::array<int32_t, 12> slice1_ranges_data{0, 1, 1, 0,    1,    1,
                                                   0, 4, 1, 1280, 1281, 1};
  auto& slice1_ranges = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {4, 3},
      sizeof(int32_t) * slice1_ranges_data.size(), slice1_ranges_data.data());
  auto& slice1_output =
      tensor_pool.CloneNativeTensorFrom(reshape1_output, {1, 1, 4, 256});
  op_wrappers.emplace_back(
      CreateSliceOp(softmax_output, slice1_output, slice1_ranges));
  // MatMulV0
  auto& kv_cache_v = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 256, 1280});
  auto& matmulv0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 4, 256});
  auto& matmulv0 = op_wrappers.emplace_back(
      CreateMatmulOp(slice0_output, kv_cache_v, matmulv0_output, false, true));
  // MatMulV1
  auto& kv_slice_v = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 256, 1});
  auto& matmulv1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 4, 256});
  auto& matmulv1 = op_wrappers.emplace_back(
      CreateMatmulOp(slice1_output, kv_slice_v, matmulv1_output, false, true));
  // Add1
  auto& add1_output = tensor_pool.CloneNativeTensorFrom(matmulv0_output);
  op_wrappers.emplace_back(
      CreateElementWiseAddOp(matmulv0_output, matmulv1_output, add1_output));
  // Reshape2
  auto& reshape2_output =
      tensor_pool.CloneNativeTensorFrom(add1_output, {1, 1, 1024});
  auto& reshape2 =
      op_wrappers.emplace_back(CreateReshapeOp(add1_output, reshape2_output));

  ASSERT_EQ(op_wrappers.size(), 14);

  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMHAOpt;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });

  ASSERT_EQ(op_wrappers.size(), 47);

  ASSERT_EQ(op_wrappers[0].IsOpCode(QnnOpCode::kSplit), true);
  const size_t sha_size = 11;
  const size_t num_head = 4;
  for (int i = 0; i < num_head; ++i) {
    ASSERT_EQ(op_wrappers[1 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseBinary);
    ASSERT_TRUE(IsElementWiseMultiply(op_wrappers[1 + sha_size * i]));
    ASSERT_EQ(op_wrappers[2 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[3 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[4 + sha_size * i].GetOpCode(), QnnOpCode::kConcat);
    ASSERT_EQ(op_wrappers[5 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseBinary);
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[5 + sha_size * i]));
    ASSERT_EQ(op_wrappers[6 + sha_size * i].GetOpCode(), QnnOpCode::kSoftmax);
    ASSERT_EQ(op_wrappers[7 + sha_size * i].GetOpCode(),
              QnnOpCode::kStridedSlice);
    ASSERT_EQ(op_wrappers[8 + sha_size * i].GetOpCode(),
              QnnOpCode::kStridedSlice);
    ASSERT_EQ(op_wrappers[9 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[10 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[11 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseBinary);
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[11 + sha_size * i]));
  }
  ASSERT_EQ(op_wrappers[45].GetOpCode(), QnnOpCode::kConcat);
  ASSERT_EQ(op_wrappers[46].GetOpCode(), QnnOpCode::kReshape);
}

TEST(MaskTransformTest, Gemma3) {
  // G2G Test case:
  //
  // ----- Before -----
  //       In0
  //        |
  //     E-wise Not
  //        |
  //       Cast
  //        |
  //      Quant
  //        |
  //       Mul
  //        |
  //       Out0
  //
  // ----- After -----
  //       In0
  //        |
  //     E-wise Select
  //        |
  //       Out0
  //

  static const std::vector<uint32_t> kDims{1, 1, 128, 1408};
  std::vector<OpWrapper> op_wrappers;
  TensorPool tensor_pool;

  QuantizeParamsWrapperVariant bool_quant_param;
  bool_quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1.0f, 0);

  // not op
  auto& pattern_input = tensor_pool.CreateNativeTensor(QNN_DATATYPE_BOOL_8,
                                                       bool_quant_param, kDims);
  auto& logic_not_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, bool_quant_param, kDims);
  op_wrappers.emplace_back(CreateElementWiseNotOp(pattern_input, logic_not_output));

  // cast op
  auto& cast_output =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kDims);
  op_wrappers.emplace_back(CreateCastOp(logic_not_output, cast_output));

  // quant op
  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(3.05185e-05f, 0);
  auto& quant_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims);
  op_wrappers.emplace_back(CreateQuantizeOp(cast_output, quant_output));

  // mul op
  QuantizeParamsWrapperVariant mul_quant_param;
  constexpr float mul_scale = 0.00305185f;
  constexpr int32_t mul_zero_point = 0;
  mul_quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(mul_scale,
                                                            mul_zero_point);
  auto& pattern_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, mul_quant_param, kDims);

  static const std::array<int16_t, 1 * 1 * 128 * 1408> mul_val{-32767};
  auto& mul_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, mul_quant_param, {mul_val.size()},
      mul_val.size() * sizeof(mul_val[0]), mul_val.data());
  op_wrappers.emplace_back(
      CreateElementWiseMulOp(cast_output, mul_const, pattern_output));

  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMHAOpt;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });
  ASSERT_EQ(op_wrappers.size(), 1);
  ASSERT_TRUE(op_wrappers[0].IsOpCode(QnnOpCode::kElementWiseSelect));
  auto& in_1 = op_wrappers[0].GetInputTensor(1);
  auto& in_2 = op_wrappers[0].GetInputTensor(2);
  ASSERT_TRUE(std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
      in_1.GetQuantParams()));
  ASSERT_TRUE(std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
      in_2.GetQuantParams()));
  auto quant_param_1 =
      std::get<ScaleOffsetQuantizeParamsWrapper>(in_1.GetQuantParams());
  ASSERT_EQ(quant_param_1.GetScale(), mul_scale);
  ASSERT_EQ(quant_param_1.GetZeroPoint(), mul_zero_point);
  auto quant_param_2 =
      std::get<ScaleOffsetQuantizeParamsWrapper>(in_2.GetQuantParams());
  ASSERT_EQ(quant_param_2.GetScale(), mul_scale);
  ASSERT_EQ(quant_param_2.GetZeroPoint(), mul_zero_point);
}

TEST(MHASHATest, FastVlm) {
  // G2G Test case: MHA -> SHA

  // ------------------- Before ---------------------
  //                       In0
  //                        |
  //                       Mul
  //                        |
  //                     Reshape0  In1   In2
  //                      /   \     \    /
  //             QIn     /     \     Add0
  //               \    /       \     /
  //                Matmul0     Matmul1
  //                    \       /
  //                     Concat
  //                        |
  //                     Reshape1
  //                        |
  //                   Mask |
  //                      \ |
  //                       Add1
  //                        |
  //                     Reshape2
  //                        |
  //                     Softmax
  //                    /       \
  //                  Slice0  Slice1 In3
  //                    |       |     |
  //              VIn   |       |  Transpose0
  //                \   |       |    /
  //                 Matmul2  Matmul3
  //                     \     /
  //                       Add2
  //                        |
  //                     Reshape3
  //                        |
  //                     Transpose1
  //                        |
  //                     Reshape4
  //                        |
  //                       Out
  //
  // -------------------- After ---------------------
  //                       In0
  //                        |
  //                       Mul    In1  In2
  //                      /   \     \  /
  //              QIn    /     \    Add0
  //                \   /       \   /
  //                Matmul0   Matmul1
  //                    \       /
  //                     Concat
  //                        |
  //                   Mask |
  //                      \ |
  //                       Add1
  //                        |
  //                     Reshape
  //                        |
  //                     Softmax
  //                    /       \
  //                  Slice0  Slice1
  //                    |       |
  //              VIn   |       |    In3
  //                \   |       |    /
  //                 Matmul2  Matmul3
  //                     \     /
  //                       Add2
  //                        |
  //                       Out
  TensorPool tensor_pool;
  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);
  std::vector<OpWrapper> op_wrappers;

  // Add0
  auto& input1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 2, 128, 64});
  auto& input2 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 2, 128, 64});
  auto& add0_output =
      tensor_pool.CloneNativeTensorFrom(input1, {1, 2, 128, 64});
  op_wrappers.emplace_back(CreateElementWiseAddOp(input1, input2, add0_output));

  // Transpose0
  auto& input3 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 128, 2, 64});
  std::array<std::uint32_t, 4> transpose0_val = {0, 2, 3, 1};
  auto& transpose0_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, quant_param, {transpose0_val.size()},
      transpose0_val.size() * sizeof(transpose0_val[0]), transpose0_val.data());
  auto& transpose0_output =
      tensor_pool.CloneNativeTensorFrom(add0_output, {1, 2, 64, 128});
  auto& transpose0 = op_wrappers.emplace_back(
      CreateTransposeOp(input3, transpose0_output, transpose0_perm));

  // Mul
  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 14, 128, 64});
  std::array<int16_t, 1> mul_val = {32767};
  auto& mul_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {mul_val.size()},
      mul_val.size() * sizeof(mul_val[0]), mul_val.data());
  auto& mul_output =
      tensor_pool.CloneNativeTensorFrom(input0, {1, 14, 128, 64});
  op_wrappers.emplace_back(
      CreateElementWiseMulOp(input0, mul_const, mul_output));

  // Reshape0
  auto& reshape0_output =
      tensor_pool.CloneNativeTensorFrom(mul_output, {1, 2, 896, 64});
  auto& reshape0 =
      op_wrappers.emplace_back(CreateReshapeOp(mul_output, reshape0_output));

  // MatMul0
  auto& q_in = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 2, 1280, 64});
  auto& matmul0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 2, 896, 1280});
  auto& matmul0 = op_wrappers.emplace_back(
      CreateMatmulOp(reshape0_output, q_in, matmul0_output, false, true));

  // MatMul1
  auto& matmul1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 2, 896, 128});
  auto& matmul1 = op_wrappers.emplace_back(CreateMatmulOp(
      reshape0_output, add0_output, matmul1_output, false, true));

  // Concat
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(matmul0_output, {1, 2, 896, 1408});
  op_wrappers.emplace_back(CreateConcatenationOp(
      {matmul0_output, matmul1_output}, concat_output, 3));

  // Reshape1
  auto& reshape1_output =
      tensor_pool.CloneNativeTensorFrom(concat_output, {2, 7, 128, 1408});
  auto& reshape1 =
      op_wrappers.emplace_back(CreateReshapeOp(concat_output, reshape1_output));

  // Add1
  auto& mask = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 1, 128, 1408});
  auto& add1_output = tensor_pool.CloneNativeTensorFrom(reshape1_output);
  op_wrappers.emplace_back(
      CreateElementWiseAddOp(reshape1_output, mask, add1_output));

  // Reshape2
  auto& reshape2_output =
      tensor_pool.CloneNativeTensorFrom(add1_output, {1, 2, 896, 1408});
  auto& reshape2 =
      op_wrappers.emplace_back(CreateReshapeOp(add1_output, reshape2_output));

  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(reshape2_output);
  auto& softmax = op_wrappers.emplace_back(
      CreateSoftmaxOp(reshape2_output, softmax_output, 1.0f));

  // Slice0
  const std::array<int32_t, 12> slice0_ranges_data{0, 1,   1, 0, 2,    1,
                                                   0, 896, 1, 0, 1280, 1};
  auto& slice0_ranges = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {4, 3},
      sizeof(int32_t) * slice0_ranges_data.size(), slice0_ranges_data.data());
  auto& slice0_output =
      tensor_pool.CloneNativeTensorFrom(softmax_output, {1, 2, 896, 1280});
  op_wrappers.emplace_back(
      CreateSliceOp(softmax_output, slice0_output, slice0_ranges));

  // Slice1
  const std::array<int32_t, 12> slice1_ranges_data{0, 1,   1, 0,    2,    1,
                                                   0, 896, 1, 1280, 1408, 1};
  auto& slice1_ranges = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {4, 3},
      sizeof(int32_t) * slice1_ranges_data.size(), slice1_ranges_data.data());
  auto& slice1_output =
      tensor_pool.CloneNativeTensorFrom(softmax_output, {1, 2, 896, 128});
  op_wrappers.emplace_back(
      CreateSliceOp(softmax_output, slice1_output, slice1_ranges));

  // MatMul2
  auto& v_in = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 2, 64, 1280});
  auto& matmul2_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 2, 896, 64});
  auto& matmul2 = op_wrappers.emplace_back(
      CreateMatmulOp(slice0_output, v_in, matmul2_output, false, true));

  // MatMul3
  auto& matmul3_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 2, 896, 64});
  auto& matmul3 = op_wrappers.emplace_back(CreateMatmulOp(
      slice1_output, transpose0_output, matmul3_output, false, true));

  // Add2
  auto& add2_output = tensor_pool.CloneNativeTensorFrom(matmul3_output);
  op_wrappers.emplace_back(
      CreateElementWiseAddOp(matmul2_output, matmul3_output, add2_output));

  // Reshape3
  auto& reshape3_output =
      tensor_pool.CloneNativeTensorFrom(add2_output, {1, 14, 128, 64});
  auto& reshape3 =
      op_wrappers.emplace_back(CreateReshapeOp(add2_output, reshape3_output));

  // Transpose1
  std::array<std::uint32_t, 4> transpose1_val = {0, 2, 1, 3};
  auto& transpose1_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, quant_param, {transpose1_val.size()},
      transpose1_val.size() * sizeof(transpose1_val[0]), transpose1_val.data());
  auto& transpose1_output =
      tensor_pool.CloneNativeTensorFrom(reshape3_output, {1, 128, 14, 64});
  auto& transpose1 = op_wrappers.emplace_back(
      CreateTransposeOp(reshape3_output, transpose1_output, transpose1_perm));

  // Reshape4
  auto& reshape4_output =
      tensor_pool.CloneNativeTensorFrom(transpose1_output, {1, 128, 896});
  auto& reshape4 = op_wrappers.emplace_back(
      CreateReshapeOp(transpose1_output, reshape4_output));

  ASSERT_EQ(op_wrappers.size(), 19);

  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMHAOptPrefill;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });
  // Check total size after G2G
  ASSERT_EQ(op_wrappers.size(), 191);

  // Check OpCode after G2G
  const size_t num_unpack = 6;
  const size_t num_head = 14;
  const size_t sha_size = 13;

  ASSERT_TRUE(op_wrappers[0].IsOpCode(QnnOpCode::kElementWiseBinary));
  ASSERT_TRUE(IsElementWiseAdd(op_wrappers[0]));
  ASSERT_TRUE(op_wrappers[1].IsOpCode(QnnOpCode::kTranspose));

  for (size_t i = 0; i < num_unpack; ++i) {
    ASSERT_TRUE(op_wrappers[2 + i].IsOpCode(QnnOpCode::kUnPack));
  }

  for (size_t i = 0; i < num_head; ++i) {
    ASSERT_TRUE(
        op_wrappers[8 + sha_size * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseMultiply(op_wrappers[8 + sha_size * i]));
    ASSERT_TRUE(op_wrappers[9 + sha_size * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(
        op_wrappers[10 + sha_size * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[10 + sha_size * i]));
    ASSERT_TRUE(op_wrappers[11 + sha_size * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(op_wrappers[12 + sha_size * i].IsOpCode(QnnOpCode::kConcat));
    ASSERT_TRUE(
        op_wrappers[13 + sha_size * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[13 + sha_size * i]));
    ASSERT_TRUE(op_wrappers[14 + sha_size * i].IsOpCode(QnnOpCode::kReshape));
    ASSERT_TRUE(op_wrappers[15 + sha_size * i].IsOpCode(QnnOpCode::kSoftmax));
    ASSERT_TRUE(
        op_wrappers[16 + sha_size * i].IsOpCode(QnnOpCode::kStridedSlice));
    ASSERT_TRUE(
        op_wrappers[17 + sha_size * i].IsOpCode(QnnOpCode::kStridedSlice));
    ASSERT_TRUE(op_wrappers[18 + sha_size * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(op_wrappers[19 + sha_size * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(
        op_wrappers[20 + sha_size * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[20 + sha_size * i]));
  }
  ASSERT_TRUE(op_wrappers[op_wrappers.size() - 1].IsOpCode(QnnOpCode::kConcat));
}

TEST(MHAOptimization, TinyGemma3Prefill) {
  // G2G Test case: MHA -> SHA
  //
  // ---------------- Before ---------------------
  //                   In0
  //                    |
  //                   Mul
  //                    |
  //               Transpose0
  //                    |
  //                Reshape0
  //      kv_cache_k  /  \  kv_slice_k
  //              \  /    \  /
  //           MatMulK0  MatMulK1
  //                 \    /
  //                  Concat
  //               mask |
  //                  \ |
  //                   Add0
  //                    |
  //                 Softmax
  //                  /  \
  // kv_cache_v  Slice0  Slice1  kv_slice_v
  //      \        /        \        /
  //       MatMulV0          MatMulV1
  //              \          /
  //               \        /
  //                  Add1
  //                    |
  //                 Reshape1
  //                    |
  //                Transpose1
  //                    |
  //                 Reshape2
  //                    |
  //                   Out0
  //
  // ---------------- After ---------------------
  //                   In0
  //                    |
  //                Transpose
  //                    |
  //                 Reshape
  //                    |
  //                  Split
  //                    | \\\
  //                    |  ...
  //                   Mul
  //      kv_cache_k  /  \  kv_slice_k
  //              \  /    \  /
  //             MatMul  MatMul
  //                 \    /
  //                 Concat
  //                    |
  //                   Add
  //                    |
  //                 Softmax
  //                 /    \
  // kv_cache_v  Slice0  Slice1  kv_slice_v
  //        \      /        \      /
  //         MatMul          MatMul
  //               \        /
  //                \      /
  //                  Add    ...
  //                   |    ///
  //                   |   ///
  //                   Concat
  //                     |
  //                  Reshape
  //                     |
  //                    Out0
  //
  TensorPool tensor_pool;
  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);
  std::vector<OpWrapper> op_wrappers;

  // Mul
  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 128, 4, 256});
  std::array<int16_t, 1> mul_val = {32767};
  auto& mul_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {mul_val.size()},
      mul_val.size() * sizeof(decltype(mul_val)::value_type), mul_val.data());
  auto& mul_output =
      tensor_pool.CloneNativeTensorFrom(input0, {1, 128, 4, 256});
  op_wrappers.emplace_back(
      CreateElementWiseMulOp(input0, mul_const, mul_output));
  // Transpose0
  std::array<std::uint32_t, 4> transpose_val = {0, 2, 1, 3};
  auto& transpose_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, quant_param, {transpose_val.size()},
      transpose_val.size() * sizeof(decltype(transpose_val)::value_type),
      transpose_val.data());
  auto& transpose0_output =
      tensor_pool.CloneNativeTensorFrom(mul_output, {1, 4, 128, 256});
  auto& transpose0 = op_wrappers.emplace_back(
      CreateTransposeOp(mul_output, transpose0_output, transpose_perm));
  // Reshape0
  auto& reshape0_output =
      tensor_pool.CloneNativeTensorFrom(transpose0_output, {1, 1, 512, 256});
  auto& reshape0 = op_wrappers.emplace_back(
      CreateReshapeOp(transpose0_output, reshape0_output));
  // MatMulK0
  auto& kv_cache_k = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 1280, 256});
  auto& matmulk0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 1280});
  auto& matmulk0 = op_wrappers.emplace_back(CreateMatmulOp(
      reshape0_output, kv_cache_k, matmulk0_output, false, true));
  // MatMulK1
  auto& kv_slice_k = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 128, 256});
  auto& matmulk1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 128});
  auto& matmulk1 = op_wrappers.emplace_back(CreateMatmulOp(
      reshape0_output, kv_slice_k, matmulk1_output, false, true));
  // Concat
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(matmulk0_output, {1, 1, 512, 1408});
  op_wrappers.emplace_back(CreateConcatenationOp(
      {matmulk0_output, matmulk1_output}, concat_output, 3));
  // Add
  auto& add0_output = tensor_pool.CloneNativeTensorFrom(concat_output);
  auto& mask = tensor_pool.CloneNativeTensorFrom(concat_output);
  op_wrappers.emplace_back(
      CreateElementWiseAddOp(concat_output, mask, add0_output));
  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(add0_output);
  auto& softmax = op_wrappers.emplace_back(
      CreateSoftmaxOp(add0_output, softmax_output, 1.0f));
  // Slice0
  const std::array<int32_t, 12> slice0_ranges_data{0, 1,   1, 0, 1,    1,
                                                   0, 512, 1, 0, 1280, 1};
  auto& slice0_ranges = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {4, 3},
      sizeof(int32_t) * slice0_ranges_data.size(), slice0_ranges_data.data());
  auto& slice0_output =
      tensor_pool.CloneNativeTensorFrom(softmax_output, {1, 1, 512, 1280});
  op_wrappers.emplace_back(
      CreateSliceOp(softmax_output, slice0_output, slice0_ranges));
  // Slice1
  const std::array<int32_t, 12> slice1_ranges_data{0, 1,   1, 0,    1,    1,
                                                   0, 512, 1, 1280, 1408, 1};
  auto& slice1_ranges = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {4, 3},
      sizeof(int32_t) * slice1_ranges_data.size(), slice1_ranges_data.data());
  auto& slice1_output =
      tensor_pool.CloneNativeTensorFrom(softmax_output, {1, 1, 512, 128});
  op_wrappers.emplace_back(
      CreateSliceOp(softmax_output, slice1_output, slice1_ranges));
  // MatMulV0
  auto& kv_cache_v = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 256, 1280});
  auto& matmulv0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 256});
  auto& matmulv0 = op_wrappers.emplace_back(
      CreateMatmulOp(slice0_output, kv_cache_v, matmulv0_output, false, true));
  // MatMulV1
  auto& kv_slice_v = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 256, 128});
  auto& matmulv1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 256});
  auto& matmulv1 = op_wrappers.emplace_back(
      CreateMatmulOp(slice1_output, kv_slice_v, matmulv1_output, false, true));
  // Add1
  auto& add1_output = tensor_pool.CloneNativeTensorFrom(matmulv0_output);
  op_wrappers.emplace_back(
      CreateElementWiseAddOp(matmulv0_output, matmulv1_output, add1_output));
  // Reshape1
  auto& reshape1_output =
      tensor_pool.CloneNativeTensorFrom(add1_output, {1, 4, 128, 256});
  auto& reshape3 =
      op_wrappers.emplace_back(CreateReshapeOp(add1_output, reshape1_output));
  // Transpose
  auto& transpose1_output =
      tensor_pool.CloneNativeTensorFrom(reshape1_output, {1, 128, 4, 256});
  auto& transpose1 = op_wrappers.emplace_back(
      CreateTransposeOp(reshape1_output, transpose1_output, transpose_perm));
  // Reshape2
  auto& reshape2_output =
      tensor_pool.CloneNativeTensorFrom(add1_output, {1, 128, 1024});
  auto& reshape2 = op_wrappers.emplace_back(
      CreateReshapeOp(transpose1_output, reshape2_output));

  ASSERT_EQ(op_wrappers.size(), 16);

  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMHAOptPrefill;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });

  ASSERT_EQ(op_wrappers.size(), 49);

  ASSERT_TRUE(op_wrappers[0].IsOpCode(QnnOpCode::kTranspose));
  ASSERT_TRUE(op_wrappers[1].IsOpCode(QnnOpCode::kSplit));
  ASSERT_TRUE(op_wrappers[2].IsOpCode(QnnOpCode::kSplit));  // split mask
  const size_t sha_size = 11;
  const size_t num_head = 4;
  for (int i = 0; i < num_head; ++i) {
    ASSERT_EQ(op_wrappers[3 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseBinary);
    ASSERT_TRUE(IsElementWiseMultiply(op_wrappers[3 + sha_size * i]));
    ASSERT_EQ(op_wrappers[4 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[5 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[6 + sha_size * i].GetOpCode(), QnnOpCode::kConcat);
    ASSERT_EQ(op_wrappers[7 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseBinary);
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[7 + sha_size * i]));
    ASSERT_EQ(op_wrappers[8 + sha_size * i].GetOpCode(), QnnOpCode::kSoftmax);
    ASSERT_EQ(op_wrappers[9 + sha_size * i].GetOpCode(),
              QnnOpCode::kStridedSlice);
    ASSERT_EQ(op_wrappers[10 + sha_size * i].GetOpCode(),
              QnnOpCode::kStridedSlice);
    ASSERT_EQ(op_wrappers[11 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[12 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[13 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseBinary);
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[13 + sha_size * i]));
  }
  ASSERT_EQ(op_wrappers[47].GetOpCode(), QnnOpCode::kConcat);
  ASSERT_EQ(op_wrappers[48].GetOpCode(), QnnOpCode::kReshape);
}

TEST(MHAOptimization, AttentionWithSelect) {
  // G2G Test case:
  //
  // -------------------- Before --------------------
  //       In0        In1
  //        |          |
  //       Mul        Mul       In2
  //        |          |         |
  //    Transpose  Transpose  Reshape
  //         \        /          |
  //          \      /           |
  //           MatMul         NotEqual
  //                \        /        \
  //                 \      /          \
  //                  Select    In3 (MHA with Select)
  //                    |        |
  //                 Softmax Transpose
  //                     \      /
  //                      MatMul
  //                        |
  //                    Transpose
  //                        |
  //                       Out0
  //
  // --------------------- After ---------------------
  //       In0     In1    In3       In2
  //        |       |      |         |
  //      Unpack  Unpack Unpack   NotEqual
  //        |       |      |     /
  //         \      |     /    Cast
  //          \     |    /     /
  //           |    |   |    Mul
  //           |    |   |   /
  //           (MHA-to-SHAs)
  //                |
  //               Pack
  //                |
  //               Out0
  //
  // SHA:
  //       In0'    In1'    In2
  //        |       |       |
  //       Mul     Mul   NotEqual
  //        |       |       |
  //        |   Transpose  Cast
  //        |     /       /
  //        MatMul     Mul
  //              \   /   \
  //               Add     (MHA with Add)
  //                |
  //             Softmax
  //                |
  //              MatMul
  //                |
  //               Out0'
  //
  TensorPool tensor_pool;
  std::vector<OpWrapper> op_wrappers;

  auto& in0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {},
                                             {1, 1024, 5, 128});
  auto& in1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {},
                                             {1, 1108, 5, 128});
  auto& in2 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {},
                                             {1, 1024, 1108});
  auto& in3 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {},
                                             {1, 1108, 5, 128});
  const std::array<float, 1> mul_val{0.5};
  auto& mul_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, {}, {mul_val.size()},
      mul_val.size() * sizeof(mul_val[0]), mul_val.data());
  // Mul0
  auto& mul_q0_trans = tensor_pool.CloneNativeTensorFrom(in0);
  op_wrappers.emplace_back(
      CreateElementWiseMulOp(in0, mul_const, mul_q0_trans));
  // Mul1
  auto& mul_k0_trans = tensor_pool.CloneNativeTensorFrom(in1);
  op_wrappers.emplace_back(
      CreateElementWiseMulOp(in1, mul_const, mul_k0_trans));
  // Transpose2
  const std::array<std::uint32_t, 4> transpose_q0_val = {0, 2, 1, 3};
  auto& transpose_q0_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, {transpose_q0_val.size()},
      transpose_q0_val.size() * sizeof(transpose_q0_val[0]),
      transpose_q0_val.data());
  auto& transpose_q0_matmul =
      tensor_pool.CloneNativeTensorFrom(mul_q0_trans, {1, 5, 1024, 128});
  auto& transpose_q0 = op_wrappers.emplace_back(
      CreateTransposeOp(mul_q0_trans, transpose_q0_matmul, transpose_q0_perm));
  // Transpose3
  const std::array<std::uint32_t, 4> transpose_k0_val = {0, 2, 3, 1};
  auto& transpose_k0_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, {transpose_k0_val.size()},
      transpose_k0_val.size() * sizeof(transpose_k0_val[0]),
      transpose_k0_val.data());
  auto& transpose_k0_matmul =
      tensor_pool.CloneNativeTensorFrom(mul_k0_trans, {1, 5, 128, 1108});
  auto& transpose_k0 = op_wrappers.emplace_back(
      CreateTransposeOp(mul_k0_trans, transpose_k0_matmul, transpose_k0_perm));
  // MatMul4
  auto& matmul_select = tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32,
                                                       {}, {1, 5, 1024, 1108});
  auto& matmul0 = op_wrappers.emplace_back(CreateMatmulOp(
      transpose_q0_matmul, transpose_k0_matmul, matmul_select, false, false));
  // Reshape5
  auto& reshape_notequal =
      tensor_pool.CloneNativeTensorFrom(in2, {1, 1, 1024, 1108});
  auto& reshape0 =
      op_wrappers.emplace_back(CreateReshapeOp(in2, reshape_notequal));
  // NotEqual6
  auto& notequal_select = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, {1, 1, 1024, 1108});
  const std::array<float, 1> zero_val{0};
  auto& zero_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, {}, {zero_val.size()},
      zero_val.size() * sizeof(zero_val[0]), zero_val.data());
  op_wrappers.emplace_back(CreateElementWiseNotEqualOp(
      reshape_notequal, zero_const, notequal_select));
  // Select7
  auto& select_softmax = tensor_pool.CloneNativeTensorFrom(matmul_select);
  const std::array<float, 1> mask_val{-2.38197633e+38};
  auto& mask_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, {}, {mask_val.size()},
      mask_val.size() * sizeof(mask_val[0]), mask_val.data());
  op_wrappers.emplace_back(CreateSelectOp(notequal_select, matmul_select,
                                          mask_const, select_softmax));
  // Softmax8
  auto& softmax_matmul = tensor_pool.CloneNativeTensorFrom(select_softmax);
  auto& softmax = op_wrappers.emplace_back(
      CreateSoftmaxOp(select_softmax, softmax_matmul, 1.0f));
  // Transpose9
  const std::array<std::uint32_t, 4> transpose_v0_val = {0, 2, 3, 1};
  auto& transpose_v0_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, {transpose_v0_val.size()},
      transpose_v0_val.size() * sizeof(transpose_v0_val[0]),
      transpose_v0_val.data());
  auto& transpose_v0_matmul =
      tensor_pool.CloneNativeTensorFrom(in3, {1, 5, 128, 1108});
  auto& transpose_v0 = op_wrappers.emplace_back(
      CreateTransposeOp(in3, transpose_v0_matmul, transpose_v0_perm));
  // MatMul10
  auto& matmul_transpose = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_FLOAT_32, {}, {1, 5, 128, 1024});
  auto& matmul_v = op_wrappers.emplace_back(CreateMatmulOp(
      transpose_v0_matmul, softmax_matmul, matmul_transpose, false, true));
  // Transpose11
  const std::array<std::uint32_t, 4> transpose_o0_val = {0, 3, 1, 2};
  auto& transpose_o0_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, {transpose_o0_val.size()},
      transpose_o0_val.size() * sizeof(transpose_o0_val[0]),
      transpose_o0_val.data());
  auto& transpose_o0_out =
      tensor_pool.CloneNativeTensorFrom(matmul_transpose, {1, 1024, 5, 128});
  auto& transpose_o0 = op_wrappers.emplace_back(
      CreateTransposeOp(matmul_transpose, transpose_o0_out, transpose_o0_perm));

  // Transform the graph.
  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMHAOpt;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });
  // Check the optimized graph is correct.
  ASSERT_EQ(op_wrappers.size(), 42);
  ASSERT_TRUE(op_wrappers[0].IsOpCode(QnnOpCode::kElementWiseBinary));
  ASSERT_TRUE(op_wrappers[1].IsOpCode(QnnOpCode::kCast));
  ASSERT_TRUE(op_wrappers[2].IsOpCode(QnnOpCode::kElementWiseBinary));
  ASSERT_TRUE(IsElementWiseMultiply(op_wrappers[2]));
  ASSERT_TRUE(op_wrappers[3].IsOpCode(QnnOpCode::kUnPack));
  ASSERT_TRUE(op_wrappers[4].IsOpCode(QnnOpCode::kUnPack));
  ASSERT_TRUE(op_wrappers[5].IsOpCode(QnnOpCode::kUnPack));

  const std::vector<QnnOpCode> sha_op_codes = {QnnOpCode::kElementWiseBinary,
                                               QnnOpCode::kElementWiseBinary,
                                               QnnOpCode::kTranspose,
                                               QnnOpCode::kMatMul,
                                               QnnOpCode::kElementWiseBinary,
                                               QnnOpCode::kSoftmax,
                                               QnnOpCode::kMatMul};
  for (int i = 6; i < 41; i = i + sha_op_codes.size()) {
    for (size_t index = 0; index < sha_op_codes.size(); ++index) {
      ASSERT_TRUE(op_wrappers[i + index].IsOpCode(sha_op_codes[index]));
    }
    ASSERT_TRUE(IsElementWiseMultiply(op_wrappers[i]));
    ASSERT_TRUE(IsElementWiseMultiply(op_wrappers[i + 1]));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[i + 4]));
  }
  ASSERT_TRUE(op_wrappers[41].IsOpCode(QnnOpCode::kPack));
}

}  // namespace
}  // namespace qnn
