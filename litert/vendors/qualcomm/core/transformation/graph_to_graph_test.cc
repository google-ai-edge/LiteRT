// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/graph_to_graph.h"

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
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

  // MatMul0: In0, In1, matmul_to_concat
  std::vector<::qnn::TensorWrapperRef> matmul0_inputs;
  matmul0_inputs.emplace_back(input0);
  matmul0_inputs.emplace_back(input1);
  std::vector<::qnn::TensorWrapperRef> matmul0_outputs;
  matmul0_outputs.emplace_back(matmul_to_concat);
  auto matmul0 =
      BuildMatmulOp(tensor_pool, matmul0_inputs, matmul0_outputs, false, true);
  std::move(matmul0.begin(), matmul0.end(), std::back_inserter(op_wrappers));

  // MatMul1: In1, In2, Out1
  std::vector<::qnn::TensorWrapperRef> matmul1_inputs;
  matmul1_inputs.emplace_back(input0);
  matmul1_inputs.emplace_back(input2);
  std::vector<::qnn::TensorWrapperRef> matmul1_outputs;
  matmul1_outputs.emplace_back(output1);
  auto matmul1 =
      BuildMatmulOp(tensor_pool, matmul1_inputs, matmul1_outputs, false, true);
  std::move(matmul1.begin(), matmul1.end(), std::back_inserter(op_wrappers));

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

  // MatMul0: In0, In1, matmul_to_concat
  std::vector<::qnn::TensorWrapperRef> matmul0_inputs;
  matmul0_inputs.emplace_back(input0);
  matmul0_inputs.emplace_back(input1);
  std::vector<::qnn::TensorWrapperRef> matmul0_outputs;
  matmul0_outputs.emplace_back(matmul_to_concat);
  auto matmul0 =
      BuildMatmulOp(tensor_pool, matmul0_inputs, matmul0_outputs, false, true);
  std::move(matmul0.begin(), matmul0.end(), std::back_inserter(op_wrappers));

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

TEST(MHAConvertTest, Gemma3Prefill) {
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
  auto mul =
      BuildElementwiseMulOp(tensor_pool, {input0, mul_const}, {mul_output});
  std::move(mul.begin(), mul.end(), std::back_inserter(op_wrappers));
  // Transpose0
  std::array<int32_t, 4> transpose_val = {0, 2, 1, 3};
  auto& transpose_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, quant_param, {transpose_val.size()},
      transpose_val.size() * sizeof(transpose_val[0]), transpose_val.data());
  auto& transpose0_output =
      tensor_pool.CloneNativeTensorFrom(mul_output, {1, 4, 128, 256});
  auto transpose0 = BuildTransposeOp(tensor_pool, {mul_output, transpose_perm},
                                     {transpose0_output});
  std::move(transpose0.begin(), transpose0.end(),
            std::back_inserter(op_wrappers));

  // Reshape0
  auto& reshape0_output =
      tensor_pool.CloneNativeTensorFrom(transpose0_output, {1, 1, 512, 256});
  auto reshape0 =
      BuildReshapeOp(tensor_pool, {transpose0_output}, {reshape0_output});
  std::move(reshape0.begin(), reshape0.end(), std::back_inserter(op_wrappers));

  // MatMulK0
  auto& kv_cache_k = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 1280, 256});
  auto& matmulk0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 1280});
  auto matmulk0 = BuildMatmulOp(tensor_pool, {reshape0_output, kv_cache_k},
                                {matmulk0_output}, false, true);
  std::move(matmulk0.begin(), matmulk0.end(), std::back_inserter(op_wrappers));
  // MatMulK1
  auto& kv_slice_k = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 128, 256});
  auto& matmulk1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 128});
  auto matmulk1 = BuildMatmulOp(tensor_pool, {reshape0_output, kv_slice_k},
                                {matmulk1_output}, false, true);
  std::move(matmulk1.begin(), matmulk1.end(), std::back_inserter(op_wrappers));
  // Concat
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(matmulk0_output, {1, 1, 512, 1408});
  auto concat = BuildConcatenationOp(
      tensor_pool, {matmulk0_output, matmulk1_output}, {concat_output}, 3);
  std::move(concat.begin(), concat.end(), std::back_inserter(op_wrappers));
  // Reshape1
  auto& reshape1_output =
      tensor_pool.CloneNativeTensorFrom(concat_output, {1, 4, 128, 1408});
  auto reshape1 =
      BuildReshapeOp(tensor_pool, {concat_output}, {reshape1_output});
  std::move(reshape1.begin(), reshape1.end(), std::back_inserter(op_wrappers));
  // Add
  auto& add0_output = tensor_pool.CloneNativeTensorFrom(reshape1_output);
  auto& mask = tensor_pool.CloneNativeTensorFrom(reshape1_output);
  auto add0 = BuildElementwiseAddOp(tensor_pool, {reshape1_output, mask},
                                    {add0_output});
  std::move(add0.begin(), add0.end(), std::back_inserter(op_wrappers));
  // Reshape2
  auto& reshape2_output =
      tensor_pool.CloneNativeTensorFrom(add0_output, {1, 1, 512, 1408});
  auto reshape2 = BuildReshapeOp(tensor_pool, {add0_output}, {reshape2_output});
  std::move(reshape2.begin(), reshape2.end(), std::back_inserter(op_wrappers));
  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(reshape2_output);
  auto softmax =
      BuildSoftmaxOp(tensor_pool, {reshape2_output}, {softmax_output}, 1.0f);
  std::move(softmax.begin(), softmax.end(), std::back_inserter(op_wrappers));
  // Slice0
  const std::array<int32_t, 4> slice0_begin_data{0, 0, 0, 0};
  auto& slice0_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {slice0_begin_data.size()},
      slice0_begin_data.size() * sizeof(slice0_begin_data[0]),
      slice0_begin_data.data());
  const std::array<int32_t, 4> slice0_size_data{1, 1, 512, 1280};
  auto& slice0_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {slice0_size_data.size()},
      slice0_size_data.size() * sizeof(slice0_size_data[0]),
      slice0_size_data.data());
  auto& slice0_output =
      tensor_pool.CloneNativeTensorFrom(reshape2_output, {1, 1, 512, 1280});
  auto slice0 =
      BuildSliceOp(tensor_pool, {softmax_output, slice0_begin, slice0_size},
                   {slice0_output});
  std::move(slice0.begin(), slice0.end(), std::back_inserter(op_wrappers));
  // Slice1
  const std::array<int32_t, 4> slice1_begin_data{0, 0, 0, 1280};
  auto& slice1_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {slice1_begin_data.size()},
      slice1_begin_data.size() * sizeof(slice1_begin_data[0]),
      slice1_begin_data.data());
  ;
  const std::array<int32_t, 4> slice1_size_data{1, 1, 512, 128};
  auto& slice1_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {slice1_size_data.size()},
      slice1_size_data.size() * sizeof(slice1_size_data[0]),
      slice1_size_data.data());
  auto& slice1_output =
      tensor_pool.CloneNativeTensorFrom(reshape2_output, {1, 1, 512, 128});
  auto slice1 =
      BuildSliceOp(tensor_pool, {softmax_output, slice1_begin, slice1_size},
                   {slice1_output});
  std::move(slice1.begin(), slice1.end(), std::back_inserter(op_wrappers));
  // MatMulV0
  auto& kv_cache_v = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 256, 1280});
  auto& matmulv0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 256});
  auto matmulv0 = BuildMatmulOp(tensor_pool, {slice0_output, kv_cache_v},
                                {matmulv0_output}, false, true);
  std::move(matmulv0.begin(), matmulv0.end(), std::back_inserter(op_wrappers));
  // MatMulV1
  auto& kv_slice_v = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 256, 128});
  auto& matmulv1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 256});
  auto matmulv1 = BuildMatmulOp(tensor_pool, {slice1_output, kv_slice_v},
                                {matmulv1_output}, false, true);
  std::move(matmulv1.begin(), matmulv1.end(), std::back_inserter(op_wrappers));
  // Add1
  auto& add1_output = tensor_pool.CloneNativeTensorFrom(matmulv0_output);
  auto add1 = BuildElementwiseAddOp(
      tensor_pool, {matmulv0_output, matmulv1_output}, {add1_output});
  std::move(add1.begin(), add1.end(), std::back_inserter(op_wrappers));
  // Reshape3
  auto& reshape3_output =
      tensor_pool.CloneNativeTensorFrom(add1_output, {1, 4, 128, 256});
  auto reshape3 = BuildReshapeOp(tensor_pool, {add1_output}, {reshape3_output});
  std::move(reshape3.begin(), reshape3.end(), std::back_inserter(op_wrappers));
  // Transpose
  auto& transpose1_output =
      tensor_pool.CloneNativeTensorFrom(reshape3_output, {1, 128, 4, 256});
  auto transpose1 = BuildTransposeOp(
      tensor_pool, {reshape3_output, transpose_perm}, {transpose1_output});
  std::move(transpose1.begin(), transpose1.end(),
            std::back_inserter(op_wrappers));
  // Reshape4
  auto& reshape4_output =
      tensor_pool.CloneNativeTensorFrom(transpose1_output, {1, 128, 1024});
  auto reshape4 =
      BuildReshapeOp(tensor_pool, {transpose1_output}, {reshape4_output});
  std::move(reshape4.begin(), reshape4.end(), std::back_inserter(op_wrappers));

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
              QnnOpCode::kElementWiseMultiply);
    ASSERT_EQ(op_wrappers[4 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[5 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[6 + sha_size * i].GetOpCode(), QnnOpCode::kConcat);
    ASSERT_EQ(op_wrappers[7 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseAdd);
    ASSERT_EQ(op_wrappers[8 + sha_size * i].GetOpCode(), QnnOpCode::kSoftmax);
    ASSERT_EQ(op_wrappers[9 + sha_size * i].GetOpCode(),
              QnnOpCode::kStridedSlice);
    ASSERT_EQ(op_wrappers[10 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[11 + sha_size * i].GetOpCode(),
              QnnOpCode::kStridedSlice);
    ASSERT_EQ(op_wrappers[12 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[13 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseAdd);
  }
  ASSERT_EQ(op_wrappers[47].GetOpCode(), QnnOpCode::kConcat);
  ASSERT_EQ(op_wrappers[48].GetOpCode(), QnnOpCode::kReshape);
}

TEST(MHAConvertTest, Gemma3Decode) {
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
  auto mul =
      BuildElementwiseMulOp(tensor_pool, {input0, mul_const}, {mul_output});
  std::move(mul.begin(), mul.end(), std::back_inserter(op_wrappers));

  // MatMulK0
  auto& kv_cache_k = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 1280, 256});
  auto& matmulk0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 4, 1280});
  auto matmulk0 = BuildMatmulOp(tensor_pool, {mul_output, kv_cache_k},
                                {matmulk0_output}, false, true);
  std::move(matmulk0.begin(), matmulk0.end(), std::back_inserter(op_wrappers));
  // MatMulK1
  auto& kv_slice_k = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 1, 256});
  auto& matmulk1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 4, 1});
  auto matmulk1 = BuildMatmulOp(tensor_pool, {mul_output, kv_slice_k},
                                {matmulk1_output}, false, true);
  std::move(matmulk1.begin(), matmulk1.end(), std::back_inserter(op_wrappers));
  // Concat
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(matmulk0_output, {1, 1, 4, 1281});
  auto concat = BuildConcatenationOp(
      tensor_pool, {matmulk0_output, matmulk1_output}, {concat_output}, 3);
  std::move(concat.begin(), concat.end(), std::back_inserter(op_wrappers));
  // Reshape0
  auto& reshape0_output =
      tensor_pool.CloneNativeTensorFrom(concat_output, {1, 4, 1, 1281});
  auto reshape0 =
      BuildReshapeOp(tensor_pool, {concat_output}, {reshape0_output});
  std::move(reshape0.begin(), reshape0.end(), std::back_inserter(op_wrappers));
  // Add
  auto& add0_output = tensor_pool.CloneNativeTensorFrom(reshape0_output);
  auto& mask = tensor_pool.CloneNativeTensorFrom(reshape0_output);
  auto add0 = BuildElementwiseAddOp(tensor_pool, {reshape0_output, mask},
                                    {add0_output});
  std::move(add0.begin(), add0.end(), std::back_inserter(op_wrappers));
  // Reshape1
  auto& reshape1_output =
      tensor_pool.CloneNativeTensorFrom(add0_output, {1, 1, 4, 1281});
  auto reshape1 = BuildReshapeOp(tensor_pool, {add0_output}, {reshape1_output});
  std::move(reshape1.begin(), reshape1.end(), std::back_inserter(op_wrappers));
  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(reshape1_output);
  auto softmax =
      BuildSoftmaxOp(tensor_pool, {reshape1_output}, {softmax_output}, 1.0f);
  std::move(softmax.begin(), softmax.end(), std::back_inserter(op_wrappers));
  // Slice0
  const std::array<int32_t, 4> slice0_begin_data{0, 0, 0, 0};
  auto& slice0_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {slice0_begin_data.size()},
      slice0_begin_data.size() * sizeof(slice0_begin_data[0]),
      slice0_begin_data.data());
  const std::array<int32_t, 4> slice0_size_data{1, 1, 4, 1280};
  auto& slice0_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {slice0_size_data.size()},
      slice0_size_data.size() * sizeof(slice0_size_data[0]),
      slice0_size_data.data());
  auto& slice0_output =
      tensor_pool.CloneNativeTensorFrom(reshape1_output, {1, 1, 4, 1280});
  auto slice0 =
      BuildSliceOp(tensor_pool, {softmax_output, slice0_begin, slice0_size},
                   {slice0_output});
  std::move(slice0.begin(), slice0.end(), std::back_inserter(op_wrappers));
  // Slice1
  const std::array<int32_t, 4> slice1_begin_data{0, 0, 0, 1280};
  auto& slice1_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {slice1_begin_data.size()},
      slice1_begin_data.size() * sizeof(slice1_begin_data[0]),
      slice1_begin_data.data());
  ;
  const std::array<int32_t, 4> slice1_size_data{1, 1, 4, 1};
  auto& slice1_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {slice1_size_data.size()},
      slice1_size_data.size() * sizeof(slice1_size_data[0]),
      slice1_size_data.data());
  auto& slice1_output =
      tensor_pool.CloneNativeTensorFrom(reshape1_output, {1, 1, 4, 256});
  auto slice1 =
      BuildSliceOp(tensor_pool, {softmax_output, slice1_begin, slice1_size},
                   {slice1_output});
  std::move(slice1.begin(), slice1.end(), std::back_inserter(op_wrappers));
  // MatMulV0
  auto& kv_cache_v = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 256, 1280});
  auto& matmulv0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 4, 256});
  auto matmulv0 = BuildMatmulOp(tensor_pool, {slice0_output, kv_cache_v},
                                {matmulv0_output}, false, true);
  std::move(matmulv0.begin(), matmulv0.end(), std::back_inserter(op_wrappers));
  // MatMulV1
  auto& kv_slice_v = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 256, 1});
  auto& matmulv1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 4, 256});
  auto matmulv1 = BuildMatmulOp(tensor_pool, {slice1_output, kv_slice_v},
                                {matmulv1_output}, false, true);
  std::move(matmulv1.begin(), matmulv1.end(), std::back_inserter(op_wrappers));
  // Add1
  auto& add1_output = tensor_pool.CloneNativeTensorFrom(matmulv0_output);
  auto add1 = BuildElementwiseAddOp(
      tensor_pool, {matmulv0_output, matmulv1_output}, {add1_output});
  std::move(add1.begin(), add1.end(), std::back_inserter(op_wrappers));
  // Reshape2
  auto& reshape2_output =
      tensor_pool.CloneNativeTensorFrom(add1_output, {1, 1, 1024});
  auto reshape2 = BuildReshapeOp(tensor_pool, {add1_output}, {reshape2_output});
  std::move(reshape2.begin(), reshape2.end(), std::back_inserter(op_wrappers));

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
              QnnOpCode::kElementWiseMultiply);
    ASSERT_EQ(op_wrappers[2 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[3 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[4 + sha_size * i].GetOpCode(), QnnOpCode::kConcat);
    ASSERT_EQ(op_wrappers[5 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseAdd);
    ASSERT_EQ(op_wrappers[6 + sha_size * i].GetOpCode(), QnnOpCode::kSoftmax);
    ASSERT_EQ(op_wrappers[7 + sha_size * i].GetOpCode(),
              QnnOpCode::kStridedSlice);
    ASSERT_EQ(op_wrappers[8 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[9 + sha_size * i].GetOpCode(),
              QnnOpCode::kStridedSlice);
    ASSERT_EQ(op_wrappers[10 + sha_size * i].GetOpCode(), QnnOpCode::kMatMul);
    ASSERT_EQ(op_wrappers[11 + sha_size * i].GetOpCode(),
              QnnOpCode::kElementWiseAdd);
  }
  ASSERT_EQ(op_wrappers[45].GetOpCode(), QnnOpCode::kConcat);
  ASSERT_EQ(op_wrappers[46].GetOpCode(), QnnOpCode::kReshape);
}

}  // namespace
}  // namespace qnn
