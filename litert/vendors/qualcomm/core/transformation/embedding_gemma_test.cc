// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <variant>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/transformation/graph_to_graph.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {

TEST(MHASHATest, EmbeddingGemma) {
  // G2G Test case: MHA -> SHA
  //
  // ---------------- Before ---------------------
  //                   QIn
  //                    |
  //                   Mul
  //                    |
  //                Transpose0
  //                    |
  //                 Reshape0
  //                    |
  //                    |  KIn
  //                    | /
  //                  MatMul0
  //                    |
  //               mask |
  //                  \ |
  //                   Add0
  //                    |
  //                 Softmax
  //                    |
  //                    |  VIn
  //                    | /
  //                  MatMul1
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
  //                  Split
  //                    | \\\
  //                    |  \\\
  //                    |   ...
  //                   Mul
  //                    |
  //                    | KIn            Mask
  //                    | /               |
  //                  MatMul            Split
  //                    |                 |
  //                    |  _______________|
  //                    | /
  //                   Add
  //                    |
  //                 Softmax
  //                    |
  //                    | VIn
  //                    | /
  //                  MatMul
  //                    |    ...
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
                                                quant_param, {1, 1024, 3, 256});
  std::array<int16_t, 1> mul_val = {32767};
  auto& mul_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {mul_val.size()},
      mul_val.size() * sizeof(mul_val[0]), mul_val.data());
  auto& mul_output =
      tensor_pool.CloneNativeTensorFrom(input0, {1, 1024, 3, 256});
  auto mul =
      BuildElementwiseMulOp(tensor_pool, {input0, mul_const}, {mul_output});
  std::move(mul.begin(), mul.end(), std::back_inserter(op_wrappers));
  // Transpose0
  std::array<int32_t, 4> transpose_val = {0, 2, 1, 3};
  auto& transpose_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, quant_param, {transpose_val.size()},
      transpose_val.size() * sizeof(transpose_val[0]), transpose_val.data());
  auto& transpose0_output =
      tensor_pool.CloneNativeTensorFrom(mul_output, {1, 3, 1024, 256});
  auto transpose0 = BuildTransposeOp(tensor_pool, {mul_output, transpose_perm},
                                     {transpose0_output});
  std::move(transpose0.begin(), transpose0.end(),
            std::back_inserter(op_wrappers));

  // Reshape0
  auto& reshape0_output =
      tensor_pool.CloneNativeTensorFrom(transpose0_output, {1, 1, 3072, 256});
  auto reshape0 =
      BuildReshapeOp(tensor_pool, {transpose0_output}, {reshape0_output});
  std::move(reshape0.begin(), reshape0.end(), std::back_inserter(op_wrappers));

  // MatMul0
  auto& k_in = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 1, 1024, 256});
  auto& matmul0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 3072, 1024});
  auto matmul0 = BuildMatmulOp(tensor_pool, {reshape0_output, k_in},
                               {matmul0_output}, false, true);
  std::move(matmul0.begin(), matmul0.end(), std::back_inserter(op_wrappers));

  // Add
  auto& add0_output = tensor_pool.CloneNativeTensorFrom(matmul0_output);
  auto& mask = tensor_pool.CloneNativeTensorFrom(matmul0_output);
  auto add0 =
      BuildElementwiseAddOp(tensor_pool, {matmul0_output, mask}, {add0_output});
  std::move(add0.begin(), add0.end(), std::back_inserter(op_wrappers));
  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(matmul0_output);
  auto softmax =
      BuildSoftmaxOp(tensor_pool, {add0_output}, {softmax_output}, 1.0f);
  std::move(softmax.begin(), softmax.end(), std::back_inserter(op_wrappers));

  // MatMul1
  auto& v_in = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 1, 1024, 256});
  auto& matmul1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 3072, 256});
  auto matmul1 = BuildMatmulOp(tensor_pool, {softmax_output, v_in},
                               {matmul1_output}, false, true);
  std::move(matmul1.begin(), matmul1.end(), std::back_inserter(op_wrappers));

  // Reshape1
  auto& reshape1_output =
      tensor_pool.CloneNativeTensorFrom(matmul1_output, {1, 3, 1024, 256});
  auto reshape1 =
      BuildReshapeOp(tensor_pool, {matmul1_output}, {reshape1_output});
  std::move(reshape1.begin(), reshape1.end(), std::back_inserter(op_wrappers));
  // Transpose1
  auto& transpose1_output =
      tensor_pool.CloneNativeTensorFrom(reshape1_output, {1, 1024, 3, 256});
  auto transpose1 = BuildTransposeOp(
      tensor_pool, {reshape1_output, transpose_perm}, {transpose1_output});
  std::move(transpose1.begin(), transpose1.end(),
            std::back_inserter(op_wrappers));

  // Reshape2
  auto& reshape2_output =
      tensor_pool.CloneNativeTensorFrom(transpose1_output, {1, 128, 1024});
  auto reshape2 =
      BuildReshapeOp(tensor_pool, {transpose1_output}, {reshape2_output});
  std::move(reshape2.begin(), reshape2.end(), std::back_inserter(op_wrappers));
  ASSERT_EQ(op_wrappers.size(), 10);

  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMatMulConvert;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });
  ASSERT_EQ(op_wrappers.size(), 20);

  ASSERT_TRUE(op_wrappers[0].IsOpCode(QnnOpCode::kTranspose));
  ASSERT_TRUE(op_wrappers[1].IsOpCode(QnnOpCode::kSplit));
  ASSERT_TRUE(op_wrappers[2].IsOpCode(QnnOpCode::kSplit));

  const size_t sha_size = 5;
  const size_t num_head = 3;
  for (size_t i = 0; i < num_head; ++i) {
    ASSERT_TRUE(op_wrappers[3 + sha_size * i].IsOpCode(
        QnnOpCode::kElementWiseMultiply));
    ASSERT_TRUE(op_wrappers[4 + sha_size * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(
        op_wrappers[5 + sha_size * i].IsOpCode(QnnOpCode::kElementWiseAdd));
    ASSERT_TRUE(op_wrappers[6 + sha_size * i].IsOpCode(QnnOpCode::kSoftmax));
    ASSERT_TRUE(op_wrappers[7 + sha_size * i].IsOpCode(QnnOpCode::kMatMul));
  }
  ASSERT_TRUE(op_wrappers[18].IsOpCode(QnnOpCode::kConcat));
  ASSERT_TRUE(op_wrappers[19].IsOpCode(QnnOpCode::kReshape));
}

}  // namespace
}  // namespace qnn
