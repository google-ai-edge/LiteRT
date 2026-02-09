// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/transformation/graph_to_graph.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {
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
  //                     Unpack        In1     In2
  //                        | \\        |       |
  //                        |  \\    Unpack   Unpack
  //        QIn             |   ..     /\\      |\\
  //         |             Mul        /  \\     | \\
  //       Unpack         /   \      /    ..    |  ..
  //       // \          /     \    Add0 -------┘
  //      //   \        /       \   /
  //     ..     └-----Matmul0   Matmul1
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
  //                    /       \     In3
  //        VIn       Slice0  Slice1   |
  //         |          |       |    Unpack
  //       Unpack       |       |     /\\
  //       //|          |       |    /  \\
  //      // └------Matmul2  Matmul3     ..
  //     ..              \     /
  //                       Add2  ..
  //                        |   //
  //                        |  //
  //                      Concat
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
  auto add0 =
      BuildElementwiseAddOp(tensor_pool, {input1, input2}, {add0_output});
  std::move(add0.begin(), add0.end(), std::back_inserter(op_wrappers));

  // Dummy Reshape Op to make sure the Add Op above is in the correct position.
  auto& dummy_reshape_output = tensor_pool.CloneNativeTensorFrom(add0_output);
  auto dummy_reshape =
      BuildReshapeOp(tensor_pool, {add0_output}, {dummy_reshape_output});
  std::move(dummy_reshape.begin(), dummy_reshape.end(),
            std::back_inserter(op_wrappers));

  // Mul
  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 14, 128, 64});
  constexpr std::array<int16_t, 1> kMulVal = {32767};
  auto& mul_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {kMulVal.size()},
      kMulVal.size() * sizeof(kMulVal[0]), kMulVal.data());
  auto& mul_output =
      tensor_pool.CloneNativeTensorFrom(input0, {1, 14, 128, 64});
  auto mul =
      BuildElementwiseMulOp(tensor_pool, {input0, mul_const}, {mul_output});
  std::move(mul.begin(), mul.end(), std::back_inserter(op_wrappers));

  // Reshape0
  auto& reshape0_output =
      tensor_pool.CloneNativeTensorFrom(mul_output, {1, 2, 896, 64});
  auto reshape0 = BuildReshapeOp(tensor_pool, {mul_output}, {reshape0_output});
  std::move(reshape0.begin(), reshape0.end(), std::back_inserter(op_wrappers));

  // MatMul0
  auto& q_in = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 2, 1280, 64});
  auto& matmul0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 2, 896, 1280});
  auto matmul0 = BuildMatmulOp(tensor_pool, {reshape0_output, q_in},
                               {matmul0_output}, false, true);
  std::move(matmul0.begin(), matmul0.end(), std::back_inserter(op_wrappers));

  // MatMul1
  auto& matmul1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 2, 896, 128});
  auto matmul1 = BuildMatmulOp(tensor_pool, {reshape0_output, add0_output},
                               {matmul1_output}, false, true);
  std::move(matmul1.begin(), matmul1.end(), std::back_inserter(op_wrappers));

  // Concat
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(matmul0_output, {1, 2, 896, 1408});
  auto concat = BuildConcatenationOp(
      tensor_pool, {matmul0_output, matmul1_output}, {concat_output}, 3);
  std::move(concat.begin(), concat.end(), std::back_inserter(op_wrappers));

  // Reshape1
  auto& reshape1_output =
      tensor_pool.CloneNativeTensorFrom(concat_output, {2, 7, 128, 1408});
  auto reshape1 =
      BuildReshapeOp(tensor_pool, {concat_output}, {reshape1_output});
  std::move(reshape1.begin(), reshape1.end(), std::back_inserter(op_wrappers));

  // Add1
  auto& mask = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 1, 128, 1408});
  auto& add1_output = tensor_pool.CloneNativeTensorFrom(reshape1_output);
  auto add1 = BuildElementwiseAddOp(tensor_pool, {reshape1_output, mask},
                                    {add1_output});
  std::move(add1.begin(), add1.end(), std::back_inserter(op_wrappers));

  // Reshape2
  auto& reshape2_output =
      tensor_pool.CloneNativeTensorFrom(add1_output, {1, 2, 896, 1408});
  auto reshape2 = BuildReshapeOp(tensor_pool, {add1_output}, {reshape2_output});
  std::move(reshape2.begin(), reshape2.end(), std::back_inserter(op_wrappers));

  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(reshape2_output);
  auto softmax =
      BuildSoftmaxOp(tensor_pool, {reshape2_output}, {softmax_output}, 1.0f);
  std::move(softmax.begin(), softmax.end(), std::back_inserter(op_wrappers));

  // Slice0
  constexpr std::array<int32_t, 4> kSlice0BeginData{0, 0, 0, 0};
  auto& slice0_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kSlice0BeginData.size()},
      kSlice0BeginData.size() * sizeof(kSlice0BeginData[0]),
      kSlice0BeginData.data());
  constexpr std::array<int32_t, 4> kSlice0SizeData{1, 2, 896, 1280};
  auto& slice0_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kSlice0SizeData.size()},
      kSlice0SizeData.size() * sizeof(kSlice0SizeData[0]),
      kSlice0SizeData.data());
  auto& slice0_output =
      tensor_pool.CloneNativeTensorFrom(softmax_output, {1, 2, 896, 1280});
  auto slice0 =
      BuildSliceOp(tensor_pool, {softmax_output, slice0_begin, slice0_size},
                   {slice0_output});
  std::move(slice0.begin(), slice0.end(), std::back_inserter(op_wrappers));

  // Slice1
  constexpr std::array<int32_t, 4> kSlice1BeginData{0, 0, 0, 1280};
  auto& slice1_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kSlice1BeginData.size()},
      kSlice1BeginData.size() * sizeof(kSlice1BeginData[0]),
      kSlice1BeginData.data());
  constexpr std::array<int32_t, 4> kSlice1SizeData{1, 2, 896, 128};
  auto& slice1_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kSlice1SizeData.size()},
      kSlice1SizeData.size() * sizeof(kSlice1SizeData[0]),
      kSlice1SizeData.data());
  auto& slice1_output =
      tensor_pool.CloneNativeTensorFrom(softmax_output, {1, 2, 896, 128});
  auto slice1 =
      BuildSliceOp(tensor_pool, {softmax_output, slice1_begin, slice1_size},
                   {slice1_output});
  std::move(slice1.begin(), slice1.end(), std::back_inserter(op_wrappers));

  // MatMul2
  auto& v_in = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 2, 64, 1280});
  auto& matmul2_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 2, 896, 64});
  auto matmul2 = BuildMatmulOp(tensor_pool, {slice0_output, v_in},
                               {matmul2_output}, false, true);
  std::move(matmul2.begin(), matmul2.end(), std::back_inserter(op_wrappers));

  // MatMul3
  auto& matmul3_input_2 =
      tensor_pool.CloneNativeTensorFrom(mul_output, {1, 2, 64, 128});
  auto& matmul3_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 2, 896, 64});
  auto matmul3 = BuildMatmulOp(tensor_pool, {slice1_output, matmul3_input_2},
                               {matmul3_output}, false, true);
  std::move(matmul3.begin(), matmul3.end(), std::back_inserter(op_wrappers));

  // Add2
  auto& add2_output = tensor_pool.CloneNativeTensorFrom(matmul3_output);
  auto add2 = BuildElementwiseAddOp(
      tensor_pool, {matmul2_output, matmul3_output}, {add2_output});
  std::move(add2.begin(), add2.end(), std::back_inserter(op_wrappers));

  // Reshape3
  auto& reshape3_output =
      tensor_pool.CloneNativeTensorFrom(add2_output, {1, 14, 128, 64});
  auto reshape3 = BuildReshapeOp(tensor_pool, {add2_output}, {reshape3_output});
  std::move(reshape3.begin(), reshape3.end(), std::back_inserter(op_wrappers));

  // Transpose1
  constexpr std::array<int32_t, 4> kTranspose1Val = {0, 2, 1, 3};
  auto& transpose1_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, quant_param, {kTranspose1Val.size()},
      kTranspose1Val.size() * sizeof(kTranspose1Val[0]), kTranspose1Val.data());
  auto& transpose1_output =
      tensor_pool.CloneNativeTensorFrom(reshape3_output, {1, 128, 14, 64});
  auto transpose1 = BuildTransposeOp(
      tensor_pool, {reshape3_output, transpose1_perm}, {transpose1_output});
  std::move(transpose1.begin(), transpose1.end(),
            std::back_inserter(op_wrappers));

  // Reshape4
  auto& reshape4_output =
      tensor_pool.CloneNativeTensorFrom(transpose1_output, {1, 128, 896});
  auto reshape4 =
      BuildReshapeOp(tensor_pool, {transpose1_output}, {reshape4_output});
  std::move(reshape4.begin(), reshape4.end(), std::back_inserter(op_wrappers));

  ASSERT_EQ(op_wrappers.size(), 19);

  const ::qnn::G2GConfig g2g_option = ::qnn::G2GConfig::kMHAOptPrefill;
  GraphToGraphTransform(g2g_option, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });
  // Check total size after G2G
  ASSERT_EQ(op_wrappers.size(), 191);

  // Check OpCode after G2G
  constexpr size_t kNumUnpack = 6;
  constexpr size_t kNumHead = 14;
  constexpr size_t kShaSize = 13;

  ASSERT_TRUE(op_wrappers[0].IsOpCode(QnnOpCode::kElementWiseBinary));
  ASSERT_TRUE(IsElementWiseAdd(op_wrappers[0]));
  ASSERT_TRUE(op_wrappers[1].IsOpCode(QnnOpCode::kReshape));

  for (size_t i = 0; i < kNumUnpack; ++i) {
    ASSERT_TRUE(op_wrappers[2 + i].IsOpCode(QnnOpCode::kUnPack));
  }

  for (size_t i = 0; i < kNumHead; ++i) {
    ASSERT_TRUE(
        op_wrappers[8 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseMultiply(op_wrappers[8 + kShaSize * i]));
    ASSERT_TRUE(op_wrappers[9 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(
        op_wrappers[10 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[10 + kShaSize * i]));
    ASSERT_TRUE(op_wrappers[11 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(op_wrappers[12 + kShaSize * i].IsOpCode(QnnOpCode::kConcat));
    ASSERT_TRUE(
        op_wrappers[13 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[13 + kShaSize * i]));
    ASSERT_TRUE(op_wrappers[14 + kShaSize * i].IsOpCode(QnnOpCode::kReshape));
    ASSERT_TRUE(op_wrappers[15 + kShaSize * i].IsOpCode(QnnOpCode::kSoftmax));
    ASSERT_TRUE(
        op_wrappers[16 + kShaSize * i].IsOpCode(QnnOpCode::kStridedSlice));
    ASSERT_TRUE(
        op_wrappers[17 + kShaSize * i].IsOpCode(QnnOpCode::kStridedSlice));
    ASSERT_TRUE(op_wrappers[18 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(op_wrappers[19 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(
        op_wrappers[20 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[20 + kShaSize * i]));
  }
  ASSERT_TRUE(op_wrappers[op_wrappers.size() - 1].IsOpCode(QnnOpCode::kConcat));
}

TEST(MHASHATest, FastVlmKvSwapped) {
  // G2G Test case: MHA -> SHA
  // ------------------- Before ---------------------
  //                       In0
  //                        |
  //                       Mul     In1   In2
  //                        |       \    /
  //                     Reshape0    Add0
  //                      /   \        |
  //             QIn     /     \   Transpose0
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
  //                  Slice0  Slice1
  //                    |       |
  //              VIn   |       |    In3
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
  //                     Unpack        In1     In2
  //                        | \\        |       |
  //                        |  \\    Unpack   Unpack
  //        QIn             |   ..     /\\      |\\
  //         |             Mul        /  \\     | \\
  //       Unpack         /   \      /    ..    |  ..
  //       // \          /     \    Add0 -------┘
  //      //   \        /       \   /
  //     ..     └-----Matmul0   Matmul1
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
  //                    /       \     In3
  //        VIn       Slice0  Slice1   |
  //         |          |       |    Unpack
  //       Unpack       |       |     /\\
  //       //|          |       |    /  \\
  //      // └------Matmul2  Matmul3     ..
  //     ..              \     /
  //                       Add2  ..
  //                        |   //
  //                        |  //
  //                      Concat
  //                        |
  //                       Out
  TensorPool tensor_pool;
  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);
  std::vector<OpWrapper> op_wrappers;

  auto& input1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 8, 128, 128});
  auto& input2 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 8, 128, 128});
  // Add0
  auto& add0_output =
      tensor_pool.CloneNativeTensorFrom(input1, {1, 8, 128, 128});
  auto add0 =
      BuildElementwiseAddOp(tensor_pool, {input1, input2}, {add0_output});
  std::move(add0.begin(), add0.end(), std::back_inserter(op_wrappers));

  // Transpose0
  constexpr std::array<int32_t, 4> kTranspose0Val = {0, 1, 3, 2};
  auto& transpose0_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kTranspose0Val.size()},
      kTranspose0Val.size() * sizeof(kTranspose0Val[0]), kTranspose0Val.data());
  auto& transpose0_output =
      tensor_pool.CloneNativeTensorFrom(add0_output, {1, 8, 128, 128});
  auto transpose0 = BuildTransposeOp(
      tensor_pool, {add0_output, transpose0_perm}, {transpose0_output});
  std::move(transpose0.begin(), transpose0.end(),
            std::back_inserter(op_wrappers));

  // Mul
  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 16, 128, 128});
  constexpr std::array<int16_t, 1> kMulVal = {32767};
  auto& mul_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {kMulVal.size()},
      kMulVal.size() * sizeof(kMulVal[0]), kMulVal.data());
  auto& mul_output =
      tensor_pool.CloneNativeTensorFrom(input0, {1, 16, 128, 128});
  auto mul =
      BuildElementwiseMulOp(tensor_pool, {input0, mul_const}, {mul_output});
  std::move(mul.begin(), mul.end(), std::back_inserter(op_wrappers));

  // Reshape0
  auto& reshape0_output =
      tensor_pool.CloneNativeTensorFrom(mul_output, {1, 8, 256, 128});
  auto reshape0 = BuildReshapeOp(tensor_pool, {mul_output}, {reshape0_output});
  std::move(reshape0.begin(), reshape0.end(), std::back_inserter(op_wrappers));

  // MatMul0
  auto& q_in = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 8, 128, 2048});
  auto& matmul0_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 8, 256, 2048});
  auto matmul0 = BuildMatmulOp(tensor_pool, {reshape0_output, q_in},
                               {matmul0_output}, false, false);
  std::move(matmul0.begin(), matmul0.end(), std::back_inserter(op_wrappers));

  // MatMul1
  auto& matmul1_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 8, 256, 128});
  auto matmul1 =
      BuildMatmulOp(tensor_pool, {reshape0_output, transpose0_output},
                    {matmul1_output}, false, false);
  std::move(matmul1.begin(), matmul1.end(), std::back_inserter(op_wrappers));

  // Concat
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(matmul0_output, {1, 8, 256, 2176});
  auto concat = BuildConcatenationOp(
      tensor_pool, {matmul0_output, matmul1_output}, {concat_output}, 3);
  std::move(concat.begin(), concat.end(), std::back_inserter(op_wrappers));

  // Reshape1
  auto& reshape1_output =
      tensor_pool.CloneNativeTensorFrom(concat_output, {8, 2, 128, 2176});
  auto reshape1 =
      BuildReshapeOp(tensor_pool, {concat_output}, {reshape1_output});
  std::move(reshape1.begin(), reshape1.end(), std::back_inserter(op_wrappers));

  // Add1
  auto& mask = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 1, 128, 2176});
  auto& add1_output = tensor_pool.CloneNativeTensorFrom(reshape1_output);
  auto add1 = BuildElementwiseAddOp(tensor_pool, {reshape1_output, mask},
                                    {add1_output});
  std::move(add1.begin(), add1.end(), std::back_inserter(op_wrappers));

  // Reshape2
  auto& reshape2_output =
      tensor_pool.CloneNativeTensorFrom(add1_output, {1, 8, 256, 2176});
  auto reshape2 = BuildReshapeOp(tensor_pool, {add1_output}, {reshape2_output});
  std::move(reshape2.begin(), reshape2.end(), std::back_inserter(op_wrappers));

  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(reshape2_output);
  auto softmax =
      BuildSoftmaxOp(tensor_pool, {reshape2_output}, {softmax_output}, 1.0f);
  std::move(softmax.begin(), softmax.end(), std::back_inserter(op_wrappers));

  // Slice0
  constexpr std::array<int32_t, 4> kSlice0BeginData{0, 0, 0, 0};
  auto& slice0_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kSlice0BeginData.size()},
      kSlice0BeginData.size() * sizeof(kSlice0BeginData[0]),
      kSlice0BeginData.data());
  constexpr std::array<int32_t, 4> kSlice0SizeData{1, 8, 256, 2048};
  auto& slice0_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kSlice0SizeData.size()},
      kSlice0SizeData.size() * sizeof(kSlice0SizeData[0]),
      kSlice0SizeData.data());
  auto& slice0_output =
      tensor_pool.CloneNativeTensorFrom(softmax_output, {1, 8, 256, 2048});
  auto slice0 =
      BuildSliceOp(tensor_pool, {softmax_output, slice0_begin, slice0_size},
                   {slice0_output});
  std::move(slice0.begin(), slice0.end(), std::back_inserter(op_wrappers));

  // Slice1
  constexpr std::array<int32_t, 4> kSlice1BeginData{0, 0, 0, 2048};
  auto& slice1_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kSlice1BeginData.size()},
      kSlice1BeginData.size() * sizeof(kSlice1BeginData[0]),
      kSlice1BeginData.data());
  constexpr std::array<int32_t, 4> kSlice1SizeData{1, 8, 256, 128};
  auto& slice1_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kSlice1SizeData.size()},
      kSlice1SizeData.size() * sizeof(kSlice1SizeData[0]),
      kSlice1SizeData.data());
  auto& slice1_output =
      tensor_pool.CloneNativeTensorFrom(softmax_output, {1, 8, 256, 128});
  auto slice1 =
      BuildSliceOp(tensor_pool, {softmax_output, slice1_begin, slice1_size},
                   {slice1_output});
  std::move(slice1.begin(), slice1.end(), std::back_inserter(op_wrappers));

  // MatMul2
  auto& v_in = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 8, 2048, 128});
  auto& matmul2_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 8, 256, 128});
  auto matmul2 = BuildMatmulOp(tensor_pool, {slice0_output, v_in},
                               {matmul2_output}, false, false);
  std::move(matmul2.begin(), matmul2.end(), std::back_inserter(op_wrappers));

  // MatMul3
  auto& matmul3_input_2 =
      tensor_pool.CloneNativeTensorFrom(mul_output, {1, 8, 128, 128});
  auto& matmul3_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 8, 256, 128});
  auto matmul3 = BuildMatmulOp(tensor_pool, {slice1_output, matmul3_input_2},
                               {matmul3_output}, false, false);
  std::move(matmul3.begin(), matmul3.end(), std::back_inserter(op_wrappers));

  // Add2
  auto& add2_output = tensor_pool.CloneNativeTensorFrom(matmul3_output);
  auto add2 = BuildElementwiseAddOp(
      tensor_pool, {matmul2_output, matmul3_output}, {add2_output});
  std::move(add2.begin(), add2.end(), std::back_inserter(op_wrappers));

  // Reshape3
  auto& reshape3_output =
      tensor_pool.CloneNativeTensorFrom(add2_output, {1, 16, 128, 128});
  auto reshape3 = BuildReshapeOp(tensor_pool, {add2_output}, {reshape3_output});
  std::move(reshape3.begin(), reshape3.end(), std::back_inserter(op_wrappers));

  // Transpose1
  constexpr std::array<int32_t, 4> kTranspose1Val = {0, 2, 1, 3};
  auto& transpose1_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, quant_param, {kTranspose1Val.size()},
      kTranspose1Val.size() * sizeof(kTranspose1Val[0]), kTranspose1Val.data());
  auto& transpose1_output =
      tensor_pool.CloneNativeTensorFrom(reshape3_output, {1, 128, 16, 128});
  auto transpose1 = BuildTransposeOp(
      tensor_pool, {reshape3_output, transpose1_perm}, {transpose1_output});
  std::move(transpose1.begin(), transpose1.end(),
            std::back_inserter(op_wrappers));

  // Reshape4
  auto& reshape4_output =
      tensor_pool.CloneNativeTensorFrom(transpose1_output, {1, 128, 2048});
  auto reshape4 =
      BuildReshapeOp(tensor_pool, {transpose1_output}, {reshape4_output});
  std::move(reshape4.begin(), reshape4.end(), std::back_inserter(op_wrappers));

  ASSERT_EQ(op_wrappers.size(), 19);

  GraphToGraphTransform(::qnn::G2GConfig::kMHAOptPrefill, op_wrappers, tensor_pool,
                        [](OpWrapper& op) { return true; });
  // Check total size after G2G
  ASSERT_EQ(op_wrappers.size(), 216);
  // Check OpCode after G2G
  constexpr size_t kNumUnpack = 6;
  constexpr size_t kNumHead = 16;
  constexpr size_t kShaSize = 13;

  ASSERT_TRUE(op_wrappers[0].IsOpCode(QnnOpCode::kElementWiseBinary));
  ASSERT_TRUE(IsElementWiseAdd(op_wrappers[0]));

  for (size_t i = 0; i < kNumUnpack; ++i) {
    ASSERT_TRUE(op_wrappers[1 + i].IsOpCode(QnnOpCode::kUnPack));
  }

  for (size_t i = 0; i < kNumHead; ++i) {
    ASSERT_TRUE(
        op_wrappers[7 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseMultiply(op_wrappers[7 + kShaSize * i]));
    ASSERT_TRUE(op_wrappers[8 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(
        op_wrappers[9 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[9 + kShaSize * i]));
    ASSERT_TRUE(op_wrappers[10 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(op_wrappers[11 + kShaSize * i].IsOpCode(QnnOpCode::kConcat));
    ASSERT_TRUE(
        op_wrappers[12 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[12 + kShaSize * i]));
    ASSERT_TRUE(op_wrappers[13 + kShaSize * i].IsOpCode(QnnOpCode::kReshape));
    ASSERT_TRUE(op_wrappers[14 + kShaSize * i].IsOpCode(QnnOpCode::kSoftmax));
    ASSERT_TRUE(
        op_wrappers[15 + kShaSize * i].IsOpCode(QnnOpCode::kStridedSlice));
    ASSERT_TRUE(
        op_wrappers[16 + kShaSize * i].IsOpCode(QnnOpCode::kStridedSlice));
    ASSERT_TRUE(op_wrappers[17 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(op_wrappers[18 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(
        op_wrappers[19 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[19 + kShaSize * i]));
  }
  ASSERT_TRUE(op_wrappers[op_wrappers.size() - 1].IsOpCode(QnnOpCode::kConcat));
}

}  // namespace
}  // namespace qnn
