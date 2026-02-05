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
TEST(MHASHATest, FastVlmKVSwapped) {
  // G2G Test case: MHA -> SHA
  // ------------------- Before ---------------------
  //                     QScaleIn
  //                        |
  //                    QScaleMul
  //                        |
  //                   QScaleReshape
  //                      /   \    
  //           KCache    /     \     KSlice
  //               \    /       \     /
  //           QKCacheMatmul  QKSliceMatmul
  //                    \       /
  //                     QKConcat
  //                        |
  //                  PreMaskReshape
  //                        |
  //                   Mask |
  //                      \ |
  //                     MaskAdd
  //                        |
  //                  PostMaskReshape
  //                        |
  //                     Softmax
  //                    /       \
  //           QKVCacheSlice  QKVSliceSlice
  //                  |         |
  //        VCache    |         |   VSlice
  //            \     |         |    /
  //         QKVCacheMatmul  QKVSliceMatmul
  //                     \     /
  //                      QKVAdd
  //                        |
  //                   QKVReshape
  //                        |
  //                   QKVTranspose
  //                        |
  //                   OProjReshape
  //                        |
  //                       Out
  //
  // -------------------- After ---------------------
  //                     QScaleIn
  //                        |
  //                     Unpack
  //                        | \\        
  //                        |  \\   KSlice
  //       KCache           |   ..    |
  //         |          QScaleMul   Unpack
  //       Unpack         /   \       / \\
  //       // \          /     \     /   \\
  //      //   \        /       \   /     ..
  //     ..   QKCacheMatmul   QKSliceMatmul
  //                    \       /
  //                     QKConcat
  //                        |
  //                   Mask |
  //                      \ |
  //                     MaskAdd
  //                        |
  //                  PostMaskReshape
  //                        |
  //                     Softmax
  //                    /           \         VSlice
  //       VCache QKVCacheSlice QKVCacheSlice   |
  //         |        |              |        Unpack
  //       Unpack     |              |         /\\
  //       //|        |              |        /  \\
  //      // └----QKVCacheMatmul  QKVCacheMatmul  ..
  //     ..              \         /
  //                        QKVAdd    ..
  //                          |      //
  //                          |     //
  //                        Concat
  //                          |

  TensorPool tensor_pool;
  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);
  std::vector<OpWrapper> op_wrappers;

  // QScaleMul
  auto& q_scale_in = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 16, 128, 128});
  constexpr std::array<int16_t, 1> kMulVal = {32767};
  auto& q_scale_mul_const = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {kMulVal.size()},
      kMulVal.size() * sizeof(kMulVal[0]), kMulVal.data());
  auto& q_scale_mul_output =
      tensor_pool.CloneNativeTensorFrom(q_scale_in, {1, 16, 128, 128});
  auto q_scale_mul = BuildElementwiseMulOp(
      tensor_pool, {q_scale_in, q_scale_mul_const}, {q_scale_mul_output});
  std::move(q_scale_mul.begin(), q_scale_mul.end(),
            std::back_inserter(op_wrappers));

  // QScaleReshape
  auto& q_scale_reshape_output =
      tensor_pool.CloneNativeTensorFrom(q_scale_mul_output, {1, 8, 256, 128});
  auto q_scale_reshape = BuildReshapeOp(tensor_pool, {q_scale_mul_output},
                                        {q_scale_reshape_output});
  std::move(q_scale_reshape.begin(), q_scale_reshape.end(),
            std::back_inserter(op_wrappers));

  // QKCacheMatmul
  auto& k_cache = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 8, 128, 2048});
  auto& q_kcache_matmul_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 8, 256, 2048});
  auto q_kcache_matmul =
      BuildMatmulOp(tensor_pool, {q_scale_reshape_output, k_cache},
                    {q_kcache_matmul_output}, false, false);
  std::move(q_kcache_matmul.begin(), q_kcache_matmul.end(),
            std::back_inserter(op_wrappers));

  // QKSliceMatmul
  auto& k_slice = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                 quant_param, {1, 8, 128, 128});
  auto& q_kslice_matmul_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 8, 256, 128});
  auto q_kslice_matmul =
      BuildMatmulOp(tensor_pool, {q_scale_reshape_output, k_slice},
                    {q_kslice_matmul_output}, false, false);
  std::move(q_kslice_matmul.begin(), q_kslice_matmul.end(),
            std::back_inserter(op_wrappers));

  // QKConcat
  auto& qk_concat_output = tensor_pool.CloneNativeTensorFrom(
      q_kcache_matmul_output, {1, 8, 256, 2176});
  auto qk_concat = BuildConcatenationOp(
      tensor_pool, {q_kcache_matmul_output, q_kslice_matmul_output},
      {qk_concat_output}, 3);
  std::move(qk_concat.begin(), qk_concat.end(),
            std::back_inserter(op_wrappers));

  // PreMaskReshape
  auto& premask_reshape_output =
      tensor_pool.CloneNativeTensorFrom(qk_concat_output, {8, 2, 128, 2176});
  auto premask_reshape =
      BuildReshapeOp(tensor_pool, {qk_concat_output}, {premask_reshape_output});
  std::move(premask_reshape.begin(), premask_reshape.end(),
            std::back_inserter(op_wrappers));

  // MaskAdd
  auto& mask = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                              quant_param, {1, 1, 128, 2176});
  auto& mask_add_output =
      tensor_pool.CloneNativeTensorFrom(premask_reshape_output);
  auto mask_add = BuildElementwiseAddOp(
      tensor_pool, {premask_reshape_output, mask}, {mask_add_output});
  std::move(mask_add.begin(), mask_add.end(), std::back_inserter(op_wrappers));

  // PostMaskReshape
  auto& postmask_reshape_output =
      tensor_pool.CloneNativeTensorFrom(mask_add_output, {1, 8, 256, 2176});
  auto postmask_reshape =
      BuildReshapeOp(tensor_pool, {mask_add_output}, {postmask_reshape_output});
  std::move(postmask_reshape.begin(), postmask_reshape.end(),
            std::back_inserter(op_wrappers));

  // Softmax
  auto& softmax_output =
      tensor_pool.CloneNativeTensorFrom(postmask_reshape_output);
  auto softmax = BuildSoftmaxOp(tensor_pool, {postmask_reshape_output},
                                {softmax_output}, 1.0f);
  std::move(softmax.begin(), softmax.end(), std::back_inserter(op_wrappers));

  // QKVCacheSlice
  constexpr std::array<int32_t, 4> kQKVCacheSliceBeginData{0, 0, 0, 0};
  auto& qk_vcache_slice_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kQKVCacheSliceBeginData.size()},
      kQKVCacheSliceBeginData.size() * sizeof(kQKVCacheSliceBeginData[0]),
      kQKVCacheSliceBeginData.data());
  constexpr std::array<int32_t, 4> kQKVCacheSliceSizeData{1, 8, 256, 2048};
  auto& qk_vcache_slice_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kQKVCacheSliceSizeData.size()},
      kQKVCacheSliceSizeData.size() * sizeof(kQKVCacheSliceSizeData[0]),
      kQKVCacheSliceSizeData.data());
  auto& qk_vcache_slice_output =
      tensor_pool.CloneNativeTensorFrom(softmax_output, {1, 8, 256, 2048});
  auto qk_vcache_slice = BuildSliceOp(
      tensor_pool,
      {softmax_output, qk_vcache_slice_begin, qk_vcache_slice_size},
      {qk_vcache_slice_output});
  std::move(qk_vcache_slice.begin(), qk_vcache_slice.end(),
            std::back_inserter(op_wrappers));

  // QKVSliceSlice
  constexpr std::array<int32_t, 4> kQKVSliceSliceBeginData{0, 0, 0, 2048};
  auto& qk_vslice_slice_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kQKVSliceSliceBeginData.size()},
      kQKVSliceSliceBeginData.size() * sizeof(kQKVSliceSliceBeginData[0]),
      kQKVSliceSliceBeginData.data());
  constexpr std::array<int32_t, 4> kQKVSliceSliceSizeData{1, 8, 256, 128};
  auto& qk_vslice_slice_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kQKVSliceSliceSizeData.size()},
      kQKVSliceSliceSizeData.size() * sizeof(kQKVSliceSliceSizeData[0]),
      kQKVSliceSliceSizeData.data());
  auto& qk_vslice_slice_output =
      tensor_pool.CloneNativeTensorFrom(softmax_output, {1, 8, 256, 128});
  auto qk_vslice_slice = BuildSliceOp(
      tensor_pool,
      {softmax_output, qk_vslice_slice_begin, qk_vslice_slice_size},
      {qk_vslice_slice_output});
  std::move(qk_vslice_slice.begin(), qk_vslice_slice.end(),
            std::back_inserter(op_wrappers));

  // QKVCacheMatmul
  auto& v_cache = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 8, 2048, 128});
  auto& qk_vcache_matmul_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 8, 256, 128});
  auto qk_vcache_matmul =
      BuildMatmulOp(tensor_pool, {qk_vcache_slice_output, v_cache},
                    {qk_vcache_matmul_output}, false, false);
  std::move(qk_vcache_matmul.begin(), qk_vcache_matmul.end(),
            std::back_inserter(op_wrappers));

  // QKVSliceMatmul
  auto& v_slice =
      tensor_pool.CloneNativeTensorFrom(q_scale_mul_output, {1, 8, 128, 128});
  auto& qk_vslice_matmul_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 8, 256, 128});
  auto qk_vslice_matmul =
      BuildMatmulOp(tensor_pool, {qk_vslice_slice_output, v_slice},
                    {qk_vslice_matmul_output}, false, false);
  std::move(qk_vslice_matmul.begin(), qk_vslice_matmul.end(),
            std::back_inserter(op_wrappers));

  // QKVAdd
  auto& qkv_add_output =
      tensor_pool.CloneNativeTensorFrom(qk_vslice_matmul_output);
  auto qkv_add = BuildElementwiseAddOp(
      tensor_pool, {qk_vcache_matmul_output, qk_vslice_matmul_output},
      {qkv_add_output});
  std::move(qkv_add.begin(), qkv_add.end(), std::back_inserter(op_wrappers));

  // QKVReshape
  auto& qkv_reshape_output =
      tensor_pool.CloneNativeTensorFrom(qkv_add_output, {1, 16, 128, 128});
  auto qkv_reshape =
      BuildReshapeOp(tensor_pool, {qkv_add_output}, {qkv_reshape_output});
  std::move(qkv_reshape.begin(), qkv_reshape.end(),
            std::back_inserter(op_wrappers));

  // QKVTranspose
  constexpr std::array<int32_t, 4> kQKVTransposeVal = {0, 2, 1, 3};
  auto& qkv_transpose_perm = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, quant_param, {kQKVTransposeVal.size()},
      kQKVTransposeVal.size() * sizeof(kQKVTransposeVal[0]),
      kQKVTransposeVal.data());
  auto& qkv_transpose_output =
      tensor_pool.CloneNativeTensorFrom(qkv_reshape_output, {1, 128, 16, 128});
  auto qkv_transpose =
      BuildTransposeOp(tensor_pool, {qkv_reshape_output, qkv_transpose_perm},
                       {qkv_transpose_output});
  std::move(qkv_transpose.begin(), qkv_transpose.end(),
            std::back_inserter(op_wrappers));

  // OProjReshape
  auto& oproj_reshape_output =
      tensor_pool.CloneNativeTensorFrom(qkv_transpose_output, {1, 128, 2048});
  auto oproj_reshape = BuildReshapeOp(tensor_pool, {qkv_transpose_output},
                                      {oproj_reshape_output});
  std::move(oproj_reshape.begin(), oproj_reshape.end(),
            std::back_inserter(op_wrappers));

  ASSERT_EQ(op_wrappers.size(), 17);

  GraphToGraphTransform(::qnn::G2GConfig::kMHAOptPrefill, op_wrappers,
                        tensor_pool, [](OpWrapper& op) { return true; });
  // Check total size after G2G
  ASSERT_EQ(op_wrappers.size(), 198);
  // Check OpCode after G2G
  constexpr size_t kNumUnpack = 5;
  constexpr size_t kNumHead = 16;
  constexpr size_t kShaSize = 12;

  for (size_t i = 0; i < kNumUnpack; ++i) {
    ASSERT_TRUE(op_wrappers[i].IsOpCode(QnnOpCode::kUnPack));
  }

  for (size_t i = 0; i < kNumHead; ++i) {
    ASSERT_TRUE(
        op_wrappers[5 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseMultiply(op_wrappers[5 + kShaSize * i]));
    ASSERT_TRUE(op_wrappers[6 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(op_wrappers[7 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(op_wrappers[8 + kShaSize * i].IsOpCode(QnnOpCode::kConcat));
    ASSERT_TRUE(
        op_wrappers[9 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[9 + kShaSize * i]));
    ASSERT_TRUE(op_wrappers[10 + kShaSize * i].IsOpCode(QnnOpCode::kReshape));
    ASSERT_TRUE(op_wrappers[11 + kShaSize * i].IsOpCode(QnnOpCode::kSoftmax));
    ASSERT_TRUE(
        op_wrappers[12 + kShaSize * i].IsOpCode(QnnOpCode::kStridedSlice));
    ASSERT_TRUE(
        op_wrappers[13 + kShaSize * i].IsOpCode(QnnOpCode::kStridedSlice));
    ASSERT_TRUE(op_wrappers[14 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(op_wrappers[15 + kShaSize * i].IsOpCode(QnnOpCode::kMatMul));
    ASSERT_TRUE(
        op_wrappers[16 + kShaSize * i].IsOpCode(QnnOpCode::kElementWiseBinary));
    ASSERT_TRUE(IsElementWiseAdd(op_wrappers[16 + kShaSize * i]));
  }
  ASSERT_TRUE(op_wrappers[op_wrappers.size() - 1].IsOpCode(QnnOpCode::kConcat));
}

}  // namespace
}  // namespace qnn
