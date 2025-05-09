// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/mha_to_sha.h"

#include <array>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {
namespace {
constexpr size_t kMulIndex = 0;
constexpr size_t kTransposeIndex = 1;
constexpr size_t kMatMulK1Index = 1;
constexpr size_t kMatMulK2Index = 2;
constexpr size_t kConcatIndex = 3;
constexpr size_t kAddIndex = 5;
constexpr size_t kSoftmaxIndex = 7;
constexpr size_t kMatMulV1Index = 10;
constexpr size_t kMatMulV2Index = 11;

std::vector<OpWrapper> TransformToSHA(std::vector<OpWrapper>& ops,
                                      size_t start_id, TensorPool& tensor_pool,
                                      const qnn::TensorWrapper& mha_input,
                                      const qnn::TensorWrapper& mha_output,
                                      const ::qnn::OpWrapper& first_mul,
                                      size_t num_heads, int32_t seq_len) {
  std::vector<OpWrapper> new_ops;

  const auto& mul_const = first_mul.GetInputTensor(1);
  const auto& mul_output_quant_param =
      first_mul.GetOutputTensor(0).GetQuantParams();

  const auto& matmulk_cache = ops[start_id + kMatMulK1Index].GetInputTensor(1);
  const auto& matmulk_cache_output =
      ops[start_id + kMatMulK1Index].GetOutputTensor(0);
  const auto& matmulk_slice = ops[start_id + kMatMulK2Index].GetInputTensor(1);
  const auto& matmulk_slice_output =
      ops[start_id + kMatMulK2Index].GetOutputTensor(0);

  const auto& concat_mha_output =
      ops[start_id + kConcatIndex].GetOutputTensor(0);

  const auto& add_mha_mask = ops[start_id + kAddIndex].GetInputTensor(1);
  const auto& add_mha_output = ops[start_id + kAddIndex].GetOutputTensor(0);

  const auto& softmax_mha_output =
      ops[start_id + kSoftmaxIndex].GetOutputTensor(0);

  const auto& matmulv_cache = ops[start_id + kMatMulV1Index].GetInputTensor(1);
  const auto& matmulv_cache_output =
      ops[start_id + kMatMulV1Index].GetOutputTensor(0);
  const auto& matmulv_slice = ops[start_id + kMatMulV2Index].GetInputTensor(1);
  const auto& matmulv_slice_output =
      ops[start_id + kMatMulV2Index].GetOutputTensor(0);

  // Slice tensor for multiplied by V
  const std::vector<uint32_t> slice1_begin_dim{4};
  const std::array<int32_t, 4> slice1_begin_data{0, 0, 0, 0};
  auto& slice1_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice1_begin_dim,
      slice1_begin_data.size() * sizeof(slice1_begin_data[0]),
      slice1_begin_data.data());
  const std::vector<uint32_t> slice1_size_dim{4};
  const std::array<int32_t, 4> slice1_size_data{1, 1, seq_len, 1280};
  auto& slice1_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice1_size_dim,
      slice1_size_data.size() * sizeof(slice1_size_data[0]),
      slice1_size_data.data());

  const std::vector<uint32_t> slice2_begin_dim{4};
  const std::array<int32_t, 4> slice2_begin_data{0, 0, 0, 1280};
  auto& slice2_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice2_begin_dim,
      slice2_begin_data.size() * sizeof(slice2_begin_data[0]),
      slice2_begin_data.data());
  const std::vector<uint32_t> slice2_size_dim{4};
  const std::array<int32_t, 4> slice2_size_data{1, 1, seq_len, seq_len};
  auto& slice2_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice2_size_dim,
      slice2_size_data.size() * sizeof(slice2_size_data[0]),
      slice2_size_data.data());
  // Split
  const std::vector<uint32_t> split_axis_dim{1};
  const std::array<int32_t, 1> split_axis_data{2};
  auto& split_axis = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, split_axis_dim,
      split_axis_data.size() * sizeof(split_axis_data[0]),
      split_axis_data.data());
  std::vector<::qnn::TensorWrapperRef> head_inputs;
  for (int i = 0; i < num_heads; ++i) {
    auto& split_output = tensor_pool.CloneNativeTensorFrom(
        mha_input, {1, 1, static_cast<uint32_t>(seq_len), 256});
    head_inputs.emplace_back(split_output);
  }
  auto split = BuildSplitOp(
      tensor_pool, {split_axis, const_cast<::qnn::TensorWrapper&>(mha_input)},
      head_inputs, num_heads);
  std::move(split.begin(), split.end(), std::back_inserter(new_ops));

  std::array<::qnn::TensorWrapper*, 4> concat_after_mha;

  for (int i = 0; i < num_heads; ++i) {
    // Mul
    auto& mul_output = tensor_pool.CloneNativeTensorFrom(
        head_inputs[i].get(), mul_output_quant_param);
    auto mul = BuildElementwiseMulOp(
        tensor_pool,
        {head_inputs[i].get(), const_cast<::qnn::TensorWrapper&>(mul_const)},
        {mul_output});
    std::move(mul.begin(), mul.end(), std::back_inserter(new_ops));
    // MatMul 1
    std::vector<uint32_t> matmul1_output_dim = matmulk_cache_output.GetDims();
    matmul1_output_dim[2] = matmul1_output_dim[2] / num_heads;
    auto& matmul1_output = tensor_pool.CloneNativeTensorFrom(
        matmulk_cache_output, matmul1_output_dim);
    auto matmul1 = BuildMatmulOp(
        tensor_pool,
        {mul_output, const_cast<::qnn::TensorWrapper&>(matmulk_cache)},
        {matmul1_output}, false, true);
    std::move(matmul1.begin(), matmul1.end(), std::back_inserter(new_ops));
    // MatMul 2
    std::vector<uint32_t> matmul2_output_dim = matmulk_slice_output.GetDims();
    matmul2_output_dim[2] = matmul2_output_dim[2] / num_heads;
    auto& matmul2_output = tensor_pool.CloneNativeTensorFrom(
        matmulk_slice_output, matmul2_output_dim);
    auto matmul2 = BuildMatmulOp(
        tensor_pool,
        {mul_output, const_cast<::qnn::TensorWrapper&>(matmulk_slice)},
        {matmul2_output}, false, true);
    std::move(matmul2.begin(), matmul2.end(), std::back_inserter(new_ops));
    // Concat
    std::vector<uint32_t> concat_output_dim = matmul1_output.GetDims();
    concat_output_dim[3] += matmul2_output.GetDim(3);
    auto& concat_output =
        tensor_pool.CloneNativeTensorFrom(concat_mha_output, concat_output_dim);
    auto concat = BuildConcatenationOp(
        tensor_pool, {matmul1_output, matmul2_output}, {concat_output}, 3);
    std::move(concat.begin(), concat.end(), std::back_inserter(new_ops));
    // Add
    auto& add_output = tensor_pool.CloneNativeTensorFrom(
        add_mha_output, concat_output.GetDims());
    auto add = BuildElementwiseAddOp(
        tensor_pool,
        {concat_output, const_cast<::qnn::TensorWrapper&>(add_mha_mask)},
        {add_output});
    std::move(add.begin(), add.end(), std::back_inserter(new_ops));
    // Softmax
    auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
        softmax_mha_output, add_output.GetDims());
    auto softmax =
        BuildSoftmaxOp(tensor_pool, {add_output}, {softmax_output}, 1.0f);
    std::move(softmax.begin(), softmax.end(), std::back_inserter(new_ops));
    // Slice 1
    const std::vector<uint32_t> slice1_output_dim{
        1, 1, static_cast<uint32_t>(seq_len), 1280};
    auto& slice1_output =
        tensor_pool.CloneNativeTensorFrom(softmax_output, slice1_output_dim);
    auto slice1 =
        BuildSliceOp(tensor_pool, {softmax_output, slice1_begin, slice1_size},
                     {slice1_output});
    std::move(slice1.begin(), slice1.end(), std::back_inserter(new_ops));
    // MatMul 1
    std::vector<uint32_t> matmul1_v_output_dim = matmulv_cache_output.GetDims();
    matmul1_v_output_dim[2] = matmul1_v_output_dim[2] / num_heads;
    auto& matmul1_v_output = tensor_pool.CloneNativeTensorFrom(
        matmulv_cache_output, matmul1_v_output_dim);
    auto matmul1_v = BuildMatmulOp(
        tensor_pool,
        {slice1_output, const_cast<::qnn::TensorWrapper&>(matmulv_cache)},
        {matmul1_v_output}, false, true);
    std::move(matmul1_v.begin(), matmul1_v.end(), std::back_inserter(new_ops));
    // Slice 2
    auto& slice2_output = tensor_pool.CloneNativeTensorFrom(
        softmax_output,
        {1, 1, static_cast<uint32_t>(seq_len), static_cast<uint32_t>(seq_len)});
    auto slice2 =
        BuildSliceOp(tensor_pool, {softmax_output, slice2_begin, slice2_size},
                     {slice2_output});
    std::move(slice2.begin(), slice2.end(), std::back_inserter(new_ops));
    // MatMul 2
    std::vector<uint32_t> matmul2_v_output_dim = matmulv_slice_output.GetDims();
    matmul2_v_output_dim[2] = matmul2_v_output_dim[2] / num_heads;
    auto& matmul2_v_output = tensor_pool.CloneNativeTensorFrom(
        matmulv_slice_output, matmul2_v_output_dim);
    auto matmul2_v = BuildMatmulOp(
        tensor_pool,
        {slice2_output, const_cast<::qnn::TensorWrapper&>(matmulv_slice)},
        {matmul2_v_output}, false, true);
    std::move(matmul2_v.begin(), matmul2_v.end(), std::back_inserter(new_ops));
    // Add
    auto& add_final_output = tensor_pool.CloneNativeTensorFrom(
        mha_output, matmul1_v_output.GetDims());
    concat_after_mha[i] = &add_final_output;
    auto add_final = BuildElementwiseAddOp(
        tensor_pool, {matmul1_v_output, matmul2_v_output}, {add_final_output});
    std::move(add_final.begin(), add_final.end(), std::back_inserter(new_ops));
  }
  // Concat
  std::vector<::qnn::TensorWrapperRef> concat_final_inputs;
  for (int i = 0; i < num_heads; ++i) {
    concat_final_inputs.emplace_back(*(concat_after_mha[i]));
  }
  auto concat_dims = mha_output.GetDims();
  concat_dims.insert(concat_dims.begin(), 1);
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(mha_output, concat_dims);
  auto concat_final = BuildConcatenationOp(tensor_pool, concat_final_inputs,
                                           {concat_output}, 3);
  std::move(concat_final.begin(), concat_final.end(),
            std::back_inserter(new_ops));
  // Reshape
  auto reshape =
      BuildReshapeOp(tensor_pool, {concat_output},
                     {const_cast<::qnn::TensorWrapper&>(mha_output)});
  std::move(reshape.begin(), reshape.end(), std::back_inserter(new_ops));

  return new_ops;
}

}  // namespace

bool OptimizeMHAPrefill(std::vector<OpWrapper>& ops, size_t start_id,
                        TensorPool& tensor_pool, size_t pattern_size) {
  QNN_LOG_INFO("[G2G] MHA optimization (Prefill)");

  std::vector<OpWrapper> new_ops;
  const auto& first_mul = ops[start_id + kMulIndex];
  const auto& pattern_input = first_mul.GetInputTensor(0);
  const auto& pattern_output =
      ops[start_id + pattern_size - 1].GetOutputTensor(0);

  // Transpose
  auto transpose_output_dims =
      ops[start_id + kTransposeIndex].GetOutputTensor(0).GetDims();
  auto& transpose_output =
      tensor_pool.CloneNativeTensorFrom(pattern_input, transpose_output_dims);
  auto transpose = BuildTransposeOp(
      tensor_pool,
      {const_cast<::qnn::TensorWrapper&>(pattern_input),
       const_cast<::qnn::TensorWrapper&>(
           ops[start_id + kTransposeIndex].GetTensorPararm(0).GetTensor())},
      {transpose_output});
  std::move(transpose.begin(), transpose.end(), std::back_inserter(new_ops));

  // Reshape
  auto& reshape_output = tensor_pool.CloneNativeTensorFrom(
      pattern_input, {transpose_output_dims[0], 1,
                      transpose_output_dims[1] * transpose_output_dims[2],
                      transpose_output_dims[3]});
  auto reshape =
      BuildReshapeOp(tensor_pool, {transpose_output}, {reshape_output});
  std::move(reshape.begin(), reshape.end(), std::back_inserter(new_ops));

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDim(2);
  const int32_t seq_len = pattern_input.GetDim(1);
  const auto& mha_input = new_ops.back().GetOutputTensor(0);
  auto sha_ops =
      TransformToSHA(ops, start_id + new_ops.size(), tensor_pool, mha_input,
                     pattern_output, first_mul, num_heads, seq_len);
  std::move(sha_ops.begin(), sha_ops.end(), std::back_inserter(new_ops));

  // Replace the matched pattern with a newly generated subgraph.
  ops.insert(ops.begin() + start_id + pattern_size,
             std::make_move_iterator(new_ops.begin()),
             std::make_move_iterator(new_ops.end()));
  ops.erase(ops.begin() + start_id, ops.begin() + start_id + pattern_size);
  return true;
}

bool OptimizeMHADecode(std::vector<OpWrapper>& ops, size_t start_id,
                       TensorPool& tensor_pool, size_t pattern_size) {
  QNN_LOG_INFO("[G2G] MHA optimization (Decode)");

  std::vector<OpWrapper> new_ops;
  const auto& first_mul = ops[start_id + kMulIndex];
  const auto& pattern_input = first_mul.GetInputTensor(0);
  const auto& pattern_output =
      ops[start_id + pattern_size - 1].GetOutputTensor(0);

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDim(2);
  const int32_t seq_len = pattern_input.GetDim(1);
  auto sha_ops =
      TransformToSHA(ops, start_id + new_ops.size(), tensor_pool, pattern_input,
                     pattern_output, first_mul, num_heads, seq_len);
  std::move(sha_ops.begin(), sha_ops.end(), std::back_inserter(new_ops));

  // Replace the matched pattern with a newly generated subgraph.
  ops.insert(ops.begin() + start_id + pattern_size,
             std::make_move_iterator(new_ops.begin()),
             std::make_move_iterator(new_ops.end()));
  ops.erase(ops.begin() + start_id, ops.begin() + start_id + pattern_size);
  return true;
}
}  // namespace qnn
