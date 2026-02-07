// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/mha_to_sha.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pack_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/unpack_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {

// Returns a boolean indicating whether the output tensor of op1 at out_index is
// connected to the input tensor of op2 at in_index.
#define IS_CONNECTED(op1, out_index, op2, in_index)     \
  (ops[start_index + op1].GetOutputTensor(out_index) == \
   ops[start_index + op2].GetInputTensor(in_index))

constexpr size_t kMulIndex = 0;
constexpr size_t kTransposePrefillIndex = 1;
constexpr size_t kReshapePrefillIndex = 2;
constexpr size_t kMatMulK1Index = 1;
constexpr size_t kMatMulK2Index = 2;
constexpr size_t kConcatIndex = 3;
constexpr size_t kReshape0Index = 4;
constexpr size_t kAddIndex = 5;
constexpr size_t kReshape1Index = 6;
constexpr size_t kSoftmaxIndex = 7;
constexpr size_t kSlice1Index = 8;
constexpr size_t kSlice2Index = 9;
constexpr size_t kMatMulV1Index = 10;
constexpr size_t kMatMulV2Index = 11;
constexpr size_t kAdd2Index = 12;
constexpr size_t kReshape2Index = 13;
constexpr size_t kTranspose2Index = 14;
constexpr size_t kReshape3Index = 15;

// QNN Slice Param ranges in the form (begin, end, stride) for each axis. To
// set the 3rd axis "end" value, we need to access ranges[3 * 2 + 2 - 1 = 7].
constexpr size_t kSlice3rdAxisEndIndex = 7;

const TensorWrapper& BuildSingleSHA(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    const TensorWrapper& sha_input, const TensorWrapper& mask, size_t num_heads,
    const OpWrapper& mul, const OpWrapper& matmul_k1,
    const OpWrapper& matmul_k2, const OpWrapper& concat, const OpWrapper& add_1,
    const OpWrapper& softmax, const OpWrapper& slice_1,
    const OpWrapper& slice_2, const OpWrapper& matmul_v1,
    const OpWrapper& matmul_v2, const OpWrapper& add_2) {
  // Mul
  const auto& mul_output = tensor_pool.CloneNativeTensorFrom(
      mul.GetOutputTensor(0), sha_input.GetDims());
  new_ops.emplace_back(
      CreateElementWiseMulOp(sha_input, mul.GetInputTensor(1), mul_output));

  // MatMul 1
  auto matmul_k1_output_dims = matmul_k1.GetOutputTensor(0).GetDims();
  matmul_k1_output_dims[2] /= num_heads;
  const auto& matmul_k1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k1.GetOutputTensor(0), matmul_k1_output_dims);
  new_ops.emplace_back(CreateMatmulOpWithSameParam(
      matmul_k1, mul_output, matmul_k1.GetInputTensor(1), matmul_k1_output));

  // MatMul 2
  auto matmul_k2_output_dims = matmul_k2.GetOutputTensor(0).GetDims();
  matmul_k2_output_dims[2] /= num_heads;
  const auto& matmul_k2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k2.GetOutputTensor(0), matmul_k2_output_dims);
  new_ops.emplace_back(CreateMatmulOpWithSameParam(
      matmul_k2, mul_output, matmul_k2.GetInputTensor(1), matmul_k2_output));

  // Concat
  auto concat_output_dims = matmul_k1_output_dims;
  concat_output_dims[3] += matmul_k2_output_dims[3];
  const auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      concat.GetOutputTensor(0), concat_output_dims);
  new_ops.emplace_back(CreateConcatenationOpWithSameParam(
      concat, {matmul_k1_output, matmul_k2_output}, concat_output));

  // Add
  const auto& add_1_output = tensor_pool.CloneNativeTensorFrom(
      add_1.GetOutputTensor(0), concat_output.GetDims());
  new_ops.emplace_back(
      CreateElementWiseAddOp(concat_output, mask, add_1_output));
  // Softmax
  const auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), add_1_output.GetDims());
  new_ops.emplace_back(
      CreateSoftmaxOpWithSameParam(softmax, add_1_output, softmax_output));

  // Slice 1
  auto slice_1_ranges = slice_1.GetTensorPararm(0).GetTensor();
  auto slice_1_rangs_data = slice_1_ranges.GetTensorData<int32_t>();
  std::vector<int32_t> sha_slice_1_ranges_data(
      slice_1_rangs_data.value().begin(), slice_1_rangs_data.value().end());
  sha_slice_1_ranges_data[kSlice3rdAxisEndIndex] /= num_heads;
  const auto& sha_slice_1_ranges = tensor_pool.CreateStaticTensor(
      slice_1_ranges.GetDataType(), slice_1_ranges.GetQuantParams(),
      slice_1_ranges.GetDims(), slice_1_ranges.GetTensorBytes(),
      sha_slice_1_ranges_data.data());
  auto slice_1_output_dims = slice_1.GetOutputTensor(0).GetDims();
  slice_1_output_dims[2] /= num_heads;
  const auto& slice_1_output = tensor_pool.CloneNativeTensorFrom(
      slice_1.GetOutputTensor(0), slice_1_output_dims);
  new_ops.emplace_back(
      CreateSliceOp(softmax_output, slice_1_output, sha_slice_1_ranges));

  // Slice 2
  auto slice_2_ranges = slice_2.GetTensorPararm(0).GetTensor();
  auto slice_2_ranges_data = slice_2_ranges.GetTensorData<int32_t>();
  std::vector<int32_t> sha_slice_2_ranges_data(
      slice_2_ranges_data.value().begin(), slice_2_ranges_data.value().end());
  sha_slice_2_ranges_data[kSlice3rdAxisEndIndex] /= num_heads;
  const auto& sha_slice_2_ranges = tensor_pool.CreateStaticTensor(
      slice_2_ranges.GetDataType(), slice_2_ranges.GetQuantParams(),
      slice_2_ranges.GetDims(), slice_2_ranges.GetTensorBytes(),
      sha_slice_2_ranges_data.data());
  auto slice_2_output_dims = slice_2.GetOutputTensor(0).GetDims();
  slice_2_output_dims[2] /= num_heads;
  const auto& slice_2_output = tensor_pool.CloneNativeTensorFrom(
      slice_2.GetOutputTensor(0), slice_2_output_dims);
  new_ops.emplace_back(
      CreateSliceOp(softmax_output, slice_2_output, sha_slice_2_ranges));

  // MatMul 1
  std::vector<uint32_t> matmul_v1_output_dims =
      matmul_v1.GetOutputTensor(0).GetDims();
  matmul_v1_output_dims[2] = matmul_v1_output_dims[2] / num_heads;
  const auto& matmul_v1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v1.GetOutputTensor(0), matmul_v1_output_dims);
  new_ops.emplace_back(CreateMatmulOpWithSameParam(matmul_v1, slice_1_output,
                                                   matmul_v1.GetInputTensor(1),
                                                   matmul_v1_output));

  // MatMul 2
  std::vector<uint32_t> matmul_v2_output_dims =
      matmul_v2.GetOutputTensor(0).GetDims();
  matmul_v2_output_dims[2] = matmul_v2_output_dims[2] / num_heads;
  const auto& matmul_v2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v2.GetOutputTensor(0), matmul_v2_output_dims);
  new_ops.emplace_back(CreateMatmulOpWithSameParam(matmul_v2, slice_2_output,
                                                   matmul_v2.GetInputTensor(1),
                                                   matmul_v2_output));

  // Add 2
  const auto& add_2_output = tensor_pool.CloneNativeTensorFrom(
      add_2.GetOutputTensor(0), matmul_v1_output.GetDims());
  new_ops.emplace_back(
      CreateElementWiseAddOp(matmul_v1_output, matmul_v2_output, add_2_output));
  return add_2_output;
}

const TensorWrapper& BuildSingleSHA(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    const TensorWrapper& v_cache, const TensorWrapper& q_slice,
    const TensorWrapper& sha_mul_input, const TensorWrapper& sha_add_input_1,
    const TensorWrapper& sha_add_input_2, const TensorWrapper& v_slice,
    const uint32_t num_attn_per_kv_heads, const OpWrapper& mul,
    const OpWrapper& matmul_k1, const OpWrapper& add_1,
    const OpWrapper& matmul_k2, const OpWrapper& concat, const OpWrapper& add_2,
    const OpWrapper& reshape_3, const OpWrapper& softmax,
    const OpWrapper& slice_1, const OpWrapper& slice_2,
    const OpWrapper& matmul_v1, const OpWrapper& matmul_v2,
    const OpWrapper& add_3) {
  // Mul
  auto mul_output_dims = mul.GetOutputTensor(0).GetDims();
  mul_output_dims.erase(mul_output_dims.begin() + 1);
  const auto& mul_output = tensor_pool.CloneNativeTensorFrom(
      mul.GetOutputTensor(0), mul_output_dims);
  new_ops.emplace_back(
      CreateElementWiseMulOp(sha_mul_input, mul.GetInputTensor(1), mul_output));

  // Matmul q
  auto matmul_q_output_dims = matmul_k1.GetOutputTensor(0).GetDims();
  matmul_q_output_dims.erase(matmul_q_output_dims.begin() + 1);
  matmul_q_output_dims[1] /= num_attn_per_kv_heads;
  const auto& matmul_q_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k1.GetOutputTensor(0), matmul_q_output_dims);
  new_ops.emplace_back(CreateMatmulOpWithSameParam(matmul_k1, mul_output,
                                                   q_slice, matmul_q_output));

  // Add
  auto add_1_output_dims = add_1.GetOutputTensor(0).GetDims();
  add_1_output_dims.erase(add_1_output_dims.begin() + 1);
  const auto& add_1_output = tensor_pool.CloneNativeTensorFrom(
      add_1.GetOutputTensor(0), add_1_output_dims);
  new_ops.emplace_back(
      CreateElementWiseAddOp(sha_add_input_1, sha_add_input_2, add_1_output));

  // Matmul k
  auto matmul_qk_output_dims = matmul_k2.GetOutputTensor(0).GetDims();
  matmul_qk_output_dims.erase(matmul_qk_output_dims.begin() + 1);
  matmul_qk_output_dims[1] /= num_attn_per_kv_heads;
  const auto& matmul_qk_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k2.GetOutputTensor(0), matmul_qk_output_dims);
  new_ops.emplace_back(CreateMatmulOpWithSameParam(
      matmul_k2, mul_output, add_1_output, matmul_qk_output));

  // Concat
  std::uint32_t adjusted_axis = 2;
  auto concat_output_dims = concat.GetOutputTensor(0).GetDims();
  concat_output_dims.erase(concat_output_dims.begin() + 1);
  concat_output_dims[1] /= num_attn_per_kv_heads;
  const auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      concat.GetOutputTensor(0), concat_output_dims);
  new_ops.emplace_back(CreateConcatenationOp(
      {matmul_q_output, matmul_qk_output}, concat_output, adjusted_axis));

  // Add 2
  auto add_2_output_dims = add_2.GetOutputTensor(0).GetDims();
  add_2_output_dims[0] = 1;
  add_2_output_dims[1] = 1;
  const auto& add_2_output = tensor_pool.CloneNativeTensorFrom(
      add_2.GetOutputTensor(0), add_2_output_dims);
  new_ops.emplace_back(CreateElementWiseAddOp(
      concat_output, add_2.GetInputTensor(1), add_2_output));

  // Reshape 3
  auto reshape_3_output_dims = reshape_3.GetOutputTensor(0).GetDims();
  reshape_3_output_dims.erase(reshape_3_output_dims.begin() + 1);
  reshape_3_output_dims[1] /= num_attn_per_kv_heads;
  const auto& reshape_3_output = tensor_pool.CloneNativeTensorFrom(
      reshape_3.GetOutputTensor(0), reshape_3_output_dims);
  new_ops.emplace_back(CreateReshapeOp(add_2_output, reshape_3_output));

  // Softmax
  auto softmax_output_dims = softmax.GetOutputTensor(0).GetDims();
  softmax_output_dims.erase(softmax_output_dims.begin() + 1);
  softmax_output_dims[1] /= num_attn_per_kv_heads;
  const auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), softmax_output_dims);
  new_ops.emplace_back(
      CreateSoftmaxOpWithSameParam(softmax, reshape_3_output, softmax_output));

  // Slice 1
  auto slice_1_param = slice_1.GetTensorPararm(0).GetTensor();
  auto slice_1_param_data = slice_1_param.GetTensorData<int32_t>();
  std::vector<int32_t> slice_1_ranges(slice_1_param_data.value().begin(),
                                      slice_1_param_data.value().end());
  slice_1_ranges.erase(slice_1_ranges.begin() + 3, slice_1_ranges.begin() + 6);
  slice_1_ranges[4] /= num_attn_per_kv_heads;
  std::vector<uint32_t> slice_1_param_dims = {
      static_cast<uint32_t>(slice_1_ranges.size() / 3), 3};
  const auto& slice_1_param_tensor = tensor_pool.CreateStaticTensor(
      slice_1_param.GetDataType(), slice_1_param.GetQuantParams(),
      slice_1_param_dims, sizeof(int32_t) * slice_1_ranges.size(),
      slice_1_ranges.data());
  auto slice_1_output_dims = slice_1.GetOutputTensor(0).GetDims();
  slice_1_output_dims.erase(slice_1_output_dims.begin() + 1);
  slice_1_output_dims[1] /= num_attn_per_kv_heads;
  const auto& slice_1_output = tensor_pool.CloneNativeTensorFrom(
      slice_1.GetOutputTensor(0), slice_1_output_dims);
  new_ops.emplace_back(
      CreateSliceOp(softmax_output, slice_1_output, slice_1_param_tensor));

  // Slice 2
  auto slice_2_param = slice_2.GetTensorPararm(0).GetTensor();
  auto slice_2_param_data = slice_2_param.GetTensorData<int32_t>();
  std::vector<int32_t> slice_2_ranges(slice_2_param_data.value().begin(),
                                      slice_2_param_data.value().end());
  slice_2_ranges.erase(slice_2_ranges.begin() + 3, slice_2_ranges.begin() + 6);
  slice_2_ranges[4] /= num_attn_per_kv_heads;
  std::vector<uint32_t> slice_2_param_dims = {
      static_cast<uint32_t>(slice_2_ranges.size() / 3), 3};
  const auto& slice_2_param_tensor = tensor_pool.CreateStaticTensor(
      slice_2_param.GetDataType(), slice_2_param.GetQuantParams(),
      slice_2_param_dims, sizeof(int32_t) * slice_2_ranges.size(),
      slice_2_ranges.data());
  auto slice_2_output_dims = slice_2.GetOutputTensor(0).GetDims();
  slice_2_output_dims.erase(slice_2_output_dims.begin() + 1);
  slice_2_output_dims[1] /= num_attn_per_kv_heads;
  const auto& slice_2_output = tensor_pool.CloneNativeTensorFrom(
      slice_2.GetOutputTensor(0), slice_2_output_dims);
  new_ops.emplace_back(
      CreateSliceOp(softmax_output, slice_2_output, slice_2_param_tensor));

  // Matmul v1
  auto matmul_v1_output_dims = matmul_v1.GetOutputTensor(0).GetDims();
  matmul_v1_output_dims.erase(matmul_v1_output_dims.begin() + 1);
  matmul_v1_output_dims[1] /= num_attn_per_kv_heads;
  const auto& matmul_v1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v1.GetOutputTensor(0), matmul_v1_output_dims);
  new_ops.emplace_back(CreateMatmulOpWithSameParam(matmul_v1, slice_1_output,
                                                   v_cache, matmul_v1_output));

  // Matmul v2
  auto matmul_v2_output_dims = matmul_v2.GetOutputTensor(0).GetDims();
  matmul_v2_output_dims.erase(matmul_v2_output_dims.begin() + 1);
  matmul_v2_output_dims[1] /= num_attn_per_kv_heads;
  const auto& matmul_v2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v2.GetOutputTensor(0), matmul_v2_output_dims);
  new_ops.emplace_back(CreateMatmulOpWithSameParam(matmul_v2, slice_2_output,
                                                   v_slice, matmul_v2_output));

  // Add 3
  auto add_3_output_dims = add_3.GetOutputTensor(0).GetDims();
  add_3_output_dims.erase(add_3_output_dims.begin() + 1);
  add_3_output_dims[1] /= num_attn_per_kv_heads;
  const auto& add_3_output = tensor_pool.CloneNativeTensorFrom(
      add_3.GetOutputTensor(0), add_3_output_dims);
  new_ops.emplace_back(
      CreateElementWiseAddOp(matmul_v1_output, matmul_v2_output, add_3_output));

  return add_3_output;
}

void CloneNamespace(const OpWrapper& source, OpWrapper& destination) {
  absl::string_view start_op_name = source.GetName();
  size_t pos = start_op_name.rfind('/');
  if (pos == absl::string_view::npos) {
    return;
  }
  destination.AddPrefixToName(absl::StrCat(start_op_name.substr(0, pos), "/"));
}

void CloneNamespace(const OpWrapper& source, std::vector<OpWrapper>& ops) {
  for (auto& op : ops) {
    CloneNamespace(source, op);
  }
}

std::vector<OpWrapper> TransformToSHA(
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    const TensorWrapper& mha_input, const TensorWrapper& mha_output,
    const OpWrapper& scaling_mul, size_t num_heads) {
  std::vector<OpWrapper> new_ops;

  const auto& split_input = ops[start_index].GetOutputTensor(0);
  // Prepare inputs for num_heads SHAs.
  std::vector<ConstTensorWrapperRef> sha_inputs;
  sha_inputs.reserve(num_heads);
  for (int i = 0; i < num_heads; ++i) {
    auto head_input_dims = split_input.GetDims();
    head_input_dims[2] /= num_heads;
    const auto& split_output =
        tensor_pool.CloneNativeTensorFrom(mha_input, head_input_dims);
    sha_inputs.emplace_back(split_output);
  }
  // Split
  std::vector<std::uint32_t> split_indice;
  split_indice.reserve(num_heads);
  for (std::uint32_t i = 1; i < num_heads; i++) {
    split_indice.emplace_back(i * split_input.GetDim(2) / num_heads);
  }
  const auto& split_indice_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {},
      {static_cast<std::uint32_t>(split_indice.size())},
      sizeof(std::uint32_t) * split_indice.size(), split_indice.data());
  auto& split = new_ops.emplace_back(
      CreateSplitOp(split_input, sha_inputs, 2, split_indice_tensor));
  CloneNamespace(ops[start_index], split);
  // Prepare outputs for num_heads SHAs.
  std::vector<ConstTensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_heads);
  // Create num_heads SHA.
  for (int i = 0; i < num_heads; ++i) {
    OpWrapper& add_1 = ops[start_index + kAddIndex];
    const auto& mask = add_1.GetInputTensor(1);
    sha_outputs.emplace_back(BuildSingleSHA(
        new_ops, tensor_pool, sha_inputs[i].get(), mask, num_heads, scaling_mul,
        ops[start_index + kMatMulK1Index], ops[start_index + kMatMulK2Index],
        ops[start_index + kConcatIndex], add_1,
        ops[start_index + kSoftmaxIndex], ops[start_index + kSlice1Index],
        ops[start_index + kSlice2Index], ops[start_index + kMatMulV1Index],
        ops[start_index + kMatMulV2Index], ops[start_index + kAdd2Index]));
  }
  // Concat
  auto concat_dims = mha_output.GetDims();
  concat_dims.insert(concat_dims.begin(), 1);
  const auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(mha_output, concat_dims);
  auto& concat_final = new_ops.emplace_back(
      CreateConcatenationOp(sha_outputs, concat_output, 3));
  CloneNamespace(ops[start_index], concat_final);
  // Reshape
  auto& reshape =
      new_ops.emplace_back(CreateReshapeOp(concat_output, mha_output));
  CloneNamespace(ops[start_index], reshape);
  return new_ops;
}

}  // namespace

size_t OptimizeMHAPrefill(std::function<bool(OpWrapper&)> validate_op_config,
                          std::vector<OpWrapper>& ops, size_t start_index,
                          TensorPool& tensor_pool, size_t pattern_size) {
  // Connection check
  if (!(IS_CONNECTED(kMulIndex, 0, kTransposePrefillIndex, 0) &&
        IS_CONNECTED(kTransposePrefillIndex, 0, kReshapePrefillIndex, 0) &&
        IS_CONNECTED(kReshapePrefillIndex, 0, kMatMulK1Index + 2, 0) &&
        IS_CONNECTED(kReshapePrefillIndex, 0, kMatMulK2Index + 2, 0) &&
        IS_CONNECTED(kMatMulK1Index + 2, 0, kConcatIndex + 2, 0) &&
        IS_CONNECTED(kMatMulK2Index + 2, 0, kConcatIndex + 2, 1) &&
        IS_CONNECTED(kConcatIndex + 2, 0, kReshape0Index + 2, 0) &&
        IS_CONNECTED(kReshape0Index + 2, 0, kAddIndex + 2, 0) &&
        IS_CONNECTED(kAddIndex + 2, 0, kReshape1Index + 2, 0) &&
        IS_CONNECTED(kReshape1Index + 2, 0, kSoftmaxIndex + 2, 0) &&
        IS_CONNECTED(kSoftmaxIndex + 2, 0, kSlice1Index + 2, 0) &&
        IS_CONNECTED(kSoftmaxIndex + 2, 0, kSlice2Index + 2, 0) &&
        IS_CONNECTED(kSlice1Index + 2, 0, kMatMulV1Index + 2, 0) &&
        IS_CONNECTED(kSlice2Index + 2, 0, kMatMulV2Index + 2, 0) &&
        IS_CONNECTED(kMatMulV1Index + 2, 0, kAdd2Index + 2, 0) &&
        IS_CONNECTED(kMatMulV2Index + 2, 0, kAdd2Index + 2, 1) &&
        IS_CONNECTED(kAdd2Index + 2, 0, kReshape2Index + 2, 0) &&
        IS_CONNECTED(kReshape2Index + 2, 0, kTranspose2Index + 2, 0) &&
        IS_CONNECTED(kTranspose2Index + 2, 0, kReshape3Index + 2, 0) &&
        IsElementWiseMultiply(ops[start_index + kMulIndex]) &&
        IsElementWiseAdd(ops[start_index + kAddIndex + 2]) &&
        IsElementWiseAdd(ops[start_index + kAdd2Index + 2]))) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] MHA optimization (Prefill)");
  std::vector<OpWrapper> new_ops;
  const auto& scaling_mul = ops[start_index + kMulIndex];
  const auto& pattern_input = scaling_mul.GetInputTensor(0);
  const auto& pattern_output =
      ops[start_index + pattern_size - 1].GetOutputTensor(0);

  // Transpose
  auto transpose_output_dims =
      ops[start_index + kTransposePrefillIndex].GetOutputTensor(0).GetDims();
  const auto& transpose_output =
      tensor_pool.CloneNativeTensorFrom(pattern_input, transpose_output_dims);
  new_ops.emplace_back(
      CreateTransposeOpWithSameParam(ops[start_index + kTransposePrefillIndex],
                                     pattern_input, transpose_output));

  // Reshape
  const auto& reshape_output = tensor_pool.CloneNativeTensorFrom(
      pattern_input, {transpose_output_dims[0], 1,
                      transpose_output_dims[1] * transpose_output_dims[2],
                      transpose_output_dims[3]});
  new_ops.emplace_back(CreateReshapeOp(transpose_output, reshape_output));

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDim(2);
  const auto& mha_input = new_ops.back().GetOutputTensor(0);
  auto sha_ops =
      TransformToSHA(ops, start_index + new_ops.size(), tensor_pool, mha_input,
                     pattern_output, scaling_mul, num_heads);
  std::move(sha_ops.begin(), sha_ops.end(), std::back_inserter(new_ops));

  // Validate new graph.
  // TODO(jiunkaiy): Disable bypassing Split int16 op validator.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](OpWrapper& op_wrapper) -> bool {
                    return op_wrapper.IsOpCode(QnnOpCode::kSplit) ||
                           validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Adjust the name to avoid a name collision in the Qnn JSON dump.
    for (size_t i = 0; i < new_ops.size(); ++i) {
      new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
    }
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size();
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

size_t OptimizeMHADecode(std::function<bool(OpWrapper&)> validate_op_config,
                         std::vector<OpWrapper>& ops, size_t start_index,
                         TensorPool& tensor_pool, size_t pattern_size) {
  // Connection check
  if (!(IS_CONNECTED(kMulIndex, 0, kMatMulK1Index, 0) &&
        IS_CONNECTED(kMulIndex, 0, kMatMulK2Index, 0) &&
        IS_CONNECTED(kMatMulK1Index, 0, kConcatIndex, 0) &&
        IS_CONNECTED(kMatMulK2Index, 0, kConcatIndex, 1) &&
        IS_CONNECTED(kConcatIndex, 0, kReshape0Index, 0) &&
        IS_CONNECTED(kReshape0Index, 0, kAddIndex, 0) &&
        IS_CONNECTED(kAddIndex, 0, kReshape1Index, 0) &&
        IS_CONNECTED(kReshape1Index, 0, kSoftmaxIndex, 0) &&
        IS_CONNECTED(kSoftmaxIndex, 0, kSlice1Index, 0) &&
        IS_CONNECTED(kSoftmaxIndex, 0, kSlice2Index, 0) &&
        IS_CONNECTED(kSlice1Index, 0, kMatMulV1Index, 0) &&
        IS_CONNECTED(kSlice2Index, 0, kMatMulV2Index, 0) &&
        IS_CONNECTED(kMatMulV1Index, 0, kAdd2Index, 0) &&
        IS_CONNECTED(kMatMulV2Index, 0, kAdd2Index, 1) &&
        IS_CONNECTED(kAdd2Index, 0, kReshape2Index, 0) &&
        IsElementWiseMultiply(ops[start_index + kMulIndex]) &&
        IsElementWiseAdd(ops[start_index + kAddIndex]) &&
        IsElementWiseAdd(ops[start_index + kAdd2Index]))) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] MHA optimization (Decode)");
  std::vector<OpWrapper> new_ops;
  const auto& scaling_mul = ops[start_index + kMulIndex];
  const auto& pattern_input = scaling_mul.GetInputTensor(0);
  const auto& pattern_output =
      ops[start_index + pattern_size - 1].GetOutputTensor(0);

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDim(2);
  auto sha_ops =
      TransformToSHA(ops, start_index + new_ops.size(), tensor_pool,
                     pattern_input, pattern_output, scaling_mul, num_heads);
  std::move(sha_ops.begin(), sha_ops.end(), std::back_inserter(new_ops));

  // Validate new graph.
  // TODO(jiunkaiy): Disable bypassing Split int16 op validator.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](OpWrapper& op_wrapper) -> bool {
                    return op_wrapper.IsOpCode(QnnOpCode::kSplit) ||
                           validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Adjust the name to avoid a name collision in the Qnn JSON dump.
    for (size_t i = 0; i < new_ops.size(); ++i) {
      new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
    }
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size();
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

size_t OptimizeMHAFastVlmPrefill(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr size_t add_1_index = -2;
  constexpr size_t mul_index = 0;
  constexpr size_t reshape_1_index = 1;
  constexpr size_t matmul_q_index = 2;
  constexpr size_t matmul_qk_index = 3;
  constexpr size_t concat_index = 4;
  constexpr size_t reshape_2_index = 5;
  constexpr size_t add_2_index = 6;
  constexpr size_t reshape_3_index = 7;
  constexpr size_t softmax_index = 8;
  constexpr size_t slice_1_index = 9;
  constexpr size_t slice_2_index = 10;
  constexpr size_t matmul_v1_index = 11;
  constexpr size_t matmul_v2_index = 12;
  constexpr size_t add_3_index = 13;
  constexpr size_t reshape_4_index = 14;
  constexpr size_t transpose_2_index = 15;
  constexpr size_t reshape_5_index = 16;

  const auto is_connected =
      [&ops, &start_index](size_t output_op_index, size_t output_tensor_index,
                           size_t input_op_index,
                           size_t input_tensor_index) -> bool {
    return ops[start_index + output_op_index].GetOutputTensor(
               output_tensor_index) ==
           ops[start_index + input_op_index].GetInputTensor(input_tensor_index);
  };
  if (!(is_connected(add_1_index, 0, matmul_qk_index, 1) &&
        is_connected(mul_index, 0, reshape_1_index, 0) &&
        is_connected(reshape_1_index, 0, matmul_q_index, 0) &&
        is_connected(reshape_1_index, 0, matmul_qk_index, 0) &&
        is_connected(matmul_q_index, 0, concat_index, 0) &&
        is_connected(matmul_qk_index, 0, concat_index, 1) &&
        is_connected(concat_index, 0, reshape_2_index, 0) &&
        is_connected(reshape_2_index, 0, add_2_index, 0) &&
        is_connected(add_2_index, 0, reshape_3_index, 0) &&
        is_connected(reshape_3_index, 0, softmax_index, 0) &&
        is_connected(softmax_index, 0, slice_1_index, 0) &&
        is_connected(softmax_index, 0, slice_2_index, 0) &&
        is_connected(slice_1_index, 0, matmul_v1_index, 0) &&
        is_connected(slice_2_index, 0, matmul_v2_index, 0) &&
        is_connected(matmul_v1_index, 0, add_3_index, 0) &&
        is_connected(matmul_v2_index, 0, add_3_index, 1) &&
        is_connected(add_3_index, 0, reshape_4_index, 0) &&
        is_connected(reshape_4_index, 0, transpose_2_index, 0) &&
        is_connected(transpose_2_index, 0, reshape_5_index, 0) &&
        IsElementWiseMultiply(ops[start_index + mul_index]) &&
        IsElementWiseAdd(ops[start_index + add_2_index]) &&
        IsElementWiseAdd(ops[start_index + add_3_index]))) {
    return 1;
  }
  QNN_LOG_INFO("[G2G] MHA optimization (fast vlm Prefill)");

  // QKV Unpack
  const auto& matmul_v1_in =
      ops[start_index + matmul_v1_index].GetInputTensor(1);
  auto unpack_1_dims =
      ops[start_index + matmul_v1_index].GetInputTensor(1).GetDims();
  uint32_t num_kv_heads = unpack_1_dims[1];
  const auto& matmul_q_in = ops[start_index + matmul_q_index].GetInputTensor(1);
  auto unpack_2_dims = matmul_q_in.GetDims();
  const auto& mul_in = ops[start_index + mul_index].GetInputTensor(0);
  auto unpack_3_dims = mul_in.GetDims();
  uint32_t num_attn_heads = unpack_3_dims[1];
  uint32_t num_attn_per_kv_heads = num_attn_heads / num_kv_heads;
  const auto& add_1_in_1 = ops[start_index + add_1_index].GetInputTensor(0);
  auto unpack_4_dims = add_1_in_1.GetDims();
  const auto& add_1_in_2 = ops[start_index + add_1_index].GetInputTensor(1);
  auto unpack_5_dims = add_1_in_2.GetDims();
  const auto& matmul_v2_in =
      ops[start_index + matmul_v2_index].GetInputTensor(1);
  auto unpack_6_dims = matmul_v2_in.GetDims();
  const auto& pattern_output =
      ops[start_index + pattern_size - 1].GetOutputTensor(0);
  auto mha_output_dims = pattern_output.GetDims();
  if (!(num_kv_heads == unpack_2_dims[1] &&
        num_kv_heads == (unpack_3_dims[1] / 7) &&
        num_kv_heads == unpack_4_dims[1] && num_kv_heads == unpack_5_dims[1] &&
        num_kv_heads == unpack_6_dims[1] &&
        num_kv_heads == (mha_output_dims[1] / 64))) {
    QNN_LOG_WARNING(
        "[G2G] num_kv heads: %d not match heads in [unpack_2: %d, unpack_3: "
        "%d, unpack_4: %d, unpack_5: %d, unpack_6: %d, "
        "mha_output: %d]",
        num_kv_heads, unpack_2_dims[1], unpack_3_dims[1] / 7, unpack_4_dims[1],
        unpack_5_dims[1], unpack_6_dims[1], mha_output_dims[1] / 64);
    return 1;
  }
  QNN_LOG_INFO("[G2G] num_kv_heads match...");

  std::vector<OpWrapper> new_ops;
  std::vector<ConstTensorWrapperRef> unpack_1_sha_inputs;
  std::vector<ConstTensorWrapperRef> unpack_2_sha_inputs;
  std::vector<ConstTensorWrapperRef> unpack_3_sha_inputs;
  std::vector<ConstTensorWrapperRef> unpack_4_sha_inputs;
  std::vector<ConstTensorWrapperRef> unpack_5_sha_inputs;
  std::vector<ConstTensorWrapperRef> unpack_6_sha_inputs;

  unpack_1_dims.erase(unpack_1_dims.begin() + 1);
  unpack_2_dims.erase(unpack_2_dims.begin() + 1);
  unpack_3_dims.erase(unpack_3_dims.begin() + 1);
  unpack_4_dims.erase(unpack_4_dims.begin() + 1);
  unpack_5_dims.erase(unpack_5_dims.begin() + 1);
  unpack_6_dims.erase(unpack_6_dims.begin() + 1);

  unpack_1_sha_inputs.reserve(num_kv_heads);
  unpack_2_sha_inputs.reserve(num_kv_heads);
  unpack_3_sha_inputs.reserve(num_attn_heads);
  unpack_4_sha_inputs.reserve(num_kv_heads);
  unpack_5_sha_inputs.reserve(num_kv_heads);
  unpack_6_sha_inputs.reserve(num_kv_heads);

  for (int i = 0; i < num_kv_heads; i++) {
    const auto& unpack_1 =
        tensor_pool.CloneNativeTensorFrom(matmul_v1_in, unpack_1_dims);
    unpack_1_sha_inputs.emplace_back(unpack_1);
    const auto& unpack_2 =
        tensor_pool.CloneNativeTensorFrom(matmul_q_in, unpack_2_dims);
    unpack_2_sha_inputs.emplace_back(unpack_2);
    for (int j = 0; j < num_attn_per_kv_heads; j++) {
      const auto& unpack_3 =
          tensor_pool.CloneNativeTensorFrom(mul_in, unpack_3_dims);
      unpack_3_sha_inputs.emplace_back(unpack_3);
    }
    const auto& unpack_4 =
        tensor_pool.CloneNativeTensorFrom(add_1_in_1, unpack_4_dims);
    unpack_4_sha_inputs.emplace_back(unpack_4);
    const auto& unpack_5 =
        tensor_pool.CloneNativeTensorFrom(add_1_in_2, unpack_5_dims);
    unpack_5_sha_inputs.emplace_back(unpack_5);
    const auto& unpack_6 =
        tensor_pool.CloneNativeTensorFrom(matmul_v2_in, unpack_6_dims);
    unpack_6_sha_inputs.emplace_back(unpack_6);
  }

  // Unpack 1-5
  new_ops.emplace_back(CreateUnpackOp(matmul_v1_in, unpack_1_sha_inputs, 1));
  new_ops.emplace_back(CreateUnpackOp(matmul_q_in, unpack_2_sha_inputs, 1));
  new_ops.emplace_back(CreateUnpackOp(mul_in, unpack_3_sha_inputs, 1));
  new_ops.emplace_back(CreateUnpackOp(add_1_in_1, unpack_4_sha_inputs, 1));
  new_ops.emplace_back(CreateUnpackOp(add_1_in_2, unpack_5_sha_inputs, 1));
  new_ops.emplace_back(CreateUnpackOp(matmul_v2_in, unpack_6_sha_inputs, 1));

  // build num_head SHAs
  std::vector<ConstTensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_attn_heads);
  for (size_t i = 0; i < num_kv_heads; ++i) {
    for (size_t j = 0; j < num_attn_per_kv_heads; ++j) {
      const auto& sha_output = BuildSingleSHA(
          new_ops, tensor_pool, unpack_1_sha_inputs[i], unpack_2_sha_inputs[i],
          unpack_3_sha_inputs[i * num_attn_per_kv_heads + j],
          unpack_4_sha_inputs[i], unpack_5_sha_inputs[i],
          unpack_6_sha_inputs[i], num_attn_per_kv_heads,
          ops[start_index + mul_index], ops[start_index + matmul_q_index],
          ops[start_index + add_1_index], ops[start_index + matmul_qk_index],
          ops[start_index + concat_index], ops[start_index + add_2_index],
          ops[start_index + reshape_3_index], ops[start_index + softmax_index],
          ops[start_index + slice_1_index], ops[start_index + slice_2_index],
          ops[start_index + matmul_v1_index],
          ops[start_index + matmul_v2_index], ops[start_index + add_3_index]);
      sha_outputs.emplace_back(sha_output);
    }
  }

  // Concat
  new_ops.emplace_back(CreateConcatenationOp(sha_outputs, pattern_output, 2));

  // Validate new graph.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](OpWrapper& op_wrapper) -> bool {
                    return validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Adjust the name to avoid a name collision in the Qnn JSON dump.
    for (size_t i = 0; i < new_ops.size(); ++i) {
      new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
    }
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size();
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}
namespace {

bool OptimizeMHATinyGemmaPrefill(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    const OpWrapper& mul, const OpWrapper& transpoe_0,
    const OpWrapper& reshape_0, const OpWrapper& matmul_k0,
    const OpWrapper& matmul_k1, const OpWrapper& concat, const OpWrapper& add_0,
    const OpWrapper& softmax, const OpWrapper& slice_0,
    const OpWrapper& slice_1, const OpWrapper& matmul_v0,
    const OpWrapper& matmul_v1, const OpWrapper& add_1,
    const OpWrapper& reshape_1, const OpWrapper& transpose_1,
    const OpWrapper& reshape_2) {
  const auto is_connected =
      [](const OpWrapper& output, size_t output_tensor_index,
         const OpWrapper& input, size_t input_tensor_index) -> bool {
    return output.GetOutputTensor(output_tensor_index) ==
           input.GetInputTensor(input_tensor_index);
  };
  if (!(is_connected(mul, 0, transpoe_0, 0) &&
        is_connected(transpoe_0, 0, reshape_0, 0) &&
        is_connected(reshape_0, 0, matmul_k0, 0) &&
        is_connected(reshape_0, 0, matmul_k1, 0) &&
        is_connected(matmul_k0, 0, concat, 0) &&
        is_connected(matmul_k1, 0, concat, 1) &&
        is_connected(concat, 0, add_0, 0) &&
        is_connected(add_0, 0, softmax, 0) &&
        is_connected(softmax, 0, slice_0, 0) &&
        is_connected(softmax, 0, slice_1, 0) &&
        is_connected(slice_0, 0, matmul_v0, 0) &&
        is_connected(slice_1, 0, matmul_v1, 0) &&
        is_connected(matmul_v0, 0, add_1, 0) &&
        is_connected(matmul_v1, 0, add_1, 1) &&
        is_connected(add_1, 0, reshape_1, 0) &&
        is_connected(reshape_1, 0, transpose_1, 0) &&
        is_connected(transpose_1, 0, reshape_2, 0) &&
        IsElementWiseMultiply(mul) && IsElementWiseAdd(add_0) &&
        IsElementWiseAdd(add_1))) {
    return false;
  }

  QNN_LOG_INFO("[G2G] MHA optimization (TinyGemma Prefill)");
  const auto& pattern_input = mul.GetInputTensor(0);
  const auto& pattern_output = reshape_2.GetOutputTensor(0);

  // Transpose
  auto transpose_output_dims = transpoe_0.GetOutputTensor(0).GetDims();
  const auto& transpose_output =
      tensor_pool.CloneNativeTensorFrom(pattern_input, transpose_output_dims);
  auto& new_transpose_0 = new_ops.emplace_back(CreateTransposeOpWithSameParam(
      transpoe_0, pattern_input, transpose_output));

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDim(2);
  const auto& mha_input = new_transpose_0.GetOutputTensor(0);

  // Prepare inputs for num_heads SHAs.
  std::vector<ConstTensorWrapperRef> sha_inputs;
  sha_inputs.reserve(num_heads);
  auto sha_input_dims = mha_input.GetDims();
  sha_input_dims[1] /= num_heads;
  for (size_t i = 0; i < num_heads; ++i) {
    const auto& sha_input =
        tensor_pool.CloneNativeTensorFrom(mha_input, sha_input_dims);
    sha_inputs.emplace_back(sha_input);
  }

  // Split
  std::vector<std::uint32_t> split_indice;
  split_indice.reserve(num_heads);
  for (std::uint32_t i = 1; i < num_heads; i++) {
    split_indice.emplace_back(i * mha_input.GetDim(1) / num_heads);
  }
  const auto& split_indice_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {},
      {static_cast<std::uint32_t>(split_indice.size())},
      sizeof(std::uint32_t) * split_indice.size(), split_indice.data());
  auto& split = new_ops.emplace_back(
      CreateSplitOp(mha_input, sha_inputs, 1, split_indice_tensor));
  CloneNamespace(mul, split);

  // Split Mask for Add
  std::vector<ConstTensorWrapperRef> splited_masks;
  splited_masks.reserve(num_heads);
  const auto& concated_mask = add_0.GetInputTensor(1);
  auto splited_mask_dims = concated_mask.GetDims();
  splited_mask_dims[2] /= num_heads;
  for (size_t i = 0; i < num_heads; ++i) {
    splited_masks.emplace_back(
        tensor_pool.CloneNativeTensorFrom(concated_mask, splited_mask_dims));
  }
  std::vector<std::uint32_t> split_mask_indice;
  split_mask_indice.reserve(num_heads);
  for (std::uint32_t i = 1; i < num_heads; i++) {
    split_mask_indice.emplace_back(i * concated_mask.GetDim(2) / num_heads);
  }
  const auto& split_mask_indice_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {},
      {static_cast<std::uint32_t>(split_mask_indice.size())},
      sizeof(std::uint32_t) * split_mask_indice.size(),
      split_mask_indice.data());
  auto& split_mask = new_ops.emplace_back(
      CreateSplitOp(concated_mask, splited_masks, 2, split_mask_indice_tensor));

  // build num_head SHAs
  std::vector<ConstTensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_heads);
  for (size_t i = 0; i < num_heads; ++i) {
    const auto& sha_output =
        BuildSingleSHA(new_ops, tensor_pool, sha_inputs[i], splited_masks[i],
                       num_heads, mul, matmul_k0, matmul_k1, concat, add_0,
                       softmax, slice_0, slice_1, matmul_v0, matmul_v1, add_1);
    sha_outputs.emplace_back(sha_output);
  }

  // Concat
  auto concat_output_dims = sha_outputs[0].get().GetDims();
  concat_output_dims[3] *= num_heads;
  const auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(sha_outputs[0], concat_output_dims);
  auto& concat_sha_output = new_ops.emplace_back(
      CreateConcatenationOp(sha_outputs, concat_output, 3));
  CloneNamespace(mul, concat_sha_output);

  // Reshape
  auto& new_reshape =
      new_ops.emplace_back(CreateReshapeOp(concat_output, pattern_output));
  CloneNamespace(mul, new_reshape);
  return true;
}

}  // namespace

size_t OptimizeMHATinyGemmaPrefillPatternWithGlobalMask(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr size_t kMulIndex = 0;
  constexpr size_t kTranspose0Index = 1;
  constexpr size_t kReshape0Index = 2;
  constexpr size_t kMatmulK0Index = 3;
  constexpr size_t kMatmulK1Index = 4;
  constexpr size_t kConcatIndex = 5;
  constexpr size_t kConcatMaskIndex = 6;
  constexpr size_t kReshapeMaskIndex = 7;
  constexpr size_t kAdd0Index = 8;
  constexpr size_t kSoftmaxIndex = 9;
  constexpr size_t kSlice0Index = 10;
  constexpr size_t kSlice1Index = 11;
  constexpr size_t kMatmulV0Index = 12;
  constexpr size_t kMatmulV1Index = 13;
  constexpr size_t kAdd1Index = 14;
  constexpr size_t kReshape1Index = 15;
  constexpr size_t kTranspose1Index = 16;
  constexpr size_t kReshape2Index = 17;

  const auto& mul = ops[start_index + kMulIndex];
  const auto& transpose_0 = ops[start_index + kTranspose0Index];
  const auto& reshape_0 = ops[start_index + kReshape0Index];
  const auto& matmul_k0 = ops[start_index + kMatmulK0Index];
  const auto& matmul_k1 = ops[start_index + kMatmulK1Index];
  const auto& concat = ops[start_index + kConcatIndex];
  const auto& add_0 = ops[start_index + kAdd0Index];
  const auto& softmax = ops[start_index + kSoftmaxIndex];
  const auto& slice_0 = ops[start_index + kSlice0Index];
  const auto& slice_1 = ops[start_index + kSlice1Index];
  const auto& matmul_v0 = ops[start_index + kMatmulV0Index];
  const auto& matmul_v1 = ops[start_index + kMatmulV1Index];
  const auto& add_1 = ops[start_index + kAdd1Index];
  const auto& reshape_1 = ops[start_index + kReshape1Index];
  const auto& transpose_1 = ops[start_index + kTranspose1Index];
  const auto& reshape_2 = ops[start_index + kReshape2Index];

  const auto& mask_concat = ops[start_index + kConcatMaskIndex];
  const auto& mask_reshape = ops[start_index + kReshapeMaskIndex];
  if (mask_concat.GetOutputTensor(0) != mask_reshape.GetInputTensor(0) ||
      mask_reshape.GetOutputTensor(0) != add_0.GetInputTensor(1)) {
    return 1;
  }

  std::vector<OpWrapper> new_ops;
  if (!OptimizeMHATinyGemmaPrefill(
          new_ops, tensor_pool, mul, transpose_0, reshape_0, matmul_k0,
          matmul_k1, concat, add_0, softmax, slice_0, slice_1, matmul_v0,
          matmul_v1, add_1, reshape_1, transpose_1, reshape_2)) {
    return 1;
  }

  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](OpWrapper& op_wrapper) -> bool {
                    return op_wrapper.IsOpCode(QnnOpCode::kSplit) ||
                           validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Adjust the name to avoid a name collision in the Qnn JSON dump.
    for (size_t i = 0; i < new_ops.size(); ++i) {
      new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
    }
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size() + 2;
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    // Only keep mask_concat and mask_reshape
    ops.erase(ops.begin() + start_index + kAdd0Index,
              ops.begin() + start_index + pattern_size);
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + kConcatMaskIndex);
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed in "
      "OptimizeMHATinyGemmaPrefillPatternWithGlobalMask. Rolling back to the "
      "original graph.");
  return 1;
}

size_t OptimizeMHATinyGemmaPrefillPattern(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr size_t kMulIndex = 0;
  constexpr size_t kTranspose0Index = 1;
  constexpr size_t kReshape0Index = 2;
  constexpr size_t kMatmulK0Index = 3;
  constexpr size_t kMatmulK1Index = 4;
  constexpr size_t kConcatIndex = 5;
  constexpr size_t kAdd0Index = 6;
  constexpr size_t kSoftmaxIndex = 7;
  constexpr size_t kSlice0Index = 8;
  constexpr size_t kSlice1Index = 9;
  constexpr size_t kMatmulV0Index = 10;
  constexpr size_t kMatmulV1Index = 11;
  constexpr size_t kAdd1Index = 12;
  constexpr size_t kReshape1Index = 13;
  constexpr size_t kTranspose1Index = 14;
  constexpr size_t kReshape2Index = 15;
  const auto& mul = ops[start_index + kMulIndex];
  const auto& transpose_0 = ops[start_index + kTranspose0Index];
  const auto& reshape_0 = ops[start_index + kReshape0Index];
  const auto& matmul_k0 = ops[start_index + kMatmulK0Index];
  const auto& matmul_k1 = ops[start_index + kMatmulK1Index];
  const auto& concat = ops[start_index + kConcatIndex];
  const auto& add_0 = ops[start_index + kAdd0Index];
  const auto& softmax = ops[start_index + kSoftmaxIndex];
  const auto& slice_0 = ops[start_index + kSlice0Index];
  const auto& slice_1 = ops[start_index + kSlice1Index];
  const auto& matmul_v0 = ops[start_index + kMatmulV0Index];
  const auto& matmul_v1 = ops[start_index + kMatmulV1Index];
  const auto& add_1 = ops[start_index + kAdd1Index];
  const auto& reshape_1 = ops[start_index + kReshape1Index];
  const auto& transpose_1 = ops[start_index + kTranspose1Index];
  const auto& reshape_2 = ops[start_index + kReshape2Index];

  std::vector<OpWrapper> new_ops;
  if (!OptimizeMHATinyGemmaPrefill(
          new_ops, tensor_pool, mul, transpose_0, reshape_0, matmul_k0,
          matmul_k1, concat, add_0, softmax, slice_0, slice_1, matmul_v0,
          matmul_v1, add_1, reshape_1, transpose_1, reshape_2)) {
    return 1;
  }

  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](OpWrapper& op_wrapper) -> bool {
                    return op_wrapper.IsOpCode(QnnOpCode::kSplit) ||
                           validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Adjust the name to avoid a name collision in the Qnn JSON dump.
    for (size_t i = 0; i < new_ops.size(); ++i) {
      new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
    }
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size();
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed in OptimizeMHATinyGemmaPrefillPattern. Rolling "
      "back to the original graph.");
  return 1;
}

size_t OptimizeMHAAttn(std::function<bool(OpWrapper&)> validate_op_config,
                       std::vector<OpWrapper>& ops, size_t attn_start_index,
                       TensorPool& tensor_pool, size_t pattern_size) {
  // attn (attention mask)
  constexpr size_t kAttnSelect = 7;
  constexpr size_t kAttnNotEqual = 6;
  constexpr size_t kAttnReshape = 5;
  // attn (QK)
  constexpr int32_t kAttnMulQ = -4;
  constexpr int32_t kAttnMulK = -3;
  constexpr int32_t kAttnTransposeQ = -2;
  constexpr int32_t kAttnTransposeK = -1;
  // attn (Softmax & V)
  constexpr int32_t kAttnSoftmax = 1;
  constexpr int32_t kAttnTransposeIn = 2;
  constexpr int32_t kAttnMatMul = 3;
  constexpr int32_t kAttnTransposeOut = 4;

  // Connection check: Reshape -> NotEqual -> Select
  size_t start_index = attn_start_index;
  if (!(IS_CONNECTED(kAttnReshape, 0, kAttnNotEqual, 0)) &&
      (IS_CONNECTED(kAttnNotEqual, 0, kAttnSelect, 0))) {
    return 1;
  }
  // attn_not_equal_op is copied from ops since ops will be modified.
  const auto& attn_not_equal_op = ops[attn_start_index + kAttnNotEqual];
  const auto& not_equal_out = attn_not_equal_op.GetOutputTensor(0);
  // Count the operations that have NotEqual as their source op.
  const auto& reshape_in =
      ops[attn_start_index + kAttnReshape].GetInputTensor(0);
  size_t num_out =
      std::count_if(ops.begin(), ops.end(), [&](const OpWrapper& op) {
        return op.IsOpCode(QnnOpCode::kElementWiseSelect) &&
               op.GetInputTensor(0) == not_equal_out;
      });
  if (num_out == 0) {
    return 1;
  }

  QNN_LOG_INFO("[G2G] MHA optimization (Attn)");
  // Handle masking.
  std::vector<OpWrapper> new_ops;
  auto not_equal_out_dims = not_equal_out.GetDims();
  not_equal_out_dims.erase(not_equal_out_dims.begin() + 1);
  const auto& select_mask =
      tensor_pool.CloneNativeTensorFrom(not_equal_out, not_equal_out_dims);
  // Change NotEqual to Equal -> Cast -> Mul.
  const auto& zero_tensor = attn_not_equal_op.GetInputTensor(1);
  new_ops.emplace_back(
      CreateElementWiseEqualOp(reshape_in, zero_tensor, select_mask));
  const auto& select_out =
      ops[attn_start_index + kAttnSelect].GetOutputTensor(0);
  auto select_out_dims = select_out.GetDims();
  select_out_dims.erase(select_out_dims.begin() + 1);

  const auto& mul_in =
      tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
  new_ops.emplace_back(CreateCastOp(select_mask, mul_in));

  const auto& select_const =
      ops[attn_start_index + kAttnSelect].GetInputTensor(2);
  // TODO(jiunkaiy): Remove this magic number (-65472) after HTP resolves
  // accuracy issues.
  float mul_const_value =
      std::max(select_const.GetTensorData<float>().value()[0], -65472.f);
  const auto& mul_const = tensor_pool.CreateStaticTensor(
      select_const.GetDataType(), select_const.GetQuantParams(),
      select_const.GetDims(), select_const.GetTensorBytes(), &mul_const_value);
  const auto& add_in =
      tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
  new_ops.emplace_back(CreateElementWiseMulOp(mul_in, mul_const, add_in));

  // Create SHAs based on Select index.
  size_t select_index = 0;
  for (size_t output_index = 0; output_index < num_out; ++output_index) {
    // Identify Select index.
    auto it_select = std::find_if(
        ops.begin() + select_index + 1, ops.end(), [&](const OpWrapper& op) {
          return op.IsOpCode(QnnOpCode::kElementWiseSelect) &&
                 op.GetInputTensor(0) == not_equal_out;
        });
    if (it_select == ops.end()) {
      QNN_LOG_ERROR("Could not find Select op with the given input tensor");
      break;
    }
    select_index = std::distance(ops.begin(), it_select);

    // Connection check based on Select index.
    start_index = select_index;
    if (!(IS_CONNECTED(0, 0, kAttnSoftmax, 0) &&
          IS_CONNECTED(kAttnSoftmax, 0, kAttnMatMul, 1) &&
          IS_CONNECTED(kAttnTransposeIn, 0, kAttnMatMul, 0) &&
          IS_CONNECTED(kAttnMatMul, 0, kAttnTransposeOut, 0))) {
      QNN_LOG_ERROR("[G2G] Connection check failed.");
      return 1;
    }
    // Identify MatMul's index.
    auto it_matmul =
        std::find_if(ops.begin(), ops.end(), [&](const OpWrapper& op) {
          return op.IsOpCode(QnnOpCode::kMatMul) &&
                 op.GetOutputTensor(0) == ops[select_index].GetInputTensor(1);
        });
    if (it_matmul == ops.end()) {
      QNN_LOG_ERROR("Could not find MatMul op with the given output tensor");
      break;
    }
    size_t matmul_qk_index = std::distance(ops.begin(), it_matmul);

    // Connection check based on Matmul index.
    start_index = matmul_qk_index;
    if (!(IS_CONNECTED(kAttnMulQ, 0, kAttnTransposeQ, 0) &&
          IS_CONNECTED(kAttnMulK, 0, kAttnTransposeK, 0) &&
          IS_CONNECTED(kAttnTransposeQ, 0, 0, 0) &&
          IS_CONNECTED(kAttnTransposeK, 0, 0, 1) &&
          IsElementWiseMultiply(ops[start_index + kAttnMulQ]) &&
          IsElementWiseMultiply(ops[start_index + kAttnMulK]))) {
      QNN_LOG_ERROR("[G2G] Connection check failed.");
      return 1;
    }
    // QKV Unpack
    const auto& mul_q_in = ops[matmul_qk_index + kAttnMulQ].GetInputTensor(0);
    auto q_unpack_dims = mul_q_in.GetDims();
    uint32_t num_heads = q_unpack_dims[2];
    const auto& mul_k_in = ops[matmul_qk_index + kAttnMulK].GetInputTensor(0);
    auto k_unpack_dims = mul_k_in.GetDims();
    const auto& transpose_v_in =
        ops[select_index + kAttnTransposeIn].GetInputTensor(0);
    auto transpose_v_perm =
        ops[select_index + kAttnTransposeIn].GetTensorPararm(0).GetTensor();
    std::vector<uint32_t> perm_data = {0, 2, 1};
    const auto& perm_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, transpose_v_perm.GetQuantParams(), {3},
        perm_data.size() * sizeof(perm_data[0]), perm_data.data());
    auto v_unpack_dims = transpose_v_in.GetDims();
    const auto& mha_out =
        ops[select_index + kAttnTransposeOut].GetOutputTensor(0);
    auto mha_out_dims = mha_out.GetDims();
    if (!(num_heads == k_unpack_dims[2] && num_heads == v_unpack_dims[2] &&
          num_heads == mha_out_dims[2])) {
      QNN_LOG_ERROR("[G2G] Num heads mismatches.");
      return 1;
    }
    q_unpack_dims.erase(q_unpack_dims.begin() + 2);
    k_unpack_dims.erase(k_unpack_dims.begin() + 2);
    v_unpack_dims.erase(v_unpack_dims.begin() + 2);
    mha_out_dims.erase(mha_out_dims.begin() + 2);
    // Prepare inputs and outputs for num_heads SHAs.
    std::vector<ConstTensorWrapperRef> q_sha_inputs;
    std::vector<ConstTensorWrapperRef> k_sha_inputs;
    std::vector<ConstTensorWrapperRef> v_sha_inputs;
    std::vector<ConstTensorWrapperRef> sha_outputs;
    q_sha_inputs.reserve(num_heads);
    k_sha_inputs.reserve(num_heads);
    v_sha_inputs.reserve(num_heads);
    sha_outputs.reserve(num_heads);

    for (int i = 0; i < num_heads; ++i) {
      const auto& q_unpack =
          tensor_pool.CloneNativeTensorFrom(mul_q_in, q_unpack_dims);
      q_sha_inputs.emplace_back(q_unpack);

      const auto& k_unpack =
          tensor_pool.CloneNativeTensorFrom(mul_k_in, k_unpack_dims);
      k_sha_inputs.emplace_back(k_unpack);

      const auto& v_unpack =
          tensor_pool.CloneNativeTensorFrom(transpose_v_in, v_unpack_dims);
      v_sha_inputs.emplace_back(v_unpack);

      const auto& sha_out =
          tensor_pool.CloneNativeTensorFrom(mha_out, mha_out_dims);
      sha_outputs.emplace_back(sha_out);
    }
    new_ops.emplace_back(CreateUnpackOp(mul_q_in, q_sha_inputs, 2));
    new_ops.emplace_back(CreateUnpackOp(mul_k_in, k_sha_inputs, 2));
    new_ops.emplace_back(CreateUnpackOp(transpose_v_in, v_sha_inputs, 2));

    for (int i = 0; i < num_heads; ++i) {
      const auto& q_matmul_in =
          tensor_pool.CloneNativeTensorFrom(q_sha_inputs[i]);
      new_ops.emplace_back(CreateElementWiseMulOp(
          q_sha_inputs[i], ops[matmul_qk_index + kAttnMulQ].GetInputTensor(1),
          q_matmul_in));

      const auto& k_transpose_in =
          tensor_pool.CloneNativeTensorFrom(k_sha_inputs[i]);
      new_ops.emplace_back(CreateElementWiseMulOp(
          k_sha_inputs[i], ops[matmul_qk_index + kAttnMulK].GetInputTensor(1),
          k_transpose_in));

      const auto& k_matmul_in = tensor_pool.CloneNativeTensorFrom(
          k_transpose_in,
          {k_unpack_dims[0], k_unpack_dims[2], k_unpack_dims[1]});
      new_ops.emplace_back(
          CreateTransposeOp(k_transpose_in, k_matmul_in, perm_tensor));
      // MatMul
      const auto& matmul_qk_out = ops[matmul_qk_index].GetOutputTensor(0);
      const auto& select_in = tensor_pool.CloneNativeTensorFrom(
          matmul_qk_out, {q_matmul_in.GetDim(0), q_matmul_in.GetDim(1),
                          k_matmul_in.GetDim(2)});
      new_ops.emplace_back(CreateMatmulOpWithSameParam(
          ops[matmul_qk_index], q_matmul_in, k_matmul_in, select_in));

      // Change Select to Add.
      const auto& softmax_in =
          tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
      new_ops.emplace_back(
          CreateElementWiseAddOp(select_in, add_in, softmax_in));

      // Softmax
      const auto& qk_softmax =
          tensor_pool.CloneNativeTensorFrom(softmax_in, select_out_dims);
      new_ops.emplace_back(CreateSoftmaxOpWithSameParam(
          ops[select_index + kAttnSoftmax], softmax_in, qk_softmax));

      // MatMul
      new_ops.emplace_back(CreateMatmulOpWithSameParam(
          ops[matmul_qk_index], qk_softmax, v_sha_inputs[i], sha_outputs[i]));
    }
    // Pack
    new_ops.emplace_back(CreatePackOp(sha_outputs, mha_out, 2));

    const bool is_valid =
        std::all_of(new_ops.begin(), new_ops.end(),
                    [validate_op_config](OpWrapper& op_wrapper) -> bool {
                      return validate_op_config(op_wrapper);
                    });
    if (is_valid) {
      // Adjust the name to avoid a name collision in the Qnn JSON dump.
      for (size_t i = 0; i < new_ops.size(); ++i) {
        new_ops[i].AddSuffixToName(absl::StrCat("_qcg2g_", i));
      }
      // Replace the matched pattern with a newly generated subgraph.
      ops.insert(ops.begin() + select_index + kAttnTransposeOut + 1,
                 std::make_move_iterator(new_ops.begin()),
                 std::make_move_iterator(new_ops.end()));
      // Erase original pattern backwards.
      ops.erase(ops.begin() + select_index,
                ops.begin() + select_index + kAttnTransposeOut + 1);
      if (output_index == 0) {
        ops.erase(ops.begin() + attn_start_index + kAttnNotEqual);
        ops.erase(ops.begin() + attn_start_index + kAttnReshape);
      }
      ops.erase(ops.begin() + matmul_qk_index + kAttnMulQ,
                ops.begin() + matmul_qk_index + 1);
    } else {
      QNN_LOG_ERROR(
          "[G2G] Validation failed. Rolling back to the original graph.");
      return 1;
    }
    new_ops.clear();
  }
  return 1;
}

}  // namespace qnn
