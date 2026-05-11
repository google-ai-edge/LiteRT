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
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
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
      mul.GetOutputTensor(0), sha_input.GetDimensions());
  new_ops.emplace_back(
      CreateElementWiseMulOp(sha_input, mul.GetInputTensor(1), mul_output));

  // MatMul 1
  auto matmul_k1_output_dims = matmul_k1.GetOutputTensor(0).GetDimensions();
  matmul_k1_output_dims[2] /= num_heads;
  const auto& matmul_k1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k1.GetOutputTensor(0), matmul_k1_output_dims);
  const std::array<ConstTensorWrapperRef, 2> matmul_k1_inputs = {
      mul_output, matmul_k1.GetInputTensor(1)};
  const std::array<ConstTensorWrapperRef, 1> matmul_k1_outputs = {
      matmul_k1_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(matmul_k1, matmul_k1_inputs, matmul_k1_outputs));

  // MatMul 2
  auto matmul_k2_output_dims = matmul_k2.GetOutputTensor(0).GetDimensions();
  matmul_k2_output_dims[2] /= num_heads;
  const auto& matmul_k2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k2.GetOutputTensor(0), matmul_k2_output_dims);
  const std::array<ConstTensorWrapperRef, 2> matmul_k2_inputs = {
      mul_output, matmul_k2.GetInputTensor(1)};
  const std::array<ConstTensorWrapperRef, 1> matmul_k2_outputs = {
      matmul_k2_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(matmul_k2, matmul_k2_inputs, matmul_k2_outputs));

  // Concat
  auto concat_output_dims = matmul_k1_output_dims;
  concat_output_dims[3] += matmul_k2_output_dims[3];
  const auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      concat.GetOutputTensor(0), concat_output_dims);
  const std::array<ConstTensorWrapperRef, 2> concat_inputs = {matmul_k1_output,
                                                              matmul_k2_output};
  const std::array<ConstTensorWrapperRef, 1> concat_outputs = {concat_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(concat, concat_inputs, concat_outputs));

  // Add
  const auto& add_1_output = tensor_pool.CloneNativeTensorFrom(
      add_1.GetOutputTensor(0), concat_output.GetDimensions());
  new_ops.emplace_back(
      CreateElementWiseAddOp(concat_output, mask, add_1_output));
  // Softmax
  const auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), add_1_output.GetDimensions());
  const std::array<ConstTensorWrapperRef, 1> softmax_inputs = {add_1_output};
  const std::array<ConstTensorWrapperRef, 1> softmax_outputs = {softmax_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(softmax, softmax_inputs, softmax_outputs));

  // Slice 1
  auto slice_1_ranges = slice_1.GetTensorParam(0).GetTensor();
  auto slice_1_rangs_data = slice_1_ranges.GetTensorData<int32_t>();
  std::vector<int32_t> sha_slice_1_ranges_data(
      slice_1_rangs_data.value().begin(), slice_1_rangs_data.value().end());
  sha_slice_1_ranges_data[kSlice3rdAxisEndIndex] /= num_heads;
  const auto& sha_slice_1_ranges = tensor_pool.CreateStaticTensor(
      slice_1_ranges.GetDataType(), slice_1_ranges.GetQuantParams(),
      slice_1_ranges.GetDimensions(), slice_1_ranges.GetTensorBytes(),
      sha_slice_1_ranges_data.data());
  auto slice_1_output_dims = slice_1.GetOutputTensor(0).GetDimensions();
  slice_1_output_dims[2] /= num_heads;
  const auto& slice_1_output = tensor_pool.CloneNativeTensorFrom(
      slice_1.GetOutputTensor(0), slice_1_output_dims);
  new_ops.emplace_back(
      CreateSliceOp(softmax_output, slice_1_output, sha_slice_1_ranges));

  // Slice 2
  auto slice_2_ranges = slice_2.GetTensorParam(0).GetTensor();
  auto slice_2_ranges_data = slice_2_ranges.GetTensorData<int32_t>();
  std::vector<int32_t> sha_slice_2_ranges_data(
      slice_2_ranges_data.value().begin(), slice_2_ranges_data.value().end());
  sha_slice_2_ranges_data[kSlice3rdAxisEndIndex] /= num_heads;
  const auto& sha_slice_2_ranges = tensor_pool.CreateStaticTensor(
      slice_2_ranges.GetDataType(), slice_2_ranges.GetQuantParams(),
      slice_2_ranges.GetDimensions(), slice_2_ranges.GetTensorBytes(),
      sha_slice_2_ranges_data.data());
  auto slice_2_output_dims = slice_2.GetOutputTensor(0).GetDimensions();
  slice_2_output_dims[2] /= num_heads;
  const auto& slice_2_output = tensor_pool.CloneNativeTensorFrom(
      slice_2.GetOutputTensor(0), slice_2_output_dims);
  new_ops.emplace_back(
      CreateSliceOp(softmax_output, slice_2_output, sha_slice_2_ranges));

  // MatMul 1
  std::vector<uint32_t> matmul_v1_output_dims =
      matmul_v1.GetOutputTensor(0).GetDimensions();
  matmul_v1_output_dims[2] = matmul_v1_output_dims[2] / num_heads;
  const auto& matmul_v1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v1.GetOutputTensor(0), matmul_v1_output_dims);
  const std::array<ConstTensorWrapperRef, 2> matmul_v1_inputs = {
      slice_1_output, matmul_v1.GetInputTensor(1)};
  const std::array<ConstTensorWrapperRef, 1> matmul_v1_outputs = {
      matmul_v1_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(matmul_v1, matmul_v1_inputs, matmul_v1_outputs));

  // MatMul 2
  std::vector<uint32_t> matmul_v2_output_dims =
      matmul_v2.GetOutputTensor(0).GetDimensions();
  matmul_v2_output_dims[2] = matmul_v2_output_dims[2] / num_heads;
  const auto& matmul_v2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v2.GetOutputTensor(0), matmul_v2_output_dims);
  const std::array<ConstTensorWrapperRef, 2> matmul_v2_inputs = {
      slice_2_output, matmul_v2.GetInputTensor(1)};
  const std::array<ConstTensorWrapperRef, 1> matmul_v2_outputs = {
      matmul_v2_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(matmul_v2, matmul_v2_inputs, matmul_v2_outputs));

  // Add 2
  const auto& add_2_output = tensor_pool.CloneNativeTensorFrom(
      add_2.GetOutputTensor(0), matmul_v1_output.GetDimensions());
  new_ops.emplace_back(
      CreateElementWiseAddOp(matmul_v1_output, matmul_v2_output, add_2_output));
  return add_2_output;
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
    auto head_input_dims = split_input.GetDimensions();
    head_input_dims[2] /= num_heads;
    const auto& split_output =
        tensor_pool.CloneNativeTensorFrom(mha_input, head_input_dims);
    sha_inputs.emplace_back(split_output);
  }
  // Split
  std::vector<std::uint32_t> split_indice;
  split_indice.reserve(num_heads);
  for (std::uint32_t i = 1; i < num_heads; i++) {
    split_indice.emplace_back(i * split_input.GetDimension(2) / num_heads);
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
  auto concat_dims = mha_output.GetDimensions();
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
std::vector<ConstTensorWrapperRef> UnpackTensor(TensorPool& tensor_pool,
                                                std::vector<OpWrapper>& new_ops,
                                                const TensorWrapper& input,
                                                size_t unpack_dims) {
  auto input_dims = input.GetDimensions();
  auto num_unpack = input_dims[unpack_dims];
  input_dims.erase(input_dims.begin() + unpack_dims);
  std::vector<ConstTensorWrapperRef> outputs;
  outputs.reserve(num_unpack);
  for (size_t i = 0; i < num_unpack; ++i) {
    outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(input, input_dims));
  }
  new_ops.emplace_back(
      CreateUnpackOp(input, outputs, static_cast<std::uint32_t>(unpack_dims)));
  return outputs;
}

TensorWrapper& BuildSingleSHAByUnpackAxis1(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    const uint32_t num_attn_per_kv_heads, const TensorWrapper& scale_mul_input,
    const TensorWrapper& k_cache, const TensorWrapper& k_slice,
    const TensorWrapper& v_cache, const TensorWrapper& v_slice,
    const OpWrapper& scale_mul, const OpWrapper& q_kcache_matmul,
    const OpWrapper& q_kslice_matmul, const OpWrapper& qk_concat,
    const OpWrapper& mask_add, const OpWrapper& post_mask_reshape,
    const OpWrapper& softmax, const OpWrapper& qk_vcache_slice,
    const OpWrapper& qk_vslice_slice, const OpWrapper& qk_vcache_matmul,
    const OpWrapper& qk_vslice_matmul, const OpWrapper& qkv_add) {
  // Scale Mul
  auto mul_output_dims = scale_mul.GetOutputTensor(0).GetDimensions();
  mul_output_dims.erase(mul_output_dims.begin() + 1);
  auto& mul_output = tensor_pool.CloneNativeTensorFrom(
      scale_mul.GetOutputTensor(0), mul_output_dims);
  new_ops.emplace_back(CreateElementWiseMulOp(
      scale_mul_input, scale_mul.GetInputTensor(1), mul_output));

  // Q KCache Matmul
  auto q_kcache_matmul_output_dims =
      q_kcache_matmul.GetOutputTensor(0).GetDimensions();
  q_kcache_matmul_output_dims.erase(q_kcache_matmul_output_dims.begin() + 1);
  q_kcache_matmul_output_dims[1] /= num_attn_per_kv_heads;
  auto& q_kcache_matmul_output = tensor_pool.CloneNativeTensorFrom(
      q_kcache_matmul.GetOutputTensor(0), q_kcache_matmul_output_dims);
  const std::array<ConstTensorWrapperRef, 2> q_kcache_matmul_inputs = {
      mul_output, k_cache};
  const std::array<ConstTensorWrapperRef, 1> q_kcache_matmul_outputs = {
      q_kcache_matmul_output};
  new_ops.emplace_back(CreateOpWithSameParams(
      q_kcache_matmul, q_kcache_matmul_inputs, q_kcache_matmul_outputs));

  // Q KSlice Matmul
  auto q_kslice_matmul_output_dims =
      q_kslice_matmul.GetOutputTensor(0).GetDimensions();
  q_kslice_matmul_output_dims.erase(q_kslice_matmul_output_dims.begin() + 1);
  q_kslice_matmul_output_dims[1] /= num_attn_per_kv_heads;
  auto& q_kslice_matmul_output = tensor_pool.CloneNativeTensorFrom(
      q_kslice_matmul.GetOutputTensor(0), q_kslice_matmul_output_dims);
  const std::array<ConstTensorWrapperRef, 2> q_kslice_matmul_inputs = {
      mul_output, k_slice};
  const std::array<ConstTensorWrapperRef, 1> q_kslice_matmul_outputs = {
      q_kslice_matmul_output};
  new_ops.emplace_back(CreateOpWithSameParams(
      q_kslice_matmul, q_kslice_matmul_inputs, q_kslice_matmul_outputs));

  // QK Concat
  std::uint32_t adjusted_axis = 2;
  auto concat_output_dims = qk_concat.GetOutputTensor(0).GetDimensions();
  concat_output_dims.erase(concat_output_dims.begin() + 1);
  concat_output_dims[1] /= num_attn_per_kv_heads;
  auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      qk_concat.GetOutputTensor(0), concat_output_dims);
  new_ops.emplace_back(
      CreateConcatenationOp({q_kcache_matmul_output, q_kslice_matmul_output},
                            concat_output, adjusted_axis));

  // Mask Add
  auto mask_add_output_dims = mask_add.GetOutputTensor(0).GetDimensions();
  mask_add_output_dims[0] = 1;
  mask_add_output_dims[1] = 1;
  auto& mask_add_output = tensor_pool.CloneNativeTensorFrom(
      mask_add.GetOutputTensor(0), mask_add_output_dims);
  new_ops.emplace_back(CreateElementWiseAddOp(
      concat_output, mask_add.GetInputTensor(1), mask_add_output));

  // Post Mask Reshape
  auto post_mask_reshape_output_dims =
      post_mask_reshape.GetOutputTensor(0).GetDimensions();
  post_mask_reshape_output_dims.erase(post_mask_reshape_output_dims.begin() +
                                      1);
  post_mask_reshape_output_dims[1] /= num_attn_per_kv_heads;
  auto& post_mask_reshape_output = tensor_pool.CloneNativeTensorFrom(
      post_mask_reshape.GetOutputTensor(0), post_mask_reshape_output_dims);
  new_ops.emplace_back(
      CreateReshapeOp(mask_add_output, post_mask_reshape_output));

  // Softmax
  auto softmax_output_dims = softmax.GetOutputTensor(0).GetDimensions();
  softmax_output_dims.erase(softmax_output_dims.begin() + 1);
  softmax_output_dims[1] /= num_attn_per_kv_heads;
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), softmax_output_dims);
  const std::array<ConstTensorWrapperRef, 1> softmax_inputs = {
      post_mask_reshape_output};
  const std::array<ConstTensorWrapperRef, 1> softmax_outputs = {softmax_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(softmax, softmax_inputs, softmax_outputs));

  // QK VCache Slice
  auto qk_vcache_slice_param = qk_vcache_slice.GetTensorParam(0).GetTensor();
  auto qk_vcache_slice_param_data =
      qk_vcache_slice_param.GetTensorData<int32_t>();
  std::vector<int32_t> qk_vcache_slice_ranges(
      qk_vcache_slice_param_data.value().begin(),
      qk_vcache_slice_param_data.value().end());
  qk_vcache_slice_ranges.erase(qk_vcache_slice_ranges.begin() + 3,
                               qk_vcache_slice_ranges.begin() + 6);
  qk_vcache_slice_ranges[4] /= num_attn_per_kv_heads;
  std::vector<uint32_t> qk_vcache_slice_param_dims = {
      static_cast<uint32_t>(qk_vcache_slice_ranges.size() / 3), 3};
  auto& qk_vcache_slice_param_tensor = tensor_pool.CreateStaticTensor(
      qk_vcache_slice_param.GetDataType(),
      qk_vcache_slice_param.GetQuantParams(), qk_vcache_slice_param_dims,
      sizeof(int32_t) * qk_vcache_slice_ranges.size(),
      qk_vcache_slice_ranges.data());
  auto qk_vcache_slice_output_dims =
      qk_vcache_slice.GetOutputTensor(0).GetDimensions();
  qk_vcache_slice_output_dims.erase(qk_vcache_slice_output_dims.begin() + 1);
  qk_vcache_slice_output_dims[1] /= num_attn_per_kv_heads;
  auto& qk_vcache_slice_output = tensor_pool.CloneNativeTensorFrom(
      qk_vcache_slice.GetOutputTensor(0), qk_vcache_slice_output_dims);
  new_ops.emplace_back(CreateSliceOp(softmax_output, qk_vcache_slice_output,
                                     qk_vcache_slice_param_tensor));

  // QK VSlice Slice
  auto qk_vslice_slice_param = qk_vslice_slice.GetTensorParam(0).GetTensor();
  auto qk_vslice_slice_param_data =
      qk_vslice_slice_param.GetTensorData<int32_t>();
  std::vector<int32_t> qk_vslice_slice_ranges(
      qk_vslice_slice_param_data.value().begin(),
      qk_vslice_slice_param_data.value().end());
  qk_vslice_slice_ranges.erase(qk_vslice_slice_ranges.begin() + 3,
                               qk_vslice_slice_ranges.begin() + 6);
  qk_vslice_slice_ranges[4] /= num_attn_per_kv_heads;
  std::vector<uint32_t> qk_vslice_slice_param_dims = {
      static_cast<uint32_t>(qk_vslice_slice_ranges.size() / 3), 3};
  auto& qk_vslice_slice_param_tensor = tensor_pool.CreateStaticTensor(
      qk_vslice_slice_param.GetDataType(),
      qk_vslice_slice_param.GetQuantParams(), qk_vslice_slice_param_dims,
      sizeof(int32_t) * qk_vslice_slice_ranges.size(),
      qk_vslice_slice_ranges.data());
  auto qk_vslice_slice_output_dims =
      qk_vslice_slice.GetOutputTensor(0).GetDimensions();
  qk_vslice_slice_output_dims.erase(qk_vslice_slice_output_dims.begin() + 1);
  qk_vslice_slice_output_dims[1] /= num_attn_per_kv_heads;
  auto& qk_vslice_slice_output = tensor_pool.CloneNativeTensorFrom(
      qk_vslice_slice.GetOutputTensor(0), qk_vslice_slice_output_dims);
  new_ops.emplace_back(CreateSliceOp(softmax_output, qk_vslice_slice_output,
                                     qk_vslice_slice_param_tensor));

  // QK VCache Matmul
  auto qk_vcache_matmul_output_dims =
      qk_vcache_matmul.GetOutputTensor(0).GetDimensions();
  qk_vcache_matmul_output_dims.erase(qk_vcache_matmul_output_dims.begin() + 1);
  qk_vcache_matmul_output_dims[1] /= num_attn_per_kv_heads;
  auto& qk_vcache_matmul_output = tensor_pool.CloneNativeTensorFrom(
      qk_vcache_matmul.GetOutputTensor(0), qk_vcache_matmul_output_dims);
  const std::array<ConstTensorWrapperRef, 2> qk_vcache_matmul_inputs = {
      qk_vcache_slice_output, v_cache};
  const std::array<ConstTensorWrapperRef, 1> qk_vcache_matmul_outputs = {
      qk_vcache_matmul_output};
  new_ops.emplace_back(CreateOpWithSameParams(
      qk_vcache_matmul, qk_vcache_matmul_inputs, qk_vcache_matmul_outputs));

  // QK VSlice Matmul
  auto qk_vslice_matmul_output_dims =
      qk_vslice_matmul.GetOutputTensor(0).GetDimensions();
  qk_vslice_matmul_output_dims.erase(qk_vslice_matmul_output_dims.begin() + 1);
  qk_vslice_matmul_output_dims[1] /= num_attn_per_kv_heads;
  auto& qk_vslice_matmul_output = tensor_pool.CloneNativeTensorFrom(
      qk_vslice_matmul.GetOutputTensor(0), qk_vslice_matmul_output_dims);
  const std::array<ConstTensorWrapperRef, 2> qk_vslice_matmul_inputs = {
      qk_vslice_slice_output, v_slice};
  const std::array<ConstTensorWrapperRef, 1> qk_vslice_matmul_outputs = {
      qk_vslice_matmul_output};
  new_ops.emplace_back(CreateOpWithSameParams(
      qk_vslice_matmul, qk_vslice_matmul_inputs, qk_vslice_matmul_outputs));

  // QKV Add
  auto qkv_add_output_dims = qkv_add.GetOutputTensor(0).GetDimensions();
  qkv_add_output_dims.erase(qkv_add_output_dims.begin() + 1);
  qkv_add_output_dims[1] /= num_attn_per_kv_heads;
  auto& qkv_add_output = tensor_pool.CloneNativeTensorFrom(
      qkv_add.GetOutputTensor(0), qkv_add_output_dims);
  new_ops.emplace_back(CreateElementWiseAddOp(
      qk_vcache_matmul_output, qk_vslice_matmul_output, qkv_add_output));

  return qkv_add_output;
}

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
  auto transpose_output_dims = ops[start_index + kTransposePrefillIndex]
                                   .GetOutputTensor(0)
                                   .GetDimensions();
  const auto& transpose_output =
      tensor_pool.CloneNativeTensorFrom(pattern_input, transpose_output_dims);
  const std::array<ConstTensorWrapperRef, 1> transpose_inputs = {pattern_input};
  const std::array<ConstTensorWrapperRef, 1> transpose_outputs = {
      transpose_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(ops[start_index + kTransposePrefillIndex],
                             transpose_inputs, transpose_outputs));

  // Reshape
  const auto& reshape_output = tensor_pool.CloneNativeTensorFrom(
      pattern_input, {transpose_output_dims[0], 1,
                      transpose_output_dims[1] * transpose_output_dims[2],
                      transpose_output_dims[3]});
  new_ops.emplace_back(CreateReshapeOp(transpose_output, reshape_output));

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDimension(2);
  const auto& mha_input = new_ops.back().GetOutputTensor(0);
  auto sha_ops =
      TransformToSHA(ops, start_index + new_ops.size(), tensor_pool, mha_input,
                     pattern_output, scaling_mul, num_heads);
  std::move(sha_ops.begin(), sha_ops.end(), std::back_inserter(new_ops));

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
  const int num_heads = pattern_input.GetDimension(2);
  auto sha_ops =
      TransformToSHA(ops, start_index + new_ops.size(), tensor_pool,
                     pattern_input, pattern_output, scaling_mul, num_heads);
  std::move(sha_ops.begin(), sha_ops.end(), std::back_inserter(new_ops));

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

size_t OptimizeMHAFastVlmPrefill(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr int32_t kKSliceAddIdx = -2;
  constexpr size_t kQScaleMulIdx = 0;
  constexpr size_t kQScaleReshapeIdx = 1;
  constexpr size_t kQKCacheMatmulIdx = 2;
  constexpr size_t kQKSliceMatmulIdx = 3;
  constexpr size_t kQKConcatIdx = 4;
  constexpr size_t kPreMaskReshapeIdx = 5;
  constexpr size_t kMaskAddIdx = 6;
  constexpr size_t kPostMaskReshapeIdx = 7;
  constexpr size_t kSoftmaxIdx = 8;
  constexpr size_t kQKVCacheSliceIdx = 9;
  constexpr size_t kQKVSliceSliceIdx = 10;
  constexpr size_t kQKVCacheMatmulIdx = 11;
  constexpr size_t kQKVSliceMatmulIdx = 12;
  constexpr size_t kQKVAddIdx = 13;
  constexpr size_t kQKVReshapeIdx = 14;
  constexpr size_t kQKVTransposeIdx = 15;
  constexpr size_t kOProjReshapeIdx = 16;

  const auto is_connected =
      [&ops, &start_index](int32_t output_op_index, size_t output_tensor_index,
                           int32_t input_op_index,
                           size_t input_tensor_index) -> bool {
    // Input/output op index might be negative.
    int32_t out_op_idx = static_cast<int32_t>(start_index) + output_op_index;
    int32_t in_op_idx = static_cast<int32_t>(start_index) + input_op_index;
    return out_op_idx >= 0 && in_op_idx >= 0 &&
           ops[out_op_idx].GetOutputTensor(output_tensor_index) ==
               ops[in_op_idx].GetInputTensor(input_tensor_index);
  };
  if (!(is_connected(kKSliceAddIdx, 0, kQKSliceMatmulIdx, 1) &&
        is_connected(kQScaleMulIdx, 0, kQScaleReshapeIdx, 0) &&
        is_connected(kQScaleReshapeIdx, 0, kQKCacheMatmulIdx, 0) &&
        is_connected(kQScaleReshapeIdx, 0, kQKSliceMatmulIdx, 0) &&
        is_connected(kQKCacheMatmulIdx, 0, kQKConcatIdx, 0) &&
        is_connected(kQKSliceMatmulIdx, 0, kQKConcatIdx, 1) &&
        is_connected(kQKConcatIdx, 0, kPreMaskReshapeIdx, 0) &&
        is_connected(kPreMaskReshapeIdx, 0, kMaskAddIdx, 0) &&
        is_connected(kMaskAddIdx, 0, kPostMaskReshapeIdx, 0) &&
        is_connected(kPostMaskReshapeIdx, 0, kSoftmaxIdx, 0) &&
        is_connected(kSoftmaxIdx, 0, kQKVCacheSliceIdx, 0) &&
        is_connected(kSoftmaxIdx, 0, kQKVSliceSliceIdx, 0) &&
        is_connected(kQKVCacheSliceIdx, 0, kQKVCacheMatmulIdx, 0) &&
        is_connected(kQKVSliceSliceIdx, 0, kQKVSliceMatmulIdx, 0) &&
        is_connected(kQKVCacheMatmulIdx, 0, kQKVAddIdx, 0) &&
        is_connected(kQKVSliceMatmulIdx, 0, kQKVAddIdx, 1) &&
        is_connected(kQKVAddIdx, 0, kQKVReshapeIdx, 0) &&
        is_connected(kQKVReshapeIdx, 0, kQKVTransposeIdx, 0) &&
        is_connected(kQKVTransposeIdx, 0, kOProjReshapeIdx, 0) &&
        IsElementWiseMultiply(ops[start_index + kQScaleMulIdx]) &&
        IsElementWiseAdd(ops[start_index + kKSliceAddIdx]) &&
        IsElementWiseAdd(ops[start_index + kMaskAddIdx]) &&
        IsElementWiseAdd(ops[start_index + kQKVAddIdx]))) {
    return 1;
  }

  // Strict check for only FastVLM with q head = 14, kv head = 2.
  static constexpr size_t kQHeads = 14;
  static constexpr size_t kKVHeads = 2;
  if (!(kKVHeads ==
            ops[start_index + kKSliceAddIdx].GetInputTensor(0).GetDimension(
                1) &&
        kKVHeads ==
            ops[start_index + kKSliceAddIdx].GetInputTensor(1).GetDimension(
                1) &&
        kQHeads ==
            ops[start_index + kQScaleMulIdx].GetInputTensor(0).GetDimension(
                1) &&
        kKVHeads == ops[start_index + kQKVCacheMatmulIdx]
                        .GetInputTensor(1)
                        .GetDimension(1) &&
        kKVHeads == ops[start_index + kQKVSliceMatmulIdx]
                        .GetInputTensor(1)
                        .GetDimension(1))) {
    QNN_LOG_WARNING(
        "[G2G] Pattern does not match Q heads: %d, KV heads &d. In pattern, "
        "k_slice_add_in_0_dims[1]: %d, k_slice_add_in_1_dims[1]: %d,"
        "q_scale_mul_dims[1]: %d, v_cache_dims[1]: %d, v_slice_dims[1]: %d",
        kQHeads, kKVHeads,
        ops[start_index + kKSliceAddIdx].GetInputTensor(0).GetDimension(1),
        ops[start_index + kKSliceAddIdx].GetInputTensor(1).GetDimension(1),
        ops[start_index + kQScaleMulIdx].GetInputTensor(0).GetDimension(1),
        ops[start_index + kQKVCacheMatmulIdx].GetInputTensor(1).GetDimension(1),
        ops[start_index + kQKVSliceMatmulIdx].GetInputTensor(1).GetDimension(
            1));
    return 1;
  }
  QNN_LOG_INFO("[G2G] MHA Optimization (FastVLM Prefill).");
  std::vector<OpWrapper> new_ops;

  // Manually unpack the Add Op inorder to reuse BuildSingleSHAByUnpackAxis1().
  const auto& k_slice_add_in_0 =
      ops[start_index + kKSliceAddIdx].GetInputTensor(0);
  auto k_slice_add_0_unpack_outputs =
      UnpackTensor(tensor_pool, new_ops, k_slice_add_in_0);

  const auto& k_slice_add_in_1 =
      ops[start_index + kKSliceAddIdx].GetInputTensor(1);
  auto k_slice_add_1_unpack_outputs =
      UnpackTensor(tensor_pool, new_ops, k_slice_add_in_1);

  const auto& k_slice_add_out_0 =
      ops[start_index + kKSliceAddIdx].GetOutputTensor(0);
  size_t num_kv_heads = k_slice_add_out_0.GetDimension(1);
  std::vector<TensorWrapperRef> k_slice_add_outputs;
  k_slice_add_outputs.reserve(num_kv_heads);

  auto k_slice_add_out_0_dims = k_slice_add_out_0.GetDimensions();
  k_slice_add_out_0_dims.erase(k_slice_add_out_0_dims.begin() + 1);
  for (int i = 0; i < num_kv_heads; i++) {
    auto& cloned_k_slice_add_out_0 = tensor_pool.CloneNativeTensorFrom(
        k_slice_add_out_0, k_slice_add_out_0_dims);
    k_slice_add_outputs.emplace_back(cloned_k_slice_add_out_0);
    new_ops.emplace_back(CreateElementWiseAddOp(k_slice_add_0_unpack_outputs[i],
                                                k_slice_add_1_unpack_outputs[i],
                                                cloned_k_slice_add_out_0));
  }

  // QKV Unpack
  const auto& k_cache = ops[start_index + kQKCacheMatmulIdx].GetInputTensor(1);
  auto k_cache_unpack_outputs = UnpackTensor(tensor_pool, new_ops, k_cache);

  const auto& scale_mul_in = ops[start_index + kQScaleMulIdx].GetInputTensor(0);
  auto scale_mul_unpack_outputs =
      UnpackTensor(tensor_pool, new_ops, scale_mul_in);

  const auto& v_cache = ops[start_index + kQKVCacheMatmulIdx].GetInputTensor(1);
  auto v_cache_unpack_outputs = UnpackTensor(tensor_pool, new_ops, v_cache);

  const auto& v_slice = ops[start_index + kQKVSliceMatmulIdx].GetInputTensor(1);
  auto v_slice_unpack_outputs = UnpackTensor(tensor_pool, new_ops, v_slice);

  auto num_attn_heads = scale_mul_unpack_outputs.size();
  auto num_attn_per_kv_heads = num_attn_heads / num_kv_heads;

  // Build num_head SHAs
  std::vector<ConstTensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_attn_heads);
  for (size_t i = 0; i < num_kv_heads; ++i) {
    for (size_t j = 0; j < num_attn_per_kv_heads; ++j) {
      auto& sha_output = BuildSingleSHAByUnpackAxis1(
          new_ops, tensor_pool, num_attn_per_kv_heads,
          scale_mul_unpack_outputs[i * num_attn_per_kv_heads + j],
          k_cache_unpack_outputs[i], k_slice_add_outputs[i],
          v_cache_unpack_outputs[i], v_slice_unpack_outputs[i],
          ops[start_index + kQScaleMulIdx],
          ops[start_index + kQKCacheMatmulIdx],
          ops[start_index + kQKSliceMatmulIdx], ops[start_index + kQKConcatIdx],
          ops[start_index + kMaskAddIdx],
          ops[start_index + kPostMaskReshapeIdx],
          ops[start_index + kSoftmaxIdx], ops[start_index + kQKVCacheSliceIdx],
          ops[start_index + kQKVSliceSliceIdx],
          ops[start_index + kQKVCacheMatmulIdx],
          ops[start_index + kQKVSliceMatmulIdx], ops[start_index + kQKVAddIdx]);
      sha_outputs.emplace_back(sha_output);
    }
  }

  // Concat
  const auto& pattern_output =
      ops[start_index + pattern_size - 1].GetOutputTensor(0);
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

size_t OptimizeMHAFastVlmDecode(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  QNN_LOG_INFO("[G2G] MHA optimization (fast vlm decode)");

  constexpr size_t kQScaleMulIdx = 0;
  constexpr size_t kQScaleReshapeIdx = 1;
  constexpr size_t kQKCacheMatmulIdx = 2;
  constexpr size_t kQKSliceMatmulIdx = 3;
  constexpr size_t kQKConcatIdx = 4;
  constexpr size_t kPreMaskReshapeIdx = 5;
  constexpr size_t kMaskAddIdx = 6;
  constexpr size_t kPostMaskReshapeIdx = 7;
  constexpr size_t kSoftmaxIdx = 8;
  constexpr size_t kQKVCacheSliceIdx = 9;
  constexpr size_t kQKVSliceSliceIdx = 10;
  constexpr size_t kQKVCacheMatmulIdx = 11;
  constexpr size_t kQKVSliceMatmulIdx = 12;
  constexpr size_t kQKVAddIdx = 13;
  constexpr size_t kQKVReshapeIdx = 14;
  const auto& q_scale_mul = ops[start_index + kQScaleMulIdx];
  const auto& q_scale_reshape = ops[start_index + kQScaleReshapeIdx];
  const auto& q_kcache_matmul = ops[start_index + kQKCacheMatmulIdx];
  const auto& q_kslice_matmul = ops[start_index + kQKSliceMatmulIdx];
  const auto& qk_concat = ops[start_index + kQKConcatIdx];
  const auto& pre_mask_reshape = ops[start_index + kPreMaskReshapeIdx];
  const auto& mask_add = ops[start_index + kMaskAddIdx];
  const auto& post_mask_reshape = ops[start_index + kPostMaskReshapeIdx];
  const auto& softmax = ops[start_index + kSoftmaxIdx];
  const auto& qk_vcache_slice = ops[start_index + kQKVCacheSliceIdx];
  const auto& qk_vslice_slice = ops[start_index + kQKVSliceSliceIdx];
  const auto& qk_vcache_matmul = ops[start_index + kQKVCacheMatmulIdx];
  const auto& qk_vslice_matmul = ops[start_index + kQKVSliceMatmulIdx];
  const auto& qkv_add = ops[start_index + kQKVAddIdx];
  const auto& qkv_reshape = ops[start_index + kQKVReshapeIdx];

  const auto is_connected =
      [](const OpWrapper& output, size_t output_tensor_index,
         const OpWrapper& input, size_t input_tensor_index) -> bool {
    return output.GetOutputTensor(output_tensor_index) ==
           input.GetInputTensor(input_tensor_index);
  };
  if (!(is_connected(q_scale_mul, 0, q_scale_reshape, 0) &&
        is_connected(q_scale_reshape, 0, q_kcache_matmul, 0) &&
        is_connected(q_scale_reshape, 0, q_kslice_matmul, 0) &&
        is_connected(q_kcache_matmul, 0, qk_concat, 0) &&
        is_connected(q_kslice_matmul, 0, qk_concat, 1) &&
        is_connected(qk_concat, 0, pre_mask_reshape, 0) &&
        is_connected(pre_mask_reshape, 0, mask_add, 0) &&
        is_connected(mask_add, 0, post_mask_reshape, 0) &&
        is_connected(post_mask_reshape, 0, softmax, 0) &&
        is_connected(softmax, 0, qk_vcache_slice, 0) &&
        is_connected(softmax, 0, qk_vslice_slice, 0) &&
        is_connected(qk_vcache_slice, 0, qk_vcache_matmul, 0) &&
        is_connected(qk_vslice_slice, 0, qk_vslice_matmul, 0) &&
        is_connected(qk_vcache_matmul, 0, qkv_add, 0) &&
        is_connected(qk_vslice_matmul, 0, qkv_add, 1) &&
        is_connected(qkv_add, 0, qkv_reshape, 0))) {
    QNN_LOG_WARNING(
        "[G2G] Failed to check connectivity when doing MHA-SHA transformation "
        "for FastVLM decode.");
    return 1;
  }

  constexpr size_t kSupportedRank = 4;
  constexpr size_t kUnpackAxis = 1;
  if (q_scale_mul.GetInputTensor(0).GetRank() != kSupportedRank ||
      q_kcache_matmul.GetInputTensor(1).GetRank() != kSupportedRank ||
      q_kslice_matmul.GetInputTensor(1).GetRank() != kSupportedRank ||
      qk_vcache_matmul.GetInputTensor(1).GetRank() != kSupportedRank ||
      qk_vslice_matmul.GetInputTensor(1).GetRank() != kSupportedRank ||
      mask_add.GetInputTensor(1).GetRank() != kSupportedRank ||
      mask_add.GetInputTensor(1).GetDimension(0) != 1 ||
      mask_add.GetInputTensor(1).GetDimension(1) != 1) {
    QNN_LOG_WARNING(
        "[G2G] Failed to check dimensions when doing MHA-SHA transformation "
        "for FastVLM decode.");
    return 1;
  }

  std::vector<OpWrapper> new_ops;

  auto scale_mul_unpack_outputs = UnpackTensor(
      tensor_pool, new_ops, q_scale_mul.GetInputTensor(0), kUnpackAxis);

  auto k_slice_unpack_outputs = UnpackTensor(
      tensor_pool, new_ops, q_kslice_matmul.GetInputTensor(1), kUnpackAxis);

  auto k_cache_unpack_outputs = UnpackTensor(
      tensor_pool, new_ops, q_kcache_matmul.GetInputTensor(1), kUnpackAxis);

  auto v_cache_unpack_outputs = UnpackTensor(
      tensor_pool, new_ops, qk_vcache_matmul.GetInputTensor(1), kUnpackAxis);

  auto v_slice_unpack_outputs = UnpackTensor(
      tensor_pool, new_ops, qk_vslice_matmul.GetInputTensor(1), kUnpackAxis);

  // Build SHA
  const auto num_attn_head = scale_mul_unpack_outputs.size();
  const auto num_kv_head = k_slice_unpack_outputs.size();
  const auto num_attn_per_kv_head = num_attn_head / num_kv_head;
  std::vector<ConstTensorWrapperRef> sha_outputs;
  for (size_t i = 0; i < num_kv_head; ++i) {
    for (size_t j = 0; j < num_attn_per_kv_head; ++j) {
      const auto& sha_output = BuildSingleSHAByUnpackAxis1(
          new_ops, tensor_pool, num_attn_per_kv_head,
          scale_mul_unpack_outputs[i * num_attn_per_kv_head + j],
          k_cache_unpack_outputs[i], k_slice_unpack_outputs[i],
          v_cache_unpack_outputs[i], v_slice_unpack_outputs[i], q_scale_mul,
          q_kcache_matmul, q_kslice_matmul, qk_concat, mask_add,
          post_mask_reshape, softmax, qk_vcache_slice, qk_vslice_slice,
          qk_vcache_matmul, qk_vslice_matmul, qkv_add);
      sha_outputs.emplace_back(sha_output);
    }
  }

  // Concat SHA outputs by the second-to-last dimension.
  const auto concat_axis = sha_outputs[0].get().GetRank() - 2;
  auto concat_sha_dims = sha_outputs[0].get().GetDimensions();
  concat_sha_dims[concat_axis] = 0;
  for (const auto& sha_output : sha_outputs) {
    concat_sha_dims[concat_axis] += sha_output.get().GetDimension(concat_axis);
  }
  const auto& concat_sha_output = tensor_pool.CloneNativeTensorFrom(
      qkv_reshape.GetInputTensor(0), concat_sha_dims);
  new_ops.emplace_back(
      CreateConcatenationOp(sha_outputs, concat_sha_output, concat_axis));
  new_ops.emplace_back(
      CreateReshapeOp(concat_sha_output, qkv_reshape.GetOutputTensor(0)));

  // Validate new graph.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
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
    QNN_LOG_INFO("[G2G] FastVLM decode optimization done.");
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
  auto transpose_output_dims = transpoe_0.GetOutputTensor(0).GetDimensions();
  const auto& transpose_output =
      tensor_pool.CloneNativeTensorFrom(pattern_input, transpose_output_dims);
  const std::array<ConstTensorWrapperRef, 1> transpose_inputs = {pattern_input};
  const std::array<ConstTensorWrapperRef, 1> transpose_outputs = {
      transpose_output};
  auto& new_transpose_0 = new_ops.emplace_back(
      CreateOpWithSameParams(transpoe_0, transpose_inputs, transpose_outputs));

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDimension(2);
  const auto& mha_input = new_transpose_0.GetOutputTensor(0);

  // Prepare inputs for num_heads SHAs.
  std::vector<ConstTensorWrapperRef> sha_inputs;
  sha_inputs.reserve(num_heads);
  auto sha_input_dims = mha_input.GetDimensions();
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
    split_indice.emplace_back(i * mha_input.GetDimension(1) / num_heads);
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
  auto splited_mask_dims = concated_mask.GetDimensions();
  splited_mask_dims[2] /= num_heads;
  for (size_t i = 0; i < num_heads; ++i) {
    splited_masks.emplace_back(
        tensor_pool.CloneNativeTensorFrom(concated_mask, splited_mask_dims));
  }
  std::vector<std::uint32_t> split_mask_indice;
  split_mask_indice.reserve(num_heads);
  for (std::uint32_t i = 1; i < num_heads; i++) {
    split_mask_indice.emplace_back(i * concated_mask.GetDimension(2) /
                                   num_heads);
  }
  const auto& split_mask_indice_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {},
      {static_cast<std::uint32_t>(split_mask_indice.size())},
      sizeof(std::uint32_t) * split_mask_indice.size(),
      split_mask_indice.data());
  new_ops.emplace_back(
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
  auto concat_output_dims = sha_outputs[0].get().GetDimensions();
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
                    return validate_op_config(op_wrapper);
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
  auto not_equal_out_dims = not_equal_out.GetDimensions();
  not_equal_out_dims.erase(not_equal_out_dims.begin() + 1);
  const auto& select_mask =
      tensor_pool.CloneNativeTensorFrom(not_equal_out, not_equal_out_dims);
  // Change NotEqual to Equal -> Cast -> Mul.
  const auto& zero_tensor = attn_not_equal_op.GetInputTensor(1);
  new_ops.emplace_back(
      CreateElementWiseEqualOp(reshape_in, zero_tensor, select_mask));
  const auto& select_out =
      ops[attn_start_index + kAttnSelect].GetOutputTensor(0);
  auto select_out_dims = select_out.GetDimensions();
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
      select_const.GetDimensions(), select_const.GetTensorBytes(),
      &mul_const_value);
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
    auto q_unpack_dims = mul_q_in.GetDimensions();
    uint32_t num_heads = q_unpack_dims[2];
    const auto& mul_k_in = ops[matmul_qk_index + kAttnMulK].GetInputTensor(0);
    auto k_unpack_dims = mul_k_in.GetDimensions();
    const auto& transpose_v_in =
        ops[select_index + kAttnTransposeIn].GetInputTensor(0);
    auto transpose_v_perm =
        ops[select_index + kAttnTransposeIn].GetTensorParam(0).GetTensor();
    std::vector<uint32_t> perm_data = {0, 2, 1};
    const auto& perm_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, transpose_v_perm.GetQuantParams(), {3},
        perm_data.size() * sizeof(perm_data[0]), perm_data.data());
    auto v_unpack_dims = transpose_v_in.GetDimensions();
    const auto& mha_out =
        ops[select_index + kAttnTransposeOut].GetOutputTensor(0);
    auto mha_out_dims = mha_out.GetDimensions();
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
          matmul_qk_out,
          {q_matmul_in.GetDimension(0), q_matmul_in.GetDimension(1),
           k_matmul_in.GetDimension(2)});
      const std::array<ConstTensorWrapperRef, 2> matmul_qk_inputs = {
          q_matmul_in, k_matmul_in};
      const std::array<ConstTensorWrapperRef, 1> matmul_qk_outputs = {
          select_in};
      new_ops.emplace_back(CreateOpWithSameParams(
          ops[matmul_qk_index], matmul_qk_inputs, matmul_qk_outputs));

      // Change Select to Add.
      const auto& softmax_in =
          tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
      new_ops.emplace_back(
          CreateElementWiseAddOp(select_in, add_in, softmax_in));

      // Softmax
      const auto& qk_softmax =
          tensor_pool.CloneNativeTensorFrom(softmax_in, select_out_dims);
      const std::array<ConstTensorWrapperRef, 1> softmax_inputs = {softmax_in};
      const std::array<ConstTensorWrapperRef, 1> softmax_outputs = {qk_softmax};
      new_ops.emplace_back(CreateOpWithSameParams(
          ops[select_index + kAttnSoftmax], softmax_inputs, softmax_outputs));

      // MatMul
      const std::array<ConstTensorWrapperRef, 2> matmul_out_inputs = {
          qk_softmax, v_sha_inputs[i]};
      const std::array<ConstTensorWrapperRef, 1> matmul_out_outputs = {
          sha_outputs[i]};
      new_ops.emplace_back(CreateOpWithSameParams(
          ops[matmul_qk_index], matmul_out_inputs, matmul_out_outputs));
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
