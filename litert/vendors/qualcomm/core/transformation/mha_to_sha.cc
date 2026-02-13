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
#include "litert/vendors/qualcomm/core/builders/fully_connected_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pack_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
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

// Emplaces the operator with updated inputs/outputs into new_ops.
// This function copies the source_op and updates the tensors according to
// inputs and outputs. The std::nullopt input/output element indicates that the
// tensor will not be updated.
OpWrapper& EmplaceOpWithIO(
    std::vector<OpWrapper>& new_ops, const OpWrapper& source_op,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& inputs,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& outputs) {
  OpWrapper ret = source_op;
  ret.UpdateTensors(inputs, outputs);
  return new_ops.emplace_back(ret);
}

TensorWrapper& BuildSingleSHA(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    TensorWrapper& sha_input, TensorWrapper& mask, size_t num_heads,
    const OpWrapper& mul, const OpWrapper& matmul_k1,
    const OpWrapper& matmul_k2, const OpWrapper& concat, const OpWrapper& add_1,
    const OpWrapper& softmax, const OpWrapper& slice_1,
    const OpWrapper& slice_2, const OpWrapper& matmul_v1,
    const OpWrapper& matmul_v2, const OpWrapper& add_2) {
  // Mul
  auto& mul_output = tensor_pool.CloneNativeTensorFrom(mul.GetOutputTensor(0),
                                                       sha_input.GetDims());
  EmplaceOpWithIO(new_ops, mul, {sha_input, std::nullopt}, {mul_output});

  // MatMul 1
  auto matmul_k1_output_dims = matmul_k1.GetOutputTensor(0).GetDims();
  matmul_k1_output_dims[2] /= num_heads;
  auto& matmul_k1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k1.GetOutputTensor(0), matmul_k1_output_dims);
  EmplaceOpWithIO(new_ops, matmul_k1, {mul_output, std::nullopt},
                  {matmul_k1_output});
  // MatMul 2
  auto matmul_k2_output_dims = matmul_k2.GetOutputTensor(0).GetDims();
  matmul_k2_output_dims[2] /= num_heads;
  auto& matmul_k2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k2.GetOutputTensor(0), matmul_k2_output_dims);
  EmplaceOpWithIO(new_ops, matmul_k2, {mul_output, std::nullopt},
                  {matmul_k2_output});
  // Concat
  auto concat_output_dims = matmul_k1_output_dims;
  concat_output_dims[3] += matmul_k2_output_dims[3];
  auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      concat.GetOutputTensor(0), concat_output_dims);
  EmplaceOpWithIO(new_ops, concat, {matmul_k1_output, matmul_k2_output},
                  {concat_output});
  // Add
  auto& add_1_output = tensor_pool.CloneNativeTensorFrom(
      add_1.GetOutputTensor(0), concat_output.GetDims());
  EmplaceOpWithIO(new_ops, add_1, {concat_output, mask}, {add_1_output});
  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), add_1_output.GetDims());
  EmplaceOpWithIO(new_ops, softmax, {add_1_output}, {softmax_output});

  // Slice 1
  auto slice_1_ranges = slice_1.GetTensorPararm(0).GetTensor();
  auto slice_1_rangs_data = slice_1_ranges.GetTensorData<int32_t>();
  std::vector<int32_t> sha_slice_1_ranges_data(
      slice_1_rangs_data.value().begin(), slice_1_rangs_data.value().end());
  sha_slice_1_ranges_data[kSlice3rdAxisEndIndex] /= num_heads;
  auto& sha_slice_1_ranges = tensor_pool.CreateStaticTensor(
      slice_1_ranges.GetDataType(), slice_1_ranges.GetQuantParams(),
      slice_1_ranges.GetDims(), slice_1_ranges.GetTensorBytes(),
      sha_slice_1_ranges_data.data());
  auto slice_1_output_dims = slice_1.GetOutputTensor(0).GetDims();
  slice_1_output_dims[2] /= num_heads;
  auto& slice_1_output = tensor_pool.CloneNativeTensorFrom(
      slice_1.GetOutputTensor(0), slice_1_output_dims);
  BuildSliceOp(new_ops.emplace_back(), softmax_output, slice_1_output,
               sha_slice_1_ranges);

  // Slice 2
  auto slice_2_ranges = slice_2.GetTensorPararm(0).GetTensor();
  auto slice_2_ranges_data = slice_2_ranges.GetTensorData<int32_t>();
  std::vector<int32_t> sha_slice_2_ranges_data(
      slice_2_ranges_data.value().begin(), slice_2_ranges_data.value().end());
  sha_slice_2_ranges_data[kSlice3rdAxisEndIndex] /= num_heads;
  auto& sha_slice_2_ranges = tensor_pool.CreateStaticTensor(
      slice_2_ranges.GetDataType(), slice_2_ranges.GetQuantParams(),
      slice_2_ranges.GetDims(), slice_2_ranges.GetTensorBytes(),
      sha_slice_2_ranges_data.data());
  auto slice_2_output_dims = slice_2.GetOutputTensor(0).GetDims();
  slice_2_output_dims[2] /= num_heads;
  auto& slice_2_output = tensor_pool.CloneNativeTensorFrom(
      slice_2.GetOutputTensor(0), slice_2_output_dims);
  BuildSliceOp(new_ops.emplace_back(), softmax_output, slice_2_output,
               sha_slice_2_ranges);

  // MatMul 1
  std::vector<uint32_t> matmul_v1_output_dims =
      matmul_v1.GetOutputTensor(0).GetDims();
  matmul_v1_output_dims[2] = matmul_v1_output_dims[2] / num_heads;
  auto& matmul_v1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v1.GetOutputTensor(0), matmul_v1_output_dims);
  EmplaceOpWithIO(new_ops, matmul_v1, {slice_1_output, std::nullopt},
                  {matmul_v1_output});

  // MatMul 2
  std::vector<uint32_t> matmul_v2_output_dims =
      matmul_v2.GetOutputTensor(0).GetDims();
  matmul_v2_output_dims[2] = matmul_v2_output_dims[2] / num_heads;
  auto& matmul_v2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v2.GetOutputTensor(0), matmul_v2_output_dims);
  EmplaceOpWithIO(new_ops, matmul_v2, {slice_2_output, std::nullopt},
                  {matmul_v2_output});
  // Add 2
  auto& add_2_output = tensor_pool.CloneNativeTensorFrom(
      add_2.GetOutputTensor(0), matmul_v1_output.GetDims());
  EmplaceOpWithIO(new_ops, add_2, {matmul_v1_output, matmul_v2_output},
                  {add_2_output});
  return add_2_output;
}

TensorWrapper& BuildSingleSHA(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    TensorWrapper& v_cache, TensorWrapper& q_slice,
    TensorWrapper& sha_mul_input, TensorWrapper& sha_add_input_1,
    TensorWrapper& sha_add_input_2, TensorWrapper& v_slice,
    const uint32_t num_attn_per_kv_heads, const OpWrapper& mul,
    const OpWrapper& matmul_k1, const OpWrapper& add_1,
    const OpWrapper& matmul_k2, const OpWrapper& concat, const OpWrapper& add_2,
    const OpWrapper& reshape_3, const OpWrapper& softmax,
    const OpWrapper& slice_1, const OpWrapper& slice_2,
    const OpWrapper& matmul_v1, const OpWrapper& matmul_v2,
    const OpWrapper& add_3, TensorWrapper* sha_add_output) {
  // Mul
  auto mul_output_dims = mul.GetOutputTensor(0).GetDims();
  mul_output_dims.erase(mul_output_dims.begin() + 1);
  auto& mul_output = tensor_pool.CloneNativeTensorFrom(mul.GetOutputTensor(0),
                                                       mul_output_dims);
  EmplaceOpWithIO(new_ops, mul, {sha_mul_input, std::nullopt}, {mul_output});

  // Matmul q
  auto matmul_q_output_dims = matmul_k1.GetOutputTensor(0).GetDims();
  matmul_q_output_dims.erase(matmul_q_output_dims.begin() + 1);
  matmul_q_output_dims[1] /= num_attn_per_kv_heads;
  auto& matmul_q_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k1.GetOutputTensor(0), matmul_q_output_dims);
  EmplaceOpWithIO(new_ops, matmul_k1, {mul_output, q_slice}, {matmul_q_output});

  // Add
  // TODO(Alen): remove hack.
  if (sha_add_output == nullptr) {
    auto add_1_output_dims = add_1.GetOutputTensor(0).GetDims();
    add_1_output_dims.erase(add_1_output_dims.begin() + 1);
    auto& add_1_output = tensor_pool.CloneNativeTensorFrom(
        add_1.GetOutputTensor(0), add_1_output_dims);
    EmplaceOpWithIO(new_ops, add_1, {sha_add_input_1, sha_add_input_2},
                    {add_1_output});
    sha_add_output = &add_1_output;
  }

  // Matmul k
  auto matmul_qk_output_dims = matmul_k2.GetOutputTensor(0).GetDims();
  matmul_qk_output_dims.erase(matmul_qk_output_dims.begin() + 1);
  matmul_qk_output_dims[1] /= num_attn_per_kv_heads;
  auto& matmul_qk_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k2.GetOutputTensor(0), matmul_qk_output_dims);
  EmplaceOpWithIO(new_ops, matmul_k2, {mul_output, *sha_add_output},
                  {matmul_qk_output});

  // Concat
  std::uint32_t adjusted_axis = 2;
  auto concat_output_dims = concat.GetOutputTensor(0).GetDims();
  concat_output_dims.erase(concat_output_dims.begin() + 1);
  concat_output_dims[1] /= num_attn_per_kv_heads;
  auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      concat.GetOutputTensor(0), concat_output_dims);
  auto concat_op =
      BuildConcatenationOp(tensor_pool, {matmul_q_output, matmul_qk_output},
                           {concat_output}, adjusted_axis);
  std::move(concat_op.begin(), concat_op.end(), std::back_inserter(new_ops));

  // Add 2
  auto add_2_output_dims = add_2.GetOutputTensor(0).GetDims();
  add_2_output_dims[0] = 1;
  add_2_output_dims[1] = 1;
  auto& add_2_output = tensor_pool.CloneNativeTensorFrom(
      add_2.GetOutputTensor(0), add_2_output_dims);
  EmplaceOpWithIO(new_ops, add_2, {concat_output, std::nullopt},
                  {add_2_output});

  // Reshape 3
  auto reshape_3_output_dims = reshape_3.GetOutputTensor(0).GetDims();
  reshape_3_output_dims.erase(reshape_3_output_dims.begin() + 1);
  reshape_3_output_dims[1] /= num_attn_per_kv_heads;
  auto& reshape_3_output = tensor_pool.CloneNativeTensorFrom(
      reshape_3.GetOutputTensor(0), reshape_3_output_dims);
  EmplaceOpWithIO(new_ops, reshape_3, {add_2_output}, {reshape_3_output});

  // Softmax
  auto softmax_output_dims = softmax.GetOutputTensor(0).GetDims();
  softmax_output_dims.erase(softmax_output_dims.begin() + 1);
  softmax_output_dims[1] /= num_attn_per_kv_heads;
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), softmax_output_dims);
  EmplaceOpWithIO(new_ops, softmax, {reshape_3_output}, {softmax_output});

  // Slice 1
  auto slice_1_param = slice_1.GetTensorPararm(0).GetTensor();
  auto slice_1_param_data = slice_1_param.GetTensorData<int32_t>();
  std::vector<int32_t> slice_1_ranges(slice_1_param_data.value().begin(),
                                      slice_1_param_data.value().end());
  slice_1_ranges.erase(slice_1_ranges.begin() + 3, slice_1_ranges.begin() + 6);
  slice_1_ranges[4] /= num_attn_per_kv_heads;
  std::vector<uint32_t> slice_1_param_dims = {
      static_cast<uint32_t>(slice_1_ranges.size() / 3), 3};
  auto& slice_1_param_tensor = tensor_pool.CreateStaticTensor(
      slice_1_param.GetDataType(), slice_1_param.GetQuantParams(),
      slice_1_param_dims, sizeof(int32_t) * slice_1_ranges.size(),
      slice_1_ranges.data());
  auto slice_1_output_dims = slice_1.GetOutputTensor(0).GetDims();
  slice_1_output_dims.erase(slice_1_output_dims.begin() + 1);
  slice_1_output_dims[1] /= num_attn_per_kv_heads;
  auto& slice_1_output = tensor_pool.CloneNativeTensorFrom(
      slice_1.GetOutputTensor(0), slice_1_output_dims);
  BuildSliceOp(new_ops.emplace_back(), softmax_output, slice_1_output,
               slice_1_param_tensor);

  // Slice 2
  auto slice_2_param = slice_2.GetTensorPararm(0).GetTensor();
  auto slice_2_param_data = slice_2_param.GetTensorData<int32_t>();
  std::vector<int32_t> slice_2_ranges(slice_2_param_data.value().begin(),
                                      slice_2_param_data.value().end());
  slice_2_ranges.erase(slice_2_ranges.begin() + 3, slice_2_ranges.begin() + 6);
  slice_2_ranges[4] /= num_attn_per_kv_heads;
  std::vector<uint32_t> slice_2_param_dims = {
      static_cast<uint32_t>(slice_2_ranges.size() / 3), 3};
  auto& slice_2_param_tensor = tensor_pool.CreateStaticTensor(
      slice_2_param.GetDataType(), slice_2_param.GetQuantParams(),
      slice_2_param_dims, sizeof(int32_t) * slice_2_ranges.size(),
      slice_2_ranges.data());
  auto slice_2_output_dims = slice_2.GetOutputTensor(0).GetDims();
  slice_2_output_dims.erase(slice_2_output_dims.begin() + 1);
  slice_2_output_dims[1] /= num_attn_per_kv_heads;
  auto& slice_2_output = tensor_pool.CloneNativeTensorFrom(
      slice_2.GetOutputTensor(0), slice_2_output_dims);
  BuildSliceOp(new_ops.emplace_back(), softmax_output, slice_2_output,
               slice_2_param_tensor);

  // Matmul v1
  auto matmul_v1_output_dims = matmul_v1.GetOutputTensor(0).GetDims();
  matmul_v1_output_dims.erase(matmul_v1_output_dims.begin() + 1);
  matmul_v1_output_dims[1] /= num_attn_per_kv_heads;
  auto& matmul_v1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v1.GetOutputTensor(0), matmul_v1_output_dims);
  EmplaceOpWithIO(new_ops, matmul_v1, {slice_1_output, v_cache},
                  {matmul_v1_output});

  // Matmul v2
  auto matmul_v2_output_dims = matmul_v2.GetOutputTensor(0).GetDims();
  matmul_v2_output_dims.erase(matmul_v2_output_dims.begin() + 1);
  matmul_v2_output_dims[1] /= num_attn_per_kv_heads;
  auto& matmul_v2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v2.GetOutputTensor(0), matmul_v2_output_dims);
  EmplaceOpWithIO(new_ops, matmul_v2, {slice_2_output, v_slice},
                  {matmul_v2_output});

  // Add 3
  auto add_3_output_dims = add_3.GetOutputTensor(0).GetDims();
  add_3_output_dims.erase(add_3_output_dims.begin() + 1);
  add_3_output_dims[1] /= num_attn_per_kv_heads;
  auto& add_3_output = tensor_pool.CloneNativeTensorFrom(
      add_3.GetOutputTensor(0), add_3_output_dims);
  EmplaceOpWithIO(new_ops, add_3, {matmul_v1_output, matmul_v2_output},
                  {add_3_output});

  return add_3_output;
}

void CloneNamespace(const OpWrapper& source, std::vector<OpWrapper>& ops) {
  absl::string_view start_op_name = source.GetName();
  size_t pos = start_op_name.rfind('/');
  if (pos == absl::string_view::npos) {
    return;
  }
  for (auto& op : ops) {
    op.AddPrefixToName(absl::StrCat(start_op_name.substr(0, pos), "/"));
  }
}

std::vector<OpWrapper> TransformToSHA(
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    const qnn::TensorWrapper& mha_input, const qnn::TensorWrapper& mha_output,
    const ::qnn::OpWrapper& scaling_mul, size_t num_heads) {
  std::vector<OpWrapper> new_ops;

  // Prepare inputs for num_heads SHAs.
  std::vector<::qnn::TensorWrapperRef> sha_inputs;
  sha_inputs.reserve(num_heads);
  for (int i = 0; i < num_heads; ++i) {
    auto head_input_dims = ops[start_index].GetOutputTensor(0).GetDims();
    head_input_dims[2] /= num_heads;
    auto& split_output =
        tensor_pool.CloneNativeTensorFrom(mha_input, head_input_dims);
    sha_inputs.emplace_back(split_output);
  }
  // Split
  const std::array<int32_t, 1> split_axis_data{2};
  auto& split_axis = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {split_axis_data.size()},
      split_axis_data.size() * sizeof(split_axis_data[0]),
      split_axis_data.data());
  auto split = BuildSplitOp(
      tensor_pool, {split_axis, const_cast<::qnn::TensorWrapper&>(mha_input)},
      sha_inputs, num_heads);
  CloneNamespace(ops[start_index], split);
  std::move(split.begin(), split.end(), std::back_inserter(new_ops));
  // Prepare outputs for num_heads SHAs.
  std::vector<::qnn::TensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_heads);
  // Create num_heads SHA.
  for (int i = 0; i < num_heads; ++i) {
    OpWrapper& add_1 = ops[start_index + kAddIndex];
    TensorWrapper& mask = const_cast<TensorWrapper&>(add_1.GetInputTensor(1));
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
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(mha_output, concat_dims);
  auto concat_final =
      BuildConcatenationOp(tensor_pool, sha_outputs, {concat_output}, 3);
  CloneNamespace(ops[start_index], concat_final);
  std::move(concat_final.begin(), concat_final.end(),
            std::back_inserter(new_ops));
  // Reshape
  auto reshape =
      BuildReshapeOp(tensor_pool, {concat_output},
                     {const_cast<::qnn::TensorWrapper&>(mha_output)});
  CloneNamespace(ops[start_index], reshape);
  std::move(reshape.begin(), reshape.end(), std::back_inserter(new_ops));
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
  auto& transpose_output =
      tensor_pool.CloneNativeTensorFrom(pattern_input, transpose_output_dims);
  EmplaceOpWithIO(new_ops, ops[start_index + kTransposePrefillIndex],
                  {const_cast<::qnn::TensorWrapper&>(pattern_input)},
                  {transpose_output});

  // Reshape
  auto& reshape_output = tensor_pool.CloneNativeTensorFrom(
      pattern_input, {transpose_output_dims[0], 1,
                      transpose_output_dims[1] * transpose_output_dims[2],
                      transpose_output_dims[3]});
  EmplaceOpWithIO(new_ops, ops[start_index + kReshapePrefillIndex],
                  {transpose_output}, {reshape_output});

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
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
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
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
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
  constexpr size_t kMulIndex = 0;
  constexpr size_t kReshape1Index = 1;
  constexpr size_t kMatmulQIndex = 2;
  constexpr size_t kMatmulQkIndex = 3;
  constexpr size_t kConcatIndex = 4;
  constexpr size_t kReshape2Index = 5;
  constexpr size_t kAdd2Index = 6;
  constexpr size_t kReshape3Index = 7;
  constexpr size_t kSoftmaxIndex = 8;
  constexpr size_t kSlice1Index = 9;
  constexpr size_t kSlice2Index = 10;
  constexpr size_t kMatmulV1Index = 11;
  constexpr size_t kMatmulV2Index = 12;
  constexpr size_t kAdd3Index = 13;
  constexpr size_t kReshape4Index = 14;
  constexpr size_t kTransposeIndex = 15;
  constexpr size_t kReshape5Index = 16;

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

  const auto is_add_op_match_in_pattern =
      [&is_connected](int32_t first_add_index) {
        return is_connected(first_add_index, 0, kMatmulQkIndex, 1) &&
               is_connected(kMulIndex, 0, kReshape1Index, 0) &&
               is_connected(kReshape1Index, 0, kMatmulQIndex, 0) &&
               is_connected(kReshape1Index, 0, kMatmulQkIndex, 0) &&
               is_connected(kMatmulQIndex, 0, kConcatIndex, 0) &&
               is_connected(kMatmulQkIndex, 0, kConcatIndex, 1) &&
               is_connected(kConcatIndex, 0, kReshape2Index, 0) &&
               is_connected(kReshape2Index, 0, kAdd2Index, 0) &&
               is_connected(kAdd2Index, 0, kReshape3Index, 0) &&
               is_connected(kReshape3Index, 0, kSoftmaxIndex, 0) &&
               is_connected(kSoftmaxIndex, 0, kSlice1Index, 0) &&
               is_connected(kSoftmaxIndex, 0, kSlice2Index, 0) &&
               is_connected(kSlice1Index, 0, kMatmulV1Index, 0) &&
               is_connected(kSlice2Index, 0, kMatmulV2Index, 0) &&
               is_connected(kMatmulV1Index, 0, kAdd3Index, 0) &&
               is_connected(kMatmulV2Index, 0, kAdd3Index, 1) &&
               is_connected(kAdd3Index, 0, kReshape4Index, 0) &&
               is_connected(kReshape4Index, 0, kTransposeIndex, 0) &&
               is_connected(kTransposeIndex, 0, kReshape5Index, 0);
      };

  int32_t add_1_index = -2;
  if (ops[start_index + add_1_index].IsOpCode(QnnOpCode::kElementWiseBinary) &&
      is_add_op_match_in_pattern(add_1_index)) {
    // Origin pattern, add_1_index is matched, do nothing.
  } else if (ops[start_index + add_1_index + 1].IsOpCode(
                 QnnOpCode::kElementWiseBinary) &&
             is_add_op_match_in_pattern(add_1_index + 1)) {
    // For new Fast VLM pattern. Tranpose Ops are fused into MatMul Ops first by
    // OptimizeTransposeMatMul, the add_1_index becomes -1.
    add_1_index += 1;
  } else {
    return 1;
  }
  QNN_LOG_INFO("[G2G] MHA optimization (fast vlm Prefill)");

  // QKV Unpack
  const auto& matmul_v1_in =
      ops[start_index + kMatmulV1Index].GetInputTensor(1);
  auto unpack_1_dims =
      ops[start_index + kMatmulV1Index].GetInputTensor(1).GetDims();
  uint32_t num_kv_heads = unpack_1_dims[1];
  const auto& matmul_q_in = ops[start_index + kMatmulQIndex].GetInputTensor(1);
  auto unpack_2_dims = matmul_q_in.GetDims();
  const auto& mul_in = ops[start_index + kMulIndex].GetInputTensor(0);
  auto unpack_3_dims = mul_in.GetDims();
  uint32_t num_attn_heads = unpack_3_dims[1];
  uint32_t num_attn_per_kv_heads = num_attn_heads / num_kv_heads;
  const auto& add_1_in_1 = ops[start_index + add_1_index].GetInputTensor(0);
  auto unpack_4_dims = add_1_in_1.GetDims();
  const auto& add_1_in_2 = ops[start_index + add_1_index].GetInputTensor(1);
  auto unpack_5_dims = add_1_in_2.GetDims();
  const auto& matmul_v2_in =
      ops[start_index + kMatmulV2Index].GetInputTensor(1);
  auto unpack_6_dims = matmul_v2_in.GetDims();
  const auto& pattern_output =
      ops[start_index + pattern_size - 1].GetOutputTensor(0);
  auto mha_output_dims = pattern_output.GetDims();

  if (!(num_kv_heads == unpack_2_dims[1] && num_kv_heads == unpack_4_dims[1] &&
        num_kv_heads == unpack_5_dims[1] && num_kv_heads == unpack_6_dims[1])) {
    QNN_LOG_WARNING(
        "[G2G] num_kv heads: %d does not match heads in [unpack_2: %d, "
        "unpack_4: %d, unpack_5: %d, unpack_6: %d]",
        num_kv_heads, unpack_2_dims[1], unpack_4_dims[1], unpack_5_dims[1],
        unpack_6_dims[1]);
    return 1;
  }

  std::vector<OpWrapper> new_ops;
  std::vector<TensorWrapperRef> unpack_1_sha_inputs;
  std::vector<TensorWrapperRef> unpack_2_sha_inputs;
  std::vector<TensorWrapperRef> unpack_3_sha_inputs;
  std::vector<TensorWrapperRef> unpack_4_sha_inputs;
  std::vector<TensorWrapperRef> unpack_5_sha_inputs;
  std::vector<TensorWrapperRef> unpack_6_sha_inputs;

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
    auto& unpack_1 =
        tensor_pool.CloneNativeTensorFrom(matmul_v1_in, unpack_1_dims);
    unpack_1_sha_inputs.emplace_back(unpack_1);
    auto& unpack_2 =
        tensor_pool.CloneNativeTensorFrom(matmul_q_in, unpack_2_dims);
    unpack_2_sha_inputs.emplace_back(unpack_2);
    for (int j = 0; j < num_attn_per_kv_heads; j++) {
      auto& unpack_3 = tensor_pool.CloneNativeTensorFrom(mul_in, unpack_3_dims);
      unpack_3_sha_inputs.emplace_back(unpack_3);
    }
    auto& unpack_4 =
        tensor_pool.CloneNativeTensorFrom(add_1_in_1, unpack_4_dims);
    unpack_4_sha_inputs.emplace_back(unpack_4);
    auto& unpack_5 =
        tensor_pool.CloneNativeTensorFrom(add_1_in_2, unpack_5_dims);
    unpack_5_sha_inputs.emplace_back(unpack_5);
    auto& unpack_6 =
        tensor_pool.CloneNativeTensorFrom(matmul_v2_in, unpack_6_dims);
    unpack_6_sha_inputs.emplace_back(unpack_6);
  }

  // Unpack 1-5
  auto unpack_1_op = BuildUnpackOp(
      tensor_pool, {const_cast<::qnn::TensorWrapper&>(matmul_v1_in)},
      unpack_1_sha_inputs, 1);
  std::move(unpack_1_op.begin(), unpack_1_op.end(),
            std::back_inserter(new_ops));
  auto unpack_2_op = BuildUnpackOp(
      tensor_pool, {const_cast<::qnn::TensorWrapper&>(matmul_q_in)},
      unpack_2_sha_inputs, 1);
  std::move(unpack_2_op.begin(), unpack_2_op.end(),
            std::back_inserter(new_ops));
  auto unpack_3_op =
      BuildUnpackOp(tensor_pool, {const_cast<::qnn::TensorWrapper&>(mul_in)},
                    unpack_3_sha_inputs, 1);
  std::move(unpack_3_op.begin(), unpack_3_op.end(),
            std::back_inserter(new_ops));
  auto unpack_4_op = BuildUnpackOp(
      tensor_pool, {const_cast<::qnn::TensorWrapper&>(add_1_in_1)},
      unpack_4_sha_inputs, 1);
  std::move(unpack_4_op.begin(), unpack_4_op.end(),
            std::back_inserter(new_ops));
  auto unpack_5_op = BuildUnpackOp(
      tensor_pool, {const_cast<::qnn::TensorWrapper&>(add_1_in_2)},
      unpack_5_sha_inputs, 1);
  std::move(unpack_5_op.begin(), unpack_5_op.end(),
            std::back_inserter(new_ops));
  auto unpack_6_op = BuildUnpackOp(
      tensor_pool, {const_cast<::qnn::TensorWrapper&>(matmul_v2_in)},
      unpack_6_sha_inputs, 1);
  std::move(unpack_6_op.begin(), unpack_6_op.end(),
            std::back_inserter(new_ops));

  // build num_head SHAs
  std::vector<TensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_attn_heads);
  for (size_t i = 0; i < num_kv_heads; ++i) {
    for (size_t j = 0; j < num_attn_per_kv_heads; ++j) {
      auto& sha_output = BuildSingleSHA(
          new_ops, tensor_pool, unpack_1_sha_inputs[i], unpack_2_sha_inputs[i],
          unpack_3_sha_inputs[i * num_attn_per_kv_heads + j],
          unpack_4_sha_inputs[i], unpack_5_sha_inputs[i],
          unpack_6_sha_inputs[i], num_attn_per_kv_heads,
          ops[start_index + kMulIndex], ops[start_index + kMatmulQIndex],
          ops[start_index + add_1_index], ops[start_index + kMatmulQkIndex],
          ops[start_index + kConcatIndex], ops[start_index + kAdd2Index],
          ops[start_index + kReshape3Index], ops[start_index + kSoftmaxIndex],
          ops[start_index + kSlice1Index], ops[start_index + kSlice2Index],
          ops[start_index + kMatmulV1Index], ops[start_index + kMatmulV2Index],
          ops[start_index + kAdd3Index], nullptr);
      sha_outputs.emplace_back(sha_output);
    }
  }

  // Concat
  auto concat_op = BuildConcatenationOp(
      tensor_pool, sha_outputs,
      {const_cast<::qnn::TensorWrapper&>(pattern_output)}, 2);
  std::move(concat_op.begin(), concat_op.end(), std::back_inserter(new_ops));

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
    QNN_LOG_INFO("[G2G] FastVLM optimization done.");
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

std::vector<ConstTensorWrapperRef> SplitFullyConnected(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    const OpWrapper& fc, const OpWrapper& reshape, size_t num_splits) {
  constexpr std::int32_t kFilterAxis = 0;

  const auto& fc_input = fc.GetInputTensor(0);
  const auto& fc_output = fc.GetOutputTensor(0);
  const auto& filter = fc.GetInputTensor(1);  // rank = 2
  std::vector<float> filter_scales;
  std::vector<std::int32_t> filter_zero_points;
  // Currently only support per-axis quant;
  if (std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
          filter.GetQuantParams())) {
    const auto& filter_quant_param =
        std::get<BwAxisScaleOffsetQuantizeParamsWrapper>(
            filter.GetQuantParams());
    filter_quant_param.GetScales(filter_scales);
    filter_quant_param.GetZeroPoints(filter_zero_points);
  } else if (std::holds_alternative<BwAxisScaleOffsetQuantizeParamsWrapper>(
                 filter.GetQuantParams())) {
    const auto& filter_quant_param =
        std::get<BwAxisScaleOffsetQuantizeParamsWrapper>(
            filter.GetQuantParams());
    filter_quant_param.GetScales(filter_scales);
    filter_quant_param.GetZeroPoints(filter_zero_points);
  } else {
    QNN_LOG_ERROR("Unsupported quant param type when split FC: %d",
                  filter.GetQuantParams().index());
    return {};
  }
  // Assume the axis of axis quant is 0.
  if (filter_scales.size() != filter_zero_points.size() &&
      filter_scales.size() != filter.GetDim(kFilterAxis)) {
    QNN_LOG_ERROR("Filter dimension mismatched: %d %d %d", filter_scales.size(),
                  filter_zero_points.size(), filter.GetDim(kFilterAxis));
    return {};
  }

  // Currently only support int8 and int4;
  auto opt_filter_data = filter.GetTensorData<int8_t>();
  if (!opt_filter_data.has_value()) {
    return {};
  }
  auto* filter_data = opt_filter_data->data();

  const std::uint32_t split_size = filter_scales.size() / num_splits;
  std::vector<ConstTensorWrapperRef> reshape_outputs;
  for (size_t i = 0; i < num_splits; ++i) {
    TensorWrapper* new_filter = nullptr;
    if (std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
            filter.GetQuantParams())) {
      AxisScaleOffsetQuantizeParamsWrapper new_filter_quant_param(
          kFilterAxis,
          absl::MakeConstSpan(filter_scales.data() + i * split_size,
                              filter_scales.data() + (i + 1) * split_size),
          absl::MakeConstSpan(
              filter_zero_points.data() + i * split_size,
              filter_zero_points.data() + (i + 1) * split_size));
      new_filter = &(tensor_pool.CreateStaticTensor(
          filter.GetDataType(), new_filter_quant_param,
          {split_size, filter.GetDim(1)}, filter.GetTensorBytes() / num_splits,
          filter_data + i * split_size));
    } else if (std::holds_alternative<BwAxisScaleOffsetQuantizeParamsWrapper>(
                   filter.GetQuantParams())) {
      BwAxisScaleOffsetQuantizeParamsWrapper new_filter_quant_param(
          std::get<BwAxisScaleOffsetQuantizeParamsWrapper>(
              filter.GetQuantParams())
              .GetBitwidth(),
          kFilterAxis,
          absl::MakeConstSpan(filter_scales.data() + i * split_size,
                              filter_scales.data() + (i + 1) * split_size),
          absl::MakeConstSpan(
              filter_zero_points.data() + i * split_size,
              filter_zero_points.data() + (i + 1) * split_size));
      new_filter = &(tensor_pool.CreateStaticTensor(
          filter.GetDataType(), new_filter_quant_param,
          {split_size, filter.GetDim(1)}, filter.GetTensorBytes() / num_splits,
          filter_data + i * split_size));
    }

    std::vector<std::uint32_t> new_fc_output_dims{
        fc_input.GetTensorNumElements() / filter.GetDim(1), split_size};
    const auto& new_fc_output =
        tensor_pool.CloneNativeTensorFrom(fc_output, new_fc_output_dims);
    new_ops.emplace_back(
        CreateFullyConnectedOp(fc_input, new_fc_output, *new_filter));

    // Reshape to keep dims
    auto reshape_dims = reshape.GetOutputTensor(0).GetDims();
    reshape_dims.back() = split_size;
    const auto& reshape_output = tensor_pool.CloneNativeTensorFrom(
        reshape.GetOutputTensor(0), reshape_dims);
    new_ops.emplace_back(CreateReshapeOp(new_fc_output, reshape_output));

    reshape_outputs.emplace_back(reshape_output);
  }
  return reshape_outputs;
}

std::vector<ConstTensorWrapperRef> SplitRoPE(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    const std::vector<ConstTensorWrapperRef>& inputs, const OpWrapper& mul_cos,
    const OpWrapper& mul_const, const OpWrapper& quantize,
    const OpWrapper& concat, const OpWrapper& mul_sin, const OpWrapper& add) {
  const auto input_dims = inputs[0].get().GetDims();
  // Reshape cos
  const auto& reshape_cos_output =
      tensor_pool.CloneNativeTensorFrom(mul_cos.GetInputTensor(1), input_dims);
  new_ops.emplace_back(
      CreateReshapeOp(mul_cos.GetInputTensor(1), reshape_cos_output));
  // Mul cos
  std::vector<ConstTensorWrapperRef> mul_cos_outputs;
  for (auto it = inputs.begin(); it != inputs.end(); ++it) {
    const auto& input = (*it).get();
    auto& mul_cos_output = tensor_pool.CloneNativeTensorFrom(
        mul_cos.GetOutputTensor(0), input_dims);
    mul_cos_outputs.emplace_back(mul_cos_output);
    new_ops.emplace_back(
        CreateElementWiseMulOp(input, reshape_cos_output, mul_cos_output));
  }
  // Reshape sin
  const auto& reshape_sin_output =
      tensor_pool.CloneNativeTensorFrom(mul_sin.GetInputTensor(1), input_dims);
  new_ops.emplace_back(
      CreateReshapeOp(mul_sin.GetInputTensor(1), reshape_sin_output));
  // Mul sin
  std::vector<ConstTensorWrapperRef> mul_sin_outputs;
  for (auto it = inputs.begin(); it != inputs.end(); ++it) {
    const auto& input = (*it).get();
    // Split
    const auto split_axis = input.GetRank() - 1;
    const size_t num_splits = 2;
    auto split_input_dims = input_dims;
    split_input_dims.back() /= num_splits;
    std::vector<ConstTensorWrapperRef> splitted_inputs;
    for (size_t i = 0; i < num_splits; ++i) {
      splitted_inputs.emplace_back(
          tensor_pool.CloneNativeTensorFrom(input, split_input_dims));
    }
    new_ops.emplace_back(
        CreateSplitOp(tensor_pool, input, splitted_inputs, split_axis));
    // Quantize (Convert)
    const auto& quantize_output = tensor_pool.CloneNativeTensorFrom(
        quantize.GetOutputTensor(0), split_input_dims);
    new_ops.emplace_back(CreateConvertOp(splitted_inputs[0], quantize_output));
    // Mul const
    const auto& mul_const_output = tensor_pool.CloneNativeTensorFrom(
        mul_const.GetOutputTensor(0), split_input_dims);
    new_ops.emplace_back(CreateElementWiseMulOp(
        splitted_inputs[0], mul_const.GetInputTensor(1), mul_const_output));
    // Concat
    const auto& concat_output = tensor_pool.CloneNativeTensorFrom(
        concat.GetOutputTensor(0), input_dims);
    new_ops.emplace_back(CreateConcatenationOp(
        {mul_const_output, quantize_output}, concat_output, split_axis));
    // Mul sin
    const auto& mul_sin_output = tensor_pool.CloneNativeTensorFrom(
        mul_sin.GetOutputTensor(0), input_dims);
    mul_sin_outputs.emplace_back(mul_sin_output);
    new_ops.emplace_back(CreateElementWiseAddOp(
        concat_output, reshape_sin_output, mul_sin_output));
  }
  // Add
  std::vector<ConstTensorWrapperRef> add_outputs;
  for (size_t i = 0; i < mul_cos_outputs.size(); ++i) {
    const auto& add_output =
        tensor_pool.CloneNativeTensorFrom(add.GetOutputTensor(0), input_dims);
    add_outputs.emplace_back(add_output);
    new_ops.emplace_back(CreateElementWiseAddOp(
        mul_cos_outputs[i], mul_sin_outputs[i], add_output));
  }
  return add_outputs;
}

size_t ExtremelyOptimizeMHAFastVlmDecode(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  QNN_LOG_INFO("[G2G] MHA optimization (fast vlm decode extremely)");
  std::vector<OpWrapper> new_ops;
  constexpr size_t kSupportedRank = 4;
  constexpr size_t kUnpackAxis = 1;

  constexpr size_t kFC0Index = 0;
  constexpr size_t kFCReshape0Index = 1;
  constexpr size_t kFCReshape1Index = 2;
  const auto& fc_0 = ops[start_index + kFC0Index];
  const auto& fc_reshape_0 = ops[start_index + kFCReshape0Index];
  const auto& fc_reshape_1 = ops[start_index + kFCReshape1Index];
  auto fc_0_outputs =
      SplitFullyConnected(new_ops, tensor_pool, fc_0, fc_reshape_0,
                          fc_reshape_1.GetOutputTensor(0).GetDim(kUnpackAxis));

  constexpr size_t kFC1Index = 3;
  constexpr size_t kFCReshape2Index = 4;
  constexpr size_t kFCReshape3Index = 5;
  const auto& fc_1 = ops[start_index + kFC1Index];
  const auto& fc_reshape_2 = ops[start_index + kFCReshape2Index];
  const auto& fc_reshape_3 = ops[start_index + kFCReshape3Index];
  auto fc_1_outputs =
      SplitFullyConnected(new_ops, tensor_pool, fc_1, fc_reshape_2,
                          fc_reshape_3.GetOutputTensor(0).GetDim(kUnpackAxis));

  constexpr size_t kFC2Index = 6;
  constexpr size_t kFCReshape4Index = 7;
  constexpr size_t kFCReshape5Index = 8;
  const auto& fc_2 = ops[start_index + kFC2Index];
  const auto& fc_reshape_4 = ops[start_index + kFCReshape4Index];
  const auto& fc_reshape_5 = ops[start_index + kFCReshape5Index];
  auto fc_2_outputs =
      SplitFullyConnected(new_ops, tensor_pool, fc_2, fc_reshape_4,
                          fc_reshape_5.GetOutputTensor(0).GetDim(kUnpackAxis));

  // Pack fc_2_outpus for graph output.
  new_ops.emplace_back(
      CreatePackOp(fc_2_outputs, fc_reshape_5.GetOutputTensor(0), kUnpackAxis));

  constexpr size_t kRoPE0MulCosIndex = 9;
  constexpr size_t kRoPE0MulConstIndex = 12;
  constexpr size_t kRoPE0QauntizeIndex = 13;
  constexpr size_t kRoPE0ConcatIndex = 14;
  constexpr size_t kRoPE0MulSinIndex = 15;
  constexpr size_t kRoPE0AddIndex = 16;
  const auto& rope_0_mul_cos = ops[start_index + kRoPE0MulCosIndex];
  const auto& rope_0_mul_const = ops[start_index + kRoPE0MulConstIndex];
  const auto& rope_0_quantize = ops[start_index + kRoPE0QauntizeIndex];
  const auto& rope_0_concat = ops[start_index + kRoPE0ConcatIndex];
  const auto& rope_0_mul_sin = ops[start_index + kRoPE0MulSinIndex];
  const auto& rope_0_add = ops[start_index + kRoPE0AddIndex];
  auto rope_0_outputs = SplitRoPE(
      new_ops, tensor_pool, fc_0_outputs, rope_0_mul_cos, rope_0_mul_const,
      rope_0_quantize, rope_0_concat, rope_0_mul_sin, rope_0_add);

  constexpr size_t kRoPE1MulCosIndex = 17;
  constexpr size_t kRoPE1MulConstIndex = 20;
  constexpr size_t kRoPE1QauntizeIndex = 21;
  constexpr size_t kRoPE1ConcatIndex = 22;
  constexpr size_t kRoPE1MulSinIndex = 23;
  constexpr size_t kRoPE1AddIndex = 24;
  const auto& rope_1_mul_cos = ops[start_index + kRoPE1MulCosIndex];
  const auto& rope_1_mul_const = ops[start_index + kRoPE1MulConstIndex];
  const auto& rope_1_quantize = ops[start_index + kRoPE1QauntizeIndex];
  const auto& rope_1_concat = ops[start_index + kRoPE1ConcatIndex];
  const auto& rope_1_mul_sin = ops[start_index + kRoPE1MulSinIndex];
  const auto& rope_1_add = ops[start_index + kRoPE1AddIndex];
  auto rope_1_outputs = SplitRoPE(
      new_ops, tensor_pool, fc_1_outputs, rope_1_mul_cos, rope_1_mul_const,
      rope_1_quantize, rope_1_concat, rope_1_mul_sin, rope_1_add);

  // Use reshape to transpose rope_1_outputs;
  constexpr size_t kRoPE1ReshapeIndex = 25;
  const auto& rope_1_reshape = ops[start_index + kRoPE1ReshapeIndex];
  auto repo_1_reshape_output_dims = rope_1_reshape.GetOutputTensor(0).GetDims();
  repo_1_reshape_output_dims.erase(repo_1_reshape_output_dims.begin() +
                                   kUnpackAxis);
  std::vector<ConstTensorWrapperRef> transposed_rope_1_outputs;
  for (const auto& rope_1_output : rope_1_outputs) {
    const auto& transposed_rope_1_output = tensor_pool.CloneNativeTensorFrom(
        rope_1_reshape.GetOutputTensor(0), repo_1_reshape_output_dims);
    transposed_rope_1_outputs.emplace_back(transposed_rope_1_output);
    new_ops.emplace_back(
        CreateReshapeOp(rope_1_output.get(), transposed_rope_1_output));
  }

  // Pack transposed_rope_1_outputs for graph output.
  new_ops.emplace_back(CreatePackOp(transposed_rope_1_outputs,
                                    rope_1_reshape.GetOutputTensor(0),
                                    kUnpackAxis));

  constexpr size_t kMulIndex = 26;
  constexpr size_t kReshape0Index = 27;
  constexpr size_t kMatmulK0Index = 28;
  constexpr size_t kMatmulK1Index = 29;
  constexpr size_t kConcatIndex = 30;
  constexpr size_t kReshape1Index = 31;
  constexpr size_t kAdd0Index = 32;
  constexpr size_t kReshape2Index = 33;
  constexpr size_t kSoftmaxIndex = 34;
  constexpr size_t kSlice0Index = 35;
  constexpr size_t kSlice1Index = 36;
  constexpr size_t kMatmulV0Index = 37;
  constexpr size_t kMatmulV1Index = 38;
  constexpr size_t kAdd1Index = 39;
  constexpr size_t kReshape3Index = 40;
  const auto& mul = ops[start_index + kMulIndex];
  const auto& reshape_0 = ops[start_index + kReshape0Index];
  const auto& matmul_k0 = ops[start_index + kMatmulK0Index];
  const auto& matmul_k1 = ops[start_index + kMatmulK1Index];
  const auto& concat = ops[start_index + kConcatIndex];
  const auto& reshape_1 = ops[start_index + kReshape1Index];
  const auto& add_0 = ops[start_index + kAdd0Index];
  const auto& reshape_2 = ops[start_index + kReshape2Index];
  const auto& softmax = ops[start_index + kSoftmaxIndex];
  const auto& slice_0 = ops[start_index + kSlice0Index];
  const auto& slice_1 = ops[start_index + kSlice1Index];
  const auto& matmul_v0 = ops[start_index + kMatmulV0Index];
  const auto& matmul_v1 = ops[start_index + kMatmulV1Index];
  const auto& add_1 = ops[start_index + kAdd1Index];
  const auto& reshape_3 = ops[start_index + kReshape3Index];

  // Unpack Cache K input
  auto unpack_cache_k_dims = matmul_k0.GetInputTensor(1).GetDims();
  const size_t num_unpack_cache_k_output = unpack_cache_k_dims[kUnpackAxis];
  unpack_cache_k_dims.erase(unpack_cache_k_dims.begin() + kUnpackAxis);
  std::vector<ConstTensorWrapperRef> unpack_cache_k_outputs;
  unpack_cache_k_outputs.reserve(num_unpack_cache_k_output);
  for (size_t i = 0; i < num_unpack_cache_k_output; ++i) {
    unpack_cache_k_outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(
        matmul_k0.GetInputTensor(1), unpack_cache_k_dims));
  }
  new_ops.emplace_back(CreateUnpackOp(matmul_k0.GetInputTensor(1),
                                      unpack_cache_k_outputs, kUnpackAxis));

  // Unpack Cache V input
  auto unpack_cache_v_dims = matmul_v0.GetInputTensor(1).GetDims();
  const size_t num_unpack_cache_v_output = unpack_cache_v_dims[kUnpackAxis];
  unpack_cache_v_dims.erase(unpack_cache_v_dims.begin() + kUnpackAxis);
  std::vector<ConstTensorWrapperRef> unpack_cache_v_outputs;
  for (size_t i = 0; i < num_unpack_cache_v_output; ++i) {
    unpack_cache_v_outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(
        matmul_v0.GetInputTensor(1), unpack_cache_v_dims));
  }
  new_ops.emplace_back(CreateUnpackOp(matmul_v0.GetInputTensor(1),
                                      unpack_cache_v_outputs, kUnpackAxis));

  // Build SHA
  const auto num_attn_head = rope_0_outputs.size();
  const auto num_kv_head = transposed_rope_1_outputs.size();
  const auto num_attn_per_kv_head = num_attn_head / num_kv_head;
  std::vector<ConstTensorWrapperRef> sha_outputs;
  for (size_t i = 0; i < num_kv_head; ++i) {
    for (size_t j = 0; j < num_attn_per_kv_head; ++j) {
      const auto& sha_output = BuildSingleSHA(
          new_ops, tensor_pool,
          const_cast<TensorWrapper&>(unpack_cache_v_outputs[i].get()),
          const_cast<TensorWrapper&>(unpack_cache_k_outputs[i].get()),
          const_cast<TensorWrapper&>(
              rope_0_outputs[i * num_attn_per_kv_head + j].get()),
          const_cast<TensorWrapper&>(add_0.GetInputTensor(0)),
          const_cast<TensorWrapper&>(add_0.GetInputTensor(1)),
          const_cast<TensorWrapper&>(fc_2_outputs[i].get()),
          num_attn_per_kv_head, mul, matmul_k0, add_0, matmul_k1, concat, add_0,
          reshape_2, softmax, slice_0, slice_1, matmul_v0, matmul_v1, add_1,
          &(const_cast<TensorWrapper&>(transposed_rope_1_outputs[i].get())));
      sha_outputs.emplace_back(sha_output);
    }
  }

  // Concat SHA outputs by last dimension
  const auto concat_axis = sha_outputs[0].get().GetRank() - 1;
  auto concat_sha_dims = sha_outputs[0].get().GetDims();
  concat_sha_dims[concat_axis] = 0;
  for (const auto& sha_output : sha_outputs) {
    concat_sha_dims[concat_axis] += sha_output.get().GetDim(concat_axis);
  }
  const auto& concat_sha_output = tensor_pool.CloneNativeTensorFrom(
      reshape_3.GetInputTensor(0), concat_sha_dims);
  new_ops.emplace_back(
      CreateConcatenationOp(sha_outputs, concat_sha_output, concat_axis));
  new_ops.emplace_back(
      CreateReshapeOp(concat_sha_output, reshape_3.GetOutputTensor(0)));

  // Validate new graph.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                    if (op_wrapper.GetOpCode() == QnnOpCode::kPack) {
                      return true;
                    }
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
    QNN_LOG_INFO("[G2G] Extremly FastVLM decode optimization done.");
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

  constexpr size_t kMulIndex = 0;
  constexpr size_t kReshape0Index = 1;
  constexpr size_t kMatmulK0Index = 2;
  constexpr size_t kMatmulK1Index = 3;
  constexpr size_t kConcatIndex = 4;
  constexpr size_t kReshape1Index = 5;
  constexpr size_t kAdd0Index = 6;
  constexpr size_t kReshape2Index = 7;
  constexpr size_t kSoftmaxIndex = 8;
  constexpr size_t kSlice0Index = 9;
  constexpr size_t kSlice1Index = 10;
  constexpr size_t kMatmulV0Index = 11;
  constexpr size_t kMatmulV1Index = 12;
  constexpr size_t kAdd1Index = 13;
  constexpr size_t kReshape3Index = 14;
  const auto& mul = ops[start_index + kMulIndex];
  const auto& reshape_0 = ops[start_index + kReshape0Index];
  const auto& matmul_k0 = ops[start_index + kMatmulK0Index];
  const auto& matmul_k1 = ops[start_index + kMatmulK1Index];
  const auto& concat = ops[start_index + kConcatIndex];
  const auto& reshape_1 = ops[start_index + kReshape1Index];
  const auto& add_0 = ops[start_index + kAdd0Index];
  const auto& reshape_2 = ops[start_index + kReshape2Index];
  const auto& softmax = ops[start_index + kSoftmaxIndex];
  const auto& slice_0 = ops[start_index + kSlice0Index];
  const auto& slice_1 = ops[start_index + kSlice1Index];
  const auto& matmul_v0 = ops[start_index + kMatmulV0Index];
  const auto& matmul_v1 = ops[start_index + kMatmulV1Index];
  const auto& add_1 = ops[start_index + kAdd1Index];
  const auto& reshape_3 = ops[start_index + kReshape3Index];

  const auto is_connected =
      [](const OpWrapper& output, size_t output_tensor_index,
         const OpWrapper& input, size_t input_tensor_index) -> bool {
    return output.GetOutputTensor(output_tensor_index) ==
           input.GetInputTensor(input_tensor_index);
  };
  if (!(is_connected(mul, 0, reshape_0, 0) &&
        is_connected(reshape_0, 0, matmul_k0, 0) &&
        is_connected(reshape_0, 0, matmul_k1, 0) &&
        is_connected(matmul_k0, 0, concat, 0) &&
        is_connected(matmul_k1, 0, concat, 1) &&
        is_connected(concat, 0, reshape_1, 0) &&
        is_connected(reshape_1, 0, add_0, 0) &&
        is_connected(add_0, 0, reshape_2, 0) &&
        is_connected(reshape_2, 0, softmax, 0) &&
        is_connected(softmax, 0, slice_0, 0) &&
        is_connected(softmax, 0, slice_1, 0) &&
        is_connected(slice_0, 0, matmul_v0, 0) &&
        is_connected(slice_1, 0, matmul_v1, 0) &&
        is_connected(matmul_v0, 0, add_1, 0) &&
        is_connected(matmul_v1, 0, add_1, 1) &&
        is_connected(add_1, 0, reshape_3, 0))) {
    QNN_LOG_WARNING(
        "[G2G] Failed to check connectivity when doing MHA-SHA transformation "
        "for FastVLM decode.");
    return 1;
  }

  constexpr size_t kSupportedRank = 4;
  constexpr size_t kUnpackAxis = 1;
  if (mul.GetInputTensor(0).GetRank() != kSupportedRank ||
      matmul_k0.GetInputTensor(1).GetRank() != kSupportedRank ||
      matmul_k1.GetInputTensor(1).GetRank() != kSupportedRank ||
      matmul_v0.GetInputTensor(1).GetRank() != kSupportedRank ||
      matmul_v1.GetInputTensor(1).GetRank() != kSupportedRank ||
      add_0.GetInputTensor(1).GetRank() != kSupportedRank ||
      add_0.GetInputTensor(1).GetDim(0) != 1 ||
      add_0.GetInputTensor(1).GetDim(1) != 1) {
    QNN_LOG_WARNING(
        "[G2G] Failed to check dimensions when doing MHA-SHA transformation "
        "for FastVLM decode.");
    return 1;
  }

  std::vector<OpWrapper> new_ops;

  // Unpack ROPE Q output
  auto unpack_rope_q_dims = mul.GetInputTensor(0).GetDims();
  const size_t num_unpack_rope_q_output = unpack_rope_q_dims[kUnpackAxis];
  unpack_rope_q_dims.erase(unpack_rope_q_dims.begin() + kUnpackAxis);
  std::vector<ConstTensorWrapperRef> unpack_rope_q_outputs;
  unpack_rope_q_outputs.reserve(num_unpack_rope_q_output);
  for (size_t i = 0; i < num_unpack_rope_q_output; ++i) {
    unpack_rope_q_outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(
        mul.GetInputTensor(0), unpack_rope_q_dims));
  }
  new_ops.emplace_back(CreateUnpackOp(mul.GetInputTensor(0),
                                      unpack_rope_q_outputs, kUnpackAxis));

  // Unpack ROPE K output
  auto unpack_rope_k_dims = matmul_k1.GetInputTensor(1).GetDims();
  const size_t num_unpack_rope_k_output = unpack_rope_k_dims[kUnpackAxis];
  unpack_rope_k_dims.erase(unpack_rope_k_dims.begin() + kUnpackAxis);
  std::vector<ConstTensorWrapperRef> unpack_rope_k_outputs;
  unpack_rope_k_outputs.reserve(num_unpack_rope_k_output);
  for (size_t i = 0; i < num_unpack_rope_k_output; ++i) {
    unpack_rope_k_outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(
        matmul_k1.GetInputTensor(1), unpack_rope_k_dims));
  }
  new_ops.emplace_back(CreateUnpackOp(matmul_k1.GetInputTensor(1),
                                      unpack_rope_k_outputs, kUnpackAxis));

  // Unpack Cache K input
  auto unpack_cache_k_dims = matmul_k0.GetInputTensor(1).GetDims();
  const size_t num_unpack_cache_k_output = unpack_cache_k_dims[kUnpackAxis];
  unpack_cache_k_dims.erase(unpack_cache_k_dims.begin() + kUnpackAxis);
  std::vector<ConstTensorWrapperRef> unpack_cache_k_outputs;
  unpack_cache_k_outputs.reserve(num_unpack_cache_k_output);
  for (size_t i = 0; i < num_unpack_cache_k_output; ++i) {
    unpack_cache_k_outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(
        matmul_k0.GetInputTensor(1), unpack_cache_k_dims));
  }
  new_ops.emplace_back(CreateUnpackOp(matmul_k0.GetInputTensor(1),
                                      unpack_cache_k_outputs, kUnpackAxis));

  // Unpack Cache V input
  auto unpack_cache_v_dims = matmul_v0.GetInputTensor(1).GetDims();
  const size_t num_unpack_cache_v_output = unpack_cache_v_dims[kUnpackAxis];
  unpack_cache_v_dims.erase(unpack_cache_v_dims.begin() + kUnpackAxis);
  std::vector<ConstTensorWrapperRef> unpack_cache_v_outputs;
  for (size_t i = 0; i < num_unpack_cache_v_output; ++i) {
    unpack_cache_v_outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(
        matmul_v0.GetInputTensor(1), unpack_cache_v_dims));
  }
  new_ops.emplace_back(CreateUnpackOp(matmul_v0.GetInputTensor(1),
                                      unpack_cache_v_outputs, kUnpackAxis));

  // Unpack Slice V input
  auto unpack_slice_v_dims = matmul_v1.GetInputTensor(1).GetDims();
  const size_t num_unpack_slice_v_output = unpack_slice_v_dims[kUnpackAxis];
  unpack_slice_v_dims.erase(unpack_slice_v_dims.begin() + kUnpackAxis);
  std::vector<ConstTensorWrapperRef> unpack_slice_v_outputs;
  for (size_t i = 0; i < num_unpack_slice_v_output; ++i) {
    unpack_slice_v_outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(
        matmul_v1.GetInputTensor(1), unpack_slice_v_dims));
  }
  new_ops.emplace_back(CreateUnpackOp(matmul_v1.GetInputTensor(1),
                                      unpack_slice_v_outputs, kUnpackAxis));

  // Build SHA
  const auto num_attn_head = unpack_rope_q_outputs.size();
  const auto num_kv_head = unpack_rope_k_outputs.size();
  const auto num_attn_per_kv_head = num_attn_head / num_kv_head;
  std::vector<ConstTensorWrapperRef> sha_outputs;
  for (size_t i = 0; i < num_kv_head; ++i) {
    for (size_t j = 0; j < num_attn_per_kv_head; ++j) {
      const auto& sha_output = BuildSingleSHA(
          new_ops, tensor_pool,
          const_cast<TensorWrapper&>(unpack_cache_v_outputs[i].get()),
          const_cast<TensorWrapper&>(unpack_cache_k_outputs[i].get()),
          const_cast<TensorWrapper&>(
              unpack_rope_q_outputs[i * num_attn_per_kv_head + j].get()),
          const_cast<TensorWrapper&>(add_0.GetInputTensor(0)),
          const_cast<TensorWrapper&>(add_0.GetInputTensor(1)),
          const_cast<TensorWrapper&>(unpack_slice_v_outputs[i].get()),
          num_attn_per_kv_head, mul, matmul_k0, add_0, matmul_k1, concat, add_0,
          reshape_2, softmax, slice_0, slice_1, matmul_v0, matmul_v1, add_1,
          &(const_cast<TensorWrapper&>(unpack_rope_k_outputs[i].get())));
      sha_outputs.emplace_back(sha_output);
    }
  }

  // Concat SHA outputs by last dimension
  const auto concat_axis = sha_outputs[0].get().GetRank() - 1;
  auto concat_sha_dims = sha_outputs[0].get().GetDims();
  concat_sha_dims[concat_axis] = 0;
  for (const auto& sha_output : sha_outputs) {
    concat_sha_dims[concat_axis] += sha_output.get().GetDim(concat_axis);
  }
  const auto& concat_sha_output = tensor_pool.CloneNativeTensorFrom(
      reshape_3.GetInputTensor(0), concat_sha_dims);
  new_ops.emplace_back(
      CreateConcatenationOp(sha_outputs, concat_sha_output, concat_axis));
  new_ops.emplace_back(
      CreateReshapeOp(concat_sha_output, reshape_3.GetOutputTensor(0)));

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
  auto transpose_output_dims = transpoe_0.GetOutputTensor(0).GetDims();
  auto& transpose_output =
      tensor_pool.CloneNativeTensorFrom(pattern_input, transpose_output_dims);
  auto& new_transpose_0 = EmplaceOpWithIO(
      new_ops, transpoe_0, {const_cast<TensorWrapper&>(pattern_input)},
      {transpose_output});

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDim(2);
  const auto& mha_input = new_transpose_0.GetOutputTensor(0);

  // Prepare inputs for num_heads SHAs.
  std::vector<TensorWrapperRef> sha_inputs;
  sha_inputs.reserve(num_heads);
  auto sha_input_dims = new_transpose_0.GetOutputTensor(0).GetDims();
  sha_input_dims[1] /= num_heads;
  for (size_t i = 0; i < num_heads; ++i) {
    auto& sha_input =
        tensor_pool.CloneNativeTensorFrom(mha_input, sha_input_dims);
    sha_inputs.emplace_back(sha_input);
  }

  // Split
  const std::array<int32_t, 1> split_axis_data{1};
  auto& split_axis = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {split_axis_data.size()},
      split_axis_data.size() * sizeof(decltype(split_axis_data)::value_type),
      split_axis_data.data());
  auto split_state_ops = BuildSplitOp(
      tensor_pool, {split_axis, const_cast<TensorWrapper&>(mha_input)},
      sha_inputs, num_heads);
  CloneNamespace(mul, split_state_ops);
  std::move(split_state_ops.begin(), split_state_ops.end(),
            std::back_inserter(new_ops));

  // Split Mask for Add
  std::vector<TensorWrapperRef> splited_masks;
  splited_masks.reserve(num_heads);
  const auto& concated_mask = add_0.GetInputTensor(1);
  auto splited_mask_dims = concated_mask.GetDims();
  splited_mask_dims[2] /= num_heads;
  for (size_t i = 0; i < num_heads; ++i) {
    splited_masks.emplace_back(
        tensor_pool.CloneNativeTensorFrom(concated_mask, splited_mask_dims));
  }
  auto split_masks_ops = BuildSplitOp(
      tensor_pool, {split_axis, const_cast<TensorWrapper&>(concated_mask)},
      splited_masks, num_heads);
  std::move(split_masks_ops.begin(), split_masks_ops.end(),
            std::back_inserter(new_ops));

  // build num_head SHAs
  std::vector<TensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_heads);
  for (size_t i = 0; i < num_heads; ++i) {
    auto& sha_output =
        BuildSingleSHA(new_ops, tensor_pool, sha_inputs[i], splited_masks[i],
                       num_heads, mul, matmul_k0, matmul_k1, concat, add_0,
                       softmax, slice_0, slice_1, matmul_v0, matmul_v1, add_1);
    sha_outputs.emplace_back(sha_output);
  }

  // Concat
  auto concat_output_dims = sha_outputs[0].get().GetDims();
  concat_output_dims[3] *= num_heads;
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(sha_outputs[0], concat_output_dims);
  auto concat_sha_output_ops =
      BuildConcatenationOp(tensor_pool, sha_outputs, {concat_output}, 3);
  CloneNamespace(mul, concat_sha_output_ops);
  std::move(concat_sha_output_ops.begin(), concat_sha_output_ops.end(),
            std::back_inserter(new_ops));

  // Reshape
  auto new_reshape_ops =
      BuildReshapeOp(tensor_pool, {concat_output},
                     {const_cast<TensorWrapper&>(pattern_output)});
  CloneNamespace(mul, new_reshape_ops);
  std::move(new_reshape_ops.begin(), new_reshape_ops.end(),
            std::back_inserter(new_ops));
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
  auto& select_mask =
      tensor_pool.CloneNativeTensorFrom(not_equal_out, not_equal_out_dims);
  // Change NotEqual to Equal -> Cast -> Mul.
  const auto& zero_tensor = attn_not_equal_op.GetInputTensor(1);
  auto equal_op =
      BuildElementwiseEqualOp(tensor_pool,
                              {const_cast<::qnn::TensorWrapper&>(reshape_in),
                               const_cast<::qnn::TensorWrapper&>(zero_tensor)},
                              {select_mask});
  std::move(equal_op.begin(), equal_op.end(), std::back_inserter(new_ops));
  const auto& select_out =
      ops[attn_start_index + kAttnSelect].GetOutputTensor(0);
  auto select_out_dims = select_out.GetDims();
  select_out_dims.erase(select_out_dims.begin() + 1);

  auto& mul_in = tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
  auto cast_select = BuildCastOp(tensor_pool, {select_mask}, {mul_in});
  std::move(cast_select.begin(), cast_select.end(),
            std::back_inserter(new_ops));

  const auto& select_const =
      ops[attn_start_index + kAttnSelect].GetInputTensor(2);
  // TODO(jiunkaiy): Remove this magic number (-65472) after HTP resolves
  // accuracy issues.
  float mul_const_value =
      std::max(select_const.GetTensorData<float>().value()[0], -65472.f);
  auto& mul_const = tensor_pool.CreateStaticTensor(
      select_const.GetDataType(), select_const.GetQuantParams(),
      select_const.GetDims(), select_const.GetTensorBytes(), &mul_const_value);
  auto& add_in = tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
  auto mul_select =
      BuildElementwiseMulOp(tensor_pool, {mul_in, mul_const}, {add_in});
  std::move(mul_select.begin(), mul_select.end(), std::back_inserter(new_ops));

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
    auto perm_tensor = tensor_pool.CreateStaticTensor(
        transpose_v_perm.GetDataType(), transpose_v_perm.GetQuantParams(), {3},
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
    std::vector<::qnn::TensorWrapperRef> q_sha_inputs;
    std::vector<::qnn::TensorWrapperRef> k_sha_inputs;
    std::vector<::qnn::TensorWrapperRef> v_sha_inputs;
    std::vector<::qnn::TensorWrapperRef> sha_outputs;
    q_sha_inputs.reserve(num_heads);
    k_sha_inputs.reserve(num_heads);
    v_sha_inputs.reserve(num_heads);
    sha_outputs.reserve(num_heads);

    for (int i = 0; i < num_heads; ++i) {
      auto& q_unpack =
          tensor_pool.CloneNativeTensorFrom(mul_q_in, q_unpack_dims);
      q_sha_inputs.emplace_back(q_unpack);

      auto& k_unpack =
          tensor_pool.CloneNativeTensorFrom(mul_k_in, k_unpack_dims);
      k_sha_inputs.emplace_back(k_unpack);

      auto& v_unpack =
          tensor_pool.CloneNativeTensorFrom(transpose_v_in, v_unpack_dims);
      v_sha_inputs.emplace_back(v_unpack);
      auto& sha_out = tensor_pool.CloneNativeTensorFrom(mha_out, mha_out_dims);
      sha_outputs.emplace_back(sha_out);
    }
    auto unpack_q_op = BuildUnpackOp(
        tensor_pool, {const_cast<::qnn::TensorWrapper&>(mul_q_in)},
        q_sha_inputs, 2);
    std::move(unpack_q_op.begin(), unpack_q_op.end(),
              std::back_inserter(new_ops));
    auto unpack_k_op = BuildUnpackOp(
        tensor_pool, {const_cast<::qnn::TensorWrapper&>(mul_k_in)},
        k_sha_inputs, 2);
    std::move(unpack_k_op.begin(), unpack_k_op.end(),
              std::back_inserter(new_ops));
    auto unpack_v_op = BuildUnpackOp(
        tensor_pool, {const_cast<::qnn::TensorWrapper&>(transpose_v_in)},
        v_sha_inputs, 2);
    std::move(unpack_v_op.begin(), unpack_v_op.end(),
              std::back_inserter(new_ops));

    for (int i = 0; i < num_heads; ++i) {
      auto& q_matmul_in = tensor_pool.CloneNativeTensorFrom(q_sha_inputs[i]);
      EmplaceOpWithIO(new_ops, ops[matmul_qk_index + kAttnMulQ],
                      {q_sha_inputs[i], std::nullopt}, {q_matmul_in});
      auto& k_transpose_in = tensor_pool.CloneNativeTensorFrom(k_sha_inputs[i]);
      EmplaceOpWithIO(new_ops, ops[matmul_qk_index + kAttnMulK],
                      {k_sha_inputs[i], std::nullopt}, {k_transpose_in});
      auto& k_matmul_in = tensor_pool.CloneNativeTensorFrom(
          k_transpose_in,
          {k_unpack_dims[0], k_unpack_dims[2], k_unpack_dims[1]});
      auto transpose_op = BuildTransposeOp(
          tensor_pool, {k_transpose_in, perm_tensor}, {k_matmul_in});
      std::move(transpose_op.begin(), transpose_op.end(),
                std::back_inserter(new_ops));
      // MatMul
      const auto& matmul_qk_out = ops[matmul_qk_index].GetOutputTensor(0);
      auto& select_in = tensor_pool.CloneNativeTensorFrom(
          matmul_qk_out, {q_matmul_in.GetDim(0), q_matmul_in.GetDim(1),
                          k_matmul_in.GetDim(2)});
      EmplaceOpWithIO(new_ops, ops[matmul_qk_index], {q_matmul_in, k_matmul_in},
                      {select_in});

      // Change Select to Add.
      auto& softmax_in =
          tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
      auto add_select =
          BuildElementwiseAddOp(tensor_pool, {select_in, add_in}, {softmax_in});
      std::move(add_select.begin(), add_select.end(),
                std::back_inserter(new_ops));

      // Softmax
      auto& qk_softmax =
          tensor_pool.CloneNativeTensorFrom(softmax_in, select_out_dims);
      EmplaceOpWithIO(new_ops, ops[select_index + kAttnSoftmax], {softmax_in},
                      {qk_softmax});
      // MatMul
      const auto& matmul_out =
          ops[select_index + kAttnMatMul].GetOutputTensor(0);
      auto matmul_out_dims = matmul_out.GetDims();
      matmul_out_dims.erase(matmul_out_dims.begin() + 1);
      EmplaceOpWithIO(new_ops, ops[matmul_qk_index],
                      {qk_softmax, v_sha_inputs[i]}, {sha_outputs[i]});
    }
    // Pack
    auto pack_op = BuildPackOp(tensor_pool, sha_outputs,
                               {const_cast<::qnn::TensorWrapper&>(mha_out)}, 2);
    std::move(pack_op.begin(), pack_op.end(), std::back_inserter(new_ops));
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

size_t OptimizeTransposeMatMul(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr size_t kAddIndex = 0;
  constexpr size_t kTransposeIndex = 1;
  constexpr size_t kMulIndex = 2;
  constexpr size_t kReshapeIndex = 3;
  constexpr size_t kMatmul0Index = 4;
  constexpr size_t kMatmul1Index = 5;
  if (!(IS_CONNECTED(kAddIndex, 0, kTransposeIndex, 0)) &&
      (IS_CONNECTED(kMulIndex, 0, kReshapeIndex, 0)) &&
      (IS_CONNECTED(kReshapeIndex, 0, kMatmul0Index, 0)) &&
      (IS_CONNECTED(kReshapeIndex, 0, kMatmul1Index, 0)) &&
      (IS_CONNECTED(kTransposeIndex, 0, kMatmul1Index, 1))) {
    return 1;
  }

  std::vector<OpWrapper> new_ops;
  auto& transpose = ops[start_index + kTransposeIndex];

  // Check if Transpose Op permute data is [0, 1, 3, 2].
  auto transpose_perm_data =
      transpose.GetTensorPararm(0).GetTensor().GetTensorData<uint32_t>();
  const auto& transpose_perm_dims =
      transpose.GetTensorPararm(0).GetTensor().GetDims();
  if (!transpose_perm_data) {
    QNN_LOG_ERROR(
        "Failed to get permute date of Transpose Op. Rolling back to the "
        "original graph.")
    return 1;
  }
  if (transpose_perm_dims.size() != 1 && transpose_perm_dims[0] != 4) {
    QNN_LOG_ERROR(
        "The permute tensor dimension of Transpose Op does not match the "
        "pattern. Rolling back to the original graph.")
    return 1;
  }
  if (transpose_perm_data.value()[2] != 3 ||
      transpose_perm_data.value()[3] != 2) {
    QNN_LOG_ERROR(
        "The permute date of Transpose Op does not match the pattern. Rolling "
        "back to the original graph.")
    return 1;
  }

  auto& matmul_1 = ops[start_index + kMatmul1Index];
  // Get adj_y in MatMul Op.
  auto adj_y_param = matmul_1.GetScalarParam(1);
  if (!adj_y_param ||
      adj_y_param->GetName() != QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) {
    return 1;
  }
  bool origin_adj_y = adj_y_param->GetValue<bool>();

  auto& transpose_in_0 = transpose.GetInputTensor(0);
  auto& matmul_in_0 = matmul_1.GetInputTensor(0);
  auto& matmul_out_0 = matmul_1.GetOutputTensor(0);

  auto new_matmul_1 =
      BuildMatmulOp(tensor_pool,
                    {const_cast<::qnn::TensorWrapper&>(matmul_in_0),
                     const_cast<::qnn::TensorWrapper&>(transpose_in_0)},
                    {const_cast<::qnn::TensorWrapper&>(matmul_out_0)},
                    /*adj_x*/ false, /*adj_y*/ !origin_adj_y);
  std::move(new_matmul_1.begin(), new_matmul_1.end(),
            std::back_inserter(new_ops));

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
    // Insert the new op at the end.
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    // Erase Transpose and MatMul 1.
    ops.erase(ops.begin() + start_index + kMatmul1Index);
    ops.erase(ops.begin() + start_index + kTransposeIndex);
    QNN_LOG_INFO("[G2G] Fused Transpose - MatMul.");
  } else {
    QNN_LOG_ERROR(
        "[G2G] Validation failed. Rolling back to the original graph.");
  }
  return 1;
}

}  // namespace qnn
