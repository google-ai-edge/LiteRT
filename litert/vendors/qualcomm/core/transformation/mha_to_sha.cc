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
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
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

TensorWrapper& BuildSingleSHA(std::vector<OpWrapper>& ops, size_t start_index,
                              std::vector<OpWrapper>& new_ops,
                              TensorPool& tensor_pool,
                              qnn::TensorWrapper& sha_input,
                              const ::qnn::OpWrapper& scaling_mul,
                              size_t num_heads) {
  // Mul
  auto& mul_output = tensor_pool.CloneNativeTensorFrom(
      scaling_mul.GetOutputTensor(0), sha_input.GetDims());
  EmplaceOpWithIO(new_ops, scaling_mul, {sha_input, std::nullopt},
                  {mul_output});
  // MatMul 1
  const auto& matmulk_cache_output =
      ops[start_index + kMatMulK1Index].GetOutputTensor(0);
  std::vector<uint32_t> matmul1_output_dim = matmulk_cache_output.GetDims();
  matmul1_output_dim[2] /= num_heads;
  auto& matmul1_output = tensor_pool.CloneNativeTensorFrom(matmulk_cache_output,
                                                           matmul1_output_dim);
  EmplaceOpWithIO(new_ops, ops[start_index + kMatMulK1Index],
                  {mul_output, std::nullopt}, {matmul1_output});
  // MatMul 2
  const auto& matmulk_slice_output =
      ops[start_index + kMatMulK2Index].GetOutputTensor(0);
  std::vector<uint32_t> matmul2_output_dim = matmulk_slice_output.GetDims();
  matmul2_output_dim[2] = matmul2_output_dim[2] / num_heads;
  auto& matmul2_output = tensor_pool.CloneNativeTensorFrom(matmulk_slice_output,
                                                           matmul2_output_dim);
  EmplaceOpWithIO(new_ops, ops[start_index + kMatMulK2Index],
                  {mul_output, std::nullopt}, {matmul2_output});
  // Concat
  std::vector<uint32_t> concat_output_dim = matmul1_output.GetDims();
  concat_output_dim[3] += matmul2_output.GetDim(3);
  auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_index + kConcatIndex].GetOutputTensor(0), concat_output_dim);
  EmplaceOpWithIO(new_ops, ops[start_index + kConcatIndex],
                  {matmul1_output, matmul2_output}, {concat_output});
  // Add
  auto& add_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_index + kAddIndex].GetOutputTensor(0), concat_output.GetDims());
  EmplaceOpWithIO(new_ops, ops[start_index + kAddIndex],
                  {concat_output, std::nullopt}, {add_output});
  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_index + kSoftmaxIndex].GetOutputTensor(0),
      add_output.GetDims());
  EmplaceOpWithIO(new_ops, ops[start_index + kSoftmaxIndex], {add_output},
                  {softmax_output});
  // Slice 1
  // Create StridedSlice param ranges.
  auto mha_slice1_param =
      ops[start_index + kSlice1Index].GetTensorPararm(0).GetTensor();
  auto mha_slice1_param_data = mha_slice1_param.GetTensorData<int32_t>();
  std::vector<int32_t> slice1_ranges(mha_slice1_param_data.value().begin(),
                                     mha_slice1_param_data.value().end());
  slice1_ranges[kSlice3rdAxisEndIndex] /= num_heads;
  auto& slice1_param_tensor = tensor_pool.CreateStaticTensor(
      mha_slice1_param.GetDataType(), mha_slice1_param.GetQuantParams(),
      mha_slice1_param.GetDims(), mha_slice1_param.GetTensorBytes(),
      slice1_ranges.data());
  // Create StridedSlice op.
  auto slice1_output_dims =
      ops[start_index + kSlice1Index].GetOutputTensor(0).GetDims();
  slice1_output_dims[2] /= num_heads;
  auto& slice1_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_index + kSlice1Index].GetOutputTensor(0), slice1_output_dims);
  EmplaceOpWithIO(new_ops, ops[start_index + kSlice1Index], {softmax_output},
                  {slice1_output});
  new_ops.back().ClearTensorParams();
  new_ops.back().AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                slice1_param_tensor);
  // MatMul 1
  const auto& matmulv_cache_output =
      ops[start_index + kMatMulV1Index].GetOutputTensor(0);

  std::vector<uint32_t> matmul1_v_output_dim = matmulv_cache_output.GetDims();
  matmul1_v_output_dim[2] = matmul1_v_output_dim[2] / num_heads;
  auto& matmul1_v_output = tensor_pool.CloneNativeTensorFrom(
      matmulv_cache_output, matmul1_v_output_dim);
  EmplaceOpWithIO(new_ops, ops[start_index + kMatMulV1Index],
                  {slice1_output, std::nullopt}, {matmul1_v_output});
  // Slice 2
  // Create StridedSlice param ranges.
  auto mha_slice2_param =
      ops[start_index + kSlice2Index].GetTensorPararm(0).GetTensor();
  auto mha_slice2_param_data = mha_slice2_param.GetTensorData<int32_t>();
  std::vector<int32_t> slice2_ranges(mha_slice2_param_data.value().begin(),
                                     mha_slice2_param_data.value().end());
  slice2_ranges[kSlice3rdAxisEndIndex] /= num_heads;
  auto& slice2_param_tensor = tensor_pool.CreateStaticTensor(
      mha_slice2_param.GetDataType(), mha_slice2_param.GetQuantParams(),
      mha_slice2_param.GetDims(), mha_slice2_param.GetTensorBytes(),
      slice2_ranges.data());
  // Create StridedSlice op.
  auto slice2_output_dims =
      ops[start_index + kSlice2Index].GetOutputTensor(0).GetDims();
  slice2_output_dims[2] /= num_heads;
  auto& slice2_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_index + kSlice2Index].GetOutputTensor(0), slice2_output_dims);
  EmplaceOpWithIO(new_ops, ops[start_index + kSlice2Index], {softmax_output},
                  {slice2_output});
  new_ops.back().ClearTensorParams();
  new_ops.back().AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                slice2_param_tensor);
  // MatMul 2
  const auto& matmulv_slice_output =
      ops[start_index + kMatMulV2Index].GetOutputTensor(0);
  std::vector<uint32_t> matmul2_v_output_dim = matmulv_slice_output.GetDims();
  matmul2_v_output_dim[2] = matmul2_v_output_dim[2] / num_heads;
  auto& matmul2_v_output = tensor_pool.CloneNativeTensorFrom(
      matmulv_slice_output, matmul2_v_output_dim);
  EmplaceOpWithIO(new_ops, ops[start_index + kMatMulV2Index],
                  {slice2_output, std::nullopt}, {matmul2_v_output});
  // Add
  auto& add_final_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_index + kAdd2Index].GetOutputTensor(0),
      matmul1_v_output.GetDims());
  EmplaceOpWithIO(new_ops, ops[start_index + kAdd2Index],
                  {matmul1_v_output, matmul2_v_output}, {add_final_output});
  return add_final_output;
}

TensorWrapper& BuildSingleSHA(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    TensorWrapper& sha_1_input, TensorWrapper& sha_2_input,
    TensorWrapper& sha_3_input, TensorWrapper& sha_4_input,
    TensorWrapper& sha_5_input, TensorWrapper& sha_6_input,
    const uint32_t num_attn_per_kv_heads, const OpWrapper& mul,
    const OpWrapper& matmul_q, const OpWrapper& add_1,
    const OpWrapper& matmul_qk, const OpWrapper& concat, const OpWrapper& add_2,
    const OpWrapper& reshape_3, const OpWrapper& softmax,
    const OpWrapper& slice_1, const OpWrapper& slice_2,
    const OpWrapper& matmul_v1, const OpWrapper& matmul_v2,
    const OpWrapper& add_3) {
  // Mul
  auto mul_output_dims = mul.GetOutputTensor(0).GetDims();
  mul_output_dims.erase(mul_output_dims.begin() + 1);
  auto& mul_output = tensor_pool.CloneNativeTensorFrom(mul.GetOutputTensor(0),
                                                       mul_output_dims);
  EmplaceOpWithIO(new_ops, mul, {sha_3_input, std::nullopt}, {mul_output});

  // Matmul q
  auto matmul_q_output_dims = matmul_q.GetOutputTensor(0).GetDims();
  matmul_q_output_dims.erase(matmul_q_output_dims.begin() + 1);
  matmul_q_output_dims[1] /= num_attn_per_kv_heads;
  auto& matmul_q_output = tensor_pool.CloneNativeTensorFrom(
      matmul_q.GetOutputTensor(0), matmul_q_output_dims);
  EmplaceOpWithIO(new_ops, matmul_q, {mul_output, sha_2_input},
                  {matmul_q_output});

  // Add
  auto add_1_output_dims = add_1.GetOutputTensor(0).GetDims();
  add_1_output_dims.erase(add_1_output_dims.begin() + 1);
  auto& add_1_output = tensor_pool.CloneNativeTensorFrom(
      add_1.GetOutputTensor(0), add_1_output_dims);
  EmplaceOpWithIO(new_ops, add_1, {sha_4_input, sha_5_input}, {add_1_output});

  // Matmul k
  auto matmul_qk_output_dims = matmul_qk.GetOutputTensor(0).GetDims();
  matmul_qk_output_dims.erase(matmul_qk_output_dims.begin() + 1);
  matmul_qk_output_dims[1] /= num_attn_per_kv_heads;
  auto& matmul_qk_output = tensor_pool.CloneNativeTensorFrom(
      matmul_qk.GetOutputTensor(0), matmul_qk_output_dims);
  EmplaceOpWithIO(new_ops, matmul_qk, {mul_output, add_1_output},
                  {matmul_qk_output});

  // Concat
  std::uint32_t adjusted_axis = 2;
  auto concat_output_dims = concat.GetOutputTensor(0).GetDims();
  concat_output_dims.erase(concat_output_dims.begin() + 1);
  concat_output_dims[1] /= num_attn_per_kv_heads;
  auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      concat.GetOutputTensor(0), concat_output_dims);
  EmplaceOpWithIO(new_ops, concat, {matmul_q_output, matmul_qk_output},
                  {concat_output});
  new_ops.back().ClearScalarParams();
  new_ops.back().AddScalarParam(QNN_OP_CONCAT_PARAM_AXIS, adjusted_axis);

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
  EmplaceOpWithIO(new_ops, slice_1, {softmax_output}, {slice_1_output});
  new_ops.back().ClearTensorParams();
  new_ops.back().AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES,
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
  EmplaceOpWithIO(new_ops, slice_2, {softmax_output}, {slice_2_output});
  new_ops.back().ClearTensorParams();
  new_ops.back().AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                slice_2_param_tensor);

  // Matmul v1
  auto matmul_v1_output_dims = matmul_v1.GetOutputTensor(0).GetDims();
  matmul_v1_output_dims.erase(matmul_v1_output_dims.begin() + 1);
  matmul_v1_output_dims[1] /= num_attn_per_kv_heads;
  auto& matmul_v1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v1.GetOutputTensor(0), matmul_v1_output_dims);
  EmplaceOpWithIO(new_ops, matmul_v1, {slice_1_output, sha_1_input},
                  {matmul_v1_output});

  // Matmul v2
  auto matmul_v2_output_dims = matmul_v2.GetOutputTensor(0).GetDims();
  matmul_v2_output_dims.erase(matmul_v2_output_dims.begin() + 1);
  matmul_v2_output_dims[1] /= num_attn_per_kv_heads;
  auto& matmul_v2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v2.GetOutputTensor(0), matmul_v2_output_dims);
  EmplaceOpWithIO(new_ops, matmul_v2, {slice_2_output, sha_6_input},
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

void CloneNamespace(OpWrapper& source, std::vector<OpWrapper>& ops) {
  absl::string_view start_op_name = source.GetOpConfig().v1.name;
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
    sha_outputs.emplace_back(
        BuildSingleSHA(ops, start_index, new_ops, tensor_pool,
                       const_cast<TensorWrapper&>(sha_inputs[i].get()),
                       scaling_mul, num_heads));
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
        IS_CONNECTED(kTranspose2Index + 2, 0, kReshape3Index + 2, 0))) {
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
        IS_CONNECTED(kAdd2Index, 0, kReshape2Index, 0))) {
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
        is_connected(transpose_2_index, 0, reshape_5_index, 0))) {
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

  // Concat 3
  auto concat_op = BuildConcatenationOp(
      tensor_pool, sha_outputs,
      {const_cast<::qnn::TensorWrapper&>(pattern_output)}, 2);
  std::move(concat_op.begin(), concat_op.end(), std::back_inserter(new_ops));

  // Validate new graph.
  // TODO(jiunkaiy): Disable bypassing Split int16 op validator.
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
    return step_size;
  }

  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

}  // namespace qnn
