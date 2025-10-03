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

#include "QnnOpDef.h"                  // from @qairt
#include "QnnTypes.h"                  // from @qairt
#include "absl/strings/str_cat.h"      // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

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

namespace {

// TODO(Alen): Should be able to utilize this function for Gemma3.
TensorWrapper& BuildSingleSHA(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    TensorWrapper& sha_input, TensorWrapper& splited_mask, size_t num_heads,
    const OpWrapper& mul, const OpWrapper& matmul_k1,
    const OpWrapper& matmul_k2, const OpWrapper& concat, const OpWrapper& add_1,
    const OpWrapper& softmax, const OpWrapper& slice_1,
    const OpWrapper& slice_2, const OpWrapper& matmul_v1,
    const OpWrapper& matmul_v2, const OpWrapper& add_2) {
  // Mul
  auto& mul_output = tensor_pool.CloneNativeTensorFrom(mul.GetOutputTensor(0),
                                                       sha_input.GetDims());
  auto& sha_mul =
      EmplaceOpWithIO(new_ops, mul, {sha_input, std::nullopt}, {mul_output});

  // MatMul 1
  auto matmul_k1_output_dims = matmul_k1.GetOutputTensor(0).GetDims();
  matmul_k1_output_dims[2] /= num_heads;
  auto& matmul_k1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k1.GetOutputTensor(0), matmul_k1_output_dims);
  auto& sha_matmul_k1 = EmplaceOpWithIO(
      new_ops, matmul_k1, {mul_output, std::nullopt}, {matmul_k1_output});

  // MatMul 2
  auto matmul_k2_output_dims = matmul_k2.GetOutputTensor(0).GetDims();
  matmul_k2_output_dims[2] /= num_heads;
  auto& matmul_k2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k2.GetOutputTensor(0), matmul_k2_output_dims);
  auto& sha_matmul_k2 = EmplaceOpWithIO(
      new_ops, matmul_k2, {mul_output, std::nullopt}, {matmul_k2_output});

  // Concat
  auto concat_output_dims = matmul_k1_output_dims;
  concat_output_dims[3] += matmul_k2_output_dims[3];
  auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      concat.GetOutputTensor(0), concat_output_dims);
  auto& sha_concat = EmplaceOpWithIO(
      new_ops, concat, {matmul_k1_output, matmul_k2_output}, {concat_output});

  // Add
  auto& add_1_output = tensor_pool.CloneNativeTensorFrom(
      add_1.GetOutputTensor(0), concat_output.GetDims());
  auto& sha_add_1 = EmplaceOpWithIO(
      new_ops, add_1, {concat_output, splited_mask}, {add_1_output});

  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), add_1_output.GetDims());
  auto& sha_softmax =
      EmplaceOpWithIO(new_ops, softmax, {add_1_output}, {softmax_output});

  // Slice 1
  // Create StridedSlice param ranges.
  const auto& slice_1_ranges = slice_1.GetTensorPararm(0).GetTensor();
  auto slice_1_ranges_data = slice_1_ranges.GetTensorData<std::int32_t>();
  std::vector<std::int32_t> sha_slice_1_ranges_data(
      slice_1_ranges_data.value().begin(), slice_1_ranges_data.value().end());
  sha_slice_1_ranges_data[kSlice3rdAxisEndIndex] /= num_heads;
  auto& sha_slice_1_ranges = tensor_pool.CreateStaticTensor(
      slice_1_ranges.GetDataType(), slice_1_ranges.GetQuantParams(),
      slice_1_ranges.GetDims(), slice_1_ranges.GetTensorBytes(),
      sha_slice_1_ranges_data.data());
  // Create StridedSlice op.
  auto slice_1_output_dims = slice_1.GetOutputTensor(0).GetDims();
  slice_1_output_dims[2] /= num_heads;
  auto& slice_1_output = tensor_pool.CloneNativeTensorFrom(
      slice_1.GetOutputTensor(0), slice_1_output_dims);
  auto& sha_slice_1 =
      EmplaceOpWithIO(new_ops, slice_1, {softmax_output}, {slice_1_output});
  sha_slice_1.ClearTensorParams();
  sha_slice_1.AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                             sha_slice_1_ranges);

  // Slice 2
  // Create StridedSlice param ranges.
  const auto& slice_2_ranges = slice_2.GetTensorPararm(0).GetTensor();
  auto slice_2_ranges_data = slice_2_ranges.GetTensorData<std::int32_t>();
  std::vector<std::int32_t> sha_slice_2_ranges_data(
      slice_2_ranges_data.value().begin(), slice_2_ranges_data.value().end());
  sha_slice_2_ranges_data[kSlice3rdAxisEndIndex] /= num_heads;
  auto& sha_slice_2_ranges = tensor_pool.CreateStaticTensor(
      slice_2_ranges.GetDataType(), slice_2_ranges.GetQuantParams(),
      slice_2_ranges.GetDims(), slice_2_ranges.GetTensorBytes(),
      sha_slice_2_ranges_data.data());
  // Create StridedSlice op.
  auto slice_2_output_dims = slice_2.GetOutputTensor(0).GetDims();
  slice_2_output_dims[2] /= num_heads;
  auto& slice_2_output = tensor_pool.CloneNativeTensorFrom(
      slice_2.GetOutputTensor(0), slice_2_output_dims);
  auto& sha_slice_2 =
      EmplaceOpWithIO(new_ops, slice_2, {softmax_output}, {slice_2_output});
  sha_slice_2.ClearTensorParams();
  sha_slice_2.AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                             sha_slice_2_ranges);

  // MatMul 1
  auto matmul_v1_output_dims = matmul_v1.GetOutputTensor(0).GetDims();
  matmul_v1_output_dims[2] /= num_heads;
  auto& matmul_v1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v1.GetOutputTensor(0), matmul_v1_output_dims);
  auto& sha_matmul_v1 = EmplaceOpWithIO(
      new_ops, matmul_v1, {slice_1_output, std::nullopt}, {matmul_v1_output});

  // MatMul 2
  auto matmul_v2_output_dims = matmul_v2.GetOutputTensor(0).GetDims();
  matmul_v2_output_dims[2] /= num_heads;
  auto& matmul_v2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v2.GetOutputTensor(0), matmul_v2_output_dims);
  auto& sha_matmul_v2 = EmplaceOpWithIO(
      new_ops, matmul_v2, {slice_2_output, std::nullopt}, {matmul_v2_output});

  // Add 2
  auto& add_2_output = tensor_pool.CloneNativeTensorFrom(
      add_2.GetOutputTensor(0), matmul_v1_output.GetDims());
  auto& sha_add_2 = EmplaceOpWithIO(
      new_ops, add_2, {matmul_v1_output, matmul_v2_output}, {add_2_output});

  return add_2_output;
}

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
        is_connected(transpose_1, 0, reshape_2, 0))) {
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
      new_ops, transpoe_0, {const_cast<::qnn::TensorWrapper&>(pattern_input)},
      {transpose_output});

  // Reshape
  auto& reshape_output = tensor_pool.CloneNativeTensorFrom(
      pattern_input, {transpose_output_dims[0], 1,
                      transpose_output_dims[1] * transpose_output_dims[2],
                      transpose_output_dims[3]});
  auto& new_reshape_0 =
      EmplaceOpWithIO(new_ops, reshape_0, {transpose_output}, {reshape_output});

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDim(2);
  const auto& mha_input = new_reshape_0.GetOutputTensor(0);

  // Prepare inputs for num_heads SHAs.
  std::vector<TensorWrapperRef> sha_inputs;
  sha_inputs.reserve(num_heads);
  auto sha_input_dims = new_reshape_0.GetOutputTensor(0).GetDims();
  sha_input_dims[2] /= num_heads;
  for (size_t i = 0; i < num_heads; ++i) {
    auto& sha_input =
        tensor_pool.CloneNativeTensorFrom(mha_input, sha_input_dims);
    sha_inputs.emplace_back(sha_input);
  }

  // Split
  const std::array<int32_t, 1> split_axis_data{2};
  auto& split_axis = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {split_axis_data.size()},
      split_axis_data.size() * sizeof(decltype(split_axis_data)::value_type),
      split_axis_data.data());
  auto split_state_ops = BuildSplitOp(
      tensor_pool, {split_axis, const_cast<::qnn::TensorWrapper&>(mha_input)},
      sha_inputs, num_heads);
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
      tensor_pool,
      {split_axis, const_cast<::qnn::TensorWrapper&>(concated_mask)},
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
  std::move(concat_sha_output_ops.begin(), concat_sha_output_ops.end(),
            std::back_inserter(new_ops));

  // Reshape
  auto new_reshape_ops =
      BuildReshapeOp(tensor_pool, {concat_output},
                     {const_cast<::qnn::TensorWrapper&>(pattern_output)});
  std::move(new_reshape_ops.begin(), new_reshape_ops.end(),
            std::back_inserter(new_ops));
  return true;
}

}  // namespace

size_t OptimizeMHATinyGemmaPrefillPattern0(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  const auto& mul = ops[start_index + 0];
  const auto& transpose_0 = ops[start_index + 1];
  const auto& reshape_0 = ops[start_index + 2];
  const auto& matmul_k0 = ops[start_index + 3];
  const auto& matmul_k1 = ops[start_index + 4];
  const auto& concat = ops[start_index + 5];
  const auto& add_0 = ops[start_index + 8];
  const auto& softmax = ops[start_index + 9];
  const auto& slice_0 = ops[start_index + 10];
  const auto& slice_1 = ops[start_index + 11];
  const auto& matmul_v0 = ops[start_index + 12];
  const auto& matmul_v1 = ops[start_index + 13];
  const auto& add_1 = ops[start_index + 14];
  const auto& reshape_1 = ops[start_index + 15];
  const auto& transpose_1 = ops[start_index + 16];
  const auto& reshape_2 = ops[start_index + 17];

  const auto& mask_concat = ops[start_index + 6];
  const auto& mask_reshape = ops[start_index + 7];
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
    size_t step_size = new_ops.size() + 2;
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    // Only keep mask_concat and mask_reshape
    ops.erase(ops.begin() + start_index + 8, ops.begin() + start_index + 18);
    ops.erase(ops.begin() + start_index, ops.begin() + start_index + 6);
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

size_t OptimizeMHATinyGemmaPrefillPattern1(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  const auto& mul = ops[start_index + 0];
  const auto& transpose_0 = ops[start_index + 1];
  const auto& reshape_0 = ops[start_index + 2];
  const auto& matmul_k0 = ops[start_index + 3];
  const auto& matmul_k1 = ops[start_index + 4];
  const auto& concat = ops[start_index + 5];
  const auto& add_0 = ops[start_index + 6];
  const auto& softmax = ops[start_index + 7];
  const auto& slice_0 = ops[start_index + 8];
  const auto& slice_1 = ops[start_index + 9];
  const auto& matmul_v0 = ops[start_index + 10];
  const auto& matmul_v1 = ops[start_index + 11];
  const auto& add_1 = ops[start_index + 12];
  const auto& reshape_1 = ops[start_index + 13];
  const auto& transpose_1 = ops[start_index + 14];
  const auto& reshape_2 = ops[start_index + 15];

  std::vector<OpWrapper> new_ops;
  if (!OptimizeMHATinyGemmaPrefill(
          new_ops, tensor_pool, mul, transpose_0, reshape_0, matmul_k0,
          matmul_k1, concat, add_0, softmax, slice_0, slice_1, matmul_v0,
          matmul_v1, add_1, reshape_1, transpose_1, reshape_2)) {
    return 1;
  }

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

}  // namespace qnn
