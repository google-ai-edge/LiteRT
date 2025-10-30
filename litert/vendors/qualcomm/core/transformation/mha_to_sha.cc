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
#include "litert/vendors/qualcomm/core/builders/pack_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
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

// Emplaces the operator with updated inputs/outputs into new_ops.
// This function copies the source_op and updates the tensors according to
// inputs and outputs. The std::nullopt input/output element indicates that the
// tensor will not be updated.
void EmplaceOpWithIO(
    std::vector<OpWrapper>& new_ops, const OpWrapper& source_op,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& inputs,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& outputs) {
  OpWrapper ret = source_op;
  ret.UpdateTensors(inputs, outputs);
  new_ops.emplace_back(ret);
}

// Retrieves the first operation index that matches the given opcode and
// includes the input_tensor. The operation is searched from start_ind and
// onwards. Returns -1 if not found.
int GetOpIndexWithInput(const std::vector<OpWrapper>& ops, QnnOpCode opcode,
                        const qnn::TensorWrapper& input_tensor,
                        size_t start_ind = 0) {
  for (size_t i = start_ind; i < ops.size(); ++i) {
    if (ops[i].IsOpCode(opcode) && ops[i].GetInputTensor(0) == input_tensor) {
      return i;
    }
  }
  return -1;
}

// Retrieves the first operation index that matches the given opcode and
// includes the output_tensor. Returns -1 if not found.
int GetOpIndexWithOutput(const std::vector<OpWrapper>& ops, QnnOpCode opcode,
                         const qnn::TensorWrapper& output_tensor) {
  for (size_t i = 0; i < ops.size(); ++i) {
    if (ops[i].IsOpCode(opcode) && ops[i].GetOutputTensor(0) == output_tensor) {
      return i;
    }
  }
  return -1;
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

size_t GetOpCountWithInput(const std::vector<OpWrapper>& ops, QnnOpCode opcode,
                           const qnn::TensorWrapper& input_tensor) {
  size_t cnt = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    if (ops[i].IsOpCode(opcode) && ops[i].GetInputTensor(0) == input_tensor) {
      cnt++;
    }
  }
  return cnt;
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

size_t OptimizeMHAAttn(std::function<bool(OpWrapper&)> validate_op_config,
                       std::vector<OpWrapper>& ops, size_t attn_start_index,
                       TensorPool& tensor_pool, size_t pattern_size) {
  // Connection check: Reshape -> NotEqual -> Select
  size_t start_index = attn_start_index;
  if (!(IS_CONNECTED(kAttnReshape, 0, kAttnNotEqual, 0)) &&
      (IS_CONNECTED(kAttnNotEqual, 0, kAttnSelect, 0))) {
    return 1;
  }
  auto attn_reshape_op = ops[attn_start_index + kAttnReshape];
  auto attn_not_equal_op = ops[attn_start_index + kAttnNotEqual];
  // Count the operations that have NotEqual as their source op.
  const auto& reshape_in = attn_reshape_op.GetInputTensor(0);
  size_t num_out = GetOpCountWithInput(ops, QnnOpCode::kElementWiseSelect,
                                       attn_not_equal_op.GetOutputTensor(0));
  if (num_out <= 0) {
    return 1;
  }

  QNN_LOG_INFO("[G2G] MHA optimization (Attn)");
  // Handle masking.
  std::vector<OpWrapper> new_ops;
  const auto& not_equal_out = attn_not_equal_op.GetOutputTensor(0);
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
      std::max(select_const.GetStaticTensorData<float>().value()[0], -65472.f);
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
    select_index = GetOpIndexWithInput(ops, QnnOpCode::kElementWiseSelect,
                                       attn_not_equal_op.GetOutputTensor(0),
                                       select_index + 1);
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
    size_t matmul_qk_index = GetOpIndexWithOutput(
        ops, QnnOpCode::kMatMul, ops[select_index].GetInputTensor(1));
    // Connection check based on Matmul index.
    start_index = matmul_qk_index;
    if (!(IS_CONNECTED(kAttnMulQ, 0, kAttnTransposeQ, 0) &&
          IS_CONNECTED(kAttnMulK, 0, kAttnTransposeK, 0) &&
          IS_CONNECTED(kAttnTransposeQ, 0, 0, 0) &&
          IS_CONNECTED(kAttnTransposeK, 0, 0, 1))) {
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
}  // namespace qnn
