// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/embedding_gemma.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "QnnTypes.h"              // from @qairt
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reduce_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace {
#define IS_CONNECTED(op1, op2)                  \
  (ops[start_index + op1].GetOutputTensor(0) == \
   ops[start_index + op2].GetInputTensor(0))

constexpr size_t kMulIndexIndex = 0;
constexpr size_t kTransposeIndex = 1;
constexpr size_t kReshapeIndex = 2;
constexpr size_t kMatMulIndex = 3;
constexpr size_t kAddIndex = 4;
constexpr size_t kSoftmaxIndex = 5;
constexpr size_t kMatMul2Index = 6;
constexpr size_t kReshape2Index = 7;
constexpr size_t kTranspose2Index = 8;
constexpr size_t kReshape3Index = 9;
}  // namespace
namespace qnn {

OpWrapper& EmplaceOpWithIO(
    std::vector<OpWrapper>& new_ops, const OpWrapper& source_op,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& inputs,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& outputs) {
  auto& ret = new_ops.emplace_back(source_op);
  ret.UpdateTensors(inputs, outputs);
  return ret;
}

TensorWrapper& BuildSingleSHA(const std::vector<OpWrapper>& original_ops,
                              size_t start_index,
                              std::vector<OpWrapper>& new_ops,
                              TensorPool& tensor_pool,
                              qnn::TensorWrapper& sha_input,
                              qnn::TensorWrapper& mask_input,
                              size_t num_heads) {
  const auto& mul_op = original_ops[start_index + kMulIndexIndex];
  const auto& matmul_op1 = original_ops[start_index + kMatMulIndex];
  const auto& add_op = original_ops[start_index + kAddIndex];
  const auto& softmax_op = original_ops[start_index + kSoftmaxIndex];
  const auto& matmul_op2 = original_ops[start_index + kMatMul2Index];
  // Mul
  auto& mul_output = tensor_pool.CloneNativeTensorFrom(
      mul_op.GetOutputTensor(0), sha_input.GetDims());
  auto& new_mul_op =
      EmplaceOpWithIO(new_ops, mul_op, {sha_input, std::nullopt}, {mul_output});

  // MatMul 1
  const auto& matmul_op1_output = matmul_op1.GetOutputTensor(0);
  std::vector<uint32_t> new_matmul1_output_dim = matmul_op1_output.GetDims();
  new_matmul1_output_dim[2] /= num_heads;
  auto& new_matmul1_output = tensor_pool.CloneNativeTensorFrom(
      matmul_op1_output, new_matmul1_output_dim);
  auto& new_matmul_op1 = EmplaceOpWithIO(
      new_ops, matmul_op1, {mul_output, std::nullopt}, {new_matmul1_output});

  TensorWrapper* softmax_input = nullptr;

  // Add
  auto& new_add_output = tensor_pool.CloneNativeTensorFrom(
      add_op.GetOutputTensor(0), new_matmul1_output_dim);
  auto& new_add_op = EmplaceOpWithIO(
      new_ops, add_op, {new_matmul1_output, mask_input}, {new_add_output});
  softmax_input = &new_add_output;

  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax_op.GetOutputTensor(0), new_add_output.GetDims());
  EmplaceOpWithIO(new_ops, softmax_op, {new_add_output}, {softmax_output});

  // MatMul 2
  auto matmul_op2_out_dim = matmul_op2.GetOutputTensor(0).GetDims();
  matmul_op2_out_dim[2] /= num_heads;
  auto& new_matmul2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_op2.GetOutputTensor(0), matmul_op2_out_dim);
  EmplaceOpWithIO(new_ops, matmul_op2, {softmax_output, std::nullopt},
                  {new_matmul2_output});
  return new_matmul2_output;
}

std::vector<OpWrapper> MHA2SHA(const std::vector<OpWrapper>& original_ops,
                               size_t start_index, TensorPool& tensor_pool,
                               size_t pattern_size) {
  std::vector<OpWrapper> new_ops;
  const auto& mul_op = original_ops[start_index + kMulIndexIndex];
  const auto& tranpose_op1 = original_ops[start_index + kTransposeIndex];
  const auto& add_op = original_ops[start_index + kAddIndex];

  const auto& pattern_input = mul_op.GetInputTensor(0);
  const auto& pattern_output =
      original_ops[start_index + pattern_size - 1].GetOutputTensor(0);

  // Transpose
  auto transpose_out_dims = tranpose_op1.GetOutputTensor(0).GetDims();
  auto& transpose_output =
      tensor_pool.CloneNativeTensorFrom(pattern_input, transpose_out_dims);
  auto& new_transpose1 = EmplaceOpWithIO(
      new_ops, tranpose_op1, {const_cast<::qnn::TensorWrapper&>(pattern_input)},
      {transpose_output});

  const uint32_t num_heads = pattern_input.GetDim(2);
  const auto& mha_input = new_transpose1.GetOutputTensor(0);  // split_in

  std::vector<::qnn::TensorWrapperRef> sha_inputs;
  sha_inputs.reserve(num_heads);
  for (int i = 0; i < num_heads; i++) {
    auto sha_input_dims = mha_input.GetDims();  // split_out_dims
    sha_input_dims[1] /= num_heads;
    auto& split_output =
        tensor_pool.CloneNativeTensorFrom(mha_input, sha_input_dims);
    sha_inputs.emplace_back(split_output);
  }

  // split from mul
  const std::array<int32_t, 1> split_axis_data{1};
  auto& split_axis = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {split_axis_data.size()},
      split_axis_data.size() * sizeof(split_axis_data[0]),
      split_axis_data.data());
  auto split_op = BuildSplitOp(
      tensor_pool, {split_axis, const_cast<TensorWrapper&>(mha_input)},
      sha_inputs, num_heads);

  std::move(split_op.begin(), split_op.end(), std::back_inserter(new_ops));

  // split from mask
  auto& mask_input = add_op.GetInputTensor(1);
  std::vector<::qnn::TensorWrapperRef> new_mask_inputs;
  new_mask_inputs.reserve(num_heads);
  for (int i = 0; i < num_heads; i++) {
    auto new_mask_input_dims = mask_input.GetDims();
    new_mask_input_dims[2] /= num_heads;
    auto& mask_split_output =
        tensor_pool.CloneNativeTensorFrom(mask_input, new_mask_input_dims);
    new_mask_inputs.emplace_back(mask_split_output);
  }

  const std::array<int32_t, 1> mask_split_axis_data{2};
  auto& mask_split_axis = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {mask_split_axis_data.size()},
      mask_split_axis_data.size() * sizeof(mask_split_axis_data[0]),
      mask_split_axis_data.data());
  auto mask_split_op = BuildSplitOp(
      tensor_pool, {mask_split_axis, const_cast<TensorWrapper&>(mask_input)},
      new_mask_inputs, num_heads);

  std::move(mask_split_op.begin(), mask_split_op.end(),
            std::back_inserter(new_ops));

  std::vector<TensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_heads);
  for (int i = 0; i < num_heads; ++i) {
    sha_outputs.emplace_back(BuildSingleSHA(
        original_ops, start_index, new_ops, tensor_pool,
        const_cast<TensorWrapper&>(sha_inputs[i].get()),
        const_cast<TensorWrapper&>(new_mask_inputs[i].get()), num_heads));
  }

  // Concat
  auto concat_dims = pattern_output.GetDims();
  concat_dims.insert(concat_dims.begin(), 1);
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(pattern_output, concat_dims);
  auto concat_final =
      BuildConcatenationOp(tensor_pool, sha_outputs, {concat_output}, 3);
  std::move(concat_final.begin(), concat_final.end(),
            std::back_inserter(new_ops));
  // Reshape
  auto reshape =
      BuildReshapeOp(tensor_pool, {concat_output},
                     {const_cast<::qnn::TensorWrapper&>(pattern_output)});
  std::move(reshape.begin(), reshape.end(), std::back_inserter(new_ops));
  return new_ops;
}

size_t TransformEmbeddingGemma(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Connection check
  bool is_connected = IS_CONNECTED(kMulIndexIndex, kTransposeIndex) &&
                      IS_CONNECTED(kTransposeIndex, kReshapeIndex) &&
                      IS_CONNECTED(kReshapeIndex, kMatMulIndex) &&
                      IS_CONNECTED(kMatMulIndex, kAddIndex) &&
                      IS_CONNECTED(kAddIndex, kSoftmaxIndex) &&
                      IS_CONNECTED(kSoftmaxIndex, kMatMul2Index) &&
                      IS_CONNECTED(kMatMul2Index, kReshape2Index) &&
                      IS_CONNECTED(kReshape2Index, kTranspose2Index) &&
                      IS_CONNECTED(kTranspose2Index, kReshape3Index);
  if (!is_connected) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] Transforming MHA to SHA in Embedding Gemma");
  // Construct the new subgraph
  auto new_ops = MHA2SHA(ops, start_index, tensor_pool, pattern_size);
  if (new_ops.empty()) {
    QNN_LOG_WARNING(
        "[G2G] Transformation failed. Rolling back to the original graph.");
    return 1;
  }
  // Validate new graph.
  bool is_valid =
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
    QNN_LOG_INFO("[G2G] Done transforming MHA to SHA in Embedding Gemma!");
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

}  // namespace qnn
