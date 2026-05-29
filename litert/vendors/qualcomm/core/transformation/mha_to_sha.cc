// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/mha_to_sha.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/pack_op_builder.h"
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

// Allow-listed (num_q_heads, num_kv_heads) shapes for the GQA MHA->SHA
// transformations. Prefill currently supports one extra shape that decode does
// not (TinyTiny with kv_heads=1) — kept asymmetric on purpose pending decode
// validation for that case.
constexpr std::array<std::pair<size_t, size_t>, 4> kSupportedGqaPrefillShapes =
    {{
        {14, 2},  // FastVLM
        {16, 8},  // Kanana
        {4, 2},   // TinyTiny
        {4, 1},   // TinyTiny (prefill-only)
    }};
constexpr std::array<std::pair<size_t, size_t>, 3> kSupportedGqaDecodeShapes = {
    {
        {14, 2},  // FastVLM
        {16, 8},  // Kanana
        {4, 2},   // TinyTiny
    }};

template <size_t N>
bool IsSupportedGqaShape(
    const std::array<std::pair<size_t, size_t>, N>& shapes, size_t num_q_heads,
    size_t num_kv_heads) {
  return std::any_of(shapes.begin(), shapes.end(), [&](const auto& shape) {
    return shape.first == num_q_heads && shape.second == num_kv_heads;
  });
}

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
  new_ops.emplace_back(CreateOpWithSameParams(
      matmul_k1, {mul_output, matmul_k1.GetInputTensor(1)},
      {matmul_k1_output}));

  // MatMul 2
  auto matmul_k2_output_dims = matmul_k2.GetOutputTensor(0).GetDimensions();
  matmul_k2_output_dims[2] /= num_heads;
  const auto& matmul_k2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_k2.GetOutputTensor(0), matmul_k2_output_dims);
  new_ops.emplace_back(CreateOpWithSameParams(
      matmul_k2, {mul_output, matmul_k2.GetInputTensor(1)},
      {matmul_k2_output}));

  // Concat
  auto concat_output_dims = matmul_k1_output_dims;
  concat_output_dims[3] += matmul_k2_output_dims[3];
  const auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      concat.GetOutputTensor(0), concat_output_dims);
  new_ops.emplace_back(CreateOpWithSameParams(
      concat, {matmul_k1_output, matmul_k2_output}, {concat_output}));

  // Add
  const auto& add_1_output = tensor_pool.CloneNativeTensorFrom(
      add_1.GetOutputTensor(0), concat_output.GetDimensions());
  new_ops.emplace_back(
      CreateElementWiseAddOp(concat_output, mask, add_1_output));
  // Softmax
  const auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), add_1_output.GetDimensions());
  new_ops.emplace_back(
      CreateOpWithSameParams(softmax, {add_1_output}, {softmax_output}));

  // Slice 1
  // QNN slice ranges are flat (begin, end, stride) triplets per axis;
  // ranges[3*axis + 1] is the "end" of that axis. Axis 2 end -> index 7.
  constexpr size_t kSlice3rdAxisEndIndex = 7;
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
  new_ops.emplace_back(CreateOpWithSameParams(
      matmul_v1, {slice_1_output, matmul_v1.GetInputTensor(1)},
      {matmul_v1_output}));

  // MatMul 2
  std::vector<uint32_t> matmul_v2_output_dims =
      matmul_v2.GetOutputTensor(0).GetDimensions();
  matmul_v2_output_dims[2] = matmul_v2_output_dims[2] / num_heads;
  const auto& matmul_v2_output = tensor_pool.CloneNativeTensorFrom(
      matmul_v2.GetOutputTensor(0), matmul_v2_output_dims);
  new_ops.emplace_back(CreateOpWithSameParams(
      matmul_v2, {slice_2_output, matmul_v2.GetInputTensor(1)},
      {matmul_v2_output}));

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

std::vector<ConstTensorWrapperRef> SplitTensor(TensorPool& tensor_pool,
                                               std::vector<OpWrapper>& new_ops,
                                               const TensorWrapper& input,
                                               size_t axis, size_t tile_size) {
  auto input_dims = input.GetDimensions();
  size_t same_size_cnt = input_dims[axis] / tile_size;
  size_t num_outputs = same_size_cnt + (input_dims[axis] % tile_size != 0);
  QNN_LOG_DEBUG("[SplitTensor] num_outputs: %zu", num_outputs);
  if (num_outputs == 1) return {input};

  std::vector<ConstTensorWrapperRef> outputs;
  outputs.reserve(num_outputs);
  // Create Regular tiles based on tile_size.
  input_dims[axis] = tile_size;
  for (size_t i = 0; i < same_size_cnt; ++i) {
    QNN_LOG_DEBUG(" tile_size(%zu) %u", i, input_dims[axis]);
    outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(input, input_dims));
  }
  // Create the last tile smaller than tile_size if needed.
  if (num_outputs > same_size_cnt) {
    input_dims[axis] = input.GetDimension(axis) % tile_size;
    QNN_LOG_DEBUG(" tile_size(%zu) %u", num_outputs - 1, input_dims[axis]);
    outputs.emplace_back(tensor_pool.CloneNativeTensorFrom(input, input_dims));
  }

  std::vector<std::uint32_t> split_indice;
  split_indice.reserve(num_outputs - 1);
  for (std::uint32_t i = 1; i < num_outputs; i++) {
    split_indice.emplace_back(i * tile_size);
  }
  const auto& split_indice_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {},
      {static_cast<std::uint32_t>(split_indice.size())},
      sizeof(split_indice[0]) * split_indice.size(), split_indice.data());
  new_ops.emplace_back(
      CreateSplitOp(input, outputs, axis, split_indice_tensor));
  return outputs;
}

TensorWrapper& BuildShaFromGqa(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    const uint32_t num_attn_per_kv_heads, const TensorWrapper& input,
    const TensorWrapper& k_cache, const TensorWrapper& k_slice,
    const TensorWrapper& v_cache, const TensorWrapper& v_slice,
    const OpWrapper* scale_mul, const OpWrapper& q_kcache_matmul,
    const OpWrapper& q_kslice_matmul, const OpWrapper& qk_concat,
    const OpWrapper& mask_add, const OpWrapper& softmax,
    const OpWrapper& qk_vcache_slice, const OpWrapper& qk_vslice_slice,
    const OpWrapper& qk_vcache_matmul, const OpWrapper& qk_vslice_matmul,
    const OpWrapper& qkv_add) {
  const TensorWrapper* matmul_input = &input;
  // Scale Mul -> change matmul_input.
  if (scale_mul) {
    auto mul_output_dims = scale_mul->GetOutputTensor(0).GetDimensions();
    mul_output_dims[1] = 1;
    matmul_input = &tensor_pool.CloneNativeTensorFrom(
        scale_mul->GetOutputTensor(0), mul_output_dims);
    new_ops.emplace_back(CreateElementWiseMulOp(
        input, scale_mul->GetInputTensor(1), *matmul_input));
  }

  // Q KCache Matmul
  auto q_kcache_matmul_output_dims =
      q_kcache_matmul.GetOutputTensor(0).GetDimensions();
  q_kcache_matmul_output_dims[1] = 1u;
  q_kcache_matmul_output_dims[2] /= num_attn_per_kv_heads;
  auto& q_kcache_matmul_output = tensor_pool.CloneNativeTensorFrom(
      q_kcache_matmul.GetOutputTensor(0), q_kcache_matmul_output_dims);
  const std::array<ConstTensorWrapperRef, 2> q_kcache_matmul_inputs = {
      *matmul_input, k_cache};
  const std::array<ConstTensorWrapperRef, 1> q_kcache_matmul_outputs = {
      q_kcache_matmul_output};
  new_ops.emplace_back(CreateOpWithSameParams(
      q_kcache_matmul, q_kcache_matmul_inputs, q_kcache_matmul_outputs));
  // Q KSlice Matmul
  auto q_kslice_matmul_output_dims =
      q_kslice_matmul.GetOutputTensor(0).GetDimensions();
  q_kslice_matmul_output_dims[1] = 1u;
  q_kslice_matmul_output_dims[2] /= num_attn_per_kv_heads;
  auto& q_kslice_matmul_output = tensor_pool.CloneNativeTensorFrom(
      q_kslice_matmul.GetOutputTensor(0), q_kslice_matmul_output_dims);
  const std::array<ConstTensorWrapperRef, 2> q_kslice_matmul_inputs = {
      *matmul_input, k_slice};
  const std::array<ConstTensorWrapperRef, 1> q_kslice_matmul_outputs = {
      q_kslice_matmul_output};
  new_ops.emplace_back(CreateOpWithSameParams(
      q_kslice_matmul, q_kslice_matmul_inputs, q_kslice_matmul_outputs));

  // QK Concat
  std::uint32_t adjusted_axis = 3u;
  auto concat_output_dims = qk_concat.GetOutputTensor(0).GetDimensions();
  concat_output_dims[1] = 1;
  concat_output_dims[2] /= num_attn_per_kv_heads;
  auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      qk_concat.GetOutputTensor(0), concat_output_dims);
  new_ops.emplace_back(
      CreateConcatenationOp({q_kcache_matmul_output, q_kslice_matmul_output},
                            concat_output, adjusted_axis));

  // Mask Add
  const auto& mask_add_out = mask_add.GetOutputTensor(0);
  auto mask_add_output_dims = mask_add_out.GetDimensions();
  mask_add_output_dims[1] = 1;
  mask_add_output_dims[2] /= num_attn_per_kv_heads;
  auto& mask_add_output = tensor_pool.CloneNativeTensorFrom(
      mask_add.GetOutputTensor(0), mask_add_output_dims);
  new_ops.emplace_back(CreateElementWiseAddOp(
      concat_output, mask_add.GetInputTensor(1), mask_add_output));
  // Softmax
  auto softmax_output_dims = softmax.GetOutputTensor(0).GetDimensions();
  softmax_output_dims[1] = 1;
  softmax_output_dims[2] /= num_attn_per_kv_heads;
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      softmax.GetOutputTensor(0), softmax_output_dims);
  const std::array<ConstTensorWrapperRef, 1> softmax_inputs = {mask_add_output};
  const std::array<ConstTensorWrapperRef, 1> softmax_outputs = {softmax_output};
  new_ops.emplace_back(
      CreateOpWithSameParams(softmax, softmax_inputs, softmax_outputs));
  // QK VCache Slice
  // QNN slice ranges are flat (begin, end, stride) triplets per axis;
  // ranges[3*axis + 1] is the "end" of that axis.
  //   axis 1 end -> index 4
  //   axis 2 end -> index 7
  constexpr size_t kSlice2ndAxisEndIndex = 4;
  constexpr size_t kSlice3rdAxisEndIndex = 7;
  auto qk_vcache_slice_param = qk_vcache_slice.GetTensorParam(0).GetTensor();
  auto qk_vcache_slice_param_data =
      qk_vcache_slice_param.GetTensorData<int32_t>();
  std::vector<int32_t> qk_vcache_slice_ranges(
      qk_vcache_slice_param_data.value().begin(),
      qk_vcache_slice_param_data.value().end());
  qk_vcache_slice_ranges[kSlice2ndAxisEndIndex] = 1;
  qk_vcache_slice_ranges[kSlice3rdAxisEndIndex] /= num_attn_per_kv_heads;
  std::vector<uint32_t> qk_vcache_slice_param_dims = {
      static_cast<uint32_t>(qk_vcache_slice_ranges.size() / 3), 3};
  auto& qk_vcache_slice_param_tensor = tensor_pool.CreateStaticTensor(
      qk_vcache_slice_param.GetDataType(),
      qk_vcache_slice_param.GetQuantParams(), qk_vcache_slice_param_dims,
      sizeof(qk_vcache_slice_ranges[0]) * qk_vcache_slice_ranges.size(),
      qk_vcache_slice_ranges.data());
  auto qk_vcache_slice_output_dims =
      qk_vcache_slice.GetOutputTensor(0).GetDimensions();
  qk_vcache_slice_output_dims[1] = 1;
  qk_vcache_slice_output_dims[2] /= num_attn_per_kv_heads;
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
  qk_vslice_slice_ranges[kSlice2ndAxisEndIndex] = 1;
  qk_vslice_slice_ranges[kSlice3rdAxisEndIndex] /= num_attn_per_kv_heads;
  std::vector<uint32_t> qk_vslice_slice_param_dims = {
      static_cast<uint32_t>(qk_vslice_slice_ranges.size() / 3), 3};
  auto& qk_vslice_slice_param_tensor = tensor_pool.CreateStaticTensor(
      qk_vslice_slice_param.GetDataType(),
      qk_vslice_slice_param.GetQuantParams(), qk_vslice_slice_param_dims,
      sizeof(qk_vslice_slice_ranges[0]) * qk_vslice_slice_ranges.size(),
      qk_vslice_slice_ranges.data());
  auto qk_vslice_slice_output_dims =
      qk_vslice_slice.GetOutputTensor(0).GetDimensions();
  qk_vslice_slice_output_dims[1] = 1;
  qk_vslice_slice_output_dims[2] /= num_attn_per_kv_heads;
  auto& qk_vslice_slice_output = tensor_pool.CloneNativeTensorFrom(
      qk_vslice_slice.GetOutputTensor(0), qk_vslice_slice_output_dims);
  new_ops.emplace_back(CreateSliceOp(softmax_output, qk_vslice_slice_output,
                                     qk_vslice_slice_param_tensor));
  // QK VCache Matmul
  auto qk_vcache_matmul_output_dims =
      qk_vcache_matmul.GetOutputTensor(0).GetDimensions();
  qk_vcache_matmul_output_dims[1] = 1;
  qk_vcache_matmul_output_dims[2] /= num_attn_per_kv_heads;
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
  qk_vslice_matmul_output_dims[1] = 1;
  qk_vslice_matmul_output_dims[2] /= num_attn_per_kv_heads;
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
  qkv_add_output_dims[1] = 1;
  qkv_add_output_dims[2] /= num_attn_per_kv_heads;
  auto& qkv_add_output = tensor_pool.CloneNativeTensorFrom(
      qkv_add.GetOutputTensor(0), qkv_add_output_dims);
  new_ops.emplace_back(CreateElementWiseAddOp(
      qk_vcache_matmul_output, qk_vslice_matmul_output, qkv_add_output));
  return qkv_add_output;
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
  new_ops.emplace_back(CreateOpWithSameParams(
      q_kcache_matmul, {mul_output, k_cache}, {q_kcache_matmul_output}));

  // Q KSlice Matmul
  auto q_kslice_matmul_output_dims =
      q_kslice_matmul.GetOutputTensor(0).GetDimensions();
  q_kslice_matmul_output_dims.erase(q_kslice_matmul_output_dims.begin() + 1);
  q_kslice_matmul_output_dims[1] /= num_attn_per_kv_heads;
  auto& q_kslice_matmul_output = tensor_pool.CloneNativeTensorFrom(
      q_kslice_matmul.GetOutputTensor(0), q_kslice_matmul_output_dims);
  new_ops.emplace_back(CreateOpWithSameParams(
      q_kslice_matmul, {mul_output, k_slice}, {q_kslice_matmul_output}));

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
  new_ops.emplace_back(CreateOpWithSameParams(
      softmax, {post_mask_reshape_output}, {softmax_output}));

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
  new_ops.emplace_back(CreateOpWithSameParams(qk_vcache_matmul,
                                              {qk_vcache_slice_output, v_cache},
                                              {qk_vcache_matmul_output}));

  // QK VSlice Matmul
  auto qk_vslice_matmul_output_dims =
      qk_vslice_matmul.GetOutputTensor(0).GetDimensions();
  qk_vslice_matmul_output_dims.erase(qk_vslice_matmul_output_dims.begin() + 1);
  qk_vslice_matmul_output_dims[1] /= num_attn_per_kv_heads;
  auto& qk_vslice_matmul_output = tensor_pool.CloneNativeTensorFrom(
      qk_vslice_matmul.GetOutputTensor(0), qk_vslice_matmul_output_dims);
  new_ops.emplace_back(CreateOpWithSameParams(qk_vslice_matmul,
                                              {qk_vslice_slice_output, v_slice},
                                              {qk_vslice_matmul_output}));

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
  new_ops.emplace_back(
      CreateOpWithSameParams(ops[start_index + kTransposePrefillIndex],
                             {pattern_input}, {transpose_output}));

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

size_t OptimizeGqaPrefill(std::function<bool(OpWrapper&)> validate_op_config,
                          std::vector<OpWrapper>& ops, size_t start_index,
                          TensorPool& tensor_pool, size_t pattern_size) {
  QNN_LOG_INFO("[G2G] GQA Optimization (Prefill)");
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
  bool has_scale_mul =
      is_connected(-1, 0, 0, 0) && IsElementWiseMultiply(ops[start_index - 1]);
  QNN_LOG_DEBUG("[G2G] GQA Optimization (Prefill): has_scale_mul %d",
                has_scale_mul);
  // Adjust indices based on scale_mul.
  start_index -= has_scale_mul;
  pattern_size += has_scale_mul;
  const size_t q_scale_reshape_idx = 0 + has_scale_mul;
  const size_t q_kcache_matmul_idx = 1 + has_scale_mul;
  const size_t q_kslice_matmul_idx = 2 + has_scale_mul;
  const size_t qk_concat_idx = 3 + has_scale_mul;
  const size_t mask_concat_idx = 4 + has_scale_mul;
  const size_t mask_add_idx = 5 + has_scale_mul;
  const size_t softmax_idx = 6 + has_scale_mul;
  const size_t qkv_cache_slice_idx = 7 + has_scale_mul;
  const size_t qkv_slice_slice_idx = 8 + has_scale_mul;
  const size_t qkv_cache_matmul_idx = 9 + has_scale_mul;
  const size_t qkv_slice_matmul_idx = 10 + has_scale_mul;
  const size_t qkv_add_idx = 11 + has_scale_mul;
  const size_t qkv_reshape_idx = 12 + has_scale_mul;
  const size_t qkv_transpose_idx = 13 + has_scale_mul;
  const size_t o_proj_reshape_idx = 14 + has_scale_mul;

  if (!(is_connected(q_scale_reshape_idx, 0, q_kcache_matmul_idx, 0) &&
        is_connected(q_scale_reshape_idx, 0, q_kslice_matmul_idx, 0) &&
        is_connected(q_kcache_matmul_idx, 0, qk_concat_idx, 0) &&
        is_connected(q_kslice_matmul_idx, 0, qk_concat_idx, 1) &&
        is_connected(qk_concat_idx, 0, mask_add_idx, 0) &&
        is_connected(mask_add_idx, 0, softmax_idx, 0) &&
        is_connected(softmax_idx, 0, qkv_cache_slice_idx, 0) &&
        is_connected(softmax_idx, 0, qkv_slice_slice_idx, 0) &&
        is_connected(qkv_cache_slice_idx, 0, qkv_cache_matmul_idx, 0) &&
        is_connected(qkv_slice_slice_idx, 0, qkv_slice_matmul_idx, 0) &&
        is_connected(qkv_cache_matmul_idx, 0, qkv_add_idx, 0) &&
        is_connected(qkv_slice_matmul_idx, 0, qkv_add_idx, 1) &&
        is_connected(qkv_add_idx, 0, qkv_reshape_idx, 0) &&
        is_connected(qkv_reshape_idx, 0, qkv_transpose_idx, 0) &&
        is_connected(qkv_transpose_idx, 0, o_proj_reshape_idx, 0) &&
        IsElementWiseAdd(ops[start_index + mask_add_idx]) &&
        IsElementWiseAdd(ops[start_index + qkv_add_idx]))) {
    return 1;
  }

  const size_t num_q_heads = ops[start_index].GetInputTensor(0).GetDimension(1);
  const size_t num_kv_heads =
      ops[start_index + q_kslice_matmul_idx].GetInputTensor(1).GetDimension(1);
  // Strict check: only well-known GQA shapes are supported. See
  // kSupportedGqaPrefillShapes.
  QNN_LOG_DEBUG(
      "[G2G] GQA Optimization (Prefill):\n  # Q Heads: %zu\n  # KV Heads: %zu",
      num_q_heads, num_kv_heads);
  if (!IsSupportedGqaShape(kSupportedGqaPrefillShapes, num_q_heads,
                           num_kv_heads)) {
    return 1;
  }
  QNN_LOG_INFO("[G2G] GQA optimization (Prefill)");
  std::vector<OpWrapper> new_ops;

  // QKV Unpack
  const auto& k_cache =
      ops[start_index + q_kcache_matmul_idx].GetInputTensor(1);
  auto k_cache_unpack_outputs = SplitTensor(tensor_pool, new_ops, k_cache);
  constexpr size_t kUnpackAxis = 1;
  auto k_slice_unpack_outputs = SplitTensor(
      tensor_pool, new_ops,
      ops[start_index + q_kslice_matmul_idx].GetInputTensor(1), kUnpackAxis);
  auto q_inputs =
      SplitTensor(tensor_pool, new_ops, ops[start_index].GetInputTensor(0));

  const auto& v_cache =
      ops[start_index + qkv_cache_matmul_idx].GetInputTensor(1);
  auto v_cache_unpack_outputs = SplitTensor(tensor_pool, new_ops, v_cache);

  const auto& v_slice =
      ops[start_index + qkv_slice_matmul_idx].GetInputTensor(1);
  auto v_slice_unpack_outputs = SplitTensor(tensor_pool, new_ops, v_slice);

  auto group_size = num_q_heads / num_kv_heads;
  // Remove unnecessary concat mask.
  auto add_op = CreateElementWiseAddOp(
      ops[start_index + mask_add_idx].GetInputTensor(0),
      ops[start_index + mask_concat_idx].GetInputTensor(0),
      ops[start_index + mask_add_idx].GetOutputTensor(0));
  const auto& mask_add_out = add_op.GetOutputTensor(0);
  auto mask_add_output_dims = mask_add_out.GetDimensions();
  // Build num_head SHAs
  std::vector<ConstTensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_q_heads);
  for (size_t i = 0; i < num_kv_heads; ++i) {
    for (size_t j = 0; j < group_size; ++j) {
      auto& sha_output = BuildShaFromGqa(
          new_ops, tensor_pool, group_size, q_inputs[i * group_size + j],
          k_cache_unpack_outputs[i], k_slice_unpack_outputs[i],
          v_cache_unpack_outputs[i], v_slice_unpack_outputs[i],
          has_scale_mul ? &ops[start_index] : nullptr,
          ops[start_index + q_kcache_matmul_idx],
          ops[start_index + q_kslice_matmul_idx],
          ops[start_index + qk_concat_idx], add_op,
          ops[start_index + softmax_idx],
          ops[start_index + qkv_cache_slice_idx],
          ops[start_index + qkv_slice_slice_idx],
          ops[start_index + qkv_cache_matmul_idx],
          ops[start_index + qkv_slice_matmul_idx],
          ops[start_index + qkv_add_idx]);
      sha_outputs.emplace_back(sha_output);
    }
  }
  const auto& qkv_reshape = ops[start_index + pattern_size - 1];
  // Concat SHA outputs by the last dimension.
  const auto concat_axis = sha_outputs[0].get().GetRank() - 1;
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

  // Clone namespace.
  CloneNamespace(ops[start_index + q_scale_reshape_idx], new_ops);
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
    QNN_LOG_INFO("[G2G] GQA optimization (Prefill) done.");
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

size_t SimplifyMaskingAdd(std::function<bool(OpWrapper&)> validate_op_config,
                          std::vector<OpWrapper>& ops, size_t start_index,
                          TensorPool& tensor_pool, size_t pattern_size) {
  constexpr size_t kMaskingPreReshapeIndex = 0;
  constexpr size_t kMaskingAddIndex = 1;
  constexpr size_t kMaskingPostReshapeIndex = 2;
  if (!(IS_CONNECTED(kMaskingPreReshapeIndex, 0, kMaskingAddIndex, 0) &&
        IS_CONNECTED(kMaskingAddIndex, 0, kMaskingPostReshapeIndex, 0))) {
    return 1;
  }
  auto& add_input =
      ops[start_index + kMaskingPreReshapeIndex].GetInputTensor(0);
  auto mask = &ops[start_index + kMaskingAddIndex].GetInputTensor(1);
  QNN_LOG_INFO("[G2G] Simplify masking");
  std::vector<OpWrapper> new_ops;
  for (size_t index = 0; index < mask->GetRank(); ++index) {
    size_t mask_dim = mask->GetDimension(index);
    size_t input_dim = add_input.GetDimension(index);
    if (!(mask_dim == input_dim || mask_dim == 1 || input_dim == 1)) {
      std::vector<qnn::ConstTensorWrapperRef> inputs;
      size_t broadcast_size = input_dim / mask_dim;
      inputs.reserve(broadcast_size);
      for (size_t i = 0; i < broadcast_size; ++i) {
        inputs.emplace_back(*mask);
      }
      auto new_dims = mask->GetDimensions();
      new_dims[index] = input_dim;
      mask = &tensor_pool.CloneNativeTensorFrom(*mask, new_dims);
      new_ops.emplace_back(CreateConcatenationOp(inputs, *mask, index));
      QNN_LOG_INFO("[G2G] Simplify masking w/ Add @ %zu", index);
      break;
    }
  }
  new_ops.emplace_back(CreateElementWiseAddOp(
      add_input, *mask,
      ops[start_index + kMaskingPostReshapeIndex].GetOutputTensor(0)));
  CloneNamespace(ops[start_index], new_ops);
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

size_t DuplicateOrRemoveConcat(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr size_t kMaskingConcatIndex = 0;
  constexpr size_t kMaskingAddIndex = 1;
  const auto& concat_op = ops[start_index + kMaskingConcatIndex];
  const std::string concat_op_name(concat_op.GetName());
  const auto& mask = concat_op.GetInputTensor(0);
  const auto& concat_output = concat_op.GetOutputTensor(0);
  const auto& add_op = ops[start_index + kMaskingAddIndex];
  if (!(IS_CONNECTED(kMaskingConcatIndex, 0, kMaskingAddIndex, 1) &&
        IsElementWiseAdd(add_op))) {
    return 1;
  }
  // Check if this concat is only for broadcast.
  size_t num_elements = concat_output.GetTensorNumElements();
  size_t input_cnt = 0;
  while (num_elements > 0) {
    auto& concat_input = concat_op.GetInputTensor(input_cnt);
    if (concat_input != mask) {
      return 1;
    }
    num_elements -= concat_input.GetTensorNumElements();
    input_cnt++;
  }

  // Find all add indices.
  std::vector<size_t> indices;
  for (size_t i = ops.size() - 1; i-- > 0;) {
    if (IsElementWiseAdd(ops[i]) && concat_output == ops[i].GetInputTensor(1)) {
      indices.push_back(i);
    }
  }
  if (indices.empty()) {
    return 1;
  }

  bool can_remove_concat = true;
  for (size_t index = 0; index < mask.GetRank(); ++index) {
    size_t mask_dim = mask.GetDimension(index);
    size_t input_dim = add_op.GetInputTensor(0).GetDimension(index);
    if (!(mask_dim == input_dim || mask_dim == 1 || input_dim == 1)) {
      can_remove_concat = false;
      break;
    }
  }

  QNN_LOG_INFO("[G2G] %s concat", can_remove_concat ? "Remove" : "Duplicate");

  // Build replacement ops first; only commit them if every one validates.
  std::vector<std::pair<size_t, OpWrapper>> add_replacements;
  std::vector<std::pair<size_t, OpWrapper>> concat_inserts;
  add_replacements.reserve(indices.size());
  if (!can_remove_concat) {
    concat_inserts.reserve(indices.size());
  }
  const std::vector<qnn::ConstTensorWrapperRef> concat_inputs(
      input_cnt, concat_op.GetInputTensor(0));
  for (size_t i : indices) {
    if (can_remove_concat) {
      auto add = CreateElementWiseAddOp(ops[i].GetInputTensor(0), mask,
                                        ops[i].GetOutputTensor(0));
      CloneNamespace(ops[i], add);
      add.AddSuffixToName(absl::StrCat("_qcg2g_", i));
      add_replacements.emplace_back(i, std::move(add));
    } else {
      const auto& duplicated_concat_output =
          tensor_pool.CloneNativeTensorFrom(concat_output);
      auto add = CreateElementWiseAddOp(ops[i].GetInputTensor(0),
                                        duplicated_concat_output,
                                        ops[i].GetOutputTensor(0));
      CloneNamespace(ops[i], add);
      add.AddSuffixToName(absl::StrCat("_qcg2g_", i));
      add_replacements.emplace_back(i, std::move(add));

      auto concat = CreateOpWithSameParams(concat_op, concat_inputs,
                                           {duplicated_concat_output});
      CloneNamespace(ops[i], concat);
      concat.AddSuffixToName(absl::StrCat("_qcg2g_", i));
      concat_inserts.emplace_back(i, std::move(concat));
    }
  }

  const auto validate = [&validate_op_config](
                            std::vector<std::pair<size_t, OpWrapper>>& items) {
    return std::all_of(items.begin(), items.end(),
                       [&validate_op_config](auto& kv) -> bool {
                         return validate_op_config(kv.second);
                       });
  };
  if (!validate(add_replacements) || !validate(concat_inserts)) {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
    return 1;
  }
  // Commit: replace add ops, then insert duplicated concat ops. `indices` is
  // in descending order, so concat insertions don't shift any index we still
  // need to touch.
  for (auto& [i, op] : add_replacements) {
    ops[i] = std::move(op);
  }
  for (auto& [i, op] : concat_inserts) {
    ops.insert(ops.begin() + i, std::move(op));
  }
  ops.erase(std::remove_if(ops.begin(), ops.end(),
                           [&concat_op_name](const auto& op) {
                             return op.GetName() == concat_op_name;
                           }),
            ops.end());
  return pattern_size;
}

size_t FuseConcatReshape(std::function<bool(OpWrapper&)> validate_op_config,
                         std::vector<OpWrapper>& ops, size_t start_index,
                         TensorPool& tensor_pool, size_t pattern_size) {
  constexpr size_t kFusedConcatIndex = 0;
  constexpr size_t kFusedReshapeIndex = 1;
  auto& concat = ops[start_index + kFusedConcatIndex];
  auto& reshape = ops[start_index + kFusedReshapeIndex];

  // Connection check
  if (!IS_CONNECTED(kFusedConcatIndex, 0, kFusedReshapeIndex, 0)) {
    return 1;
  }
  // Check the indices with different dim.
  const auto& reshape_input_dims = reshape.GetInputTensor(0).GetDimensions();
  const auto& reshape_output_dims = reshape.GetOutputTensor(0).GetDimensions();
  std::vector<size_t> diff_indices;
  diff_indices.reserve(
      std::max(reshape_output_dims.size(), reshape_output_dims.size()));
  for (size_t i = 0; i < reshape_input_dims.size(); ++i) {
    if (reshape_input_dims[i] != reshape_output_dims[i]) {
      diff_indices.emplace_back(i);
    }
  }
  constexpr size_t kDiffIndices = 2;
  if (diff_indices.size() != kDiffIndices ||
      reshape_output_dims[diff_indices[0]] != 1) {
    return 1;
  }

  // Change concat axis form diff_indices[0] to diff_indices[1], and remove
  // reshape. Example: 2, 128 -> 1, 256 => concat at the 2nd axis instead of the
  // 1st.
  QNN_LOG_INFO("[G2G] convert-reshape fusion");
  std::vector<ConstTensorWrapperRef> concat_inputs;
  concat_inputs.reserve(concat.GetInputTensorCount());
  for (size_t i = 0; i < concat.GetInputTensorCount(); ++i) {
    concat_inputs.emplace_back(concat.GetInputTensor(i));
  }
  auto new_concat = CreateConcatenationOp(
      concat_inputs, {reshape.GetOutputTensor(0)}, diff_indices[1]);
  new_concat.AddSuffixToName(absl::StrCat("_qcg2g_0"));
  CloneNamespace(concat, new_concat);
  if (validate_op_config(new_concat)) {
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    ops.emplace(ops.begin() + start_index, std::move(new_concat));
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
  }
  return 1;
}

size_t OptimizeGqaDecode(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
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
  // Case 1 (offset: 2): mul -> reshape -> gqa
  // Case 2 (offset: 1): rehsapae -> gqa
  // Case 3 (offset: 0): other -> gqa
  size_t offset = 0;
  if (is_connected(-1, 0, 0, 0) && is_connected(-1, 0, 1, 0) &&
      ops[start_index - 1].IsOpCode(QnnOpCode::kReshape)) {
    offset++;
    start_index--;
    if (is_connected(-1, 0, 0, 0) &&
        IsElementWiseMultiply(ops[start_index - 1])) {
      offset++;
      start_index--;
    }
  }
  pattern_size += offset;
  // `offset` is computed by at most two `++` above; the only reachable values
  // are 0, 1, or 2.
  if (offset == 2) {
    QNN_LOG_DEBUG("[G2G] GQA Optimization (Decode): mul -> reshape -> GQA");
  } else if (offset == 1) {
    QNN_LOG_DEBUG("[G2G] GQA Optimization (Decode): reshape -> GQA");
  } else {
    QNN_LOG_DEBUG("[G2G] GQA Optimization (Decode): GQA");
  }

  const size_t q_kcache_matmul_idx = 0 + offset;
  const size_t q_kslice_matmul_idx = 1 + offset;
  const size_t qk_concat_idx = 2 + offset;
  const size_t mask_add_idx = 3 + offset;
  const size_t softmax_idx = 4 + offset;
  const size_t qkv_cache_slice_idx = 5 + offset;
  const size_t qkv_slice_slice_idx = 6 + offset;
  const size_t qkv_cache_matmul_idx = 7 + offset;
  const size_t qkv_slice_matmul_idx = 8 + offset;
  const size_t qkv_add_idx = 9 + offset;
  const size_t qkv_reshape_idx = 10 + offset;

  if (!(ops[start_index + q_kcache_matmul_idx].GetInputTensor(0) ==
            ops[start_index + q_kslice_matmul_idx].GetInputTensor(0) &&
        is_connected(q_kcache_matmul_idx, 0, qk_concat_idx, 0) &&
        is_connected(q_kslice_matmul_idx, 0, qk_concat_idx, 1) &&
        is_connected(qk_concat_idx, 0, mask_add_idx, 0) &&
        is_connected(mask_add_idx, 0, softmax_idx, 0) &&
        is_connected(softmax_idx, 0, qkv_cache_slice_idx, 0) &&
        is_connected(softmax_idx, 0, qkv_slice_slice_idx, 0) &&
        is_connected(qkv_cache_slice_idx, 0, qkv_cache_matmul_idx, 0) &&
        is_connected(qkv_slice_slice_idx, 0, qkv_slice_matmul_idx, 0) &&
        is_connected(qkv_cache_matmul_idx, 0, qkv_add_idx, 0) &&
        is_connected(qkv_slice_matmul_idx, 0, qkv_add_idx, 1) &&
        is_connected(qkv_add_idx, 0, qkv_reshape_idx, 0))) {
    QNN_LOG_WARNING(
        "[G2G] Failed to check connectivity when doing MHA-SHA transformation "
        "for GQA decode.");
    return 1;
  }

  constexpr size_t kSupportedRank = 4;
  constexpr size_t kUnpackAxis = 1;
  // Guard the dimension reads below: every tensor we index by axis must have
  // the expected rank, otherwise GetDimension() may read past the end.
  if (ops[start_index].GetInputTensor(0).GetRank() != kSupportedRank ||
      ops[start_index + q_kcache_matmul_idx].GetInputTensor(1).GetRank() !=
          kSupportedRank ||
      ops[start_index + q_kslice_matmul_idx].GetInputTensor(1).GetRank() !=
          kSupportedRank ||
      ops[start_index + qkv_cache_matmul_idx].GetInputTensor(1).GetRank() !=
          kSupportedRank ||
      ops[start_index + qkv_slice_matmul_idx].GetInputTensor(1).GetRank() !=
          kSupportedRank) {
    QNN_LOG_WARNING(
        "[G2G] Failed to check ranks when doing MHA-SHA transformation "
        "for GQA decode.");
    return 1;
  }
  size_t split_index = 1;
  size_t num_q_heads =
      ops[start_index].GetInputTensor(0).GetDimension(split_index);
  if (num_q_heads == 1) {
    split_index = 2;
    num_q_heads = ops[start_index].GetInputTensor(0).GetDimension(split_index);
  }
  const size_t num_kv_heads =
      ops[start_index + q_kslice_matmul_idx].GetInputTensor(1).GetDimension(1);
  // Strict check: only well-known GQA shapes are supported. See
  // kSupportedGqaDecodeShapes.
  QNN_LOG_DEBUG(
      "[G2G] GQA Optimization (Decode):\n  # Q Heads: %zu\n  # KV Heads: %zu",
      num_q_heads, num_kv_heads);
  if (!IsSupportedGqaShape(kSupportedGqaDecodeShapes, num_q_heads,
                           num_kv_heads)) {
    return 1;
  }
  QNN_LOG_INFO("[G2G] GQA optimization (Decode)");

  std::vector<OpWrapper> new_ops;
  auto q_inputs = SplitTensor(tensor_pool, new_ops,
                              ops[start_index].GetInputTensor(0), split_index);
  auto k_cache_outputs = SplitTensor(
      tensor_pool, new_ops,
      ops[start_index + q_kcache_matmul_idx].GetInputTensor(1), kUnpackAxis);
  auto k_slice_outputs = SplitTensor(
      tensor_pool, new_ops,
      ops[start_index + q_kslice_matmul_idx].GetInputTensor(1), kUnpackAxis);
  auto v_cache_outputs = SplitTensor(
      tensor_pool, new_ops,
      ops[start_index + qkv_cache_matmul_idx].GetInputTensor(1), kUnpackAxis);
  auto v_slice_outputs = SplitTensor(
      tensor_pool, new_ops,
      ops[start_index + qkv_slice_matmul_idx].GetInputTensor(1), kUnpackAxis);

  // Build SHA
  const auto group_size = num_q_heads / num_kv_heads;
  std::vector<ConstTensorWrapperRef> sha_outputs;
  for (size_t i = 0; i < num_kv_heads; ++i) {
    for (size_t j = 0; j < group_size; ++j) {
      const auto& sha_output = BuildShaFromGqa(
          new_ops, tensor_pool, group_size, q_inputs[i * group_size + j],
          k_cache_outputs[i], k_slice_outputs[i], v_cache_outputs[i],
          v_slice_outputs[i],
          IsElementWiseMultiply(ops[start_index]) ? &ops[start_index] : nullptr,
          ops[start_index + q_kcache_matmul_idx],
          ops[start_index + q_kslice_matmul_idx], ops[start_index + qk_concat_idx],
          ops[start_index + mask_add_idx], ops[start_index + softmax_idx],
          ops[start_index + qkv_cache_slice_idx],
          ops[start_index + qkv_slice_slice_idx],
          ops[start_index + qkv_cache_matmul_idx],
          ops[start_index + qkv_slice_matmul_idx], ops[start_index + qkv_add_idx]);
      sha_outputs.emplace_back(sha_output);
    }
  }

  // Concat SHA outputs by the last dimension.
  const auto concat_axis = sha_outputs[0].get().GetRank() - 1;
  auto concat_sha_dims = sha_outputs[0].get().GetDimensions();
  concat_sha_dims[concat_axis] = 0;
  for (const auto& sha_output : sha_outputs) {
    concat_sha_dims[concat_axis] += sha_output.get().GetDimension(concat_axis);
  }
  const auto& concat_sha_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_index + qkv_reshape_idx].GetInputTensor(0), concat_sha_dims);
  new_ops.emplace_back(
      CreateConcatenationOp(sha_outputs, concat_sha_output, concat_axis));
  new_ops.emplace_back(CreateReshapeOp(
      concat_sha_output, ops[start_index + qkv_reshape_idx].GetOutputTensor(0)));
  // Clone namespace.
  CloneNamespace(ops[start_index], new_ops);
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
    QNN_LOG_INFO("[G2G] GQA optimization (Decode) done.");
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
  auto& new_transpose_0 = new_ops.emplace_back(
      CreateOpWithSameParams(transpoe_0, {pattern_input}, {transpose_output}));

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
      new_ops.emplace_back(CreateOpWithSameParams(
          ops[matmul_qk_index], {q_matmul_in, k_matmul_in}, {select_in}));

      // Change Select to Add.
      const auto& softmax_in =
          tensor_pool.CloneNativeTensorFrom(select_out, select_out_dims);
      new_ops.emplace_back(
          CreateElementWiseAddOp(select_in, add_in, softmax_in));

      // Softmax
      const auto& qk_softmax =
          tensor_pool.CloneNativeTensorFrom(softmax_in, select_out_dims);
      new_ops.emplace_back(CreateOpWithSameParams(
          ops[select_index + kAttnSoftmax], {softmax_in}, {qk_softmax}));

      // MatMul
      new_ops.emplace_back(CreateOpWithSameParams(ops[matmul_qk_index],
                                                  {qk_softmax, v_sha_inputs[i]},
                                                  {sha_outputs[i]}));
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
