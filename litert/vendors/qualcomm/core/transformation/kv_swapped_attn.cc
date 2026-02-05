// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/unpack_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/transformation/mha_to_sha.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
size_t OptimizeKvSwappedFastVlmPrefill(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  static constexpr size_t kQScaleMulIdx = 0;
  static constexpr size_t kQScaleReshapeIdx = 1;
  static constexpr size_t kQKCacheMatmulIdx = 2;
  static constexpr size_t kQKSliceMatmulIdx = 3;
  static constexpr size_t kQKConcatIdx = 4;
  static constexpr size_t kPreMaskReshapeIdx = 5;
  static constexpr size_t kMaskAddIdx = 6;
  static constexpr size_t kPostMaskReshapeIdx = 7;
  static constexpr size_t kSoftmaxIdx = 8;
  static constexpr size_t kQKVCacheSliceIdx = 9;
  static constexpr size_t kQKVSliceSliceIdx = 10;
  static constexpr size_t kQKVCacheMatmulIdx = 11;
  static constexpr size_t kQKVSliceMatmulIdx = 12;
  static constexpr size_t kQKVAddIdx = 13;
  static constexpr size_t kQKVReshapeIdx = 14;
  static constexpr size_t kQKVTransposeIdx = 15;
  static constexpr size_t kOProjReshapeIdx = 16;

  const auto is_connected =
      [&ops, &start_index](size_t output_op_index, size_t output_tensor_index,
                           size_t input_op_index,
                           size_t input_tensor_index) -> bool {
    return ops[start_index + output_op_index].GetOutputTensor(
               output_tensor_index) ==
           ops[start_index + input_op_index].GetInputTensor(input_tensor_index);
  };

  if (!(is_connected(kQScaleMulIdx, 0, kQScaleReshapeIdx, 0) &&
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
        is_connected(kQKVTransposeIdx, 0, kOProjReshapeIdx, 0))) {
    return 1;
  }
  QNN_LOG_INFO("[G2G] Kv-swapped attention pattern matched.");

  std::vector<OpWrapper> new_ops;

  const auto& k_cache = ops[start_index + kQKCacheMatmulIdx].GetInputTensor(1);
  auto k_cache_unpack_outputs = UnpackTensor(tensor_pool, new_ops, k_cache);

  const auto& k_slice = ops[start_index + kQKSliceMatmulIdx].GetInputTensor(1);
  auto k_slice_unpack_outputs = UnpackTensor(tensor_pool, new_ops, k_slice);

  const auto& scale_mul_in = ops[start_index + kQScaleMulIdx].GetInputTensor(0);
  auto scale_mul_unpack_outputs = UnpackTensor(tensor_pool, new_ops, scale_mul_in);

  const auto& v_cache = ops[start_index + kQKVCacheMatmulIdx].GetInputTensor(1);
  auto v_cache_unpack_outputs = UnpackTensor(tensor_pool, new_ops, v_cache);

  const auto& v_slice = ops[start_index + kQKVSliceMatmulIdx].GetInputTensor(1);
  auto v_slice_unpack_outputs = UnpackTensor(tensor_pool, new_ops, v_slice);

  auto num_kv_heads = k_cache_unpack_outputs.size();
  auto num_attn_heads = scale_mul_unpack_outputs.size();
  auto num_attn_per_kv_heads = num_attn_heads / num_kv_heads;

  if (!(num_kv_heads == k_cache.GetDimension(1) &&
        num_kv_heads == k_slice.GetDimension(1) &&
        num_kv_heads == v_cache.GetDimension(1) &&
        num_kv_heads == v_slice.GetDimension(1))) {
    QNN_LOG_WARNING(
        "[G2G] num_kv heads: %d does not match heads in [k_cache: %d, "
        "k_slice: %d, v_cache: %d, v_slice: %d]",
        num_kv_heads, k_cache.GetDimension(1), k_slice.GetDimension(1),
        v_cache.GetDimension(1), v_slice.GetDimension(1));
    return 1;
  }
  // build num_head SHAs
  std::vector<TensorWrapperRef> sha_outputs;
  sha_outputs.reserve(num_attn_heads);
  for (size_t i = 0; i < num_kv_heads; ++i) {
    for (size_t j = 0; j < num_attn_per_kv_heads; ++j) {
      auto& sha_output = BuildSingleSHAByUnpackAxis1(
          new_ops, tensor_pool, num_attn_per_kv_heads,
          scale_mul_unpack_outputs[i * num_attn_per_kv_heads + j],
          k_cache_unpack_outputs[i], k_slice_unpack_outputs[i],
          v_cache_unpack_outputs[i], v_slice_unpack_outputs[i],
          ops[start_index + kQScaleMulIdx], ops[start_index + kQKCacheMatmulIdx],
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
    QNN_LOG_INFO("[G2G] Optimized kv-swapped attention.");
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}
}  // namespace qnn
