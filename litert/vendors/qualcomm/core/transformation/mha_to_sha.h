// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MHA_TO_SHA_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MHA_TO_SHA_H_

#include <cstddef>
#include <functional>
#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {
std::vector<TensorWrapperRef> UnpackTensor(TensorPool& tensor_pool,
                                           std::vector<OpWrapper>& new_ops,
                                           const TensorWrapper& input,
                                           size_t unpack_dims = 1);

TensorWrapper& BuildSingleSHAByUnpackAxis1(
    std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
    const uint32_t num_attn_per_kv_heads, TensorWrapper& scale_mul_input,
    TensorWrapper& k_cache, TensorWrapper& k_slice, TensorWrapper& v_cache,
    TensorWrapper& v_slice, const OpWrapper& scale_mul,
    const OpWrapper& q_kcache_matmul, const OpWrapper& q_kslice_matmul,
    const OpWrapper& qk_concat, const OpWrapper& mask_add,
    const OpWrapper& post_mask_reshape, const OpWrapper& softmax,
    const OpWrapper& qk_vcache_slice, const OpWrapper& qk_vslice_slice,
    const OpWrapper& qk_vcache_matmul, const OpWrapper& qk_vslice_matmul,
    const OpWrapper& qkv_add);

size_t OptimizeMHAPrefill(std::function<bool(OpWrapper&)> validate_op_config,
                          std::vector<OpWrapper>& ops, size_t start_index,
                          TensorPool& tensor_pool, size_t pattern_size);

size_t OptimizeMHADecode(std::function<bool(OpWrapper&)> validate_op_config,
                         std::vector<OpWrapper>& ops, size_t start_index,
                         TensorPool& tensor_pool, size_t pattern_size);

size_t OptimizeMHAFastVlmPrefill(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size);

size_t OptimizeMHATinyGemmaPrefillPatternWithGlobalMask(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size);

size_t OptimizeMHATinyGemmaPrefillPattern(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size);

size_t OptimizeMHAAttn(std::function<bool(OpWrapper&)> validate_op_config,
                       std::vector<OpWrapper>& ops, size_t start_index,
                       TensorPool& tensor_pool, size_t pattern_size);
}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MHA_TO_SHA_H_
