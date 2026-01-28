// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/matmul_convert.h"

#include <cstddef>
#include <functional>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

size_t FuseMatMulConvertDecode(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  auto& matmul = ops[start_index];
  auto& convert = ops[start_index + 1];
  // Connection check
  if (matmul.GetOutputTensor(0) != convert.GetInputTensor(0)) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] MatMul-convert fusion (Decode)");
  auto new_matmul = CreateMatmulOpWithSameParam(
      matmul, matmul.GetInputTensor(0), matmul.GetInputTensor(1),
      convert.GetOutputTensor(0));
  if (validate_op_config(new_matmul)) {
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);
    ops.emplace(ops.begin() + start_index, std::move(new_matmul));
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
  }
  return 1;
}

size_t FuseMatMulConvertPrefill(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  auto& matmul = ops[start_index];
  auto& convert = ops[start_index + 2];
  // Connection check
  if (matmul.GetOutputTensor(0) != convert.GetInputTensor(0)) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] MatMul-convert fusion (Prefill)");
  auto new_matmul = CreateMatmulOpWithSameParam(
      matmul, matmul.GetInputTensor(0), matmul.GetInputTensor(1),
      convert.GetOutputTensor(0));
  if (validate_op_config(new_matmul)) {
    ops.erase(ops.begin() + start_index + 2);
    ops.erase(ops.begin() + start_index);
    ops.emplace(ops.begin() + start_index, std::move(new_matmul));
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
  }
  return 1;
}

}  // namespace qnn
