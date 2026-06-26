// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/matmul_convert.h"

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
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
  auto new_matmul = CreateOpWithSameParams(
      matmul, {matmul.GetInputTensor(0), matmul.GetInputTensor(1)},
      {convert.GetOutputTensor(0)});
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
  auto new_matmul = CreateOpWithSameParams(
      matmul, {matmul.GetInputTensor(0), matmul.GetInputTensor(1)},
      {convert.GetOutputTensor(0)});
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

size_t FuseConvertMatMulPrefill(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  auto& convert = ops[start_index];
  auto& matmul = ops[start_index+1];
  // Connection check
  if (convert.GetOutputTensor(0) != matmul.GetInputTensor(1)) {
    return 1;
  }
  // Safety check: the Convert's output must be used only by this MatMul.
  // If it's shared across multiple consumers, removing the Convert would
  // orphan the other consumers (e.g. when the same K-slice tensor feeds
  // multiple attention head MatMuls).
  const TensorWrapper& convert_output = convert.GetOutputTensor(0);
  for (size_t i = 0; i < ops.size(); ++i) {
    if (i == start_index || i == start_index + 1) {
      continue;  // skip the Convert itself and the MatMul being fused
    }
    for (const auto& t : ops[i].GetAllTensors()) {
      if (&t.get() == &convert_output) {
        QNN_LOG_WARNING(
            "[G2G] Convert-MatMul fusion (Prefill) skipped: Convert output "
            "is used by multiple consumers.");
        return 1;
      }
    }
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] Convert-MatMul fusion (Prefill)");
  auto new_matmul = CreateOpWithSameParams(
      matmul, {matmul.GetInputTensor(0), convert.GetInputTensor(0)},
      {matmul.GetOutputTensor(0)});
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

}  // namespace qnn
