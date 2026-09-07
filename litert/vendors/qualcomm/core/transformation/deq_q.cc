// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/deq_q.h"

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

size_t ConvertDequantizeQuantize(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  size_t dequantize_index = start_index - 1;
  if (!ops[dequantize_index].IsOpCode(QnnOpCode::kDequantize)) {
    dequantize_index--;
  }
  if (!ops[dequantize_index].IsOpCode(QnnOpCode::kDequantize)) {
    QNN_LOG_INFO("[G2G] Cannot find dequantize-quantize.")
    return 1;
  }
  auto& dequantize = ops[dequantize_index];
  auto& quantize = ops[start_index];

  const auto& pattern_input = dequantize.GetInputTensor(0);
  const auto& dequantize_output = dequantize.GetOutputTensor(0);
  const auto& quantize_input = quantize.GetInputTensor(0);
  const auto& pattern_output = quantize.GetOutputTensor(0);

  if (dequantize_output != quantize_input || !pattern_input.IsQuantI8() ||
      !dequantize_output.IsF32() || !pattern_output.IsQuantI16()) {
    return 1;
  }

  QNN_LOG_INFO("[G2G] Dequantize-quantize to Convert");
  QNN_LOG_INFO("[G2G]   Dequantize OP Name: %s", dequantize.GetName().data());
  QNN_LOG_INFO("[G2G]   Quantize OP Name: %s", quantize.GetName().data());
  auto new_convert = CreateConvertOp(pattern_input, pattern_output);
  if (validate_op_config(new_convert)) {
    ops.erase(ops.begin() + start_index);
    ops.erase(ops.begin() + dequantize_index);
    ops.emplace(ops.begin() + dequantize_index, std::move(new_convert));
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
  }
  return 1;
}

}  // namespace qnn
