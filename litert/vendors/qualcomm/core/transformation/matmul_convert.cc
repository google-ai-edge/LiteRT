// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/matmul_convert.h"

#include <array>
#include <vector>

#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

bool FuseMatMulConvertDecode(std::vector<OpWrapper>& ops, size_t start_id,
                             TensorPool& tensor_pool, size_t pattern_size) {
  if (&ops[start_id].GetOutputTensor(0) ==
      &ops[start_id + 1].GetInputTensor(0)) {
    ops[start_id].StealOutputs(ops[start_id + 1]);
    ops.erase(ops.begin() + start_id + 1);
    QNN_LOG_INFO("[G2G] MatMul-convert fusion (Decode)");
    return true;
  } else {
    return false;
  }
}

bool FuseMatMulConvertPrefill(std::vector<OpWrapper>& ops, size_t start_id,
                              TensorPool& tensor_pool, size_t pattern_size) {
  if (&ops[start_id].GetOutputTensor(0) ==
      &ops[start_id + 2].GetInputTensor(0)) {
    ops[start_id].StealOutputs(ops[start_id + 2]);
    ops.erase(ops.begin() + start_id + 2);
    QNN_LOG_INFO("[G2G] MatMul-convert fusion (Prefill)");
    return true;
  } else {
    return false;
  }
}

}  // namespace qnn
