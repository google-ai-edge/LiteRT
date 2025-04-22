// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/graph_to_graph.h"

#include <array>
#include <functional>
#include <vector>

#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/transformation/matmul_convert.h"
#include "litert/vendors/qualcomm/core/transformation/mha_to_sha.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace qnn {

namespace {

constexpr size_t kQnnOpCodeSize = static_cast<size_t>(QnnOpCode::kUnknown);

std::vector<size_t> CreateBadMatchTable(
    const std::vector<QnnOpCode>& pattern_ops) {
  std::vector<size_t> table(kQnnOpCodeSize, pattern_ops.size());
  for (size_t i = 0; i < pattern_ops.size() - 1; ++i) {
    table[static_cast<size_t>(pattern_ops[i])] = pattern_ops.size() - i - 1;
  }
  return table;
}

// Returns the start index of the matched operator pattern beginning at
// start_index; returns -1 if no match is found in the graph.
int GetPatternStartIndex(size_t start_index, const std::vector<OpWrapper>& ops,
                         const std::vector<QnnOpCode>& pattern_ops,
                         const std::vector<size_t>& bad_match_table) {
  size_t end_index = start_index + (pattern_ops.size() - 1);
  while (end_index < ops.size()) {
    bool found_pattern = true;
    for (size_t i = 0; i < pattern_ops.size(); ++i) {
      if (!ops[end_index - i].IsOpCode(
              pattern_ops[pattern_ops.size() - i - 1])) {
        found_pattern = false;
        break;
      }
    }
    if (found_pattern) {
      return end_index - (pattern_ops.size() - 1);
    } else {
      end_index +=
          bad_match_table[static_cast<size_t>(ops[end_index].GetOpCode())];
    }
  }
  return -1;
}

// Returns the number of indices to skip for the next pattern match.
// This function attempts to transform a specific pattern into optimized one
// and returns the size of the skippable indices for the next index check.
typedef size_t (*G2GTransform)(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size);
void Transform(std::function<bool(OpWrapper&)> validate_op_config,
               std::vector<OpWrapper>& ops, TensorPool& tensor_pool,
               const std::vector<QnnOpCode>& pattern_ops,
               G2GTransform custom_transform) {
  auto bad_match_table = CreateBadMatchTable(pattern_ops);
  size_t start_index = 0;
  while ((start_index + (pattern_ops.size() - 1)) < ops.size()) {
    if (auto pattern_start_index = GetPatternStartIndex(
            start_index, ops, pattern_ops, bad_match_table);
        pattern_start_index != -1) {
      start_index +=
          custom_transform(validate_op_config, ops, pattern_start_index,
                           tensor_pool, pattern_ops.size());
    } else {
      break;
    }
  }
}

}  // namespace

// TODO (jiunkaiy): Add more G2G transformation.
void GraphToGraphTransform(const G2GConfig g2g_option,
                           std::vector<OpWrapper>& ops, TensorPool& tensor_pool,
                           std::function<bool(OpWrapper&)> validate_op_config) {
  if (g2g_option == G2GConfig::kOff) {
    return;
  }

  // MatMul-convert Fusion
  if (g2g_option == G2GConfig::kMatMulConvert ||
      g2g_option == G2GConfig::kMHAOptPrefill ||
      g2g_option == G2GConfig::kMHAOpt) {
    const std::vector<QnnOpCode> matmul_convert_decode = {
        QnnOpCode::kMatMul,
        QnnOpCode::kConvert,
    };
    Transform(validate_op_config, ops, tensor_pool, matmul_convert_decode,
              FuseMatMulConvertDecode);
    const std::vector<QnnOpCode> matmul_convert_prefill = {
        QnnOpCode::kMatMul,
        QnnOpCode::kMatMul,
        QnnOpCode::kConvert,
    };
    Transform(validate_op_config, ops, tensor_pool, matmul_convert_prefill,
              FuseMatMulConvertPrefill);
  }
  // MHA Optimization
  if (g2g_option == G2GConfig::kMHAOpt) {
    const std::vector<QnnOpCode> gemma3_mha_decode = {
        QnnOpCode::kElementWiseMultiply,
        QnnOpCode::kMatMul,
        QnnOpCode::kMatMul,
        QnnOpCode::kConcat,
        QnnOpCode::kReshape,
        QnnOpCode::kElementWiseAdd,
        QnnOpCode::kReshape,
        QnnOpCode::kSoftmax,
        QnnOpCode::kStridedSlice,
        QnnOpCode::kStridedSlice,
        QnnOpCode::kMatMul,
        QnnOpCode::kMatMul,
        QnnOpCode::kElementWiseAdd,
        QnnOpCode::kReshape,
    };
    Transform(validate_op_config, ops, tensor_pool, gemma3_mha_decode,
              OptimizeMHADecode);
  }
  if (g2g_option == G2GConfig::kMHAOptPrefill ||
      g2g_option == G2GConfig::kMHAOpt) {
    const std::vector<QnnOpCode> gemma3_mha_prefill = {
        QnnOpCode::kElementWiseMultiply,
        QnnOpCode::kTranspose,
        QnnOpCode::kReshape,
        QnnOpCode::kMatMul,
        QnnOpCode::kMatMul,
        QnnOpCode::kConcat,
        QnnOpCode::kReshape,
        QnnOpCode::kElementWiseAdd,
        QnnOpCode::kReshape,
        QnnOpCode::kSoftmax,
        QnnOpCode::kStridedSlice,
        QnnOpCode::kStridedSlice,
        QnnOpCode::kMatMul,
        QnnOpCode::kMatMul,
        QnnOpCode::kElementWiseAdd,
        QnnOpCode::kReshape,
        QnnOpCode::kTranspose,
        QnnOpCode::kReshape,
    };
    Transform(validate_op_config, ops, tensor_pool, gemma3_mha_prefill,
              OptimizeMHAPrefill);
  }
}
}  // namespace qnn
