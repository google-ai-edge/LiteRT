// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/onehot_fc.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/embedding_lookup_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {

bool IsOneHotFullyConnectedBranch(const std::vector<OpWrapper>& ops,
                                  size_t start_index, size_t slice,
                                  size_t indices, size_t one_hot, size_t less,
                                  size_t greater_equal, size_t logical_or,
                                  size_t not_equal, size_t logical_and,
                                  size_t broadcast, size_t select, size_t fc,
                                  size_t fc_reshape, size_t model_reshape) {
  const auto& slice_op = ops[start_index + slice];
  const auto& indices_op = ops[start_index + indices];
  const auto& one_hot_op = ops[start_index + one_hot];
  const auto& less_op = ops[start_index + less];
  const auto& greater_equal_op = ops[start_index + greater_equal];
  const auto& or_op = ops[start_index + logical_or];
  const auto& not_equal_op = ops[start_index + not_equal];
  const auto& and_op = ops[start_index + logical_and];
  const auto& broadcast_op = ops[start_index + broadcast];
  const auto& select_op = ops[start_index + select];
  const auto& fc_op = ops[start_index + fc];
  const auto& fc_reshape_op = ops[start_index + fc_reshape];
  const auto& model_reshape_op = ops[start_index + model_reshape];

  return slice_op.GetOutputTensor(0) == indices_op.GetInputTensor(0) &&
         indices_op.GetOutputTensor(0) == one_hot_op.GetInputTensor(0) &&
         indices_op.GetOutputTensor(0) == less_op.GetInputTensor(0) &&
         indices_op.GetOutputTensor(0) == greater_equal_op.GetInputTensor(0) &&
         less_op.GetOutputTensor(0) == or_op.GetInputTensor(0) &&
         greater_equal_op.GetOutputTensor(0) == or_op.GetInputTensor(1) &&
         indices_op.GetOutputTensor(0) == not_equal_op.GetInputTensor(0) &&
         or_op.GetOutputTensor(0) == and_op.GetInputTensor(0) &&
         not_equal_op.GetOutputTensor(0) == and_op.GetInputTensor(1) &&
         and_op.GetOutputTensor(0) == broadcast_op.GetInputTensor(0) &&
         broadcast_op.GetOutputTensor(0) == select_op.GetInputTensor(0) &&
         one_hot_op.GetOutputTensor(0) == select_op.GetInputTensor(2) &&
         fc_op.GetInputCount() == 2 &&
         select_op.GetOutputTensor(0) == fc_op.GetInputTensor(0) &&
         fc_op.GetOutputTensor(0) == fc_reshape_op.GetInputTensor(0) &&
         fc_reshape_op.GetOutputTensor(0) ==
             model_reshape_op.GetInputTensor(0) &&
         IsElementWiseLess(less_op) &&
         IsElementWiseGreaterEqual(greater_equal_op) &&
         IsElementWiseOr(or_op) && IsElementWiseNotEqual(not_equal_op) &&
         IsElementWiseAnd(and_op);
}

std::optional<OpWrapper> CreateGatherFromFc(TensorPool& tensor_pool,
                                            const OpWrapper& indices_op,
                                            const OpWrapper& fc_op,
                                            const TensorWrapper& output) {
  const auto& indices = indices_op.GetOutputTensor(0);
  const auto& weight = fc_op.GetInputTensor(1);
  if (!weight.IsTensorStatic() || !weight.IsF32() || weight.GetRank() != 2 ||
      output.GetRank() != indices.GetRank() + weight.GetRank() - 1 ||
      output.GetDimension(output.GetRank() - 1) != weight.GetDimension(0) ||
      indices.GetTensorNumElements() !=
          output.GetTensorNumElements() / weight.GetDimension(0)) {
    QNN_LOG_WARNING("[G2G] OneHot-FC: unsupported float32 Gather dimensions.");
    return std::nullopt;
  }

  const auto weight_data = weight.GetTensorData<float>();
  if (!weight_data.has_value()) {
    QNN_LOG_WARNING("[G2G] OneHot-FC: failed to read float FC weight data.");
    return std::nullopt;
  }

  const uint32_t rows = weight.GetDimension(0);
  const uint32_t cols = weight.GetDimension(1);
  std::vector<float> transposed_weight(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      transposed_weight[col * rows + row] = (*weight_data)[row * cols + col];
    }
  }

  const auto& table = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, weight.GetQuantParams(), {cols, rows},
      transposed_weight.size() * sizeof(transposed_weight[0]),
      transposed_weight.data());
  return CreateGatherOp(table, indices, output, /*axis=*/0);
}

}  // namespace

size_t TransformOneHotFc(std::function<bool(OpWrapper&)> validate_op_config,
                         std::vector<OpWrapper>& ops, size_t start_index,
                         TensorPool& tensor_pool, size_t pattern_size) {
  constexpr size_t kFirstSlice = 0;
  constexpr size_t kFirstIndices = 1;
  constexpr size_t kFirstOneHot = 2;
  constexpr size_t kFirstLess = 3;
  constexpr size_t kFirstGreaterEqual = 4;
  constexpr size_t kFirstOr = 5;
  constexpr size_t kFirstNotEqual = 6;
  constexpr size_t kFirstAnd = 7;
  constexpr size_t kFirstBroadcast = 8;
  constexpr size_t kFirstSelect = 9;
  constexpr size_t kSecondSlice = 10;
  constexpr size_t kSecondIndices = 11;
  constexpr size_t kSecondOneHot = 12;
  constexpr size_t kSecondLess = 13;
  constexpr size_t kSecondGreaterEqual = 14;
  constexpr size_t kSecondOr = 15;
  constexpr size_t kSecondNotEqual = 16;
  constexpr size_t kSecondAnd = 17;
  constexpr size_t kSecondBroadcast = 18;
  constexpr size_t kSecondSelect = 19;
  constexpr size_t kFirstFc = 20;
  constexpr size_t kFirstFcReshape = 21;
  constexpr size_t kFirstModelReshape = 22;
  constexpr size_t kSecondFc = 23;
  constexpr size_t kSecondFcReshape = 24;
  constexpr size_t kSecondModelReshape = 25;
  constexpr size_t kPatternSize = 26;

  if (pattern_size != kPatternSize) {
    QNN_LOG_WARNING("[G2G] OneHot-FC: unsupported pattern size %zu.",
                    pattern_size);
    return 1;
  }
  QNN_LOG_INFO("[G2G] OneHot-FC pattern matched at op %zu; validating.",
               start_index);
  if (!IsOneHotFullyConnectedBranch(
          ops, start_index, kFirstSlice, kFirstIndices, kFirstOneHot,
          kFirstLess, kFirstGreaterEqual, kFirstOr, kFirstNotEqual, kFirstAnd,
          kFirstBroadcast, kFirstSelect, kFirstFc, kFirstFcReshape,
          kFirstModelReshape) ||
      !IsOneHotFullyConnectedBranch(
          ops, start_index, kSecondSlice, kSecondIndices, kSecondOneHot,
          kSecondLess, kSecondGreaterEqual, kSecondOr, kSecondNotEqual,
          kSecondAnd, kSecondBroadcast, kSecondSelect, kSecondFc,
          kSecondFcReshape, kSecondModelReshape)) {
    QNN_LOG_WARNING("[G2G] OneHot-FC: branch connectivity check failed.");
    return 1;
  }

  const auto& first_fc_op = ops[start_index + kFirstFc];
  const auto& second_fc_op = ops[start_index + kSecondFc];
  const auto& first_gather_output =
      ops[start_index + kFirstFcReshape].GetOutputTensor(0);
  const auto& second_gather_output =
      ops[start_index + kSecondFcReshape].GetOutputTensor(0);

  auto first_gather =
      CreateGatherFromFc(tensor_pool, ops[start_index + kFirstIndices],
                         first_fc_op, first_gather_output);
  auto second_gather =
      CreateGatherFromFc(tensor_pool, ops[start_index + kSecondIndices],
                         second_fc_op, second_gather_output);
  if (!first_gather || !second_gather) {
    QNN_LOG_WARNING("[G2G] OneHot-FC: failed to create Gather ops.");
    return 1;
  }
  auto first_reshape =
      CreateReshapeOp(first_gather_output,
                      ops[start_index + kFirstModelReshape].GetOutputTensor(0));
  auto second_reshape = CreateReshapeOp(
      second_gather_output,
      ops[start_index + kSecondModelReshape].GetOutputTensor(0));
  if (!validate_op_config(*first_gather) ||
      !validate_op_config(*second_gather) ||
      !validate_op_config(first_reshape) ||
      !validate_op_config(second_reshape)) {
    QNN_LOG_WARNING("[G2G] OneHot-FC: Gather validation failed.");
    return 1;
  }

  QNN_LOG_INFO("[G2G] OneHot-FC: transforming both branches to Gather.");
  std::vector<OpWrapper> new_ops;
  new_ops.emplace_back(ops[start_index + kFirstSlice]);
  new_ops.emplace_back(ops[start_index + kFirstIndices]);
  new_ops.emplace_back(std::move(*first_gather));
  new_ops.emplace_back(std::move(first_reshape));
  new_ops.emplace_back(ops[start_index + kSecondSlice]);
  new_ops.emplace_back(ops[start_index + kSecondIndices]);
  new_ops.emplace_back(std::move(*second_gather));
  new_ops.emplace_back(std::move(second_reshape));
  ops.erase(ops.begin() + start_index,
            ops.begin() + start_index + pattern_size);
  ops.insert(ops.begin() + start_index,
             std::make_move_iterator(new_ops.begin()),
             std::make_move_iterator(new_ops.end()));
  QNN_LOG_INFO("[G2G] OneHot-FC: transform completed.");
  return new_ops.size();
}

}  // namespace qnn
