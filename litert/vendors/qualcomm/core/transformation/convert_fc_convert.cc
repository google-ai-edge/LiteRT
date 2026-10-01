// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/convert_fc_convert.h"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/utils/log.h"

namespace qnn {
namespace {

bool IsFullyConnectedBranch(const OpWrapper& input_convert, const OpWrapper& fc,
                            const OpWrapper& fc_reshape,
                            const OpWrapper& reshape,
                            const OpWrapper& output_convert,
                            const OpWrapper& rms_norm) {
  return fc.GetInputCount() >= 2 && input_convert.GetOutputTensor(0) ==
                                       fc.GetInputTensor(0) &&
         fc.GetOutputTensor(0) == fc_reshape.GetInputTensor(0) &&
         fc_reshape.GetOutputTensor(0) == reshape.GetInputTensor(0) &&
         reshape.GetOutputTensor(0) == output_convert.GetInputTensor(0) &&
         output_convert.GetOutputTensor(0) == rms_norm.GetInputTensor(0) &&
         input_convert.GetInputTensor(0).IsQuantI16() &&
         output_convert.GetOutputTensor(0).IsQuantI16();
}

void AddFullyConnectedBranch(std::vector<OpWrapper>& new_ops,
                             TensorPool& tensor_pool,
                             const TensorWrapper& int16_activation,
                             const OpWrapper& fc,
                             const OpWrapper& fc_reshape,
                             const OpWrapper& reshape,
                             const OpWrapper& output_convert,
                             const OpWrapper& rms_norm) {
  std::vector<ConstTensorWrapperRef> fc_inputs;
  fc_inputs.reserve(fc.GetInputCount());
  fc_inputs.emplace_back(int16_activation);
  for (size_t index = 1; index < fc.GetInputCount(); ++index) {
    fc_inputs.emplace_back(fc.GetInputTensor(index));
  }
  const auto& fc_output = tensor_pool.CloneNativeTensorFrom(
      output_convert.GetOutputTensor(0), fc.GetOutputTensor(0).GetDimensions());
  new_ops.emplace_back(CreateOpWithSameParams(fc, fc_inputs, {fc_output}));
  const auto& fc_reshape_output = tensor_pool.CloneNativeTensorFrom(
      output_convert.GetOutputTensor(0),
      fc_reshape.GetOutputTensor(0).GetDimensions());
  new_ops.emplace_back(CreateReshapeOp(fc_output, fc_reshape_output));
  new_ops.emplace_back(
      CreateReshapeOp(fc_reshape_output, output_convert.GetOutputTensor(0)));
  new_ops.emplace_back(rms_norm);
}

bool IsFcConvertBranch(const OpWrapper& input_convert, const OpWrapper& fc,
                       const OpWrapper& fc_reshape,
                       const OpWrapper& output_convert) {
  return fc.GetInputCount() >= 2 &&
         input_convert.GetOutputTensor(0) == fc.GetInputTensor(0) &&
         fc.GetOutputTensor(0) == fc_reshape.GetInputTensor(0) &&
         fc_reshape.GetOutputTensor(0) == output_convert.GetInputTensor(0) &&
         output_convert.GetOutputTensor(0).IsQuantI16();
}

void AddFcConvertBranch(std::vector<OpWrapper>& new_ops, TensorPool& tensor_pool,
                        const TensorWrapper& int16_activation,
                        const OpWrapper& fc, const OpWrapper& fc_reshape,
                        const OpWrapper& output_convert) {
  std::vector<ConstTensorWrapperRef> fc_inputs;
  fc_inputs.emplace_back(int16_activation);
  for (size_t index = 1; index < fc.GetInputCount(); ++index) {
    fc_inputs.emplace_back(fc.GetInputTensor(index));
  }
  const auto& fc_output = tensor_pool.CloneNativeTensorFrom(
      output_convert.GetOutputTensor(0), fc.GetOutputTensor(0).GetDimensions());
  new_ops.emplace_back(CreateOpWithSameParams(fc, fc_inputs, {fc_output}));
  new_ops.emplace_back(
      CreateReshapeOp(fc_output, output_convert.GetOutputTensor(0)));
}

bool Replace(std::function<bool(OpWrapper&)> validate_op_config,
             std::vector<OpWrapper>& ops, size_t start_index,
             size_t pattern_size, std::vector<OpWrapper>& new_ops) {
  if (!std::all_of(new_ops.begin(), new_ops.end(), validate_op_config)) {
    return false;
  }
  ops.erase(ops.begin() + start_index,
            ops.begin() + start_index + pattern_size);
  ops.insert(ops.begin() + start_index,
             std::make_move_iterator(new_ops.begin()),
             std::make_move_iterator(new_ops.end()));
  return true;
}

}  // namespace

size_t TransformConvertFcConvert(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr size_t kInputConvert = 0;
  constexpr size_t kFirstFc = 1;
  constexpr size_t kFirstFcReshape = 2;
  constexpr size_t kFirstReshape = 3;
  constexpr size_t kSecondFc = 4;
  constexpr size_t kSecondFcReshape = 5;
  constexpr size_t kSecondReshape = 6;
  constexpr size_t kThirdFc = 7;
  constexpr size_t kThirdFcReshape = 8;
  constexpr size_t kThirdReshape = 9;
  constexpr size_t kFirstOutputConvert = 10;
  constexpr size_t kFirstRmsNorm = 11;
  constexpr size_t kSecondOutputConvert = 12;
  constexpr size_t kSecondRmsNorm = 13;
  constexpr size_t kThirdOutputConvert = 14;
  constexpr size_t kThirdRmsNorm = 15;
  constexpr size_t kPatternSize = 16;

  if (pattern_size != kPatternSize) return 1;
  const auto& input_convert = ops[start_index + kInputConvert];
  const auto& first_fc = ops[start_index + kFirstFc];
  const auto& second_fc = ops[start_index + kSecondFc];
  const auto& third_fc = ops[start_index + kThirdFc];
  const auto& first_output_convert = ops[start_index + kFirstOutputConvert];
  const auto& second_output_convert = ops[start_index + kSecondOutputConvert];
  const auto& third_output_convert = ops[start_index + kThirdOutputConvert];
  if (!IsFullyConnectedBranch(
          input_convert, first_fc, ops[start_index + kFirstFcReshape],
          ops[start_index + kFirstReshape], first_output_convert,
          ops[start_index + kFirstRmsNorm]) ||
      !IsFullyConnectedBranch(
          input_convert, second_fc, ops[start_index + kSecondFcReshape],
          ops[start_index + kSecondReshape], second_output_convert,
          ops[start_index + kSecondRmsNorm]) ||
      !IsFullyConnectedBranch(
          input_convert, third_fc, ops[start_index + kThirdFcReshape],
          ops[start_index + kThirdReshape], third_output_convert,
          ops[start_index + kThirdRmsNorm])) {
    QNN_LOG_WARNING("[G2G] Convert-FC-Convert: connectivity check failed.");
    return 1;
  }

  QNN_LOG_INFO("[G2G] Convert-FC-Convert pattern matched at op %zu.",
               start_index);
  const auto& int16_activation = input_convert.GetInputTensor(0);
  std::vector<OpWrapper> new_ops;
  AddFullyConnectedBranch(
      new_ops, tensor_pool, int16_activation, first_fc,
      ops[start_index + kFirstFcReshape], ops[start_index + kFirstReshape],
      first_output_convert, ops[start_index + kFirstRmsNorm]);
  AddFullyConnectedBranch(
      new_ops, tensor_pool, int16_activation, second_fc,
      ops[start_index + kSecondFcReshape], ops[start_index + kSecondReshape],
      second_output_convert, ops[start_index + kSecondRmsNorm]);
  AddFullyConnectedBranch(
      new_ops, tensor_pool, int16_activation, third_fc,
      ops[start_index + kThirdFcReshape], ops[start_index + kThirdReshape],
      third_output_convert, ops[start_index + kThirdRmsNorm]);
  if (!std::all_of(new_ops.begin(), new_ops.end(), validate_op_config)) {
    QNN_LOG_WARNING("[G2G] Convert-FC-Convert: transformed ops failed validation.");
    return 1;
  }
  ops.erase(ops.begin() + start_index,
            ops.begin() + start_index + pattern_size);
  ops.insert(ops.begin() + start_index,
             std::make_move_iterator(new_ops.begin()),
             std::make_move_iterator(new_ops.end()));
  return new_ops.size();
}

size_t TransformConvertReshapeFcConvert(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr size_t kPatternSize = 5;
  if (pattern_size != kPatternSize) return 1;
  const auto& input_convert = ops[start_index];
  const auto& reshape = ops[start_index + 1];
  const auto& fc = ops[start_index + 2];
  const auto& fc_reshape = ops[start_index + 3];
  const auto& output_convert = ops[start_index + 4];
  QNN_LOG_INFO(
      "[G2G] Convert-Reshape-FC-Convert pattern matched at op %zu.",
      start_index);
  if (input_convert.GetInputTensor(0).IsQuantI16() &&
      input_convert.GetOutputTensor(0) == reshape.GetInputTensor(0) &&
      reshape.GetOutputTensor(0) == fc.GetInputTensor(0) &&
      IsFcConvertBranch(reshape, fc, fc_reshape, output_convert)) {
    const auto& int16_reshape = tensor_pool.CloneNativeTensorFrom(
        output_convert.GetOutputTensor(0), reshape.GetOutputTensor(0).GetDimensions());
    std::vector<OpWrapper> new_ops;
    new_ops.emplace_back(CreateReshapeOp(input_convert.GetInputTensor(0),
                                         int16_reshape));
    AddFcConvertBranch(new_ops, tensor_pool, int16_reshape, fc, fc_reshape,
                       output_convert);
    if (Replace(validate_op_config, ops, start_index, pattern_size, new_ops)) {
      QNN_LOG_INFO("[G2G] Convert-Reshape-FC-Convert transformed.");
      return new_ops.size();
    }
  }
  return 1;
}

size_t TransformConvertTwoFcConvertGeluConvert(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr size_t kPatternSize = 8;
  if (pattern_size != kPatternSize) return 1;
  const auto& input_convert = ops[start_index];
  const auto& first_fc = ops[start_index + 1];
  const auto& first_fc_reshape = ops[start_index + 2];
  const auto& second_fc = ops[start_index + 3];
  const auto& second_fc_reshape = ops[start_index + 4];
  const auto& first_output_convert = ops[start_index + 5];
  const auto& gelu = ops[start_index + 6];
  const auto& second_output_convert = ops[start_index + 7];
  QNN_LOG_INFO(
      "[G2G] Convert-Two-FC-Convert-GELU-Convert pattern matched at op %zu.",
      start_index);
  if (!input_convert.GetInputTensor(0).IsQuantI16() ||
      !IsFcConvertBranch(input_convert, first_fc, first_fc_reshape,
                         first_output_convert) ||
      !IsFcConvertBranch(input_convert, second_fc, second_fc_reshape,
                         second_output_convert) ||
      first_output_convert.GetOutputTensor(0) != gelu.GetInputTensor(0)) {
    return 1;
  }
  std::vector<OpWrapper> new_ops;
  AddFcConvertBranch(new_ops, tensor_pool, input_convert.GetInputTensor(0),
                     first_fc, first_fc_reshape, first_output_convert);
  new_ops.emplace_back(gelu);
  AddFcConvertBranch(new_ops, tensor_pool, input_convert.GetInputTensor(0),
                     second_fc, second_fc_reshape, second_output_convert);
  if (Replace(validate_op_config, ops, start_index, pattern_size, new_ops)) {
    QNN_LOG_INFO("[G2G] Convert-Two-FC-Convert-GELU-Convert transformed.");
    return new_ops.size();
  }
  return 1;
}

size_t TransformConvertFcReshapeConvert(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
  constexpr size_t kPatternSize = 4;
  if (pattern_size != kPatternSize) return 1;
  const auto& input_convert = ops[start_index];
  const auto& fc = ops[start_index + 1];
  const auto& fc_reshape = ops[start_index + 2];
  const auto& output_convert = ops[start_index + 3];
  QNN_LOG_INFO("[G2G] Convert-FC-Reshape-Convert pattern matched at op %zu.",
               start_index);
  if (!input_convert.GetInputTensor(0).IsQuantI16() ||
      !IsFcConvertBranch(input_convert, fc, fc_reshape, output_convert)) {
    return 1;
  }
  std::vector<OpWrapper> new_ops;
  AddFcConvertBranch(new_ops, tensor_pool, input_convert.GetInputTensor(0), fc,
                     fc_reshape, output_convert);
  if (Replace(validate_op_config, ops, start_index, pattern_size, new_ops)) {
    QNN_LOG_INFO("[G2G] Convert-FC-Reshape-Convert transformed.");
    return new_ops.size();
  }
  return 1;
}

}  // namespace qnn
