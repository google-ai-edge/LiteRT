// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/lstm_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {
namespace {

constexpr size_t kInput = 0;
constexpr size_t kInputToInputWeights = 1;
constexpr size_t kInputToForgetWeights = 2;
constexpr size_t kInputToCellWeights = 3;
constexpr size_t kInputToOutputWeights = 4;
constexpr size_t kRecurrentToInputWeights = 5;
constexpr size_t kRecurrentToForgetWeights = 6;
constexpr size_t kRecurrentToCellWeights = 7;
constexpr size_t kRecurrentToOutputWeights = 8;
constexpr size_t kCellToInputWeights = 9;
constexpr size_t kCellToForgetWeights = 10;
constexpr size_t kCellToOutputWeights = 11;
constexpr size_t kInputGateBias = 12;
constexpr size_t kForgetGateBias = 13;
constexpr size_t kCellGateBias = 14;
constexpr size_t kOutputGateBias = 15;
constexpr size_t kProjectionWeights = 16;
constexpr size_t kProjectionBias = 17;
constexpr size_t kOutputState = 18;
constexpr size_t kCellState = 19;
constexpr size_t kInputLayerNormCoefficients = 20;
constexpr size_t kForgetLayerNormCoefficients = 21;
constexpr size_t kCellLayerNormCoefficients = 22;
constexpr size_t kOutputLayerNormCoefficients = 23;

constexpr size_t kNumInputsNoLayerNorm = 20;
constexpr size_t kNumInputsWithLayerNorm = 24;

constexpr size_t kOutput = 0;

constexpr size_t kQnnInputLayerNormCoefficients = 12;
constexpr size_t kQnnForgetLayerNormCoefficients = 13;
constexpr size_t kQnnCellLayerNormCoefficients = 14;
constexpr size_t kQnnOutputLayerNormCoefficients = 15;

constexpr float kGateQScale = 1.0f / 4096;

}  // namespace

std::vector<OpWrapper> BuildLstmOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, float cell_clip,
    float proj_clip, bool time_major) {
  std::vector<OpWrapper> res;

  // FULL LSTM has 20 canonical inputs, plus four optional LayerNorm inputs.
  const bool has_layer_norm = inputs.size() == kNumInputsWithLayerNorm;
  if ((!has_layer_norm && inputs.size() != kNumInputsNoLayerNorm) ||
      outputs.empty()) {
    QNN_LOG_ERROR(
        "LiteRT Lstm lowering expects %zu or %zu canonical FULL input slots "
        "and at least one TFLite output, got %zu inputs and %zu outputs.",
        kNumInputsNoLayerNorm, kNumInputsWithLayerNorm, inputs.size(),
        outputs.size());
    return res;
  }

  TensorWrapper& output_tensor = outputs[kOutput].get();
  const TensorWrapper& cell_state_tensor = inputs[kCellState].get();

  // A 2D LSTM maps its TFLite output to QNN out[2]. A 3D sequence LSTM maps
  // the full sequence to out[0], while out[2] remains the final 2D step.
  const bool is_sequence = output_tensor.GetRank() == 3;
  TensorWrapper& input_layer_norm =
      has_layer_norm ? inputs[kInputLayerNormCoefficients].get()
                     : tensor_pool.CreateNullTensor();
  TensorWrapper& forget_layer_norm =
      has_layer_norm ? inputs[kForgetLayerNormCoefficients].get()
                     : tensor_pool.CreateNullTensor();
  TensorWrapper& cell_layer_norm =
      has_layer_norm ? inputs[kCellLayerNormCoefficients].get()
                     : tensor_pool.CreateNullTensor();
  TensorWrapper& output_layer_norm =
      has_layer_norm ? inputs[kOutputLayerNormCoefficients].get()
                     : tensor_pool.CreateNullTensor();

  std::vector<ConstTensorWrapperRef> lstm_inputs{
      inputs[kInput].get(),
      inputs[kInputToForgetWeights].get(),
      inputs[kInputToCellWeights].get(),
      inputs[kInputToOutputWeights].get(),
      inputs[kRecurrentToForgetWeights].get(),
      inputs[kRecurrentToCellWeights].get(),
      inputs[kRecurrentToOutputWeights].get(),
      inputs[kForgetGateBias].get(),
      inputs[kCellGateBias].get(),
      inputs[kOutputGateBias].get(),
      inputs[kOutputState].get(),
      inputs[kCellState].get(),
      input_layer_norm,
      forget_layer_norm,
      cell_layer_norm,
      output_layer_norm,
      inputs[kInputToInputWeights].get(),
      inputs[kRecurrentToInputWeights].get(),
      inputs[kCellToInputWeights].get(),
      inputs[kCellToForgetWeights].get(),
      inputs[kCellToOutputWeights].get(),
      inputs[kInputGateBias].get(),
      inputs[kProjectionWeights].get(),
      inputs[kProjectionBias].get(),
      tensor_pool.CreateNullTensor()};

  const std::vector<std::uint32_t>& seq_dims = output_tensor.GetDimensions();
  std::vector<std::uint32_t> step_dims;
  if (is_sequence) {
    // Extract [batch, output] from either [time, batch, output] or
    // [batch, time, output].
    step_dims = {seq_dims[time_major ? 1 : 0], seq_dims.back()};
  }

  // QNN requires out[0] output state, out[1] cell state, and out[2] output.
  TensorWrapper& output_state_out =
      is_sequence ? output_tensor
                  : tensor_pool.CloneNativeTensorFrom(output_tensor);
  TensorWrapper& cell_state_out =
      tensor_pool.CloneNativeTensorFrom(cell_state_tensor);
  TensorWrapper& final_step_out =
      is_sequence ? tensor_pool.CloneNativeTensorFrom(output_tensor, step_dims)
                  : output_tensor;

  const std::vector<ConstTensorWrapperRef> lstm_outputs{
      output_state_out, cell_state_out, final_step_out};

  res.emplace_back(CreateLstmOp(lstm_inputs, lstm_outputs, cell_clip, proj_clip,
                                time_major, QNN_OP_LSTM_DIRECTION_FORWARD));
  return res;
}

OpWrapper CreateLstmOp(const std::vector<ConstTensorWrapperRef>& inputs,
                       const std::vector<ConstTensorWrapperRef>& outputs,
                       float cell_clip, float proj_clip, bool time_major,
                       std::uint32_t direction) {
  OpWrapper op(GetUniqueOpName(QNN_OP_LSTM), QNN_OP_LSTM, QnnOpCode::kLstm);
  for (const auto& input : inputs) {
    op.AddInputTensor(input);
  }
  for (const auto& output : outputs) {
    op.AddOutputTensor(output);
  }

  op.AddScalarParam<std::uint32_t>(QNN_OP_LSTM_PARAM_DIRECTION, direction);
  op.AddScalarParam<float>(QNN_OP_LSTM_PARAM_CELL_CLIP_THRESHOLD, cell_clip);
  op.AddScalarParam<float>(QNN_OP_LSTM_PARAM_OUTPUT_CLIP_THRESHOLD, proj_clip);
  op.AddScalarParam<bool>(QNN_OP_LSTM_PARAM_TIME_MAJOR, time_major);

  // Quantized non-LayerNorm FULL uses a 2^-12 gate qscales. LayerNorm omits
  // them to match qnn-delegate and QNN HTP requirements.
  const bool has_layer_norm =
      !inputs[kQnnInputLayerNormCoefficients].get().IsTensorNull() ||
      !inputs[kQnnForgetLayerNormCoefficients].get().IsTensorNull() ||
      !inputs[kQnnCellLayerNormCoefficients].get().IsTensorNull() ||
      !inputs[kQnnOutputLayerNormCoefficients].get().IsTensorNull();
  if (inputs.front().get().IsQuant() && !has_layer_norm) {
    op.AddScalarParam<float>(QNN_OP_LSTM_PARAM_INPUT_GATE_QSCALE, kGateQScale);
    op.AddScalarParam<float>(QNN_OP_LSTM_PARAM_FORGET_GATE_QSCALE, kGateQScale);
    op.AddScalarParam<float>(QNN_OP_LSTM_PARAM_CELL_GATE_QSCALE, kGateQScale);
    op.AddScalarParam<float>(QNN_OP_LSTM_PARAM_OUTPUT_GATE_QSCALE, kGateQScale);
  }

  return op;
}

}  // namespace qnn
