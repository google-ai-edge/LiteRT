// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "third_party/odml/litert/ml_drift/tflite/lstm_parser.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/object_reader.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/lstm_shared.h"

namespace litert::ml_drift {
namespace {

::ml_drift::Value* CreateNewSimilarValue(::ml_drift::GraphFloat32* graph,
                                         const ::ml_drift::Value* old_value) {
  ::ml_drift::Value* new_value = graph->NewValue();
  new_value->quant_params = old_value->quant_params;
  new_value->tensor.shape = old_value->tensor.shape;
  new_value->tensor.type = old_value->tensor.type;
  new_value->tensor.ref = -1;
  return new_value;
}

void GetFullyConnectedNode(int weights_tensor_id, int bias_tensor_id,
                           ObjectReader* reader, ::ml_drift::Node* node) {
  const TfLiteTensor* weights_tensor =
      reader->GetInputTensor(weights_tensor_id);
  TfLiteAffineQuantization* quant_params =
      static_cast<TfLiteAffineQuantization*>(
          weights_tensor->quantization.params);
  if (weights_tensor->type == kTfLiteInt8) {
    // uniform/per channel int8 quantization
    node->operation.type =
        ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT8);
    ::ml_drift::FullyConnectedInt8Attributes fc_attr;

    fc_attr.scale.shape = ::ml_drift::OHWI(quant_params->scale->size, 1, 1, 1);
    fc_attr.scale.data.resize(quant_params->scale->size);
    std::memcpy(fc_attr.scale.data.data(), quant_params->scale->data,
                quant_params->scale->size * sizeof(float));

    fc_attr.zero_point.shape =
        ::ml_drift::OHWI(quant_params->scale->size, 1, 1, 1);
    fc_attr.zero_point.data.resize(quant_params->scale->size);
    if (quant_params->zero_point->size == quant_params->scale->size) {
      std::memcpy(fc_attr.zero_point.data.data(),
                  quant_params->zero_point->data,
                  quant_params->zero_point->size * sizeof(int32_t));
    } else {
      std::fill(fc_attr.zero_point.data.begin(), fc_attr.zero_point.data.end(),
                quant_params->zero_point->data[0]);
    }

    fc_attr.weights.data.resize(weights_tensor->bytes);
    std::memcpy(fc_attr.weights.data.data(), weights_tensor->data.int8,
                weights_tensor->bytes);
    int tensor_id = reader->GetTensorId(weights_tensor_id);
    fc_attr.weights.id = tensor_id;
    fc_attr.weights.shape.o = weights_tensor->dims->data[0];
    fc_attr.weights.shape.h = 1;
    fc_attr.weights.shape.w = 1;
    fc_attr.weights.shape.i = weights_tensor->dims->data[1];
    if (bias_tensor_id != -1 &&
        reader->IsNodeInputTensorPresent(bias_tensor_id)) {
      reader->ReadTensor(bias_tensor_id, &(fc_attr.bias),
                         ReadTensorFlags::kNoExtraBytes);
    }
    node->operation.attributes = std::move(fc_attr);
  } else {
    node->operation.type = ToString(::ml_drift::OperationType::FULLY_CONNECTED);
    ::ml_drift::FullyConnectedAttributes fc_attr;
    ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32> weights;
    reader->ReadTensor(weights_tensor_id, &weights,
                       ReadTensorFlags::kExtraBytes);
    fc_attr.weights.data = std::move(weights.data);
    fc_attr.weights.id = weights.id;
    fc_attr.weights.shape.o = weights.shape.h;
    fc_attr.weights.shape.h = 1;
    fc_attr.weights.shape.w = 1;
    fc_attr.weights.shape.i = weights.shape.w;
    if (bias_tensor_id != -1 &&
        reader->IsNodeInputTensorPresent(bias_tensor_id)) {
      reader->ReadTensor(bias_tensor_id, &(fc_attr.bias),
                         ReadTensorFlags::kNoExtraBytes);
    }
    node->operation.attributes = std::move(fc_attr);
  }
}

bool HasTensor(const TfLiteNode* node, const int index) {
  return (index < node->inputs->size) &&
         (node->inputs->data[index] != kTfLiteOptionalTensor);
}

bool HasCifg(const TfLiteNode* node) {
  return !HasTensor(
      node, tflite::ops::builtin::lstm::full::kInputToInputWeightsTensor);
}

bool HasPeephole(const TfLiteNode* node) {
  // Use forget weights to detect peephole instead of input weights as input
  // weights may be missing for cifg.
  return HasTensor(
      node, tflite::ops::builtin::lstm::full::kCellToForgetWeightsTensor);
}

bool HasNormalization(const TfLiteNode* node) {
  return HasTensor(
      node,
      tflite::ops::builtin::lstm::full::kForgetLayerNormCoefficientsTensor);
}

bool HasProjection(const TfLiteNode* node) {
  return HasTensor(node,
                   tflite::ops::builtin::lstm::full::kProjectionWeightsTensor);
}

// Builds subgraph for a single LSTM gate.
// Returns a Value representing the gate's output.
// High-level parameters:
//   - Has normalization (if true: provide normalization weights).
//   - Has peephole connection (if true: provide peephole weights).
//   - Which activation function to use.
// Note: no support for aux input.
//
// Implements the following:
// (*: matrix multiply, .*: elementwise multiply, +: elementwise add):
//   temp = input_weights * input_tensor + recurrent_weights * output_state;
//   if (peephole):
//     temp += peephole_weights .* cell_state;
//   if (layer normalization):
//     gate = activate(normalization_weights .* mean_stddev_norm(temp) + bias);
//   else:
//     gate = activate(temp + bias);
//
void BuildLstmGate(::ml_drift::GraphFloat32* graph, ObjectReader* reader,
                   ::ml_drift::Value* output_state,
                   ::ml_drift::Value* cell_state, int input_weight_id,
                   int recurrent_weight_id, int cell_weight_id, int bias_id,
                   int normalization_weight_id,
                   const TfLiteFusedActivation activation, bool has_peephole,
                   bool has_normalization, ::ml_drift::Value** gate_out) {
  ::ml_drift::Value* input_times_weights =
      CreateNewSimilarValue(graph, cell_state);
  {
    // #1 matrix multiplication: input_weights * input_tensor
    // If has no normalization, also adds bias.
    ::ml_drift::Node* node = graph->NewNode();
    int input_bias_id = !has_normalization ? bias_id : -1;
    GetFullyConnectedNode(input_weight_id, input_bias_id, reader, node);
    reader->AddInput(node, tflite::ops::builtin::lstm::full::kInputTensor);
    graph->SetProducer(node->id, input_times_weights->id);
  }

  ::ml_drift::Value* output_state_times_weights =
      CreateNewSimilarValue(graph, cell_state);
  {
    // #2 matrix multiplication: recurrent_weights * output_state
    ::ml_drift::Node* node = graph->NewNode();
    GetFullyConnectedNode(recurrent_weight_id, -1, reader, node);
    graph->AddConsumer(node->id, output_state->id);
    graph->SetProducer(node->id, output_state_times_weights->id);
  }

  ::ml_drift::Value* cell_state_times_weights;
  if (has_peephole) {
    // #3 elementwise multiplication: cell_weight .* cell_state
    cell_state_times_weights = CreateNewSimilarValue(graph, cell_state);
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::MUL);
    ::ml_drift::ElementwiseAttributes attr;
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
        weights;
    reader->ReadTensor(cell_weight_id, &weights,
                       ReadTensorFlags::kNoExtraBytes);
    attr.param = std::move(weights);
    node->operation.attributes = std::move(attr);
    graph->AddConsumer(node->id, cell_state->id);
    graph->SetProducer(node->id, cell_state_times_weights->id);
  }

  ::ml_drift::Value* gate_before_normalization =
      CreateNewSimilarValue(graph, cell_state);
  ::ml_drift::Node* add_node = graph->NewNode();
  {
    // #4 elementwise addition: #1 + #2 + #3
    add_node->operation.type = ToString(::ml_drift::OperationType::ADD);
    graph->AddConsumer(add_node->id, input_times_weights->id);
    graph->AddConsumer(add_node->id, output_state_times_weights->id);
    if (has_peephole) {
      graph->AddConsumer(add_node->id, cell_state_times_weights->id);
    }
    graph->SetProducer(add_node->id, gate_before_normalization->id);
  }

  if (!has_normalization) {
    // #5 Activation function: activate(temp + bias)
    // Bias is added in node #1.
    HandleFusedActivation(activation, graph, add_node);
    *gate_out = gate_before_normalization;
    return;
  }

  ::ml_drift::Value* normalized_gate =
      CreateNewSimilarValue(graph, gate_before_normalization);
  {
    // #6 Normalization: normalize(temp)
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::MEAN_STDDEV_NORMALIZATION);
    graph->AddConsumer(node->id, gate_before_normalization->id);
    graph->SetProducer(node->id, normalized_gate->id);
  }
  ::ml_drift::Value* reweighted_normalized_gate =
      CreateNewSimilarValue(graph, normalized_gate);
  {
    // #7 Elementwise multiplication: norm_weights .* #6
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::MUL);
    ::ml_drift::ElementwiseAttributes attr;
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
        norm_weights;
    reader->ReadTensor(normalization_weight_id, &norm_weights,
                       ReadTensorFlags::kNoExtraBytes);
    attr.param = std::move(norm_weights);
    node->operation.attributes = std::move(attr);
    graph->AddConsumer(node->id, normalized_gate->id);
    graph->SetProducer(node->id, reweighted_normalized_gate->id);
  }
  ::ml_drift::Value* gate =
      CreateNewSimilarValue(graph, reweighted_normalized_gate);
  {
    // #8 Elementwise add: #7 + bias
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::ADD);
    ::ml_drift::ElementwiseAttributes attr;
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> bias;
    reader->ReadTensor(bias_id, &bias, ReadTensorFlags::kNoExtraBytes);
    attr.param = std::move(bias);
    node->operation.attributes = std::move(attr);
    graph->AddConsumer(node->id, reweighted_normalized_gate->id);
    graph->SetProducer(node->id, gate->id);

    // #9: Activation function
    HandleFusedActivation(activation, graph, node);
  }
  *gate_out = gate;
}

// Builds subgraph for LSTM cell state update.
// Returns a Value representing the updated cell state.
// High-level parameters:
//  - clip: if > 0, clamp the resulting cell state to [-clip, +clip].
//
// Implements the following:
// (*: matrix multiply, .*: elementwise multiply, +: elementwise add):
//
//   cell_state_new = clip(forget_gate .* cell_state + input_gate .* cell_gate);
//
void BuildCellStateUpdate(::ml_drift::GraphFloat32* graph, ObjectReader* reader,
                          ::ml_drift::Value* forget_gate,
                          ::ml_drift::Value* input_gate,
                          ::ml_drift::Value* cell_gate, float cell_clip,
                          ::ml_drift::Value** cell_state_new) {
  ::ml_drift::Value* cell_state =
      reader->ReadValue(tflite::ops::builtin::lstm::full::kCellStateTensor);
  ::ml_drift::Value* cell_state_contrib =
      CreateNewSimilarValue(graph, cell_gate);
  {
    // #1 elementwise multiplication: forget_gate .* cell_state
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::MUL);
    graph->AddConsumer(node->id, forget_gate->id);
    graph->AddConsumer(node->id, cell_state->id);
    graph->SetProducer(node->id, cell_state_contrib->id);
  }
  ::ml_drift::Value* cell_gate_contrib =
      CreateNewSimilarValue(graph, cell_gate);
  {
    // #2 elementwise multiplication: input_gate .* cell_gate
    // Note, with CIFG input_gate is equal to 1-forget_gate.
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::MUL);
    graph->AddConsumer(node->id, input_gate->id);
    graph->AddConsumer(node->id, cell_gate->id);
    graph->SetProducer(node->id, cell_gate_contrib->id);
  }
  ::ml_drift::Value* new_cell_state = CreateNewSimilarValue(graph, cell_gate);
  {
    // #3 elementwise add: #1 + #2
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::ADD);
    graph->AddConsumer(node->id, cell_state_contrib->id);
    graph->AddConsumer(node->id, cell_gate_contrib->id);
    graph->SetProducer(node->id, new_cell_state->id);
  }

  if (cell_clip <= 0.0f) {
    *cell_state_new = new_cell_state;
    return;
  }

  ::ml_drift::Value* max_clipped_state =
      CreateNewSimilarValue(graph, new_cell_state);
  {
    // #4 elementwise minimum: min(#3, clip)
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::MINIMUM);
    ::ml_drift::ElementwiseAttributes attr;
    attr.param = cell_clip;
    node->operation.attributes = std::move(attr);
    graph->AddConsumer(node->id, new_cell_state->id);
    graph->SetProducer(node->id, max_clipped_state->id);
  }
  ::ml_drift::Value* clipped_cell_state =
      CreateNewSimilarValue(graph, max_clipped_state);
  {
    // #5 elementwise maximum: max(#4, -clip)
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::MAXIMUM);
    ::ml_drift::ElementwiseAttributes attr;
    attr.param = -cell_clip;
    node->operation.attributes = std::move(attr);
    graph->AddConsumer(node->id, max_clipped_state->id);
    graph->SetProducer(node->id, clipped_cell_state->id);
  }
  *cell_state_new = clipped_cell_state;
}

// Build subgraph for LSTM output state update.
// Returns value representing the updated output state.
// High-level parameters:
//   - Has projection (if true, provide projection_weights).
//   - Has projection bias (only with projection).
//   - clip: clamp the projection output to [-clip, clip].
//   - Which activation function to use.
// Note the updated output state does not depend on the old output state
// directly, only through the output gate.
//
// Implements the following:
// (*: matrix multiply, .*: elementwise multiply, +: elementwise add):
//
//   temp = output_gate .* activate(cell_state);
//   if (projection):
//     output_state_new = clip(projection_weights * temp + projection_bias);
//   else:
//     output_state_new = temp;
//
void BuildOutputStateUpdate(::ml_drift::GraphFloat32* graph,
                            ObjectReader* reader,
                            ::ml_drift::Value* output_state,
                            ::ml_drift::Value* output_gate,
                            ::ml_drift::Value* cell_state,
                            TfLiteFusedActivation activation,
                            bool has_projection, float proj_clip,
                            ::ml_drift::Value** output_state_new) {
  ::ml_drift::Value* activated_state = CreateNewSimilarValue(graph, cell_state);
  {
    // #1 activation: activate(cell_state)
    ::ml_drift::Node* node = graph->NewNode();
    if (activation == kTfLiteActTanh) {
      node->operation.type = ToString(::ml_drift::OperationType::TANH);
    } else {
      node->operation.type = ToString(::ml_drift::OperationType::SIGMOID);
    }
    graph->AddConsumer(node->id, cell_state->id);
    graph->SetProducer(node->id, activated_state->id);
  }

  ::ml_drift::Value* new_output_state =
      CreateNewSimilarValue(graph, cell_state);
  {
    // #2 elementwise multiplication: output_gate .* #1
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::MUL);
    graph->AddConsumer(node->id, activated_state->id);
    graph->AddConsumer(node->id, output_gate->id);
    graph->SetProducer(node->id, new_output_state->id);
  }

  if (!has_projection) {
    *output_state_new = new_output_state;
    return;
  }

  ::ml_drift::Value* projected_output_state =
      CreateNewSimilarValue(graph, output_state);
  {
    // #3 matrix multiplication: projection_weights * #2 + projection_bias
    ::ml_drift::Node* node = graph->NewNode();

    GetFullyConnectedNode(
        tflite::ops::builtin::lstm::full::kProjectionWeightsTensor,
        tflite::ops::builtin::lstm::full::kProjectionBiasTensor, reader, node);

    graph->AddConsumer(node->id, new_output_state->id);
    graph->SetProducer(node->id, projected_output_state->id);
  }

  if (proj_clip <= 0.0f) {
    *output_state_new = projected_output_state;
    return;
  }

  ::ml_drift::Value* max_clipped_state =
      CreateNewSimilarValue(graph, projected_output_state);
  {
    // #4 elementwise minimum: min(#3, clip)
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::MINIMUM);
    ::ml_drift::ElementwiseAttributes attr;
    attr.param = proj_clip;
    node->operation.attributes = std::move(attr);
    graph->AddConsumer(node->id, projected_output_state->id);
    graph->SetProducer(node->id, max_clipped_state->id);
  }
  ::ml_drift::Value* clipped_output_state =
      CreateNewSimilarValue(graph, max_clipped_state);
  {
    // #5 elementwise maximum: max(#4, -clip)
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::MAXIMUM);
    ::ml_drift::ElementwiseAttributes attr;
    attr.param = -proj_clip;
    node->operation.attributes = std::move(attr);
    graph->AddConsumer(node->id, max_clipped_state->id);
    graph->SetProducer(node->id, clipped_output_state->id);
  }
  *output_state_new = clipped_output_state;
}

}  // namespace

// Build subgraph for a single LSTM OP.
// Returns a mapping for the used variable tensors' updated Values.
//
// High-level parameters:
//   - Has CIFG:
//       If false, calculate input_gate regularly.
//       If true, calculate input_gate to 1-forget_gate.
//   - Has peephole: see BuildLstmGate. Applies to all gates.
//   - Has normalization: see BuildLstmGate. Applies to all gates.
//   - Has projection, projection_bias, proj_clip: see BuildOutputStateUpdate
//   - Which activation to use:
//       Applies to only cell gate and output state update.
//       Other gates always use Sigmoid.
//
void ParseLSTMAttributes(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    ::ml_drift::GraphFloat32* graph, ObjectReader* reader,
    const TfLiteLSTMParams* params,
    absl::flat_hash_map<int, ::ml_drift::ValueId>* new_variable_input_values) {
  const bool has_cifg = HasCifg(tflite_node);
  const bool has_peephole = HasPeephole(tflite_node);
  const bool has_normalization = HasNormalization(tflite_node);
  const bool has_projection = HasProjection(tflite_node);

  ::ml_drift::Value* old_cell_state =
      reader->ReadValue(tflite::ops::builtin::lstm::full::kCellStateTensor);
  ::ml_drift::Value* old_output_state =
      reader->ReadValue(tflite::ops::builtin::lstm::full::kOutputStateTensor);

  ::ml_drift::Value* forget_gate;
  BuildLstmGate(
      graph, reader, old_output_state, old_cell_state,
      tflite::ops::builtin::lstm::full::kInputToForgetWeightsTensor,
      tflite::ops::builtin::lstm::full::kRecurrentToForgetWeightsTensor,
      tflite::ops::builtin::lstm::full::kCellToForgetWeightsTensor,
      tflite::ops::builtin::lstm::full::kForgetGateBiasTensor,
      tflite::ops::builtin::lstm::full::kForgetLayerNormCoefficientsTensor,
      kTfLiteActSigmoid, has_peephole, has_normalization, &forget_gate);

  ::ml_drift::Value* input_gate;
  if (has_cifg) {
    // When using cifg, input_gate is computed as (1 - forget_gate).
    ::ml_drift::Node* node = graph->NewNode();
    input_gate = CreateNewSimilarValue(graph, forget_gate);

    node->operation.type = ToString(::ml_drift::OperationType::SUB);
    ::ml_drift::ElementwiseAttributes attr;
    attr.param = 1.0f;
    attr.runtime_tensor_is_second = true;
    node->operation.attributes = std::move(attr);
    graph->AddConsumer(node->id, forget_gate->id);
    graph->SetProducer(node->id, input_gate->id);
  } else {
    BuildLstmGate(
        graph, reader, old_output_state, old_cell_state,
        tflite::ops::builtin::lstm::full::kInputToInputWeightsTensor,
        tflite::ops::builtin::lstm::full::kRecurrentToInputWeightsTensor,
        tflite::ops::builtin::lstm::full::kCellToInputWeightsTensor,
        tflite::ops::builtin::lstm::full::kInputGateBiasTensor,
        tflite::ops::builtin::lstm::full::kInputLayerNormCoefficientsTensor,
        kTfLiteActSigmoid, has_peephole, has_normalization, &input_gate);
  }

  // Cell state will not have peephole connections to itself
  ::ml_drift::Value* cell_gate;
  BuildLstmGate(
      graph, reader, old_output_state, old_cell_state,
      tflite::ops::builtin::lstm::full::kInputToCellWeightsTensor,
      tflite::ops::builtin::lstm::full::kRecurrentToCellWeightsTensor,
      /*cell_weight_id=*/-1,
      tflite::ops::builtin::lstm::full::kCellGateBiasTensor,
      tflite::ops::builtin::lstm::full::kCellLayerNormCoefficientsTensor,
      params->activation, /*has_peephole=*/false, has_normalization,
      &cell_gate);

  ::ml_drift::Value* new_cell_state;
  BuildCellStateUpdate(graph, reader, forget_gate, input_gate, cell_gate,
                       params->cell_clip, &new_cell_state);

  ::ml_drift::Value* output_gate;
  BuildLstmGate(
      graph, reader, old_output_state, new_cell_state,
      tflite::ops::builtin::lstm::full::kInputToOutputWeightsTensor,
      tflite::ops::builtin::lstm::full::kRecurrentToOutputWeightsTensor,
      tflite::ops::builtin::lstm::full::kCellToOutputWeightsTensor,
      tflite::ops::builtin::lstm::full::kOutputGateBiasTensor,
      tflite::ops::builtin::lstm::full::kOutputLayerNormCoefficientsTensor,
      kTfLiteActSigmoid, has_peephole, has_normalization, &output_gate);

  ::ml_drift::Value* new_output_state;
  BuildOutputStateUpdate(graph, reader, old_output_state, output_gate,
                         new_cell_state, params->activation, has_projection,
                         params->proj_clip, &new_output_state);

  {
    // Copy updated output state to output.
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::COPY);
    graph->AddConsumer(node->id, new_output_state->id);
    reader->AddOutput(node, tflite::ops::builtin::lstm::full::kOutputTensor);
  }

  new_variable_input_values->clear();
  new_variable_input_values->emplace(
      tflite::ops::builtin::lstm::full::kCellStateTensor, new_cell_state->id);
  new_variable_input_values->emplace(
      tflite::ops::builtin::lstm::full::kOutputStateTensor,
      new_output_state->id);
}

}  // namespace litert::ml_drift
