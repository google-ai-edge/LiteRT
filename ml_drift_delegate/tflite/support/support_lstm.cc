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

#include "ml_drift_delegate/tflite/support/support_lstm.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/kernels/lstm_shared.h"

namespace litert::ml_drift::ir {
namespace {

using ::tflite::ops::builtin::lstm::full::kCellGateBiasTensor;
using ::tflite::ops::builtin::lstm::full::kCellLayerNormCoefficientsTensor;
using ::tflite::ops::builtin::lstm::full::kCellStateTensor;
using ::tflite::ops::builtin::lstm::full::kCellToForgetWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kCellToInputWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kCellToOutputWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kForgetGateBiasTensor;
using ::tflite::ops::builtin::lstm::full::kForgetLayerNormCoefficientsTensor;
using ::tflite::ops::builtin::lstm::full::kInputGateBiasTensor;
using ::tflite::ops::builtin::lstm::full::kInputLayerNormCoefficientsTensor;
using ::tflite::ops::builtin::lstm::full::kInputTensor;
using ::tflite::ops::builtin::lstm::full::kInputToCellWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kInputToForgetWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kInputToInputWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kInputToOutputWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kOutputGateBiasTensor;
using ::tflite::ops::builtin::lstm::full::kOutputLayerNormCoefficientsTensor;
using ::tflite::ops::builtin::lstm::full::kOutputStateTensor;
using ::tflite::ops::builtin::lstm::full::kProjectionBiasTensor;
using ::tflite::ops::builtin::lstm::full::kProjectionWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kRecurrentToCellWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kRecurrentToForgetWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kRecurrentToInputWeightsTensor;
using ::tflite::ops::builtin::lstm::full::kRecurrentToOutputWeightsTensor;

// Helpers for full LSTM.
inline bool HasTensor(const TfLiteNode* node, const int index) {
  return (index < node->inputs->size) &&
         (node->inputs->data[index] != kTfLiteOptionalTensor);
}

inline bool HasCifg(const TfLiteNode* node) {
  return !HasTensor(node, kInputToInputWeightsTensor);
}

inline bool HasPeephole(const TfLiteNode* node) {
  return HasTensor(node, kCellToForgetWeightsTensor);
}

inline bool HasNormalization(const TfLiteNode* node) {
  return HasTensor(node, kForgetLayerNormCoefficientsTensor);
}

inline bool HasProjection(const TfLiteNode* node) {
  return HasTensor(node, kProjectionWeightsTensor);
}

bool CheckGateTensors(const TfLiteContext* context, const TfLiteNode* node,
                      int input_weight_id, int recurrent_weight_id,
                      int cell_weight_id, int bias_id,
                      int normalization_weight_id, bool has_peephole,
                      bool has_normalization, std::string* error) {
  const absl::flat_hash_set<TfLiteType> supported_weights_dtypes = {
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
  };
  const absl::flat_hash_set<TfLiteType> supported_bias_dtypes = {
      kTfLiteFloat16,
      kTfLiteFloat32,
  };

  if (!CheckTensorDtype(context->tensors[node->inputs->data[input_weight_id]],
                        supported_weights_dtypes, "input weight", *error)) {
    return false;
  }
  if (!CheckTensorDtype(
          context->tensors[node->inputs->data[recurrent_weight_id]],
          supported_weights_dtypes, "recurrent weight", *error)) {
    return false;
  }
  if (!CheckTensorDtype(context->tensors[node->inputs->data[bias_id]],
                        supported_bias_dtypes, "bias", *error)) {
    return false;
  }

  if (has_peephole && HasTensor(node, cell_weight_id) &&
      !CheckTensorDtype(context->tensors[node->inputs->data[cell_weight_id]],
                        supported_bias_dtypes, "peephole weight", *error)) {
    return false;
  }

  if (has_normalization && HasTensor(node, normalization_weight_id) &&
      !CheckTensorDtype(
          context->tensors[node->inputs->data[normalization_weight_id]],
          supported_bias_dtypes, "layer norm coefficient", *error)) {
    return false;
  }
  return true;
}

bool CheckFullGates(const TfLiteContext* context, const TfLiteNode* node,
                    bool has_cifg, bool has_peephole, bool has_normalization,
                    std::string* error) {
  // Check forget gate tensors.
  if (!CheckGateTensors(context, node, kInputToForgetWeightsTensor,
                        kRecurrentToForgetWeightsTensor,
                        kCellToForgetWeightsTensor, kForgetGateBiasTensor,
                        kForgetLayerNormCoefficientsTensor, has_peephole,
                        has_normalization, error)) {
    return false;
  }

  // Check input gate tensors.
  if (!has_cifg) {
    if (!CheckGateTensors(context, node, kInputToInputWeightsTensor,
                          kRecurrentToInputWeightsTensor,
                          kCellToInputWeightsTensor, kInputGateBiasTensor,
                          kInputLayerNormCoefficientsTensor, has_peephole,
                          has_normalization, error)) {
      return false;
    }
  }

  // Check cell gate tensors.
  if (!CheckGateTensors(context, node, kInputToCellWeightsTensor,
                        kRecurrentToCellWeightsTensor, -1, kCellGateBiasTensor,
                        kCellLayerNormCoefficientsTensor,
                        /*has_peephole=*/false, has_normalization, error)) {
    return false;
  }

  // Check output gate tensors.
  if (!CheckGateTensors(context, node, kInputToOutputWeightsTensor,
                        kRecurrentToOutputWeightsTensor,
                        kCellToOutputWeightsTensor, kOutputGateBiasTensor,
                        kOutputLayerNormCoefficientsTensor, has_peephole,
                        has_normalization, error)) {
    return false;
  }

  return true;
}

bool CheckFullDtypes(const TfLiteContext* context, const TfLiteNode* node,
                     bool has_projection, std::string* error) {
  const absl::flat_hash_set<TfLiteType> supported_input_dtypes = {
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
  };
  const TfLiteTensor& input =
      context->tensors[node->inputs->data[kInputTensor]];
  if (!CheckTensorDtype(input, supported_input_dtypes, "input", *error)) {
    return false;
  }
  if (!CheckNotConstant(input, "input", *error)) {
    return false;
  }

  const TfLiteTensor& cell_state =
      context->tensors[node->inputs->data[kCellStateTensor]];
  if (!CheckTensorDtype(cell_state, supported_input_dtypes, "cell state",
                        *error)) {
    return false;
  }
  if (!CheckNotConstant(cell_state, "cell state", *error)) {
    return false;
  }

  const TfLiteTensor& output_state =
      context->tensors[node->inputs->data[kOutputStateTensor]];
  if (!CheckTensorDtype(output_state, supported_input_dtypes, "output state",
                        *error)) {
    return false;
  }
  if (!CheckNotConstant(output_state, "output state", *error)) {
    return false;
  }

  // Check projection tensors.
  if (has_projection) {
    const absl::flat_hash_set<TfLiteType> supported_weights_dtypes = {
        kTfLiteFloat16,
        kTfLiteFloat32,
        kTfLiteInt8,
    };
    const absl::flat_hash_set<TfLiteType> supported_bias_dtypes = {
        kTfLiteFloat16,
        kTfLiteFloat32,
    };
    if (!CheckTensorDtype(
            context->tensors[node->inputs->data[kProjectionWeightsTensor]],
            supported_weights_dtypes, "projection weight", *error)) {
      return false;
    }
    if (HasTensor(node, kProjectionBiasTensor) &&
        !CheckTensorDtype(
            context->tensors[node->inputs->data[kProjectionBiasTensor]],
            supported_bias_dtypes, "projection bias", *error)) {
      return false;
    }
  }

  return true;
}

// Full LSTM subgraph.
// Supports 24 inputs and 1 output.
//
// For full LSTM cells, see this blog post:
// https://colah.github.io/posts/2015-08-Understanding-LSTMs/
// In addition to Peephole connections and Combined Input Forget Gates (CIFG)
// described in that post, this code also adds the following optional features:
// - Configurable activations (sigmoid or TANH)
// - L2 Normalization of gates: https://arxiv.org/abs/1607.06450
// - Output projection:
//     https://www.isca-speech.org/archive/interspeech_2014/i14_0338.html
// - Configurable clipping of cell state and output state.
bool CheckFull(const TfLiteContext* context, const TfLiteNode* node,
               const TfLiteRegistration* registration,
               const TfLiteLSTMParams* tf_options, std::string* error) {
  const bool has_cifg = HasCifg(node);
  const bool has_peephole = HasPeephole(node);
  const bool has_normalization = HasNormalization(node);
  const bool has_projection = HasProjection(node);

  // Check input and output tensor counts.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/24,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *node->inputs, "inputs", *error)) {
    return false;
  }
  if (!ValidateTensorIds(*context, *node->outputs, "outputs", *error)) {
    return false;
  }

  // Check tensor dtypes.
  if (!CheckFullDtypes(context, node, has_projection, error)) {
    return false;
  }

  // Check gate tensors.
  if (!CheckFullGates(context, node, has_cifg, has_peephole, has_normalization,
                      error)) {
    return false;
  }

  // Check activation.
  if (tf_options->activation != kTfLiteActSigmoid &&
      tf_options->activation != kTfLiteActTanh) {
    absl::StrAppend(error, "Only sigmoid or tanh activation is supported.");
    return false;
  }

  return true;
}

// Helpers for basic LSTM.
inline bool CheckBasicParameters(const TfLiteLSTMParams* tf_options,
                                 std::string* error) {
  if (tf_options->activation != kTfLiteActTanh) {
    absl::StrAppend(error, "Only TANH activation is supported.");
    return false;
  }
  if (tf_options->cell_clip != 0.0f) {
    absl::StrAppend(error, "cell_clip is not supported.");
    return false;
  }
  if (tf_options->proj_clip != 0.0f) {
    absl::StrAppend(error, "proj_clip is not supported.");
    return false;
  }
  return true;
}

// Basic LSTM Cell:
//
//  1name = name is at input  index 1
//  name1 = name is at output index 1
//
//    0input     1prev_activ
//       \        /
//        [[concat]]
//             \
//       concat_temp2  2weights  3biases
//              \      /        /
//             [[fully-connected]]
//               \
//         activ_temp3    4prev_state
//                 \      /
//                 [[LSTM]]
//                 /      \
//           new_state1    activation0
//
bool CheckBasic(const TfLiteContext* context, const TfLiteNode* node,
                const TfLiteRegistration* registration,
                const TfLiteLSTMParams* tf_options, std::string* error) {
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/5,
                              /*expected_outputs=*/4, *error)) {
    return false;
  }
  if (!CheckBasicParameters(tf_options, error)) {
    return false;
  }

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *node->inputs, "inputs", *error)) {
    return false;
  }
  if (!ValidateTensorIds(*context, *node->outputs, "outputs", *error)) {
    return false;
  }

  // Check tensor properties.
  const absl::flat_hash_set<TfLiteType> supported_input_dtypes = {
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
  };
  const absl::flat_hash_set<TfLiteType> supported_weights_dtypes = {
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
  };
  const absl::flat_hash_set<TfLiteType> supported_bias_dtypes = {
      kTfLiteFloat16,
      kTfLiteFloat32,
  };

  // Check runtime inputs.
  for (int i : {0, 1, 4}) {
    const TfLiteTensor& tensor = context->tensors[node->inputs->data[i]];
    if (!CheckNotConstant(tensor, absl::StrCat("inputs[", i, "]"), *error)) {
      return false;
    }
    if (!CheckTensorDtype(tensor, supported_input_dtypes,
                          absl::StrCat("inputs[", i, "]"), *error)) {
      return false;
    }
    if (tensor.sparsity) {
      absl::StrAppend(error, "Sparsity is not supported for input ", i);
      return false;
    }
  }

  // Check other inputs.
  const TfLiteTensor& weights = context->tensors[node->inputs->data[2]];
  if (!CheckTensorDtype(weights, supported_weights_dtypes, "inputs[2]",
                        *error)) {
    return false;
  }
  if (weights.sparsity) {
    absl::StrAppend(error, "Sparsity is not supported for weights.");
    return false;
  }

  const TfLiteTensor& bias = context->tensors[node->inputs->data[3]];
  if (!CheckTensorDtype(bias, supported_bias_dtypes, "inputs[3]", *error)) {
    return false;
  }
  if (bias.sparsity) {
    absl::StrAppend(error, "Sparsity is not supported for bias.");
    return false;
  }

  // Check main input and output tensor dimensions are the same.
  const TfLiteTensor& prev_activ = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& prev_state = context->tensors[node->inputs->data[4]];
  const TfLiteTensor& output_activ = context->tensors[node->outputs->data[0]];
  const TfLiteTensor& output_state = context->tensors[node->outputs->data[1]];
  if (!::tflite::HaveSameShapes(&prev_state, &output_state)) {
    absl::StrAppend(error, "prev_state and output_state shapes mismatch.");
    return false;
  }
  if (!::tflite::HaveSameShapes(&prev_activ, &output_activ)) {
    absl::StrAppend(error, "prev_activ and output_activ shapes mismatch.");
    return false;
  }
  return true;
}

}  // namespace

bool IsLstmSupported(const TfLiteContext* absl_nonnull context,
                     const TfLiteNode* absl_nonnull node,
                     const TfLiteRegistration* absl_nonnull registration,
                     const int supported_max_version,
                     std::string* absl_nonnull error) {
  if (registration->version < 1 ||
      registration->version > supported_max_version) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  const auto* params = static_cast<const TfLiteLSTMParams*>(node->builtin_data);
  if (!params) {
    absl::StrAppend(error, "Missing TfLiteLSTMParams.");
    return false;
  }

  bool supported = false;
  switch (params->kernel_type) {
    case kTfLiteLSTMBasicKernel:
      supported = CheckBasic(context, node, registration, params, error);
      break;
    case kTfLiteLSTMFullKernel:
      supported = CheckFull(context, node, registration, params, error);
      break;
    default:
      absl::StrAppend(error, "Unsupported kernel type: ", params->kernel_type);
      break;
  }

  return supported;
}

}  // namespace litert::ml_drift::ir
