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

#include "third_party/odml/litert/ml_drift/tflite/operation_parser.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/model_builder_helper.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/tools/versioning/gpu_compatibility.h"

namespace litert::ml_drift {

absl::Status TFLiteOperationParser::ValidateSupport(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration,
    const ParserValidationOptions& options) {
  RETURN_IF_ERROR(
      CheckMaxSupportedOpVersion(registration, options.max_version));
  if (options.check_gpu_compatibility) {
    RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration, options.gpu_flags));
  }
  if (options.min_inputs != -1 || options.max_inputs != -1) {
    const int inputs = tflite_node->inputs->size;
    if (options.min_inputs != -1 && inputs < options.min_inputs) {
      return absl::InternalError(
          absl::StrCat("Invalid number of inputs: ", inputs,
                       ", while expected at least ", options.min_inputs, "."));
    }
    if (options.max_inputs != -1 && inputs > options.max_inputs) {
      return absl::InternalError(
          absl::StrCat("Invalid number of inputs: ", inputs,
                       ", while expected at most ", options.max_inputs, "."));
    }
  }
  if (options.num_outputs != -1 &&
      tflite_node->outputs->size != options.num_outputs) {
    return absl::InternalError(
        absl::StrCat("Expected ", options.num_outputs,
                     " output tensor(s), but node has ",
                     tflite_node->outputs->size, " output(s)."));
  }
  if (options.required_runtime_inputs != -1) {
    const int runtime_inputs =
        GetNumberOfRuntimeInputsForNode(context, tflite_node);
    if (runtime_inputs != options.required_runtime_inputs) {
      return absl::InternalError(absl::StrCat(
          "Expected ", options.required_runtime_inputs,
          " runtime input tensor(s), but node has ", runtime_inputs,
          " runtime input(s)."));
    }
  }
  if (options.required_const_inputs != -1) {
    const int const_inputs =
        GetNumberOfConstInputsForNode(context, tflite_node);
    if (const_inputs != options.required_const_inputs) {
      return absl::InternalError(absl::StrCat(
          "Expected ", options.required_const_inputs,
          " const input tensor(s), but node has ", const_inputs,
          " const input(s)."));
    }
  }
  RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
  return absl::OkStatus();
}

absl::Status CheckMaxSupportedOpVersion(const TfLiteRegistration* registration,
                                        int max_version) {
  const int op_version = registration->version;
  if (max_version != -1 && op_version > max_version) {
    return absl::UnimplementedError(
        absl::StrCat("Max version supported: ", max_version,
                     ". Requested version ", op_version, "."));
  }
  return absl::OkStatus();
}

// Helper functions for IsSupported() checkings
absl::Status PreGetInputTensor(const TfLiteContext* context,
                               const TfLiteNode* tflite_node, int32_t index,
                               const TfLiteTensor** input) {
  if (tflite::GetInputSafe(context, tflite_node, index, input) != kTfLiteOk) {
    return absl::UnavailableError(
        absl::StrCat("input #", index, " tensor is not available"));
  }
  return absl::OkStatus();
}

absl::Status PreGetOutputTensor(const TfLiteContext* context,
                                const TfLiteNode* tflite_node, int32_t index,
                                TfLiteTensor** output) {
  if (tflite::GetOutputSafe(context, tflite_node, index, output) != kTfLiteOk) {
    return absl::UnavailableError(
        absl::StrCat("output #", index, " tensor is not available"));
  }
  return absl::OkStatus();
}

absl::Status CheckTensorShape(const TfLiteIntArray* dims,
                              const char* tensor_name) {
  switch (dims->size) {
    case 0:
      // scalar layout
    case 1:
      // B layout
    case 2:
      // BC layout
    case 3:
      // BWC layout
    case 4:
      // BHWC layout
    case 5:
      // BHWDC layout
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Tensor \"", tensor_name ? tensor_name : "nullptr",
                       "\" has bad input dims size: ", dims->size, "."));
  }
}

template <>
absl::Status CheckAllDimensions<::ml_drift::Scalar>(
    const TfLiteIntArray* dimensions) {
  if (dimensions->size < 0) {
    return absl::InvalidArgumentError("Invalid Scalar dimensions");
  }
  for (int i = 0; i < dimensions->size; ++i) {
    if (dimensions->data[i] != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat(tflite::GetShapeDebugString(dimensions),
                       "  cannot be reduced to scalar."));
    }
  }
  return absl::OkStatus();
}

template <>
absl::Status CheckAllDimensions<::ml_drift::Linear>(
    const TfLiteIntArray* dimensions) {
  if (IsLinearConvertible(dimensions)) return absl::OkStatus();
  return absl::InvalidArgumentError("Isn't linear convertible.");
}

template <>
absl::Status CheckAllDimensions<::ml_drift::HWC>(
    const TfLiteIntArray* dimensions) {
  if (dimensions->size == 3) {
    return absl::OkStatus();
  }
  if (dimensions->size == 4) {
    if (dimensions->data[0] != 1) {
      return absl::UnimplementedError("Batch size is not equal to 1.");
    }
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Expected a 3D tensor of shape HxWxC or a 4D tensor of "
                   "shape 1xHxWxC but got ",
                   tflite::GetShapeDebugString(dimensions)));
}

template <>
absl::Status CheckAllDimensions<::ml_drift::HW>(
    const TfLiteIntArray* dimensions) {
  if (dimensions->size != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected a 2D tensor of shape HxW but got ",
                     tflite::GetShapeDebugString(dimensions)));
  }
  return absl::OkStatus();
}

template <>
absl::Status CheckAllDimensions<::ml_drift::OHWI>(
    const TfLiteIntArray* dimensions) {
  if (dimensions->size != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected a 4D tensor of shape OxHxWxI but got ",
                     tflite::GetShapeDebugString(dimensions)));
  }
  return absl::OkStatus();
}

template <>
absl::Status CheckAllDimensions<::ml_drift::BHWC>(
    const TfLiteIntArray* dimensions) {
  return CheckTensorShape(dimensions);
}

template <>
absl::Status PreCheckCopyData(const TfLiteTensor& src, float* dst) {
  if (src.data.raw_const == nullptr) {
    return absl::InvalidArgumentError("src has no data.");
  }
  switch (src.type) {
    case kTfLiteFloat32:
    case kTfLiteFloat16:
    case kTfLiteInt4:
    case kTfLiteInt8:
    case kTfLiteUInt8:
    case kTfLiteInt32:
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          "Unsupported data type for float32 tensor");
  }
}

absl::Status PreCheckAxisFromIndex(const TfLiteTensor& tflite_tensor,
                                   int index) {
  const TfLiteIntArray* dims = tflite_tensor.dims;
  if (index < 0) {
    index = dims->size + index;
  }
  if (index < 0 || index >= dims->size) {
    return absl::OutOfRangeError("Index for axis out of range");
  }
  switch (dims->size) {
    case 1:
    case 2:
    case 3:
    case 4:
      return absl::OkStatus();
    default:
      return absl::UnavailableError("Unknown layout.");
  }
  return absl::OkStatus();
}

absl::Status PreCheckTensorShape(const TfLiteTensor& tflite_tensor) {
  return CheckTensorShape(tflite_tensor.dims, tflite_tensor.name);
}

absl::Status PreCheckTfLiteShape(const TfLiteTensor& tflite_tensor) {
  const TfLiteIntArray* dims = tflite_tensor.dims;
  switch (dims->size) {
    case 1:
      // C layout
    case 2:
      // WC layout
    case 3:
      // HWC layout
    case 4:
      // BHWC layout
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Tensor \"", tflite_tensor.name ? tflite_tensor.name : "nullptr",
          "\" has bad input dims size: ", dims->size, "."));
  }
  return absl::OkStatus();
}

absl::Status GetTensorId(const TfLiteContext* context, const TfLiteNode* node,
                         int input_id, int* tensor_id) {
  if (input_id < 0 || input_id >= node->inputs->size) {
    return absl::OutOfRangeError(
        absl::StrCat("Invalid input tensor index: ", input_id));
  }
  *tensor_id = node->inputs->data[input_id];
  if (*tensor_id < 0 || *tensor_id > context->tensors_size) {
    return absl::OutOfRangeError(
        absl::StrCat("Invalid Tensor index: ", *tensor_id));
  }
  return absl::OkStatus();
}

absl::Status PreCheckNewTensorWithDifferentType(const TfLiteContext* context,
                                                const int original_tensor_index,
                                                TfLiteType new_type,
                                                TfLiteTensor* new_tensor) {
  const TfLiteTensor& original_tensor = context->tensors[original_tensor_index];
  new_tensor->type = new_type;
  new_tensor->allocation_type = kTfLiteArenaRw;
  // instead of copying, just use the original tensor's dims for checking
  new_tensor->dims = original_tensor.dims;

  return absl::OkStatus();
}

absl::Status PreCheckReadValue(const TfLiteContext* context,
                               const TfLiteNode* node, uint32_t idx) {
  int32_t tensor_idx = 0;
  RETURN_IF_ERROR(GetTensorId(context, node, idx, &(tensor_idx)));
  return PreCheckReadValueByTensorIdx(context, tensor_idx);
}

absl::Status PreCheckReadValueByTensorIdx(const TfLiteContext* context,
                                          uint32_t tensor_idx) {
  // The checkings in ReadNonConstantTensor
  if (tensor_idx < 0 || tensor_idx >= context->tensors_size) {
    return absl::OutOfRangeError(
        absl::StrCat("PreCheckReadValue: input tensor index: ", tensor_idx));
  }

  TfLiteTensor* tflite_tensor = &context->tensors[tensor_idx];
  if ((tflite_tensor->type == kTfLiteInt8 ||
       tflite_tensor->type == kTfLiteUInt8) &&
      tflite_tensor->quantization.type ==
          TfLiteQuantizationType::kTfLiteAffineQuantization) {
    TfLiteTensor fp_tflite_tensor;
    RETURN_IF_ERROR(PreCheckNewTensorWithDifferentType(
        context, tensor_idx, kTfLiteFloat32, &fp_tflite_tensor));
    RETURN_IF_ERROR(
        CheckTensorShape(fp_tflite_tensor.dims, tflite_tensor->name));
    // checkings in PopulateQuantParams
    TfLiteAffineQuantization* quantization_data =
        reinterpret_cast<TfLiteAffineQuantization*>(
            tflite_tensor->quantization.params);
    if (quantization_data->scale->size > 1) {
      return absl::InvalidArgumentError("Unsupported quantization scale size");
    }
  } else {
    RETURN_IF_ERROR(CheckTensorShape(tflite_tensor->dims, tflite_tensor->name));
  }

  return absl::OkStatus();
}

absl::Status PreCheckReadQuantizedValueByTensorIdx(const TfLiteContext* context,
                                                   uint32_t tensor_idx) {
  TfLiteTensor* tflite_tensor = &context->tensors[tensor_idx];
  if (tflite_tensor->quantization.params == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "PreCheckReadQuantizedValueByTensorIdx: Empty quantization params: ",
        tensor_idx));
  }
  if (tflite_tensor->quantization.type != kTfLiteAffineQuantization &&
      tflite_tensor->quantization.type != kTfLiteBlockwiseQuantization) {
    return absl::InvalidArgumentError(absl::StrCat(
        "PreCheckReadQuantizedValueByTensorIdx: Tensor is not quantized: ",
        tensor_idx));
  }
  if (tflite_tensor->type != kTfLiteInt8 &&
      tflite_tensor->type != kTfLiteInt4 &&
      tflite_tensor->type != kTfLiteInt2) {
    return absl::InvalidArgumentError(
        "Expected Int8, Int4, or Int2 quantized tensor: " +
        std::to_string(tensor_idx));
  }
  if (tflite_tensor->dims->size != 2 && tflite_tensor->dims->size != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("PreCheckReadQuantizedValueByTensorIdx: Expected 2D or 4D "
                     "quantized tensor: ",
                     tensor_idx));
  }
  if (tflite_tensor->dims->size == 4) {
    if (tflite_tensor->dims->data[1] != 1 ||
        tflite_tensor->dims->data[2] != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("PreCheckReadQuantizedValueByTensorIdx: Expected 4D "
                       "quantized tensor with height of 1 and width of 1: ",
                       tensor_idx));
    }
  }
  return absl::OkStatus();
}

absl::Status PreCheckRuntimeOrConstantInput(const TfLiteContext* context,
                                            const TfLiteNode* node,
                                            uint32_t tensor_idx) {
  if (!PreCheckReadValue(context, node, tensor_idx).ok()) {
    // checkings in NewConstNode
    const TfLiteTensor* input = nullptr;
    RETURN_IF_ERROR(PreGetInputTensor(context, node, tensor_idx, &input));
    if (input->type == kTfLiteFloat32) {
      ::ml_drift::TensorFloat32 t;
      RETURN_IF_ERROR(PreCheckTensorToTensor(input, &t));
    } else if (input->type == kTfLiteBool) {
      ::ml_drift::TensorBool t;
      RETURN_IF_ERROR(PreCheckTensorToTensor(input, &t));
    } else if (input->type == kTfLiteInt32) {
      ::ml_drift::TensorInt32 t;
      RETURN_IF_ERROR(PreCheckTensorToTensor(input, &t));
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported tensor type in NewConstNode: ",
                       std::string(TfLiteTypeGetName(input->type))));
    }
  }
  return absl::OkStatus();
}

absl::Status PreCheckOutput(const TfLiteContext* context,
                            const TfLiteNode* node, int id) {
  if (node->outputs->size <= id) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Data id ", id, " must be less than tflite node outputs size ",
        node->outputs->size));
  }
  uint32_t tensor_idx = node->outputs->data[id];
  return PreCheckReadValueByTensorIdx(context, tensor_idx);
}

absl::Status PreCheckOutputs(const TfLiteContext* context,
                             const TfLiteNode* node) {
  for (int i = 0; i < node->outputs->size; ++i) {
    RETURN_IF_ERROR(PreCheckOutput(context, node, i));
  }
  return absl::OkStatus();
}

absl::Status PreCheckMaybeFuseActivation(
    const TfLiteNode* node, TfLiteFusedActivation fused_activation) {
  if (node->outputs->size != 1) {
    return absl::InternalError("Number of outputs != 1");
  }
  return PreCheckMaybeFuseActivationSkipSize(node, fused_activation);
}

absl::Status PreCheckMaybeFuseActivationSkipSize(
    const TfLiteNode* node, TfLiteFusedActivation fused_activation) {
  switch (fused_activation) {
    case kTfLiteActNone:
      // Nothing to do here
    case kTfLiteActRelu:
    case kTfLiteActReluN1To1:
    case kTfLiteActRelu6:
    case kTfLiteActTanh:
    case kTfLiteActSigmoid:
      return absl::OkStatus();
    default:
      return absl::NotFoundError(
          absl::StrCat("Unsupported fused activation: ", fused_activation));
  }
  return absl::OkStatus();
}

absl::Status PreCheckMaybeFuseActivationForElementwiseNode(
    ::ml_drift::OperationType operation_type, const TfLiteNode* tflite_node) {
  TfLiteFusedActivation activation = kTfLiteActNone;
  switch (operation_type) {
    case ::ml_drift::OperationType::MUL: {
      const TfLiteMulParams* tf_options;
      if (PreCheckBuiltinData(tflite_node, &tf_options).ok()) {
        activation = tf_options->activation;
      }
      break;
    }
    case ::ml_drift::OperationType::ADD: {
      const TfLiteAddParams* tf_options;
      if (PreCheckBuiltinData(tflite_node, &tf_options).ok()) {
        activation = tf_options->activation;
      }
      break;
    }
    case ::ml_drift::OperationType::SUB: {
      const TfLiteSubParams* tf_options;
      if (PreCheckBuiltinData(tflite_node, &tf_options).ok()) {
        activation = tf_options->activation;
      }
      break;
    }
    case ::ml_drift::OperationType::DIV: {
      const TfLiteDivParams* tf_options;
      if (PreCheckBuiltinData(tflite_node, &tf_options).ok()) {
        activation = tf_options->activation;
      }
      break;
    }
    default:
      // No activation expected.
      activation = kTfLiteActNone;
  }

  if (activation) {
    return PreCheckMaybeFuseActivation(tflite_node, activation);
  }

  return absl::OkStatus();
}

}  // namespace litert::ml_drift
