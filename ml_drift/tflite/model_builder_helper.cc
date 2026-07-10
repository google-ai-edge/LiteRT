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

#include "third_party/odml/litert/ml_drift/tflite/model_builder_helper.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "third_party/FP16/include/fp16.h"
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/context_util.h"
#include "tflite/kernels/internal/portable_tensor_utils.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift {
namespace {

std::string GetTensorDebugString(const TfLiteTensor* tensor) {
  return std::string("{\n  type: ") + TfLiteTypeGetName(tensor->type) +
         "\n  data: {...}\n  dims: " +
         tflite::GetShapeDebugString(tensor->dims) + "\n}";
}

// Creates a node that consumes output from the given node. Because output need
// to stay the same, newly created node will inherit the output from the given
// node, which will in turn get newly created copy of output. This is necessary
// to preserve reference consistency if another node was pointing at that
// output:
//   node(output)
// will turn into:
//   node(copy(output)) <- passthrough_node(output)
::ml_drift::Node* NewPassthroughNode(::ml_drift::GraphFloat32* graph,
                                     ::ml_drift::Node* node,
                                     const ::ml_drift::Value* output) {
  ::ml_drift::Node* passthru_node = graph->NewNode();
  // Make copies for every output in the original node.
  graph->SetProducer(passthru_node->id, output->id);
  ::ml_drift::Value* copy_output = graph->NewValue();
  graph->SetProducer(node->id, copy_output->id);
  graph->AddConsumer(passthru_node->id, copy_output->id);
  copy_output->tensor = output->tensor;
  copy_output->tensor.ref = -1;
  return passthru_node;
}

}  // namespace

absl::Status GetNodeAndRegistration(TfLiteContext* context, int node_id,
                                    TfLiteNode** tflite_node,
                                    TfLiteRegistration** registration) {
  if (context->GetNodeAndRegistration(context, node_id, tflite_node,
                                      registration) != kTfLiteOk) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Couldn't get node and registration info for op: ", node_id));
  }
  return absl::OkStatus();
}

::ml_drift::DataType ToDataType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return ::ml_drift::DataType::FLOAT32;
    case kTfLiteInt16:
      return ::ml_drift::DataType::INT16;
    case kTfLiteInt32:
      return ::ml_drift::DataType::INT32;
    case kTfLiteInt64:
      return ::ml_drift::DataType::INT64;
    case kTfLiteFloat16:
      return ::ml_drift::DataType::FLOAT16;
    case kTfLiteBFloat16:
      return ::ml_drift::DataType::BFLOAT16;
    case kTfLiteInt8:
      return ::ml_drift::DataType::INT8;
    case kTfLiteUInt32:
      return ::ml_drift::DataType::UINT32;
    case kTfLiteUInt16:
      return ::ml_drift::DataType::UINT16;
    case kTfLiteUInt8:
      return ::ml_drift::DataType::UINT8;
    case kTfLiteBool:
      return ::ml_drift::DataType::BOOL;
    default:
      return ::ml_drift::DataType::UNKNOWN;
  }
}

::ml_drift::BHWC ExtractTensorShape(const TfLiteIntArray* dims) {
  const int size = dims->size;
  if (size == 0) return ::ml_drift::BHWC(1, 1, 1, 1);
  if (size == 1) return ::ml_drift::BHWC(dims->data[0], 1, 1, 1);
  if (size == 2) return ::ml_drift::BHWC(dims->data[0], 1, 1, dims->data[1]);
  if (size == 3)
    return ::ml_drift::BHWC(dims->data[0], 1, dims->data[1], dims->data[2]);
  return ::ml_drift::BHWC(dims->data[0], dims->data[1], dims->data[2],
                          dims->data[3]);
}

::ml_drift::BHWC ExtractTensorShape(const TfLiteTensor* tflite_tensor) {
  return ExtractTensorShape(tflite_tensor->dims);
}

::ml_drift::Axis ExtractAxisFromIndex(const TfLiteTensor& tflite_tensor,
                                      int index) {
  const TfLiteIntArray* dims = tflite_tensor.dims;
  if (index < 0) index = dims->size + index;
  std::vector<::ml_drift::Axis> index_to_axis;
  if (dims->size == 1) {
    index_to_axis = {::ml_drift::Axis::BATCH};
  } else if (dims->size == 2) {
    index_to_axis = {::ml_drift::Axis::BATCH, ::ml_drift::Axis::CHANNELS};
  } else if (dims->size == 3) {
    index_to_axis = {::ml_drift::Axis::BATCH, ::ml_drift::Axis::WIDTH,
                     ::ml_drift::Axis::CHANNELS};
  } else {
    index_to_axis = {::ml_drift::Axis::BATCH, ::ml_drift::Axis::HEIGHT,
                     ::ml_drift::Axis::WIDTH, ::ml_drift::Axis::CHANNELS};
  }
  return index_to_axis[index];
}

void PopulateQuantParams(const TfLiteTensor& tensor,
                         ::ml_drift::QuantizationParams* quant_params) {
  const TfLiteQuantization& quant = tensor.quantization;
  ABSL_QCHECK_EQ(quant.type, TfLiteQuantizationType::kTfLiteAffineQuantization);
  const TfLiteAffineQuantization* params =
      static_cast<const TfLiteAffineQuantization*>(quant.params);
  ABSL_QCHECK_EQ(params->scale->size, 1);
  const float scale = params->scale->data[0];
  const float zero_point = static_cast<float>(params->zero_point->data[0]);

  float qmin_value = 0;
  float qmax_value = 0;
  if (tensor.type == kTfLiteUInt8) {
    qmin_value = static_cast<float>(std::numeric_limits<uint8_t>::min());
    qmax_value = static_cast<float>(std::numeric_limits<uint8_t>::max());
  } else if (tensor.type == kTfLiteInt8) {
    qmin_value = static_cast<float>(std::numeric_limits<int8_t>::min());
    qmax_value = static_cast<float>(std::numeric_limits<int8_t>::max());
  } else {
    ABSL_LOG(FATAL) << absl::StrCat("Type invalid for quantized tensor: ",
                                    std::string(tensor.name));
  }
  quant_params->min = scale * (static_cast<float>(qmin_value) - zero_point);
  quant_params->max = scale * (static_cast<float>(qmax_value) - zero_point);
  quant_params->scale = scale;
}

int GetNumberOfRuntimeInputsForNode(const TfLiteContext* context,
                                    const TfLiteNode* tflite_node) {
  int number_of_runtime_inputs = 0;
  for (int i = 0; i < tflite::NumInputs(tflite_node); i++) {
    const TfLiteTensor* tensor =
        tflite::GetOptionalInputTensor(context, tflite_node, i);
    if (tensor != nullptr && !tflite::IsConstantTensor(tensor)) {
      number_of_runtime_inputs++;
    }
  }
  return number_of_runtime_inputs;
}

int GetNumberOfConstInputsForNode(const TfLiteContext* context,
                                  const TfLiteNode* tflite_node) {
  return tflite::NumInputs(tflite_node) -
         GetNumberOfRuntimeInputsForNode(context, tflite_node);
}

absl::Status CheckInputsOutputs(const TfLiteContext* context,
                                const TfLiteNode* tflite_node,
                                int runtime_inputs, int outputs) {
  const int runtime_inputs_from_model =
      GetNumberOfRuntimeInputsForNode(context, tflite_node);
  if (runtime_inputs_from_model != runtime_inputs) {
    return absl::InternalError(absl::StrCat(
        "Expected ", runtime_inputs, " runtime input tensor(s), but node has ",
        runtime_inputs_from_model, " runtime input(s)."));
  }
  const int outputs_from_model = tflite::NumOutputs(tflite_node);
  if (outputs_from_model != outputs) {
    return absl::InternalError(absl::StrCat("Expected ", outputs,
                                            " output tensor(s), but node has ",
                                            outputs_from_model, " output(s)."));
  }
  return absl::OkStatus();
}

absl::Status CheckInputsConstsOutputs(const TfLiteContext* context,
                                      const TfLiteNode* tflite_node,
                                      int runtime_inputs, int const_inputs,
                                      int outputs) {
  const int const_inputs_from_model =
      GetNumberOfConstInputsForNode(context, tflite_node);
  if (const_inputs_from_model != const_inputs) {
    return absl::InternalError(absl::StrCat(
        "Expected ", const_inputs, " const input tensor(s), but node has ",
        const_inputs_from_model, " const input(s)."));
  }
  return CheckInputsOutputs(context, tflite_node, runtime_inputs, outputs);
}

void ConvertFloat16ToFloat32(size_t num_elements, const uint16_t* src,
                             float* dst) {
  for (size_t i = 0; i < num_elements; i++) {
    *dst++ = fp16_ieee_to_fp32_value(*src++);
  }
}

template <>
void CopyData<float>(const TfLiteTensor& src, float* dst) {
  const TfLiteType dtype = src.type;
  if (dtype == kTfLiteFloat32 ||  //
      dtype == kTfLiteFloat16 ||  //
      dtype == kTfLiteInt4 ||     //
      dtype == kTfLiteInt8 ||     //
      dtype == kTfLiteUInt8 ||    //
      dtype == kTfLiteInt32) {
    CopyFloat32Data(&src, dst);
    return;
  }
  ABSL_LOG(FATAL) << absl::StrCat(GetTensorDebugString(&src),
                                  " has unsupported dtype.");
}

void CopyFloat32Data(const TfLiteTensor* tfl_tensor, float* dst) {
  const TfLiteType dtype = tfl_tensor->type;
  if (dtype == kTfLiteFloat32) {
    std::memcpy(dst, tfl_tensor->data.f, tfl_tensor->bytes);
  } else if (dtype == kTfLiteFloat16) {
    ConvertFloat16ToFloat32(
        tflite::NumElements(tfl_tensor),
        reinterpret_cast<uint16_t const*>(tfl_tensor->data.f16), dst);
  } else if (dtype == kTfLiteInt4) {
    // Unpack the int4 data into int8 data and then dequantize it.
    // The temporary `bytes_unpacked` may have one more byte if the
    // number of elements is odd but the dequantized `dst` will have the
    // correct number of elements by DequantizeConstantTensor().
    const size_t bytes_unpacked = tfl_tensor->bytes * 2;
    auto unpacked_input_data = std::make_unique<int8_t[]>(bytes_unpacked);
    tflite::tensor_utils::UnpackPackedIntToInt8(tfl_tensor->data.int8,
                                                bytes_unpacked, /*bit_width=*/4,
                                                unpacked_input_data.get());
    const int8_t* input_data = unpacked_input_data.get();
    DequantizeConstantTensor(*tfl_tensor, input_data, dst);
  } else if (dtype == kTfLiteInt8) {
    DequantizeConstantTensor(*tfl_tensor, tfl_tensor->data.int8, dst);
  } else if (dtype == kTfLiteUInt8) {
    DequantizeConstantTensor(*tfl_tensor, tfl_tensor->data.uint8, dst);
  } else if (dtype == kTfLiteInt32) {
    DequantizeConstantTensor(*tfl_tensor, tfl_tensor->data.i32, dst);
  }
}

std::string GetDimensionString(const TfLiteIntArray* dimensions) {
  return absl::StrJoin(tflite::TfLiteIntArrayView(dimensions), "x");
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::Scalar* shape) {
  for (int i = 0; i < dims->size; ++i) ABSL_QCHECK_EQ(dims->data[i], 1);
  shape->v = 1;
}

bool IsLinearConvertible(const TfLiteIntArray* dims) {
  if (dims->size <= 0) return false;
  for (int i = 0; i < dims->size - 1; ++i) {
    if (dims->data[i] != 1) return false;
  }
  return true;
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::Linear* shape) {
  ABSL_QCHECK(IsLinearConvertible(dims));
  shape->v = dims->data[dims->size - 1];
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::HWC* shape) {
  if (dims->size == 3) {
    shape->h = dims->data[0];
    shape->w = dims->data[1];
    shape->c = dims->data[2];
    return;
  }
  if (dims->size == 4) {
    ABSL_QCHECK_EQ(dims->data[0], 1);
    shape->h = dims->data[1];
    shape->w = dims->data[2];
    shape->c = dims->data[3];
    return;
  }
  ABSL_LOG(FATAL) << absl::StrCat(
      "Expected a 3D tensor of shape HxWxC or a 4D tensor of "
      "shape 1xHxWxC but got ",
      GetDimensionString(dims));
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::HW* shape) {
  ABSL_QCHECK_EQ(dims->size, 2);
  shape->h = dims->data[0];
  shape->w = dims->data[1];
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::OHWI* shape) {
  ABSL_QCHECK_EQ(dims->size, 4);
  shape->o = dims->data[0];
  shape->h = dims->data[1];
  shape->w = dims->data[2];
  shape->i = dims->data[3];
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::BHWC* shape) {
  *shape = ExtractTensorShape(dims);
}

void HandleFusedActivation(TfLiteFusedActivation fused_activation,
                           ::ml_drift::GraphFloat32* graph,
                           ::ml_drift::Node* node) {
  const auto outputs = graph->FindOutputs(node->id);
  switch (fused_activation) {
    case kTfLiteActNone:
      return;
    case kTfLiteActRelu:
    case kTfLiteActReluN1To1:
    case kTfLiteActRelu6: {
      ::ml_drift::ReLUAttributes attr;
      attr.activation_max =
          fused_activation == kTfLiteActRelu
              ? 0.0f
              : (fused_activation == kTfLiteActReluN1To1 ? 1.0f : 6.0f);
      attr.activation_min =
          fused_activation == kTfLiteActReluN1To1 ? -1.0f : 0.0f;
      ::ml_drift::Node* activation_node =
          NewPassthroughNode(graph, node, outputs[0]);
      activation_node->operation.type =
          ToString(::ml_drift::OperationType::RELU);
      activation_node->operation.attributes = attr;
      return;
    }
    case kTfLiteActTanh: {
      ::ml_drift::Node* activation_node =
          NewPassthroughNode(graph, node, outputs[0]);
      activation_node->operation.type =
          ToString(::ml_drift::OperationType::TANH);
      return;
    }
    case kTfLiteActSigmoid: {
      ::ml_drift::Node* activation_node =
          NewPassthroughNode(graph, node, outputs[0]);
      activation_node->operation.type =
          ToString(::ml_drift::OperationType::SIGMOID);
      return;
    }
    case kTfLiteActSignBit:
      ::ml_drift::Node* activation_node =
          NewPassthroughNode(graph, node, outputs[0]);
      activation_node->operation.type =
          ToString(::ml_drift::OperationType::SIGN);
      return;
    // DO NOT add `default:` for compiler checks.
  }
}

// Scan dimensions from right to left and return false if there is a mismatch
// and the mismatch isn't 1.
bool IsBroadcastable(const TfLiteIntArray* dims1, const TfLiteIntArray* dims2) {
  int idx1 = dims1->size - 1;
  int idx2 = dims2->size - 1;
  for (int i = std::max(idx1, idx2); i >= 0; --i) {
    int data1 = idx1 < 0 ? 1 : dims1->data[idx1];
    int data2 = idx2 < 0 ? 1 : dims2->data[idx2];
    if (data1 != data2 && data1 != 1 && data2 != 1) return false;
    --idx1;
    --idx2;
  }
  return true;
}

}  // namespace litert::ml_drift
