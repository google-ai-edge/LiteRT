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

#include "ml_drift_delegate/tflite/model_builder_helper.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "fp16.h"  // from @FP16
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
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
      return ::ml_drift::DataType::kFloat32;
    case kTfLiteInt16:
      return ::ml_drift::DataType::kInt16;
    case kTfLiteInt32:
      return ::ml_drift::DataType::kInt32;
    case kTfLiteInt64:
      return ::ml_drift::DataType::kInt64;
    case kTfLiteFloat16:
      return ::ml_drift::DataType::kFloat16;
    case kTfLiteBFloat16:
      return ::ml_drift::DataType::kBfloat16;
    case kTfLiteInt8:
      return ::ml_drift::DataType::kInt8;
    case kTfLiteInt4:
      return ::ml_drift::DataType::kInt4;
    case kTfLiteInt2:
      return ::ml_drift::DataType::kInt2;
    case kTfLiteUInt32:
      return ::ml_drift::DataType::kUint32;
    case kTfLiteUInt16:
      return ::ml_drift::DataType::kUint16;
    case kTfLiteUInt8:
      return ::ml_drift::DataType::kUint8;
    case kTfLiteBool:
      return ::ml_drift::DataType::kBool;
    default:
      return ::ml_drift::DataType::kUnknown;
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
    index_to_axis = {::ml_drift::Axis::kBatch};
  } else if (dims->size == 2) {
    index_to_axis = {::ml_drift::Axis::kBatch, ::ml_drift::Axis::kChannels};
  } else if (dims->size == 3) {
    index_to_axis = {::ml_drift::Axis::kBatch, ::ml_drift::Axis::kWidth,
                     ::ml_drift::Axis::kChannels};
  } else {
    index_to_axis = {::ml_drift::Axis::kBatch, ::ml_drift::Axis::kHeight,
                     ::ml_drift::Axis::kWidth, ::ml_drift::Axis::kChannels};
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
  } else if (tensor.type == kTfLiteUInt4) {
    qmin_value = 0.0f;
    qmax_value = 15.0f;
  } else if (tensor.type == kTfLiteInt4) {
    qmin_value = -8.0f;
    qmax_value = 7.0f;
  } else if (tensor.type == kTfLiteInt2) {
    qmin_value = -2.0f;
    qmax_value = 1.0f;
  } else {
    ABSL_LOG(FATAL) << absl::StrCat("Type invalid for quantized tensor: ",
                                    tensor.name ? tensor.name : "unknown");
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
      dtype == kTfLiteInt32 ||    //
      dtype == kTfLiteBool) {
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
  } else if (dtype == kTfLiteBool) {
    const int num_elements = tflite::NumElements(tfl_tensor);
    for (int i = 0; i < num_elements; ++i) {
      dst[i] = static_cast<float>(tfl_tensor->data.b[i]);
    }
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
          ToString(::ml_drift::OperationType::kRelu);
      activation_node->operation.attributes = attr;
      return;
    }
    case kTfLiteActTanh: {
      ::ml_drift::Node* activation_node =
          NewPassthroughNode(graph, node, outputs[0]);
      activation_node->operation.type =
          ToString(::ml_drift::OperationType::kTanh);
      return;
    }
    case kTfLiteActSigmoid: {
      ::ml_drift::Node* activation_node =
          NewPassthroughNode(graph, node, outputs[0]);
      activation_node->operation.type =
          ToString(::ml_drift::OperationType::kSigmoid);
      return;
    }
    case kTfLiteActSignBit:
      ::ml_drift::Node* activation_node =
          NewPassthroughNode(graph, node, outputs[0]);
      activation_node->operation.type =
          ToString(::ml_drift::OperationType::kSign);
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

namespace {

// Returns element `index` of the blockwise scale or zero point tensor `param`.
float ReadBlockwiseV2Param(const TfLiteTensor& param, int64_t index) {
  switch (param.type) {
    case kTfLiteFloat32:
      return param.data.f[index];
    case kTfLiteFloat16:
      return fp16_ieee_to_fp32_value(param.data.f16[index].data);
    case kTfLiteInt32:
      return static_cast<float>(param.data.i32[index]);
    case kTfLiteInt16:
      return static_cast<float>(param.data.i16[index]);
    case kTfLiteInt8:
      return static_cast<float>(param.data.int8[index]);
    default:
      ABSL_LOG(FATAL) << "Unsupported blockwise quantization parameter type: "
                      << TfLiteTypeGetName(param.type);
  }
}

// Returns the index, in the blockwise scale or zero point tensor `param`, of
// the block that holds the element at `coords` of the quantized tensor. A
// `param` extent of 1 broadcasts along that dimension.
int64_t BlockwiseV2ParamIndex(const TfLiteTensor& param,
                              const TfLiteIntArray& block_shape,
                              const int* coords) {
  int64_t index = 0;
  for (int d = 0; d < param.dims->size; ++d) {
    const int extent = param.dims->data[d];
    index =
        index * extent + (extent == 1 ? 0 : coords[d] / block_shape.data[d]);
  }
  return index;
}

// Returns the element size in bytes of a blockwise quantization scale or zero
// point tensor, or UnimplementedError if `type` is not supported for that role.
absl::StatusOr<size_t> GetBlockwiseV2ParamElementSize(TfLiteType type,
                                                      bool is_scale) {
  switch (type) {
    case kTfLiteFloat32:
      return sizeof(float);
    case kTfLiteFloat16:
      if (is_scale) return sizeof(uint16_t);
      break;
    case kTfLiteInt32:
      if (!is_scale) return sizeof(int32_t);
      break;
    case kTfLiteInt16:
      if (!is_scale) return sizeof(int16_t);
      break;
    case kTfLiteInt8:
      if (!is_scale) return sizeof(int8_t);
      break;
    default:
      break;
  }
  return absl::UnimplementedError(absl::StrCat(
      "Unsupported blockwise quantization ", is_scale ? "scale" : "zero point",
      " type: ", TfLiteTypeGetName(type)));
}

// Returns `&tensors[index]` if `index` is within `[0, tensors_size)`, or
// nullptr otherwise.
const TfLiteTensor* GetTensorAt(const TfLiteTensor* tensors,
                                size_t tensors_size, int index) {
  return index >= 0 && static_cast<size_t>(index) < tensors_size
             ? &tensors[index]
             : nullptr;
}

// Validates a blockwise quantization scale or zero point tensor `param` against
// the quantized `tensor` and `block_shape`:
//  * supported element dtype for scale vs. zero point,
//  * constant allocation with non-null data pointer,
//  * same rank as `tensor`,
//  * per-axis extent of either 1 (broadcast) or ceil(dim / block_shape[d]),
//  * buffer byte size large enough for all elements without integer overflow.
absl::Status ValidateBlockwiseV2Param(const TfLiteTensor& tensor,
                                      const TfLiteIntArray& block_shape,
                                      const TfLiteTensor& param,
                                      bool is_scale) {
  ABSL_ASSIGN_OR_RETURN(const size_t element_size,
                        GetBlockwiseV2ParamElementSize(param.type, is_scale));
  if (!tflite::IsConstantTensor(&param) || param.data.raw_const == nullptr) {
    return absl::UnimplementedError(
        "Blockwise quantization scales and zero points must be constant.");
  }
  if (param.dims == nullptr || param.dims->size != tensor.dims->size) {
    return absl::InvalidArgumentError(
        "Blockwise quantization scales and zero points must have the rank of "
        "the quantized tensor.");
  }
  const size_t max_elements = param.bytes / element_size;
  size_t num_elements = 1;
  for (int d = 0; d < param.dims->size; ++d) {
    const int block = block_shape.data[d];
    const int num_blocks = 1 + (tensor.dims->data[d] - 1) / block;
    const int extent = param.dims->data[d];
    if (extent != 1 && extent != num_blocks) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Blockwise quantization scales and zero points must have an extent "
          "of 1 or of the number of blocks along every dimension, got ",
          extent, " for ", num_blocks, " blocks along dimension ", d, "."));
    }
    if (static_cast<size_t>(extent) > max_elements / num_elements) {
      return absl::InvalidArgumentError(
          "Blockwise quantization scale or zero point buffer is too small.");
    }
    num_elements *= static_cast<size_t>(extent);
  }
  if (num_elements > max_elements) {
    return absl::InvalidArgumentError(
        "Blockwise quantization scale or zero point buffer is too small.");
  }
  return absl::OkStatus();
}

// Indices of the dimensions of blockwise quantized FullyConnected weights.
struct BlockwiseV2WeightsDims {
  int o = 0;
  int i = 1;
  // Dimension of the weights batch, -1 for 2-D FULLY_CONNECTED weights.
  int h = -1;
};

absl::StatusOr<BlockwiseV2WeightsDims> GetBlockwiseV2WeightsDims(
    const TfLiteIntArray& dims, bool oi_transposed) {
  if (dims.size == 2) {
    return BlockwiseV2WeightsDims{.o = 0, .i = 1, .h = -1};
  }
  if (dims.size == 3) {
    return oi_transposed ? BlockwiseV2WeightsDims{.o = 2, .i = 1, .h = 0}
                         : BlockwiseV2WeightsDims{.o = 1, .i = 2, .h = 0};
  }
  if (dims.size == 4 && dims.data[0] == 1) {
    return oi_transposed ? BlockwiseV2WeightsDims{.o = 3, .i = 2, .h = 1}
                         : BlockwiseV2WeightsDims{.o = 2, .i = 3, .h = 1};
  }
  return absl::UnimplementedError(
      "Blockwise quantized weights must be [O, I], [H, I, O], [H, O, I], "
      "[1, H, I, O] or [1, H, O, I].");
}

}  // namespace

absl::StatusOr<BlockwiseV2Params> GetBlockwiseV2Params(
    const TfLiteTensor* tensors, size_t tensors_size,
    const TfLiteTensor& tensor) {
  if (tensor.quantization.type != kTfLiteBlockwiseQuantizationV2 ||
      tensor.quantization.params == nullptr) {
    return absl::InvalidArgumentError(
        "Tensor has no N-D blockwise quantization.");
  }
  if (tensor.type != kTfLiteInt2 && tensor.type != kTfLiteInt4 &&
      tensor.type != kTfLiteInt8) {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported blockwise quantized tensor type: ",
                     TfLiteTypeGetName(tensor.type)));
  }
  const auto* quant = static_cast<const TfLiteBlockwiseQuantizationV2*>(
      tensor.quantization.params);
  if (tensor.dims == nullptr || quant->block_shape == nullptr ||
      quant->block_shape->size != tensor.dims->size) {
    return absl::InvalidArgumentError(
        "Blockwise quantization needs one block extent per dimension.");
  }
  for (int d = 0; d < tensor.dims->size; ++d) {
    if (tensor.dims->data[d] <= 0 || quant->block_shape->data[d] <= 0) {
      return absl::InvalidArgumentError(
          "Blockwise quantized tensors need positive dimensions and block "
          "extents.");
    }
  }
  BlockwiseV2Params params;
  params.scale = GetTensorAt(tensors, tensors_size, quant->scale);
  if (params.scale == nullptr) {
    return absl::InvalidArgumentError(
        "Invalid blockwise quantization scale tensor index.");
  }
  ABSL_RETURN_IF_ERROR(ValidateBlockwiseV2Param(
      tensor, *quant->block_shape, *params.scale, /*is_scale=*/true));
  if (quant->zero_point != kTfLiteOptionalTensor) {
    params.zero_point = GetTensorAt(tensors, tensors_size, quant->zero_point);
    if (params.zero_point == nullptr) {
      return absl::InvalidArgumentError(
          "Invalid blockwise quantization zero point tensor index.");
    }
    ABSL_RETURN_IF_ERROR(ValidateBlockwiseV2Param(
        tensor, *quant->block_shape, *params.zero_point, /*is_scale=*/false));
  }
  return params;
}

void DequantizeBlockwiseV2(const TfLiteTensor& tensor,
                           const BlockwiseV2Params& params, float* dst) {
  const auto* quant = static_cast<const TfLiteBlockwiseQuantizationV2*>(
      tensor.quantization.params);
  const TfLiteIntArray& block_shape = *quant->block_shape;
  const int64_t num_elements = tflite::NumElements(&tensor);
  ABSL_QCHECK_GE(num_elements, 0);
  ABSL_QCHECK_LE(num_elements, std::numeric_limits<int>::max());
  ABSL_QCHECK(tensor.data.raw_const != nullptr);
  int64_t required_bytes = num_elements;
  if (tensor.type == kTfLiteInt4) {
    required_bytes = (num_elements + 1) / 2;
  } else if (tensor.type == kTfLiteInt2) {
    required_bytes = (num_elements + 3) / 4;
  } else {
    ABSL_QCHECK_EQ(tensor.type, kTfLiteInt8);
  }
  ABSL_QCHECK_GE(tensor.bytes, static_cast<size_t>(required_bytes));
  const int8_t* quantized = tensor.data.int8;
  std::vector<int8_t> unpacked;
  if (tensor.type != kTfLiteInt8) {
    unpacked.resize(num_elements);
    tflite::tensor_utils::UnpackPackedIntToInt8(
        tensor.data.int8, static_cast<int>(num_elements),
        /*bit_width=*/tensor.type == kTfLiteInt2 ? 2 : 4, unpacked.data());
    quantized = unpacked.data();
  }
  const int rank = tensor.dims->size;
  std::vector<int> coords(rank, 0);
  for (int64_t e = 0; e < num_elements; ++e) {
    const float scale = ReadBlockwiseV2Param(
        *params.scale,
        BlockwiseV2ParamIndex(*params.scale, block_shape, coords.data()));
    const float zero_point =
        params.zero_point == nullptr
            ? 0.0f
            : ReadBlockwiseV2Param(
                  *params.zero_point,
                  BlockwiseV2ParamIndex(*params.zero_point, block_shape,
                                        coords.data()));
    dst[e] = scale * (static_cast<float>(quantized[e]) - zero_point);
    for (int d = rank - 1; d >= 0; --d) {
      if (++coords[d] < tensor.dims->data[d]) break;
      coords[d] = 0;
    }
  }
}

absl::Status GetBlockwiseV2WeightsInfo(const TfLiteTensor& tensor,
                                       const BlockwiseV2Params& params,
                                       BlockwiseV2WeightsInfo* info,
                                       bool oi_transposed) {
  const auto* quant = static_cast<const TfLiteBlockwiseQuantizationV2*>(
      tensor.quantization.params);
  ABSL_ASSIGN_OR_RETURN(
      const BlockwiseV2WeightsDims dims,
      GetBlockwiseV2WeightsDims(*tensor.dims, oi_transposed));
  const int in_channels = tensor.dims->data[dims.i];
  const int block = quant->block_shape->data[dims.i];
  const int scale_blocks = params.scale->dims->data[dims.i];
  const int zp_blocks = params.zero_point != nullptr
                            ? params.zero_point->dims->data[dims.i]
                            : 1;
  if (scale_blocks != 1 && zp_blocks != 1 && scale_blocks != zp_blocks) {
    return absl::InvalidArgumentError(
        "Blockwise quantized scale and zero point must have the same number "
        "of blocks along the input channels.");
  }
  const int num_blocks = std::max(scale_blocks, zp_blocks);
  if (num_blocks > 1 &&
      (in_channels % block != 0 || num_blocks != in_channels / block)) {
    return absl::UnimplementedError(
        "Blockwise quantized weights need blocks of the same size along the "
        "input channels.");
  }
  info->weights_shape = ::ml_drift::OHWI(
      tensor.dims->data[dims.o], dims.h < 0 ? 1 : tensor.dims->data[dims.h], 1,
      in_channels);
  info->scale_shape = ::ml_drift::OHWI(info->weights_shape.o,
                                       info->weights_shape.h, 1, num_blocks);
  info->has_zero_point = params.zero_point != nullptr;
  info->oi_transposed = oi_transposed;
  return absl::OkStatus();
}

std::vector<float> ReadBlockwiseV2WeightsParam(
    const TfLiteTensor& tensor, const TfLiteTensor& param,
    const BlockwiseV2WeightsInfo& info) {
  const auto* quant = static_cast<const TfLiteBlockwiseQuantizationV2*>(
      tensor.quantization.params);
  const absl::StatusOr<BlockwiseV2WeightsDims> dims =
      GetBlockwiseV2WeightsDims(*tensor.dims, info.oi_transposed);
  ABSL_QCHECK_OK(dims.status());
  const ::ml_drift::OHWI& shape = info.scale_shape;
  // Input channels covered by each of the `shape.i` values.
  const int block = info.weights_shape.i / shape.i;
  std::vector<float> values(shape.DimensionsProduct());
  int coords[4] = {0, 0, 0, 0};
  for (int o = 0; o < shape.o; ++o) {
    coords[dims->o] = o;
    for (int h = 0; h < shape.h; ++h) {
      if (dims->h >= 0) coords[dims->h] = h;
      for (int b = 0; b < shape.i; ++b) {
        coords[dims->i] = b * block;
        values[(o * shape.h + h) * shape.i + b] = ReadBlockwiseV2Param(
            param, BlockwiseV2ParamIndex(param, *quant->block_shape, coords));
      }
    }
  }
  return values;
}

}  // namespace litert::ml_drift
