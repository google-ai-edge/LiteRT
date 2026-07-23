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

#include "ml_drift_delegate/tflite/model_builder.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "fp16.h"  // from @FP16
#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/transformations/model_transformations.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/custom_parsers.h"
#include "ml_drift_delegate/tflite/lstm_parser.h"
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/core/api/op_resolver.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/c/c_api.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/delegates/utils.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/internal/portable_tensor_utils.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/kernels/lstm_shared.h"
#include "tflite/model_builder.h"
#include "tflite/tools/versioning/gpu_compatibility.h"
#include "tflite/tools/versioning/op_signature.h"
#include "tflite/util.h"

namespace litert::ml_drift {
namespace {

inline std::string GetTensorDebugString(const TfLiteTensor* tensor) {
  return absl::StrCat("{\n  type: ", TfLiteTypeGetName(tensor->type),
                      "\n  data: {...}\n  dims: ",
                      tflite::GetShapeDebugString(tensor->dims), "\n}");
}

::ml_drift::FullyConnectedAttributes GetFullyConnectedAttributes(
    int weights_node_input_index, int bias_node_input_index,
    ObjectReader* reader) {
  ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32> weights;
  reader->ReadTensor(weights_node_input_index, &weights,
                     ReadTensorFlags::kExtraBytes);
  ::ml_drift::FullyConnectedAttributes attr;
  attr.weights.data = std::move(weights.data);
  attr.weights.id = weights.id;
  attr.weights.shape.h = 1;
  attr.weights.shape.w = 1;
  attr.weights.shape.o = weights.shape.h;
  attr.weights.shape.i = weights.shape.w;
  if (reader->IsNodeInputTensorPresent(bias_node_input_index)) {
    reader->ReadTensor(bias_node_input_index, &attr.bias,
                       ReadTensorFlags::kNoExtraBytes);
  }
  return attr;
}

// if copy_weights is true, input can be int4 and then it will be unpacked to
// int8.
::ml_drift::FullyConnectedInt8Attributes GetFullyConnectedInt8Attributes(
    int weights_node_input_index, int bias_node_input_index,
    ObjectReader* reader, bool copy_weights = true) {
  const TfLiteTensor* weights_tensor =
      reader->GetInputTensor(weights_node_input_index);

  const bool supported_int8 = weights_tensor->type == kTfLiteInt8;
  const bool supported_int4 =
      copy_weights && weights_tensor->type == kTfLiteInt4;
  const bool supported_int2 =
      copy_weights && weights_tensor->type == kTfLiteInt2;
  ABSL_QCHECK(supported_int8 || supported_int4 || supported_int2)
      << "Unsupported weights type in GetFullyConnectedInt8Attributes.";

  ::ml_drift::FullyConnectedInt8Attributes attr;
  ::ml_drift::BHWC weights_shape = ExtractTensorShape(weights_tensor);
  attr.weights.shape.o = weights_shape.b;
  attr.weights.shape.h = weights_shape.h;
  attr.weights.shape.w = weights_shape.w;
  attr.weights.shape.i = weights_shape.c;
  if (copy_weights) {
    if (weights_tensor->type == kTfLiteInt8) {
      std::vector<int8_t> temp(weights_tensor->bytes + XNN_EXTRA_BYTES);
      std::memcpy(&temp[0], weights_tensor->data.raw, weights_tensor->bytes);
      attr.weights.data = std::move(temp);
    } else if (weights_tensor->type == kTfLiteInt4) {
      // Unpack the int4 data into int8 data.
      attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                               XNN_EXTRA_BYTES);
      tflite::tensor_utils::UnpackPackedIntToInt8(
          weights_tensor->data.int8, attr.weights.shape.DimensionsProduct(),
          /*bit_width=*/4, &attr.weights.data[0]);
    } else if (weights_tensor->type == kTfLiteInt2) {
      // Unpack the int2 data into int8 data.
      attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                               XNN_EXTRA_BYTES);
      tflite::tensor_utils::UnpackPackedIntToInt8(
          weights_tensor->data.int8, attr.weights.shape.DimensionsProduct(),
          /*bit_width=*/2, &attr.weights.data[0]);
    }
  } else {
    attr.weights.spanned_data =
        absl::MakeSpan(weights_tensor->data.int8, weights_tensor->bytes);
  }

  if (weights_tensor->quantization.type == kTfLiteAffineQuantization) {
    const auto* qparams = reinterpret_cast<TfLiteAffineQuantization*>(
        weights_tensor->quantization.params);
    if (copy_weights) {
      attr.scale.data = std::vector<float>(
          qparams->scale->data, qparams->scale->data + qparams->scale->size);
    } else {
      attr.scale.spanned_data =
          absl::MakeSpan(qparams->scale->data, qparams->scale->size);
    }
    attr.scale.shape = ::ml_drift::OHWI(qparams->scale->size, 1, 1, 1);
    if (copy_weights) {
      attr.zero_point.data = std::vector<int32_t>(
          qparams->zero_point->data,
          qparams->zero_point->data + qparams->zero_point->size);
      attr.zero_point.data.resize(qparams->scale->size,
                                  attr.zero_point.data[0]);
    } else {
      // If the zero points have been de-duplicated, we need to broadcast them.
      if (qparams->zero_point->size != qparams->scale->size) {
        attr.zero_point.data.assign(qparams->scale->size,
                                    qparams->zero_point->data[0]);
      } else {
        attr.zero_point.spanned_data = absl::MakeSpan(
            qparams->zero_point->data, qparams->zero_point->size);
      }
    }
    // TFLite assumes that scale and zero_point have the same logical shape.
    // Since TFLite de-duplicates zero_point and its physical storage shape
    // might  be shrinked from its logical shape, we should set zero_point's
    // logical shape by scale's shape.
    attr.zero_point.shape = ::ml_drift::OHWI(qparams->scale->size, 1, 1, 1);
    if (reader->IsNodeInputTensorPresent(bias_node_input_index)) {
      reader->ReadTensor(bias_node_input_index, &attr.bias,
                         ReadTensorFlags::kNoExtraBytes);
    }
    return attr;
  }

  if (weights_tensor->quantization.type == kTfLiteBlockwiseQuantization) {
    const auto* qparams = reinterpret_cast<TfLiteBlockwiseQuantization*>(
        weights_tensor->quantization.params);
    const TfLiteTensor* scale = reader->GetTensor(qparams->scale);
    if (copy_weights) {
      if (scale->type == kTfLiteFloat32) {
        attr.scale.data = std::vector<float>(
            scale->data.f, scale->data.f + scale->bytes / sizeof(float));
      } else if (scale->type == kTfLiteFloat16) {
        const auto* scale2 = reinterpret_cast<TfLiteFloat16*>(scale->data.f16);
        for (int i = 0; i < scale->bytes / sizeof(TfLiteFloat16); ++i) {
          attr.scale.data.push_back(fp16_ieee_to_fp32_value(scale2[i].data));
        }
      } else {
        ABSL_LOG(FATAL) << "Unimplemented scale dtype: " << scale->type;
      }
    } else {
      if (scale->type == kTfLiteFloat32) {
        attr.scale.spanned_data =
            absl::MakeSpan(scale->data.f, scale->bytes / sizeof(float));
      } else {
        ABSL_LOG(FATAL) << "Unimplemented scale dtype: " << scale->type;
      }
    }
    attr.scale.shape = ::ml_drift::OHWI(
        attr.weights.shape.o, 1, 1, attr.weights.shape.i / qparams->blocksize);
    // Add zero points if they are available.
    if (qparams->zero_point >= 0) {
      const TfLiteTensor* zp = reader->GetTensor(qparams->zero_point);
      if (copy_weights) {
        if (zp->type == kTfLiteInt32) {
          attr.zero_point.data = std::vector<int32_t>(
              zp->data.i32, zp->data.i32 + zp->bytes / sizeof(int32_t));
        } else if (zp->type == kTfLiteInt64) {
          auto zp2 = absl::MakeSpan(reinterpret_cast<int64_t*>(zp->data.i64),
                                    zp->bytes / sizeof(int64_t));
          attr.zero_point.data.assign(zp2.begin(), zp2.end());
        } else {
          ABSL_LOG(FATAL) << "Unimplemented zero_point dtype: " << zp->type;
        }
      } else {
        if (zp->type == kTfLiteInt32) {
          attr.zero_point.spanned_data =
              absl::MakeSpan(zp->data.i32, zp->bytes / sizeof(int32_t));
        } else {
          ABSL_LOG(FATAL) << "Unimplemented zero_point dtype: " << zp->type;
        }
      }
      attr.zero_point.shape = attr.scale.shape;
    }
    if (reader->IsNodeInputTensorPresent(bias_node_input_index)) {
      reader->ReadTensor(bias_node_input_index, &attr.bias,
                         ReadTensorFlags::kNoExtraBytes);
    }
    return attr;
  }

  ABSL_LOG(FATAL) << "Unimplemented quantization.";
}

::ml_drift::FullyConnectedInt4Attributes GetFullyConnectedInt4Attributes(
    int weights_node_input_index, int bias_node_input_index,
    ObjectReader* reader, bool copy_weights = true) {
  auto attr8 = GetFullyConnectedInt8Attributes(
      weights_node_input_index, bias_node_input_index, reader, copy_weights);
  ::ml_drift::FullyConnectedInt4Attributes attr4;
  attr4.weights = std::move(attr8.weights);
  attr4.scale = std::move(attr8.scale);
  attr4.zero_point = std::move(attr8.zero_point);
  attr4.bias = std::move(attr8.bias);
  attr4.op_name = std::move(attr8.op_name);
  return attr4;
}

// Sets the output shape and expected output shape of a fully connected node.
void SetFullyConnectedOutputShape(const TfLiteTensor* src_tensor,
                                  const TfLiteTensor* weights_tensor,
                                  ObjectReader* reader,
                                  ::ml_drift::BHWC& output_shape,
                                  ::ml_drift::BHWC& expected_output_shape) {
  ::ml_drift::BHWC input_shape = ExtractTensorShape(src_tensor);
  ::ml_drift::BHWC weights_shape = ExtractTensorShape(weights_tensor);
  output_shape = ExtractTensorShape(reader->GetOutputTensor(0));
  expected_output_shape = input_shape;
  expected_output_shape.c = weights_shape.b;
}

bool IsFullyConnectedOutputReshapeNeeded(const TfLiteTensor* src_tensor,
                                         const TfLiteTensor* weights_tensor,
                                         ObjectReader* reader) {
  ::ml_drift::BHWC output_shape;
  ::ml_drift::BHWC expected_output_shape;
  SetFullyConnectedOutputShape(src_tensor, weights_tensor, reader, output_shape,
                               expected_output_shape);
  return output_shape != expected_output_shape;
}

void ReshapeFullyConnectedOutput(const TfLiteTensor* src_tensor,
                                 const TfLiteTensor* weights_tensor,
                                 ::ml_drift::GraphFloat32* graph,
                                 ObjectReader* reader, ::ml_drift::Node* node) {
  ::ml_drift::BHWC output_shape;
  ::ml_drift::BHWC expected_output_shape;
  SetFullyConnectedOutputShape(src_tensor, weights_tensor, reader, output_shape,
                               expected_output_shape);
  ::ml_drift::Value* copy_value = graph->NewValue();
  auto input_value = graph->FindInputs(node->id)[0];
  copy_value->tensor.type = input_value->tensor.type;
  copy_value->tensor.shape = expected_output_shape;
  ::ml_drift::Node* node_reshape = graph->NewNode();
  node_reshape->operation.type = ToString(::ml_drift::OperationType::RESHAPE);
  ::ml_drift::ReshapeAttributes reshape_attr;
  reshape_attr.new_shape = output_shape;
  node_reshape->operation.attributes = reshape_attr;
  graph->SetProducer(node->id, copy_value->id);
  graph->AddConsumer(node_reshape->id, copy_value->id);
  reader->AddOutputs(node_reshape);
}

void ConfigSharedWeightFullyConnectedNode(
    const int tflite_weights_tensor_index,
    const TfLiteTensor* const weights_tensor, ObjectReader* reader,
    ::ml_drift::GraphFloat32* graph, ::ml_drift::Node* node) {
  ::ml_drift::Value* input;
  reader->ReadQuantizedValueByTensorIdx(tflite_weights_tensor_index, &input);
  const auto shape =
      ::ml_drift::OHWI(input->tensor.shape.b, input->tensor.shape.h,
                       input->tensor.shape.w, input->tensor.shape.c);

  ::ml_drift::OHWI scale_shape = ::ml_drift::OHWI(shape.o, 1, 1, 1);
  if (weights_tensor->quantization.type == kTfLiteBlockwiseQuantization) {
    const auto* qparams = reinterpret_cast<const TfLiteBlockwiseQuantization*>(
        weights_tensor->quantization.params);
    scale_shape.i = shape.i / qparams->blocksize;
  }

  if (weights_tensor->type == kTfLiteInt8) {
    node->operation.type =
        ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT8);
    ::ml_drift::FullyConnectedInt8Attributes attr;
    attr.weights.shape = shape;
    attr.scale.shape = scale_shape;
    node->operation.attributes = std::move(attr);
  } else if (weights_tensor->type == kTfLiteInt4) {
    node->operation.type =
        ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT4);
    ::ml_drift::FullyConnectedInt4Attributes attr;
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4>
        int4_weights;
    int4_weights.shape = shape;
    attr.weights = int4_weights;
    attr.scale.shape = scale_shape;
    node->operation.attributes = std::move(attr);
  } else if (weights_tensor->type == kTfLiteInt2) {
    node->operation.type =
        ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT2);
    ::ml_drift::FullyConnectedInt2Attributes attr;
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT2>
        int2_weights;
    int2_weights.shape = shape;
    attr.weights = int2_weights;
    attr.scale.shape = scale_shape;
    node->operation.attributes = std::move(attr);
  }
  graph->AddConsumer(node->id, input->id);
}

void ConfigSharedBiasFullyConnectedNode(bool bias_shared,
                                        TfLiteIntArray* inputs,
                                        int bias_tensor_index,
                                        ObjectReader* reader,
                                        ::ml_drift::GraphFloat32* graph,
                                        ::ml_drift::Node* node) {
  if (!bias_shared) return;
  ::ml_drift::Value* bias =
      reader->ReadValueByTensorIdx(inputs->data[bias_tensor_index]);
  if (bias->tensor.shape.b != 1 && bias->tensor.shape.c == 1) {
    bias->tensor.shape = ::ml_drift::BHWC(1, 1, 1, bias->tensor.shape.b);
  }
  graph->AddConsumer(node->id, bias->id);
}

bool IsLogicalOpCode(int32_t builtin_code) {
  return builtin_code == kTfLiteBuiltinLogicalAnd ||
         builtin_code == kTfLiteBuiltinLogicalOr ||
         builtin_code == kTfLiteBuiltinLogicalNot ||
         builtin_code == kTfLiteBuiltinBitwiseXor;
}

bool IsCompareOpCode(int32_t builtin_code) {
  return builtin_code == kTfLiteBuiltinGreater ||
         builtin_code == kTfLiteBuiltinGreaterEqual ||
         builtin_code == kTfLiteBuiltinLess ||
         builtin_code == kTfLiteBuiltinLessEqual ||
         builtin_code == kTfLiteBuiltinEqual ||
         builtin_code == kTfLiteBuiltinNotEqual;
}

bool IsBf16SupportedOp(int32_t builtin_code) {
  return builtin_code == kTfLiteBuiltinAbs ||
         builtin_code == kTfLiteBuiltinAdd ||
         builtin_code == kTfLiteBuiltinAtan2 ||
         builtin_code == kTfLiteBuiltinStablehloCbrt ||
         builtin_code == kTfLiteBuiltinCeil ||
         builtin_code == kTfLiteBuiltinStablehloClamp ||
         builtin_code == kTfLiteBuiltinConv2d ||
         builtin_code == kTfLiteBuiltinCos ||
         builtin_code == kTfLiteBuiltinDiv ||
         builtin_code == kTfLiteBuiltinExp ||
         builtin_code == kTfLiteBuiltinFloor ||
         builtin_code == kTfLiteBuiltinGather ||
         builtin_code == kTfLiteBuiltinLog ||
         builtin_code == kTfLiteBuiltinLogistic ||
         builtin_code == kTfLiteBuiltinMaximum ||
         builtin_code == kTfLiteBuiltinMinimum ||
         builtin_code == kTfLiteBuiltinMul ||
         builtin_code == kTfLiteBuiltinNeg ||
         builtin_code == kTfLiteBuiltinPow ||
         builtin_code == kTfLiteBuiltinRound ||
         builtin_code == kTfLiteBuiltinRsqrt ||
         builtin_code == kTfLiteBuiltinSign ||
         builtin_code == kTfLiteBuiltinSin ||
         builtin_code == kTfLiteBuiltinSqrt ||
         builtin_code == kTfLiteBuiltinSub ||
         builtin_code == kTfLiteBuiltinTanh;
}

bool IsIntSupportedOp(int32_t builtin_code) {
  return builtin_code == kTfLiteBuiltinAbs ||
         builtin_code == kTfLiteBuiltinAdd ||
         builtin_code == kTfLiteBuiltinAtan2 ||
         builtin_code == kTfLiteBuiltinCos ||
         builtin_code == kTfLiteBuiltinDiv ||
         builtin_code == kTfLiteBuiltinFloorDiv ||
         builtin_code == kTfLiteBuiltinFloorMod ||
         builtin_code == kTfLiteBuiltinMaximum ||
         builtin_code == kTfLiteBuiltinMinimum ||
         builtin_code == kTfLiteBuiltinMul ||
         builtin_code == kTfLiteBuiltinNeg ||
         builtin_code == kTfLiteBuiltinReduceMax ||
         builtin_code == kTfLiteBuiltinReduceMin ||
         builtin_code == kTfLiteBuiltinReduceProd ||
         builtin_code == kTfLiteBuiltinRightShift ||
         builtin_code == kTfLiteBuiltinSign ||
         builtin_code == kTfLiteBuiltinSin ||
         builtin_code == kTfLiteBuiltinStablehloClamp ||
         builtin_code == kTfLiteBuiltinSub ||
         builtin_code == kTfLiteBuiltinSum ||
         builtin_code == kTfLiteBuiltinStablehloRemainder ||
         builtin_code == kTfLiteBuiltinTile;
}

bool IsIntSupportedCompositeOrCustomOp(
    int32_t builtin_code, TfLiteNode* tflite_node,
    const TfLiteRegistration* registration,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory) {
  if (composite_parser_factory == nullptr) {
    return false;
  }
  if (builtin_code == kTfLiteBuiltinStablehloComposite) {
    const auto* composite_params =
        reinterpret_cast<const TfLiteStablehloCompositeParams*>(
            tflite_node->builtin_data);
    return composite_params != nullptr &&
           composite_parser_factory->SupportsIntegerTypes(
               composite_params->name);
  }
  if (builtin_code == kTfLiteBuiltinCustom && registration != nullptr &&
      registration->custom_name != nullptr) {
    return composite_parser_factory->SupportsIntegerTypes(
        registration->custom_name);
  }
  return false;
}

bool IsBoolSupportedCompositeOp(
    int32_t builtin_code, TfLiteNode* tflite_node,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory) {
  if (builtin_code != kTfLiteBuiltinStablehloComposite ||
      composite_parser_factory == nullptr) {
    return false;
  }
  const auto* composite_params =
      reinterpret_cast<const TfLiteStablehloCompositeParams*>(
          tflite_node->builtin_data);
  return composite_parser_factory->SupportsBoolTypes(composite_params->name);
}

bool SupportAllPrecisionOp(int32_t builtin_code) {
  return builtin_code == kTfLiteBuiltinStablehloBroadcastInDim ||
         builtin_code == kTfLiteBuiltinBitcast ||
         builtin_code == kTfLiteBuiltinCast ||
         builtin_code == kTfLiteBuiltinConcatenation ||
         builtin_code == kTfLiteBuiltinDynamicUpdateSlice ||
         builtin_code == kTfLiteBuiltinGather ||
         builtin_code == kTfLiteBuiltinPack ||
         builtin_code == kTfLiteBuiltinPad ||
         builtin_code == kTfLiteBuiltinPadv2 ||
         builtin_code == kTfLiteBuiltinReshape ||
         builtin_code == kTfLiteBuiltinReverseV2 ||
         builtin_code == kTfLiteBuiltinSelect ||
         builtin_code == kTfLiteBuiltinSelectV2 ||
         builtin_code == kTfLiteBuiltinSlice ||
         builtin_code == kTfLiteBuiltinStridedSlice ||
         builtin_code == kTfLiteBuiltinTranspose;
}

inline ::ml_drift::HW ToHW(int h, int w) {
  return ::ml_drift::HW(std::max(1, h), std::max(1, w));
}

class ArgMaxOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.num_outputs = 1,
                                          .required_runtime_inputs = 1,
                                          .required_const_inputs = 1,
                                          .check_gpu_compatibility = false}));

    ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::INT32>
        dims_tensor;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, &dims_tensor,
                                       ReadTensorFlags::kNoExtraBytes));

    const TfLiteTensor* src_tensor = nullptr;
    TfLiteTensor* dst_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, 0, &src_tensor));
    ABSL_RETURN_IF_ERROR(
        PreGetOutputTensor(context, tflite_node, 0, &dst_tensor));
    ABSL_RETURN_IF_ERROR(
        PreCheckAxisFromIndex(*src_tensor, dims_tensor.data[0]));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckTensorShape(*src_tensor));
    ABSL_RETURN_IF_ERROR(PreCheckTensorShape(*dst_tensor));

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::MAX_INDEX);
    ::ml_drift::MaxIndexAttributes attr;
    const TfLiteTensor* src_tensor = reader->GetInputTensor(0);
    const TfLiteTensor* dst_tensor = reader->GetOutputTensor(0);
    ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::INT32>
        dims_tensor;
    reader->ReadTensor(1, &dims_tensor, ReadTensorFlags::kNoExtraBytes);
    attr.dim = ExtractAxisFromIndex(*src_tensor, dims_tensor.data[0]);
    reader->AddInput(node, 0);
    if (src_tensor->dims->size != dst_tensor->dims->size) {
      // GPU delegates does not support implicit shapes transformations
      // adding explicit Reshape
      ::ml_drift::BHWC arg_max_shape = ExtractTensorShape(src_tensor);
      arg_max_shape.set(attr.dim, 1);
      ::ml_drift::Node* node_reshape = graph->NewNode();
      node_reshape->operation.type =
          ToString(::ml_drift::OperationType::RESHAPE);
      ::ml_drift::ReshapeAttributes reshape_attr;
      reshape_attr.new_shape = ExtractTensorShape(dst_tensor);
      node_reshape->operation.attributes = reshape_attr;
      ::ml_drift::Value* arg_max_result = graph->NewValue();
      arg_max_result->tensor.type = ToDataType(dst_tensor->type);
      arg_max_result->tensor.shape = arg_max_shape;

      graph->SetProducer(node->id, arg_max_result->id);
      graph->AddConsumer(node_reshape->id, arg_max_result->id);
      node->operation.attributes = std::move(attr);
      reader->AddOutputs(node_reshape);
    } else {
      node->operation.attributes = std::move(attr);
      reader->AddOutputs(node);
    }
  }
};

::ml_drift::Value* NewConstNode(::ml_drift::TensorFloat32 t,
                                ::ml_drift::GraphFloat32* graph) {
  ::ml_drift::Node* node = graph->NewNode();
  node->operation.type = ToString(::ml_drift::OperationType::CONSTANT);
  ::ml_drift::Value* value = graph->NewValue();
  graph->SetProducer(node->id, value->id);
  // Keep data inside this tensor.
  value->tensor.ref = t.id;
  value->tensor.type = t.kType;
  value->tensor.shape = t.shape;
  ::ml_drift::ConstTensorAttributes attr;
  attr.tensor = std::move(t);
  node->operation.attributes = std::move(attr);
  return value;
}

class BatchedMatMulOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.num_outputs = 1}));
    const int runtime_inputs =
        GetNumberOfRuntimeInputsForNode(context, tflite_node);
    if (runtime_inputs == 1) {
      // Second input is constant, replace with Convolution2D
      ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
      const TfLiteTensor* second_input = nullptr;
      ABSL_RETURN_IF_ERROR(
          PreGetInputTensor(context, tflite_node, 1, &second_input));
      if (!tflite::IsConstantTensor(second_input)) {
        // first input must be runtime and second is a constant tensor
        return absl::UnavailableError(
            "Not supported batched mat mul case: non-constant tensor");
      }
      if (second_input->dims->size == 2) {
        ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32>
            dummy_weights;
        ABSL_RETURN_IF_ERROR(
            PreCheckReadTensor(context, tflite_node, 1, &dummy_weights));
      } else if (second_input->dims->size == 3 ||
                 second_input->dims->size == 4) {
        ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>
            dummy_weights;
        ABSL_RETURN_IF_ERROR(
            PreCheckReadTensor(context, tflite_node, 1, &dummy_weights));
      } else {
        return absl::UnavailableError(
            "Not supported batched mat mul case: second input has unsupported "
            "rank");
      }
    } else if (runtime_inputs == 2) {
      ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
      ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    } else {
      return absl::UnavailableError(
          "Not supported batched mat mul case: too many inputs");
    }
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    const TfLiteTensor* src1_tensor = reader->GetInputTensor(1);
    if (reader->GetNumberOfRuntimeInputs() == 1 &&
        src1_tensor->dims->size == 2) {
      ::ml_drift::Node* node = graph->NewNode();
      node->operation.type =
          ToString(::ml_drift::OperationType::FULLY_CONNECTED);
      reader->AddInput(node, 0);
      reader->AddOutputs(node);

      ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32> weights;
      reader->ReadTensor(1, &weights, ReadTensorFlags::kNoExtraBytes);
      ::ml_drift::FullyConnectedAttributes attr;
      attr.weights.data.resize(weights.shape.w * weights.shape.h +
                               XNN_EXTRA_BYTES / sizeof(float));
      for (int i = 0; i < weights.shape.w; ++i) {
        for (int j = 0; j < weights.shape.h; ++j) {
          attr.weights.data[i * weights.shape.h + j] =
              weights.data[j * weights.shape.w + i];
        }
      }
      attr.weights.id = weights.id;
      attr.weights.shape =
          ::ml_drift::OHWI(weights.shape.w, 1, 1, weights.shape.h);
      node->operation.attributes = std::move(attr);
      return;
    }

    const TfLiteTensor* src0_tensor = reader->GetInputTensor(0);
    const TfLiteTensor* dst0_tensor = reader->GetOutputTensor(0);
    ::ml_drift::Value* input0 = reader->ReadValue(0);
    ::ml_drift::Value* input1 = nullptr;
    if (reader->GetNumberOfRuntimeInputs() == 2) {
      input1 = reader->ReadValue(1);
    } else {
      ABSL_CHECK_EQ(reader->GetNumberOfRuntimeInputs(), 1);
      ::ml_drift::TensorFloat32 tensor;
      reader->ReadTensor(1, &tensor, ReadTensorFlags::kNoExtraBytes);
      input1 = NewConstNode(std::move(tensor), graph);
    }
    ::ml_drift::Value* output =
        reader->ReadValueByTensorIdx(tflite_node->outputs->data[0]);
    // MLDrift supports batched matmul with single batch.
    // Model can have model batch in addition to matmul batch. In this case we
    // reshape inputs/outputs to have single batch in MLDrift. For example
    // 2x4x128x32 (2 is model batch, 4 is matmul batch) will be reshaped to
    // 1x8x128x32.
    // If shape is 2d(MxN) MLDrift treats it as Mx1x1xN. In this case we need
    // to make reshape to get 1x1xMxN.
    ::ml_drift::Value* left_value = input0;
    ::ml_drift::Value* right_value = input1;
    ::ml_drift::Value* result_value = output;
    if (input0->tensor.shape.b != 1) {
      ::ml_drift::Node* reshape_left = graph->NewNode();
      reshape_left->operation.type =
          ToString(::ml_drift::OperationType::RESHAPE);
      ::ml_drift::ReshapeAttributes reshape_attr;
      if (src0_tensor->dims->size == 2) {
        // reshape left, Mx1x1xK -> 1x1xMxK
        reshape_attr.new_shape = ::ml_drift::BHWC(1, 1, input0->tensor.shape.b,
                                                  input0->tensor.shape.c);
      } else {
        // reshape left, B0xB1xMxK -> 1xBxMxK, B = B0 * B1
        reshape_attr.new_shape =
            ::ml_drift::BHWC(1, input0->tensor.shape.b * input0->tensor.shape.h,
                             input0->tensor.shape.w, input0->tensor.shape.c);
      }
      reshape_left->operation.attributes = reshape_attr;
      graph->AddConsumer(reshape_left->id, input0->id);
      left_value = graph->NewValue();
      left_value->tensor.type = input0->tensor.type;
      left_value->tensor.shape = reshape_attr.new_shape;
      graph->SetProducer(reshape_left->id, left_value->id);
    }
    if (input1->tensor.shape.b != 1) {
      ::ml_drift::Node* reshape_right = graph->NewNode();
      reshape_right->operation.type =
          ToString(::ml_drift::OperationType::RESHAPE);
      ::ml_drift::ReshapeAttributes reshape_attr;
      if (src1_tensor->dims->size == 2) {
        // reshape right, Kx1x1xN -> 1x1xKxN
        reshape_attr.new_shape = ::ml_drift::BHWC(1, 1, input1->tensor.shape.b,
                                                  input1->tensor.shape.c);
      } else {
        // reshape right, B0xB1xKxN -> 1xBxKxN, B = B0 * B1
        reshape_attr.new_shape =
            ::ml_drift::BHWC(1, input1->tensor.shape.b * input1->tensor.shape.h,
                             input1->tensor.shape.w, input1->tensor.shape.c);
      }
      reshape_right->operation.attributes = std::move(reshape_attr);
      graph->AddConsumer(reshape_right->id, input1->id);
      right_value = graph->NewValue();
      right_value->tensor.type = input1->tensor.type;
      right_value->tensor.shape = reshape_attr.new_shape;
      graph->SetProducer(reshape_right->id, right_value->id);
    }
    if (output->tensor.shape.b != 1) {
      result_value = graph->NewValue();
      result_value->tensor.type = output->tensor.type;
      if (dst0_tensor->dims->size == 2) {
        // reshape output, Mx1x1xN -> 1x1xMxN
        result_value->tensor.shape = ::ml_drift::BHWC(
            1, 1, output->tensor.shape.b, output->tensor.shape.c);
      } else {
        // reshape output, B0xB1xMxN -> 1xBxMxN, B = B0 * B1
        result_value->tensor.shape =
            ::ml_drift::BHWC(1, output->tensor.shape.b * output->tensor.shape.h,
                             output->tensor.shape.w, output->tensor.shape.c);
      }
    }
    ::ml_drift::Node* bmm = graph->NewNode();
    auto tflite_options = reinterpret_cast<const TfLiteBatchMatMulParams*>(
        tflite_node->builtin_data);
    ::ml_drift::BatchedMatMulAttributes attr;
    attr.transpose_left = tflite_options->adj_x;
    attr.transpose_right = tflite_options->adj_y;
    bmm->operation.attributes = std::move(attr);
    bmm->operation.type = ToString(::ml_drift::OperationType::BATCHED_MATMUL);
    graph->AddConsumer(bmm->id, left_value->id);
    graph->AddConsumer(bmm->id, right_value->id);
    graph->SetProducer(bmm->id, result_value->id);
    if (output->tensor.shape.b != 1) {
      // reshape result to original shape
      ::ml_drift::Node* reshape_result = graph->NewNode();
      reshape_result->operation.type =
          ToString(::ml_drift::OperationType::RESHAPE);
      ::ml_drift::ReshapeAttributes reshape_attr;
      reshape_attr.new_shape = output->tensor.shape;
      reshape_result->operation.attributes = std::move(reshape_attr);
      graph->AddConsumer(reshape_result->id, result_value->id);
      graph->SetProducer(reshape_result->id, output->id);
    }
  }
};

class BitcastOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(
        ValidateSupport(context, tflite_node, registration, {}));
    const TfLiteTensor* src_tensor = nullptr;
    TfLiteTensor* dst_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, 0, &src_tensor));
    ABSL_RETURN_IF_ERROR(
        PreGetOutputTensor(context, tflite_node, 0, &dst_tensor));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    if (src_tensor->dims->size != dst_tensor->dims->size) {
      ABSL_RETURN_IF_ERROR(PreCheckTensorShape(*dst_tensor));
    }

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    const TfLiteTensor* src_tensor = reader->GetInputTensor(0);
    const TfLiteTensor* dst_tensor = reader->GetOutputTensor(0);
    if (src_tensor->dims->size >
        dst_tensor->dims->size) {  // decrease precision size
      // bitcast -> reshape.
      // Ex: for si8 -> si32 we might have shapes such as
      //     (2, 2, 4) -> (bitcast) -> (2, 2, 1) -> (reshape) -> (2, 2)
      ::ml_drift::Node* bitcast_node = graph->NewNode();
      bitcast_node->operation.type =
          ToString(::ml_drift::OperationType::BITCAST);
      reader->AddInput(bitcast_node, 0);
      const auto bitcast_input = graph->FindInputs(bitcast_node->id)[0];
      ::ml_drift::BHWC interim_shape = bitcast_input->tensor.shape;
      interim_shape.c = 1;

      // Add reshape if needed (no reshape for 2d)
      const ::ml_drift::BHWC output_shape = ExtractTensorShape(dst_tensor);
      if (output_shape != interim_shape) {
        // Create interim value
        ::ml_drift::Value* interim_val = graph->NewValue();
        interim_val->tensor.type = ToDataType(dst_tensor->type);
        interim_val->tensor.shape = interim_shape;
        graph->SetProducer(bitcast_node->id, interim_val->id);

        // Add reshape
        ::ml_drift::Node* reshape_node = graph->NewNode();
        reshape_node->operation.type =
            ToString(::ml_drift::OperationType::RESHAPE);
        ::ml_drift::ReshapeAttributes reshape_attr;
        reshape_attr.new_shape = output_shape;
        reshape_node->operation.attributes = std::move(reshape_attr);
        graph->AddConsumer(reshape_node->id, interim_val->id);
        reader->AddOutputs(reshape_node);
      } else {
        reader->AddOutputs(bitcast_node);
      }
    } else if (src_tensor->dims->size <
               dst_tensor->dims->size) {  // increase precision size
      // reshape -> bitcast
      // Ex: for si32 -> si8 we might have shapes such as
      //     (2, 2) -> (reshape) -> (2, 2, 1) -> (bitcast) -> (2, 2, 4)
      ::ml_drift::Node* reshape_node = graph->NewNode();
      reshape_node->operation.type =
          ToString(::ml_drift::OperationType::RESHAPE);
      reader->AddInput(reshape_node, 0);
      ::ml_drift::Value* interim_val = graph->NewValue();
      interim_val->tensor.type = ToDataType(src_tensor->type);
      ::ml_drift::BHWC interim_shape = ExtractTensorShape(dst_tensor);
      interim_shape.c = 1;
      interim_val->tensor.shape = interim_shape;
      ::ml_drift::ReshapeAttributes reshape_attr;
      reshape_attr.new_shape = interim_shape;
      reshape_node->operation.attributes = std::move(reshape_attr);
      graph->SetProducer(reshape_node->id, interim_val->id);

      // Add bitcast
      ::ml_drift::Node* bitcast_node = graph->NewNode();
      bitcast_node->operation.type =
          ToString(::ml_drift::OperationType::BITCAST);
      graph->AddConsumer(bitcast_node->id, interim_val->id);
      reader->AddOutputs(bitcast_node);
    } else {  // maintain precision size
      // bitcast
      // Ex: for f32 -> si32 we might have shapes such as
      //     (2, 2, 4) -> (bitcast) -> (2, 2, 4)
      ::ml_drift::Node* bitcast_node = graph->NewNode();
      bitcast_node->operation.type =
          ToString(::ml_drift::OperationType::BITCAST);
      reader->AddInput(bitcast_node, 0);
      reader->AddOutputs(bitcast_node);
    }
  }
};

class BroadcastInDimOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(
        ValidateSupport(context, tflite_node, registration, {}));
    const TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, 0, &input_tensor));
    ABSL_RETURN_IF_ERROR(
        PreGetOutputTensor(context, tflite_node, 0, &output_tensor));
    if (!CanGetIndicesMap(input_tensor->dims->size)) {
      return absl::InvalidArgumentError("Only supports 1D-4D input tensor.");
    }
    if (!CanGetIndicesMap(output_tensor->dims->size)) {
      return absl::InvalidArgumentError("Only supports 1D-4D output tensor.");
    }

    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32> indices;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, &indices,
                                       ReadTensorFlags::kNoExtraBytes));
    ABSL_RETURN_IF_ERROR(CheckIndices(input_tensor, output_tensor, indices));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));

    ABSL_RETURN_IF_ERROR(PreCheckTensorShape(*output_tensor));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    // Ensure valid inputs
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32> indices;
    reader->ReadTensor(1, &indices, ReadTensorFlags::kNoExtraBytes);
    TfLiteTensor* input_tensor = reader->GetInputTensor(0);
    TfLiteTensor* output_tensor = reader->GetOutputTensor(0);

    // Add transpose node
    ::ml_drift::Node* transpose_node = graph->NewNode();
    transpose_node->operation.type =
        ToString(::ml_drift::OperationType::TRANSPOSE);
    reader->AddInput(transpose_node, 0);
    const auto input = graph->FindInputs(transpose_node->id)[0];
    ::ml_drift::Value* interim_val = graph->NewValue();
    interim_val->tensor.type = input->tensor.type;

    // Determine transpose permutation
    const auto input_map = GetIndicesMap(input_tensor->dims->size);
    const auto output_map = GetIndicesMap(output_tensor->dims->size);
    std::vector<int> indices_mapped(4);
    for (int d = 0; d < 4; ++d) {
      indices_mapped[input_map.at(d)] = output_map.at(d);
    }
    ::ml_drift::BHWC perm(indices_mapped[0], indices_mapped[1],
                          indices_mapped[2], indices_mapped[3]);
    ::ml_drift::TransposeAttributes attr;
    attr.perm = perm;
    transpose_node->operation.attributes = std::move(attr);

    // Get intermediate shape
    const ::ml_drift::BHWC input_shape = input->tensor.shape;
    const std::vector<int> input_shape_vec = {input_shape.b, input_shape.h,
                                              input_shape.w, input_shape.c};
    std::vector<int> interim_shape_vec(4);
    for (int d = 0; d < 4; ++d) {
      interim_shape_vec[d] = input_shape_vec[indices_mapped[d]];
    }
    const ::ml_drift::BHWC interim_shape =
        ::ml_drift::BHWC(interim_shape_vec[0], interim_shape_vec[1],
                         interim_shape_vec[2], interim_shape_vec[3]);
    interim_val->tensor.shape = interim_shape;

    // Add tile node if necessary
    const ::ml_drift::BHWC output_shape = ExtractTensorShape(output_tensor);
    if (interim_shape != output_shape) {
      graph->SetProducer(transpose_node->id, interim_val->id);
      ::ml_drift::Node* tile_node = graph->NewNode();
      tile_node->operation.type = ToString(::ml_drift::OperationType::TILE);
      graph->AddConsumer(tile_node->id, interim_val->id);
      reader->AddOutputs(tile_node);
    } else {
      reader->AddOutputs(transpose_node);
    }
  }

 private:
  // Modeled after ExtractTensorShape in model_builder_helper
  static inline bool CanGetIndicesMap(int dims) { return dims > 0 && dims < 5; }
  static std::vector<int> GetIndicesMap(int dims) {
    if (dims == 2) return {0, 3, 2, 1};
    if (dims == 3) return {0, 2, 3, 1};
    return {0, 1, 2, 3};
  }

  absl::Status CheckIndices(
      const TfLiteTensor* input_tensor, TfLiteTensor* output_tensor,
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>&
          indices) const {
    for (int d = 0; d < indices.data.size(); ++d) {
      if (indices.data[d] < 0 || indices.data[d] >= output_tensor->dims->size) {
        return absl::InvalidArgumentError(
            "Require 0 <= index val < rank(result)");
      }
      if (input_tensor->dims->data[d] != 1 &&
          input_tensor->dims->data[d] !=
              output_tensor->dims->data[indices.data[d]]) {
        return absl::InvalidArgumentError(
            "Requires dim(input, d) = 1 || dim(input, d) = dim(output, "
            "index[d])");
      }
    }
    const std::set<int> indices_set(indices.data.begin(), indices.data.end());
    if (indices.data.size() != indices_set.size()) {
      return absl::InvalidArgumentError("Index values must be unique");
    }
    return absl::OkStatus();
  }
};

class CastOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.min_inputs = 1,
                                          .max_inputs = 1,
                                          .num_outputs = 1,
                                          .required_runtime_inputs = 1,
                                          .required_const_inputs = 0,
                                          .check_gpu_compatibility = false}));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    // TODO(b/189917229): add proper op support checking.
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::CAST);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);
  }
};

class CbrtOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(
        ValidateSupport(context, tflite_node, registration, {}));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    // cbrt(x) = pow(x, 1/3)
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::POW);
    reader->AddInput(node, 0);
    ::ml_drift::ElementwiseAttributes attr;
    attr.param = 1.0f / 3.0f;
    node->operation.attributes = std::move(attr);
    reader->AddOutputs(node);
  }
};

class ClampOperationsParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(
        ValidateSupport(context, tflite_node, registration, {}));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 2));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    // Expect clamp(min, value, max) where min and max are runtime
    // tensors. They either are the same shape as value, or are scalars.
    // clamp(min, value, max) = min(max(value, min), max))
    ::ml_drift::Node* max_node = graph->NewNode();
    max_node->operation.type = ToString(::ml_drift::OperationType::MAXIMUM);
    ::ml_drift::Node* min_node = graph->NewNode();
    min_node->operation.type = ToString(::ml_drift::OperationType::MINIMUM);

    reader->AddInput(max_node, 0);
    reader->AddInput(max_node, 1);
    auto input = graph->FindInputs(max_node->id)[1];

    ::ml_drift::Value* interim = graph->NewValue();
    interim->tensor.type = input->tensor.type;
    interim->tensor.shape = input->tensor.shape;

    reader->AddInput(min_node, 2);

    graph->SetProducer(max_node->id, interim->id);
    graph->AddConsumer(min_node->id, interim->id);

    reader->AddOutputs(min_node);
  }
};

class ConcatenationOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 6}));

    // tensor availability checking
    std::vector<::ml_drift::BHWC> input_shapes;
    for (uint32_t idx = 0; idx < tflite_node->inputs->size; ++idx) {
      const TfLiteTensor* input = nullptr;
      ABSL_RETURN_IF_ERROR(
          PreGetInputTensor(context, tflite_node, idx, &input));
      ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, idx));
      input_shapes.push_back(ExtractTensorShape(input));
    }

    if (GetAxis(input_shapes) == ::ml_drift::Axis::UNKNOWN) {
      return absl::InvalidArgumentError(
          "Couldn't find an axis to concatenate by.");
    }

    const auto* params = static_cast<const TfLiteConcatenationParams*>(
        tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteConcatenationParams.");
    }
    ABSL_RETURN_IF_ERROR(
        PreCheckMaybeFuseActivation(tflite_node, params->activation));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::ConcatAttributes attr;
    // Read inputs first to make sure const node is added to a graph before
    // concat node to ensure topological order.
    std::vector<const ::ml_drift::Value*> inputs;
    for (uint32_t idx = 0; idx < tflite_node->inputs->size; ++idx) {
      ::ml_drift::Value* input = reader->IsConstantTensor(idx)
                                     ? reader->AddConstInput(idx, /*layout=*/{})
                                     : reader->ReadValue(idx);
      inputs.push_back(input);
    }

    for (int i = 0; i < inputs.size(); ++i) {
      for (int j = 0; j < i; ++j) {
        if (inputs[i] == inputs[j]) {
          ::ml_drift::Node* node_copy = graph->NewNode();
          node_copy->operation.type = ToString(::ml_drift::OperationType::COPY);
          graph->AddConsumer(node_copy->id, inputs[j]->id);
          ::ml_drift::Value* copy_value = graph->NewValue();
          copy_value->tensor.type = inputs[j]->tensor.type;
          copy_value->tensor.shape = inputs[j]->tensor.shape;
          graph->SetProducer(node_copy->id, copy_value->id);
          inputs[i] = copy_value;
          break;
        }
      }
    }

    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::CONCAT);
    reader->AddOutputs(node);
    for (int i = 0; i < inputs.size(); ++i) {
      graph->AddConsumer(node->id, inputs[i]->id);
    }

    std::vector<::ml_drift::BHWC> input_shapes;
    for (auto input : graph->FindInputs(node->id)) {
      input_shapes.push_back(input->tensor.shape);
    }
    attr.axis = GetAxis(input_shapes);

    // Guess axis.
    ::ml_drift::BHWC output_shape =
        graph->FindOutputs(node->id)[0]->tensor.shape;
    for (auto input : graph->FindInputs(node->id)) {
      if (input->tensor.shape.h != output_shape.h) {
        attr.axis = ::ml_drift::Axis::HEIGHT;
        break;
      }
      if (input->tensor.shape.w != output_shape.w) {
        attr.axis = ::ml_drift::Axis::WIDTH;
        break;
      }
      if (input->tensor.shape.c != output_shape.c) {
        attr.axis = ::ml_drift::Axis::CHANNELS;
        break;
      }
    }
    const auto* params = static_cast<const TfLiteConcatenationParams*>(
        tflite_node->builtin_data);
    HandleFusedActivation(params->activation, graph, node);
    node->operation.attributes = attr;
  }

 private:
  static ::ml_drift::Axis GetAxis(
      const std::vector<::ml_drift::BHWC>& input_shapes) {
    ::ml_drift::Axis axis = ::ml_drift::Axis::BATCH;
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].h != input_shapes[i].h &&
          input_shapes[0].w != input_shapes[i].w &&
          input_shapes[0].c != input_shapes[i].c) {
        axis = ::ml_drift::Axis::HEIGHT;
        break;
      }
    }
    if (axis == ::ml_drift::Axis::BATCH) return axis;
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].b != input_shapes[i].b &&
          input_shapes[0].w != input_shapes[i].w &&
          input_shapes[0].c != input_shapes[i].c) {
        axis = ::ml_drift::Axis::WIDTH;
        break;
      }
    }
    if (axis == ::ml_drift::Axis::HEIGHT) return axis;
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].b != input_shapes[i].b &&
          input_shapes[0].h != input_shapes[i].h &&
          input_shapes[0].c != input_shapes[i].c) {
        axis = ::ml_drift::Axis::CHANNELS;
        break;
      }
    }
    if (axis == ::ml_drift::Axis::WIDTH) return axis;
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].b != input_shapes[i].b &&
          input_shapes[0].w != input_shapes[i].w &&
          input_shapes[0].h != input_shapes[i].h) {
        return ::ml_drift::Axis::UNKNOWN;
      }
    }
    return axis;
  }
};

class Conv2DOperationParser : public TFLiteOperationParser {
  enum { kInputSrcId, kInputWeightsId, kInputBiasId };
  enum { kOutputId };

 public:
  explicit Conv2DOperationParser(const ModelBuilderOptions& options)
      : options_(options) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 6}));

    // TODO: b/372428865 - Get tensor sharing information in IsSupported.
    // To make checkings more completed, sharing information is required.
    const TfLiteTensor* input_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, kInputSrcId, &input_tensor));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, kInputSrcId));
    if (tflite::IsConstantTensor(input_tensor)) {
      return absl::InvalidArgumentError("input must be a runtime tensor");
    }

    const TfLiteTensor* weights_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node,
                                           kInputWeightsId, &weights_tensor));

    const auto* params =
        static_cast<const TfLiteConvParams*>(tflite_node->builtin_data);
    if (!params) return absl::InvalidArgumentError("Missing TfLiteConvParams.");

    // Check for quantized weights sharing when they are convertible to fully
    // connected layer.
    if (IsConvertibleToFC(weights_tensor)) {
      int weights_tensor_idx = 0;
      ABSL_RETURN_IF_ERROR(GetTensorId(context, tflite_node, kInputWeightsId,
                                       &weights_tensor_idx));
      ABSL_RETURN_IF_ERROR(
          PreCheckReadQuantizedValueByTensorIdx(context, weights_tensor_idx));
    }

    if (tflite::GetVariableInput(const_cast<TfLiteContext*>(context),
                                 tflite_node, kInputBiasId)) {
      ABSL_RETURN_IF_ERROR(
          PreCheckReadValue(context, tflite_node, kInputBiasId));
    }

    // TODO: who/impjdi - Check ReadAttributes.
    ABSL_RETURN_IF_ERROR(
        PreCheckMaybeFuseActivation(tflite_node, params->activation));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    reader->AllowSharingInput(kInputWeightsId);
    reader->AllowSharingInput(kInputBiasId);
    if (reader->SharingEnabled() &&
        reader->GetSharingInfoByNodeInputIndex(kInputWeightsId).IsShared()) {
      return ParseWithSharedWeights(tflite_node, registration, graph, reader);
    }
    const TfLiteTensor* weights_tensor =
        reader->GetInputTensor(kInputWeightsId);
    const auto* params =
        static_cast<const TfLiteConvParams*>(tflite_node->builtin_data);
    ::ml_drift::Convolution2DAttributes attr =
        ReadAttributes(tflite_node, params, reader);

    const int runtime_inputs = reader->GetNumberOfRuntimeInputs();
    if (runtime_inputs == 2) {
      // Weights are runtime input.
      ::ml_drift::Node* node = graph->NewNode();
      node->operation.type =
          ToString(::ml_drift::OperationType::CONVOLUTION_2D);
      node->operation.attributes = std::move(attr);
      reader->AddInput(node, 0);
      reader->AddInput(node, 1);
      reader->AddOutputs(node);
      HandleFusedActivation(params->activation, graph, node);
      return;
    }

    // Weights are constant.
    if (IsConvertibleToFC(weights_tensor) &&
        (weights_tensor->type == kTfLiteInt8 ||
         (!options_.enable_raw_weights_propagation &&
          weights_tensor->type == kTfLiteInt4))) {
      ::ml_drift::Node* node = graph->NewNode();
      reader->AddInput(node, 0);
      node->operation.attributes = GetFullyConnectedInt8Attributes(
          kInputWeightsId, kInputBiasId, reader,
          /*copy_weights=*/!options_.enable_raw_weights_propagation);
      // TODO: b/378522761 - add support for int4 quantized weights.
      node->operation.type =
          ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT8);
      reader->AddOutputs(node);
      HandleFusedActivation(params->activation, graph, node);
      return;
    }

    const ::ml_drift::BHWC src_shape =
        ExtractTensorShape(reader->GetInputTensor(0));
    const ::ml_drift::BHWC dst_shape =
        ExtractTensorShape(reader->GetOutputTensor(0));
    const auto& conv_weights_shape = std::visit(
        [](const auto& weights) { return weights.shape; }, attr.weights);
    const int src_group_size = conv_weights_shape.i;
    if (conv_weights_shape.i == 1 && src_shape.c == dst_shape.c) {
      // when weights shape input channels = 1 =>
      // groups count = src_shape channels =>
      // when src channels == dst channels && weights input channels == 1 =>
      // CONVOLUTION_2D equivalent to DEPTHWISE_CONVOLUTION
      const auto& conv_weights = GetFloatWeights(attr);
      ::ml_drift::DepthwiseConvolution2DAttributes dw_attr;
      auto& dw_weights = dw_attr.weights.emplace<::ml_drift::Tensor<
          ::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();
      dw_weights.id = conv_weights.id;
      dw_weights.shape =
          ::ml_drift::OHWI(conv_weights.shape.i, conv_weights.shape.h,
                           conv_weights.shape.w, conv_weights.shape.o);
      dw_weights.data.resize(dw_weights.shape.DimensionsProduct() +
                             XNN_EXTRA_BYTES / sizeof(float));
      for (int o = 0; o < dw_weights.shape.o; ++o) {
        for (int h = 0; h < dw_weights.shape.h; ++h) {
          for (int w = 0; w < dw_weights.shape.w; ++w) {
            for (int i = 0; i < dw_weights.shape.i; ++i) {
              dw_weights.data[dw_weights.shape.LinearIndex({o, h, w, i})] =
                  conv_weights
                      .data[conv_weights.shape.LinearIndex({i, h, w, o})];
            }
          }
        }
      }
      dw_attr.bias = attr.bias;
      dw_attr.strides = attr.strides;
      dw_attr.dilations = attr.dilations;
      dw_attr.padding = attr.padding;
      ::ml_drift::Node* node = graph->NewNode();
      node->operation.type =
          ToString(::ml_drift::OperationType::DEPTHWISE_CONVOLUTION);
      node->operation.attributes = std::move(dw_attr);
      reader->AddInput(node, 0);
      reader->AddOutputs(node);
      HandleFusedActivation(params->activation, graph, node);
      return;
    }
    const int dst_group_size = conv_weights_shape.o / attr.groups;
    const bool supported_grouped_conv =
        src_group_size % 4 == 0 && dst_group_size % 4 == 0;
    if (attr.groups != 1 && !supported_grouped_conv) {
      // Not supported case, replace with usual convolutions:
      ResolveGroupedConvolution(attr, params, reader, graph);
      return;
    }
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::CONVOLUTION_2D);
    node->operation.attributes = std::move(attr);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);
    HandleFusedActivation(params->activation, graph, node);
  }

 private:
  bool IsConvertibleToFC(const TfLiteTensor* weights_tensor) {
    return weights_tensor->dims->size == 4 &&
           weights_tensor->dims->data[1] == 1 &&
           weights_tensor->dims->data[2] == 1 &&
           weights_tensor->quantization.type ==
               TfLiteQuantizationType::kTfLiteAffineQuantization &&
           (weights_tensor->type == kTfLiteInt4 ||
            weights_tensor->type == kTfLiteInt8);
  }

  void ParseWithSharedWeights(const TfLiteNode* tflite_node,
                              const TfLiteRegistration* registration,
                              ::ml_drift::GraphFloat32* graph,
                              ObjectReader* reader) {
    const TfLiteTensor* weights_tensor =
        reader->GetInputTensor(kInputWeightsId);
    const auto* params =
        static_cast<const TfLiteConvParams*>(tflite_node->builtin_data);

    const ObjectReader::ConstantInputSharingInfo weights_share =
        reader->GetSharingInfoByNodeInputIndex(kInputWeightsId);
    const ObjectReader::ConstantInputSharingInfo bias_share =
        reader->GetSharingInfoByNodeInputIndex(kInputBiasId);
    const bool shared_bias = bias_share.IsShared();

    // Reduce convolution operation to a fully connected node.
    if (IsConvertibleToFC(weights_tensor)) {
      ::ml_drift::Node* node = graph->NewNode();
      reader->AddInput(node, 0);
      ConfigSharedWeightFullyConnectedNode(
          tflite_node->inputs->data[kInputWeightsId], weights_tensor, reader,
          graph, node);
      ConfigSharedBiasFullyConnectedNode(shared_bias, tflite_node->inputs,
                                         kInputBiasId, reader, graph, node);

      const auto node_inputs = graph->FindInputs(node->id);
      reader->SetSharedTensor(node_inputs[1]->id, weights_share.PreferredId(),
                              tflite_node->inputs->data[kInputWeightsId],
                              /*dequant_forced=*/false,
                              /*layout=*/std::nullopt);
      if (shared_bias) {
        reader->SetSharedTensor(node_inputs[2]->id, bias_share.PreferredId(),
                                tflite_node->inputs->data[kInputBiasId],
                                /*dequant_forced=*/false,
                                ::ml_drift::Layout::LINEAR);
      }
      const TfLiteTensor* src_tensor = reader->GetInputTensor(kInputSrcId);
      if (IsFullyConnectedOutputReshapeNeeded(src_tensor, weights_tensor,
                                              reader)) {
        ReshapeFullyConnectedOutput(src_tensor, weights_tensor, graph, reader,
                                    node);
      } else {
        reader->AddOutputs(node);
      }
      HandleFusedActivation(params->activation, graph, node);
      return;
    }

    // Build a convolution node.
    ::ml_drift::Node* node = graph->NewNode();
    ::ml_drift::Convolution2DAttributes attr =
        ReadAttributes(tflite_node, params, reader);
    node->operation.type = ToString(::ml_drift::OperationType::CONVOLUTION_2D);
    node->operation.attributes = std::move(attr);
    reader->AddInput(node, 0);
    // If the weights are shared, they will be passed as a runtime input.
    reader->AddInput(node, 1);
    if (shared_bias) {
      ::ml_drift::Value* bias_input =
          reader->ReadValueByTensorIdx(tflite_node->inputs->data[kInputBiasId]);

      // Runtime bias value needs to have all its values only in channels
      // dimension, all other dimensions must be equal to 1.
      // ExtractTensorShape() puts the dim[0] to batch if original tensor's
      // dims->size == 1. So we need to swap batch and channels if this
      // occurs.
      if (bias_input->tensor.shape.b != 1 && bias_input->tensor.shape.c == 1) {
        bias_input->tensor.shape =
            ::ml_drift::BHWC(1, 1, 1, bias_input->tensor.shape.b);
      }

      graph->AddConsumer(node->id, bias_input->id);
    }
    // Convolution2D does not support reading quantized weights, so we
    // force dequantization before sharing.
    bool dequant_forced = weights_tensor->quantization.type ==
                          TfLiteQuantizationType::kTfLiteAffineQuantization;
    const auto node_inputs = graph->FindInputs(node->id);
    reader->SetSharedTensor(node_inputs[1]->id, weights_share.PreferredId(),
                            tflite_node->inputs->data[kInputWeightsId],
                            dequant_forced,
                            /*layout=*/std::nullopt);
    if (shared_bias) {
      reader->SetSharedTensor(node_inputs[2]->id, bias_share.PreferredId(),
                              tflite_node->inputs->data[kInputBiasId],
                              /*dequant_forced=*/false,
                              ::ml_drift::Layout::LINEAR);
    }
    reader->AddOutputs(node);
    HandleFusedActivation(params->activation, graph, node);
  }

  ::ml_drift::Convolution2DAttributes ReadAttributes(
      const TfLiteNode* tflite_node, const TfLiteConvParams* tf_options,
      ObjectReader* reader) {
    ::ml_drift::Convolution2DAttributes attr;
    const TfLiteTensor* src_tensor = reader->GetInputTensor(kInputSrcId);
    const ::ml_drift::BHWC src_shape = ExtractTensorShape(src_tensor);
    const ObjectReader::ConstantInputSharingInfo weights_share =
        reader->GetSharingInfoByNodeInputIndex(kInputWeightsId);
    const int runtime_inputs = reader->GetNumberOfRuntimeInputs();
    if (runtime_inputs == 2 || weights_share.IsShared()) {
      auto& weights = attr.weights.emplace<::ml_drift::Tensor<
          ::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();
      const TfLiteTensor* weights_tensor =
          reader->GetInputTensor(kInputWeightsId);
      const ::ml_drift::BHWC weights_shape = ExtractTensorShape(weights_tensor);
      weights.shape = ::ml_drift::OHWI(weights_shape.b, weights_shape.h,
                                       weights_shape.w, weights_shape.c);
      attr.groups = 1;
    } else {
      const TfLiteTensor* tflite_tensor =
          reader->GetInputTensor(kInputWeightsId);
      if (tflite_tensor->type == kTfLiteInt4) {
        auto& weights = attr.weights.emplace<
            ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4>>();
        reader->ReadTensor(
            kInputWeightsId, &weights, ReadTensorFlags::kExtraBytes,
            options_.enable_spanned_weights, &attr.scale, &attr.zero_point);
      } else if (tflite_tensor->type == kTfLiteInt8) {
        auto& weights = attr.weights.emplace<
            ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>>();
        reader->ReadTensor(
            kInputWeightsId, &weights, ReadTensorFlags::kExtraBytes,
            options_.enable_spanned_weights, &attr.scale, &attr.zero_point);
      } else {
        auto& weights = attr.weights.emplace<::ml_drift::Tensor<
            ::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();
        reader->ReadTensor(kInputWeightsId, &weights,
                           ReadTensorFlags::kExtraBytes,
                           options_.enable_spanned_weights, /*scale=*/nullptr,
                           /*zero_point=*/nullptr);
      }
      attr.groups =
          src_shape.c /
          std::visit([](const auto& w) { return w.shape; }, attr.weights).i;
    }
    if (reader->IsNodeInputTensorPresent(kInputBiasId)) {
      reader->ReadTensor(kInputBiasId, &attr.bias,
                         ReadTensorFlags::kNoExtraBytes,
                         options_.enable_spanned_weights);
    }
    attr.strides = ToHW(tf_options->stride_height, tf_options->stride_width);
    attr.dilations = ::ml_drift::HW(tf_options->dilation_height_factor,
                                    tf_options->dilation_width_factor);
    UpdatePadding(tf_options->padding, src_shape, &attr);
    return attr;
  }

  // Replace single grouped convolution(N = groups count) with this sequence:
  //  split input to N tensors in channels dim
  //  N usual convs
  //  concat N tensors to 1 output in channels dim
  void ResolveGroupedConvolution(
      const ::ml_drift::Convolution2DAttributes& attr,
      const TfLiteConvParams* tf_options, ObjectReader* reader,
      ::ml_drift::GraphFloat32* graph) {
    const TfLiteTensor* src_tensor = reader->GetInputTensor(kInputSrcId);
    const TfLiteTensor* dst_tensor = reader->GetOutputTensor(0);
    const ::ml_drift::BHWC src_shape = ExtractTensorShape(src_tensor);
    const ::ml_drift::BHWC dst_shape = ExtractTensorShape(dst_tensor);

    ::ml_drift::DataType src_type = ::ml_drift::DataType::FLOAT32;
    if (src_tensor->type == kTfLiteFloat16) {
      src_type = ::ml_drift::DataType::FLOAT16;
    }
    ::ml_drift::DataType dst_type = ::ml_drift::DataType::FLOAT32;
    if (dst_tensor->type == kTfLiteFloat16) {
      dst_type = ::ml_drift::DataType::FLOAT16;
    }

    const auto& weights_shape =
        std::visit([](const auto& w) { return w.shape; }, attr.weights);
    const int src_group_size = weights_shape.i;
    const int dst_group_size = weights_shape.o / attr.groups;

    ::ml_drift::Node* split_node = graph->NewNode();
    split_node->operation.type = ToString(::ml_drift::OperationType::SPLIT);
    {
      ::ml_drift::SplitAttributes attr;
      attr.axis = ::ml_drift::Axis::CHANNELS;
      split_node->operation.attributes = std::move(attr);
    }
    reader->AddInput(split_node, 0);

    std::vector<::ml_drift::Node*> conv_nodes(attr.groups);
    std::vector<::ml_drift::Value*> conv_src(attr.groups);
    std::vector<::ml_drift::Value*> conv_dst(attr.groups);
    const auto& weights = GetFloatWeights(attr);
    for (int i = 0; i < attr.groups; ++i) {
      conv_nodes[i] = graph->NewNode();
      conv_src[i] = graph->NewValue();
      conv_dst[i] = graph->NewValue();
      conv_src[i]->tensor.shape = src_shape;
      conv_src[i]->tensor.type = src_type;
      conv_src[i]->tensor.shape.c = src_group_size;
      conv_dst[i]->tensor.shape = dst_shape;
      conv_dst[i]->tensor.type = dst_type;
      conv_dst[i]->tensor.shape.c = dst_group_size;
      ::ml_drift::Convolution2DAttributes conv_attr;
      conv_attr = attr;
      conv_attr.groups = 1;
      auto& conv_weights = conv_attr.weights.emplace<::ml_drift::Tensor<
          ::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();
      conv_weights.id = -1;
      conv_weights.shape.o = dst_group_size;
      conv_weights.data.resize(conv_weights.shape.DimensionsProduct() +
                               XNN_EXTRA_BYTES / sizeof(float));
      for (int out_i = 0; out_i < dst_group_size; ++out_i) {
        for (int in_i = 0; in_i < src_group_size; ++in_i) {
          for (int ky = 0; ky < weights_shape.h; ++ky) {
            for (int kx = 0; kx < weights_shape.w; ++kx) {
              const int src_index = weights_shape.LinearIndex(
                  {{i * dst_group_size + out_i, ky, kx, in_i}});
              const int dst_index =
                  conv_weights.shape.LinearIndex({{out_i, ky, kx, in_i}});
              conv_weights.data[dst_index] = weights.data[src_index];
            }
          }
        }
      }
      conv_attr.bias.shape.v = dst_group_size;
      conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
      for (int out_i = 0; out_i < dst_group_size; ++out_i) {
        if (i * dst_group_size + out_i < attr.bias.data.size()) {
          conv_attr.bias.data[out_i] =
              attr.bias.data[i * dst_group_size + out_i];
        } else {
          conv_attr.bias.data[out_i] = 0.0f;
        }
      }
      conv_nodes[i]->operation.type =
          ToString(::ml_drift::OperationType::CONVOLUTION_2D);
      conv_nodes[i]->operation.attributes = conv_attr;

      graph->SetProducer(split_node->id, conv_src[i]->id);
      graph->AddConsumer(conv_nodes[i]->id, conv_src[i]->id);
      graph->SetProducer(conv_nodes[i]->id, conv_dst[i]->id);
    }

    ::ml_drift::Node* concat_node = graph->NewNode();
    {
      ::ml_drift::ConcatAttributes concat_attr;
      concat_attr.axis = ::ml_drift::Axis::CHANNELS;
      concat_node->operation.type = ToString(::ml_drift::OperationType::CONCAT);
      concat_node->operation.attributes = concat_attr;
    }
    for (int i = 0; i < attr.groups; ++i) {
      graph->AddConsumer(concat_node->id, conv_dst[i]->id);
    }
    reader->AddOutputs(concat_node);
    HandleFusedActivation(tf_options->activation, graph, concat_node);
  }

  ModelBuilderOptions options_;
};

class CumsumOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 1}));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    ::ml_drift::CumsumAttributes attr;
    const TfLiteTensor* input_tensor = reader->GetInputTensor(0);
    const TfLiteTensor* axis_tensor = reader->GetInputTensor(1);
    const TfLiteIntArray* shape = input_tensor->dims;
    const int tflite_axis = tflite::GetTensorData<int32_t>(axis_tensor)[0];
    const ::ml_drift::Axis axes[4] = {
        ::ml_drift::Axis::BATCH, ::ml_drift::Axis::HEIGHT,
        ::ml_drift::Axis::WIDTH, ::ml_drift::Axis::CHANNELS};
    attr.axis = axes[tflite_axis + 4 - shape->size];
    node->operation.type = ToString(::ml_drift::OperationType::CUMSUM);
    node->operation.attributes = std::move(attr);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);
  }
};

class DepthwiseConvolutionOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 6}));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    const int runtime_inputs =
        GetNumberOfRuntimeInputsForNode(context, tflite_node);
    if (runtime_inputs == 2) {
      ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    } else if (runtime_inputs == 1) {
      ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
          dummy_weights;
      ABSL_RETURN_IF_ERROR(
          PreCheckReadTensor(context, tflite_node, 1, &dummy_weights));
    } else {
      return absl::InvalidArgumentError(
          "Unsupported Depthwise Convolution case.");
    }

    const auto* params = static_cast<const TfLiteDepthwiseConvParams*>(
        tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteDepthwiseConvParams.");
    }
    ABSL_RETURN_IF_ERROR(
        PreCheckMaybeFuseActivation(tflite_node, params->activation));

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::DEPTHWISE_CONVOLUTION);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::DepthwiseConvolution2DAttributes attr;
    auto& weights = attr.weights.emplace<
        ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();
    const int runtime_inputs = reader->GetNumberOfRuntimeInputs();
    if (runtime_inputs == 2) {
      reader->AddInput(node, 1);
      auto weights_shape = graph->FindInputs(node->id)[1]->tensor.shape;
      weights.shape = ::ml_drift::OHWI(weights_shape.b, weights_shape.h,
                                       weights_shape.w, weights_shape.c);
    } else {  // runtime_inputs == 1;
      reader->ReadTensor(1, &weights, ReadTensorFlags::kExtraBytes);
    }
    if (reader->IsNodeInputTensorPresent(2)) {
      reader->ReadTensor(2, &attr.bias, ReadTensorFlags::kNoExtraBytes);
    }
    const auto* params = static_cast<const TfLiteDepthwiseConvParams*>(
        tflite_node->builtin_data);
    attr.strides = ToHW(params->stride_height, params->stride_width);
    attr.dilations = ::ml_drift::HW(std::max(1, params->dilation_height_factor),
                                    std::max(1, params->dilation_width_factor));
    UpdatePadding(params->padding, graph->FindInputs(node->id)[0]->tensor.shape,
                  &attr);
    HandleFusedActivation(params->activation, graph, node);
    const int depth_multiplier = params->depth_multiplier;
    if (depth_multiplier != 1) {
      const TfLiteTensor* input = reader->GetInputTensor(0);
      const TfLiteTensor* filter = reader->GetInputTensor(1);
      const TfLiteTensor* output = reader->GetOutputTensor(0);
      TransposeWeights(input, filter, output, depth_multiplier, &attr);
    }
    node->operation.attributes = std::move(attr);
  }

 private:
  // TFLite CPU stores weights as:
  //   [1, kernel_height, kernel_width, input_depth * depth_multiplier]
  // TFLite GPU stores weights as:
  //   [depth_multiplier, kernel_height, kernel_width, input_depth]
  static void TransposeWeights(
      const TfLiteTensor* input, const TfLiteTensor* filter,
      const TfLiteTensor* output, int depth_multiplier,
      ::ml_drift::DepthwiseConvolution2DAttributes* attr) {
    const int input_depth = input->dims->data[3];
    const int filter_height = filter->dims->data[1];
    const int filter_width = filter->dims->data[2];
    const int kernel_spatial_size = filter_height * filter_width;

    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32> weights;
    const auto& src_weights = GetFloatWeights(*attr);
    weights.id = src_weights.id;
    weights.shape = ::ml_drift::OHWI(depth_multiplier, filter_height,
                                     filter_width, input_depth);
    weights.data.resize(weights.shape.DimensionsProduct() +
                        XNN_EXTRA_BYTES / sizeof(float));
    float* dst = weights.data.data();
    const float* src = src_weights.data.data();

    if (kernel_spatial_size == 1) {  // Optimized for 1x1 convolutions
      float* dst_ptr = dst;
      const size_t copy_size = depth_multiplier * sizeof(float);
      for (int i = 0; i < input_depth; ++i) {
        const float* src_ptr = src + i * depth_multiplier;
        std::memcpy(dst_ptr, src_ptr, copy_size);
        dst_ptr += depth_multiplier;
      }
    } else {  // General case for > 1x1 convolutions
      const int src_outer_stride = input_depth * depth_multiplier;
      const int dst_m_stride = kernel_spatial_size * input_depth;
      for (int m = 0; m < depth_multiplier; ++m) {
        for (int s = 0; s < kernel_spatial_size; ++s) {
          const float* current_src = src + s * src_outer_stride + m;
          float* current_dst = dst + m * dst_m_stride + s * input_depth;
          for (int i = 0; i < input_depth; ++i) {
            current_dst[i] = current_src[i * depth_multiplier];
          }
        }
      }
    }
    attr->weights.emplace<
        ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>(
        std::move(weights));
  }
};

class DepthToSpaceOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(
        ValidateSupport(context, tflite_node, registration, {}));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));

    const auto* params =
        static_cast<const TfLiteDepthToSpaceParams*>(tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteDepthToSpaceParams.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::DEPTH_TO_SPACE);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);
    const auto* params =
        static_cast<const TfLiteDepthToSpaceParams*>(tflite_node->builtin_data);
    ::ml_drift::SpaceToDepthAttributes attr;
    attr.block_size = params->block_size;
    node->operation.attributes = attr;
  }
};

class DequantizeOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 3}));
    const int runtime_inputs =
        GetNumberOfRuntimeInputsForNode(context, tflite_node);
    if (runtime_inputs == 0) {
      const TfLiteTensor* input = nullptr;
      ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));
      if (input->dims->size < 2 || input->dims->size > 4) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported dims for tensor: ", input->name,
                         " Got dims:", input->dims->size));
      }
      if (input->dims->size == 2) {
        ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32>
            dummy_tensor_2d;
        ABSL_RETURN_IF_ERROR(
            PreCheckReadTensor(context, tflite_node, 0, &dummy_tensor_2d));
      } else if (input->dims->size == 3) {
        ::ml_drift::Tensor<::ml_drift::HWC, ::ml_drift::DataType::FLOAT32>
            dummy_tensor_3d;
        ABSL_RETURN_IF_ERROR(
            PreCheckReadTensor(context, tflite_node, 0, &dummy_tensor_3d));
      } else if (input->dims->size == 4) {
        ::ml_drift::TensorFloat32 dummy_tensor;
        ABSL_RETURN_IF_ERROR(
            PreCheckReadTensor(context, tflite_node, 0, &dummy_tensor));
      }
      return absl::OkStatus();
    }

    // IsSupported would have to check three things that's related to Populate:
    // CASE 1:
    //   if (tensor.quantization.type != AffineQuant)
    //      return false;
    // This is not needed because this function is called when this is true
    // (see L.42)
    //
    // CASE 2:
    //  if (static_cast<AffineQuant*>(quant.params)->scale->size > 1)
    //    return false;
    // This is the only check have to do here.
    //
    // CASE 3:
    //  if (!int8 && !uint8)
    //    return false;
    // Same with case 1; This function is called when int8 or uint8 (L.38-39)
    const TfLiteTensor* input = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));

    TfLiteAffineQuantization* quantization_data =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    if (!quantization_data) {
      if (runtime_inputs == 1) {
        // DEQUANTIZE op is preceded by DENSIFY op and doesn't have any
        // quantization params. And the DENSIFY is no longer supported.
        return absl::InvalidArgumentError(
            "Dequantize for Densify is not supported");
      }
      return absl::InvalidArgumentError(
          "Encountered Dequantize input with no quant params");
    } else {
      if (quantization_data->scale->size > 1) {
        return absl::InvalidArgumentError(
            "Unsupported quantization scale size");
      }
    }
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    return absl::OkStatus();
  }

  // Read a constant tensor with broadcasting to 4D.
  void ReadConstTensor(ObjectReader* reader, int input_index,
                       ::ml_drift::TensorFloat32* tensor) {
    const TfLiteTensor* input = reader->GetInputTensor(input_index);
    if (input->dims->size == 2) {
      ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32>
          tensor_2d;
      reader->ReadTensor(input_index, &tensor_2d,
                         ReadTensorFlags::kNoExtraBytes);
      tensor->id = tensor_2d.id;
      tensor->shape =
          ::ml_drift::BHWC(1, 1, tensor_2d.shape.h, tensor_2d.shape.w);
      tensor->data = std::move(tensor_2d.data);
    } else if (input->dims->size == 3) {
      ::ml_drift::Tensor<::ml_drift::HWC, ::ml_drift::DataType::FLOAT32>
          tensor_3d;
      reader->ReadTensor(input_index, &tensor_3d,
                         ReadTensorFlags::kNoExtraBytes);
      tensor->id = tensor_3d.id;
      tensor->shape = ::ml_drift::BHWC(1, tensor_3d.shape.h, tensor_3d.shape.w,
                                       tensor_3d.shape.c);
      tensor->data = std::move(tensor_3d.data);
    } else if (input->dims->size == 4) {
      reader->ReadTensor(input_index, tensor, ReadTensorFlags::kNoExtraBytes);
    }
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    // 'Dequantize' is rewritten as QuantizeAndDequantize since we are dealing
    // with floating-point versions of the original tensors.
    const int runtime_inputs = reader->GetNumberOfRuntimeInputs();
    if (runtime_inputs == 0) {
      // constant input, can be dequantized here
      ::ml_drift::TensorFloat32 attr_tensor;
      ReadConstTensor(reader, /*input_index=*/0, &attr_tensor);
      ::ml_drift::ConstTensorAttributes attr;
      attr.tensor = std::move(attr_tensor);
      ::ml_drift::Node* node = graph->NewNode();
      node->operation.attributes = attr;
      node->operation.type = ToString(::ml_drift::OperationType::CONSTANT);
      reader->AddOutputs(node);
      return;
    }
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::QUANTIZE_AND_DEQUANTIZE);
    // Non-constant dequantization.
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    // Quantization attributes should already be present in the input tensor.
    auto input_value = graph->FindInputs(node->id)[0];
    if (input_value->quant_params) {
      ::ml_drift::QuantizeAndDequantizeAttributes attr;
      attr.min = input_value->quant_params.value().min;
      attr.max = input_value->quant_params.value().max;
      attr.scale = input_value->quant_params.value().scale;
      node->operation.attributes = attr;
    } else {
      // This case should be caught by IsSupported.
      ABSL_LOG(FATAL) << "Encountered Dequantize input with no quant params";
    }
  }
};

class DynamicUpdateSliceOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(
        ValidateSupport(context, tflite_node, registration, {}));
    const TfLiteTensor* operand = nullptr;
    const TfLiteTensor* update = nullptr;
    const TfLiteTensor* start_indices = nullptr;
    TfLiteTensor* output = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &operand));
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &update));
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, 2, &start_indices));
    ABSL_RETURN_IF_ERROR(PreGetOutputTensor(context, tflite_node, 0, &output));

    if (tflite::IsConstantTensor(start_indices)) {
      return absl::InvalidArgumentError(
          "start_indices must be a runtime tensor");
    }
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    if (!CanMapAxis(operand)) {
      return absl::InvalidArgumentError("operand dims must be 1D-4D.");
    }
    const ::ml_drift::BHWC operand_shape = MapAxis(operand);
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    if (!CanMapAxis(update)) {
      return absl::InvalidArgumentError("update dims must be 1D-4D.");
    }
    const ::ml_drift::BHWC update_slice_shape = MapAxis(update);
    if (update_slice_shape.b > operand_shape.b ||
        update_slice_shape.h > operand_shape.h ||
        update_slice_shape.w > operand_shape.w ||
        update_slice_shape.c > operand_shape.c) {
      return absl::InternalError(absl::StrCat(
          "Updated_slice shape must be less or equal to the "
          "array to update, but they are updated_slice: [",
          update_slice_shape.b, update_slice_shape.h, update_slice_shape.w,
          update_slice_shape.c, "], array_to_update: [", operand_shape.b,
          operand_shape.h, operand_shape.w, operand_shape.c, "]"));
    }
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 2));
    if (!CanMapAxis(start_indices)) {
      return absl::InvalidArgumentError("start_indices dims must be 1D-4D.");
    }
    int idx = tflite_node->outputs->data[0];
    ABSL_RETURN_IF_ERROR(PreCheckReadValueByTensorIdx(context, idx));
    if (!CanMapAxis(output)) {
      return absl::InvalidArgumentError("output dims must be 1D-4D.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    const TfLiteTensor* start_indices = reader->GetInputTensor(2);
    ::ml_drift::Value* start_indices_value = reader->ReadValue(2);
    const ::ml_drift::BHWC tflite_style_start_indices_shape =
        MapAxis(start_indices);
    if (start_indices_value->tensor.shape != tflite_style_start_indices_shape) {
      // Add reshape node for start_indices.
      ::ml_drift::Node* reshape_node = graph->NewNode();
      reshape_node->operation.type =
          ToString(::ml_drift::OperationType::RESHAPE);
      ::ml_drift::ReshapeAttributes start_indices_reshape_attr;
      start_indices_reshape_attr.new_shape = tflite_style_start_indices_shape;
      reshape_node->operation.attributes =
          std::move(start_indices_reshape_attr);
      graph->AddConsumer(reshape_node->id, start_indices_value->id);
      // Add output value for the reshape node.
      ::ml_drift::Value* reshape_output_value = graph->NewValue();
      reshape_output_value->tensor.type = start_indices_value->tensor.type;
      reshape_output_value->tensor.shape = tflite_style_start_indices_shape;
      graph->SetProducer(reshape_node->id, reshape_output_value->id);
      start_indices_value = reshape_output_value;
    }
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::DYNAMIC_UPDATE_SLICE);

    reader->AddInput(node, 0);  // array_to_update
    reader->AddInput(node, 1);  // updated_slice

    // Add start_indices as input.
    graph->AddConsumer(node->id, start_indices_value->id);
    reader->AddOutputs(node);
  }

 private:
  static bool CanMapAxis(const TfLiteTensor* tensor) {
    return tensor->dims && tensor->dims->size > 0 && tensor->dims->size < 5;
  }

  // Map the dimension of the tflite input tensor to kernel inputs.
  // Currently the kernel will take HWC to perform the dynamic update slice.
  // MUST call CanMapAxis a priori.
  static ::ml_drift::BHWC MapAxis(const TfLiteTensor* tensor) {
    const TfLiteIntArray* dims = tensor->dims;
    const int* data = dims->data;
    return dims->size == 1   ? ::ml_drift::BHWC(1, 1, 1, data[0])
           : dims->size == 2 ? ::ml_drift::BHWC(1, 1, data[0], data[1])
           : dims->size == 3
               ? ::ml_drift::BHWC(1, data[0], data[1], data[2])
               : ::ml_drift::BHWC(data[0], data[1], data[2], data[3]);
  }
};

class ElementwiseOperationParser : public TFLiteOperationParser {
 public:
  explicit ElementwiseOperationParser(::ml_drift::OperationType operation_type)
      : operation_type_(operation_type) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    const int kSupportedOpVersion =
        operation_type_ == ::ml_drift::OperationType::ABS         ? 5
        : operation_type_ == ::ml_drift::OperationType::FLOOR_DIV ? 3
        : operation_type_ == ::ml_drift::OperationType::MAXIMUM   ? 4
        : operation_type_ == ::ml_drift::OperationType::MINIMUM   ? 4
        : operation_type_ == ::ml_drift::OperationType::MUL       ? 8
        : operation_type_ == ::ml_drift::OperationType::ADD       ? 6
        : operation_type_ == ::ml_drift::OperationType::GELU      ? 3
                                                                  : 2;
    ABSL_RETURN_IF_ERROR(ValidateSupport(
        context, tflite_node, registration,
        {.max_version = kSupportedOpVersion,
         .gpu_flags = tflite::GpuCompatibilityFlags::kEnhancedBroadcast}));

    bool need_broadcast = false;
    if (IsOneArgumentOperation()) {
      ABSL_RETURN_IF_ERROR(
          CheckInputsConstsOutputs(context, tflite_node,
                                   /*required_runtime_inputs=*/1,
                                   /*required_const_inputs=*/0,
                                   /*required_outputs=*/1));
      ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    } else if (IsTwoArgumentOperation() &&
               CheckInputsConstsOutputs(context, tflite_node,
                                        /*required_runtime_inputs=*/2,
                                        /*required_const_inputs=*/0,
                                        /*required_outputs=*/1)
                   .ok()) {
      if (tflite_node->inputs->size != 2) {
        return absl::InvalidArgumentError("Applies only two input tensors");
      }

      const TfLiteTensor* input0 = nullptr;
      const TfLiteTensor* input1 = nullptr;
      ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input0));
      ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &input1));
      if (input0 == input1) {
        if (operation_type_ != ::ml_drift::OperationType::MUL &&
            operation_type_ != ::ml_drift::OperationType::ADD) {
          return absl::UnimplementedError(
              "No support of few identical inputs in the same operation.");
        }
        ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
      } else {
        if (operation_type_ == ::ml_drift::OperationType::MUL ||
            operation_type_ == ::ml_drift::OperationType::ADD) {
          // The "larger" input tensor must be bound to 1st input and the
          // "smaller" input tensor must be bound to 2nd input.
          ABSL_RETURN_IF_ERROR(PreCheckTensorShape(*input0));
          ABSL_RETURN_IF_ERROR(PreCheckTensorShape(*input1));
        }

        ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
        ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));

        if (input0->dims->size != input1->dims->size) {
          need_broadcast = IsBroadcastable(input0->dims, input1->dims);
          if (!need_broadcast) {
            return absl::InvalidArgumentError(
                "Inputs dimensions sizes don't match, broadcasting isn't "
                "applicable.");
          }
        }

        if (need_broadcast) {
          ABSL_RETURN_IF_ERROR(PreCheckTfLiteShape(*input0));
          ABSL_RETURN_IF_ERROR(PreCheckTfLiteShape(*input1));
        }
      }
    } else if (IsTwoArgumentOperation()) {
      ABSL_RETURN_IF_ERROR(
          CheckInputsConstsOutputs(context, tflite_node,
                                   /*required_runtime_inputs=*/1,
                                   /*required_const_inputs=*/1,
                                   /*required_outputs=*/1));
      if (!CanParseInputsWithConstTensor(context, tflite_node)) {
        return absl::InvalidArgumentError(
            "Can't parse inputs with const tensors.");
      }
    } else {
      return absl::InvalidArgumentError("Incorrect operation type passed");
    }
    if (need_broadcast) {
      ABSL_RETURN_IF_ERROR(
          PreCheckReadValueByTensorIdx(context, tflite_node->outputs->data[0]));
      TfLiteTensor* output = nullptr;
      ABSL_RETURN_IF_ERROR(
          PreGetOutputTensor(context, tflite_node, 0, &output));
      ABSL_RETURN_IF_ERROR(PreCheckTfLiteShape(*output));
    }
    ABSL_RETURN_IF_ERROR(PreCheckMaybeFuseActivationForElementwiseNode(
        operation_type_, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    if (IsBroadcastNeeded(tflite_node, reader)) {
      AddOpWithBroadcastReshape(operation_type_, tflite_node, reader, graph);
      return;
    }
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(operation_type_);
    if (operation_type_ == ::ml_drift::OperationType::ADD) {
      ::ml_drift::ElementwiseAttributes attr;
      node->operation.attributes = std::move(attr);
    }
    if (IsOneArgumentOperation()) {
      if (operation_type_ == ::ml_drift::OperationType::GELU) {
        auto tflite_options = reinterpret_cast<const TfLiteGeluParams*>(
            tflite_node->builtin_data);
        if (tflite_options->approximate) {
          node->operation.type =
              ToString(::ml_drift::OperationType::GELU_TANH_APPROX);
        }
      }

      reader->AddInput(node, 0);
    } else if (IsTwoArgumentOperation() &&
               reader
                   ->VerifyInputsConstsOutputs(tflite_node,
                                               /*runtime_inputs=*/2,
                                               /*const_inputs=*/0,
                                               /*outputs=*/1)
                   .ok()) {
      const TfLiteTensor* input0 = reader->GetInputTensor(0);
      const TfLiteTensor* input1 = reader->GetInputTensor(1);

      // TODO(b/166831113): Support the same inputs for operations.
      if (input0 == input1) {
        if (operation_type_ == ::ml_drift::OperationType::MUL) {
          // replace MUL(A, A) with SQUARE(A)
          node->operation.type = ToString(::ml_drift::OperationType::SQUARE);
          reader->AddInput(node, 0);
        } else if (operation_type_ == ::ml_drift::OperationType::ADD) {
          // replace ADD(A, A) with MUL(A, 2.0)
          node->operation.type = ToString(::ml_drift::OperationType::MUL);
          ::ml_drift::ElementwiseAttributes attr;
          attr.param = 2.0f;
          node->operation.attributes = std::move(attr);
          reader->AddInput(node, 0);
        }
      } else {
        int input_tensor0 = 0;
        int input_tensor1 = 1;
        SwapInputs(operation_type_, input0, input1, &input_tensor0,
                   &input_tensor1);
        reader->AddInput(node, input_tensor0);
        reader->AddInput(node, input_tensor1);
      }
    } else if (IsTwoArgumentOperation()) {
      ::ml_drift::ElementwiseAttributes attr;
      ParseInputsWithConstTensor(node, reader, graph, &attr.param);
      attr.runtime_tensor_is_second =
          tflite::IsConstantTensor(reader->GetInputTensor(0));
      node->operation.attributes = std::move(attr);
    }

    reader->AddOutputs(node);
    HandleFusedActivation(operation_type_, tflite_node, graph, node);
  }

 private:
  // Swap the inputs for MUL and ADD operations.
  void SwapInputs(::ml_drift::OperationType operation_type,
                  const TfLiteTensor* input0, const TfLiteTensor* input1,
                  int* input_tensor0, int* input_tensor1) {
    if (operation_type != ::ml_drift::OperationType::MUL &&
        operation_type != ::ml_drift::OperationType::ADD) {
      return;
    }

    // The "larger" input tensor must be bound to 1st input and the
    // "smaller" input tensor must be bound to 2nd input.
    const ::ml_drift::BHWC shape0 =
        ExtractTensorShapeWithTfLiteBroadcast(input0);
    const ::ml_drift::BHWC shape1 =
        ExtractTensorShapeWithTfLiteBroadcast(input1);
    if (shape0.b <= shape1.b && shape0.h <= shape1.h && shape0.w <= shape1.w &&
        shape0.c == shape1.c) {
      *input_tensor0 = 1;
      *input_tensor1 = 0;
    }
  }

  // Check if broadcast is needed.
  bool IsBroadcastNeeded(const TfLiteNode* tflite_node,
                         ObjectReader* reader) const {
    if (!IsTwoArgumentOperation() ||
        !reader
             ->VerifyInputsConstsOutputs(tflite_node,
                                         /*runtime_inputs=*/2,
                                         /*const_inputs=*/0,
                                         /*outputs=*/1)
             .ok()) {
      return false;
    }
    const TfLiteTensor* input0 = reader->GetInputTensor(0);
    const TfLiteTensor* input1 = reader->GetInputTensor(1);
    if (input0 == input1) return false;

    // Check if the inputs are broadcastable and dimensions are
    // synchronized. If true, add reshapes nodes for the inputs and then
    // create the node. e.g. [1, 128, 512] + [128, 512] => [1, 1, 128, 512]
    // + [1, 1, 128, 512]
    if (!IsBroadcastable(input0->dims, input1->dims) ||
        input0->dims->size == input1->dims->size) {
      return false;
    }

    return true;
  }

  // Should only be called if IsBroadcastNeeded returns true.
  // Inserts additional reshapes for broadcast.
  void AddOpWithBroadcastReshape(const ::ml_drift::OperationType operation_type,
                                 const TfLiteNode* tflite_node,
                                 ObjectReader* reader,
                                 ::ml_drift::GraphFloat32* graph) {
    const TfLiteTensor* input0 = reader->GetInputTensor(0);
    const TfLiteTensor* input1 = reader->GetInputTensor(1);

    int input_tensor0 = 0;
    int input_tensor1 = 1;
    SwapInputs(operation_type_, input0, input1, &input_tensor0, &input_tensor1);
    ::ml_drift::Value* value0 = reader->ReadValue(input_tensor0);
    ::ml_drift::Value* value1 = reader->ReadValue(input_tensor1);

    ::ml_drift::BHWC input0_shape = ExtractTensorShapeWithTfLiteBroadcast(
        reader->GetInputTensor(input_tensor0));
    ::ml_drift::BHWC input1_shape = ExtractTensorShapeWithTfLiteBroadcast(
        reader->GetInputTensor(input_tensor1));

    // Add reshape node for input0
    ::ml_drift::Node* reshape_node0 = graph->NewNode();
    reshape_node0->operation.type =
        ToString(::ml_drift::OperationType::RESHAPE);
    ::ml_drift::ReshapeAttributes reshape_attr;
    reshape_attr.new_shape = input0_shape;
    reshape_node0->operation.attributes = std::move(reshape_attr);
    graph->AddConsumer(reshape_node0->id, value0->id);
    ::ml_drift::Value* reshape_output_value = graph->NewValue();
    reshape_output_value->tensor.type = value0->tensor.type;
    reshape_output_value->tensor.shape = input0_shape;
    graph->SetProducer(reshape_node0->id, reshape_output_value->id);

    // Add reshape node for input1
    ::ml_drift::Node* reshape_node1 = graph->NewNode();
    reshape_node1->operation.type =
        ToString(::ml_drift::OperationType::RESHAPE);
    ::ml_drift::ReshapeAttributes reshape_attr1;
    reshape_attr1.new_shape = input1_shape;
    reshape_node1->operation.attributes = std::move(reshape_attr1);
    graph->AddConsumer(reshape_node1->id, value1->id);
    ::ml_drift::Value* reshape_output_value1 = graph->NewValue();
    reshape_output_value1->tensor.type = value1->tensor.type;
    reshape_output_value1->tensor.shape = input1_shape;
    graph->SetProducer(reshape_node1->id, reshape_output_value1->id);

    // Link the reshape node to the original node
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(operation_type_);
    ::ml_drift::ElementwiseAttributes attr;
    node->operation.attributes = std::move(attr);
    graph->AddConsumer(node->id, reshape_output_value->id);
    graph->AddConsumer(node->id, reshape_output_value1->id);

    // Create output value for the node.
    ::ml_drift::Value* output_value = graph->NewValue();
    graph->SetProducer(node->id, output_value->id);
    // Set the output shape to the tflite output tensor shape.
    const TfLiteTensor* output_tensor = reader->GetOutputTensor(0);
    ::ml_drift::BHWC tflite_style_output_shape =
        ExtractTensorShapeWithTfLiteBroadcast(output_tensor);
    output_value->tensor.shape = tflite_style_output_shape;
    output_value->tensor.type = ToDataType(output_tensor->type);

    // Reshape the output to the ml_drift output tensor shape.
    ::ml_drift::Node* reshape = graph->NewNode();
    reshape->operation.type = ToString(::ml_drift::OperationType::RESHAPE);
    graph->AddConsumer(reshape->id, output_value->id);
    reader->AddOutput(reshape, 0);
    ::ml_drift::ReshapeAttributes output_reshape_attr;
    output_reshape_attr.new_shape =
        graph->FindOutputs(reshape->id)[0]->tensor.shape;
    reshape->operation.attributes = std::move(output_reshape_attr);

    HandleFusedActivation(operation_type, tflite_node, graph, reshape);
  }

  static inline bool IsFloat32Convertible(TfLiteType dtype) {
    return dtype == kTfLiteFloat32 ||  //
           dtype == kTfLiteFloat16 ||  //
           dtype == kTfLiteInt4 ||     //
           dtype == kTfLiteInt8 ||     //
           dtype == kTfLiteUInt8 ||    //
           dtype == kTfLiteInt32;
  }

  // Check if tfl_tensor can be converted to Tensor<Scalar, DataType::FLOAT32>.
  static bool IsScalarFloat32Convertible(const TfLiteTensor* tfl_tensor) {
    return IsFloat32Convertible(tfl_tensor->type) &&
           tflite::NumElements(tfl_tensor) == 1;
  }

  // Check if tfl_tensor can be converted to Tensor<Scalar, DataType::INT32>.
  static bool IsScalarInt32Convertible(const TfLiteTensor* tfl_tensor) {
    const TfLiteType dtype = tfl_tensor->type;
    return (dtype == kTfLiteBool ||     //
            dtype == kTfLiteFloat32 ||  //
            dtype == kTfLiteInt8 ||     //
            dtype == kTfLiteUInt8 ||    //
            dtype == kTfLiteInt16 ||    //
            dtype == kTfLiteUInt16 ||   //
            dtype == kTfLiteInt32) &&   //
           tflite::NumElements(tfl_tensor) == 1;
  }

  // Check if tfl_tensor can be converted to Tensor<Linear, DataType::FLOAT32>.
  static bool IsLinearFloat32Convertible(const TfLiteTensor* tfl_tensor) {
    const TfLiteIntArray* dims = tfl_tensor->dims;
    return IsFloat32Convertible(tfl_tensor->type) &&
           tflite::NumElements(dims->data, dims->size - 1) == 1;
  }

  // Check if tfl_tensor can be converted to Tensor<::ml_drift::BHWC,
  // DataType::FLOAT32>.
  static bool IsBhwcFloat32Convertible(const TfLiteTensor* tfl_tensor) {
    return IsFloat32Convertible(tfl_tensor->type) && tfl_tensor->dims->size < 5;
  }

  static bool CanParseInputsWithConstTensor(const TfLiteContext* context,
                                            const TfLiteNode* tflite_node) {
    TfLiteTensor* input0 = context->tensors + tflite_node->inputs->data[0];
    if (!input0) {
      ABSL_LOG(ERROR) << "Couldn't get the 1st input tensor.";
      return false;
    }
    TfLiteTensor* input1 = context->tensors + tflite_node->inputs->data[1];
    if (!input1) {
      ABSL_LOG(ERROR) << "Couldn't get the 2nd input tensor.";
      return false;
    }
    const bool constant_tensor0 = tflite::IsConstantTensor(input0);
    const bool constant_tensor1 = tflite::IsConstantTensor(input1);

    // If both are constant tensors, it should be pre-computed.
    if (constant_tensor0 && constant_tensor1) {
      ABSL_LOG(ERROR) << "No runtime input tensors.";
      return false;
    }

    // Simple case when both are runtime tensors.
    if (!constant_tensor0 && !constant_tensor1) return true;

    // Create aliases for constant and runtime tensors.
    int runtime_tensor_index;
    int constant_tensor_index;
    const TfLiteTensor* constant_tensor;
    if (constant_tensor0) {
      runtime_tensor_index = 1;
      constant_tensor_index = 0;
      constant_tensor = input0;
    } else {
      runtime_tensor_index = 0;
      constant_tensor_index = 1;
      constant_tensor = input1;
    }

    if (constant_tensor->sparsity) {
      ABSL_LOG(ERROR) << GetTensorDebugString(constant_tensor) << " is sparse.";
      return false;
    }

    const TfLiteIntArray* constant_dims = constant_tensor->dims;
    const bool convertible_to_f32 =
        constant_tensor->type == kTfLiteFloat32 ||
        constant_tensor->type == kTfLiteFloat16 ||
        (constant_tensor->quantization.type ==
             TfLiteQuantizationType::kTfLiteAffineQuantization &&
         (constant_tensor->type == kTfLiteInt8 ||
          constant_tensor->type == kTfLiteUInt8 ||
          constant_tensor->type == kTfLiteInt4));
    if (tflite::NumElements(constant_dims) == 1) {
      if (convertible_to_f32) {
        if (IsScalarFloat32Convertible(constant_tensor)) return true;
      } else if (constant_tensor->type == kTfLiteInt32) {
        if (IsScalarInt32Convertible(constant_tensor)) return true;
      }
      ABSL_LOG(ERROR) << GetTensorDebugString(constant_tensor)
                      << " isn't scalar-convertible.";
      return false;
    }
    if (!convertible_to_f32) return true;
    if (IsLinearConvertible(constant_dims)) {
      if (IsLinearFloat32Convertible(constant_tensor)) return true;
      ABSL_LOG(ERROR) << GetTensorDebugString(constant_tensor)
                      << " isn't linear-convertible.";
      return false;
    }
    if (IsBhwcFloat32Convertible(constant_tensor)) return true;
    ABSL_LOG(ERROR) << GetTensorDebugString(constant_tensor)
                    << " isn't ::ml_drift::BHWC-convertible.";
    return false;
  }

  // Specialization of TfLiteTensorToTensor<Tensor<Scalar, DataType::FLOAT32>>.
  static ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::FLOAT32>
  ConvertToScalarFloat32Tensor(const TfLiteTensor* tfl_tensor) {
    ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::FLOAT32>
        mld_tensor;
    mld_tensor.data.resize(1);
    CopyFloat32Data(tfl_tensor, &mld_tensor.data[0]);
    mld_tensor.shape.v = 1;
    return mld_tensor;
  }

  // Specialization of TfLiteTensorToTensor<Tensor<Scalar, DataType::INT32>>.
  static ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::INT32>
  ConvertToScalarInt32Tensor(const TfLiteTensor* tfl_tensor) {
    const TfLiteType dtype = tfl_tensor->type;
    ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::INT32>
        mld_tensor;
    mld_tensor.data.push_back(
        dtype == kTfLiteFloat32  ? tfl_tensor->data.f[0]
        : dtype == kTfLiteBool   ? tfl_tensor->data.b[0]
        : dtype == kTfLiteInt8   ? tfl_tensor->data.int8[0]
        : dtype == kTfLiteUInt8  ? tfl_tensor->data.uint8[0]
        : dtype == kTfLiteInt16  ? tfl_tensor->data.i16[0]
        : dtype == kTfLiteUInt16 ? tfl_tensor->data.ui16[0]
                                 : tfl_tensor->data.i32[0]);
    mld_tensor.shape.v = 1;
    return mld_tensor;
  }

  // Specialization of TfLIteTensorToTensor<Tensor<Linear, DataType::FLOAT32>>.
  static ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
  ConvertToLinearFloat32Tensor(const TfLiteTensor* tfl_tensor) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
        mld_tensor;
    const int n = tflite::NumElements(tfl_tensor);
    mld_tensor.data.resize(n);
    CopyFloat32Data(tfl_tensor, &mld_tensor.data[0]);
    mld_tensor.shape.v = n;
    return mld_tensor;
  }

  // Specialization of TfLIteTensorToTensor<Tensor<::ml_drift::BHWC,
  // DataType::FLOAT32>>.
  static ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>
  ConvertToBhwcFloat32Tensor(const TfLiteTensor* tfl_tensor) {
    ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>
        mld_tensor;
    mld_tensor.data.resize(tflite::NumElements(tfl_tensor));
    CopyFloat32Data(tfl_tensor, &mld_tensor.data[0]);
    mld_tensor.shape = ExtractTensorShape(tfl_tensor);
    return mld_tensor;
  }

  static void ParseInputsWithConstTensor(
      ::ml_drift::Node* node, ObjectReader* reader,
      ::ml_drift::GraphFloat32* graph,
      ::ml_drift::TensorOrScalar* tensor_or_scalar) {
    // Determine runtime/constant tensors.
    const TfLiteTensor* input0 = reader->GetInputTensor(0);
    const TfLiteTensor* input1 = reader->GetInputTensor(1);
    const bool constant_tensor0 = tflite::IsConstantTensor(input0);
    const bool constant_tensor1 = tflite::IsConstantTensor(input1);

    // Simple case when both are runtime tensors.
    if (!constant_tensor0 && !constant_tensor1) {
      reader->AddInput(node, 0);
      reader->AddInput(node, 1);
      return;
    }

    // Create aliases for constant and runtime tensors.
    int runtime_tensor_index;
    int constant_tensor_index;
    const TfLiteTensor* constant_tensor;
    if (constant_tensor0) {
      runtime_tensor_index = 1;
      constant_tensor_index = 0;
      constant_tensor = input0;
    } else {
      runtime_tensor_index = 0;
      constant_tensor_index = 1;
      constant_tensor = input1;
    }

    reader->AddInput(node, runtime_tensor_index);
    const TfLiteIntArray* constant_dims = constant_tensor->dims;
    const bool convertible_to_f32 =
        constant_tensor->type == kTfLiteFloat32 ||
        constant_tensor->type == kTfLiteFloat16 ||
        (constant_tensor->quantization.type ==
             TfLiteQuantizationType::kTfLiteAffineQuantization &&
         (constant_tensor->type == kTfLiteInt8 ||
          constant_tensor->type == kTfLiteUInt8 ||
          constant_tensor->type == kTfLiteInt4));
    if (constant_dims->size < 1 || tflite::NumElements(constant_dims) == 1) {
      if (convertible_to_f32) {
        const ::ml_drift::Tensor<::ml_drift::Scalar,
                                 ::ml_drift::DataType::FLOAT32>
            t = ConvertToScalarFloat32Tensor(constant_tensor);
        *tensor_or_scalar = t.data[0];
        return;
      }
      if (constant_tensor->type == kTfLiteInt32) {
        const ::ml_drift::Tensor<::ml_drift::Scalar,
                                 ::ml_drift::DataType::INT32>
            t = ConvertToScalarInt32Tensor(constant_tensor);
        *tensor_or_scalar = t.data[0];
        return;
      }
    }
    if (!convertible_to_f32) {
      if (reader->IsConstantTensor(constant_tensor_index)) {
        const ::ml_drift::Value* input =
            reader->AddConstInput(constant_tensor_index, /*layout=*/{});
        graph->AddConsumer(node->id, input->id);
      } else {
        reader->AddInput(node, constant_tensor_index);
      }
      return;
    }
    if (IsLinearConvertible(constant_dims)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
          tensor = ConvertToLinearFloat32Tensor(constant_tensor);
      *tensor_or_scalar = std::move(tensor);
      return;
    }
    if (constant_dims->size < 5) {
      ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>
          tensor = ConvertToBhwcFloat32Tensor(constant_tensor);
      if (constant_dims->size == 2) {
        tensor.shape = ::ml_drift::BHWC(1, 1, tensor.shape.b, tensor.shape.c);
      } else if (constant_dims->size == 3) {
        tensor.shape =
            ::ml_drift::BHWC(1, tensor.shape.b, tensor.shape.w, tensor.shape.c);
      }
      *tensor_or_scalar = std::move(tensor);
    }
  }

  static void HandleFusedActivation(::ml_drift::OperationType operation_type,
                                    const TfLiteNode* tflite_node,
                                    ::ml_drift::GraphFloat32* graph,
                                    ::ml_drift::Node* node) {
    TfLiteFusedActivation activation = kTfLiteActNone;
    switch (operation_type) {
      case ::ml_drift::OperationType::ADD:
        if (const auto* params = static_cast<const TfLiteAddParams*>(
                tflite_node->builtin_data)) {
          activation = params->activation;
        }
        break;
      case ::ml_drift::OperationType::DIV:
        if (const auto* params = static_cast<const TfLiteDivParams*>(
                tflite_node->builtin_data)) {
          activation = params->activation;
        }
        break;
      case ::ml_drift::OperationType::MUL:
        if (const auto* params = static_cast<const TfLiteMulParams*>(
                tflite_node->builtin_data)) {
          activation = params->activation;
        }
        break;
      case ::ml_drift::OperationType::SUB:
        if (const auto* params = static_cast<const TfLiteSubParams*>(
                tflite_node->builtin_data)) {
          activation = params->activation;
        }
        break;
      default:
        break;
    }
    ::litert::ml_drift::HandleFusedActivation(activation, graph, node);
  }

  bool IsOneArgumentOperation() const {
    switch (operation_type_) {
      case ::ml_drift::OperationType::ABS:
      case ::ml_drift::OperationType::CEIL:
      case ::ml_drift::OperationType::COPY:
      case ::ml_drift::OperationType::COS:
      case ::ml_drift::OperationType::ELU:
      case ::ml_drift::OperationType::EXP:
      case ::ml_drift::OperationType::FLOOR:
      case ::ml_drift::OperationType::GELU:
      case ::ml_drift::OperationType::LOG:
      case ::ml_drift::OperationType::LOGICAL_NOT:
      case ::ml_drift::OperationType::NEG:
      case ::ml_drift::OperationType::ROUND:
      case ::ml_drift::OperationType::RSQRT:
      case ::ml_drift::OperationType::SIGMOID:
      case ::ml_drift::OperationType::SIGN:
      case ::ml_drift::OperationType::SIN:
      case ::ml_drift::OperationType::SQRT:
      case ::ml_drift::OperationType::SQUARE:
      case ::ml_drift::OperationType::TANH:
        return true;
      default:
        return false;
    }
  }

  bool IsTwoArgumentOperation() const {
    switch (operation_type_) {
      case ::ml_drift::OperationType::ADD:
      case ::ml_drift::OperationType::ATAN2:
      case ::ml_drift::OperationType::DIV:
      case ::ml_drift::OperationType::EQUAL:
      case ::ml_drift::OperationType::FLOOR_DIV:
      case ::ml_drift::OperationType::FLOOR_MOD:
      case ::ml_drift::OperationType::GREATER:
      case ::ml_drift::OperationType::GREATER_EQUAL:
      case ::ml_drift::OperationType::LESS:
      case ::ml_drift::OperationType::LESS_EQUAL:
      case ::ml_drift::OperationType::LOGICAL_AND:
      case ::ml_drift::OperationType::LOGICAL_OR:
      case ::ml_drift::OperationType::LOGICAL_XOR:
      case ::ml_drift::OperationType::MAXIMUM:
      case ::ml_drift::OperationType::MINIMUM:
      case ::ml_drift::OperationType::MUL:
      case ::ml_drift::OperationType::NOT_EQUAL:
      case ::ml_drift::OperationType::POW:
      case ::ml_drift::OperationType::SHIFT_LEFT:
      case ::ml_drift::OperationType::SHIFT_RIGHT:
      case ::ml_drift::OperationType::SQUARED_DIFF:
      case ::ml_drift::OperationType::SUB:
        return true;
      default:
        return false;
    }
  }

  static inline bool CanExtractTensorShape(const TfLiteTensor* tflite_tensor) {
    return tflite_tensor->dims->size < 5;
  }

  // Extracts shape from TfLiteTensor. And expand the shape to 4D based on the
  // broadcasting needs.
  static ::ml_drift::BHWC ExtractTensorShapeWithTfLiteBroadcast(
      const TfLiteTensor* tflite_tensor) {
    const TfLiteIntArray* dims = tflite_tensor->dims;
    if (dims->size == 0) return ::ml_drift::BHWC(1, 1, 1, 1);
    if (dims->size == 1) return ::ml_drift::BHWC(1, 1, 1, dims->data[0]);
    if (dims->size == 2)
      return ::ml_drift::BHWC(1, 1, dims->data[0], dims->data[1]);
    if (dims->size == 3) {
      return ::ml_drift::BHWC(1, dims->data[0], dims->data[1], dims->data[2]);
    }
    return ::ml_drift::BHWC(dims->data[0], dims->data[1], dims->data[2],
                            dims->data[3]);
  }

  absl::Status PreCheckInputsWithConstTensor(const TfLiteContext* context,
                                             const TfLiteNode* tflite_node,
                                             const std::string& opname) {
    const TfLiteTensor* input0;
    if (tflite::GetInputSafe(context, tflite_node, 0, &input0) != kTfLiteOk) {
      return absl::InvalidArgumentError(
          "Couldn't get the 1st input tensor for " + opname);
    }
    const TfLiteTensor* input1;
    if (tflite::GetInputSafe(context, tflite_node, 1, &input1) != kTfLiteOk) {
      return absl::InvalidArgumentError(
          "Couldn't get the 2nd input tensor for " + opname);
    }
    const bool constant_tensor0 = tflite::IsConstantTensor(input0);
    const bool constant_tensor1 = tflite::IsConstantTensor(input1);

    const bool runtime_tensor0 = !constant_tensor0;
    const bool runtime_tensor1 = !constant_tensor1;
    if (!runtime_tensor0 || !runtime_tensor1) {
      int runtime_tensor_index = 0;
      int constant_tensor_index = 1;
      TfLiteIntArray* constant_dims = input1->dims;
      if (constant_tensor0 && runtime_tensor1) {
        runtime_tensor_index = 1;
        constant_tensor_index = 0;
        constant_dims = input0->dims;
      }
      const TfLiteTensor* constant_tensor =
          tflite::GetInput(context, tflite_node, constant_tensor_index);
      bool convertible_to_f32 = constant_tensor->type == kTfLiteFloat32 ||
                                constant_tensor->type == kTfLiteFloat16;
      convertible_to_f32 |=
          constant_tensor->quantization.type ==
              TfLiteQuantizationType::kTfLiteAffineQuantization &&
          (constant_tensor->type == kTfLiteInt8 ||
           constant_tensor->type == kTfLiteUInt8 ||
           constant_tensor->type == kTfLiteInt4);
      if (!convertible_to_f32 && !(constant_tensor->type == kTfLiteInt32)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported constant tensor type: ",
                         TfLiteTypeGetName(constant_tensor->type), " ",
                         constant_tensor->name));
      }
      if (constant_tensor->type == kTfLiteInt32) {
        return absl::OkStatus();
      }
      if (constant_dims->size <= 0 || tflite::NumElements(constant_dims) == 1) {
        return absl::OkStatus();
      }
      if (IsLinearConvertible(constant_dims)) {
        ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
            tensor;
        ABSL_RETURN_IF_ERROR(PreCheckTensorToTensor(constant_tensor, &tensor));
      } else if (constant_dims->size <= 4) {
        ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>
            tensor;
        ABSL_RETURN_IF_ERROR(PreCheckTensorToTensor(constant_tensor, &tensor));
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected to get constant tensor dimension <= 4, but got: ",
            constant_dims->size));
      }
    }
    return absl::OkStatus();
  }

  ::ml_drift::OperationType operation_type_;
};

class EmbeddingLookupOperationParser : public TFLiteOperationParser {
  enum { kInputSrcId, kInputWeightsId, kInputScaleId, kInputZeroPointId };

 public:
  absl::Status IsSupported(const TfLiteContext* context, const TfLiteNode* node,
                           const TfLiteRegistration* registration) final {
    const TfLiteTensor* value;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, node, kInputWeightsId, &value));

    if (value->quantization.params == nullptr &&
        value->type != kTfLiteFloat32) {
      return absl::InvalidArgumentError(
          "EMBEDDING_LOOKUP: Empty quantization params.");
    }

    const TfLiteTensor* weights_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, node, kInputWeightsId, &weights_tensor));

    if (weights_tensor->type == kTfLiteInt8 ||
        weights_tensor->type == kTfLiteInt4 ||
        weights_tensor->type == kTfLiteInt2) {
      if (weights_tensor->quantization.params == nullptr) {
        return absl::InvalidArgumentError(
            "EMBEDDING_LOOKUP: Empty quantization params.");
      }
      int weights_tensor_idx = 0;
      ABSL_RETURN_IF_ERROR(
          GetTensorId(context, node, kInputWeightsId, &weights_tensor_idx));
      ABSL_RETURN_IF_ERROR(
          PreCheckReadQuantizedValueByTensorIdx(context, weights_tensor_idx));
    } else if (weights_tensor->type != kTfLiteFloat32) {
      return absl::InvalidArgumentError(
          "EMBEDDING_LOOKUP: Unsupported weights type.");
    }
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, node, 0));

    ABSL_RETURN_IF_ERROR(ValidateSupport(context, node, registration, {}));
    ABSL_RETURN_IF_ERROR(
        PreCheckReadValueByTensorIdx(context, node->outputs->data[0]));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    reader->AllowSharingInput(kInputWeightsId);
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::EMBEDDING_LOOKUP);
    const TfLiteTensor weights_tensor = *reader->GetInputTensor(1);
    const ObjectReader::ConstantInputSharingInfo weights_share =
        reader->GetSharingInfoByNodeInputIndex(kInputWeightsId);
    reader->AddInput(node, 0);
    if (weights_share.IsShared()) {
      int tensor_idx = tflite_node->inputs->data[kInputWeightsId];
      ::ml_drift::Value* weights_val;
      if (weights_tensor.type == kTfLiteInt8 ||
          weights_tensor.type == kTfLiteInt4 ||
          weights_tensor.type == kTfLiteInt2) {
        reader->ReadQuantizedValueByTensorIdx(tensor_idx, &weights_val);
      } else {
        // TODO: who/fengwuyao - Check in IsSupported instead of crashing here.
        ABSL_LOG(FATAL)
            << "EMBEDDING_LOOKUP: Unsupported external weights type: "
            << weights_tensor.type;
      }

      ::ml_drift::EmbeddingLookupAttributes attr;
      attr.original_weights_shape = ::ml_drift::OHWI(
          weights_val->tensor.shape.b, weights_val->tensor.shape.h,
          weights_val->tensor.shape.w, weights_val->tensor.shape.c);
      attr.weights_type =
          weights_tensor.type == kTfLiteInt8
              ? ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt8
          : weights_tensor.type == kTfLiteInt4
              ? ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt4
              : ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt2;
      attr.scale_zp_shape =
          ::ml_drift::OHWI(attr.original_weights_shape.o, 1, 1, 1);
      if (weights_tensor.quantization.type == kTfLiteBlockwiseQuantization) {
        const auto* qparams =
            reinterpret_cast<const TfLiteBlockwiseQuantization*>(
                weights_tensor.quantization.params);
        attr.scale_zp_shape.i =
            attr.original_weights_shape.i / qparams->blocksize;
      }

      graph->AddConsumer(node->id, weights_val->id);
      reader->SetSharedTensor(weights_val->id, weights_share.PreferredId(),
                              tflite_node->inputs->data[kInputWeightsId],
                              /*dequant_forced=*/false,
                              /*layout=*/std::nullopt);
      node->operation.attributes = std::move(attr);
    } else {
      TfLiteAffineQuantization* quantization_data = nullptr;
      if (weights_tensor.quantization.type == kTfLiteAffineQuantization) {
        quantization_data = reinterpret_cast<TfLiteAffineQuantization*>(
            weights_tensor.quantization.params);
      } else if (weights_tensor.quantization.type ==
                 kTfLiteBlockwiseQuantization) {
        ABSL_LOG(FATAL) << "EMBEDDING_LOOKUP: Use external weights for "
                           "blockwise quantization.";
      }
      if (weights_tensor.type == kTfLiteInt8) {
        SetInt8Attributes(reader, node, quantization_data);
      } else if (weights_tensor.type == kTfLiteInt4) {
        SetInt4Attributes(reader, node, quantization_data);
      } else if (weights_tensor.type == kTfLiteInt2) {
        SetInt2Attributes(reader, node, quantization_data);
      } else if (weights_tensor.type == kTfLiteFloat32) {
        SetFloat32Attributes(reader, node, quantization_data);
      }
    }

    reader->AddOutputs(node);
  }

 private:
  // TODO: b/352629255 - Refactor the code to remove duplication.
  void SetInt2Attributes(ObjectReader* reader, ::ml_drift::Node* node,
                         TfLiteAffineQuantization* quantization_data) {
    ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::UINT8>
        lookup_tensor;
    reader->ReadTensor(1, &lookup_tensor, ReadTensorFlags::kExtraBytes);
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
        scale_tensor;
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
        zero_point_tensor;

    scale_tensor.shape = ::ml_drift::OHWI(lookup_tensor.shape.h, 1, 1, 1);
    scale_tensor.data.resize(scale_tensor.shape.DimensionsProduct());
    std::memcpy(scale_tensor.data.data(), &quantization_data->scale->data[0],
                sizeof(float) * quantization_data->scale->size);

    zero_point_tensor.shape = ::ml_drift::OHWI(lookup_tensor.shape.h, 1, 1, 1);
    zero_point_tensor.data.resize(zero_point_tensor.shape.DimensionsProduct());
    if (quantization_data->zero_point->size == quantization_data->scale->size) {
      std::memcpy(zero_point_tensor.data.data(),
                  &quantization_data->zero_point->data[0],
                  sizeof(float) * quantization_data->zero_point->size);
    } else {
      std::fill(zero_point_tensor.data.begin(), zero_point_tensor.data.end(),
                quantization_data->zero_point->data[0]);
    }
    // Construct attributes.
    ::ml_drift::EmbeddingLookupAttributes attr;
    attr.original_weights_shape =
        ::ml_drift::OHWI(lookup_tensor.shape.h, 1, 1, lookup_tensor.shape.w);
    attr.weights_scale = scale_tensor;
    attr.weights_zero_point = zero_point_tensor;
    attr.weights_type =
        ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt2;
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::UINT8>
        weights_int2;
    weights_int2.data = lookup_tensor.data;
    weights_int2.shape.h = 1;
    weights_int2.shape.w = 1;
    weights_int2.shape.o = lookup_tensor.shape.h;
    weights_int2.shape.i = lookup_tensor.shape.w;
    attr.weights = weights_int2;
    node->operation.attributes = std::move(attr);
  }

  void SetInt4Attributes(ObjectReader* reader, ::ml_drift::Node* node,
                         TfLiteAffineQuantization* quantization_data) {
    ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::UINT8>
        lookup_tensor;
    reader->ReadTensor(1, &lookup_tensor, ReadTensorFlags::kExtraBytes);
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
        scale_tensor;
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
        zero_point_tensor;

    scale_tensor.shape = ::ml_drift::OHWI(lookup_tensor.shape.h, 1, 1, 1);
    scale_tensor.data.resize(scale_tensor.shape.DimensionsProduct());
    std::memcpy(scale_tensor.data.data(), &quantization_data->scale->data[0],
                sizeof(float) * quantization_data->scale->size);

    zero_point_tensor.shape = ::ml_drift::OHWI(lookup_tensor.shape.h, 1, 1, 1);
    zero_point_tensor.data.resize(zero_point_tensor.shape.DimensionsProduct());
    if (quantization_data->zero_point->size == quantization_data->scale->size) {
      std::memcpy(zero_point_tensor.data.data(),
                  &quantization_data->zero_point->data[0],
                  sizeof(float) * quantization_data->zero_point->size);
    } else {
      std::fill(zero_point_tensor.data.begin(), zero_point_tensor.data.end(),
                quantization_data->zero_point->data[0]);
    }
    // Construct attributes.
    ::ml_drift::EmbeddingLookupAttributes attr;
    attr.original_weights_shape =
        ::ml_drift::OHWI(lookup_tensor.shape.h, 1, 1, lookup_tensor.shape.w);
    attr.weights_scale = scale_tensor;
    attr.weights_zero_point = zero_point_tensor;
    attr.weights_type =
        ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt4;
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::UINT8>
        weights_int4;
    weights_int4.data = lookup_tensor.data;
    weights_int4.shape.h = 1;
    weights_int4.shape.w = 1;
    weights_int4.shape.o = lookup_tensor.shape.h;
    weights_int4.shape.i = lookup_tensor.shape.w;
    attr.weights = weights_int4;
    node->operation.attributes = std::move(attr);
  }

  void SetInt8Attributes(ObjectReader* reader, ::ml_drift::Node* node,
                         TfLiteAffineQuantization* quantization_data) {
    ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::INT8>
        lookup_tensor;
    reader->ReadTensor(1, &lookup_tensor, ReadTensorFlags::kExtraBytes);
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
        scale_tensor;
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
        zero_point_tensor;

    scale_tensor.shape = ::ml_drift::OHWI(lookup_tensor.shape.h, 1, 1, 1);
    scale_tensor.data.resize(scale_tensor.shape.DimensionsProduct());
    std::memcpy(scale_tensor.data.data(), &quantization_data->scale->data[0],
                sizeof(float) * quantization_data->scale->size);

    zero_point_tensor.shape = ::ml_drift::OHWI(lookup_tensor.shape.h, 1, 1, 1);
    zero_point_tensor.data.resize(zero_point_tensor.shape.DimensionsProduct());
    if (quantization_data->zero_point->size == quantization_data->scale->size) {
      std::memcpy(zero_point_tensor.data.data(),
                  &quantization_data->zero_point->data[0],
                  sizeof(float) * quantization_data->zero_point->size);
    } else {
      std::fill(zero_point_tensor.data.begin(), zero_point_tensor.data.end(),
                quantization_data->zero_point->data[0]);
    }
    // Construct attributes.
    ::ml_drift::EmbeddingLookupAttributes attr;
    attr.original_weights_shape =
        ::ml_drift::OHWI(lookup_tensor.shape.h, 1, 1, lookup_tensor.shape.w);
    attr.weights_scale = scale_tensor;
    attr.weights_zero_point = zero_point_tensor;
    attr.weights_type =
        ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt8;
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>
        weights_int8;
    weights_int8.data = lookup_tensor.data;
    weights_int8.shape.h = 1;
    weights_int8.shape.w = 1;
    weights_int8.shape.o = lookup_tensor.shape.h;
    weights_int8.shape.i = lookup_tensor.shape.w;
    attr.weights = weights_int8;
    node->operation.attributes = std::move(attr);
  }

  void SetFloat32Attributes(ObjectReader* reader, ::ml_drift::Node* node,
                            TfLiteAffineQuantization* quantization_data) {
    ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32>
        lookup_tensor;
    reader->ReadTensor(1, &lookup_tensor, ReadTensorFlags::kExtraBytes);
    // Construct attributes.
    ::ml_drift::EmbeddingLookupAttributes attr;
    attr.original_weights_shape =
        ::ml_drift::OHWI(lookup_tensor.shape.h, 1, 1, lookup_tensor.shape.w);
    attr.weights_type =
        ::ml_drift::EmbeddingLookupAttributes::WeightsType::kFloat32;

    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
        weights_float32;
    weights_float32.data = lookup_tensor.data;
    weights_float32.shape.h = 1;
    weights_float32.shape.w = 1;
    weights_float32.shape.o = lookup_tensor.shape.h;
    weights_float32.shape.i = lookup_tensor.shape.w;
    attr.weights = weights_float32;
    node->operation.attributes = std::move(attr);
  }
};

class FullyConnectedOperationParser : public TFLiteOperationParser {
  enum { kInputSrcId, kInputWeightsId, kInputBiasId };
  enum { kOutputId };

 public:
  explicit FullyConnectedOperationParser(const ModelBuilderOptions& options)
      : options_(options) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 14}));

    // TODO(b/372428865) : Get tensor sharing information in IsSupported.
    //            To make checkings more completed, sharing information is
    //            required.
    const TfLiteTensor* input_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, kInputSrcId, &input_tensor));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, kInputSrcId));

    const TfLiteTensor* weights_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node,
                                           kInputWeightsId, &weights_tensor));
    if (weights_tensor->quantization.type == kTfLiteAffineQuantization) {
      // Check for quantized weight sharing.
      int weights_tensor_idx = 0;
      ABSL_RETURN_IF_ERROR(GetTensorId(context, tflite_node, kInputWeightsId,
                                       &weights_tensor_idx));
      ABSL_RETURN_IF_ERROR(
          PreCheckReadQuantizedValueByTensorIdx(context, weights_tensor_idx));
      if (options_.enable_raw_weights_propagation) {
        if (weights_tensor->type != kTfLiteInt8) {
          return absl::UnimplementedError(
              "Expected int8 weights for raw weights propagation.");
        }
        ::ml_drift::BHWC weights_shape;
        SetAllDimensions(weights_tensor->dims, &weights_shape);
        if (weights_shape.h != 1 || weights_shape.w != 1) {
          return absl::UnimplementedError(
              "Expected height and width of 1 for quantized weights.");
        }
      }
    } else if (weights_tensor->quantization.type ==
               kTfLiteBlockwiseQuantization) {
      // TODO: who/impjdi - Add meaningful check.
    } else {
      ABSL_RETURN_IF_ERROR(
          PreCheckReadValue(context, tflite_node, kInputWeightsId));
    }

    if (tflite::GetVariableInput(const_cast<TfLiteContext*>(context),
                                 tflite_node, kInputBiasId)) {
      ABSL_RETURN_IF_ERROR(
          PreCheckReadValue(context, tflite_node, kInputBiasId));
    }

    TfLiteTensor* output_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetOutputTensor(context, tflite_node, kOutputId, &output_tensor));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    const auto* params = static_cast<const TfLiteFullyConnectedParams*>(
        tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteFullyConnectedParams.");
    }
    if (params->weights_format != kTfLiteFullyConnectedWeightsFormatDefault) {
      return absl::UnimplementedError(
          "Unsupported fully connected weights format.");
    }
    ABSL_RETURN_IF_ERROR(
        PreCheckMaybeFuseActivation(tflite_node, params->activation));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    reader->AllowSharingInput(kInputWeightsId);
    reader->AllowSharingInput(kInputBiasId);
    const auto* params = static_cast<const TfLiteFullyConnectedParams*>(
        tflite_node->builtin_data);

    const ObjectReader::ConstantInputSharingInfo weights_share =
        reader->GetSharingInfoByNodeInputIndex(kInputWeightsId);
    const ObjectReader::ConstantInputSharingInfo bias_share =
        reader->GetSharingInfoByNodeInputIndex(kInputBiasId);
    const TfLiteTensor* src_tensor = reader->GetInputTensor(kInputSrcId);
    const TfLiteTensor* weights_tensor =
        reader->GetInputTensor(kInputWeightsId);
    if (reader->GetNumberOfRuntimeInputs() == 2 || weights_share.IsShared()) {
      ::ml_drift::Node* node = graph->NewNode();
      reader->AddInput(node, 0);
      if (weights_share.IsShared() &&
          (weights_tensor->quantization.type == kTfLiteAffineQuantization ||
           weights_tensor->quantization.type == kTfLiteBlockwiseQuantization)) {
        ConfigSharedWeightFullyConnectedNode(
            tflite_node->inputs->data[kInputWeightsId], weights_tensor, reader,
            graph, node);
      } else {
        node->operation.type =
            ToString(::ml_drift::OperationType::FULLY_CONNECTED);
        reader->AddInput(node, 1);
      }

      ConfigSharedBiasFullyConnectedNode(bias_share.IsShared(),
                                         tflite_node->inputs, kInputBiasId,
                                         reader, graph, node);
      if (weights_share.IsShared()) {
        auto node_inputs = graph->FindInputs(node->id);
        reader->SetSharedTensor(node_inputs[1]->id, weights_share.PreferredId(),
                                tflite_node->inputs->data[kInputWeightsId],
                                /*dequant_forced=*/false,
                                /*layout=*/std::nullopt);
        if (bias_share.IsShared()) {
          reader->SetSharedTensor(node_inputs[2]->id, bias_share.PreferredId(),
                                  tflite_node->inputs->data[kInputBiasId],
                                  /*dequant_forced=*/false,
                                  ::ml_drift::Layout::LINEAR);
        }
      }
      if (IsFullyConnectedOutputReshapeNeeded(src_tensor, weights_tensor,
                                              reader)) {
        ReshapeFullyConnectedOutput(src_tensor, weights_tensor, graph, reader,
                                    node);
      } else {
        reader->AddOutputs(node);
      }
      // `weights_tensor` may have been invalidated by AddOutputs() or
      // ReshapeFullyConnectedOutput().
      weights_tensor = reader->GetInputTensor(kInputWeightsId);

      if (weights_share.IsShared() &&
          (weights_tensor->quantization.type == kTfLiteAffineQuantization ||
           weights_tensor->quantization.type == kTfLiteBlockwiseQuantization)) {
        HandleFusedActivation(params->activation, graph, node);
      } else {
        ::ml_drift::FullyConnectedAttributes attr;
        HandleFusedActivation(params->activation, graph, node);
        node->operation.attributes = std::move(attr);
      }
      return;
    }

    if (weights_tensor->quantization.type == kTfLiteAffineQuantization) {
      if (weights_tensor->type == kTfLiteInt8 ||
          (!options_.enable_raw_weights_propagation &&
           (weights_tensor->type == kTfLiteInt4 ||
            weights_tensor->type == kTfLiteInt2))) {
        ::ml_drift::Node* node = graph->NewNode();
        reader->AddInput(node, 0);
        node->operation.attributes = GetFullyConnectedInt8Attributes(
            kInputWeightsId, kInputBiasId, reader,
            /*copy_weights=*/!options_.enable_raw_weights_propagation);
        // TODO: b/378522761 - add support for int2/int4 quantized weights.
        node->operation.type =
            ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT8);
        reader->AddOutputs(node);
        HandleFusedActivation(params->activation, graph, node);
        return;
      }
      ABSL_LOG(FATAL) << absl::UnimplementedError(
          "FULLY_CONNECTED with affine quantization.");
    }

    if (weights_tensor->quantization.type == kTfLiteBlockwiseQuantization) {
      if (weights_tensor->type == kTfLiteInt4) {
        ::ml_drift::Node* node = graph->NewNode();
        reader->AddInput(node, 0);
        node->operation.attributes = GetFullyConnectedInt4Attributes(
            kInputWeightsId, kInputBiasId, reader,
            /*copy_weights=*/!options_.enable_raw_weights_propagation);
        node->operation.type =
            ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT4);
        reader->AddOutputs(node);
        HandleFusedActivation(params->activation, graph, node);
        return;
      }
      ABSL_LOG(FATAL) << absl::UnimplementedError(
          "FULLY_CONNECTED with blockwise quantization.");
    }

    // This branch builds floating point attributes
    ::ml_drift::Node* node = graph->NewNode();
    reader->AddInput(node, 0);

    ::ml_drift::FullyConnectedAttributes attr =
        GetFullyConnectedAttributes(kInputWeightsId, kInputBiasId, reader);

    auto input = graph->FindInputs(node->id)[0];
    ::ml_drift::Node* conv = node;
    if (input->tensor.shape.h != 1 || input->tensor.shape.w != 1) {
      // In Gpu delegates assume that height and width = 1 for FullyConnected
      // Using usual convolution2d when height or width != 1
      ::ml_drift::Convolution2DAttributes conv_attr;
      conv_attr.strides = ::ml_drift::HW(1, 1);
      conv_attr.dilations = ::ml_drift::HW(1, 1);
      conv_attr.padding.appended = ::ml_drift::HW(0, 0);
      conv_attr.padding.prepended = ::ml_drift::HW(0, 0);
      conv_attr.weights = attr.weights;
      conv_attr.bias = attr.bias;
      conv->operation.type =
          ToString(::ml_drift::OperationType::CONVOLUTION_2D);
      conv->operation.attributes = std::move(conv_attr);
    } else {
      conv->operation.type =
          ToString(::ml_drift::OperationType::FULLY_CONNECTED);
      conv->operation.attributes = std::move(attr);
    }
    if (IsFullyConnectedOutputReshapeNeeded(src_tensor, weights_tensor,
                                            reader)) {
      ReshapeFullyConnectedOutput(src_tensor, weights_tensor, graph, reader,
                                  node);
    } else {
      reader->AddOutputs(node);
    }
    HandleFusedActivation(params->activation, graph, conv);
  }

 private:
  ModelBuilderOptions options_;
};

class GatherOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    // CheckGpuDelegateCompatibility makes sure we can only have one constant
    // input.
    const TfLiteTensor* tfl_input = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, 0, &tfl_input));

    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    if (!tflite_node->builtin_data) {
      return absl::InvalidArgumentError("Missing TfLiteGatherParams.");
    }
    TfLiteTensor* tfl_output;
    ABSL_RETURN_IF_ERROR(
        PreGetOutputTensor(context, tflite_node, 0, &tfl_output));
    if (tfl_input->type != tfl_output->type) {
      return absl::InvalidArgumentError("Input / output dtype mismatch.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    const auto* params =
        static_cast<const TfLiteGatherParams*>(tflite_node->builtin_data);
    const TfLiteTensor* input_tensor = reader->GetInputTensor(0);
    const TfLiteTensor* indices_tensor = reader->GetInputTensor(1);

    const bool indices_are_const = reader->IsConstantTensor(1);
    const bool indices_are_1d =
        indices_tensor && indices_tensor->dims->size == 1;

    ::ml_drift::Value* indices_value = nullptr;

    // Insert a RESHAPE if indices tensor [N] is mis-auto-expanded to [N,1,1,1].
    if (!indices_are_const && indices_are_1d) {
      const ::ml_drift::Value* original_indices = reader->ReadValue(1);
      const ::ml_drift::BHWC new_shape(1, 1, 1,
                                       original_indices->tensor.shape.b);
      ::ml_drift::Node* reshape_node = graph->NewNode();
      reshape_node->operation.type =
          ToString(::ml_drift::OperationType::RESHAPE);
      ::ml_drift::ReshapeAttributes reshape_attr;
      reshape_attr.new_shape = new_shape;
      reshape_node->operation.attributes = std::move(reshape_attr);
      graph->AddConsumer(reshape_node->id, original_indices->id);
      indices_value = graph->NewValue();
      indices_value->tensor.type = original_indices->tensor.type;
      indices_value->tensor.shape = new_shape;
      graph->SetProducer(reshape_node->id, indices_value->id);
    }

    // Insert the GATHER.
    ::ml_drift::Node* gather_node = graph->NewNode();
    gather_node->operation.type = ToString(::ml_drift::OperationType::GATHER);
    ::ml_drift::GatherAttributes gather_attr;
    gather_attr.axis = ExtractAxisFromIndex(*input_tensor, params->axis);
    gather_node->operation.attributes = std::move(gather_attr);

    if (reader->IsConstantTensor(0)) {
      ::ml_drift::Value* value = reader->AddConstInput(0, /*layout=*/{});
      graph->AddConsumer(gather_node->id, value->id);
    } else {
      reader->AddInput(gather_node, 0);
    }
    if (indices_are_const) {
      indices_value = reader->AddConstInput(1, /*layout=*/{});
      graph->AddConsumer(gather_node->id, indices_value->id);
    } else if (indices_are_1d) {
      graph->AddConsumer(gather_node->id, indices_value->id);
    } else {
      reader->AddInput(gather_node, 1);
    }
    reader->AddOutputs(gather_node);
  }
};

class HardSwishOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::HARD_SWISH);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);
  }
};

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
// For full LSTM cells, see this blog post:
// https://colah.github.io/posts/2015-08-Understanding-LSTMs/
// In addition to Peephole connections and Combined Input Forget Gates (CIFG)
// described in that post, this code also adds the following optional features:
// - Configurable activations (sigmoid or TANH)
// - L2 Normalization of gates: https://arxiv.org/abs/1607.06450
// - Output projection:
//     https://www.isca-speech.org/archive/interspeech_2014/i14_0338.html
// - Configurable clipping of cell state and output state.
class LSTMOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 4}));

    const auto* params =
        static_cast<const TfLiteLSTMParams*>(tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteLSTMParams.");
    }
    switch (params->kernel_type) {
      case kTfLiteLSTMFullKernel:
        ABSL_RETURN_IF_ERROR(
            CheckFull(context, tflite_node, registration, params));
        break;
      case kTfLiteLSTMBasicKernel:
        ABSL_RETURN_IF_ERROR(
            CheckBasic(context, tflite_node, registration, params));
        break;
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    const auto* params =
        static_cast<const TfLiteLSTMParams*>(tflite_node->builtin_data);
    switch (params->kernel_type) {
      case kTfLiteLSTMFullKernel:
        ParseLSTMAttributes(tflite_node, registration, graph, reader, params,
                            &new_variable_input_value_map_);
        break;
      case kTfLiteLSTMBasicKernel:
        ParseBasic(tflite_node, registration, graph, reader, params);
        break;
    }
  }

  absl::flat_hash_map<int, ::ml_drift::ValueId>
  GetNewValueIdsForVariableInputNodes() final {
    return new_variable_input_value_map_;
  }

 private:
  absl::Status CheckBasic(const TfLiteContext* context,
                          const TfLiteNode* tflite_node,
                          const TfLiteRegistration* registration,
                          const TfLiteLSTMParams* tf_options) {
    if (tflite_node->inputs->size != 5) {
      return absl::InvalidArgumentError("LSTM should have 5 input tensors");
    }
    if (tflite_node->outputs->size != 4) {
      return absl::InvalidArgumentError("LSTM should have 4 output tensors");
    }
    // checked in CheckGpuDelegateCompatibility
    ABSL_RETURN_IF_ERROR(CheckBasicParameters(tf_options));

    // checking for GetFullyConnectedAttributes
    ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32>
        dummy_weights;
    ABSL_RETURN_IF_ERROR(
        PreCheckReadTensor(context, tflite_node, 2, &dummy_weights));

    int concat_tensor_idx = tflite_node->outputs->data[2];
    ABSL_RETURN_IF_ERROR(
        PreCheckReadValueByTensorIdx(context, concat_tensor_idx));

    int activ_tensor_idx = tflite_node->outputs->data[3];
    ABSL_RETURN_IF_ERROR(
        PreCheckReadValueByTensorIdx(context, activ_tensor_idx));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 4));
    ABSL_RETURN_IF_ERROR(PreCheckOutput(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutput(context, tflite_node, 1));
    return absl::OkStatus();
  }

  void ParseBasic(const TfLiteNode* tflite_node,
                  const TfLiteRegistration* registration,
                  ::ml_drift::GraphFloat32* graph, ObjectReader* reader,
                  const TfLiteLSTMParams* tf_options) {
    ::ml_drift::Node* concat_node = graph->NewNode();
    concat_node->operation.type = ToString(::ml_drift::OperationType::CONCAT);
    ::ml_drift::ConcatAttributes concat_attr;
    concat_attr.axis = ::ml_drift::Axis::CHANNELS;
    concat_node->operation.attributes = concat_attr;

    ::ml_drift::Node* fc_node = graph->NewNode();
    fc_node->operation.type =
        ToString(::ml_drift::OperationType::FULLY_CONNECTED);
    ::ml_drift::FullyConnectedAttributes fc_attr = GetFullyConnectedAttributes(
        /*weights_node_input_index=*/2,
        /*bias_node_input_index=*/3, reader);
    fc_node->operation.attributes = std::move(fc_attr);

    ::ml_drift::Node* lstm_node = graph->NewNode();
    lstm_node->operation.type = ToString(::ml_drift::OperationType::LSTM);
    ::ml_drift::LstmAttributes lstm_attr;
    lstm_attr.kernel_type = ::ml_drift::LstmKernelType::BASIC;
    lstm_node->operation.attributes = lstm_attr;

    ::ml_drift::Value* concat_temp =
        reader->ReadValueByTensorIdx(tflite_node->outputs->data[2]);
    ::ml_drift::Value* activ_temp =
        reader->ReadValueByTensorIdx(tflite_node->outputs->data[3]);

    reader->AddInput(concat_node, 0);  // input
    reader->AddInput(concat_node, 1);  // prev_activ
    graph->SetProducer(concat_node->id, concat_temp->id);

    graph->AddConsumer(fc_node->id, concat_temp->id);
    graph->SetProducer(fc_node->id, activ_temp->id);

    graph->AddConsumer(lstm_node->id, activ_temp->id);
    reader->AddInput(lstm_node, 4);   // prev_state
    reader->AddOutput(lstm_node, 1);  // new_state
    reader->AddOutput(lstm_node, 0);  // activation
  }

  absl::Status CheckBasicParameters(const TfLiteLSTMParams* tf_options) {
    if (tf_options->activation != kTfLiteActTanh) {
      return absl::UnimplementedError("Only TANH activation is supported.");
    }
    if (tf_options->cell_clip != 0.0f) {
      return absl::UnimplementedError("cell_clip is not supported.");
    }
    if (tf_options->proj_clip != 0.0f) {
      return absl::UnimplementedError("proj_clip is not supported.");
    }
    return absl::OkStatus();
  }

  absl::Status CheckFullyConnected(const TfLiteContext* context,
                                   const TfLiteNode* tflite_node,
                                   int weight_tensor_id) {
    const TfLiteTensor* weights_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node,
                                           weight_tensor_id, &weights_tensor));
    TfLiteAffineQuantization* quant_params =
        static_cast<TfLiteAffineQuantization*>(
            weights_tensor->quantization.params);
    if (weights_tensor->type == kTfLiteInt8 && quant_params->scale->size == 1) {
      int tensor_id;
      ABSL_RETURN_IF_ERROR(
          GetTensorId(context, tflite_node, weight_tensor_id, &tensor_id));
    } else {
      ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32>
          dummy_weights;
      ABSL_RETURN_IF_ERROR(PreCheckReadTensor(
          context, tflite_node, weight_tensor_id, &dummy_weights));
    }
    return absl::OkStatus();
  }

  absl::Status CheckBuildLstmGate(const TfLiteContext* context,
                                  const TfLiteNode* tflite_node,
                                  int input_weight_id, int recurrent_weight_id,
                                  int cell_weight_id, int bias_id,
                                  int normalization_weight_id,
                                  const TfLiteFusedActivation activation,
                                  bool has_peephole, bool has_normalization) {
    ABSL_RETURN_IF_ERROR(
        CheckFullyConnected(context, tflite_node, input_weight_id));
    ABSL_RETURN_IF_ERROR(
        CheckFullyConnected(context, tflite_node, recurrent_weight_id));

    if (has_peephole) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
          dummy_weights;
      ABSL_RETURN_IF_ERROR(PreCheckReadTensor(context, tflite_node,
                                              cell_weight_id, &dummy_weights));
    }

    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
        dummy_norm_weights;
    ABSL_RETURN_IF_ERROR(PreCheckReadTensor(
        context, tflite_node, normalization_weight_id, &dummy_norm_weights));

    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
        dummy_bias;
    ABSL_RETURN_IF_ERROR(
        PreCheckReadTensor(context, tflite_node, bias_id, &dummy_bias));

    ABSL_RETURN_IF_ERROR(PreCheckMaybeFuseActivation(tflite_node, activation));

    return absl::OkStatus();
  }

  absl::Status CheckFull(const TfLiteContext* context,
                         const TfLiteNode* tflite_node,
                         const TfLiteRegistration* registration,
                         const TfLiteLSTMParams* params) {
    // checkings extracted from ParseLSTMAttributes
    const bool has_cifg = HasCifg(tflite_node);
    const bool has_peephole = HasPeephole(tflite_node);
    const bool has_normalization = HasNormalization(tflite_node);

    // Since value can't be pre-read here (tensor_to_value and
    // quant_conversion_map are required).
    // Therefore Batched execution can't be checked here.
    // Batched LSTM operation will error quitely in ParseFull.

    ABSL_RETURN_IF_ERROR(
        PreCheckReadValue(context, tflite_node,
                          tflite::ops::builtin::lstm::full::kCellStateTensor));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(
        context, tflite_node,
        tflite::ops::builtin::lstm::full::kOutputStateTensor));

    ABSL_RETURN_IF_ERROR(CheckBuildLstmGate(
        context, tflite_node,
        tflite::ops::builtin::lstm::full::kInputToForgetWeightsTensor,
        tflite::ops::builtin::lstm::full::kRecurrentToForgetWeightsTensor,
        tflite::ops::builtin::lstm::full::kCellToForgetWeightsTensor,
        tflite::ops::builtin::lstm::full::kForgetGateBiasTensor,
        tflite::ops::builtin::lstm::full::kForgetLayerNormCoefficientsTensor,
        kTfLiteActSigmoid, has_peephole, has_normalization));

    ABSL_RETURN_IF_ERROR(CheckBuildLstmGate(
        context, tflite_node,
        tflite::ops::builtin::lstm::full::kInputToForgetWeightsTensor,
        tflite::ops::builtin::lstm::full::kRecurrentToForgetWeightsTensor,
        tflite::ops::builtin::lstm::full::kCellToForgetWeightsTensor,
        tflite::ops::builtin::lstm::full::kForgetGateBiasTensor,
        tflite::ops::builtin::lstm::full::kForgetLayerNormCoefficientsTensor,
        kTfLiteActSigmoid, has_peephole, has_normalization));

    if (!has_cifg) {
      ABSL_RETURN_IF_ERROR(CheckBuildLstmGate(
          context, tflite_node,
          tflite::ops::builtin::lstm::full::kInputToInputWeightsTensor,
          tflite::ops::builtin::lstm::full::kRecurrentToInputWeightsTensor,
          tflite::ops::builtin::lstm::full::kCellToInputWeightsTensor,
          tflite::ops::builtin::lstm::full::kInputGateBiasTensor,
          tflite::ops::builtin::lstm::full::kInputLayerNormCoefficientsTensor,
          kTfLiteActSigmoid, has_peephole, has_normalization));
    }

    ABSL_RETURN_IF_ERROR(CheckBuildLstmGate(
        context, tflite_node,
        tflite::ops::builtin::lstm::full::kInputToCellWeightsTensor,
        tflite::ops::builtin::lstm::full::kRecurrentToCellWeightsTensor,
        /*cell_weight_id=*/-1,
        tflite::ops::builtin::lstm::full::kCellGateBiasTensor,
        tflite::ops::builtin::lstm::full::kCellLayerNormCoefficientsTensor,
        params->activation, /*has_peephole=*/false, has_normalization));

    ABSL_RETURN_IF_ERROR(CheckBuildLstmGate(
        context, tflite_node,
        tflite::ops::builtin::lstm::full::kInputToOutputWeightsTensor,
        tflite::ops::builtin::lstm::full::kRecurrentToOutputWeightsTensor,
        tflite::ops::builtin::lstm::full::kCellToOutputWeightsTensor,
        tflite::ops::builtin::lstm::full::kOutputGateBiasTensor,
        tflite::ops::builtin::lstm::full::kOutputLayerNormCoefficientsTensor,
        kTfLiteActSigmoid, has_peephole, has_normalization));

    ABSL_RETURN_IF_ERROR(PreCheckOutput(
        context, tflite_node, tflite::ops::builtin::lstm::full::kOutputTensor));

    if (params->activation != kTfLiteActSigmoid &&
        params->activation != kTfLiteActTanh) {
      return absl::InvalidArgumentError(
          "Only sigmoid or tanh activation is supported.");
    }
    return absl::OkStatus();
  }

  // Helper functions copied from lstm_parser.cc
  inline bool HasTensor(const TfLiteNode* node, const int index) const {
    return index < node->inputs->size &&
           node->inputs->data[index] != kTfLiteOptionalTensor;
  }

  inline bool HasCifg(const TfLiteNode* node) const {
    return !HasTensor(
        node, tflite::ops::builtin::lstm::full::kInputToInputWeightsTensor);
  }

  inline bool HasPeephole(const TfLiteNode* node) const {
    return HasTensor(
        node, tflite::ops::builtin::lstm::full::kCellToForgetWeightsTensor);
  }

  inline bool HasNormalization(const TfLiteNode* node) const {
    return HasTensor(
        node,
        tflite::ops::builtin::lstm::full::kForgetLayerNormCoefficientsTensor);
  }

  inline bool HasProjection(const TfLiteNode* node) const {
    return HasTensor(
        node, tflite::ops::builtin::lstm::full::kProjectionWeightsTensor);
  }

  absl::flat_hash_map<int, ::ml_drift::ValueId> new_variable_input_value_map_;
};

class OneHotOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 1}));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    ::ml_drift::OneHotAttributes attr;
    const TfLiteTensor* on_tensor = reader->GetInputTensor(2);
    const TfLiteTensor* off_tensor = reader->GetInputTensor(3);
    attr.on_value = tflite::GetTensorData<float>(on_tensor)[0];
    attr.off_value = tflite::GetTensorData<float>(off_tensor)[0];
    node->operation.type = ToString(::ml_drift::OperationType::ONE_HOT);
    node->operation.attributes = std::move(attr);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);
  }
};

class PackOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));

    if (tflite_node->inputs->size == 1) {
      ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
      ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    } else {
      const auto* params =
          static_cast<const TfLitePackParams*>(tflite_node->builtin_data);
      if (!params) {
        return absl::InvalidArgumentError("Missing TfLitePackParams.");
      }

      ABSL_RETURN_IF_ERROR(
          ValidateSupport(context, tflite_node, registration, {}));

      for (uint32_t idx = 0; idx < tflite_node->inputs->size; ++idx) {
        ABSL_RETURN_IF_ERROR(
            PreCheckRuntimeOrConstantInput(context, tflite_node, idx));
      }

      TfLiteTensor* output = nullptr;
      ABSL_RETURN_IF_ERROR(
          PreGetOutputTensor(context, tflite_node, 0, &output));
      ABSL_RETURN_IF_ERROR(PreCheckAxisFromIndex(*output, params->axis));
      ABSL_RETURN_IF_ERROR(PreCheckTensorShape(*output));
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    if (tflite_node->inputs->size == 1) {
      // Pack with single input can be replaced with Reshape
      ::ml_drift::Node* node = graph->NewNode();
      node->operation.type = ToString(::ml_drift::OperationType::RESHAPE);
      reader->AddInput(node, 0);
      reader->AddOutputs(node);
      // New shape comes from output shape.
      ::ml_drift::ReshapeAttributes attr;
      attr.new_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
      node->operation.attributes = attr;
      return;
    } else {
      // Pack with few inputs can be replaced with Concat
      const auto* params =
          static_cast<const TfLitePackParams*>(tflite_node->builtin_data);

      // Read inputs first to make sure const node is added to a graph before
      // concat node to ensure topological order.
      std::vector<const ::ml_drift::Value*> inputs;
      inputs.reserve(tflite_node->inputs->size);
      for (uint32_t idx = 0; idx < tflite_node->inputs->size; ++idx) {
        ::ml_drift::Value* input =
            reader->IsConstantTensor(idx)
                ? reader->AddConstInput(idx, /*layout=*/{})
                : reader->ReadValue(idx);
        inputs.push_back(input);
      }

      const TfLiteTensor* output = reader->GetOutputTensor(0);
      ::ml_drift::ConcatAttributes attr;
      attr.axis = ExtractAxisFromIndex(*output, params->axis);
      const ::ml_drift::BHWC output_shape = ExtractTensorShape(output);
      ::ml_drift::BHWC input_required_shape = output_shape;
      input_required_shape.set(attr.axis, 1);
      for (int i = 0; i < inputs.size(); ++i) {
        ::ml_drift::BHWC input_shape = inputs[i]->tensor.shape;
        if (input_shape != input_required_shape) {
          // GPU delegates does not support implicit shapes transformations
          // adding explicit Reshape
          ::ml_drift::Node* node_reshape = graph->NewNode();
          node_reshape->operation.type =
              ToString(::ml_drift::OperationType::RESHAPE);
          ::ml_drift::ReshapeAttributes reshape_attr;
          reshape_attr.new_shape = input_required_shape;
          node_reshape->operation.attributes = reshape_attr;
          graph->AddConsumer(node_reshape->id, inputs[i]->id);
          ::ml_drift::Value* copy_value = graph->NewValue();
          copy_value->tensor.type = inputs[i]->tensor.type;
          copy_value->tensor.shape = input_required_shape;
          graph->SetProducer(node_reshape->id, copy_value->id);
          inputs[i] = copy_value;
        }
      }

      ::ml_drift::Node* node = graph->NewNode();
      node->operation.type = ToString(::ml_drift::OperationType::CONCAT);
      reader->AddOutputs(node);
      for (const ::ml_drift::Value* input : inputs) {
        graph->AddConsumer(node->id, input->id);
      }
      node->operation.attributes = attr;
      return;
    }
  }
};

class PReLUOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));

    const TfLiteTensor* input = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    const ::ml_drift::BHWC input_shape = ExtractTensorShape(input);

    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
        linear_alpha;
    absl::Status status = PreReadTensor(context, tflite_node, 1, &linear_alpha,
                                        ReadTensorFlags::kNoExtraBytes);
    if (status.ok()) {
      if (linear_alpha.shape.v != input_shape.c) {
        return absl::InvalidArgumentError(
            "Linear alpha shape does not match the number of input channels.");
      }
    } else {
      ::ml_drift::Tensor<::ml_drift::HWC, ::ml_drift::DataType::FLOAT32>
          hwc_alpha;
      ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, &hwc_alpha,
                                         ReadTensorFlags::kNoExtraBytes));
      if (hwc_alpha.shape.h != input_shape.h ||
          hwc_alpha.shape.w != input_shape.w ||
          hwc_alpha.shape.c != input_shape.c) {
        return absl::InvalidArgumentError(
            "Alpha shape does not match input shape.");
      }
    }

    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::PRELU);
    reader->AddInput(node, 0);

    ::ml_drift::PReLUAttributes attr;
    if (reader->IsLinearTensor(1)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> t;
      reader->ReadTensor(1, &t, ReadTensorFlags::kNoExtraBytes);
      attr.alpha = std::move(t);
    } else {
      ::ml_drift::Tensor<::ml_drift::HWC, ::ml_drift::DataType::FLOAT32> t;
      reader->ReadTensor(1, &t, ReadTensorFlags::kNoExtraBytes);
      attr.alpha = std::move(t);
    }
    node->operation.attributes = std::move(attr);
    reader->AddOutputs(node);
  }
};

class PadOperationParser : public TFLiteOperationParser {
 public:
  explicit PadOperationParser(bool mirror_pad,
                              const ModelBuilderOptions& options)
      : mirror_pad_(mirror_pad), options_(options) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 5}));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));

    ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::INT32> paddings;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, &paddings,
                                       ReadTensorFlags::kNoExtraBytes));

    if (registration->builtin_code == kTfLiteBuiltinPadv2 &&
        tflite_node->inputs->size == 3) {
      ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::FLOAT32>
          dummy_const_tensor;
      ABSL_RETURN_IF_ERROR(
          PreCheckReadTensor(context, tflite_node, 2, &dummy_const_tensor));
    }

    if (!(paddings.shape.h == 4 && paddings.shape.w == 2) &&
        !(paddings.shape.h == 3 && paddings.shape.w == 2)) {
      // It shouldn't fail here since it's checked at IsSupported().
      return absl::InvalidArgumentError(
          "Paddings tensor has unexpected shape.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::PAD);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::PadAttributes attr;
    if (mirror_pad_) {
      attr.type = ::ml_drift::PaddingContentType::REFLECT;
    } else /*zero pad*/ {
      attr.type = ::ml_drift::PaddingContentType::ZEROS;
    }

    ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::INT32> paddings;
    reader->ReadTensor(1, &paddings, ReadTensorFlags::kNoExtraBytes);

    if (registration->builtin_code == kTfLiteBuiltinPadv2 &&
        tflite_node->inputs->size == 3) {
      ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::FLOAT32>
          const_tensor;
      reader->ReadTensor(2, &const_tensor, ReadTensorFlags::kNoExtraBytes);
      attr.constant_values = const_tensor.data[0];
      if (options_.enable_infinite_float_capping) {
        const bool input_is_fp16 =
            reader->GetInputTensor(0)->type == kTfLiteFloat16;
        const bool use_half =
            options_.enable_reduced_precision || input_is_fp16;
        if (attr.constant_values == std::numeric_limits<float>::infinity()) {
          attr.constant_values = use_half ? ::ml_drift::kMaxHalf
                                          : std::numeric_limits<float>::max();
        } else if (attr.constant_values ==
                   -std::numeric_limits<float>::infinity()) {
          attr.constant_values = use_half
                                     ? -::ml_drift::kMaxHalf
                                     : std::numeric_limits<float>::lowest();
        }
      }
    }

    if (paddings.shape.h == 4 && paddings.shape.w == 2) {
      // 4x2 tensor with paddings.
      attr.prepended = ::ml_drift::BHWC(paddings.data[0], paddings.data[2],
                                        paddings.data[4], paddings.data[6]);
      attr.appended = ::ml_drift::BHWC(paddings.data[1], paddings.data[3],
                                       paddings.data[5], paddings.data[7]);
    } else if (paddings.shape.h == 3 && paddings.shape.w == 2) {
      // 3x2 tensor with paddings.
      attr.prepended = ::ml_drift::BHWC(1, paddings.data[0], paddings.data[2],
                                        paddings.data[4]);
      attr.appended = ::ml_drift::BHWC(1, paddings.data[1], paddings.data[3],
                                       paddings.data[5]);
    }
    node->operation.attributes = attr;
  }

 private:
  bool mirror_pad_ = false;
  ModelBuilderOptions options_;
};

class Pooling2DOperationParser : public TFLiteOperationParser {
 public:
  explicit Pooling2DOperationParser(::ml_drift::PoolingType type)
      : type_(type) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 2}));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    const TfLitePoolParams* params =
        static_cast<const TfLitePoolParams*>(tflite_node->custom_initial_data);
    if (!params) {
      params = static_cast<const TfLitePoolParams*>(tflite_node->builtin_data);
    }
    if (!params) return absl::InvalidArgumentError("Missing TfLitePoolParams.");
    ABSL_RETURN_IF_ERROR(
        PreCheckMaybeFuseActivationSkipSize(tflite_node, params->activation));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::POOLING_2D);
    reader->AddInput(node, 0);
    reader->AddOutput(node, 0);

    ::ml_drift::Pooling2DAttributes attr;
    attr.type = type_;

    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;

    // Check whether there are custom options encoded. It happens if operation
    // is MaxPoolingWithArgmax2D. There is no way to read
    // tflite_node->builtin_code, so, simply check whether custom data is
    // available.
    const TfLitePoolParams* params =
        static_cast<const TfLitePoolParams*>(tflite_node->custom_initial_data);
    if (!params) {
      params = static_cast<const TfLitePoolParams*>(tflite_node->builtin_data);
    }

    HandleFusedActivation(params->activation, graph, node);
    // Second output is optional. It is not required, it but must be added after
    // MaybeAddFusedActivation function is called
    if (reader->IsNodeOutputTensorPresent(1)) reader->AddOutput(node, 1);

    // First output is the result of pooling operation, while second output is
    // indices used for pooling.
    auto outputs = graph->FindOutputs(node->id);
    attr.output_indices = outputs.size() == 2;
    if (attr.output_indices) {
      // Fix data type for output indices. In the model it is set as float32.
      outputs[1]->tensor.type = ::ml_drift::DataType::INT32;
    }
    attr.kernel = ToHW(params->filter_height, params->filter_width);
    attr.strides = ToHW(params->stride_height, params->stride_width);
    UpdatePadding(params->padding, input_shape, &attr);
    node->operation.attributes = attr;
  }

 private:
  const ::ml_drift::PoolingType type_;
};

class ReduceOperationParser : public TFLiteOperationParser {
 public:
  explicit ReduceOperationParser(::ml_drift::OperationType operation_type)
      : operation_type_(operation_type) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));

    const auto* params =
        static_cast<const TfLiteReducerParams*>(tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteReducerParams.");
    }
    const TfLiteTensor* input = nullptr;
    const TfLiteTensor* axes = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &axes));
    for (int i = 0; i < tflite::NumElements(axes->dims); i++) {
      ABSL_RETURN_IF_ERROR(PreCheckAxisFromIndex(*input, axes->data.i32[i]));
    }
    if (!params->keep_dims) {
      TfLiteTensor* output = nullptr;  // reader->GetOutputTensor(0);
      ABSL_RETURN_IF_ERROR(
          PreGetOutputTensor(context, tflite_node, 0, &output));
      ABSL_RETURN_IF_ERROR(PreCheckTensorShape(*output));
    }
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(operation_type_);
    reader->AddInput(node, 0);

    ::ml_drift::ReduceAttributes attr;
    const TfLiteTensor* input = reader->GetInputTensor(0);
    const TfLiteTensor* axes = reader->GetInputTensor(1);
    for (int i = 0; i < tflite::NumElements(axes->dims); i++) {
      ::ml_drift::Axis axis = ExtractAxisFromIndex(*input, axes->data.i32[i]);
      attr.dims.insert(axis);
    }
    node->operation.attributes = attr;

    const auto* params =
        static_cast<const TfLiteReducerParams*>(tflite_node->builtin_data);
    if (!params->keep_dims) {
      // GPU delegates does not support implicit shapes transformations
      // adding explicit Reshape
      const auto& input_tensor = graph->FindInputs(node->id)[0]->tensor;
      auto reduce_output_shape = input_tensor.shape;
      for (auto axis : attr.dims) {
        reduce_output_shape.set(axis, 1);
      }
      ::ml_drift::Node* node_reshape = graph->NewNode();
      node_reshape->operation.type =
          ToString(::ml_drift::OperationType::RESHAPE);
      ::ml_drift::ReshapeAttributes reshape_attr;
      const TfLiteTensor* output = reader->GetOutputTensor(0);
      reshape_attr.new_shape = ExtractTensorShape(output);
      node_reshape->operation.attributes = reshape_attr;
      ::ml_drift::Value* reduce_result = graph->NewValue();
      reduce_result->tensor.type = input_tensor.type;
      reduce_result->tensor.shape = reduce_output_shape;

      graph->SetProducer(node->id, reduce_result->id);
      graph->AddConsumer(node_reshape->id, reduce_result->id);
      reader->AddOutputs(node_reshape);
    } else {
      reader->AddOutputs(node);
    }
  }

 private:
  const ::ml_drift::OperationType operation_type_;
};

class QuantizeOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));

    TfLiteTensor* output = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetOutputTensor(context, tflite_node, 0, &output));
    if (output->quantization.params == nullptr) {
      return absl::InvalidArgumentError(
          "Encountered Quantize output with no quant params");
    }
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    // 'Quantize' is rewritten as QuantizeAndDequantize since we are dealing
    // with floating-point versions of the original tensors.
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::QUANTIZE_AND_DEQUANTIZE);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    // Quantization attributes should already be present in the output tensor.
    auto output_value = graph->FindOutputs(node->id)[0];
    ::ml_drift::QuantizeAndDequantizeAttributes attr;
    attr.min = output_value->quant_params.value().min;
    attr.max = output_value->quant_params.value().max;
    attr.scale = output_value->quant_params.value().scale;

    node->operation.attributes = attr;
  }
};

class ReLUOperationParser : public TFLiteOperationParser {
 public:
  explicit ReLUOperationParser(int activation_min, int activation_max)
      : activation_min_(activation_min), activation_max_(activation_max) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 2,
                                          .num_outputs = 1,
                                          .required_runtime_inputs = 1,
                                          .required_const_inputs = 0}));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::RELU);
    reader->AddInput(node, 0);

    ::ml_drift::ReLUAttributes attr;
    const auto* params =
        static_cast<const TfLiteLeakyReluParams*>(tflite_node->builtin_data);
    attr.alpha = params ? params->alpha : 0;
    attr.activation_min = activation_min_;
    attr.activation_max = activation_max_;
    node->operation.attributes = std::move(attr);
    reader->AddOutputs(node);
  }

 private:
  const int activation_min_;
  const int activation_max_;
};

class RemainderOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    // remainder(x, y) = x - d * y
    // if float: d = floor(x / y)
    // if int: d = x / y
    ::ml_drift::Node* div_node = graph->NewNode();
    div_node->operation.type = ToString(::ml_drift::OperationType::DIV);

    reader->AddInput(div_node, 0);
    reader->AddInput(div_node, 1);
    const auto input = graph->FindInputs(div_node->id)[0];

    ::ml_drift::Value* d = graph->NewValue();
    const ::ml_drift::DataType type = input->tensor.type;
    const ::ml_drift::BHWC shape = input->tensor.shape;
    d->tensor.type = type;
    d->tensor.shape = shape;
    graph->SetProducer(div_node->id, d->id);

    ::ml_drift::Node* mul_node = graph->NewNode();
    mul_node->operation.type = ToString(::ml_drift::OperationType::MUL);
    reader->AddInput(mul_node, 1);
    if (type == ::ml_drift::DataType::FLOAT16 ||
        type == ::ml_drift::DataType::FLOAT32) {
      ::ml_drift::Node* floor_node = graph->NewNode();
      floor_node->operation.type = ToString(::ml_drift::OperationType::FLOOR);
      graph->AddConsumer(floor_node->id, d->id);
      ::ml_drift::Value* floor_result = graph->NewValue();
      floor_result->tensor.type = type;
      floor_result->tensor.shape = shape;
      graph->SetProducer(floor_node->id, floor_result->id);
      graph->AddConsumer(mul_node->id, floor_result->id);
    } else {
      graph->AddConsumer(mul_node->id, d->id);
    }
    ::ml_drift::Value* mul_output = graph->NewValue();
    mul_output->tensor.type = type;
    mul_output->tensor.shape = shape;
    graph->SetProducer(mul_node->id, mul_output->id);

    ::ml_drift::Node* sub_node = graph->NewNode();
    sub_node->operation.type = ToString(::ml_drift::OperationType::SUB);
    reader->AddInput(sub_node, 0);
    graph->AddConsumer(sub_node->id, mul_output->id);
    reader->AddOutputs(sub_node);
  }
};

class ResamplerOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    reader->AddInput(node, 0);  // src
    reader->AddInput(node, 1);  // warp
    reader->AddOutputs(node);

    node->operation.type = ToString(::ml_drift::OperationType::RESAMPLER);

    auto src_shape = graph->FindInputs(node->id)[0]->tensor.shape;
    auto warp_shape = graph->FindInputs(node->id)[1]->tensor.shape;

    auto output_value = graph->FindOutputs(node->id)[0];
    output_value->tensor.shape =
        ::ml_drift::BHWC(src_shape.b, warp_shape.h, warp_shape.w, src_shape.c);
  }
};

class ReshapeOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    // TODO(eignasheva): add shape checking
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    // Here we may have extra inputs. Other tensors were supposed to
    // define new shape, but in TFLite these are ignored.
    // TODO(akulik): check that shapes match?

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::RESHAPE);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    // New shape comes from output shape.
    ::ml_drift::ReshapeAttributes attr;
    attr.new_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
    node->operation.attributes = attr;
  }
};

class Resize2DOperationParser : public TFLiteOperationParser {
 public:
  explicit Resize2DOperationParser(::ml_drift::SamplingType sampling_type)
      : sampling_type_(sampling_type) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 3));
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    if (sampling_type_ == ::ml_drift::SamplingType::BILINEAR) {
      const auto* params = static_cast<const TfLiteResizeBilinearParams*>(
          tflite_node->builtin_data);
      if (!params) {
        return absl::InvalidArgumentError(
            "Missing TfLiteResizeBilinearParams.");
      }
      if (params->align_corners && params->half_pixel_centers) {
        return absl::InternalError(
            "If half_pixel_centers is True, align_corners must be False.");
      }
    } else if (sampling_type_ == ::ml_drift::SamplingType::NEAREST) {
      if (!static_cast<const TfLiteResizeNearestNeighborParams*>(
              tflite_node->builtin_data)) {
        return absl::InvalidArgumentError(
            "Missing TfLiteResizeNearestNeighborParams.");
      }
    } else {
      return absl::UnimplementedError("Unsupported sampling type.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::RESIZE);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);
    // Here we may have extra inputs. Other tensors were supposed to
    // define new shape, but in TFLite these are ignored.

    ::ml_drift::Resize2DAttributes attr;
    if (sampling_type_ == ::ml_drift::SamplingType::BILINEAR) {
      const auto* params = static_cast<const TfLiteResizeBilinearParams*>(
          tflite_node->builtin_data);
      attr.align_corners = params->align_corners;
      attr.half_pixel_centers = params->half_pixel_centers;
    } else {
      const auto* params =
          static_cast<const TfLiteResizeNearestNeighborParams*>(
              tflite_node->builtin_data);
      attr.align_corners = params->align_corners;
      attr.half_pixel_centers = params->half_pixel_centers;
    }
    attr.type = sampling_type_;
    attr.new_shape.CopyAllDefinedAxis(
        graph->FindOutputs(node->id)[0]->tensor.shape);
    node->operation.attributes = std::move(attr);
  }

 private:
  ::ml_drift::SamplingType sampling_type_ = ::ml_drift::SamplingType::UNKNOWN;
};

class ReverseOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    const TfLiteTensor* input = nullptr;
    const TfLiteTensor* axes = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &axes));
    for (int i = 0; i < tflite::NumElements(axes->dims); i++) {
      ABSL_RETURN_IF_ERROR(PreCheckAxisFromIndex(*input, axes->data.i32[i]));
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::REVERSE);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::ReverseAttributes attr;
    const TfLiteTensor* input = reader->GetInputTensor(0);
    const TfLiteTensor* axes = reader->GetInputTensor(1);
    for (int i = 0; i < tflite::NumElements(axes->dims); i++) {
      ::ml_drift::Axis axis = ExtractAxisFromIndex(*input, axes->data.i32[i]);
      attr.axes.insert(axis);
    }
    node->operation.attributes = std::move(attr);
  }
};

class SelectV2OperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    ABSL_RETURN_IF_ERROR(CheckGpuDelegateCompatibility(
        tflite::GetOpSignature(context, tflite_node, registration)));

    ABSL_RETURN_IF_ERROR(
        PreCheckRuntimeOrConstantInput(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(
        PreCheckRuntimeOrConstantInput(context, tflite_node, 1));
    ABSL_RETURN_IF_ERROR(
        PreCheckRuntimeOrConstantInput(context, tflite_node, 2));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  absl::Status CheckGpuDelegateCompatibility(
      const tflite::OpSignature& op_sig) {
    if (op_sig.inputs.size() != 3 || op_sig.outputs.size() != 1) {
      return absl::InvalidArgumentError("Expected 3 inputs and 1 output");
    }
    // Only supports float inputs with non-broadcastable or scalar if/else.
    absl::Status error = absl::InvalidArgumentError(
        "Cond must be float or bool type, if, else tensors must be "
        "either be same the shape as output or constant, scalar.");
    if (op_sig.inputs.at(0).type != kTfLiteBool &&
        op_sig.inputs.at(0).type != kTfLiteFloat16 &&
        op_sig.inputs.at(0).type != kTfLiteFloat32) {
      return error;
    }
    std::vector<int32_t> output_dims = op_sig.outputs[0].dims;
    if (!op_sig.inputs.at(1).dims.empty() &&
        (op_sig.inputs.at(1).dims != output_dims) &&
        (op_sig.inputs.at(1).dims.size() > 1 ||
         op_sig.inputs.at(1).dims[0] > 1)) {
      return error;
    }
    if (!op_sig.inputs.at(2).dims.empty() &&
        (op_sig.inputs.at(2).dims != output_dims) &&
        (op_sig.inputs.at(2).dims.size() > 1 ||
         op_sig.inputs.at(2).dims[0] > 1)) {
      return error;
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::SelectV2Attributes attr;
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::SELECT_V2);

    {  // cond tensor
      constexpr int kIndex = 0;
      if (reader->IsConstantTensor(kIndex)) {
        const ::ml_drift::Value* input =
            reader->AddConstInput(kIndex, /*layout=*/{});
        graph->AddConsumer(node->id, input->id);
      } else {
        reader->AddInput(node, kIndex);
      }
    }

    // num_dims == 3; convert HWC to 1HWC for constant tensors
    const SizedLayout constants_layout = {
        /*layout_3d=*/::ml_drift::Layout::HWC};
    {  // then tensor
      constexpr int kIndex = 1;
      if (reader->IsConstantTensor(kIndex)) {
        const ::ml_drift::Value* input =
            reader->AddConstInput(kIndex, constants_layout);
        graph->AddConsumer(node->id, input->id);
      } else {
        reader->AddInput(node, kIndex);
      }
    }
    {  // else tensor
      constexpr int kIndex = 2;
      if (reader->IsConstantTensor(kIndex)) {
        const ::ml_drift::Value* input =
            reader->AddConstInput(kIndex, constants_layout);
        graph->AddConsumer(node->id, input->id);
      } else {
        reader->AddInput(node, kIndex);
      }
    }

    reader->AddOutputs(node);
    node->operation.attributes = std::move(attr);
  }
};

class SliceOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(
        context, tflite_node, registration,
        {.max_version = 8, .min_inputs = 3, .check_gpu_compatibility = false}));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));

    const TfLiteTensor* tfl_input = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, 0, &tfl_input));
    if (tflite::IsConstantTensor(tfl_input)) {
      return absl::InvalidArgumentError("Constant input is not supported.");
    }
    const int input_dims = tfl_input->dims->size;

    ::ml_drift::SliceAttributes attr;
    attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32> starts,
        sizes;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, &starts,
                                       ReadTensorFlags::kNoExtraBytes));
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 2, &sizes,
                                       ReadTensorFlags::kNoExtraBytes));
    if (starts.data.size() != sizes.data.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("The dimensionality of `starts` (", starts.data.size(),
                       "D) must match `sizes` (", sizes.data.size(), "D)."));
    }
    ::ml_drift::BHWC bhwc_starts(0, 0, 0, 0);
    ::ml_drift::BHWC bhwc_sizes = ExtractTensorShape(tfl_input);
    if (input_dims == 4) {
      // input in ::ml_drift::BHWC layout
      if (starts.data.size() == 4) {
        bhwc_starts.b = starts.data[0];
        bhwc_starts.h = starts.data[1];
        bhwc_starts.w = starts.data[2];
        bhwc_starts.c = starts.data[3];
        bhwc_sizes.b = sizes.data[0];
        bhwc_sizes.h = sizes.data[1];
        bhwc_sizes.w = sizes.data[2];
        bhwc_sizes.c = sizes.data[3];
      } else if (starts.data.size() == 3) {
        // if input is 4D(::ml_drift::BHWC) and args 3D, we assume that args in
        // HWC layout
        bhwc_starts.h = starts.data[0];
        bhwc_starts.w = starts.data[1];
        bhwc_starts.c = starts.data[2];
        bhwc_sizes.h = sizes.data[0];
        bhwc_sizes.w = sizes.data[1];
        bhwc_sizes.c = sizes.data[2];
      } else {
        return absl::UnimplementedError(
            "Slicing starts count must be 3 or 4 for 4d input");
      }
    } else if (input_dims == 3) {
      // input in BWC layout
      if (starts.data.size() == 3) {
        bhwc_starts.b = starts.data[0];
        bhwc_starts.w = starts.data[1];
        bhwc_starts.c = starts.data[2];
        bhwc_sizes.b = sizes.data[0];
        bhwc_sizes.w = sizes.data[1];
        bhwc_sizes.c = sizes.data[2];
      } else {
        return absl::UnimplementedError(
            "Slicing starts count must be 3 for 3d input");
      }
    } else if (input_dims == 2) {
      if (starts.data.size() == 2) {
        bhwc_starts.b = starts.data[0];
        bhwc_starts.c = starts.data[1];
        bhwc_sizes.b = sizes.data[0];
        bhwc_sizes.c = sizes.data[1];
      } else {
        return absl::UnimplementedError(
            "Slicing starts count must be 2 for 2d input");
      }
    } else if (input_dims == 1) {
      if (starts.data.size() == 1) {
        bhwc_starts.b = starts.data[0];
        bhwc_sizes.b = sizes.data[0];
      } else {
        return absl::UnimplementedError(
            "Slicing starts count must be 1 for 1d input");
      }
    } else {
      return absl::UnimplementedError(
          "Slicing is unsupported for input with rank > 4");
    }
    const ::ml_drift::BHWC in_shape = ExtractTensorShape(tfl_input);
    if (bhwc_sizes.b == -1) bhwc_sizes.b = in_shape.b - bhwc_starts.b;
    if (bhwc_sizes.h == -1) bhwc_sizes.h = in_shape.h - bhwc_starts.h;
    if (bhwc_sizes.w == -1) bhwc_sizes.w = in_shape.w - bhwc_starts.w;
    if (bhwc_sizes.c == -1) bhwc_sizes.c = in_shape.c - bhwc_starts.c;
    attr.starts = bhwc_starts;
    attr.ends = ::ml_drift::BHWC(
        bhwc_starts.b + bhwc_sizes.b, bhwc_starts.h + bhwc_sizes.h,
        bhwc_starts.w + bhwc_sizes.w, bhwc_starts.c + bhwc_sizes.c);
    UpdateIfNegative(in_shape, &attr);

    TfLiteTensor* tfl_output = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetOutputTensor(context, tflite_node, 0, &tfl_output));
    const ::ml_drift::BHWC out_shape = ExtractTensorShape(tfl_output);
    if ((attr.ends.b - attr.starts.b) != out_shape.b) {
      return absl::UnimplementedError("Output batch don't match");
    }
    if ((attr.ends.h - attr.starts.h) != out_shape.h) {
      return absl::UnimplementedError("Output height doesn't match");
    }
    if ((attr.ends.w - attr.starts.w) != out_shape.w) {
      return absl::UnimplementedError("Output width doesn't match");
    }
    if ((attr.ends.c - attr.starts.c) != out_shape.c) {
      return absl::UnimplementedError("Output channels don't match");
    }

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::SLICE);
    reader->AddOutputs(node);
    ::ml_drift::Value* input = reader->ReadValue(0);
    graph->AddConsumer(node->id, input->id);

    const TfLiteTensor* tfl_input = reader->GetInputTensor(0);
    const int input_dims = tfl_input->dims->size;

    ::ml_drift::SliceAttributes attr;
    attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32> starts,
        sizes;
    reader->ReadTensor(1, &starts, ReadTensorFlags::kNoExtraBytes);
    reader->ReadTensor(2, &sizes, ReadTensorFlags::kNoExtraBytes);

    ::ml_drift::BHWC bhwc_starts(0, 0, 0, 0);
    ::ml_drift::BHWC bhwc_sizes = input->tensor.shape;
    if (input_dims == 4) {
      // input in ::ml_drift::BHWC layout
      if (starts.data.size() == 4) {
        bhwc_starts.b = starts.data[0];
        bhwc_starts.h = starts.data[1];
        bhwc_starts.w = starts.data[2];
        bhwc_starts.c = starts.data[3];
        bhwc_sizes.b = sizes.data[0];
        bhwc_sizes.h = sizes.data[1];
        bhwc_sizes.w = sizes.data[2];
        bhwc_sizes.c = sizes.data[3];
      } else if (starts.data.size() == 3) {
        // if input is 4D(::ml_drift::BHWC) and args 3D, we assume that args in
        // HWC layout
        bhwc_starts.h = starts.data[0];
        bhwc_starts.w = starts.data[1];
        bhwc_starts.c = starts.data[2];
        bhwc_sizes.h = sizes.data[0];
        bhwc_sizes.w = sizes.data[1];
        bhwc_sizes.c = sizes.data[2];
      }
    } else if (input_dims == 3) {
      // input in BWC layout
      if (starts.data.size() == 3) {
        bhwc_starts.b = starts.data[0];
        bhwc_starts.w = starts.data[1];
        bhwc_starts.c = starts.data[2];
        bhwc_sizes.b = sizes.data[0];
        bhwc_sizes.w = sizes.data[1];
        bhwc_sizes.c = sizes.data[2];
      }
    } else if (input_dims == 2) {
      if (starts.data.size() == 2) {
        bhwc_starts.b = starts.data[0];
        bhwc_starts.c = starts.data[1];
        bhwc_sizes.b = sizes.data[0];
        bhwc_sizes.c = sizes.data[1];
      }
    } else if (input_dims == 1) {
      if (starts.data.size() == 1) {
        bhwc_starts.b = starts.data[0];
        bhwc_sizes.b = sizes.data[0];
      }
    }
    const auto& in_shape = input->tensor.shape;
    if (bhwc_sizes.b == -1) {
      bhwc_sizes.b = in_shape.b - bhwc_starts.b;
    }
    if (bhwc_sizes.h == -1) {
      bhwc_sizes.h = in_shape.h - bhwc_starts.h;
    }
    if (bhwc_sizes.w == -1) {
      bhwc_sizes.w = in_shape.w - bhwc_starts.w;
    }
    if (bhwc_sizes.c == -1) {
      bhwc_sizes.c = in_shape.c - bhwc_starts.c;
    }
    attr.starts = bhwc_starts;
    attr.ends = ::ml_drift::BHWC(
        bhwc_starts.b + bhwc_sizes.b, bhwc_starts.h + bhwc_sizes.h,
        bhwc_starts.w + bhwc_sizes.w, bhwc_starts.c + bhwc_sizes.c);
    UpdateIfNegative(in_shape, &attr);
    node->operation.attributes = std::move(attr);
  }

 private:
  static void UpdateIfNegative(const ::ml_drift::BHWC& input_shape,
                               ::ml_drift::SliceAttributes* attr) {
    if (attr->ends.b < 0) attr->ends.b += input_shape.b;
    if (attr->ends.h < 0) attr->ends.h += input_shape.h;
    if (attr->ends.w < 0) attr->ends.w += input_shape.w;
    if (attr->ends.c < 0) attr->ends.c += input_shape.c;
  }
};

class SoftmaxOperationParser : public TFLiteOperationParser {
 public:
  explicit SoftmaxOperationParser(const ModelBuilderOptions& options)
      : options_(options) {}
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 4,
                                          .num_outputs = 1,
                                          .required_runtime_inputs = 1,
                                          .required_const_inputs = 0}));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));

    const auto* params =
        static_cast<const TfLiteSoftmaxParams*>(tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteSoftmaxParams.");
    }
    if (params->beta != 1) {
      // there is multiply by scalar operation fused in softmax. Make a layer
      // out of it before softmax.
      return absl::UnimplementedError("Softmax.beta != 1 is not supported.");
      // auto mul_node = reader->NewPassthroughNode(node);
      // mul_node->operation.type = ToString(OperationType::MUL);
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* softmax_node = nullptr;
    if (options_.enable_infinite_float_capping) {
      const bool input_is_fp16 =
          reader->GetInputTensor(0)->type == kTfLiteFloat16;
      const bool use_half = options_.enable_reduced_precision || input_is_fp16;
      const float cap_value =
          use_half ? ::ml_drift::kMaxHalf : std::numeric_limits<float>::max();
      ::ml_drift::Node* max_node = graph->NewNode();
      max_node->operation.type = ToString(::ml_drift::OperationType::MAXIMUM);
      max_node->operation.attributes =
          ::ml_drift::ElementwiseAttributes{/*param=*/-cap_value};
      reader->AddInput(max_node, 0);
      ::ml_drift::Value* max_input_value = graph->FindInputs(max_node->id)[0];
      ::ml_drift::Value* max_output_value = graph->NewValue();
      max_output_value->tensor.type = max_input_value->tensor.type;
      max_output_value->tensor.shape = max_input_value->tensor.shape;
      graph->SetProducer(max_node->id, max_output_value->id);

      ::ml_drift::Node* min_node = graph->NewNode();
      min_node->operation.type = ToString(::ml_drift::OperationType::MINIMUM);
      min_node->operation.attributes =
          ::ml_drift::ElementwiseAttributes{/*param=*/cap_value};
      graph->AddConsumer(min_node->id, max_output_value->id);
      ::ml_drift::Value* min_output_value = graph->NewValue();
      min_output_value->tensor.type = max_output_value->tensor.type;
      min_output_value->tensor.shape = max_output_value->tensor.shape;
      graph->SetProducer(min_node->id, min_output_value->id);

      softmax_node = graph->NewNode();
      softmax_node->operation.type =
          ToString(::ml_drift::OperationType::SOFTMAX);
      graph->AddConsumer(softmax_node->id, min_output_value->id);
      reader->AddOutputs(softmax_node);
    } else {
      softmax_node = graph->NewNode();
      softmax_node->operation.type =
          ToString(::ml_drift::OperationType::SOFTMAX);
      reader->AddInput(softmax_node, 0);
      reader->AddOutputs(softmax_node);
    }
    ::ml_drift::SoftmaxAttributes attr;
    attr.axis = ::ml_drift::Axis::CHANNELS;  // always by channels
    softmax_node->operation.attributes = attr;
  }

 private:
  ModelBuilderOptions options_;
};

class SpaceToDepthOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(context, tflite_node, registration,
                                         {.max_version = 2,
                                          .num_outputs = 1,
                                          .required_runtime_inputs = 1,
                                          .required_const_inputs = 0}));
    // TODO(impjdi): Dims check.
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    const auto* params =
        static_cast<const TfLiteSpaceToDepthParams*>(tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteSpaceToDepthParams.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::SPACE_TO_DEPTH);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);
    const auto* params =
        static_cast<const TfLiteSpaceToDepthParams*>(tflite_node->builtin_data);
    ::ml_drift::SpaceToDepthAttributes attr;
    attr.block_size = params->block_size;
    node->operation.attributes = attr;
  }
};

class SplitOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    const auto* params =
        static_cast<const TfLiteSplitParams*>(tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteSplitParams.");
    }
    if (params->num_splits == 1) {
      ABSL_RETURN_IF_ERROR(
          ValidateSupport(context, tflite_node, registration, {}));
      ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
      return absl::OkStatus();
    }
    const TfLiteTensor* input = nullptr;
    const TfLiteTensor* axis_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &input));
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, 0, &axis_tensor));
    ABSL_RETURN_IF_ERROR(
        PreCheckAxisFromIndex(*input, axis_tensor->data.i32[0]));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    for (int i = 0; i < tflite_node->outputs->size; ++i) {
      ABSL_RETURN_IF_ERROR(PreCheckOutput(context, tflite_node, i));
    }

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    const auto* params =
        static_cast<const TfLiteSplitParams*>(tflite_node->builtin_data);
    if (params->num_splits == 1) {
      // Adding Identity reshape that will be removed.
      ::ml_drift::Node* node = graph->NewNode();
      node->operation.type = ToString(::ml_drift::OperationType::RESHAPE);
      reader->AddInput(node, 0);
      reader->AddOutputs(node);
      // New shape comes from output shape.
      ::ml_drift::ReshapeAttributes attr;
      attr.new_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
      node->operation.attributes = attr;
      return;
    }
    const TfLiteTensor* input = reader->GetInputTensor(1);
    const TfLiteTensor* axis_tensor = reader->GetInputTensor(0);
    ::ml_drift::SplitAttributes attr;
    attr.axis = ExtractAxisFromIndex(*input, axis_tensor->data.i32[0]);

    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::SPLIT);
    node->operation.attributes = attr;
    reader->AddInput(node, 1);
    reader->AddOutputs(node);
  }
};

class SplitVOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(ValidateSupport(
        context, tflite_node, registration,
        {.required_runtime_inputs = 1, .required_const_inputs = 2}));
    const auto* params =
        static_cast<const TfLiteSplitVParams*>(tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteSplitVParams.");
    }

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));

    if (params->num_splits == 1) {
      return absl::OkStatus();
    }

    const TfLiteTensor* input = nullptr;
    const TfLiteTensor* axis_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, 2, &axis_tensor));
    ABSL_RETURN_IF_ERROR(
        PreCheckAxisFromIndex(*input, axis_tensor->data.i32[0]));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    const auto* params =
        static_cast<const TfLiteSplitVParams*>(tflite_node->builtin_data);
    if (params->num_splits == 1) {
      // Adding Identity reshape that will be removed.
      ::ml_drift::Node* node = graph->NewNode();
      node->operation.type = ToString(::ml_drift::OperationType::RESHAPE);
      reader->AddInput(node, 0);
      reader->AddOutputs(node);
      // New shape comes from output shape.
      ::ml_drift::ReshapeAttributes attr;
      attr.new_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
      node->operation.attributes = attr;
      return;
    }
    const TfLiteTensor* input = reader->GetInputTensor(0);
    const TfLiteTensor* axis_tensor = reader->GetInputTensor(2);
    ::ml_drift::SplitAttributes attr;
    attr.axis = ExtractAxisFromIndex(*input, axis_tensor->data.i32[0]);

    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::SPLIT);
    node->operation.attributes = attr;
    reader->AddInput(node, 0);
    reader->AddOutputs(node);
  }
};

class StridedSliceOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    ABSL_RETURN_IF_ERROR(CheckGpuDelegateCompatibility(
        tflite_node,
        tflite::GetOpSignature(context, tflite_node, registration)));

    const TfLiteTensor* input_tensor;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, 0, &input_tensor));
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>
        starts_sizes;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, &starts_sizes,
                                       ReadTensorFlags::kNoExtraBytes));

    const bool read_attr_without_batch =
        input_tensor->dims->size == starts_sizes.data.size() + 1;
    const bool read_attr_with_batch =
        input_tensor->dims->size == starts_sizes.data.size();
    if (!read_attr_without_batch && !read_attr_with_batch) {
      // Error: Must be catched in IsSupported()
      return absl::UnimplementedError(
          "STRIDED_SLICE input/params dimensions mismatch.");
    }

    const auto* params =
        static_cast<const TfLiteStridedSliceParams*>(tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteStridedSliceParams.");
    }

    ::ml_drift::SliceAttributes attr;
    attr.starts = ::ml_drift::BHWC(0, 0, 0, 0);
    attr.ends = ExtractTensorShape(input_tensor);
    attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
    if (read_attr_without_batch) {
      ABSL_RETURN_IF_ERROR(
          PreReadAttribsWithoutBatch(context, tflite_node, &attr));
    } else if (read_attr_with_batch) {
      ABSL_RETURN_IF_ERROR(
          PreReadAttribsWithBatch(context, tflite_node, &attr));
    }
    const ::ml_drift::BHWC in_shape = ExtractTensorShape(input_tensor);
    UpdateIfNegative(in_shape, &attr);
    const int begin_mask =
        UpdateMask(params->begin_mask, input_tensor->dims->size);
    const int end_mask = UpdateMask(params->end_mask, input_tensor->dims->size);
    UpdateWithMask(begin_mask, end_mask, in_shape, &attr);
    if (attr.strides.b == 0 || attr.strides.h == 0 || attr.strides.w == 0 ||
        attr.strides.c == 0) {
      return absl::InvalidArgumentError("stride values must be non-zero");
    }
    if (attr.strides.b < 0 || attr.strides.h < 0 || attr.strides.w < 0 ||
        attr.strides.c < 0) {
      return absl::UnimplementedError("Reverse slices are not supported.");
    }

    ::ml_drift::BHWC ref_shape;
    ref_shape.b =
        ::ml_drift::DivideRoundUp(attr.ends.b - attr.starts.b, attr.strides.b);
    ref_shape.h =
        ::ml_drift::DivideRoundUp(attr.ends.h - attr.starts.h, attr.strides.h);
    ref_shape.w =
        ::ml_drift::DivideRoundUp(attr.ends.w - attr.starts.w, attr.strides.w);
    ref_shape.c =
        ::ml_drift::DivideRoundUp(attr.ends.c - attr.starts.c, attr.strides.c);

    TfLiteTensor* output = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetOutputTensor(context, tflite_node, 0, &output));
    const ::ml_drift::BHWC out_shape = ExtractTensorShape(output);
    if (ref_shape != out_shape) {
      return absl::UnimplementedError("Output shape mismatch");
    }

    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  absl::Status CheckGpuDelegateCompatibility(
      const TfLiteNode* tflite_node, const tflite::OpSignature& op_sig) {
    const auto* params =
        static_cast<const TfLiteStridedSliceParams*>(tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteStridedSliceParams.");
    }
    if (params->ellipsis_mask) {
      return absl::UnimplementedError("Slice does not support ellipsis_mask.");
    }
    if (params->new_axis_mask) {
      return absl::UnimplementedError("Slice does not support new_axis_mask.");
    }
    if (params->shrink_axis_mask) {
      return absl::UnimplementedError(
          "Slice does not support shrink_axis_mask parameter. ");
    }
    if (op_sig.inputs.size() < 4) {
      return absl::UnimplementedError("STRIDED_SLICE requires 4 inputs.");
    }
    const auto& input_dims = op_sig.inputs.at(0).dims.size();
    const auto& start_dims = op_sig.inputs.at(1).dims[0];
    const bool read_attr_without_batch = input_dims == start_dims + 1;
    const bool read_attr_with_batch = input_dims == start_dims;
    if (!read_attr_without_batch && !read_attr_with_batch) {
      return absl::UnimplementedError(
          "STRIDED_SLICE input/params dimensions mismatch.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::SLICE);
    ::ml_drift::Value* input = reader->ReadValue(0);
    graph->AddConsumer(node->id, input->id);

    const TfLiteTensor* input_tensor = reader->GetInputTensor(0);

    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>
        starts_sizes;
    reader->ReadTensor(1, &starts_sizes, ReadTensorFlags::kNoExtraBytes);

    const bool read_attr_without_batch =
        input_tensor->dims->size == starts_sizes.data.size() + 1;
    const bool read_attr_with_batch =
        input_tensor->dims->size == starts_sizes.data.size();

    const auto* params =
        static_cast<const TfLiteStridedSliceParams*>(tflite_node->builtin_data);

    ::ml_drift::SliceAttributes attr;
    attr.starts = ::ml_drift::BHWC(0, 0, 0, 0);
    attr.ends = input->tensor.shape;
    attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
    if (read_attr_without_batch) {
      ReadAttribsWithoutBatch(reader, &attr);
    } else if (read_attr_with_batch) {
      ReadAttribsWithBatch(reader, &attr);
    }
    UpdateIfNegative(input->tensor.shape, &attr);
    const int begin_mask =
        UpdateMask(params->begin_mask, input_tensor->dims->size);
    const int end_mask = UpdateMask(params->end_mask, input_tensor->dims->size);
    UpdateWithMask(begin_mask, end_mask, input->tensor.shape, &attr);

    ::ml_drift::BHWC ref_shape;
    ref_shape.b =
        ::ml_drift::DivideRoundUp(attr.ends.b - attr.starts.b, attr.strides.b);
    ref_shape.h =
        ::ml_drift::DivideRoundUp(attr.ends.h - attr.starts.h, attr.strides.h);
    ref_shape.w =
        ::ml_drift::DivideRoundUp(attr.ends.w - attr.starts.w, attr.strides.w);
    ref_shape.c =
        ::ml_drift::DivideRoundUp(attr.ends.c - attr.starts.c, attr.strides.c);

    node->operation.attributes = attr;
    reader->AddOutputs(node);
  }

 private:
  static void UpdateWithMask(int begin_mask, int end_mask,
                             const ::ml_drift::BHWC& input_shape,
                             ::ml_drift::SliceAttributes* attr) {
    if (begin_mask & 1) attr->starts.b = 0;
    if (begin_mask & 2) attr->starts.h = 0;
    if (begin_mask & 4) attr->starts.w = 0;
    if (begin_mask & 8) attr->starts.c = 0;

    if (end_mask & 1) attr->ends.b = input_shape.b;
    if (end_mask & 2) attr->ends.h = input_shape.h;
    if (end_mask & 4) attr->ends.w = input_shape.w;
    if (end_mask & 8) attr->ends.c = input_shape.c;
  }

  static void UpdateIfNegative(const ::ml_drift::BHWC& input_shape,
                               ::ml_drift::SliceAttributes* attr) {
    if (attr->ends.b < 0) attr->ends.b += input_shape.b;
    if (attr->ends.h < 0) attr->ends.h += input_shape.h;
    if (attr->ends.w < 0) attr->ends.w += input_shape.w;
    if (attr->ends.c < 0) attr->ends.c += input_shape.c;

    if (attr->starts.b < 0) attr->starts.b += input_shape.b;
    if (attr->starts.h < 0) attr->starts.h += input_shape.h;
    if (attr->starts.w < 0) attr->starts.w += input_shape.w;
    if (attr->starts.c < 0) attr->starts.c += input_shape.c;
  }

  static void ReadBhwc(const ObjectReader* reader, int node_input_index,
                       ::ml_drift::BHWC* bhwc) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32> t;
    reader->ReadTensor(node_input_index, &t, ReadTensorFlags::kNoExtraBytes);
    if (t.data.size() == 4) {
      *bhwc = ::ml_drift::BHWC(t.data[0], t.data[1], t.data[2], t.data[3]);
    } else if (t.data.size() == 3) {
      *bhwc = ::ml_drift::BHWC(t.data[0], bhwc->h, t.data[1], t.data[2]);
    } else if (t.data.size() == 2) {
      *bhwc = ::ml_drift::BHWC(t.data[0], bhwc->h, bhwc->w, t.data[1]);
    } else if (t.data.size() == 1) {
      *bhwc = ::ml_drift::BHWC(t.data[0], bhwc->h, bhwc->w, bhwc->c);
    }
  }

  static void ReadAttribsWithBatch(const ObjectReader* reader,
                                   ::ml_drift::SliceAttributes* attr) {
    ReadBhwc(reader, 1, &attr->starts);
    ReadBhwc(reader, 2, &attr->ends);
    ReadBhwc(reader, 3, &attr->strides);
  }

  static void ReadHwc(const ObjectReader* reader, int node_input_index,
                      ::ml_drift::BHWC* bhwc) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32> t;
    reader->ReadTensor(node_input_index, &t, ReadTensorFlags::kNoExtraBytes);
    if (t.data.size() == 3) {
      *bhwc = ::ml_drift::BHWC(bhwc->b, t.data[0], t.data[1], t.data[2]);
    } else if (t.data.size() == 2) {
      *bhwc = ::ml_drift::BHWC(bhwc->b, bhwc->h, t.data[0], t.data[1]);
    } else if (t.data.size() == 1) {
      *bhwc = ::ml_drift::BHWC(bhwc->b, bhwc->h, bhwc->w, t.data[0]);
    }
  }

  static void ReadAttribsWithoutBatch(const ObjectReader* reader,
                                      ::ml_drift::SliceAttributes* attr) {
    ReadHwc(reader, 1, &attr->starts);
    ReadHwc(reader, 2, &attr->ends);
    ReadHwc(reader, 3, &attr->strides);
  }

  static absl::Status CanReadBhwc(const TfLiteContext* context,
                                  const TfLiteNode* tflite_node,
                                  int node_input_index,
                                  ::ml_drift::BHWC* bhwc) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32> t;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, node_input_index,
                                       &t, ReadTensorFlags::kNoExtraBytes));
    if (t.data.size() == 4) {
      *bhwc = ::ml_drift::BHWC(t.data[0], t.data[1], t.data[2], t.data[3]);
    } else if (t.data.size() == 3) {
      *bhwc = ::ml_drift::BHWC(t.data[0], bhwc->h, t.data[1], t.data[2]);
    } else if (t.data.size() == 2) {
      *bhwc = ::ml_drift::BHWC(t.data[0], bhwc->h, bhwc->w, t.data[1]);
    } else if (t.data.size() == 1) {
      *bhwc = ::ml_drift::BHWC(t.data[0], bhwc->h, bhwc->w, bhwc->c);
    }
    return absl::OkStatus();
  }

  static absl::Status PreReadAttribsWithBatch(
      const TfLiteContext* context, const TfLiteNode* tflite_node,
      ::ml_drift::SliceAttributes* attr) {
    ABSL_RETURN_IF_ERROR(CanReadBhwc(context, tflite_node, 1, &attr->starts));
    ABSL_RETURN_IF_ERROR(CanReadBhwc(context, tflite_node, 2, &attr->ends));
    ABSL_RETURN_IF_ERROR(CanReadBhwc(context, tflite_node, 3, &attr->strides));
    return absl::OkStatus();
  }

  static absl::Status CanReadHwc(const TfLiteContext* context,
                                 const TfLiteNode* tflite_node,
                                 int node_input_index, ::ml_drift::BHWC* bhwc) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32> t;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, node_input_index,
                                       &t, ReadTensorFlags::kNoExtraBytes));
    if (t.data.size() == 3) {
      *bhwc = ::ml_drift::BHWC(bhwc->b, t.data[0], t.data[1], t.data[2]);
    } else if (t.data.size() == 2) {
      *bhwc = ::ml_drift::BHWC(bhwc->b, bhwc->h, t.data[0], t.data[1]);
    } else if (t.data.size() == 1) {
      *bhwc = ::ml_drift::BHWC(bhwc->b, bhwc->h, bhwc->w, t.data[0]);
    }
    return absl::OkStatus();
  }

  static absl::Status PreReadAttribsWithoutBatch(
      const TfLiteContext* context, const TfLiteNode* tflite_node,
      ::ml_drift::SliceAttributes* attr) {
    ABSL_RETURN_IF_ERROR(CanReadHwc(context, tflite_node, 1, &attr->starts));
    ABSL_RETURN_IF_ERROR(CanReadHwc(context, tflite_node, 2, &attr->ends));
    ABSL_RETURN_IF_ERROR(CanReadHwc(context, tflite_node, 3, &attr->strides));
    return absl::OkStatus();
  }

  int UpdateMask(int mask, int src_dims_count) {
    if (src_dims_count == 4) {
      return mask;
    } else if (src_dims_count == 3) {
      // cwb -> cw0b
      int b_bit = mask & 1;
      int w_bit = mask & 2;
      int c_bit = mask & 4;
      mask = (c_bit << 1) | (w_bit << 1) | b_bit;
      return mask;
    } else if (src_dims_count == 2) {
      // cb -> c00b
      int b_bit = mask & 1;
      int c_bit = mask & 2;
      mask = (c_bit << 2) | b_bit;
      return mask;
    } else if (src_dims_count == 1) {
      // b -> 000b
      return mask;
    } else {
      return mask;
    }
  }
};

class TileOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::TILE);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);
  }
};

class TopKOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));

    const TfLiteTensor* k_tensor = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &k_tensor));
    if (!tflite::IsConstantTensor(k_tensor)) {
      return absl::InvalidArgumentError("TopK k must be constant tensor.");
    }
    if (tflite::NumElements(k_tensor) != 1) {
      return absl::InvalidArgumentError("TopK k must be scalar.");
    }

    if (tflite_node->outputs->size != 2) {
      return absl::InvalidArgumentError("TopK requires 2 output tensors");
    }
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    const TfLiteTensor* input_tensor = reader->GetInputTensor(0);
    const bool is_1d = input_tensor && input_tensor->dims->size == 1;

    ::ml_drift::Value* input_value = nullptr;

    // Insert a RESHAPE if input tensor [N] is mis-auto-expanded to [N,1,1,1].
    if (is_1d) {
      const ::ml_drift::Value* original_input = reader->ReadValue(0);
      const ::ml_drift::BHWC new_shape(1, 1, 1,
                                       original_input->tensor.shape.b);
      ::ml_drift::Node* reshape_node = graph->NewNode();
      reshape_node->operation.type =
          ToString(::ml_drift::OperationType::RESHAPE);
      ::ml_drift::ReshapeAttributes reshape_attr;
      reshape_attr.new_shape = new_shape;
      reshape_node->operation.attributes = std::move(reshape_attr);
      graph->AddConsumer(reshape_node->id, original_input->id);
      input_value = graph->NewValue();
      input_value->tensor.type = original_input->tensor.type;
      input_value->tensor.shape = new_shape;
      graph->SetProducer(reshape_node->id, input_value->id);
    }

    // Insert the TOP_K.
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::TOP_K);
    ::ml_drift::TopKAttributes attr;
    ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::INT32>
        k_tensor;
    reader->ReadTensor(1, &k_tensor, ReadTensorFlags::kNoExtraBytes);
    attr.k = k_tensor.data[0];
    node->operation.attributes = std::move(attr);

    if (is_1d) {
      graph->AddConsumer(node->id, input_value->id);
    } else {
      reader->AddInput(node, 0);
    }

    // Reshape the outputs back to the standard 1D layout.
    if (is_1d) {
      for (int i = 0; i < tflite_node->outputs->size; ++i) {
        ::ml_drift::Value* top_k_interm_output = graph->NewValue();
        top_k_interm_output->tensor.shape = ::ml_drift::BHWC(1, 1, 1, attr.k);
        top_k_interm_output->tensor.type =
            ToDataType(reader->GetOutputTensor(i)->type);
        graph->SetProducer(node->id, top_k_interm_output->id);
        ::ml_drift::Node* reshape_node = graph->NewNode();
        reshape_node->operation.type =
            ToString(::ml_drift::OperationType::RESHAPE);
        ::ml_drift::ReshapeAttributes reshape_attr;
        reshape_attr.new_shape = ::ml_drift::BHWC(attr.k, 1, 1, 1);
        reshape_node->operation.attributes = std::move(reshape_attr);
        graph->AddConsumer(reshape_node->id, top_k_interm_output->id);
        reader->AddOutput(reshape_node, i);
      }
    } else {
      reader->AddOutputs(node);
    }
  }
};

class TransposeConvBuiltinOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 3));
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 2));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    const auto* params = static_cast<const TfLiteTransposeConvParams*>(
        tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteTransposeConvParams.");
    }

    const int runtime_inputs =
        GetNumberOfRuntimeInputsForNode(context, tflite_node);
    if (runtime_inputs == 2) {
      ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    } else {
      ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
          dummy_weights;
      ABSL_RETURN_IF_ERROR(
          PreCheckReadTensor(context, tflite_node, 1, &dummy_weights));
    }

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::CONVOLUTION_TRANSPOSED);
    ::ml_drift::Value* input = reader->ReadValue(2);
    graph->AddConsumer(node->id, input->id);
    reader->AddOutputs(node);

    const auto* params = static_cast<const TfLiteTransposeConvParams*>(
        tflite_node->builtin_data);

    ::ml_drift::ConvolutionTransposedAttributes attr;
    attr.stride =
        params ? ::ml_drift::HW(params->stride_height, params->stride_width)
               : ::ml_drift::HW(1, 1);
    const int runtime_inputs = reader->GetNumberOfRuntimeInputs();
    if (runtime_inputs == 2) {
      reader->AddInput(node, 1);
      auto weights_shape = graph->FindInputs(node->id)[1]->tensor.shape;
      attr.weights.shape = ::ml_drift::OHWI(weights_shape.b, weights_shape.h,
                                            weights_shape.w, weights_shape.c);
    } else {  // runtime_inputs == 1;
      reader->ReadTensor(1, &attr.weights, ReadTensorFlags::kExtraBytes);
    }
    if (reader->IsNodeInputTensorPresent(3)) {
      reader->ReadTensor(3, &attr.bias, ReadTensorFlags::kNoExtraBytes);
    }

    UpdatePadding(params->padding, graph->FindInputs(node->id)[0]->tensor.shape,
                  &attr);
    node->operation.attributes = std::move(attr);
  }
};

class TransposeConvCustomOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
        dummy_weights;
    ABSL_RETURN_IF_ERROR(
        PreCheckReadTensor(context, tflite_node, 1, &dummy_weights));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::CONVOLUTION_TRANSPOSED);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    const auto* params = static_cast<const TfLiteTransposeConvParams*>(
        tflite_node->custom_initial_data);

    ::ml_drift::ConvolutionTransposedAttributes attr;
    attr.stride =
        params ? ::ml_drift::HW(params->stride_height, params->stride_width)
               : ::ml_drift::HW(1, 1);
    reader->ReadTensor(1, &attr.weights, ReadTensorFlags::kExtraBytes);
    if (reader->IsNodeInputTensorPresent(2)) {
      reader->ReadTensor(2, &attr.bias, ReadTensorFlags::kNoExtraBytes);
    }

    UpdatePadding(params ? params->padding : kTfLitePaddingUnknown,
                  graph->FindInputs(node->id)[0]->tensor.shape, &attr);
    node->operation.attributes = std::move(attr);
  }
};

class TransposeOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 9));
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32> perm;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, &perm,
                                       ReadTensorFlags::kNoExtraBytes));
    if (perm.data.size() > 4 || perm.data.size() < 2) {
      return absl::InvalidArgumentError(
          "Permutation for transpose is invalid.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::TRANSPOSE);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::TransposeAttributes attr;
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32> perm;
    reader->ReadTensor(1, &perm, ReadTensorFlags::kNoExtraBytes);
    std::map<::ml_drift::Axis, int> axis_to_index = {
        {::ml_drift::Axis::BATCH, 0},
        {::ml_drift::Axis::HEIGHT, 1},
        {::ml_drift::Axis::WIDTH, 2},
        {::ml_drift::Axis::CHANNELS, 3}};
    if (perm.data.size() == 4) {
      attr.perm = ::ml_drift::BHWC(perm.data[0], perm.data[1], perm.data[2],
                                   perm.data[3]);
    } else if (perm.data.size() == 3) {
      std::vector<::ml_drift::Axis> index_to_axis = {
          ::ml_drift::Axis::BATCH, ::ml_drift::Axis::WIDTH,
          ::ml_drift::Axis::CHANNELS};
      attr.perm.b = axis_to_index[index_to_axis[perm.data[0]]];
      attr.perm.h = 1;
      attr.perm.w = axis_to_index[index_to_axis[perm.data[1]]];
      attr.perm.c = axis_to_index[index_to_axis[perm.data[2]]];
    } else if (perm.data.size() == 2) {
      std::vector<::ml_drift::Axis> index_to_axis = {
          ::ml_drift::Axis::BATCH, ::ml_drift::Axis::CHANNELS};
      attr.perm.b = axis_to_index[index_to_axis[perm.data[0]]];
      attr.perm.h = 1;
      attr.perm.w = 2;
      attr.perm.c = axis_to_index[index_to_axis[perm.data[1]]];
    }

    node->operation.attributes = attr;
  }
};

class UnpackOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    const auto* params =
        static_cast<const TfLiteUnpackParams*>(tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("Missing TfLiteUnpackParams.");
    }
    if (params->num == 1) {
      ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
      ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
      return absl::OkStatus();
    }

    const TfLiteTensor* input = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));
    ABSL_RETURN_IF_ERROR(PreCheckTensorShape(*input));
    ABSL_RETURN_IF_ERROR(PreCheckAxisFromIndex(*input, params->axis));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    for (int i = 0; i < tflite_node->outputs->size; ++i) {
      TfLiteTensor* output = nullptr;
      ABSL_RETURN_IF_ERROR(
          PreGetOutputTensor(context, tflite_node, 0, &output));
      ABSL_RETURN_IF_ERROR(PreCheckTensorShape(*output));
      ABSL_RETURN_IF_ERROR(PreCheckOutput(context, tflite_node, i));
    }

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    const auto* params =
        static_cast<const TfLiteUnpackParams*>(tflite_node->builtin_data);
    if (params->num == 1) {
      // Adding Identity reshape that will be removed.
      ::ml_drift::Node* node = graph->NewNode();
      node->operation.type = ToString(::ml_drift::OperationType::RESHAPE);
      reader->AddInput(node, 0);
      reader->AddOutputs(node);
      // New shape comes from output shape.
      ::ml_drift::ReshapeAttributes attr;
      attr.new_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
      node->operation.attributes = attr;
      return;
    }
    const TfLiteTensor* input = reader->GetInputTensor(0);
    const ::ml_drift::BHWC input_shape = ExtractTensorShape(input);
    ::ml_drift::SplitAttributes attr;
    attr.axis = ExtractAxisFromIndex(*input, params->axis);
    ::ml_drift::BHWC output_required_shape = input_shape;
    output_required_shape.set(attr.axis, 1);

    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::SPLIT);
    node->operation.attributes = attr;
    reader->AddInput(node, 0);
    auto input_value = graph->FindInputs(node->id)[0];
    for (int i = 0; i < tflite_node->outputs->size; ++i) {
      const TfLiteTensor* output = reader->GetOutputTensor(i);
      const ::ml_drift::BHWC output_shape = ExtractTensorShape(output);
      if (output_shape != output_required_shape) {
        // GPU delegates does not support implicit shapes transformations
        // adding explicit Reshape
        ::ml_drift::Value* copy_value = graph->NewValue();
        copy_value->tensor.type = input_value->tensor.type;
        copy_value->tensor.shape = output_required_shape;
        graph->SetProducer(node->id, copy_value->id);
        ::ml_drift::Node* node_reshape = graph->NewNode();
        node_reshape->operation.type =
            ToString(::ml_drift::OperationType::RESHAPE);
        ::ml_drift::ReshapeAttributes reshape_attr;
        reshape_attr.new_shape = output_shape;
        node_reshape->operation.attributes = reshape_attr;
        graph->AddConsumer(node_reshape->id, copy_value->id);
        reader->AddOutput(node_reshape, i);
      } else {
        reader->AddOutput(node, i);
      }
    }
  }
};

class Unpooling2DOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(tflite::CheckGpuDelegateCompatibility(
        context, tflite_node, registration));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    const auto* params =
        static_cast<const TfLitePoolParams*>(tflite_node->custom_initial_data);
    if (!params) return absl::InvalidArgumentError("Missing TfLitePoolParams.");
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::MAX_UNPOOLING_2D);
    reader->AddInput(node, 0);
    reader->AddInput(node, 1);
    reader->AddOutputs(node);
    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;
    ::ml_drift::MaxUnpooling2DAttributes attr;

    const auto* params =
        static_cast<const TfLitePoolParams*>(tflite_node->custom_initial_data);
    attr.kernel = ToHW(params->filter_height, params->filter_width);
    attr.strides = ToHW(params->stride_height, params->stride_width);
    UpdatePadding(params->padding, input_shape, &attr);

    node->operation.attributes = attr;

    auto output_value = graph->FindOutputs(node->id)[0];
    output_value->tensor.shape = CalculateOutputShape(input_shape, attr);
  }
};

// TODO: b/368214363 - Remove GroupNormParser class when its dependent models
// are transited to StableHloComposite op.
class GroupNormParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    const TfLiteTensor* input = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));
    const ::ml_drift::BHWC input_shape = ExtractTensorShape(input);

    ::ml_drift::GroupNormAttributes attr;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, attr.gamma,
                                       ReadTensorFlags::kNoExtraBytes));
    if (attr.gamma.has_value() && input_shape.c != attr.gamma.value().shape.v) {
      return absl::InternalError(
          "Scale tensor channels don't match input tensor channels.");
    }
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 2, attr.beta,
                                       ReadTensorFlags::kNoExtraBytes));
    if (attr.beta.has_value() && attr.beta.value().shape.v != 0 &&
        input_shape.c != attr.beta.value().shape.v) {
      return absl::InternalError(
          "Bias tensor channels don't match input tensor channels.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = "group_norm";
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::GroupNormAttributes attr;
    if (reader->IsNodeInputTensorPresent(1)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> t;
      reader->ReadTensor(1, &t, ReadTensorFlags::kNoExtraBytes);
      attr.gamma = std::move(t);
    }
    if (reader->IsNodeInputTensorPresent(2)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> t;
      reader->ReadTensor(2, &t, ReadTensorFlags::kNoExtraBytes);
      attr.beta = std::move(t);
    }
    attr.groups = 32;
    attr.epsilon = 1e-6;
    const uint8_t* buffer_t =
        reinterpret_cast<const uint8_t*>(tflite_node->custom_initial_data);
    if (buffer_t) {
      size_t length = tflite_node->custom_initial_data_size;
      const flexbuffers::Map& m =
          flexbuffers::GetRoot(buffer_t, length).AsMap();
      attr.groups = m["num_groups"].AsInt32();
      attr.epsilon = m["epsilon"].AsFloat();
    }

    node->operation.attributes = attr;
  }
};

// TODO: b/368214363 - Remove LayerNormParser class when its dependent models
// are transited to StableHloComposite op.
class LayerNormParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    const TfLiteTensor* input = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));
    const ::ml_drift::BHWC input_shape = ExtractTensorShape(input);

    ::ml_drift::LayerNormAttributes attr;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, attr.scale,
                                       ReadTensorFlags::kNoExtraBytes));
    if (attr.scale.has_value() && input_shape.c != attr.scale.value().shape.v) {
      return absl::InternalError(
          "Scale tensor channels don't match input tensor channels.");
    }
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 2, attr.bias,
                                       ReadTensorFlags::kNoExtraBytes));
    if (attr.bias.has_value() && attr.bias.value().shape.v != 0 &&
        input_shape.c != attr.bias.value().shape.v) {
      return absl::InternalError(
          "Bias tensor channels don't match input tensor channels.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = "layer_norm";
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::LayerNormAttributes attr;
    if (reader->IsNodeInputTensorPresent(1)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> t;
      reader->ReadTensor(1, &t, ReadTensorFlags::kNoExtraBytes);
      attr.scale = std::move(t);
    }
    if (reader->IsNodeInputTensorPresent(2)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> t;
      reader->ReadTensor(2, &t, ReadTensorFlags::kNoExtraBytes);
      attr.bias = std::move(t);
    }
    const uint8_t* buffer_t =
        reinterpret_cast<const uint8_t*>(tflite_node->custom_initial_data);
    size_t length = tflite_node->custom_initial_data_size;
    const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
    attr.epsilon = m["epsilon"].AsFloat();
    node->operation.attributes = std::move(attr);
  }
};

class PixelShuffleParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::DEPTH_TO_SPACE);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::SpaceToDepthAttributes attr;

    const uint8_t* buffer_t =
        reinterpret_cast<const uint8_t*>(tflite_node->custom_initial_data);
    size_t length = tflite_node->custom_initial_data_size;
    const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
    attr.block_size = m["block_size"].AsInt32();
    node->operation.attributes = attr;
  }
};

class PositionalEmbeddingParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    if (tflite_node->inputs->size != 2) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid number of inputs: ", tflite_node->inputs->size,
                       ", while expected 2."));
    }
    if (tflite_node->outputs->size != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid number of outputs: ", tflite_node->outputs->size,
          ", while expected 1."));
    }
    const TfLiteTensor* src = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &src));
    const ::ml_drift::BHWC src_sh = ExtractTensorShape(src);
    const TfLiteTensor* pos = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &pos));
    const ::ml_drift::BHWC pos_sh = ExtractTensorShape(pos);
    if (src_sh.w != pos_sh.w) {
      return absl::InvalidArgumentError(
          "src and pos must have the same width for PositionalEmbedding.");
    }

    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::POSITIONAL_EMBEDDING);
    reader->AddInput(node, 0);
    reader->AddInput(node, 1);
    reader->AddOutputs(node);
  }
};

class RoPEParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    if (tflite_node->inputs->size == 2) {
      if (tflite_node->outputs->size != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid number of outputs: ", tflite_node->outputs->size,
            ", while expected 1."));
      }
    } else if (tflite_node->inputs->size == 3) {
      if (tflite_node->outputs->size != 2) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid number of outputs: ", tflite_node->outputs->size,
            ", while expected 2."));
      }
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid number of inputs: ", tflite_node->inputs->size,
                       ", while expected 2 or 3."));
    }

    const int pos_tensor_idx = tflite_node->inputs->size == 2 ? 1 : 2;

    const TfLiteTensor* src = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &src));
    const ::ml_drift::BHWC src_sh = ExtractTensorShape(src);
    const TfLiteTensor* pos = nullptr;
    ABSL_RETURN_IF_ERROR(
        PreGetInputTensor(context, tflite_node, pos_tensor_idx, &pos));
    const ::ml_drift::BHWC pos_sh = ExtractTensorShape(pos);
    if (src_sh.w != pos_sh.w) {
      return absl::InvalidArgumentError(
          "src and pos must have the same width for PositionalEmbedding.");
    }
    ABSL_RETURN_IF_ERROR(
        PreCheckRuntimeOrConstantInput(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(
        PreCheckRuntimeOrConstantInput(context, tflite_node, 1));
    if (tflite_node->inputs->size != 2) {
      ABSL_RETURN_IF_ERROR(
          PreCheckRuntimeOrConstantInput(context, tflite_node, 2));
    }
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::ROPE);
    {
      constexpr int kIndex = 0;
      if (reader->IsConstantTensor(kIndex)) {
        const ::ml_drift::Value* input =
            reader->AddConstInput(kIndex, /*layout=*/{});
        graph->AddConsumer(node->id, input->id);
      } else {
        reader->AddInput(node, kIndex);
      }
    }
    {
      constexpr int kIndex = 1;
      if (reader->IsConstantTensor(kIndex)) {
        const ::ml_drift::Value* input =
            reader->AddConstInput(kIndex, /*layout=*/{});
        graph->AddConsumer(node->id, input->id);
      } else {
        reader->AddInput(node, kIndex);
      }
    }
    if (tflite_node->inputs->size > 2) {
      constexpr int kIndex = 2;
      if (reader->IsConstantTensor(kIndex)) {
        const ::ml_drift::Value* input =
            reader->AddConstInput(kIndex, /*layout=*/{});
        graph->AddConsumer(node->id, input->id);
      } else {
        reader->AddInput(node, kIndex);
      }
    }
    reader->AddOutputs(node);
  }
};

// Checks that the reduction axes match the expected reduction axes.
absl::Status ExpectReductionAxes(const flexbuffers::Map& flexbuffer_map,
                                 std::vector<int64_t> expected_reduction_axes,
                                 absl::string_view operation_name) {
  const flexbuffers::Vector reduction_axes_vec =
      flexbuffer_map["_TENSOR_V1_reduction_axes"]
          .AsMap()["TENSOR_DATA"]
          .AsVector();

  std::vector<int64_t> reduction_axes;
  reduction_axes.reserve(reduction_axes_vec.size());
  for (int n = 0; n < reduction_axes_vec.size(); ++n) {
    reduction_axes.push_back(reduction_axes_vec[n].AsInt64());
  }

  if (reduction_axes != expected_reduction_axes) {
    return absl::InternalError(absl::StrCat(
        operation_name, " has unexpected reduction axes. Expected axes [",
        absl::StrJoin(expected_reduction_axes, ", "), "] but got [",
        absl::StrJoin(reduction_axes, ", "), "]"));
  }

  return absl::OkStatus();
}

class CompositeGroupNormParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final {
    const TfLiteTensor* input = context->tensors + tflite_node->inputs->data[0];
    if (!input) {
      return absl::InvalidArgumentError("GroupNorm is missing input tensor.");
    }
    if (!input->dims || input->dims->size > 4) {
      std::vector<int> input_dims;
      if (input->dims) {
        input_dims.assign(input->dims->data,
                          input->dims->data + input->dims->size);
      }
      return absl::InvalidArgumentError(
          absl::StrCat("GroupNorm has bad input tensor dims. Expected at most "
                       "4D input, got ",
                       absl::StrJoin(input_dims, ", ")));
    }
    const TfLiteTensor* gamma =
        tflite_node->inputs->size > 1
            ? context->tensors + tflite_node->inputs->data[1]
            : nullptr;
    if (gamma) {
      if (!gamma->dims || gamma->dims->size != 1) {
        return absl::InvalidArgumentError(
            "GroupNorm has bad gamma tensor dims.");
      }
      if (input->dims->data[input->dims->size - 1] != gamma->dims->data[0]) {
        return absl::InternalError(
            "Scale tensor channels don't match input tensor channels.");
      }
    }
    const TfLiteTensor* beta =
        tflite_node->inputs->size > 2
            ? context->tensors + tflite_node->inputs->data[2]
            : nullptr;
    if (beta) {
      if (!beta->dims || beta->dims->size != 1) {
        return absl::InvalidArgumentError(
            "GroupNorm has bad beta tensor dims.");
      }
      if (input->dims->data[input->dims->size - 1] != beta->dims->data[0]) {
        return absl::InternalError(
            "Beta tensor channels don't match input tensor channels.");
      }
    }
    const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
        tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("GroupNorm is missing params.");
    }
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    const int tensor_dims_size = input->dims->size;

    if (!flexbuffer_map["_TENSOR_V1_reduction_axes"].IsNull()) {
      // Check that reduction axes are
      //   - [1, 2, ..., n-1] if sub_type is set (default GroupNorm behavior)
      //   - [n-1] if sub_type is not set
      //     (for backward compatibility, see b/489399819)
      bool has_sub_type = !flexbuffer_map["sub_type"].IsNull();
      std::vector<int64_t> expected_reduction_axes;
      if (has_sub_type) {
        expected_reduction_axes.resize(tensor_dims_size - 1);
        std::iota(expected_reduction_axes.begin(),
                  expected_reduction_axes.end(), 1);
      } else {
        expected_reduction_axes.push_back(tensor_dims_size - 1);
      }

      ABSL_RETURN_IF_ERROR(ExpectReductionAxes(
          flexbuffer_map, expected_reduction_axes, "GroupNorm"));
    }

    if (!flexbuffer_map["channel_axis"].IsNull() &&
        flexbuffer_map["channel_axis"].AsInt32() != tensor_dims_size - 1) {
      return absl::InternalError(absl::StrCat(
          "Only channel-last tensor is supported for GroupNorm. Expected axis ",
          tensor_dims_size - 1, " but got ",
          flexbuffer_map["channel_axis"].AsInt32()));
    }
    if (flexbuffer_map["num_groups"].IsNull()) {
      return absl::InvalidArgumentError("GroupNorm is missing num_groups.");
    }
    if (flexbuffer_map["epsilon"].IsNull()) {
      return absl::InvalidArgumentError("GroupNorm is missing epsilon.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::GROUP_NORM);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::GroupNormAttributes attr;
    if (reader->IsNodeInputTensorPresent(1)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
          gamma;
      reader->ReadTensor(1, &gamma, ReadTensorFlags::kNoExtraBytes);
      attr.gamma = std::move(gamma);
    }
    if (reader->IsNodeInputTensorPresent(2)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
          beta;
      reader->ReadTensor(2, &beta, ReadTensorFlags::kNoExtraBytes);
      attr.beta = std::move(beta);
    }
    const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
        tflite_node->builtin_data);
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    attr.groups = flexbuffer_map["num_groups"].AsInt32();
    attr.epsilon = flexbuffer_map["epsilon"].AsFloat();
    node->operation.attributes = std::move(attr);
  }
};

class CompositeLayerNormParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final {
    const TfLiteTensor* input = context->tensors + tflite_node->inputs->data[0];
    if (!input) {
      return absl::InvalidArgumentError("LayerNorm is missing input tensor.");
    }
    if (!input->dims || input->dims->size > 4) {
      std::vector<int> input_dims;
      if (input->dims) {
        input_dims.assign(input->dims->data,
                          input->dims->data + input->dims->size);
      }
      return absl::InvalidArgumentError(
          absl::StrCat("LayerNorm has bad input tensor dims. Expected at most "
                       "4D input, got ",
                       absl::StrJoin(input_dims, ", ")));
    }
    const TfLiteTensor* scale =
        tflite_node->inputs->size > 1
            ? context->tensors + tflite_node->inputs->data[1]
            : nullptr;
    if (scale) {
      if (!scale->dims || scale->dims->size != 1) {
        return absl::InvalidArgumentError(
            "LayerNorm has bad scale tensor dims.");
      }
      if (input->dims->data[input->dims->size - 1] != scale->dims->data[0]) {
        return absl::InternalError(
            "Scale tensor channels don't match input tensor channels.");
      }
    }
    const TfLiteTensor* bias =
        tflite_node->inputs->size > 2
            ? context->tensors + tflite_node->inputs->data[2]
            : nullptr;
    if (bias) {
      if (!bias->dims || bias->dims->size != 1) {
        return absl::InvalidArgumentError(
            "LayerNorm has bad beta tensor dims.");
      }
      if (input->dims->data[input->dims->size - 1] != bias->dims->data[0]) {
        return absl::InternalError(
            "Bias tensor channels don't match input tensor channels.");
      }
    }
    const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
        tflite_node->builtin_data);
    if (!params) {
      return absl::InvalidArgumentError("LayerNorm is missing params.");
    }
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (flexbuffer_map["epsilon"].IsNull()) {
      return absl::InvalidArgumentError("LayerNorm is missing epsilon.");
    }
    if (!flexbuffer_map["_TENSOR_V1_reduction_axes"].IsNull()) {
      ABSL_RETURN_IF_ERROR(ExpectReductionAxes(
          flexbuffer_map, {input->dims->size - 1}, "LayerNorm"));
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(::ml_drift::OperationType::LAYER_NORM);
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::LayerNormAttributes attr;
    if (reader->IsNodeInputTensorPresent(1)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
          scale;
      reader->ReadTensor(1, &scale, ReadTensorFlags::kNoExtraBytes);
      attr.scale = std::move(scale);
    }
    if (reader->IsNodeInputTensorPresent(2)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
          bias;
      reader->ReadTensor(2, &bias, ReadTensorFlags::kNoExtraBytes);
      attr.bias = std::move(bias);
    }
    const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
        tflite_node->builtin_data);
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    attr.epsilon = flexbuffer_map["epsilon"].AsFloat();
    node->operation.attributes = std::move(attr);
  }
};

class CompositeRmsNormParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    const TfLiteTensor* input = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));
    const ::ml_drift::BHWC input_shape = ExtractTensorShape(input);

    ::ml_drift::RmsNormAttributes attr;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, attr.scale,
                                       ReadTensorFlags::kNoExtraBytes));
    if (attr.scale.has_value() && input_shape.c != attr.scale.value().shape.v) {
      return absl::InternalError(
          "Scale tensor channels don't match input tensor channels.");
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type = "rms_norm";
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::RmsNormAttributes attr;
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> t;
    reader->ReadTensor(1, &t, ReadTensorFlags::kNoExtraBytes);
    attr.scale = std::move(t);

    const TfLiteStablehloCompositeParams* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(
            tflite_node->builtin_data);
    const flexbuffers::Map m =
        flexbuffers::GetRoot(composite_params->attributes,
                             composite_params->attributes_size)
            .AsMap();
    attr.epsilon = m["epsilon"].AsFloat();
    node->operation.attributes = std::move(attr);
  }
};

class CompositeSdpaParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration*) final {
    if (tflite_node->inputs->size < 3 || tflite_node->inputs->size > 4) {
      return absl::FailedPreconditionError(
          absl::StrCat("Invalid number of inputs: ", tflite_node->inputs->size,
                       ", while expected 3 or 4."));
    }

    TfLiteTensor* q_tensor = context->tensors + tflite_node->inputs->data[0];
    TfLiteTensor* k_tensor = context->tensors + tflite_node->inputs->data[1];
    TfLiteTensor* v_tensor = context->tensors + tflite_node->inputs->data[2];

    const ::ml_drift::BHWC q_shape = ExtractTensorShape(q_tensor);
    const ::ml_drift::BHWC k_shape = ExtractTensorShape(k_tensor);
    const ::ml_drift::BHWC v_shape = ExtractTensorShape(v_tensor);

    const bool channels_match =
        q_shape.c == k_shape.c && q_shape.c == v_shape.c;
    if (!channels_match) {
      return absl::InternalError(absl::StrCat(
          "Q, K and V tensors' channels must match, but they are: Q.c = ",
          q_shape.c, ", K.c = ", k_shape.c, ", and V.c = ", v_shape.c, "."));
    }
    if (k_shape != v_shape) {
      return absl::InternalError(absl::StrCat(
          "K and V tensors' shapes must identical, but they are: K = ",
          ToString(k_shape), ", V = ", ToString(v_shape), "."));
    }
    if (tflite_node->inputs->size == 4) {
      TfLiteTensor* mask_tensor =
          context->tensors + tflite_node->inputs->data[3];
      const ::ml_drift::BHWC mask_shape = ExtractTensorShape(mask_tensor);
      if (k_shape.h != mask_shape.c) {
        return absl::InternalError(absl::StrCat(
            "K tensor's height must match Mask tensor's channels, but they "
            "are: "
            "K = ",
            ToString(k_shape), ", Mask = ", ToString(mask_shape), "."));
      }
    }
    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node, const TfLiteRegistration*,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    reader->GetNumberOfRuntimeInputs();
    ::ml_drift::Node* node = graph->NewNode();
    node->operation.type =
        ToString(::ml_drift::OperationType::SCALED_DOT_PRODUCT_ATTENTION);
    reader->AddInput(node, 0);  // Q
    reader->AddInput(node, 1);  // K
    reader->AddInput(node, 2);  // V

    // Mask is optional and can be either a constant tensor or a runtime tensor.
    if (const TfLiteTensor* input3 = reader->GetInputTensor(3)) {
      if (tflite::IsConstantTensor(input3)) {  // constant
        ::ml_drift::TensorFloat32 tensor;
        reader->ReadTensor(3, &tensor, ReadTensorFlags::kNoExtraBytes);
        ::ml_drift::Value* value = NewConstNode(std::move(tensor), graph);
        graph->AddConsumer(node->id, value->id);
      } else {  // runtime
        reader->AddInput(node, 3);
      }
    }
    reader->AddOutputs(node);

    const TfLiteStablehloCompositeParams* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(
            tflite_node->builtin_data);
    const uint8_t* buffer_t = composite_params->attributes;
    ::ml_drift::ScaledDotProductAttentionAttributes attr;
    if (buffer_t) {
      size_t length = composite_params->attributes_size;
      const flexbuffers::Map& m =
          flexbuffers::GetRoot(buffer_t, length).AsMap();
      if (!m["scale"].IsNull()) {
        attr.scale = m["scale"].AsFloat();
      }
    }
    node->operation.attributes = std::move(attr);
  }
};

class RmsNormParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
    ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

    const TfLiteTensor* input = nullptr;
    ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input));
    const ::ml_drift::BHWC input_shape = ExtractTensorShape(input);

    ::ml_drift::RmsNormAttributes attr;
    ABSL_RETURN_IF_ERROR(PreReadTensor(context, tflite_node, 1, attr.scale,
                                       ReadTensorFlags::kNoExtraBytes));
    if (attr.scale.has_value() && input_shape.c != attr.scale.value().shape.v) {
      return absl::InternalError(
          "Scale tensor channels don't match input tensor channels.");
    }

    return absl::OkStatus();
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = "rms_norm";
    reader->AddInput(node, 0);
    reader->AddOutputs(node);

    ::ml_drift::RmsNormAttributes attr;
    if (reader->IsNodeInputTensorPresent(1)) {
      ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> t;
      reader->ReadTensor(1, &t, ReadTensorFlags::kNoExtraBytes);
      attr.scale = std::move(t);
    }
    const uint8_t* buffer_t =
        reinterpret_cast<const uint8_t*>(tflite_node->custom_initial_data);
    size_t length = tflite_node->custom_initial_data_size;
    const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
    attr.epsilon = m["epsilon"].AsFloat();
    node->operation.attributes = std::move(attr);
  }
};

class UnsupportedOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return absl::UnimplementedError("Operation is not supported.");
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {}
};

bool IsAllAllowedTensors(TfLiteContext* context,
                         const TfLiteIntArray* tensor_indices,
                         const std::vector<TfLiteType>& allowed_types,
                         std::string* unsupported_details) {
  for (int i = 0; i < tensor_indices->size; ++i) {
    int tensor_idx = tensor_indices->data[i];
    if (tensor_idx == kTfLiteOptionalTensor) continue;
    const TfLiteTensor* t = &context->tensors[tensor_idx];
    if (t->dims && t->dims->size >= 5) {
      *unsupported_details +=
          "Tensor dimensions must be less than 5. " + std::string(t->name);
      return false;
    }
    bool type_supported = false;
    for (auto allowed_type : allowed_types) {
      if (t->type == allowed_type) {
        type_supported = true;
        break;
      }
    }
    if (t->allocation_type == kTfLiteArenaRw && !type_supported) {
      *unsupported_details += "Tensor type(" +
                              std::string(TfLiteTypeGetName(t->type)) +
                              ") is not supported. " + std::string(t->name);
      return false;
    }
  }
  return true;
}

}  // namespace

std::unique_ptr<TFLiteOperationParser> NewOperationParser(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    bool allow_quant_ops, const ModelBuilderOptions& options,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory) {
  const auto builtin_code = registration->builtin_code;
  if (excluded_ops != nullptr &&
      excluded_ops->contains(
          static_cast<TfLiteBuiltinOperator>(builtin_code))) {
    return std::make_unique<UnsupportedOperationParser>();
  }
  switch (builtin_code) {
    case kTfLiteBuiltinAbs:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::ABS);
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinAddN:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::ADD);
    case kTfLiteBuiltinArgMax:
      return std::make_unique<ArgMaxOperationParser>();
    case kTfLiteBuiltinAtan2:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::ATAN2);
    case kTfLiteBuiltinAveragePool2d:
      return std::make_unique<Pooling2DOperationParser>(
          ::ml_drift::PoolingType::AVERAGE);
    case kTfLiteBuiltinBatchMatmul:
      return std::make_unique<BatchedMatMulOperationParser>();
    case kTfLiteBuiltinBitcast:
      return std::make_unique<BitcastOperationParser>();
    case kTfLiteBuiltinBitwiseXor:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::LOGICAL_XOR);
    case kTfLiteBuiltinStablehloBroadcastInDim:
      return std::make_unique<BroadcastInDimOperationParser>();
    case kTfLiteBuiltinCast:
      return std::make_unique<CastOperationParser>();
    case kTfLiteBuiltinStablehloCbrt:
      return std::make_unique<CbrtOperationParser>();
    case kTfLiteBuiltinCeil:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::CEIL);
    case kTfLiteBuiltinConcatenation:
      return std::make_unique<ConcatenationOperationParser>();
    case kTfLiteBuiltinConv2d:
      return std::make_unique<Conv2DOperationParser>(options);
    case kTfLiteBuiltinCos:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::COS);
    case kTfLiteBuiltinCumsum:
      return std::make_unique<CumsumOperationParser>();
    case kTfLiteBuiltinDepthwiseConv2d:
      return std::make_unique<DepthwiseConvolutionOperationParser>();
    case kTfLiteBuiltinDepthToSpace:
      return std::make_unique<DepthToSpaceOperationParser>();
    case kTfLiteBuiltinDequantize:
      if (allow_quant_ops) return std::make_unique<DequantizeOperationParser>();
      break;
    case kTfLiteBuiltinDiv:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::DIV);
    case kTfLiteBuiltinDynamicUpdateSlice:
      return std::make_unique<DynamicUpdateSliceOperationParser>();
    case kTfLiteBuiltinEqual:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::EQUAL);
    case kTfLiteBuiltinElu:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::ELU);
    case kTfLiteBuiltinEmbeddingLookup:
      return std::make_unique<EmbeddingLookupOperationParser>();
    case kTfLiteBuiltinExp:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::EXP);
    case kTfLiteBuiltinFloor:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::FLOOR);
    case kTfLiteBuiltinFloorDiv:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::FLOOR_DIV);
    case kTfLiteBuiltinFloorMod:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::FLOOR_MOD);
    case kTfLiteBuiltinFullyConnected:
      return std::make_unique<FullyConnectedOperationParser>(options);
    case kTfLiteBuiltinGather:
      return std::make_unique<GatherOperationParser>();
    case kTfLiteBuiltinGelu:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::GELU);
    case kTfLiteBuiltinGreater:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::GREATER);
    case kTfLiteBuiltinGreaterEqual:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::GREATER_EQUAL);
    case kTfLiteBuiltinHardSwish:
      return std::make_unique<HardSwishOperationParser>();
    case kTfLiteBuiltinLess:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::LESS);
    case kTfLiteBuiltinLessEqual:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::LESS_EQUAL);
    case kTfLiteBuiltinLogicalAnd:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::LOGICAL_AND);
    case kTfLiteBuiltinLogicalNot:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::LOGICAL_NOT);
    case kTfLiteBuiltinLogicalOr:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::LOGICAL_OR);
    case kTfLiteBuiltinLogistic:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::SIGMOID);
    case kTfLiteBuiltinLog:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::LOG);
    case kTfLiteBuiltinLstm:
      return std::make_unique<LSTMOperationParser>();
    case kTfLiteBuiltinMaximum:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::MAXIMUM);
    case kTfLiteBuiltinMaxPool2d:
      return std::make_unique<Pooling2DOperationParser>(
          ::ml_drift::PoolingType::MAX);
    case kTfLiteBuiltinMean:
      return std::make_unique<ReduceOperationParser>(
          ::ml_drift::OperationType::MEAN);
    case kTfLiteBuiltinMinimum:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::MINIMUM);
    case kTfLiteBuiltinMirrorPad:
      return std::make_unique<PadOperationParser>(/*mirror_pad=*/true, options);
    case kTfLiteBuiltinMul:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::MUL);
    case kTfLiteBuiltinNeg:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::NEG);
    case kTfLiteBuiltinNotEqual:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::NOT_EQUAL);
    case kTfLiteBuiltinOneHot:
      return std::make_unique<OneHotOperationParser>();
    case kTfLiteBuiltinPack:
      return std::make_unique<PackOperationParser>();
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2:
      return std::make_unique<PadOperationParser>(/*mirror_pad=*/false,
                                                  options);
    case kTfLiteBuiltinPow:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::POW);
    case kTfLiteBuiltinReduceAll:
      return std::make_unique<ReduceOperationParser>(
          ::ml_drift::OperationType::REDUCE_ALL);
    case kTfLiteBuiltinReduceAny:
      return std::make_unique<ReduceOperationParser>(
          ::ml_drift::OperationType::REDUCE_ANY);
    case kTfLiteBuiltinReduceMax:
      return std::make_unique<ReduceOperationParser>(
          ::ml_drift::OperationType::REDUCE_MAXIMUM);
    case kTfLiteBuiltinReduceMin:
      return std::make_unique<ReduceOperationParser>(
          ::ml_drift::OperationType::REDUCE_MINIMUM);
    case kTfLiteBuiltinReduceProd:
      return std::make_unique<ReduceOperationParser>(
          ::ml_drift::OperationType::REDUCE_PRODUCT);
    case kTfLiteBuiltinQuantize:
      if (allow_quant_ops) {
        return std::make_unique<QuantizeOperationParser>();
      }
      break;
    case kTfLiteBuiltinRelu:
      return std::make_unique<ReLUOperationParser>(0, 0);
    case kTfLiteBuiltinRelu6:
      return std::make_unique<ReLUOperationParser>(0, 6);
    case kTfLiteBuiltinRelu0To1:
      return std::make_unique<ReLUOperationParser>(0.0, 1.0);
    case kTfLiteBuiltinReluN1To1:
      return std::make_unique<ReLUOperationParser>(-1.0, 1.0);
    case kTfLiteBuiltinLeakyRelu:
      return std::make_unique<ReLUOperationParser>(0, 0);
    case kTfLiteBuiltinPrelu:
      return std::make_unique<PReLUOperationParser>();
    case kTfLiteBuiltinStablehloRemainder:
      return std::make_unique<RemainderOperationParser>();
    case kTfLiteBuiltinReshape:
      return std::make_unique<ReshapeOperationParser>();
    case kTfLiteBuiltinResizeBilinear:
      return std::make_unique<Resize2DOperationParser>(
          ::ml_drift::SamplingType::BILINEAR);
    case kTfLiteBuiltinResizeNearestNeighbor:
      return std::make_unique<Resize2DOperationParser>(
          ::ml_drift::SamplingType::NEAREST);
    case kTfLiteBuiltinReverseV2:
      return std::make_unique<ReverseOperationParser>();
    case kTfLiteBuiltinRound:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::ROUND);
    case kTfLiteBuiltinRsqrt:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::RSQRT);
    case kTfLiteBuiltinSelect:
    case kTfLiteBuiltinSelectV2:
      return std::make_unique<SelectV2OperationParser>();
    case kTfLiteBuiltinStablehloShiftLeft:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::SHIFT_LEFT);
    case kTfLiteBuiltinRightShift:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::SHIFT_RIGHT);
    case kTfLiteBuiltinSign:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::SIGN);
    case kTfLiteBuiltinSin:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::SIN);
    case kTfLiteBuiltinSlice:
      return std::make_unique<SliceOperationParser>();
    case kTfLiteBuiltinSoftmax:
      return std::make_unique<SoftmaxOperationParser>(options);
    case kTfLiteBuiltinSpaceToDepth:
      return std::make_unique<SpaceToDepthOperationParser>();
    case kTfLiteBuiltinSplit:
      return std::make_unique<SplitOperationParser>();
    case kTfLiteBuiltinSplitV:
      return std::make_unique<SplitVOperationParser>();
    case kTfLiteBuiltinSqrt:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::SQRT);
    case kTfLiteBuiltinSquare:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::SQUARE);
    case kTfLiteBuiltinSquaredDifference:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::SQUARED_DIFF);
    case kTfLiteBuiltinStridedSlice:
      return std::make_unique<StridedSliceOperationParser>();
    case kTfLiteBuiltinSub:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::SUB);
    case kTfLiteBuiltinSum:
      return std::make_unique<ReduceOperationParser>(
          ::ml_drift::OperationType::REDUCE_SUM);
    case kTfLiteBuiltinTanh:
      return std::make_unique<ElementwiseOperationParser>(
          ::ml_drift::OperationType::TANH);
    case kTfLiteBuiltinTile:
      return std::make_unique<TileOperationParser>();
    case kTfLiteBuiltinTopkV2:
      return std::make_unique<TopKOperationParser>();
    case kTfLiteBuiltinTranspose:
      return std::make_unique<TransposeOperationParser>();
    case kTfLiteBuiltinTransposeConv:
      return std::make_unique<TransposeConvBuiltinOperationParser>();
    case kTfLiteBuiltinUnpack:
      return std::make_unique<UnpackOperationParser>();
    case kTfLiteBuiltinStablehloClamp:
      return std::make_unique<ClampOperationsParser>();
    case kTfLiteBuiltinCustom: {
      const absl::string_view custom_name = registration->custom_name;
      // TODO: b/368214363 - Remove custom op of GroupNorm and LayerNorm when
      // their dependent models are transited to StableHloComposite op.
      if (custom_name == "custom_call.GroupNorm") {
        return std::make_unique<GroupNormParser>();
      }
      if (custom_name == "custom_call.LayerNorm") {
        return std::make_unique<LayerNormParser>();
      }
      if (custom_name == "custom_call.RmsNorm") {
        return std::make_unique<RmsNormParser>();
      }
      if (custom_name == "custom_call.PixelShuffle") {
        return std::make_unique<PixelShuffleParser>();
      }
      if (custom_name == "custom_call.absolute_positional_embedding") {
        return std::make_unique<PositionalEmbeddingParser>();
      }
      if (custom_name == "custom_call.rotary_positional_embedding") {
        return std::make_unique<RoPEParser>();
      }
      if (custom_name == "Convolution2DTransposeBias") {
        return std::make_unique<TransposeConvCustomOperationParser>();
      }
      if (custom_name == "MaxPoolingWithArgmax2D") {
        return std::make_unique<Pooling2DOperationParser>(
            ::ml_drift::PoolingType::MAX);
      }
      if (custom_name == "MaxUnpooling2D") {
        return std::make_unique<Unpooling2DOperationParser>();
      }
      if (custom_name == "Resampler") {
        return std::make_unique<ResamplerOperationParser>();
      }
      if (composite_parser_factory &&
          composite_parser_factory->SupportsIntegerTypes(custom_name)) {
        return composite_parser_factory->Create(custom_name);
      }
      return NewCustomOperationParser(registration->custom_name);
    }

    case kTfLiteBuiltinStablehloComposite: {
      const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
          tflite_node->builtin_data);
      if (!params) {
        ABSL_LOG(ERROR) << "Missing StableHLO composite params.";
        return std::make_unique<UnsupportedOperationParser>();
      }
      if (!std::strcmp(params->name, "odml.group_norm")) {
        const flexbuffers::Map flexbuffer_map =
            flexbuffers::GetRoot(params->attributes, params->attributes_size)
                .AsMap();
        if (!flexbuffer_map["sub_type"].IsNull()) {
          // sub_type: 0=GroupNorm, 1=LayerNorm.
          if (flexbuffer_map["sub_type"].AsInt32() == 0) {
            return std::make_unique<CompositeGroupNormParser>();
          } else {
            return std::make_unique<CompositeLayerNormParser>();
          }
          // Below is the legacy treatment for backwards compatibility.
          // It had some issues, for instance it was backwards, see b/489399819.
        } else if (flexbuffer_map["_TENSOR_V1_reduction_axes"].IsNull()) {
          return std::make_unique<CompositeLayerNormParser>();
        } else {
          return std::make_unique<CompositeGroupNormParser>();
        }
      }
      if (!std::strcmp(params->name, "odml.rms_norm")) {
        return std::make_unique<CompositeRmsNormParser>();
      }
      if (!std::strcmp(params->name, "odml.scaled_dot_product_attention")) {
        return std::make_unique<CompositeSdpaParser>();
      }
      if (composite_parser_factory) {
        return composite_parser_factory->Create(params->name);
      }
      ABSL_LOG(ERROR) << "Unknown StableHLO composite params: " << params->name;
      return std::make_unique<UnsupportedOperationParser>();
    }
  }
  return std::make_unique<UnsupportedOperationParser>();
}

std::unique_ptr<TFLiteOperationParser> NewOperationParser(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    bool allow_quant_ops,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops) {
  return NewOperationParser(tflite_node, registration, allow_quant_ops,
                            /*options=*/{}, excluded_ops,
                            /*composite_parser_factory=*/nullptr);
}

absl::Status CheckIfSupportedNode(
    const TfLiteContext* context, const TfLiteNode* node,
    const TfLiteRegistration* registration, bool allow_quant_ops,
    const ModelBuilderOptions& options,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory) {
  return NewOperationParser(node, registration, allow_quant_ops, options,
                            excluded_ops, composite_parser_factory)
      ->IsSupported(context, node, registration);
}

// TODO(impjdi): Check number of input/output tensors and their dimensions.
// TODO(impjdi): Check ops' parameters.
TfLiteIntArray* GetOpsToReplaceWithOptions(
    TfLiteContext* context, bool allow_quant_ops,
    const ModelBuilderOptions& options, int max_delegated_partitions,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops,
    int start_node_index, int end_node_index,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory) {
  tflite::delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool {
    const auto status =
        CheckIfSupportedNode(context, node, registration, allow_quant_ops,
                             options, excluded_ops, composite_parser_factory);
    if (!status.ok()) {
      if (unsupported_details) {
        *unsupported_details = std::string(status.message());
      }
      return false;
    }

    auto add_int_types = [](std::vector<TfLiteType>* types) {
      types->push_back(kTfLiteInt8);
      types->push_back(kTfLiteInt16);
      types->push_back(kTfLiteInt32);
    };
    auto add_uint_types = [](std::vector<TfLiteType>* types) {
      types->push_back(kTfLiteUInt8);
      types->push_back(kTfLiteUInt16);
      types->push_back(kTfLiteUInt32);
    };

    std::vector<TfLiteType> allowed_in_types = {kTfLiteFloat32, kTfLiteFloat16};
    if (options.allow_bool_tensors) {
      allowed_in_types.push_back(kTfLiteBool);
    }
    std::vector<TfLiteType> allowed_out_types = {kTfLiteFloat32,
                                                 kTfLiteFloat16};
    if (allow_quant_ops) {
      // Since we only check non-constant tensors, type cannot be Int32.
      allowed_in_types.push_back(kTfLiteInt2);
      allowed_in_types.push_back(kTfLiteInt4);
      allowed_in_types.push_back(kTfLiteInt8);
      allowed_in_types.push_back(kTfLiteUInt8);
      allowed_out_types.push_back(kTfLiteInt8);
      allowed_out_types.push_back(kTfLiteUInt8);
    }
    if (registration->builtin_code == kTfLiteBuiltinArgMax) {
      allowed_out_types = {kTfLiteInt32};
    }
    if (registration->builtin_code == kTfLiteBuiltinReduceAll ||
        registration->builtin_code == kTfLiteBuiltinReduceAny) {
      if (options.allow_bool_tensors) {
        allowed_in_types = {kTfLiteBool};
        allowed_out_types = {kTfLiteBool};
      }
    }
    if (registration->builtin_code == kTfLiteBuiltinEmbeddingLookup ||
        registration->builtin_code == kTfLiteBuiltinOneHot ||
        registration->builtin_code == kTfLiteBuiltinDynamicUpdateSlice) {
      allowed_in_types.push_back(kTfLiteInt32);
    }
    if (IsCompareOpCode(registration->builtin_code)) {
      add_int_types(&allowed_in_types);
      add_uint_types(&allowed_in_types);
      if (options.allow_bool_tensors) {
        allowed_out_types.push_back(kTfLiteBool);
      }
    }
    if (IsLogicalOpCode(registration->builtin_code)) {
      if (options.allow_bool_tensors) {
        allowed_in_types = {kTfLiteBool, kTfLiteInt8, kTfLiteInt16,
                            kTfLiteInt32};
        allowed_out_types = {kTfLiteBool, kTfLiteInt8, kTfLiteInt16,
                             kTfLiteInt32};
      } else {
        allowed_in_types = {kTfLiteInt8, kTfLiteInt16, kTfLiteInt32};
        allowed_out_types = {kTfLiteInt8, kTfLiteInt16, kTfLiteInt32};
      }
    }
    if (registration->builtin_code == kTfLiteBuiltinTopkV2) {
      allowed_out_types.push_back(kTfLiteInt32);
    }
    if (IsBf16SupportedOp(registration->builtin_code)) {
      allowed_in_types.push_back(kTfLiteBFloat16);
      allowed_out_types.push_back(kTfLiteBFloat16);
    }
    if (IsIntSupportedOp(registration->builtin_code)) {
      add_int_types(&allowed_in_types);
      add_int_types(&allowed_out_types);
    }
    if (IsBoolSupportedCompositeOp(registration->builtin_code, node,
                                   composite_parser_factory)) {
      if (options.allow_bool_tensors) {
        allowed_in_types.push_back(kTfLiteBool);
        allowed_out_types.push_back(kTfLiteBool);
      }
    }
    if (IsIntSupportedCompositeOrCustomOp(registration->builtin_code, node,
                                          registration,
                                          composite_parser_factory)) {
      allowed_in_types.push_back(kTfLiteInt32);
    }
    if (SupportAllPrecisionOp(registration->builtin_code)) {
      add_int_types(&allowed_in_types);
      add_uint_types(&allowed_in_types);
      if (options.allow_bool_tensors) {
        allowed_in_types.push_back(kTfLiteBool);
      }
      allowed_in_types.push_back(kTfLiteBFloat16);
      add_int_types(&allowed_out_types);
      add_uint_types(&allowed_out_types);
      if (options.allow_bool_tensors) {
        allowed_out_types.push_back(kTfLiteBool);
      }
      allowed_out_types.push_back(kTfLiteBFloat16);
    }
    if (registration->builtin_code == kTfLiteBuiltinStablehloShiftLeft) {
      add_int_types(&allowed_in_types);
      add_int_types(&allowed_out_types);
    }
    if (!IsAllAllowedTensors(context, node->inputs, allowed_in_types,
                             unsupported_details) ||
        !IsAllAllowedTensors(context, node->outputs, allowed_out_types,
                             unsupported_details)) {
      return false;
    }
    return true;
  };

  tflite::delegates::FP16GraphPartitionHelper partition_helper(
      context, node_supported_fn);
  std::set<std::string> unsupported_nodes_info;
  if (partition_helper.Partition(&unsupported_nodes_info, start_node_index,
                                 end_node_index) != kTfLiteOk) {
    return TfLiteIntArrayCreate(0);
  }

  // By default, we simply get 1st largest partition as 'max_delegate_partions'
  // is set to 1 by default.
  std::vector<int> ops_to_replace =
      partition_helper.GetNodesOfFirstNLargestPartitions(
          max_delegated_partitions);

  if (!unsupported_nodes_info.empty() &&
      partition_helper.num_total_nodes() > ops_to_replace.size()) {
    std::string unsupported = absl::StrJoin(unsupported_nodes_info, "\n");
    std::string error_message = absl::StrCat(
        "Following operations are not supported by GPU delegate:\n",
        unsupported, "\n");
    if (!ops_to_replace.empty()) {
      absl::StrAppend(
          &error_message, ops_to_replace.size(),
          " operations will run on the GPU, and the remaining ",
          partition_helper.num_total_nodes() - ops_to_replace.size());
    } else {
      absl::StrAppend(&error_message,
                      "No operations will run on the GPU, and all ",
                      partition_helper.num_total_nodes());
    }
    absl::StrAppend(&error_message, " operations will run on the CPU.");
    TF_LITE_KERNEL_LOG(context, "%s", error_message.c_str());
  }
  return tflite::ConvertVectorToTfLiteIntArray(ops_to_replace);
}

TfLiteIntArray* GetOpsToReplace(
    TfLiteContext* context, bool allow_quant_ops, int max_delegated_partitions,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops,
    int start_node_index, int end_node_index,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory) {
  return GetOpsToReplaceWithOptions(
      context, allow_quant_ops, /*options=*/{}, max_delegated_partitions,
      excluded_ops, start_node_index, end_node_index, composite_parser_factory);
}

// Creates inputs and outputs passed by io_tensors parameters in the resulting
// graph. We force it to make sure that delegated subgraph has same order of
// inputs and outputs with the original one. When delegated model is built from
// the tflite model representation tensors are created lazily, so there is no
// guarantee that the order will match the source model tensors order.
absl::Status PrecreateIOTensors(
    TfLiteContext* context, ::ml_drift::GraphFloat32* graph,
    const std::vector<int>& io_ids,
    absl::flat_hash_map<int, int>* quant_conversion_map,
    absl::flat_hash_map<int, ::ml_drift::Value*>* tensor_to_value,
    bool is_output) {
  for (const auto& id : io_ids) {
    const TfLiteTensor& tflite_tensor = context->tensors[id];
    if (tflite::IsConstantTensor(&tflite_tensor)) continue;
    ::ml_drift::Value* value = ObjectReader::ReadNonConstantTensor(
        context, tensor_to_value, quant_conversion_map, graph, id);
    if (is_output) graph->AddKnownGraphOutput(value);
  }
  return absl::OkStatus();
}

absl::Status CopyVariableTensorOutputs(
    TfLiteNode* tflite_node, TfLiteRegistration* registration,
    ::ml_drift::GraphFloat32* graph, ObjectReader& reader,
    const absl::flat_hash_map<int, ::ml_drift::ValueId>&
        new_variable_tensor_values) {
  absl::flat_hash_map<int, ::ml_drift::ValueId> new_variable_tensor_values_copy(
      new_variable_tensor_values);
  // Retrieve the final value id for the variable input tensors.
  for (int i = 0; i < tflite_node->inputs->size; i++) {
    const int tensor_idx = tflite_node->inputs->data[i];
    if (!reader.CanReadValueByTensorIdx(tensor_idx)) continue;
    ::ml_drift::Value* value = reader.ReadValueByTensorIdx(tensor_idx);
    if (value && value->tensor.is_variable_input) {
      if (new_variable_tensor_values_copy.find(i) ==
          new_variable_tensor_values_copy.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat(tflite::GetOpNameByRegistration(*registration),
                         " did not provide a new value for the variable input "
                         "tensor with index ",
                         tensor_idx));
      } else {
        ::ml_drift::Node* node = graph->NewNode();
        node->operation.type = ToString(::ml_drift::OperationType::COPY);
        graph->AddConsumer(node->id, new_variable_tensor_values_copy.at(i));
        reader.AddUpdate(node, i);
        new_variable_tensor_values_copy.erase(
            new_variable_tensor_values_copy.find(i));
      }
    }
  }
  if (!new_variable_tensor_values_copy.empty()) {
    return absl::InvalidArgumentError(
        "More input variable tensors asked to be copied than present on the "
        "node");
  }
  return absl::OkStatus();
}

absl::Status BuildModel(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const ModelBuilderOptions& options, ::ml_drift::GraphFloat32* graph,
    absl::flat_hash_map<int, int>* quant_conversion_map,
    SharedConstTensorsMap* shared_tensors,
    const TensorIndexToBufferIdMap* tensor_to_buffer_id_map,
    const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory) {
  std::vector<int> inputs(delegate_params->input_tensors->size);
  std::vector<int> outputs(delegate_params->output_tensors->size);
  for (int i = 0; i < delegate_params->input_tensors->size; i++) {
    inputs[i] = delegate_params->input_tensors->data[i];
  }
  for (int i = 0; i < delegate_params->output_tensors->size; i++) {
    outputs[i] = delegate_params->output_tensors->data[i];
  }
  return BuildModelEnforceIO(
      context, delegate_params, options, inputs, outputs, graph,
      quant_conversion_map, shared_tensors, tensor_to_buffer_id_map,
      tensor_to_external_buffer_id_map, composite_parser_factory);
}

absl::Status BuildModelEnforceIO(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const ModelBuilderOptions& options, const std::vector<int>& input_ids,
    const std::vector<int>& output_ids, ::ml_drift::GraphFloat32* graph,
    absl::flat_hash_map<int, int>* quant_conversion_map,
    SharedConstTensorsMap* shared_tensors,
    const TensorIndexToBufferIdMap* tensor_to_buffer_id_map,
    const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory) {
  std::vector<int> filtered_input_ids;
  filtered_input_ids.reserve(input_ids.size());
  for (int input_id : input_ids) {
    const bool shared_external_tensor =
        shared_tensors != nullptr &&
        tensor_to_external_buffer_id_map != nullptr &&
        tensor_to_external_buffer_id_map->find(input_id) !=
            tensor_to_external_buffer_id_map->end();
    if (!shared_external_tensor) {
      filtered_input_ids.push_back(input_id);
    }
  }
  std::vector<std::unique_ptr<TFLiteOperationParser>> operations;
  std::vector<int> tflite_nodes;
  for (int i = 0; i < delegate_params->nodes_to_replace->size; ++i) {
    TfLiteNode* tflite_node = nullptr;
    TfLiteRegistration* registration = nullptr;
    ABSL_RETURN_IF_ERROR(GetNodeAndRegistration(
        context, delegate_params->nodes_to_replace->data[i], &tflite_node,
        &registration));
    if (registration->builtin_code == kTfLiteBuiltinDequantize &&
        context->tensors[tflite_node->inputs->data[0]].type ==
            TfLiteType::kTfLiteFloat16 &&
        context->tensors[tflite_node->inputs->data[0]].allocation_type ==
            TfLiteAllocationType::kTfLiteMmapRo) {
      // Ignore Fp16 Dequantize nodes only if they are the final nodes before
      // weights, i.e., no other nodes preceded them.
      continue;
    }
    auto op_parser = NewOperationParser(
        tflite_node, registration,
        /*allow_quant_ops=*/quant_conversion_map != nullptr, options,
        /*excluded_ops=*/nullptr, composite_parser_factory);
    if (!op_parser) {
      return absl::UnimplementedError(
          absl::StrCat("Operation ", registration->builtin_code, "(",
                       registration->custom_name,
                       ") is not supported by TFLite GPU Delegate."));
    }
    // If the StableHlo Composite op is supported, then ML Drift will lower it
    // by itself, rather than using the default subgraph stored in the
    // flatbuffer, so the default subgraph should be marked as skippable to
    // avoid the waste of being delegated to ML Drift Delegate by the Runtime.
    if (registration->builtin_code == kTfLiteBuiltinStablehloComposite) {
      auto* current_subgraph = static_cast<::tflite::Subgraph*>(context->impl_);
      const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
          tflite_node->builtin_data);
      if (!params) {
        return absl::InternalError("Missing StableHLO composite op params.");
      }
      if (current_subgraph->MarkSubgraphAsDelegationSkippable(
              params->subgraph_index) != kTfLiteOk) {
        return absl::InternalError(
            "Failed to mark subgraph as delegation skippable.");
      }
    }
    operations.push_back(std::move(op_parser));
    tflite_nodes.push_back(i);
  }
  absl::flat_hash_map<int, ::ml_drift::Value*> tensor_to_value;
  std::vector<::ml_drift::ValueId> variable_inputs_to_value_id;
  ABSL_RETURN_IF_ERROR(PrecreateIOTensors(context, graph, filtered_input_ids,
                                          quant_conversion_map,
                                          &tensor_to_value,
                                          /*is_output=*/false));
  ABSL_RETURN_IF_ERROR(PrecreateIOTensors(
      context, graph, output_ids, quant_conversion_map, &tensor_to_value,
      /*is_output=*/true));
  for (int i = 0; i < operations.size(); ++i) {
    TfLiteNode* tflite_node;
    TfLiteRegistration* registration;
    ABSL_RETURN_IF_ERROR(GetNodeAndRegistration(
        context, delegate_params->nodes_to_replace->data[tflite_nodes[i]],
        &tflite_node, &registration));
    ObjectReader reader(graph, context, tflite_node, &tensor_to_value,
                        quant_conversion_map, tensor_to_buffer_id_map,
                        tensor_to_external_buffer_id_map, shared_tensors);
    operations[i]->Parse(tflite_node, registration, graph, &reader);
    absl::flat_hash_map<int, ::ml_drift::ValueId>
        new_value_for_variable_input_tensors =
            operations[i]->GetNewValueIdsForVariableInputNodes();
    ABSL_RETURN_IF_ERROR(
        CopyVariableTensorOutputs(tflite_node, registration, graph, reader,
                                  new_value_for_variable_input_tensors));
  }

  // Variable input tensors expect to be unchanged throughout model execution.
  // They need to be an output of the graph in order to have them unchanged.
  for (auto value_id : variable_inputs_to_value_id) {
    if (!graph->IsGraphOutput(value_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Variable input tensors must be a graph output. Value ",
                       value_id, " is not a graph output"));
    }
  }
  return absl::OkStatus();
}

absl::Status BuildFinalModel(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const ModelBuilderOptions& options, ::ml_drift::GraphFloat32* graph,
    absl::flat_hash_map<int, int>* quant_conversion_map,
    SharedConstTensorsMap* shared_tensors,
    const TensorIndexToBufferIdMap* tensor_to_buffer_id_map,
    const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map,
    TFLiteStablehloCompositeParserFactory* composite_parser_factory) {
  ABSL_RETURN_IF_ERROR(
      BuildModel(context, delegate_params, options, graph, quant_conversion_map,
                 shared_tensors, tensor_to_buffer_id_map,
                 tensor_to_external_buffer_id_map, composite_parser_factory));

  return ApplyGpuModelTransformations(graph);
}

namespace {

class DelegateContext {
 public:
  struct DelegateData {
    std::vector<int> input_ids;
    std::vector<int> output_ids;
    ::ml_drift::GraphFloat32* graph;
    ModelBuilderOptions options;
    std::unique_ptr<absl::flat_hash_map<int, int>> quant_conversion_map;
  };
  bool Init(TfLiteContext* context,
            const TfLiteDelegateParams* delegate_params) {
    const auto* delegate_data =
        reinterpret_cast<DelegateData*>(delegate_params->delegate->data_);
    if (!delegate_data->graph) {
      TF_LITE_KERNEL_LOG(context, "Invalid delegate_data->graph");
      return false;
    }
    auto build_status = BuildModelEnforceIO(
        context, delegate_params, delegate_data->options,
        delegate_data->input_ids, delegate_data->output_ids,
        delegate_data->graph, delegate_data->quant_conversion_map.get(),
        /*shared_tensors=*/nullptr);
    if (!build_status.ok()) {
      TF_LITE_KERNEL_LOG(context, "%s",
                         std::string(build_status.message()).c_str());
      return false;
    }
    return true;
  }
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteRegistration registration{};
  registration.init = [](TfLiteContext* context, const char* buffer,
                         size_t) -> void* {
    auto* delegate_context = new DelegateContext();
    if (!delegate_context->Init(
            context, reinterpret_cast<const TfLiteDelegateParams*>(buffer))) {
      delete delegate_context;
      return nullptr;
    }
    return delegate_context;
  };
  registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<DelegateContext*>(buffer);
  };
  registration.prepare = [](TfLiteContext* context,
                            TfLiteNode* node) -> TfLiteStatus {
    return node->user_data ? kTfLiteOk : kTfLiteError;
  };

  const auto* delegate_data =
      reinterpret_cast<const DelegateContext::DelegateData*>(delegate->data_);
  TfLiteIntArray* ops_to_replace = GetOpsToReplaceWithOptions(
      context, static_cast<bool>(delegate_data->quant_conversion_map),
      delegate_data->options, /*max_delegated_partitions=*/1,
      /*excluded_ops=*/nullptr, /*start_node_index=*/0,
      /*end_node_index=*/std::numeric_limits<int>::max(),
      /*composite_parser_factory=*/nullptr);
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, registration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace

absl::Status BuildFromFlatBuffer(const tflite::FlatBufferModel& flatbuffer,
                                 const tflite::OpResolver& op_resolver,
                                 ::ml_drift::GraphFloat32* graph,
                                 bool allow_quant_ops,
                                 bool apply_model_transformations) {
  return BuildFromFlatBuffer(flatbuffer, op_resolver, /*options=*/{}, graph,
                             allow_quant_ops, apply_model_transformations);
}

absl::Status BuildFromFlatBuffer(const tflite::FlatBufferModel& flatbuffer,
                                 const tflite::OpResolver& op_resolver,
                                 const ModelBuilderOptions& options,
                                 ::ml_drift::GraphFloat32* graph,
                                 bool allow_quant_ops,
                                 bool apply_model_transformations) {
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(flatbuffer, op_resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk || !interpreter) {
    return absl::InternalError("Unable to prepare TfLite interpreter.");
  }
  TfLiteDelegate delegate = TfLiteDelegateCreate();

  DelegateContext::DelegateData delegate_data{
      interpreter->inputs(), interpreter->outputs(), graph, options};
  if (allow_quant_ops) {
    delegate_data.quant_conversion_map =
        std::make_unique<absl::flat_hash_map<int, int>>();
  }

  delegate.data_ = &delegate_data;
  delegate.flags = kTfLiteDelegateFlagsNone;
  delegate.Prepare = DelegatePrepare;
  delegate.CopyFromBufferHandle = nullptr;
  delegate.CopyToBufferHandle = nullptr;
  delegate.FreeBufferHandle = nullptr;

  if (interpreter->ModifyGraphWithDelegate(&delegate) != kTfLiteOk) {
    return absl::InternalError("Conversion from TfLite model failed.");
  }

  if (apply_model_transformations) {
    return ApplyGpuModelTransformations(graph);
  }

  return absl::OkStatus();
}

}  // namespace litert::ml_drift
