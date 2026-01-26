// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/c/litert_builder.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Builder
//

LiteRtStatus LiteRtBuilderBuildTensor(
    LiteRtBuilder builder, LiteRtTensorTypeId tensor_type_id,
    LiteRtRankedTensorType ranked_tensor_type,
    LiteRtUnrankedTensorType unranked_tensor_type, LiteRtWeights weights,
    LiteRtQuantizationTypeId quantization_type_id,
    LiteRtQuantizationPerTensor per_tensor_quantization,
    LiteRtQuantizationPerChannel per_channel_quantization, const char* name,
    LiteRtTensor* new_tensor) {
  // Pack tensor type to internal type.
  TensorType tensor_type;
  tensor_type.first = tensor_type_id;
  if (tensor_type_id == kLiteRtRankedTensorType) {
    tensor_type.second.ranked_tensor_type = ranked_tensor_type;
  } else if (tensor_type_id == kLiteRtUnrankedTensorType) {
    tensor_type.second.unranked_tensor_type = unranked_tensor_type;
    if (weights) {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }

  // Pack quantization type to internal type.
  Quantization quantization;

  switch (quantization_type_id) {
    case kLiteRtQuantizationPerTensor:
      quantization.first = kLiteRtQuantizationPerTensor;
      quantization.second.per_tensor = per_tensor_quantization;
      break;
    case kLiteRtQuantizationPerChannel:
      quantization.first = kLiteRtQuantizationPerChannel;
      quantization.second.per_channel = per_channel_quantization;
      break;
    case kLiteRtQuantizationNone:
      quantization.first = kLiteRtQuantizationNone;
      break;
    default:
      return kLiteRtStatusErrorInvalidArgument;
  }

  // Pack weights to internal type.
  LiteRtWeightsT weights_t;
  if (weights) {
    weights_t.SetBufferId(weights->GetBufferId());
    ::litert::internal::BufferManager::BufferId buffer_id =
        weights->GetBufferId();
    if (buffer_id == 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    ::litert::internal::BufferManager* buffer_manager =
        weights->GetBufferManager();
    if (buffer_manager == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    weights_t.SetBufferId(buffer_id);
    weights_t.SetBufferManager(buffer_manager);
  } else {
    // Set the buffer manager to nullptr, this allows the internal builder to
    // identify this is a null weights.
    weights_t.SetBufferManager(nullptr);
  }

  std::optional<std::string> tensor_name;
  if (name != nullptr) {
    tensor_name = name;
  }

  *new_tensor =
      &builder->BuildTensor(weights_t, quantization, tensor_type, tensor_name);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildWeights(LiteRtBuilder builder,
                                       const uint8_t* data,
                                       LiteRtParamIndex size,
                                       LiteRtTensor tensor,

                                       LiteRtWeights* new_weights) {
  if (builder == nullptr || tensor == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *new_weights = &builder->BuildWeights(data, size, tensor);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildOp(LiteRtBuilder builder, LiteRtOpCode op_code,
                                  LiteRtParamIndex num_inputs,
                                  LiteRtTensor* inputs,
                                  LiteRtParamIndex num_outputs,
                                  LiteRtTensor* outputs, LiteRtOp* new_op) {
  if (builder == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::vector<LiteRtTensor> input_tensors;
  std::vector<LiteRtTensor> output_tensors;

  for (int i = 0; i < num_inputs; ++i) {
    if (inputs[i] == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    input_tensors.push_back(inputs[i]);
  }

  for (int i = 0; i < num_outputs; ++i) {
    if (outputs[i] == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    output_tensors.push_back(outputs[i]);
  }

  *new_op = &builder->BuildOp(op_code, input_tensors, output_tensors);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderEraseOp(LiteRtBuilder builder, LiteRtOp op_to_erase) {
  if (builder == nullptr || op_to_erase == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  builder->EraseOp(op_to_erase);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildAddOpOption(LiteRtBuilder builder, LiteRtOp op,
                                           uint32_t* fused_activation) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_AddOptions;
  auto options = std::make_unique<tflite::AddOptionsT>();
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildBatchMatmulOpOption(
    LiteRtBuilder builder, LiteRtOp op, bool* adj_x, bool* adj_y,
    bool* asymmetric_quantize_input) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
  auto options = std::make_unique<tflite::BatchMatMulOptionsT>();
  options->adj_x = *adj_x;
  options->adj_y = *adj_y;
  options->asymmetric_quantize_inputs = *asymmetric_quantize_input;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildConcatenationOpOption(LiteRtBuilder builder,
                                                     LiteRtOp op,
                                                     uint32_t* fused_activation,
                                                     int32_t* axis) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  auto options = std::make_unique<tflite::ConcatenationOptionsT>();
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation);
  options->axis = *axis;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildDivOpOption(LiteRtBuilder builder, LiteRtOp op,
                                           uint32_t* fused_activation) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflDiv) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DivOptions;
  auto options = std::make_unique<tflite::DivOptionsT>();
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildFullyConnectedOpOption(
    LiteRtBuilder builder, LiteRtOp op, uint32_t* fused_activation,
    uint32_t* weights_format, bool* keep_num_dims,
    uint32_t* quantized_bias_type, bool* asymmetric_quantize_input) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_FullyConnectedOptions;
  auto options = std::make_unique<tflite::FullyConnectedOptionsT>();
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation);
  options->weights_format =
      static_cast<tflite::FullyConnectedOptionsWeightsFormat>(*weights_format);
  options->keep_num_dims = *keep_num_dims;
  options->quantized_bias_type =
      static_cast<tflite::TensorType>(*quantized_bias_type);
  options->asymmetric_quantize_inputs = *asymmetric_quantize_input;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildMulOpOption(LiteRtBuilder builder, LiteRtOp op,
                                           uint32_t* fused_activation) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflMul) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_MulOptions;
  auto options = std::make_unique<tflite::MulOptionsT>();
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildSoftmaxOpOption(LiteRtBuilder builder,
                                               LiteRtOp op, float* beta) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflSoftmax) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SoftmaxOptions;
  auto options = std::make_unique<tflite::SoftmaxOptionsT>();
  options->beta = *beta;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildStridedSliceOpOption(
    LiteRtBuilder builder, LiteRtOp op, int32_t* begin_mask, int32_t* end_mask,
    int32_t* ellipsis_mask, int32_t* new_axis_mask, int32_t* shrink_axis_mask,
    bool* offset) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_StridedSliceOptions;
  auto options = std::make_unique<tflite::StridedSliceOptionsT>();
  options->begin_mask = *begin_mask;
  options->end_mask = *end_mask;
  options->ellipsis_mask = *ellipsis_mask;
  options->new_axis_mask = *new_axis_mask;
  options->shrink_axis_mask = *shrink_axis_mask;
  options->offset = *offset;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildSubOpOption(LiteRtBuilder builder, LiteRtOp op,
                                           uint32_t* fused_activation) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflSub) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SubOptions;
  auto options = std::make_unique<tflite::SubOptionsT>();
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildReshapeOpOption(LiteRtBuilder builder,
                                               LiteRtOp op, int32_t* new_shape,
                                               int32_t new_shape_size) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflReshape) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape.assign(new_shape, new_shape + new_shape_size);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildSumOpOption(LiteRtBuilder builder, LiteRtOp op,
                                           bool* keepdims) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflSum) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReducerOptions;
  auto options = std::make_unique<tflite::ReducerOptionsT>();
  options->keep_dims = *keepdims;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildReduceMaxOpOption(LiteRtBuilder builder,
                                                 LiteRtOp op, bool* keepdims) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflReduceMax) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReducerOptions;
  auto options = std::make_unique<tflite::ReducerOptionsT>();
  options->keep_dims = *keepdims;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildReduceMinOpOption(LiteRtBuilder builder,
                                                 LiteRtOp op, bool* keepdims) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflReduceMin) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReducerOptions;
  auto options = std::make_unique<tflite::ReducerOptionsT>();
  options->keep_dims = *keepdims;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildReduceAnyOpOption(LiteRtBuilder builder,
                                                 LiteRtOp op, bool* keepdims) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflReduceAny) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReducerOptions;
  auto options = std::make_unique<tflite::ReducerOptionsT>();
  options->keep_dims = *keepdims;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildReduceAllOpOption(LiteRtBuilder builder,
                                                 LiteRtOp op, bool* keepdims) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflReduceAll) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReducerOptions;
  auto options = std::make_unique<tflite::ReducerOptionsT>();
  options->keep_dims = *keepdims;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildPackOpOption(LiteRtBuilder builder, LiteRtOp op,
                                            int32_t* axis,
                                            int32_t* values_count) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflPack) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  auto options = std::make_unique<tflite::PackOptionsT>();
  options->axis = *axis;
  options->values_count = *values_count;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildUnpackOpOption(LiteRtBuilder builder,
                                              LiteRtOp op, int32_t* axis,
                                              int32_t* num) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflUnpack) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_UnpackOptions;
  auto options = std::make_unique<tflite::UnpackOptionsT>();
  options->axis = *axis;
  options->num = *num;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildGatherOpOption(LiteRtBuilder builder,
                                              LiteRtOp op, int32_t* axis,
                                              int32_t* batch_dims) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflGather) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = *axis;
  options->batch_dims = *batch_dims;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildMeanOpOption(LiteRtBuilder builder, LiteRtOp op,
                                            bool* keepdims) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflMean) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReducerOptions;
  auto options = std::make_unique<tflite::ReducerOptionsT>();
  options->keep_dims = *keepdims;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildSplitOpOption(LiteRtBuilder builder, LiteRtOp op,
                                             int32_t* num_splits) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflSplit) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SplitOptions;
  auto options = std::make_unique<tflite::SplitOptionsT>();
  options->num_splits = *num_splits;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildConv2dOpOption(
    LiteRtBuilder builder, LiteRtOp op, uint32_t* padding, int32_t* stride_w,
    int32_t* stride_h, int32_t* dilation_w_factor, int32_t* dilation_h_factor,
    uint32_t* fused_activation_function) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = static_cast<tflite::Padding>(*padding);
  options->stride_w = *stride_w;
  options->stride_h = *stride_h;
  options->dilation_w_factor = *dilation_w_factor;
  options->dilation_h_factor = *dilation_h_factor;
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation_function);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildConv3dOpOption(
    LiteRtBuilder builder, LiteRtOp op, uint32_t* padding, int32_t* stride_w,
    int32_t* stride_h, int32_t* stride_d, int32_t* dilation_w_factor,
    int32_t* dilation_h_factor, int32_t* dilation_d_factor,
    uint32_t* fused_activation_function) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv3DOptions;
  auto options = std::make_unique<tflite::Conv3DOptionsT>();
  options->padding = static_cast<tflite::Padding>(*padding);
  options->stride_w = *stride_w;
  options->stride_h = *stride_h;
  options->stride_d = *stride_d;
  options->dilation_w_factor = *dilation_w_factor;
  options->dilation_h_factor = *dilation_h_factor;
  options->dilation_d_factor = *dilation_d_factor;
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation_function);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildDepthwiseConv2dOpOption(
    LiteRtBuilder builder, LiteRtOp op, uint32_t* padding, int32_t* stride_w,
    int32_t* stride_h, int32_t* depth_multiplier,
    uint32_t* fused_activation_function, int32_t* dilation_w_factor,
    int32_t* dilation_h_factor) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  auto options = std::make_unique<tflite::DepthwiseConv2DOptionsT>();
  options->padding = static_cast<tflite::Padding>(*padding);
  options->stride_w = *stride_w;
  options->stride_h = *stride_h;
  options->depth_multiplier = *depth_multiplier;
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation_function);
  options->dilation_w_factor = *dilation_w_factor;
  options->dilation_h_factor = *dilation_h_factor;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildTransposeConvOpOption(
    LiteRtBuilder builder, LiteRtOp op, uint32_t* padding, int32_t* stride_w,
    int32_t* stride_h, uint32_t* fused_activation_function) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflTransposeConv) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_TransposeConvOptions;
  auto options = std::make_unique<tflite::TransposeConvOptionsT>();
  options->padding = static_cast<tflite::Padding>(*padding);
  options->stride_w = *stride_w;
  options->stride_h = *stride_h;
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation_function);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildAveragePool2dOpOption(
    LiteRtBuilder builder, LiteRtOp op, uint32_t* padding, int32_t* stride_w,
    int32_t* stride_h, int32_t* filter_width, int32_t* filter_height,
    uint32_t* fused_activation_function) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  auto options = std::make_unique<tflite::Pool2DOptionsT>();
  options->padding = static_cast<tflite::Padding>(*padding);
  options->stride_w = *stride_w;
  options->stride_h = *stride_h;
  options->filter_width = *filter_width;
  options->filter_height = *filter_height;
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation_function);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildMaxPool2dOpOption(
    LiteRtBuilder builder, LiteRtOp op, uint32_t* padding, int32_t* stride_w,
    int32_t* stride_h, int32_t* filter_width, int32_t* filter_height,
    uint32_t* fused_activation_function) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflMaxPool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  auto options = std::make_unique<tflite::Pool2DOptionsT>();
  options->padding = static_cast<tflite::Padding>(*padding);
  options->stride_w = *stride_w;
  options->stride_h = *stride_h;
  options->filter_width = *filter_width;
  options->filter_height = *filter_height;
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation_function);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildL2Pool2dOpOption(
    LiteRtBuilder builder, LiteRtOp op, uint32_t* padding, int32_t* stride_w,
    int32_t* stride_h, int32_t* filter_width, int32_t* filter_height,
    uint32_t* fused_activation_function) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflL2Pool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  auto options = std::make_unique<tflite::Pool2DOptionsT>();
  options->padding = static_cast<tflite::Padding>(*padding);
  options->stride_w = *stride_w;
  options->stride_h = *stride_h;
  options->filter_width = *filter_width;
  options->filter_height = *filter_height;
  options->fused_activation_function =
      static_cast<tflite::ActivationFunctionType>(*fused_activation_function);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildResizeBilinearOpOption(
    LiteRtBuilder builder, LiteRtOp op, bool* align_corners,
    bool* half_pixel_centers) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflResizeBilinear) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ResizeBilinearOptions;
  auto options = std::make_unique<tflite::ResizeBilinearOptionsT>();
  options->align_corners = *align_corners;
  options->half_pixel_centers = *half_pixel_centers;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildLeakyReluOpOption(LiteRtBuilder builder,
                                                 LiteRtOp op, float* alpha) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflLeakyRelu) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_LeakyReluOptions;
  auto options = std::make_unique<tflite::LeakyReluOptionsT>();
  options->alpha = *alpha;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildDepthToSpaceOpOption(LiteRtBuilder builder,
                                                    LiteRtOp op,
                                                    int32_t* block_size) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflDepthToSpace) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthToSpaceOptions;
  auto options = std::make_unique<tflite::DepthToSpaceOptionsT>();
  options->block_size = *block_size;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildSpaceToDepthOpOption(LiteRtBuilder builder,
                                                    LiteRtOp op,
                                                    int32_t* block_size) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflSpaceToDepth) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SpaceToDepthOptions;
  auto options = std::make_unique<tflite::SpaceToDepthOptionsT>();
  options->block_size = *block_size;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildResizeNearestNeighborOpOption(
    LiteRtBuilder builder, LiteRtOp op, bool* align_corners,
    bool* half_pixel_centers) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflResizeNearestNeighbor) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ResizeNearestNeighborOptions;
  auto options = std::make_unique<tflite::ResizeNearestNeighborOptionsT>();
  options->align_corners = *align_corners;
  options->half_pixel_centers = *half_pixel_centers;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildCumsumOpOption(LiteRtBuilder builder,
                                              LiteRtOp op, bool* exclusive,
                                              bool* reverse) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflCumsum) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_CumsumOptions;
  auto options = std::make_unique<tflite::CumsumOptionsT>();
  options->exclusive = *exclusive;
  options->reverse = *reverse;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildGeluOpOption(LiteRtBuilder builder, LiteRtOp op,
                                            bool* approximate) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflGelu) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GeluOptions;
  auto options = std::make_unique<tflite::GeluOptionsT>();
  options->approximate = *approximate;
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildMirrorPadOpOption(LiteRtBuilder builder,
                                                 LiteRtOp op, uint32_t* mode) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflMirrorPad) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_MirrorPadOptions;
  auto options = std::make_unique<tflite::MirrorPadOptionsT>();
  options->mode = static_cast<tflite::MirrorPadMode>(*mode);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBuilderBuildSqueezeOpOption(LiteRtBuilder builder,
                                               LiteRtOp op,
                                               const int32_t* squeeze_dims,
                                               int32_t num_squeeze_dims) {
  if (builder == nullptr || op == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!builder->IsOpAllocated(op)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->OpCode() != kLiteRtOpCodeTflSqueeze) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SqueezeOptions;
  auto options = std::make_unique<tflite::SqueezeOptionsT>();
  options->squeeze_dims.assign(squeeze_dims, squeeze_dims + num_squeeze_dims);
  tfl_options.value = options.release();
  litert::internal::SetTflOptions(*op, std::move(tfl_options));
  return kLiteRtStatusOk;
}

#ifdef __cplusplus
}  // extern "C"
#endif
