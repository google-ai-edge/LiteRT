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
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
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

  *new_tensor =
      &builder->BuildTensor(weights_t, quantization, tensor_type, name);
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

#ifdef __cplusplus
}  // extern "C"
#endif
