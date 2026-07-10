// Copyright 2026 Google LLC.
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

#include "third_party/odml/litert/ml_drift/delegate/composite/moe_experts_parser.h"

#include <string>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/object_reader.h"
#include "third_party/odml/litert/ml_drift/tflite/operation_parser.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift {
namespace {

constexpr int kFp32InputCount = 7;
constexpr int kInt8InputCount = 10;

constexpr int kInputSrc = 0;
constexpr int kInputTopWeights = 1;
constexpr int kInputTopIndices = 2;
constexpr int kInputFp32GateWeight = 3;
constexpr int kInputFp32Ff1Weight = 4;
constexpr int kInputFp32LinearWeight = 5;
constexpr int kInputFp32PerExpertScale = 6;
constexpr int kInputInt8GateWeight = 3;
constexpr int kInputInt8GateScale = 4;
constexpr int kInputInt8Ff1Weight = 5;
constexpr int kInputInt8Ff1Scale = 6;
constexpr int kInputInt8LinearWeight = 7;
constexpr int kInputInt8LinearScale = 8;
constexpr int kInputInt8PerExpertScale = 9;

bool IsMissing(const flexbuffers::Map& map, absl::string_view key) {
  return map[std::string(key)].IsNull();
}

absl::Status ValidateShape(const ::ml_drift::BHWC& shape,
                           const ::ml_drift::BHWC& expected,
                           absl::string_view name) {
  if (shape != expected) {
    return absl::InvalidArgumentError(absl::StrCat(
        "moe expected ", name, " shape [", expected.b, ", ",
        expected.h, ", ", expected.w, ", ", expected.c, "], got [", shape.b,
        ", ", shape.h, ", ", shape.w, ", ", shape.c, "]."));
  }
  return absl::OkStatus();
}

absl::Status ValidateExpertWeight(const TfLiteTensor* tensor, int experts,
                                  int out_channels, int in_channels,
                                  absl::string_view name) {
  const ::ml_drift::BHWC shape = ExtractTensorShape(tensor);
  return ValidateShape(
      shape, ::ml_drift::BHWC(out_channels, experts, 1, in_channels), name);
}

absl::Status ValidateScale(const TfLiteTensor* tensor, int channels,
                           absl::string_view name) {
  if (tensor->type != kTfLiteFloat32 && tensor->type != kTfLiteFloat16) {
    return absl::InvalidArgumentError(absl::StrCat(
        "moe expects ", name, " to be float32 or float16."));
  }
  const ::ml_drift::BHWC shape = ExtractTensorShape(tensor);
  return ValidateShape(shape, ::ml_drift::BHWC(1, 1, 1, channels), name);
}

absl::Status ValidateExpertScale(const TfLiteTensor* tensor, int experts,
                                 int out_channels, absl::string_view name) {
  if (tensor->type != kTfLiteFloat32 && tensor->type != kTfLiteFloat16) {
    return absl::InvalidArgumentError(absl::StrCat(
        "moe expects ", name, " to be float32 or float16."));
  }
  const ::ml_drift::BHWC shape = ExtractTensorShape(tensor);
  return ValidateShape(shape, ::ml_drift::BHWC(out_channels, experts, 1, 1),
                       name);
}

absl::Status ValidateInt8ZeroPoint(const TfLiteTensor* tensor,
                                   absl::string_view name) {
  if (tensor->quantization.type != kTfLiteAffineQuantization ||
      tensor->quantization.params == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "moe expects affine quantization for ", name, "."));
  }
  const auto* quant =
      static_cast<const TfLiteAffineQuantization*>(tensor->quantization.params);
  if (!quant->zero_point) {
    return absl::InvalidArgumentError(
        absl::StrCat("moe missing zero points for ", name, "."));
  }
  for (int i = 0; i < quant->zero_point->size; ++i) {
    if (quant->zero_point->data[i] != 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("moe only supports symmetric int8 "
                       "weights; non-zero zero point in ",
                       name, "."));
    }
  }
  return absl::OkStatus();
}

bool IsMissingCustomOptions(const TfLiteNode *tflite_node,
                            const TfLiteRegistration *registration) {
  return registration != nullptr &&
         registration->builtin_code == kTfLiteBuiltinCustom &&
         (tflite_node->custom_initial_data == nullptr ||
          tflite_node->custom_initial_data_size == 0);
}

absl::StatusOr<flexbuffers::Map> ReadAttributeMap(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration) {
  if (registration == nullptr) {
    return absl::InvalidArgumentError("moe is missing registration.");
  }
  if (registration->builtin_code == kTfLiteBuiltinCustom) {
      if (IsMissingCustomOptions(tflite_node, registration)) {
      return absl::InvalidArgumentError("moe is missing custom options.");
    }
    return flexbuffers::GetRoot(
               static_cast<const uint8_t*>(tflite_node->custom_initial_data),
               tflite_node->custom_initial_data_size)
        .AsMap();
  }
  return absl::InvalidArgumentError("moe has unsupported op carrier.");
}

absl::StatusOr<MoeExpertsAttributes> ReadAttributes(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration) {
  ASSIGN_OR_RETURN(const flexbuffers::Map map,
                   ReadAttributeMap(tflite_node, registration));
  for (absl::string_view key : {"num_experts", "num_active_experts",
                                "model_dim", "hidden_dim", "weight_type"}) {
    if (IsMissing(map, key)) {
      return absl::InvalidArgumentError(
          absl::StrCat("moe is missing ", key, "."));
    }
  }
  if (!IsMissing(map, "activation") &&
      map["activation"].AsString().str() != "gelu") {
    return absl::InvalidArgumentError(
        "moe only supports activation='gelu'.");
  }
  if (!IsMissing(map, "renormalized_top_weights") &&
      !map["renormalized_top_weights"].AsBool()) {
    return absl::InvalidArgumentError(
        "moe expects renormalized_top_weights=true.");
  }

  MoeExpertsAttributes attr;
  attr.num_experts = map["num_experts"].AsInt32();
  attr.num_active_experts = map["num_active_experts"].AsInt32();
  attr.model_dim = map["model_dim"].AsInt32();
  attr.hidden_dim = map["hidden_dim"].AsInt32();
  const std::string weight_type = map["weight_type"].AsString().str();
  if (weight_type == "fp32") {
    attr.weight_type = MoeExpertsAttributes::WeightType::kFp32;
  } else if (weight_type == "int8") {
    attr.weight_type = MoeExpertsAttributes::WeightType::kInt8;
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "moe unsupported weight_type: ", weight_type));
  }

  if (attr.num_experts <= 0 || attr.num_active_experts <= 0 ||
      attr.model_dim <= 0 || attr.hidden_dim <= 0) {
    return absl::InvalidArgumentError(
        "moe dimensions must be positive.");
  }
  if (attr.num_active_experts > attr.num_experts) {
    return absl::InvalidArgumentError(
        "moe num_active_experts must be <= num_experts.");
  }
  return attr;
}
absl::StatusOr<MoeExpertsAttributes> InferAttributesFromTensors(
    const TfLiteNode *tflite_node, const TfLiteTensor *src,
    const TfLiteTensor *top_indices, const TfLiteTensor *gate_weight,
    const TfLiteTensor *linear_weight) {
  MoeExpertsAttributes attr;
  if (tflite_node->inputs->size == kFp32InputCount) {
    attr.weight_type = MoeExpertsAttributes::WeightType::kFp32;
  } else if (tflite_node->inputs->size == kInt8InputCount) {
    attr.weight_type = MoeExpertsAttributes::WeightType::kInt8;
  } else {
    return absl::InvalidArgumentError(
        "moe cannot infer weight_type from input count.");
  }

  const ::ml_drift::BHWC src_shape = ExtractTensorShape(src);
  const ::ml_drift::BHWC top_indices_shape = ExtractTensorShape(top_indices);
  const ::ml_drift::BHWC gate_shape = ExtractTensorShape(gate_weight);
  const ::ml_drift::BHWC linear_shape = ExtractTensorShape(linear_weight);
  attr.model_dim = src_shape.c;
  attr.hidden_dim = gate_shape.b;
  attr.num_experts = gate_shape.h;
  attr.num_active_experts = top_indices_shape.c;

  if (attr.num_experts <= 0 || attr.num_active_experts <= 0 ||
      attr.model_dim <= 0 || attr.hidden_dim <= 0) {
    return absl::InvalidArgumentError(
        "moe inferred dimensions must be positive.");
  }
  if (attr.num_active_experts > attr.num_experts) {
    return absl::InvalidArgumentError(
        "moe inferred num_active_experts must be <= num_experts.");
  }
  if (src_shape.b != 1 || src_shape.h != 1 ||
      top_indices_shape.b != 1 || top_indices_shape.h != 1 ||
      top_indices_shape.w != src_shape.w || gate_shape.w != 1 ||
      gate_shape.c != attr.model_dim || linear_shape.b != attr.model_dim ||
      linear_shape.h != attr.num_experts || linear_shape.w != 1 ||
      linear_shape.c != attr.hidden_dim) {
    return absl::InvalidArgumentError(
        "moe cannot infer consistent attributes from shapes.");
  }
  return attr;
}

absl::StatusOr<MoeExpertsAttributes> ReadAttributesOrInfer(
    const TfLiteContext *context, const TfLiteNode *tflite_node,
    const TfLiteRegistration *registration) {
  absl::StatusOr<MoeExpertsAttributes> attr =
      ReadAttributes(tflite_node, registration);
  if (attr.ok() || !IsMissingCustomOptions(tflite_node, registration)) {
    return attr;
  }
  if (tflite_node->inputs->size != kFp32InputCount &&
      tflite_node->inputs->size != kInt8InputCount) {
    return attr.status();
  }

  const TfLiteTensor *src = nullptr;
  const TfLiteTensor *top_indices = nullptr;
  const TfLiteTensor *gate_weight = nullptr;
  const TfLiteTensor *linear_weight = nullptr;
  RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, kInputSrc, &src));
  RETURN_IF_ERROR(
      PreGetInputTensor(context, tflite_node, kInputTopIndices, &top_indices));
  RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, kInputInt8GateWeight,
                                    &gate_weight));
  const int linear_weight_index =
      tflite_node->inputs->size == kFp32InputCount ? kInputFp32LinearWeight
                                                   : kInputInt8LinearWeight;
  RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, linear_weight_index,
                                    &linear_weight));
  return InferAttributesFromTensors(tflite_node, src, top_indices, gate_weight,
                                    linear_weight);
}

absl::StatusOr<MoeExpertsAttributes> ReadAttributesOrInfer(
    const TfLiteNode *tflite_node, const TfLiteRegistration *registration,
    ObjectReader *reader) {
  absl::StatusOr<MoeExpertsAttributes> attr =
      ReadAttributes(tflite_node, registration);
  if (attr.ok() || !IsMissingCustomOptions(tflite_node, registration)) {
    return attr;
  }
  if (tflite_node->inputs->size != kFp32InputCount &&
      tflite_node->inputs->size != kInt8InputCount) {
    return attr.status();
  }
  const int linear_weight_index =
      tflite_node->inputs->size == kFp32InputCount ? kInputFp32LinearWeight
                                                   : kInputInt8LinearWeight;
  return InferAttributesFromTensors(
      tflite_node, reader->GetInputTensor(kInputSrc),
      reader->GetInputTensor(kInputTopIndices),
      reader->GetInputTensor(kInputInt8GateWeight),
      reader->GetInputTensor(linear_weight_index));
}

void AddQuantizedConstInputPreserveShape(::ml_drift::GraphFloat32* graph,
                                         ObjectReader* reader,
                                         int node_input_index,
                                         ::ml_drift::Node* node) {
  const TfLiteTensor* tensor = reader->GetInputTensor(node_input_index);
  ::ml_drift::Value* input = graph->NewValue();
  input->tensor.type = ToDataType(tensor->type);
  input->tensor.shape = ExtractTensorShape(tensor);
  input->tensor.ref = reader->GetTensorId(node_input_index);
  input->tensor.is_variable_input = tensor->is_variable;
  graph->AddConsumer(node->id, input->id);
}

void AddConstInput(::ml_drift::GraphFloat32* graph, ObjectReader* reader,
                   int node_input_index, ::ml_drift::Node* node) {
  const ::ml_drift::Value* input =
      reader->AddConstInput(node_input_index, /*layout=*/{});
  graph->AddConsumer(node->id, input->id);
}

}  // namespace

absl::Status MoeExpertsOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration) {
  ASSIGN_OR_RETURN(const MoeExpertsAttributes attr,
                   ReadAttributesOrInfer(context, tflite_node, registration));
  const int expected_inputs =
      attr.weight_type == MoeExpertsAttributes::WeightType::kFp32
          ? kFp32InputCount
          : kInt8InputCount;
  if (tflite_node->inputs->size != expected_inputs) {
    return absl::UnavailableError(
        absl::StrCat("moe expects ", expected_inputs, " inputs."));
  }
  if (tflite_node->outputs->size != 1) {
    return absl::InvalidArgumentError("moe expects 1 output.");
  }

  RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, kInputSrc));
  RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, kInputTopWeights));
  RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, kInputTopIndices));
  RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

  const TfLiteTensor* src = nullptr;
  RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, kInputSrc, &src));
  const ::ml_drift::BHWC src_shape = ExtractTensorShape(src);
  if (src_shape.b != 1 || src_shape.h != 1 || src_shape.c != attr.model_dim) {
    return absl::UnavailableError(
        "moe expects src shape [1, seq, model_dim] or "
        "[1, 1, seq, model_dim].");
  }
  const int sequence_size = src_shape.w;

  const TfLiteTensor* top_weights = nullptr;
  RETURN_IF_ERROR(
      PreGetInputTensor(context, tflite_node, kInputTopWeights, &top_weights));
  RETURN_IF_ERROR(ValidateShape(
      ExtractTensorShape(top_weights),
      ::ml_drift::BHWC(1, 1, sequence_size, attr.num_active_experts),
      "top_weights"));

  const TfLiteTensor* top_indices = nullptr;
  RETURN_IF_ERROR(
      PreGetInputTensor(context, tflite_node, kInputTopIndices, &top_indices));
  if (top_indices->type != kTfLiteInt32) {
    return absl::InvalidArgumentError(
        "moe expects int32 top_indices.");
  }
  RETURN_IF_ERROR(ValidateShape(
      ExtractTensorShape(top_indices),
      ::ml_drift::BHWC(1, 1, sequence_size, attr.num_active_experts),
      "top_indices"));

  if (attr.weight_type == MoeExpertsAttributes::WeightType::kFp32) {
    const TfLiteTensor* gate_weight = nullptr;
    const TfLiteTensor* ff1_weight = nullptr;
    const TfLiteTensor* linear_weight = nullptr;
    const TfLiteTensor* per_expert_scale = nullptr;
    RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node,
                                      kInputFp32GateWeight, &gate_weight));
    RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, kInputFp32Ff1Weight,
                                      &ff1_weight));
    RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node,
                                      kInputFp32LinearWeight, &linear_weight));
    RETURN_IF_ERROR(PreGetInputTensor(
        context, tflite_node, kInputFp32PerExpertScale, &per_expert_scale));
    if (!tflite::IsConstantTensor(gate_weight) ||
        !tflite::IsConstantTensor(ff1_weight) ||
        !tflite::IsConstantTensor(linear_weight) ||
        !tflite::IsConstantTensor(per_expert_scale)) {
      return absl::InvalidArgumentError(
          "moe v1 expects constant weights and scales.");
    }
    RETURN_IF_ERROR(ValidateExpertWeight(gate_weight, attr.num_experts,
                                         attr.hidden_dim, attr.model_dim,
                                         "ff_gate_weight"));
    RETURN_IF_ERROR(ValidateExpertWeight(ff1_weight, attr.num_experts,
                                         attr.hidden_dim, attr.model_dim,
                                         "ff1_weight"));
    RETURN_IF_ERROR(ValidateExpertWeight(linear_weight, attr.num_experts,
                                         attr.model_dim, attr.hidden_dim,
                                         "linear_weight"));
    RETURN_IF_ERROR(
        ValidateScale(per_expert_scale, attr.num_experts, "per_expert_scale"));
  } else {
    const int weight_indices[] = {kInputInt8GateWeight, kInputInt8Ff1Weight,
                                  kInputInt8LinearWeight};
    absl::string_view weight_names[] = {"ff_gate_weight", "ff1_weight",
                                        "linear_weight"};
    for (int i = 0; i < 3; ++i) {
      const TfLiteTensor* weight = nullptr;
      RETURN_IF_ERROR(
          PreGetInputTensor(context, tflite_node, weight_indices[i], &weight));
      if (!tflite::IsConstantTensor(weight)) {
        return absl::InvalidArgumentError(
            "moe v1 expects constant int8 weights.");
      }
      if (weight->type != kTfLiteInt8) {
        return absl::InvalidArgumentError(absl::StrCat(
            "moe expects int8 ", weight_names[i], "."));
      }
      RETURN_IF_ERROR(ValidateInt8ZeroPoint(weight, weight_names[i]));
    }
    const TfLiteTensor* gate_weight = nullptr;
    const TfLiteTensor* ff1_weight = nullptr;
    const TfLiteTensor* linear_weight = nullptr;
    RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node,
                                      kInputInt8GateWeight, &gate_weight));
    RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, kInputInt8Ff1Weight,
                                      &ff1_weight));
    RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node,
                                      kInputInt8LinearWeight, &linear_weight));
    RETURN_IF_ERROR(ValidateExpertWeight(gate_weight, attr.num_experts,
                                         attr.hidden_dim, attr.model_dim,
                                         "ff_gate_weight"));
    RETURN_IF_ERROR(ValidateExpertWeight(ff1_weight, attr.num_experts,
                                         attr.hidden_dim, attr.model_dim,
                                         "ff1_weight"));
    RETURN_IF_ERROR(ValidateExpertWeight(linear_weight, attr.num_experts,
                                         attr.model_dim, attr.hidden_dim,
                                         "linear_weight"));

    const TfLiteTensor* gate_scale = nullptr;
    const TfLiteTensor* ff1_scale = nullptr;
    const TfLiteTensor* linear_scale = nullptr;
    const TfLiteTensor* per_expert_scale = nullptr;
    RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, kInputInt8GateScale,
                                      &gate_scale));
    RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, kInputInt8Ff1Scale,
                                      &ff1_scale));
    RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node,
                                      kInputInt8LinearScale, &linear_scale));
    RETURN_IF_ERROR(PreGetInputTensor(
        context, tflite_node, kInputInt8PerExpertScale, &per_expert_scale));
    if (!tflite::IsConstantTensor(gate_scale) ||
        !tflite::IsConstantTensor(ff1_scale) ||
        !tflite::IsConstantTensor(linear_scale) ||
        !tflite::IsConstantTensor(per_expert_scale)) {
      return absl::InvalidArgumentError(
          "moe v1 expects constant int8 scales.");
    }
    RETURN_IF_ERROR(ValidateExpertScale(gate_scale, attr.num_experts,
                                        attr.hidden_dim, "ff_gate_scale"));
    RETURN_IF_ERROR(ValidateExpertScale(ff1_scale, attr.num_experts,
                                        attr.hidden_dim, "ff1_scale"));
    RETURN_IF_ERROR(ValidateExpertScale(linear_scale, attr.num_experts,
                                        attr.model_dim, "linear_scale"));
    RETURN_IF_ERROR(
        ValidateScale(per_expert_scale, attr.num_experts, "per_expert_scale"));
  }

  return absl::OkStatus();
}

void MoeExpertsOperationParser::Parse(const TfLiteNode* tflite_node,
                                      const TfLiteRegistration* registration,
                                      ::ml_drift::GraphFloat32* graph,
                                      ObjectReader* reader) {
  absl::StatusOr<MoeExpertsAttributes> attr_or =
      ReadAttributesOrInfer(tflite_node, registration ,reader);
  ABSL_CHECK_OK(attr_or.status());
  MoeExpertsAttributes attr = std::move(attr_or.value());

  ::ml_drift::Node* node = graph->NewNode();
  node->operation.type = kMoeExpertsType;
  reader->AddInput(node, kInputSrc);
  reader->AddInput(node, kInputTopWeights);
  reader->AddInput(node, kInputTopIndices);

  if (attr.weight_type == MoeExpertsAttributes::WeightType::kFp32) {
    AddConstInput(graph, reader, kInputFp32GateWeight, node);
    AddConstInput(graph, reader, kInputFp32Ff1Weight, node);
    AddConstInput(graph, reader, kInputFp32LinearWeight, node);
    AddConstInput(graph, reader, kInputFp32PerExpertScale, node);
  } else {
    reader->ReadTensor(kInputInt8GateScale, &attr.ff_gate_scale.emplace(),
                       ReadTensorFlags::kNoExtraBytes);
    reader->ReadTensor(kInputInt8Ff1Scale, &attr.ff1_scale.emplace(),
                       ReadTensorFlags::kNoExtraBytes);
    reader->ReadTensor(kInputInt8LinearScale, &attr.linear_scale.emplace(),
                       ReadTensorFlags::kNoExtraBytes);
    AddQuantizedConstInputPreserveShape(graph, reader, kInputInt8GateWeight,
                                        node);
    AddConstInput(graph, reader, kInputInt8GateScale, node);
    AddQuantizedConstInputPreserveShape(graph, reader, kInputInt8Ff1Weight,
                                        node);
    AddConstInput(graph, reader, kInputInt8Ff1Scale, node);
    AddQuantizedConstInputPreserveShape(graph, reader, kInputInt8LinearWeight,
                                        node);
    AddConstInput(graph, reader, kInputInt8LinearScale, node);
    AddConstInput(graph, reader, kInputInt8PerExpertScale, node);
  }
  reader->AddOutputs(node);
  node->operation.attributes = std::move(attr);
}

}  // namespace litert::ml_drift
