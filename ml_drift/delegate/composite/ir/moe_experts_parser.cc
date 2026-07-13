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

#include "third_party/odml/litert/ml_drift/delegate/composite/ir/moe_experts_parser.h"

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/custom_ir_operation_parser.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/model_builder_helper.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

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
        "moe expected ", name, " shape [", expected.b, ", ", expected.h, ", ",
        expected.w, ", ", expected.c, "], got [", shape.b, ", ", shape.h, ", ",
        shape.w, ", ", shape.c, "]."));
  }
  return absl::OkStatus();
}

absl::Status ValidateExpertWeight(const TfLiteTensor* tensor, int experts,
                                  int out_channels, int in_channels,
                                  absl::string_view name) {
  const ::ml_drift::BHWC shape = ::litert::ml_drift::ExtractTensorShape(tensor);
  return ValidateShape(
      shape, ::ml_drift::BHWC(out_channels, experts, 1, in_channels), name);
}

absl::Status ValidateScale(const TfLiteTensor* tensor, int channels,
                           absl::string_view name) {
  if (tensor->type != kTfLiteFloat32 && tensor->type != kTfLiteFloat16) {
    return absl::InvalidArgumentError(
        absl::StrCat("moe expects ", name, " to be float32 or float16."));
  }
  const ::ml_drift::BHWC shape = ::litert::ml_drift::ExtractTensorShape(tensor);
  return ValidateShape(shape, ::ml_drift::BHWC(1, 1, 1, channels), name);
}

absl::Status ValidateExpertScale(const TfLiteTensor* tensor, int experts,
                                 int out_channels, absl::string_view name) {
  if (tensor->type != kTfLiteFloat32 && tensor->type != kTfLiteFloat16) {
    return absl::InvalidArgumentError(
        absl::StrCat("moe expects ", name, " to be float32 or float16."));
  }
  const ::ml_drift::BHWC shape = ::litert::ml_drift::ExtractTensorShape(tensor);
  return ValidateShape(shape, ::ml_drift::BHWC(out_channels, experts, 1, 1),
                       name);
}

absl::Status ValidateInt8ZeroPoint(const TfLiteTensor* tensor,
                                   absl::string_view name) {
  if (tensor->quantization.type != kTfLiteAffineQuantization ||
      tensor->quantization.params == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("moe expects affine quantization for ", name, "."));
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

bool IsMissingCustomOptions(const TfLiteNode* tflite_node,
                            const TfLiteRegistration* registration) {
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
  auto map_or = ReadAttributeMap(tflite_node, registration);
  if (!map_or.ok()) return map_or.status();
  const flexbuffers::Map map = map_or.value();

  for (absl::string_view key : {"num_experts", "num_active_experts",
                                "model_dim", "hidden_dim", "weight_type"}) {
    if (IsMissing(map, key)) {
      return absl::InvalidArgumentError(
          absl::StrCat("moe is missing ", key, "."));
    }
  }
  if (!IsMissing(map, "activation") &&
      map["activation"].AsString().str() != "gelu") {
    return absl::InvalidArgumentError("moe only supports activation='gelu'.");
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
    return absl::InvalidArgumentError(
        absl::StrCat("moe unsupported weight_type: ", weight_type));
  }

  if (attr.num_experts <= 0 || attr.num_active_experts <= 0 ||
      attr.model_dim <= 0 || attr.hidden_dim <= 0) {
    return absl::InvalidArgumentError("moe dimensions must be positive.");
  }
  if (attr.num_active_experts > attr.num_experts) {
    return absl::InvalidArgumentError(
        "moe num_active_experts must be <= num_experts.");
  }
  return attr;
}

absl::StatusOr<MoeExpertsAttributes> InferAttributesFromTensors(
    const TfLiteNode* tflite_node, const TfLiteTensor* src,
    const TfLiteTensor* top_indices, const TfLiteTensor* gate_weight,
    const TfLiteTensor* linear_weight) {
  MoeExpertsAttributes attr;
  if (tflite_node->inputs->size == kFp32InputCount) {
    attr.weight_type = MoeExpertsAttributes::WeightType::kFp32;
  } else if (tflite_node->inputs->size == kInt8InputCount) {
    attr.weight_type = MoeExpertsAttributes::WeightType::kInt8;
  } else {
    return absl::InvalidArgumentError(
        "moe cannot infer weight_type from input count.");
  }

  const ::ml_drift::BHWC src_shape =
      ::litert::ml_drift::ExtractTensorShape(src);
  const ::ml_drift::BHWC top_indices_shape =
      ::litert::ml_drift::ExtractTensorShape(top_indices);
  const ::ml_drift::BHWC gate_shape =
      ::litert::ml_drift::ExtractTensorShape(gate_weight);
  const ::ml_drift::BHWC linear_shape =
      ::litert::ml_drift::ExtractTensorShape(linear_weight);
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
  if (src_shape.b != 1 || src_shape.h != 1 || top_indices_shape.b != 1 ||
      top_indices_shape.h != 1 || top_indices_shape.w != src_shape.w ||
      gate_shape.w != 1 || gate_shape.c != attr.model_dim ||
      linear_shape.b != attr.model_dim || linear_shape.h != attr.num_experts ||
      linear_shape.w != 1 || linear_shape.c != attr.hidden_dim) {
    return absl::InvalidArgumentError(
        "moe cannot infer consistent attributes from shapes.");
  }
  return attr;
}

absl::Status GetInputTensor(const TfLiteContext* context,
                            const TfLiteNode* tflite_node, int index,
                            const TfLiteTensor** tensor) {
  if (index < 0 || index >= tflite_node->inputs->size) {
    return absl::InvalidArgumentError("moe invalid input index.");
  }
  int tensor_index = tflite_node->inputs->data[index];
  if (tensor_index == kTfLiteOptionalTensor) {
    return absl::InvalidArgumentError("moe input is optional tensor.");
  }
  *tensor = &context->tensors[tensor_index];
  return absl::OkStatus();
}

absl::StatusOr<MoeExpertsAttributes> ReadAttributesOrInfer(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration) {
  absl::StatusOr<MoeExpertsAttributes> attr =
      ReadAttributes(tflite_node, registration);
  if (attr.ok() || !IsMissingCustomOptions(tflite_node, registration)) {
    return attr;
  }
  if (tflite_node->inputs->size != kFp32InputCount &&
      tflite_node->inputs->size != kInt8InputCount) {
    return attr.status();
  }

  const TfLiteTensor* src = nullptr;
  const TfLiteTensor* top_indices = nullptr;
  const TfLiteTensor* gate_weight = nullptr;
  const TfLiteTensor* linear_weight = nullptr;
  auto status = GetInputTensor(context, tflite_node, kInputSrc, &src);
  if (!status.ok()) return status;
  status = GetInputTensor(context, tflite_node, kInputTopIndices, &top_indices);
  if (!status.ok()) return status;
  status =
      GetInputTensor(context, tflite_node, kInputInt8GateWeight, &gate_weight);
  if (!status.ok()) return status;
  const int linear_weight_index = tflite_node->inputs->size == kFp32InputCount
                                      ? kInputFp32LinearWeight
                                      : kInputInt8LinearWeight;
  status =
      GetInputTensor(context, tflite_node, linear_weight_index, &linear_weight);
  if (!status.ok()) return status;

  return InferAttributesFromTensors(tflite_node, src, top_indices, gate_weight,
                                    linear_weight);
}

absl::Status MoeExpertsIsSupported(const TfLiteContext* context,
                                   const TfLiteNode* tflite_node,
                                   const TfLiteRegistration* registration) {
  auto attr_or = ReadAttributesOrInfer(context, tflite_node, registration);
  if (!attr_or.ok()) return attr_or.status();
  const MoeExpertsAttributes attr = attr_or.value();

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

  const TfLiteTensor* src = nullptr;
  auto status = GetInputTensor(context, tflite_node, kInputSrc, &src);
  if (!status.ok()) return status;
  const ::ml_drift::BHWC src_shape =
      ::litert::ml_drift::ExtractTensorShape(src);
  if (src_shape.b != 1 || src_shape.h != 1 || src_shape.c != attr.model_dim) {
    return absl::UnavailableError(
        "moe expects src shape [1, seq, model_dim] or "
        "[1, 1, seq, model_dim].");
  }
  const int sequence_size = src_shape.w;

  const TfLiteTensor* top_weights = nullptr;
  status = GetInputTensor(context, tflite_node, kInputTopWeights, &top_weights);
  if (!status.ok()) return status;
  status = ValidateShape(
      ::litert::ml_drift::ExtractTensorShape(top_weights),
      ::ml_drift::BHWC(1, 1, sequence_size, attr.num_active_experts),
      "top_weights");
  if (!status.ok()) return status;

  const TfLiteTensor* top_indices = nullptr;
  status = GetInputTensor(context, tflite_node, kInputTopIndices, &top_indices);
  if (!status.ok()) return status;
  if (top_indices->type != kTfLiteInt32) {
    return absl::InvalidArgumentError("moe expects int32 top_indices.");
  }
  status = ValidateShape(
      ::litert::ml_drift::ExtractTensorShape(top_indices),
      ::ml_drift::BHWC(1, 1, sequence_size, attr.num_active_experts),
      "top_indices");
  if (!status.ok()) return status;

  if (attr.weight_type == MoeExpertsAttributes::WeightType::kFp32) {
    const TfLiteTensor* gate_weight = nullptr;
    const TfLiteTensor* ff1_weight = nullptr;
    const TfLiteTensor* linear_weight = nullptr;
    const TfLiteTensor* per_expert_scale = nullptr;
    status = GetInputTensor(context, tflite_node, kInputFp32GateWeight,
                            &gate_weight);
    if (!status.ok()) return status;
    status =
        GetInputTensor(context, tflite_node, kInputFp32Ff1Weight, &ff1_weight);
    if (!status.ok()) return status;
    status = GetInputTensor(context, tflite_node, kInputFp32LinearWeight,
                            &linear_weight);
    if (!status.ok()) return status;
    status = GetInputTensor(context, tflite_node, kInputFp32PerExpertScale,
                            &per_expert_scale);
    if (!status.ok()) return status;

    if (!tflite::IsConstantTensor(gate_weight) ||
        !tflite::IsConstantTensor(ff1_weight) ||
        !tflite::IsConstantTensor(linear_weight) ||
        !tflite::IsConstantTensor(per_expert_scale)) {
      return absl::InvalidArgumentError(
          "moe v1 expects constant weights and scales.");
    }
    status =
        ValidateExpertWeight(gate_weight, attr.num_experts, attr.hidden_dim,
                             attr.model_dim, "ff_gate_weight");
    if (!status.ok()) return status;
    status = ValidateExpertWeight(ff1_weight, attr.num_experts, attr.hidden_dim,
                                  attr.model_dim, "ff1_weight");
    if (!status.ok()) return status;
    status =
        ValidateExpertWeight(linear_weight, attr.num_experts, attr.model_dim,
                             attr.hidden_dim, "linear_weight");
    if (!status.ok()) return status;
    status =
        ValidateScale(per_expert_scale, attr.num_experts, "per_expert_scale");
    if (!status.ok()) return status;
  } else {
    const int weight_indices[] = {kInputInt8GateWeight, kInputInt8Ff1Weight,
                                  kInputInt8LinearWeight};
    absl::string_view weight_names[] = {"ff_gate_weight", "ff1_weight",
                                        "linear_weight"};
    for (int i = 0; i < 3; ++i) {
      const TfLiteTensor* weight = nullptr;
      status = GetInputTensor(context, tflite_node, weight_indices[i], &weight);
      if (!status.ok()) return status;
      if (!tflite::IsConstantTensor(weight)) {
        return absl::InvalidArgumentError(
            "moe v1 expects constant int8 weights.");
      }
      if (weight->type != kTfLiteInt8) {
        return absl::InvalidArgumentError(
            absl::StrCat("moe expects int8 ", weight_names[i], "."));
      }
      status = ValidateInt8ZeroPoint(weight, weight_names[i]);
      if (!status.ok()) return status;
    }
    const TfLiteTensor* gate_weight = nullptr;
    const TfLiteTensor* ff1_weight = nullptr;
    const TfLiteTensor* linear_weight = nullptr;
    status = GetInputTensor(context, tflite_node, kInputInt8GateWeight,
                            &gate_weight);
    if (!status.ok()) return status;
    status =
        GetInputTensor(context, tflite_node, kInputInt8Ff1Weight, &ff1_weight);
    if (!status.ok()) return status;
    status = GetInputTensor(context, tflite_node, kInputInt8LinearWeight,
                            &linear_weight);
    if (!status.ok()) return status;

    status =
        ValidateExpertWeight(gate_weight, attr.num_experts, attr.hidden_dim,
                             attr.model_dim, "ff_gate_weight");
    if (!status.ok()) return status;
    status = ValidateExpertWeight(ff1_weight, attr.num_experts, attr.hidden_dim,
                                  attr.model_dim, "ff1_weight");
    if (!status.ok()) return status;
    status =
        ValidateExpertWeight(linear_weight, attr.num_experts, attr.model_dim,
                             attr.hidden_dim, "linear_weight");
    if (!status.ok()) return status;

    const TfLiteTensor* gate_scale = nullptr;
    const TfLiteTensor* ff1_scale = nullptr;
    const TfLiteTensor* linear_scale = nullptr;
    const TfLiteTensor* per_expert_scale = nullptr;
    status =
        GetInputTensor(context, tflite_node, kInputInt8GateScale, &gate_scale);
    if (!status.ok()) return status;
    status =
        GetInputTensor(context, tflite_node, kInputInt8Ff1Scale, &ff1_scale);
    if (!status.ok()) return status;
    status = GetInputTensor(context, tflite_node, kInputInt8LinearScale,
                            &linear_scale);
    if (!status.ok()) return status;
    status = GetInputTensor(context, tflite_node, kInputInt8PerExpertScale,
                            &per_expert_scale);
    if (!status.ok()) return status;

    if (!tflite::IsConstantTensor(gate_scale) ||
        !tflite::IsConstantTensor(ff1_scale) ||
        !tflite::IsConstantTensor(linear_scale) ||
        !tflite::IsConstantTensor(per_expert_scale)) {
      return absl::InvalidArgumentError("moe v1 expects constant int8 scales.");
    }
    status = ValidateExpertScale(gate_scale, attr.num_experts, attr.hidden_dim,
                                 "ff_gate_scale");
    if (!status.ok()) return status;
    status = ValidateExpertScale(ff1_scale, attr.num_experts, attr.hidden_dim,
                                 "ff1_scale");
    if (!status.ok()) return status;
    status = ValidateExpertScale(linear_scale, attr.num_experts, attr.model_dim,
                                 "linear_scale");
    if (!status.ok()) return status;
    status =
        ValidateScale(per_expert_scale, attr.num_experts, "per_expert_scale");
    if (!status.ok()) return status;
  }

  return absl::OkStatus();
}

void MoeExpertsConvert(
    const TfLiteContext& context, const TfLiteNode& tflite_node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& /*options*/,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = "moe_experts";

  auto attr_or = ReadAttributesOrInfer(&context, &tflite_node, &registration);
  if (attr_or.ok()) {
    MoeExpertsAttributes attr = attr_or.value();
    if (attr.weight_type == MoeExpertsAttributes::WeightType::kInt8) {
      auto get_scale = [&](int index) -> MoeScaleTensor {
        const TfLiteTensor* t =
            &context.tensors[tflite_node.inputs->data[index]];
        auto shape = ::litert::ml_drift::ExtractTensorShape(t);
        MoeScaleTensor res;
        res.shape = ::ml_drift::OHWI(shape.b, shape.h, shape.w, shape.c);
        res.data.resize(res.shape.DimensionsProduct());
        ::litert::ml_drift::ir::CopyFloat32Data(t, res.data.data());
        return res;
      };
      attr.ff_gate_scale = get_scale(kInputInt8GateScale);
      attr.ff1_scale = get_scale(kInputInt8Ff1Scale);
      attr.linear_scale = get_scale(kInputInt8LinearScale);
    }
    op->attr = attr;
  }

  for (int i = 0; i < tflite_node.inputs->size; ++i) {
    if (attr_or.ok() && attr_or.value().weight_type ==
                            MoeExpertsAttributes::WeightType::kInt8) {
      if (i == kInputInt8GateScale || i == kInputInt8Ff1Scale ||
          i == kInputInt8LinearScale) {
        continue;
      }
    }
    int tflite_tensor_id = tflite_node.inputs->data[i];
    if (tflite_tensor_id != kTfLiteOptionalTensor) {
      ir_model.AddConsumer(tensor_map[tflite_tensor_id], op->id);
    }
  }

  for (int i = 0; i < tflite_node.outputs->size; ++i) {
    int tflite_tensor_id = tflite_node.outputs->data[i];
    if (tflite_tensor_id != kTfLiteOptionalTensor) {
      ir_model.SetProducer(tensor_map[tflite_tensor_id], op->id);
    }
  }
}

}  // namespace

CustomIrOpParser GetMoeExpertsParser() {
  return {
      .is_supported = MoeExpertsIsSupported,
      .convert = MoeExpertsConvert,
  };
}

}  // namespace litert::ml_drift::ir
