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

#include "ml_drift_delegate/delegate/composite/symmetric_quantized_fully_connected.h"

#include <any>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

using ::ml_drift::GpuModelBuilder;
using ::ml_drift::OHWI;
using ::ml_drift::OperationType;
using ::ml_drift::ValueId;

OHWI GetWeightsShape(const ::ml_drift::FullyConnectedInt8Attributes& attr) {
  return attr.weights.shape;
}

// Int4 and int2 weights are a variant of packed and unpacked tensors.
template <typename AttrT>
OHWI GetWeightsShape(const AttrT& attr) {
  return std::visit([](const auto& weights) { return weights.shape; },
                    attr.weights);
}

// `inputs` are (src, weights, scale) or (src, weights, bias, scale).
template <typename AttrT>
absl::Status CreateFullyConnected(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::CreateGpuModelInfo& create_info, OperationType op_type,
    const AttrT& attr, const std::vector<ValueId>& inputs, ValueId output,
    GpuModelBuilder* model_builder) {
  ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]));
  ABSL_ASSIGN_OR_RETURN(auto weights, model_builder->GetTensor(inputs[1]));
  GpuModelBuilder::TensorHandle scale;
  GpuModelBuilder::TensorHandle bias;
  GpuModelBuilder::TensorHandle* bias_ptr = nullptr;
  if (inputs.size() == 3) {
    ABSL_ASSIGN_OR_RETURN(scale, model_builder->GetTensor(inputs[2]));
    if (!attr.bias.data.empty()) {
      bias = model_builder->AddConstantTensor(attr.bias,
                                              src.tensor_desc.GetDataType());
      bias_ptr = &bias;
    }
  } else {
    ABSL_ASSIGN_OR_RETURN(bias, model_builder->GetTensor(inputs[2]));
    ABSL_ASSIGN_OR_RETURN(scale, model_builder->GetTensor(inputs[3]));
    bias_ptr = &bias;
  }

  const OHWI weights_shape = GetWeightsShape(attr);
  const bool prefer_textures =
      create_info.hints.Check(::ml_drift::ModelHints::kPreferTextureWeights);
  const auto external_weights =
      [&](const ::ml_drift::WeightsDescription& weights_desc) {
        return ::ml_drift::CreateExternalWeights(
            weights, weights_desc, weights_shape, attr.scale.shape, &scale,
            /*zero_point=*/nullptr, /*sum_i=*/nullptr);
      };
  GpuModelBuilder::TensorHandle out;
  switch (op_type) {
    case OperationType::kFullyConnectedInt8:
      out = model_builder->FullyConnectedInt8ExternalWeights(
          src,
          external_weights(::ml_drift::GetFullyConnectedInt8WeightsDesc(
              gpu_info, weights_shape, prefer_textures)),
          bias_ptr);
      break;
    case OperationType::kFullyConnectedInt4:
      out = model_builder->FullyConnectedInt4ExternalWeights(
          src,
          external_weights(::ml_drift::GetFullyConnectedInt4WeightsDesc(
              gpu_info, weights_shape, prefer_textures)),
          bias_ptr);
      break;
    case OperationType::kFullyConnectedInt2:
      out = model_builder->FullyConnectedInt2ExternalWeights(
          src,
          external_weights(::ml_drift::GetFullyConnectedInt2WeightsDesc(
              gpu_info, weights_shape, prefer_textures)),
          bias_ptr);
      break;
    default:
      return absl::InternalError(
          "Expected an int2, int4 or int8 FullyConnected.");
  }
  return model_builder->UpdateOutputTensor(out, output);
}

template <typename AttrT>
absl::Status CreateIfSymmetric(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::CreateGpuModelInfo& create_info, OperationType op_type,
    const std::any& attributes, const std::vector<ValueId>& inputs,
    ValueId output, GpuModelBuilder* model_builder) {
  const auto* attr = std::any_cast<AttrT>(&attributes);
  if (attr == nullptr) {
    return absl::InvalidArgumentError(
        "Missing or mismatched FullyConnected attributes.");
  }
  if (inputs.size() != 3 && inputs.size() != 4) {
    // Inline constant weights (1-2 inputs) or asymmetric external weights with
    // bias / weights_sum_i (5+ inputs) are handled by ML Drift's selector.
    return absl::UnimplementedError(
        "Not an external-weight FullyConnected with 3 or 4 inputs.");
  }
  if (attr->zero_point.shape.o == 0) {
    return CreateFullyConnected(gpu_info, create_info, op_type, *attr, inputs,
                                output, model_builder);
  } else if (inputs.size() == 3) {
    return absl::InvalidArgumentError(
        "Quantized FullyConnected with a zero point requires at least 4 "
        "inputs.");
  } else {
    // 4-input (src, weights, scale, zero_point): fall back to ML Drift's
    // selector, which binds the zero-point tensor.
    return absl::UnimplementedError(
        "Quantized FullyConnected has a zero point.");
  }
}

absl::Status CreateFromAttributes(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    absl::string_view op_type, const std::any& attributes,
    const std::vector<ValueId>& inputs, const std::vector<ValueId>& outputs,
    GpuModelBuilder* model_builder) {
  if (outputs.size() != 1) {
    return absl::UnimplementedError("Expected a single FullyConnected output.");
  }
  if (op_type == ToString(OperationType::kFullyConnectedInt8)) {
    return CreateIfSymmetric<::ml_drift::FullyConnectedInt8Attributes>(
        gpu_info, create_info, OperationType::kFullyConnectedInt8, attributes,
        inputs, outputs[0], model_builder);
  } else if (op_type == ToString(OperationType::kFullyConnectedInt4)) {
    return CreateIfSymmetric<::ml_drift::FullyConnectedInt4Attributes>(
        gpu_info, create_info, OperationType::kFullyConnectedInt4, attributes,
        inputs, outputs[0], model_builder);
  } else if (op_type == ToString(OperationType::kFullyConnectedInt2)) {
    return CreateIfSymmetric<::ml_drift::FullyConnectedInt2Attributes>(
        gpu_info, create_info, OperationType::kFullyConnectedInt2, attributes,
        inputs, outputs[0], model_builder);
  }
  return absl::UnimplementedError(
      "Not an int2, int4 or int8 FullyConnected op.");
}

template <typename TensorT>
std::vector<ValueId> GetIds(const std::vector<TensorT*>& tensors) {
  std::vector<ValueId> ids;
  ids.reserve(tensors.size());
  for (const auto* tensor : tensors) {
    ids.push_back(tensor->id);
  }
  return ids;
}

}  // namespace

absl::Status CreateSymmetricQuantizedFullyConnectedFromNode(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  return CreateFromAttributes(gpu_info, create_info, node.operation.type,
                              node.operation.attributes, GetIds(inputs),
                              GetIds(outputs), model_builder);
}

absl::Status CreateSymmetricQuantizedFullyConnectedFromIrOp(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& op,
    ::ml_drift::GpuModelBuilder* model_builder) {
  return CreateFromAttributes(gpu_info, create_info, op.name, op.attr,
                              GetIds(inputs), GetIds(outputs), model_builder);
}

}  // namespace litert::ml_drift
