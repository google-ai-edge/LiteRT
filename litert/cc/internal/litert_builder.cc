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

#include "litert/cc/internal/litert_builder.h"

#include <optional>
#include <string>
#include <vector>

#include "litert/c/litert_builder.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"

namespace litert {

Expected<Tensor> Builder::BuildTensor(const RankedTensorSpec& spec) const {
  // tensor holds the newly created tensor.
  LiteRtTensor tensor;
  LiteRtRankedTensorType ranked_tensor_type_litert =
      static_cast<LiteRtRankedTensorType>(spec.ranked_tensor_type);

  LiteRtWeights litert_weights;
  if (spec.weights.has_value()) {
    litert_weights = spec.weights->Get();
  } else {
    litert_weights = nullptr;
  }

  LiteRtQuantizationTypeId quantization_type_id = kLiteRtQuantizationNone;
  LiteRtQuantizationPerTensor litert_per_tensor_quantization;
  if (spec.per_tensor_quantization.has_value()) {
    litert_per_tensor_quantization = *spec.per_tensor_quantization;
    quantization_type_id = kLiteRtQuantizationPerTensor;
  }
  LiteRtQuantizationPerChannel litert_per_channel_quantization;
  if (spec.per_channel_quantization.has_value()) {
    litert_per_channel_quantization = *spec.per_channel_quantization;
    quantization_type_id = kLiteRtQuantizationPerChannel;
  }
  internal::AssertOk(LiteRtBuilderBuildTensor, Get(),
                     kLiteRtRankedTensorType, ranked_tensor_type_litert,
                     LiteRtUnrankedTensorType(), litert_weights,
                     quantization_type_id, litert_per_tensor_quantization,
                     litert_per_channel_quantization,
                     spec.tensor_name.value_or("").c_str(), &tensor);
  return Tensor(tensor);
}

Expected<Tensor> Builder::BuildScalar(LiteRtElementType element_type,
                                      std::optional<std::string> name) const {
  LiteRtTensor tensor;
  LiteRtUnrankedTensorType unranked_tensor_type;
  unranked_tensor_type.element_type = element_type;
  internal::AssertOk(
      LiteRtBuilderBuildTensor, Get(), kLiteRtUnrankedTensorType,
      LiteRtRankedTensorType(), unranked_tensor_type, LiteRtWeights(),
      kLiteRtQuantizationNone, LiteRtQuantizationPerTensor(),
      LiteRtQuantizationPerChannel(), name.value_or("").c_str(), &tensor);
  return Tensor(tensor);
}

Op Builder::BuildOp(LiteRtOpCode op_code, OpInputs& inputs,
                    OpOutputs& outputs) const {
  LiteRtOp litert_op;
  std::vector<LiteRtTensor> input_tensors;
  input_tensors.reserve(inputs.size());
  for (const auto& input : inputs) {
    input_tensors.push_back(input.Get());
  }
  std::vector<LiteRtTensor> output_tensors;
  output_tensors.reserve(outputs.size());
  for (const auto& output : outputs) {
    output_tensors.push_back(output.Get());
  }
  internal::AssertOk(LiteRtBuilderBuildOp, Get(), op_code,
                     input_tensors.size(), input_tensors.data(),
                     output_tensors.size(), output_tensors.data(), &litert_op);
  return Op(litert_op);
}

}  // namespace litert
