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

// Reusable model-level input-shape override logic, extracted from
// apply_input_shapes_main.cc so that apply_plugin can call it inline without
// an intermediate .tflite file.

#ifndef ODML_LITERT_LITERT_TOOLS_APPLY_INPUT_SHAPES_H_
#define ODML_LITERT_LITERT_TOOLS_APPLY_INPUT_SHAPES_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::tools {

namespace internal {

inline Expected<Dims> ParseShape(absl::string_view shape_str) {
  Dims shape;
  for (absl::string_view dim_str : absl::StrSplit(shape_str, ':')) {
    int32_t dim;
    if (!absl::SimpleAtoi(dim_str, &dim)) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          "Invalid dimension in shape string: " + std::string(dim_str));
    }
    shape.push_back(dim);
  }
  return shape;
}

struct NameAndShape {
  std::string name;
  std::string shape_str;
};

inline Expected<NameAndShape> ParseNameAndShape(absl::string_view input) {
  std::vector<absl::string_view> parts =
      absl::StrSplit(input, absl::MaxSplits('@', 1));
  if (parts.size() != 2) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid input format (expected name@shape): " +
                          std::string(input));
  }
  return NameAndShape{std::string(parts[0]), std::string(parts[1])};
}

inline Expected<void> UpdateTensorType(LiteRtTensor tensor, const Dims& shape) {
  LiteRtElementType element_type = kLiteRtElementTypeNone;
  if (tensor->Type().first == kLiteRtRankedTensorType) {
    element_type = tensor->Type().second.ranked_tensor_type.element_type;
  } else if (tensor->Type().first == kLiteRtUnrankedTensorType) {
    element_type = tensor->Type().second.unranked_tensor_type.element_type;
  }

  if (element_type == kLiteRtElementTypeNone) {
    return Unexpected(kLiteRtStatusErrorUnsupported, "Unknown input type");
  }

  tensor->SetType(MakeRankedTensorType(element_type, absl::MakeSpan(shape)));
  return {};
}

}  // namespace internal

// Apply input shape overrides to a model in memory, then run shape inference.
//
// model                - The in-memory model to mutate.
// signature_key        - Signature to resolve tensors through (empty = first).
// positional_inputs    - Shapes by position, e.g. {"1:224:224:3", "1:10"}.
//                        Count must match the number of model inputs exactly.
// name_inputs          - Shapes by tensor name, e.g. {"arg0@1:224:224:3"}.
//
// At most one of positional_inputs or name_inputs may be non-empty.
inline Expected<void> ApplyInputShapes(
    LiteRtModelT* model, const std::string& signature_key,
    const std::vector<std::string>& positional_inputs,
    const std::vector<std::string>& name_inputs) {
  LiteRtSubgraph subgraph = nullptr;
  LiteRtSignature signature = nullptr;
  std::optional<LiteRtSignatureT> default_signature;

  if (!signature_key.empty()) {
    auto sig_res = model->FindSignature(signature_key);
    if (!sig_res) {
      return Unexpected(sig_res.Error().Status(),
                        "Signature not found: " + signature_key);
    }
    signature = &sig_res->get();
    subgraph = &signature->GetSubgraph();
  } else if (!model->Signatures().empty()) {
    signature = model->Signatures()[0];
    subgraph = &signature->GetSubgraph();
  } else {
    if (model->NumSubgraphs() == 0) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Model has no subgraphs");
    }
    subgraph = model->MainSubgraph();
  }

  if (!positional_inputs.empty()) {
    if (positional_inputs.size() != subgraph->Inputs().size()) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrFormat("Number of inputs provided (%zu) does not "
                          "match model inputs (%zu)",
                          positional_inputs.size(), subgraph->Inputs().size()));
    }
    for (size_t i = 0; i < positional_inputs.size(); ++i) {
      auto shape_res = internal::ParseShape(positional_inputs[i]);
      if (!shape_res) return shape_res.Error();
      auto status = internal::UpdateTensorType(subgraph->Inputs()[i], *shape_res);
      if (!status) return status;
    }
  } else if (!name_inputs.empty()) {
    for (const auto& input_str : name_inputs) {
      auto name_and_shape = internal::ParseNameAndShape(input_str);
      if (!name_and_shape) return name_and_shape.Error();
      auto shape_res = internal::ParseShape(name_and_shape->shape_str);
      if (!shape_res) return shape_res.Error();

      LiteRtTensor found_tensor = nullptr;
      for (auto tensor : subgraph->Tensors()) {
        if (tensor->Name() == name_and_shape->name) {
          found_tensor = tensor;
          break;
        }
      }
      if (!found_tensor) {
        return Unexpected(kLiteRtStatusErrorNotFound,
                          "Tensor not found: " + name_and_shape->name);
      }
      auto status = internal::UpdateTensorType(found_tensor, *shape_res);
      if (!status) return status;
    }
  }

  litert::internal::ShapeInferenceEngine engine(model);
  auto status = engine.InferSubgraphShapes(subgraph);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Shape inference failed");
  }

  return {};
}

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_TOOLS_APPLY_INPUT_SHAPES_H_
