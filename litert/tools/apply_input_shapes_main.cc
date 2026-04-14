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

// Example usage:
// bazel run //third_party/odml/litert/litert/tools:apply_input_shapes_main -- \
//   --model_path=/path/to/model.tflite \
//   --output_path=/path/to/output_model.tflite \
//   --input=1:224:224:3 \
//   --input=1:10
//
// Or by tensor name:
// bazel run //third_party/odml/litert/litert/tools:apply_input_shapes_main -- \
//   --model_path=/path/to/model.tflite \
//   --output_path=/path/to/output_model.tflite \
//   --input_name=arg0@1:224:224:3
//
// Or by signature input name:
// bazel run //third_party/odml/litert/litert/tools:apply_input_shapes_main -- \
//   --model_path=/path/to/model.tflite \
//   --output_path=/path/to/output_model.tflite \
//   --signature=serving_default \
//   --signature_name=image@1:224:224:3

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/model/shape_inference.h"
#include "litert/core/model/shape_inference_types.h"

ABSL_FLAG(std::string, model_path, "", "Path to the model file.");
ABSL_FLAG(std::string, output_path, "", "Path to the output model file.");
ABSL_FLAG(std::vector<std::string>, input, {},
          "Input shapes, e.g. --input=1:224:224:3 --input=1:10.");
ABSL_FLAG(
    std::string, signature, "",
    "Signature key to use. Defaults to the first signature if not provided.");
ABSL_FLAG(std::vector<std::string>, input_name, {},
          "Input shapes by tensor name, e.g. --input_name=arg0@1:224:224:3.");
ABSL_FLAG(std::vector<std::string>, signature_name, {},
          "Input shapes by signature name, e.g. "
          "--signature_name=image@1:224:224:3.");

namespace litert::internal {
namespace {

bool WriteBufferToFile(const std::string& path, const uint8_t* data,
                       size_t size) {
  std::ofstream file(path, std::ios::out | std::ios::binary);
  if (!file) {
    return false;
  }
  file.write(reinterpret_cast<const char*>(data), size);
  return file.good();
}

Expected<Dims> ParseShape(absl::string_view shape_str) {
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

Expected<NameAndShape> ParseNameAndShape(absl::string_view input) {
  std::vector<absl::string_view> parts =
      absl::StrSplit(input, absl::MaxSplits('@', 1));
  if (parts.size() != 2) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Invalid input format (expected name@shape or name:shape): " +
            std::string(input));
  }
  return NameAndShape{std::string(parts[0]), std::string(parts[1])};
}

Expected<void> UpdateTensorType(LiteRtTensor tensor, const Dims& shape) {
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

Expected<void> ApplyInputShapes(
    const std::string& model_path, const std::string& output_path,
    const std::string& signature_key,
    const std::vector<std::string>& positional_inputs,
    const std::vector<std::string>& name_inputs,
    const std::vector<std::string>& signature_name_inputs) {
  // Use internal LoadModelFromFile directly
  auto model_res = litert::internal::LoadModelFromFile(model_path);
  if (!model_res) {
    return Unexpected(model_res.Error().Status(),
                      "Failed to load model: " + model_res.Error().Message());
  }
  auto model = std::move(*model_res);

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
      auto shape_res = ParseShape(positional_inputs[i]);
      if (!shape_res) return shape_res.Error();
      auto status = UpdateTensorType(subgraph->Inputs()[i], *shape_res);
      if (!status) return status;
    }
  } else if (!name_inputs.empty()) {
    for (const auto& input_str : name_inputs) {
      auto name_and_shape = ParseNameAndShape(input_str);
      if (!name_and_shape) return name_and_shape.Error();
      auto shape_res = ParseShape(name_and_shape->shape_str);
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
      auto status = UpdateTensorType(found_tensor, *shape_res);
      if (!status) return status;
    }
  } else if (!signature_name_inputs.empty()) {
    if (signature == nullptr) {
      default_signature = MakeDefaultSignature(subgraph);
      signature = &*default_signature;
    }
    for (const auto& input_str : signature_name_inputs) {
      auto name_and_shape = ParseNameAndShape(input_str);
      if (!name_and_shape) return name_and_shape.Error();
      auto shape_res = ParseShape(name_and_shape->shape_str);
      if (!shape_res) return shape_res.Error();

      auto tensor_res = signature->FindInputTensor(name_and_shape->name);
      if (!tensor_res) {
        return Unexpected(tensor_res.Error().Status(),
                          "Signature input not found: " + name_and_shape->name);
      }
      auto status = UpdateTensorType(*tensor_res, *shape_res);
      if (!status) return status;
    }
  }

  litert::internal::ShapeInferenceEngine engine(model.get());
  auto status = engine.InferSubgraphShapes(subgraph);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Shape inference failed");
  }

  auto serialized = litert::internal::SerializeModel(std::move(*model));
  if (!serialized) {
    return serialized.Error();
  }

  if (!WriteBufferToFile(output_path, serialized->Data(), serialized->Size())) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to write output file");
  }

  return {};
}

}  // namespace
}  // namespace litert::internal

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const auto model_path = absl::GetFlag(FLAGS_model_path);
  const auto output_path = absl::GetFlag(FLAGS_output_path);
  const auto signature_key = absl::GetFlag(FLAGS_signature);
  const auto positional_inputs = absl::GetFlag(FLAGS_input);
  const auto name_inputs = absl::GetFlag(FLAGS_input_name);
  const auto signature_name_inputs = absl::GetFlag(FLAGS_signature_name);

  if (model_path.empty() || output_path.empty()) {
    ABSL_LOG(ERROR) << "--model_path and --output_path are required.";
    return 1;
  }

  int num_input_methods = 0;
  if (!positional_inputs.empty()) num_input_methods++;
  if (!name_inputs.empty()) num_input_methods++;
  if (!signature_name_inputs.empty()) num_input_methods++;

  if (num_input_methods > 1) {
    ABSL_LOG(ERROR) << "Only one of --input, --input_name, or --signature_name "
                       "can be used.";
    return 1;
  }

  auto result = litert::internal::ApplyInputShapes(
      model_path, output_path, signature_key, positional_inputs, name_inputs,
      signature_name_inputs);
  if (!result) {
    ABSL_LOG(ERROR) << result.Error().Message();
    return 1;
  }

  return 0;
}
