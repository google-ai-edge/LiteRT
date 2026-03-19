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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
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

Expected<void> ApplyInputShapes(const std::string& model_path,
                                const std::string& output_path,
                                const std::vector<std::string>& input_strs) {
  // Use internal LoadModelFromFile directly
  auto model_res = litert::internal::LoadModelFromFile(model_path);
  if (!model_res) {
    return Unexpected(model_res.Error().Status(),
                      "Failed to load model: " + model_res.Error().Message());
  }
  auto model = std::move(*model_res);

  if (model->Subgraphs().empty()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Model has no subgraphs");
  }
  auto& subgraph = model->Subgraph(LiteRtModelT::kMainSubgraphIndex);

  if (input_strs.size() != subgraph.Inputs().size()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrFormat("Number of inputs provided (%d) does not "
                        "match model inputs (%d)",
                        input_strs.size(), subgraph.Inputs().size()));
  }

  for (size_t i = 0; i < input_strs.size(); ++i) {
    auto shape_res = ParseShape(input_strs[i]);
    if (!shape_res) {
      return shape_res.Error();
    }
    Dims shape = *shape_res;
    auto& input_tensor = subgraph.Input(i);

    LiteRtElementType element_type = kLiteRtElementTypeNone;
    if (input_tensor.Type().first == kLiteRtRankedTensorType) {
      element_type = input_tensor.Type().second.ranked_tensor_type.element_type;
    } else if (input_tensor.Type().first == kLiteRtUnrankedTensorType) {
      element_type =
          input_tensor.Type().second.unranked_tensor_type.element_type;
    }

    if (element_type == kLiteRtElementTypeNone) {
      return Unexpected(kLiteRtStatusErrorUnsupported, "Unknown input type");
    }

    input_tensor.SetType(
        MakeRankedTensorType(element_type, absl::MakeSpan(shape)));
  }

  litert::internal::ShapeInferenceEngine engine(model.get());
  auto status = engine.InferSubgraphShapes(&subgraph);
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
  const auto input_shapes = absl::GetFlag(FLAGS_input);

  if (model_path.empty() || output_path.empty()) {
    ABSL_LOG(ERROR) << "--model_path and --output_path are required.";
    return 1;
  }

  auto result =
      litert::internal::ApplyInputShapes(model_path, output_path, input_shapes);
  if (!result) {
    ABSL_LOG(ERROR) << result.Error().Message();
    return 1;
  }

  return 0;
}
