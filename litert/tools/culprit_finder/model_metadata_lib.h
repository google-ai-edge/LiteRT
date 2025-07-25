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

#ifndef ODML_LITERT_TOOLS_CULPRIT_FINDER_MODEL_METADATA_LIB_H_
#define ODML_LITERT_TOOLS_CULPRIT_FINDER_MODEL_METADATA_LIB_H_
#include <memory>
#include <unordered_map>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "tflite/interpreter.h"
#include "tflite/profiling/model_runtime_info.h"
#include "tflite/tools/logging.h"

namespace litert::tools {

// ModelMetadata is used to store the metadata of the model and provide methods
// to get the node identifier, tensor identifier, node shapes, output tensors of
// a node, node ids in a range from an execution plan, etc.
class ModelMetadata {
 public:
  explicit ModelMetadata(
      const tflite::profiling::ModelRuntimeDetails& model_runtime_details) {
    for (const auto& subgraph : model_runtime_details.subgraphs()) {
      // Only interested in the default subgraph (single signature) for now.
      if (subgraph.subgraph_id() == 0) {
        for (const auto& node : subgraph.nodes()) {
          node_index_to_node_proto_[node.id()] = node;
        }
        break;
      }
    }
  };

  ~ModelMetadata() = default;

  static litert::Expected<std::unique_ptr<ModelMetadata>> Create(
      tflite::Interpreter* interpreter) {
    tflite::profiling::ModelRuntimeDetails model_runtime_details;
    if (tflite::profiling::GenerateModelRuntimeInfo(
            *interpreter, model_runtime_details) != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to generate model runtime info";
      return litert::Unexpected(LiteRtStatus::kLiteRtStatusErrorRuntimeFailure,
                                "Failed to generate model runtime info");
    }
    return std::make_unique<ModelMetadata>(model_runtime_details);
  }

  // Returns a vector of output tensor indices for the given node.
  std::vector<int> GetOutputTensorsOfNode(int node_id);

 private:
  std::unordered_map<int, tflite::profiling::Node> node_index_to_node_proto_;
};
}  // namespace litert::tools

#endif  // ODML_LITERT_TOOLS_CULPRIT_FINDER_MODEL_METADATA_LIB_H_
