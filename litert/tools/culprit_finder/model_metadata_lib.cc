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

#include "litert/tools/culprit_finder/model_metadata_lib.h"

#include <string>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl

namespace litert::tools {

namespace {
// Returns a string representing the shape of the edge. The shape is in the
// format of "[<dim_1>,<dim_2>,...]."
std::string EdgeShapeToString(const tflite::profiling::Edge& edge) {
  const std::string shape_string = absl::StrJoin(edge.shape(), ",");
  return tflite::profiling::Edge::DataType_Name(edge.data_type()) + "[" +
         shape_string + "]";
}
}  // namespace

std::vector<int> ModelMetadata::GetOutputTensorsOfNode(int node_id) {
  std::vector<int> output_tensors;
  output_tensors.reserve(node_index_to_node_proto_[node_id].outputs_size());

  for (int output_tensor_id : node_index_to_node_proto_[node_id].outputs()) {
    output_tensors.push_back(output_tensor_id);
  }
  return output_tensors;
}

std::vector<int> ModelMetadata::GetNodeIdsInRange(int start_node,
                                                  int end_node) {
  std::vector<int> node_ids;
  node_ids.reserve(interpreter_->execution_plan().size());
  for (int node_id : interpreter_->execution_plan()) {
    if (node_id >= start_node && node_id <= end_node) {
      node_ids.push_back(node_id);
    }
  }
  return node_ids;
}

std::string ModelMetadata::GetNodeIdentifier(int node_index, bool with_index) {
  auto node_proto = node_index_to_node_proto_[node_index];
  if (with_index) {
    return absl::StrFormat("[%s]:%d", node_proto.name(), node_index);
  }
  return node_proto.name();
}

std::string ModelMetadata::GetTensorIdentifier(int tensor_index) {
  if (!tensor_index_to_src_node_.contains(tensor_index)) {
    // This is an input tensor.
    return absl::StrFormat("(INPUT)->%d", tensor_index);
  }
  return absl::StrFormat(
      "(%s)->%d",
      GetNodeIdentifier(tensor_index_to_src_node_[tensor_index],
                        /*with_index=*/true),
      tensor_index);
}

std::string ModelMetadata::GetNodeShapes(int node_index) {
  const auto& node_proto = node_index_to_node_proto_[node_index];
  const std::string input_shapes =
      absl::StrJoin(node_proto.inputs(), ",", [&](std::string* out, int input) {
        auto input_edge = tensor_index_to_edge_proto_[input];
        if (input_edge.data_type() == tflite::profiling::Edge::UNKNOWN_TYPE) {
          return;
        }
        out->append(EdgeShapeToString(input_edge));
      });
  const std::string output_shapes = absl::StrJoin(
      node_proto.outputs(), ",", [&](std::string* out, int output) {
        auto output_edge = tensor_index_to_edge_proto_[output];
        if (output_edge.data_type() == tflite::profiling::Edge::UNKNOWN_TYPE) {
          return;
        }
        out->append(EdgeShapeToString(output_edge));
      });
  return "(" + input_shapes + ") -> (" + output_shapes + ")";
}

}  // namespace litert::tools
