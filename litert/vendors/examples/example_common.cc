// Copyright 2024 Google LLC.
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

#include "litert/vendors/examples/example_common.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::example {
namespace {

struct SchemaLine {
  std::string key;
  std::string value;

  template <typename S>
  SchemaLine(S&& key, S&& value)
      : key(std::forward<S>(key)), value(std::forward<S>(value)) {}
};

Expected<SchemaLine> ParseSchemaLine(absl::string_view str) {
  const std::vector<std::string> split =
      absl::StrSplit(str, absl::MaxSplits(kLineDelim, 1));
  if (split.size() != 2) {
    LITERT_LOG(LITERT_ERROR, "Invalid schema line format, expected 2 colons");
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return SchemaLine(std::string(split[0]), std::string(split[1]));
}

template <typename T>
Expected<std::vector<T>> ParseLoI(absl::string_view str,
                                  absl::string_view delim = kFirstDelim) {
  std::vector<T> ts;
  for (const auto& i : absl::StrSplit(str, delim)) {
    T t;
    if (!absl::SimpleAtoi(i, &t)) {
      LITERT_LOG(LITERT_ERROR, "Could not parse index");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    ts.push_back(t);
  }
  return ts;
}

template <typename T>
Expected<std::vector<T>> ParseLoF(absl::string_view str,
                                  absl::string_view delim = kFirstDelim) {
  std::vector<T> ts;
  for (const auto& i : absl::StrSplit(str, delim)) {
    T t;
    if (!absl::SimpleAtof(i, &t)) {
      LITERT_LOG(LITERT_ERROR, "Could not parse float");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    ts.push_back(t);
  }
  return ts;
}

Expected<std::vector<ExampleTensor>> ParseTensors(absl::string_view str) {
  std::vector<ExampleTensor> tensors;
  for (const auto& t : absl::StrSplit(str, kFirstDelim)) {
    LITERT_ASSIGN_OR_RETURN(auto tensor, ExampleTensor::Parse(t));
    tensors.push_back(std::move(tensor));
  }
  return tensors;
}

Expected<std::vector<ExampleOp>> ParseOps(absl::string_view str) {
  std::vector<ExampleOp> ops;
  for (const auto& op_str : absl::StrSplit(str, kSecondDelim)) {
    LITERT_ASSIGN_OR_RETURN(auto op, ExampleOp::Parse(op_str));
    ops.push_back(std::move(op));
  }
  return ops;
}

// Intermediate struct for parsing the schema.
struct SchemaStrings {
  std::string version;
  std::string inputs;
  std::string outputs;
  std::string const_map;
  std::string tensors;
  std::string ops;

  static Expected<SchemaStrings> Parse(absl::string_view str) {
    const std::vector<std::string> split = absl::StrSplit(str, '\n');
    if (std::size(split) == 5) {
      // Legacy format without const_map
      LITERT_ASSIGN_OR_RETURN(auto version, ParseSchemaLine(split[0]));
      LITERT_ASSIGN_OR_RETURN(auto inputs, ParseSchemaLine(split[1]));
      LITERT_ASSIGN_OR_RETURN(auto outputs, ParseSchemaLine(split[2]));
      LITERT_ASSIGN_OR_RETURN(auto tensors, ParseSchemaLine(split[3]));
      LITERT_ASSIGN_OR_RETURN(auto ops, ParseSchemaLine(split[4]));
      return SchemaStrings{version.value,    inputs.value,  outputs.value,
                           /*const_map=*/"", tensors.value, ops.value};
    }

    if (std::size(split) != 6) {
      LITERT_LOG(LITERT_ERROR,
                 "Invalid graph format, expected 5 or 6 lines, got %d",
                 std::size(split));
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    LITERT_ASSIGN_OR_RETURN(auto version, ParseSchemaLine(split[0]));
    if (version.key != "version") {
      LITERT_LOG(LITERT_ERROR,
                 "Invalid schema line format, expected 'version'");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    LITERT_ASSIGN_OR_RETURN(auto inputs, ParseSchemaLine(split[1]));
    if (inputs.key != "inputs") {
      LITERT_LOG(LITERT_ERROR, "Invalid schema line format, expected 'inputs'");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    LITERT_ASSIGN_OR_RETURN(auto outputs, ParseSchemaLine(split[2]));
    if (outputs.key != "outputs") {
      LITERT_LOG(LITERT_ERROR,
                 "Invalid schema line format, expected 'outputs'");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    LITERT_ASSIGN_OR_RETURN(auto const_map, ParseSchemaLine(split[3]));
    if (const_map.key != "const_map") {
      LITERT_LOG(LITERT_ERROR,
                 "Invalid schema line format, expected 'const_map'");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    LITERT_ASSIGN_OR_RETURN(auto tensors, ParseSchemaLine(split[4]));
    if (tensors.key != "tensors") {
      LITERT_LOG(LITERT_ERROR,
                 "Invalid schema line format, expected 'tensors'");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    LITERT_ASSIGN_OR_RETURN(auto ops, ParseSchemaLine(split[5]));
    if (ops.key != "ops") {
      LITERT_LOG(LITERT_ERROR, "Invalid schema line format, expected 'ops'");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    return SchemaStrings{version.value,   inputs.value,  outputs.value,
                         const_map.value, tensors.value, ops.value};
  }
};

}  // namespace

Expected<ExampleTensor> ExampleTensor::Parse(absl::string_view str) {
  if (!(::litert::StartsWith(str, kOpen) && ::litert::EndsWith(str, kClose))) {
    LITERT_LOG(
        LITERT_ERROR,
        "Invalid tensor format, must start and end with square brackets");
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  LITERT_ASSIGN_OR_RETURN(
      auto dims, (ParseLoI<Dim>(str.substr(1, str.size() - 2), kThirdDelim)));
  return ExampleTensor(std::move(dims));
}

Expected<ExampleOp> ExampleOp::Parse(absl::string_view str) {
  ExampleOp op;

  const auto inputs_start = str.find(kOpen) + 1;
  LITERT_ASSIGN_OR_RETURN(op.code,
                          ParseOpCode(str.substr(0, inputs_start - 1)));

  auto inputs_str = str.substr(inputs_start, str.find(kClose) - inputs_start);
  LITERT_ASSIGN_OR_RETURN(op.inputs, (ParseLoI<Index>(inputs_str)));

  const auto outputs_start = str.rfind(kOpen) + 1;
  auto outputs_str =
      str.substr(outputs_start, str.rfind(kClose) - outputs_start);

  LITERT_ASSIGN_OR_RETURN(op.outputs, (ParseLoI<Index>(outputs_str)));

  return op;
}

Expected<ExampleGraph> ExampleGraph::Parse(BufferRef<uint8_t> serialized) {
  return Parse(serialized.StrView());
}

Expected<ExampleGraph> ExampleGraph::Parse(absl::string_view serialized) {
  ExampleGraph graph;

  LITERT_ASSIGN_OR_RETURN(auto schema, SchemaStrings::Parse(serialized));
  graph.version_ = schema.version;
  LITERT_ASSIGN_OR_RETURN(graph.inputs_, (ParseLoI<Index>(schema.inputs)));
  LITERT_ASSIGN_OR_RETURN(graph.outputs_, (ParseLoI<Index>(schema.outputs)));

  for (const auto& entry : absl::StrSplit(schema.const_map, kFirstDelim)) {
    if (entry.empty()) {
      continue;
    }
    std::vector<std::string> kv = absl::StrSplit(entry, ':');
    if (kv.size() != 2) {
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    Index idx;
    uint32_t buf_id;
    if (!absl::SimpleAtoi(kv[0], &idx) || !absl::SimpleAtoi(kv[1], &buf_id)) {
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    graph.const_map_[idx] = buf_id;
  }

  LITERT_ASSIGN_OR_RETURN(graph.tensors_, ParseTensors(schema.tensors));
  LITERT_ASSIGN_OR_RETURN(graph.ops_, ParseOps(schema.ops));

  return graph;
}

Expected<ExampleGlobalGraph> ExampleGlobalGraph::Parse(
    BufferRef<uint8_t> serialized) {
  ExampleGlobalGraph global_graph;
  const std::vector<std::string> lines =
      absl::StrSplit(serialized.StrView(), '\n');

  std::string current_mode = "";
  std::string current_key;
  std::vector<std::string> buffer_accumulator;

  auto flush_accumulator = [&]() -> Expected<void> {
    if (current_mode == "SUBGRAPH") {
      std::string content = absl::StrJoin(buffer_accumulator, "\n");
      LITERT_ASSIGN_OR_RETURN(auto graph, ExampleGraph::Parse(content));
      global_graph.subgraphs_[current_key] = std::move(graph);
    } else if (current_mode == "BUFFER") {
      if (buffer_accumulator.size() < 2) {
        return Error(kLiteRtStatusErrorInvalidArgument);
      }
      LITERT_ASSIGN_OR_RETURN(auto tensor,
                              ExampleTensor::Parse(buffer_accumulator[0]));
      absl::string_view data_line = buffer_accumulator[1];
      if (!absl::StartsWith(data_line, "DATA:")) {
        return Error(kLiteRtStatusErrorInvalidArgument);
      }
      data_line.remove_prefix(5);
      LITERT_ASSIGN_OR_RETURN(auto data, ParseLoF<float>(data_line, ","));
      tensor.data = std::move(data);
      uint32_t id;
      if (!absl::SimpleAtoi(current_key, &id)) {
        return Error(kLiteRtStatusErrorInvalidArgument);
      }
      global_graph.buffers_[id] = std::move(tensor);
    }
    buffer_accumulator.clear();
    return {};
  };

  for (const auto& line : lines) {
    if (absl::StartsWith(line, "SUBGRAPH ")) {
      LITERT_RETURN_IF_ERROR(flush_accumulator());
      current_mode = "SUBGRAPH";
      current_key = line.substr(9);
    } else if (absl::StartsWith(line, "BUFFER ")) {
      LITERT_RETURN_IF_ERROR(flush_accumulator());
      current_mode = "BUFFER";
      current_key = line.substr(7);
    } else if (!line.empty()) {
      buffer_accumulator.push_back(line);
    }
  }
  LITERT_RETURN_IF_ERROR(flush_accumulator());

  return global_graph;
}

Expected<std::vector<Data>> Execute(const ExampleGraph& graph,
                                    const std::vector<Data>& inputs) {
  // Validate.
  if (inputs.size() != graph.Inputs().size()) {
    LITERT_LOG(LITERT_ERROR, "Expected %d inputs, got %d",
               graph.Inputs().size(), inputs.size());
    return Error(kLiteRtStatusErrorInvalidArgument);
  }

  // Setup state
  std::vector<Data> intermediates;
  for (const auto& tensor : graph.Tensors()) {
    if (!tensor.data.empty()) {
      intermediates.push_back(tensor.data);
    } else {
      intermediates.push_back(Data(tensor.NumElements()));
    }
  }
  for (auto i = 0; i < graph.Inputs().size(); ++i) {
    intermediates[graph.Inputs()[i]] = inputs[i];
  }

  auto binary = [](float lhs, float rhs, float& result, OpCode code) {
    if (code == OpCode::kMul) {
      result = lhs * rhs;
    } else {
      result = lhs - rhs;
    }
  };

  // Execute loop.
  for (const auto& op : graph.Ops()) {
    if (op.code == OpCode::kRmsNorm) {
      LITERT_LOG(LITERT_ERROR, "RmsNorm not supported");
      return Error(kLiteRtStatusErrorUnsupported);
    }

    const auto num_elements = intermediates[op.inputs[0]].size();
    for (auto i = 0; i < num_elements; ++i) {
      binary(intermediates[op.inputs[0]][i], intermediates[op.inputs[1]][i],
             intermediates[op.outputs[0]][i], op.code);
    }
  }

  // Move results back to caller.
  std::vector<Data> results(graph.Outputs().size());
  for (auto i = 0; i < graph.Outputs().size(); ++i) {
    results[i] = std::move(intermediates[graph.Outputs()[i]]);
  }

  return results;
}

}  // namespace litert::example
