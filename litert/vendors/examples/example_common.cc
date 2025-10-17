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

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

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
  const std::vector<std::string> split = absl::StrSplit(str, kLineDelim);
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
  std::string inputs;
  std::string outputs;
  std::string tensors;
  std::string ops;

  static Expected<SchemaStrings> Parse(absl::string_view str) {
    const std::vector<std::string> split = absl::StrSplit(str, '\n');
    if (std::size(split) != 4) {
      LITERT_LOG(LITERT_ERROR, "Invalid graph format, expected 4 lines, got %d",
                 std::size(split));
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    LITERT_ASSIGN_OR_RETURN(auto inputs, ParseSchemaLine(split[0]));
    if (inputs.key != "inputs") {
      LITERT_LOG(LITERT_ERROR, "Invalid schema line format, expected 'inputs'");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    LITERT_ASSIGN_OR_RETURN(auto outputs, ParseSchemaLine(split[1]));
    if (outputs.key != "outputs") {
      LITERT_LOG(LITERT_ERROR,
                 "Invalid schema line format, expected 'outputs'");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    LITERT_ASSIGN_OR_RETURN(auto tensors, ParseSchemaLine(split[2]));
    if (tensors.key != "tensors") {
      LITERT_LOG(LITERT_ERROR,
                 "Invalid schema line format, expected 'tensors'");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    LITERT_ASSIGN_OR_RETURN(auto ops, ParseSchemaLine(split[3]));
    if (ops.key != "ops") {
      LITERT_LOG(LITERT_ERROR, "Invalid schema line format, expected 'ops'");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    return SchemaStrings{inputs.value, outputs.value, tensors.value, ops.value};
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
  ExampleGraph graph;

  LITERT_ASSIGN_OR_RETURN(auto schema,
                          SchemaStrings::Parse(serialized.StrView()));
  LITERT_ASSIGN_OR_RETURN(graph.inputs_, (ParseLoI<Index>(schema.inputs)));
  LITERT_ASSIGN_OR_RETURN(graph.outputs_, (ParseLoI<Index>(schema.outputs)));

  LITERT_ASSIGN_OR_RETURN(graph.tensors_, ParseTensors(schema.tensors));
  LITERT_ASSIGN_OR_RETURN(graph.ops_, ParseOps(schema.ops));

  return graph;
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
    intermediates.push_back(Data(tensor.NumElements()));
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
