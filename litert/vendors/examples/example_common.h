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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_EXAMPLE_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_EXAMPLE_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"

namespace litert::example {

constexpr char kPluginManufacturer[] = "ExampleSocManufacturer";
constexpr char kPluginSocModel[] = "ExampleSocModel";
constexpr char kIncompatiblePluginSocModel[] = "UnsupportedSocModel";

static constexpr absl::string_view kFirstDelim = ",";
static constexpr absl::string_view kSecondDelim = "~";
static constexpr absl::string_view kThirdDelim = "x";
static constexpr absl::string_view kLineDelim = ":";

using Dim = uint32_t;
using Dims = std::vector<Dim>;
using Index = int32_t;
using Inds = std::vector<Index>;
using Data = std::vector<float>;

enum class OpCode {
  kMul,
  kSub,
  kRmsNorm,  // NOTE: No runtime support.
};

template <typename Sink>
void AbslStringify(Sink& sink, OpCode code) {
  switch (code) {
    case OpCode::kMul:
      sink.Append("mul");
      break;
    case OpCode::kSub:
      sink.Append("sub");
      break;
    case OpCode::kRmsNorm:
      sink.Append("rms_norm");
      break;
  }
}

inline Expected<OpCode> ParseOpCode(absl::string_view str) {
  if (str == "mul") {
    return OpCode::kMul;
  } else if (str == "sub") {
    return OpCode::kSub;
  } else if (str == "rms_norm") {
    return OpCode::kRmsNorm;
  } else {
    LITERT_LOG(LITERT_ERROR, "Invalid op code when parsing");
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
}

struct ExampleTensor {
 private:
  static constexpr absl::string_view kOpen = "[";
  static constexpr absl::string_view kClose = "]";

 public:
  enum class Type {
    kFloat32,  // NOTE: Only f32 supported.
  };

  bool operator==(const ExampleTensor& other) const {
    return dims == other.dims && data == other.data && type == other.type;
  }

  bool operator!=(const ExampleTensor& other) const {
    return !(*this == other);
  }

  size_t NumElements() const {
    return absl::c_accumulate(dims, 1, std::multiplies<Dim>());
  }

  Dims dims;
  Data data = {};
  Type type = Type::kFloat32;

  ExampleTensor() = default;

  explicit ExampleTensor(Dims dims, Data data = {})
      : dims(std::move(dims)), data(std::move(data)) {}

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ExampleTensor& t) {
    absl::Format(&sink, "%s%s%s", kOpen, absl::StrJoin(t.dims, kThirdDelim),
                 kClose);
  }

  static Expected<ExampleTensor> Parse(absl::string_view str);
};

struct ExampleOp {
 private:
  static constexpr absl::string_view kOpen = "(";
  static constexpr absl::string_view kClose = ")";

 public:
  OpCode code;
  Inds inputs;
  Inds outputs;

  ExampleOp(OpCode code, Inds inputs, Inds outputs)
      : code(code), inputs(std::move(inputs)), outputs(std::move(outputs)) {}

  ExampleOp() = default;

  bool operator==(const ExampleOp& other) const {
    return code == other.code && inputs == other.inputs &&
           outputs == other.outputs;
  }

  bool operator!=(const ExampleOp& other) const { return !(*this == other); }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ExampleOp& op) {
    absl::Format(&sink, "%v%s%v%s%s%v%s", op.code, kOpen,
                 absl::StrJoin(op.inputs, kFirstDelim), kClose, kOpen,
                 absl::StrJoin(op.outputs, kFirstDelim), kClose);
  }

  static Expected<ExampleOp> Parse(absl::string_view str);
};

// An in memory representation of an example IR graph. Used to both
// construct a program and read a serialized program.
class ExampleGraph {
 private:
  // clang-format off
static constexpr absl::string_view kSchema = R"(version:%s
inputs:%s
outputs:%s
const_map:%s
tensors:%s
ops:%s)";
  // clang-format on
 public:
  // Build IR (compilation).
  template <class... Args>
  Index EmplaceTensor(Args&&... args) {
    tensors_.emplace_back(std::forward<Args>(args)...);
    return tensors_.size() - 1;
  }
  template <class... Args>
  void EmplaceOp(Args&&... args) {
    ops_.emplace_back(std::forward<Args>(args)...);
  }

  // Serialize IR (compilation).
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ExampleGraph& g) {
    std::vector<std::string> const_map_strs;
    const_map_strs.reserve(g.const_map_.size());
    for (const auto& [idx, buf_id] : g.const_map_) {
      const_map_strs.push_back(absl::StrFormat("%d:%d", idx, buf_id));
    }
    absl::Format(&sink, kSchema, g.version_,
                 absl::StrJoin(g.inputs_, kFirstDelim),
                 absl::StrJoin(g.outputs_, kFirstDelim),
                 absl::StrJoin(const_map_strs, kFirstDelim),
                 absl::StrJoin(g.tensors_, kFirstDelim),
                 absl::StrJoin(g.ops_, kSecondDelim));
  }

  Expected<OwningBufferRef<uint8_t>> Serialize() const {
    const auto serialized = absl::StrFormat("%v", *this);
    return OwningBufferRef<uint8_t>(absl::string_view(serialized));
  }

  // Read IR (execution).
  const std::string& version() const { return version_; }
  const std::vector<ExampleTensor>& Tensors() const { return tensors_; }
  std::vector<ExampleTensor>& MutableTensors() { return tensors_; }
  const std::vector<ExampleOp>& Ops() const { return ops_; }
  const Inds& Inputs() const { return inputs_; }
  const Inds& Outputs() const { return outputs_; }
  const std::map<Index, uint32_t>& ConstMap() const { return const_map_; }

  template <class... Is>
  void SetInputs(Is&&... is) {
    inputs_ = Inds{std::forward<Is>(is)...};
  }

  template <class... Os>
  void SetOutputs(Os&&... os) {
    outputs_ = Inds{std::forward<Os>(os)...};
  }

  void AddConstMap(Index tensor_idx, uint32_t buf_id) {
    const_map_[tensor_idx] = buf_id;
  }

  void SetVersion(std::string version) { version_ = std::move(version); }

  // Parse IR (execution).
  static Expected<ExampleGraph> Parse(BufferRef<uint8_t> serialized);
  static Expected<ExampleGraph> Parse(absl::string_view serialized);

 private:
  std::string version_;
  std::vector<ExampleTensor> tensors_;
  std::vector<ExampleOp> ops_;
  Inds inputs_;
  Inds outputs_;
  std::map<Index, uint32_t> const_map_;
};

class ExampleGlobalGraph {
 public:
  std::map<std::string, ExampleGraph> subgraphs_;
  std::map<uint32_t, ExampleTensor> buffers_;

  static Expected<ExampleGlobalGraph> Parse(BufferRef<uint8_t> serialized);

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ExampleGlobalGraph& g) {
    for (const auto& [id, buffer] : g.buffers_) {
      absl::Format(&sink, "BUFFER %d\n%v\nDATA:%s\n", id, buffer,
                   absl::StrJoin(buffer.data, ","));
    }
    for (const auto& [name, subgraph] : g.subgraphs_) {
      absl::Format(&sink, "SUBGRAPH %s\n%v\n", name, subgraph);
    }
  }

  Expected<OwningBufferRef<uint8_t>> Serialize() const {
    const auto serialized = absl::StrFormat("%v", *this);
    return OwningBufferRef<uint8_t>(absl::string_view(serialized));
  }
};

// Executes the graph, returning output tensors.
Expected<std::vector<Data>> Execute(const ExampleGraph& graph,
                                    const std::vector<Data>& inputs);

}  // namespace litert::example

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_EXAMPLE_COMMON_H_
