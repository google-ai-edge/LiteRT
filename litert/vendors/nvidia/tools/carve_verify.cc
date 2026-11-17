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

#include "litert/vendors/nvidia/tools/carve_verify.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/hash/hash.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/vendors/nvidia/tools/carve_model.h"
#include "tflite/c/common.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter_options.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"

namespace litert::tools {
namespace {

// Decompositions inside decompositions end somewhere.
constexpr int kMaxExpansionDepth = 16;

// The number of an operand that is left out (index -1).
constexpr uint64_t kNoTensor = 0x6e6f6e65;

bool IsAppended(uint64_t offset) { return offset > 1; }

// One of the two models, with a number for each value its graphs compute: two
// values have the same number when they are the same expression of the
// subgraph's inputs and of constants, taken with the options of every
// operator and the attributes of every tensor on the way. The numbers are
// hashes: equal graphs give equal numbers, and different graphs give different
// numbers but for a collision of a 64-bit hash.
class Graph {
 public:
  // A composite whose decomposition is subgraph `expand_from` or a later one
  // counts as the ops of its decomposition.
  Graph(litert::BufferRef<uint8_t> bytes, size_t expand_from)
      : bytes_(bytes), expand_from_(expand_from) {}

  absl::Status Load() {
    if (!litert::internal::VerifyFlatbuffer(bytes_.Data(), bytes_.Size())) {
      return absl::DataLossError("not a valid TFLite flatbuffer");
    }
    model_.reset(tflite::GetModel(bytes_.Data())->UnPack());
    for (const auto& buffer : model_->buffers) {
      if (IsAppended(buffer->offset) &&
          (buffer->offset > bytes_.Size() ||
           buffer->size > bytes_.Size() - buffer->offset)) {
        return absl::DataLossError("a buffer lies outside the model");
      }
    }
    attributes_.resize(model_->subgraphs.size());
    for (size_t s = 0; s < model_->subgraphs.size(); ++s) {
      attributes_[s].resize(model_->subgraphs[s]->tensors.size());
    }
    expanded_.assign(model_->subgraphs.size(), 0);
    return absl::OkStatus();
  }

  const tflite::ModelT& model() const { return *model_; }
  size_t num_operators() const { return num_operators_; }
  uint64_t constant_bytes() const { return constant_bytes_; }
  // How many composites each subgraph was put in the place of.
  const std::vector<size_t>& expanded() const { return expanded_; }

  template <class T>
  uint64_t HashOf(const T& object) {
    builder_.Clear();
    builder_.Finish(T::TableType::Pack(builder_, &object));
    return absl::HashOf(absl::string_view(
        reinterpret_cast<const char*>(builder_.GetBufferPointer()),
        builder_.GetSize()));
  }

  absl::string_view BufferData(uint32_t index) const {
    if (index >= model_->buffers.size()) return {};
    const auto& buffer = *model_->buffers[index];
    if (IsAppended(buffer.offset)) {
      return absl::string_view(
          reinterpret_cast<const char*>(bytes_.Data()) + buffer.offset,
          buffer.size);
    }
    return absl::string_view(reinterpret_cast<const char*>(buffer.data.data()),
                             buffer.data.size());
  }

  // The numbers of the outputs of subgraph `s` as a function of its inputs.
  absl::StatusOr<std::vector<uint64_t>> Outputs(size_t s) {
    const auto& subgraph = *model_->subgraphs[s];
    std::vector<uint64_t> inputs;
    for (size_t i = 0; i < subgraph.inputs.size(); ++i) {
      if (!InRange(subgraph, subgraph.inputs[i])) {
        return Error(s, "an input index is out of range");
      }
      auto attributes = Attributes(s, subgraph.inputs[i]);
      if (!attributes.ok()) return attributes.status();
      inputs.push_back(absl::HashOf(i, *attributes));
    }
    return Evaluate(s, inputs, 0);
  }

  // The tensors of subgraph `s` that nothing uses: no operator, not the
  // subgraph's inputs or outputs, not the quantization of a tensor in use.
  size_t CountUnused(size_t s) const {
    const auto& subgraph = *model_->subgraphs[s];
    std::vector<bool> used(subgraph.tensors.size(), false);
    auto use = [&](const std::vector<int32_t>& indices) {
      for (const int32_t index : indices) {
        if (InRange(subgraph, index)) used[index] = true;
      }
    };
    for (const auto& op : subgraph.operators) {
      use(op->inputs);
      use(op->outputs);
      use(op->intermediates);
    }
    use(subgraph.inputs);
    use(subgraph.outputs);
    for (bool changed = true; changed;) {
      changed = false;
      for (size_t t = 0; t < subgraph.tensors.size(); ++t) {
        if (!used[t]) continue;
        for (const int32_t* ref : QuantizationTensors(*subgraph.tensors[t])) {
          if (InRange(subgraph, *ref) && !used[*ref]) {
            used[*ref] = true;
            changed = true;
          }
        }
      }
    }
    return std::count(used.begin(), used.end(), false);
  }

  // What a signature is to its caller: its key, its subgraph, and the names
  // and positions of its inputs and outputs.
  std::string Describe(const tflite::SignatureDefT& signature) const {
    std::string text = absl::StrCat(signature.signature_key, " -> subgraph ",
                                    signature.subgraph_index);
    if (signature.subgraph_index >= model_->subgraphs.size()) return text;
    const auto& subgraph = *model_->subgraphs[signature.subgraph_index];
    auto list = [&](const char* label, const auto& maps,
                    const std::vector<int32_t>& order) {
      absl::StrAppend(&text, "; ", label, ":");
      for (const auto& map : maps) {
        const auto position =
            std::find(order.begin(), order.end(),
                      static_cast<int32_t>(map->tensor_index)) -
            order.begin();
        const std::string tensor_name =
            map->tensor_index < subgraph.tensors.size()
                ? subgraph.tensors[map->tensor_index]->name
                : "?";
        absl::StrAppend(&text, " ", map->name, "@", position, "=", tensor_name);
      }
    };
    list("inputs", signature.inputs, subgraph.inputs);
    list("outputs", signature.outputs, subgraph.outputs);
    return text;
  }

 private:
  static bool InRange(const tflite::SubGraphT& subgraph, int32_t tensor) {
    return tensor >= 0 && static_cast<size_t>(tensor) < subgraph.tensors.size();
  }

  absl::Status Error(size_t s, absl::string_view message) const {
    return absl::FailedPreconditionError(absl::StrCat(
        "subgraph ", s, " (", model_->subgraphs[s]->name, "): ", message));
  }

  std::string TensorName(size_t s, int32_t tensor) const {
    return absl::StrCat("tensor ", tensor, " (",
                        model_->subgraphs[s]->tensors[tensor]->name, ")");
  }

  // Whether the tensor is a constant: its buffer gives the interpreter data,
  // which may be no bytes at all for a tensor without elements (an empty
  // shape). The unpacked model cannot tell an empty buffer from none.
  bool HasData(const tflite::TensorT& tensor) const {
    if (tensor.external_buffer != 0) return true;
    const auto* buffers = tflite::GetModel(bytes_.Data())->buffers();
    if (tensor.buffer == 0 || buffers == nullptr ||
        tensor.buffer >= buffers->size()) {
      return false;
    }
    const auto* buffer = buffers->Get(tensor.buffer);
    return buffer->data() != nullptr || IsAppended(buffer->offset());
  }

  uint64_t BufferHash(uint32_t index) {
    auto [it, inserted] = buffer_hashes_.try_emplace(index, 0);
    if (inserted) {
      const absl::string_view data = BufferData(index);
      constant_bytes_ += data.size();
      it->second = absl::HashOf(data);
    }
    return it->second;
  }

  // The number of everything a tensor is besides its value and its name: the
  // type, the shape, the quantization, and the scale and zero point tensors
  // the quantization names, which must be constants of the same subgraph.
  absl::StatusOr<uint64_t> Attributes(size_t s, int32_t t) {
    if (attributes_[s][t]) return *attributes_[s][t];
    const auto& subgraph = *model_->subgraphs[s];
    tflite::TensorT copy = *subgraph.tensors[t];
    copy.name.clear();
    copy.buffer = 0;
    copy.external_buffer = 0;
    std::vector<uint64_t> parameters;
    for (int32_t* ref : QuantizationTensors(copy)) {
      if (*ref == -1) continue;
      if (!InRange(subgraph, *ref) || !HasData(*subgraph.tensors[*ref])) {
        return Error(s, absl::StrCat("the quantization of ", TensorName(s, t),
                                     " names tensor ", *ref,
                                     ", which is not a constant of the "
                                     "subgraph"));
      }
      auto parameter = Constant(s, *ref);
      if (!parameter.ok()) return parameter.status();
      parameters.push_back(*parameter);
      *ref = 0;
    }
    const uint64_t number = absl::HashOf(HashOf(copy), parameters);
    attributes_[s][t] = number;
    return number;
  }

  absl::StatusOr<uint64_t> Constant(size_t s, int32_t t) {
    auto attributes = Attributes(s, t);
    if (!attributes.ok()) return attributes.status();
    const auto& tensor = *model_->subgraphs[s]->tensors[t];
    // An external constant is the slice of the weights its id names; the
    // table of the slices is compared with the rest of the model.
    return absl::HashOf(
        *attributes, tensor.external_buffer,
        tensor.external_buffer != 0 ? uint64_t{0} : BufferHash(tensor.buffer));
  }

  // The decomposition to put in the place of `op`, if it is a composite that
  // counts as its decomposition.
  std::optional<size_t> Expansion(const tflite::OperatorT& op) const {
    if (op.opcode_index >= model_->operator_codes.size() ||
        model_->operator_codes[op.opcode_index]->builtin_code !=
            tflite::BuiltinOperator_STABLEHLO_COMPOSITE) {
      return std::nullopt;
    }
    const auto* options = op.builtin_options_2.AsStableHLOCompositeOptions();
    if (options == nullptr || options->decomposition_subgraph_index < 0) {
      return std::nullopt;
    }
    const size_t index = options->decomposition_subgraph_index;
    return index >= expand_from_ ? std::optional<size_t>(index) : std::nullopt;
  }

  absl::StatusOr<std::vector<uint64_t>> Evaluate(
      size_t s, const std::vector<uint64_t>& inputs, int depth) {
    const auto& subgraph = *model_->subgraphs[s];
    if (depth > kMaxExpansionDepth) {
      return Error(s, "decompositions are nested too deeply");
    }
    if (inputs.size() != subgraph.inputs.size()) {
      return Error(s, "called with the wrong number of inputs");
    }
    std::vector<std::optional<uint64_t>> numbers(subgraph.tensors.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      const int32_t input = subgraph.inputs[i];
      if (!InRange(subgraph, input) || numbers[input]) {
        return Error(s, "an input is out of range or listed twice");
      }
      numbers[input] = inputs[i];
    }
    // The number of a tensor that an operator or the subgraph's outputs read.
    auto read = [&](int32_t t,
                    absl::string_view reader) -> absl::StatusOr<uint64_t> {
      if (!InRange(subgraph, t)) {
        return Error(s, absl::StrCat(reader, " reads a tensor out of range"));
      }
      if (!numbers[t]) {
        if (!HasData(*subgraph.tensors[t])) {
          return Error(s, absl::StrCat(TensorName(s, t), " is read by ", reader,
                                       " before it is written"));
        }
        auto constant = Constant(s, t);
        if (!constant.ok()) return constant.status();
        numbers[t] = *constant;
      }
      return *numbers[t];
    };

    for (size_t o = 0; o < subgraph.operators.size(); ++o) {
      const auto& op = *subgraph.operators[o];
      const std::string op_name = absl::StrCat("operator ", o);
      ++num_operators_;
      std::vector<uint64_t> operands;
      for (const int32_t input : op.inputs) {
        if (input == -1) {
          operands.push_back(kNoTensor);
          continue;
        }
        auto operand = read(input, op_name);
        if (!operand.ok()) return operand.status();
        operands.push_back(*operand);
      }
      for (const int32_t output : op.outputs) {
        if (!InRange(subgraph, output) || numbers[output] ||
            HasData(*subgraph.tensors[output])) {
          return Error(s, absl::StrCat(op_name,
                                       " writes a tensor that is out "
                                       "of range, a constant or already "
                                       "written"));
        }
      }

      const auto expansion = Expansion(op);
      if (!expansion) {
        tflite::OperatorT copy = op;
        copy.inputs.clear();
        copy.outputs.clear();
        copy.opcode_index = 0;
        if (op.opcode_index >= model_->operator_codes.size()) {
          return Error(s, absl::StrCat(op_name, " has no operator code"));
        }
        const uint64_t op_number = absl::HashOf(
            HashOf(copy), HashOf(*model_->operator_codes[op.opcode_index]),
            operands);
        for (size_t j = 0; j < op.outputs.size(); ++j) {
          auto attributes = Attributes(s, op.outputs[j]);
          if (!attributes.ok()) return attributes.status();
          numbers[op.outputs[j]] = absl::HashOf(op_number, j, *attributes);
        }
        continue;
      }

      // A composite of this run: its decomposition computes its outputs from
      // its inputs, and a kernel that runs the decomposition reads the
      // tensors of the decomposition, so they must be what the composite's
      // own tensors are.
      const size_t d = *expansion;
      if (d >= model_->subgraphs.size()) {
        return Error(s, absl::StrCat(op_name, " has no decomposition"));
      }
      const auto& decomposition = *model_->subgraphs[d];
      if (decomposition.inputs.size() != op.inputs.size() ||
          decomposition.outputs.size() != op.outputs.size()) {
        return Error(s, absl::StrCat(op_name,
                                     " and its decomposition differ "
                                     "in the number of inputs or outputs"));
      }
      auto same_tensor = [&](int32_t outer, int32_t inner) -> absl::Status {
        if (!InRange(subgraph, outer) || !InRange(decomposition, inner)) {
          return Error(s, absl::StrCat(op_name,
                                       " or its decomposition names "
                                       "a tensor out of range"));
        }
        auto a = Attributes(s, outer);
        if (!a.ok()) return a.status();
        auto b = Attributes(d, inner);
        if (!b.ok()) return b.status();
        if (*a != *b) {
          return Error(s, absl::StrCat(TensorName(s, outer), " of ", op_name,
                                       " differs from ", TensorName(d, inner),
                                       " of its decomposition in type, shape "
                                       "or quantization"));
        }
        return absl::OkStatus();
      };
      for (size_t i = 0; i < op.inputs.size(); ++i) {
        if (auto status = same_tensor(op.inputs[i], decomposition.inputs[i]);
            !status.ok()) {
          return status;
        }
      }
      auto results = Evaluate(d, operands, depth + 1);
      if (!results.ok()) return results.status();
      for (size_t j = 0; j < op.outputs.size(); ++j) {
        if (auto status = same_tensor(op.outputs[j], decomposition.outputs[j]);
            !status.ok()) {
          return status;
        }
        numbers[op.outputs[j]] = (*results)[j];
      }
      ++expanded_[d];
    }

    std::vector<uint64_t> outputs;
    for (const int32_t output : subgraph.outputs) {
      auto number = read(output, "the subgraph's outputs");
      if (!number.ok()) return number.status();
      outputs.push_back(*number);
    }
    return outputs;
  }

  const litert::BufferRef<uint8_t> bytes_;
  const size_t expand_from_;
  std::unique_ptr<tflite::ModelT> model_;
  flatbuffers::FlatBufferBuilder builder_;
  // The attributes number of each tensor of each subgraph, once computed.
  std::vector<std::vector<std::optional<uint64_t>>> attributes_;
  absl::flat_hash_map<uint32_t, uint64_t> buffer_hashes_;
  std::vector<size_t> expanded_;
  size_t num_operators_ = 0;
  uint64_t constant_bytes_ = 0;
};

// NUMERIC CHECK

// A hash that is the same in every run, to seed the inputs.
uint64_t StableHash(absl::string_view text) {
  uint64_t hash = 0xcbf29ce484222325;
  for (const char c : text) {
    hash = (hash ^ static_cast<uint8_t>(c)) * 0x100000001b3;
  }
  return hash;
}

void FillInput(TfLiteTensor* tensor, uint64_t seed) {
  std::mt19937_64 random(seed);
  switch (tensor->type) {
    case kTfLiteFloat32: {
      std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
      const size_t count = tensor->bytes / sizeof(float);
      for (size_t i = 0; i < count; ++i) tensor->data.f[i] = uniform(random);
      break;
    }
    case kTfLiteFloat16: {
      // Halves that are exact: 0, +-0.25, +-0.5, +-1, 2.
      constexpr uint16_t kHalves[] = {0x0000, 0x3400, 0xb400, 0x3800,
                                      0xb800, 0x3c00, 0xbc00, 0x4000};
      const size_t count = tensor->bytes / sizeof(uint16_t);
      for (size_t i = 0; i < count; ++i) {
        std::memcpy(tensor->data.raw + i * sizeof(uint16_t),
                    &kHalves[random() % 8], sizeof(uint16_t));
      }
      break;
    }
    case kTfLiteBool:
      // Masks: everything visible, so that no row of an attention is empty.
      std::memset(tensor->data.raw, 1, tensor->bytes);
      break;
    case kTfLiteInt2:
    case kTfLiteInt4:
    case kTfLiteUInt4:
    case kTfLiteInt8:
    case kTfLiteUInt8:
    case kTfLiteInt16:
      // Quantized data: every code is a value.
      for (size_t i = 0; i < tensor->bytes; ++i) {
        tensor->data.raw[i] = static_cast<char>(random());
      }
      break;
    default:
      // Positions and indices: 0 is in range everywhere.
      std::memset(tensor->data.raw, 0, tensor->bytes);
      break;
  }
}

struct OutputDigest {
  std::string name;
  size_t bytes = 0;
  uint64_t hash = 0;
  // Of the floating-point elements, how many are not numbers.
  size_t nans = 0;
  size_t floats = 0;
};

struct SignatureResult {
  std::string key;
  std::vector<OutputDigest> outputs;
};

absl::StatusOr<std::vector<SignatureResult>> RunOnCpu(
    litert::BufferRef<uint8_t> bytes, const NumericCheckOptions& options) {
  auto error = [](absl::string_view message) {
    return absl::InternalError(absl::StrCat("CPU run: ", message));
  };
  const auto model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(bytes.Data()), bytes.Size());
  if (model == nullptr) return error("cannot read the model");
  if (const auto* external = model->GetModel()->external_buffers();
      external != nullptr && external->size() > 0) {
    return absl::UnimplementedError(
        "the model keeps constants in external buffers, which this check "
        "does not load");
  }

  // As LiteRT's compiled model sets the interpreter up for the CPU. The
  // delegate is declared first: the interpreter uses it until it is destroyed.
  TfLiteXNNPackDelegateOptions xnnpack_options =
      TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_options.num_threads = options.num_threads;
  const std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> xnnpack(
      TfLiteXNNPackDelegateCreate(&xnnpack_options),
      TfLiteXNNPackDelegateDelete);
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterOptions interpreter_options;
  interpreter_options.SetShloCompositeInlining(true);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder builder(*model, resolver, &interpreter_options);
  if (builder(&interpreter, options.num_threads) != kTfLiteOk ||
      interpreter == nullptr) {
    return error("cannot build the interpreter");
  }
  if (interpreter->ModifyGraphWithDelegate(xnnpack.get()) != kTfLiteOk) {
    return error("cannot apply XNNPACK");
  }

  std::vector<std::string> keys = options.signatures;
  if (keys.empty()) {
    for (const std::string* key : interpreter->signature_keys()) {
      keys.push_back(*key);
    }
  }
  std::vector<SignatureResult> results;
  // A model without signatures has its first subgraph to run.
  const bool unnamed = keys.empty();
  if (unnamed) keys.emplace_back();
  for (const auto& key : keys) {
    auto* runner =
        interpreter->GetSignatureRunner(unnamed ? nullptr : key.c_str());
    if (runner == nullptr) {
      return error(absl::StrCat("no signature '", key, "'"));
    }
    if (runner->AllocateTensors() != kTfLiteOk) {
      return error(absl::StrCat("cannot allocate the tensors of '", key, "'"));
    }
    for (const char* name : runner->input_names()) {
      FillInput(runner->input_tensor(name),
                StableHash(absl::StrCat(key, "/", name)));
    }
    if (runner->Invoke() != kTfLiteOk) {
      return error(absl::StrCat("'", key, "' failed to run"));
    }
    auto& result = results.emplace_back();
    result.key = key;
    for (const char* name : runner->output_names()) {
      const TfLiteTensor* tensor = runner->output_tensor(name);
      auto& output = result.outputs.emplace_back();
      output.name = name;
      output.bytes = tensor->bytes;
      output.hash =
          absl::HashOf(absl::string_view(tensor->data.raw, tensor->bytes));
      if (tensor->type == kTfLiteFloat32) {
        output.floats = tensor->bytes / sizeof(float);
        for (size_t i = 0; i < output.floats; ++i) {
          output.nans += std::isnan(tensor->data.f[i]) ? 1 : 0;
        }
      }
    }
  }
  return results;
}

}  // namespace

absl::Status VerifyCarvedStructure(litert::BufferRef<uint8_t> original,
                                   litert::BufferRef<uint8_t> carved,
                                   std::string* summary) {
  Graph before(original, std::numeric_limits<size_t>::max());
  if (auto status = before.Load(); !status.ok()) return status;
  const auto& a = before.model();
  Graph after(carved, a.subgraphs.size());
  if (auto status = after.Load(); !status.ok()) return status;
  const auto& b = after.model();
  auto differs = [](absl::string_view what) {
    return absl::FailedPreconditionError(
        absl::StrCat("the carved model differs from the original in ", what));
  };

  // What a model holds besides its graphs.
  if (a.version != b.version || a.description != b.description) {
    return differs("its version or description");
  }
  if (b.operator_codes.size() < a.operator_codes.size()) {
    return differs("its operator codes");
  }
  for (size_t i = 0; i < b.operator_codes.size(); ++i) {
    // The composite's code is the one a rewrite may add.
    const bool same = i < a.operator_codes.size()
                          ? before.HashOf(*a.operator_codes[i]) ==
                                after.HashOf(*b.operator_codes[i])
                          : b.operator_codes[i]->builtin_code ==
                                tflite::BuiltinOperator_STABLEHLO_COMPOSITE;
    if (!same) return differs(absl::StrCat("operator code ", i));
  }
  if (a.metadata.size() != b.metadata.size() ||
      a.metadata_buffer.size() != b.metadata_buffer.size()) {
    return differs("its metadata");
  }
  for (size_t i = 0; i < a.metadata.size(); ++i) {
    if (a.metadata[i]->name != b.metadata[i]->name ||
        before.BufferData(a.metadata[i]->buffer) !=
            after.BufferData(b.metadata[i]->buffer)) {
      return differs(absl::StrCat("metadata ", a.metadata[i]->name));
    }
  }
  if (a.external_buffers.size() != b.external_buffers.size() ||
      a.external_buffer_groups.size() != b.external_buffer_groups.size()) {
    return differs("its external buffers");
  }
  for (size_t i = 0; i < a.external_buffers.size(); ++i) {
    if (before.HashOf(*a.external_buffers[i]) !=
        after.HashOf(*b.external_buffers[i])) {
      return differs(absl::StrCat("external buffer ", i));
    }
  }
  for (size_t i = 0; i < a.external_buffer_groups.size(); ++i) {
    if (a.external_buffer_groups[i]->name !=
        b.external_buffer_groups[i]->name) {
      return differs(absl::StrCat("external buffer group ", i));
    }
  }
  if (a.signature_defs.size() != b.signature_defs.size()) {
    return differs("the number of signatures");
  }
  for (size_t i = 0; i < a.signature_defs.size(); ++i) {
    const std::string x = before.Describe(*a.signature_defs[i]);
    const std::string y = after.Describe(*b.signature_defs[i]);
    if (x != y) {
      return differs(absl::StrCat("signature ", i, ": ", x, " became ", y));
    }
  }

  // The graphs.
  if (b.subgraphs.size() < a.subgraphs.size()) {
    return differs("the number of subgraphs");
  }
  size_t unused = 0;
  for (size_t s = 0; s < a.subgraphs.size(); ++s) {
    if (a.subgraphs[s]->name != b.subgraphs[s]->name) {
      return differs(absl::StrCat("the name of subgraph ", s));
    }
    auto x = before.Outputs(s);
    if (!x.ok()) {
      return absl::FailedPreconditionError(
          absl::StrCat("the original model: ", x.status().message()));
    }
    auto y = after.Outputs(s);
    if (!y.ok()) {
      return absl::FailedPreconditionError(
          absl::StrCat("the carved model: ", y.status().message()));
    }
    if (x->size() != y->size()) {
      return differs(absl::StrCat("the outputs of subgraph ", s));
    }
    for (size_t i = 0; i < x->size(); ++i) {
      if ((*x)[i] != (*y)[i]) {
        return differs(absl::StrCat("what output ", i, " of subgraph ", s, " (",
                                    a.subgraphs[s]->name, ") computes"));
      }
    }
    const size_t unused_before = before.CountUnused(s);
    const size_t unused_after = after.CountUnused(s);
    if (unused_after > unused_before) {
      return absl::FailedPreconditionError(absl::StrCat(
          "the carved model leaves ", unused_after - unused_before,
          " tensors unused in subgraph ", s, " (", a.subgraphs[s]->name, ")"));
    }
    unused += unused_after;
  }
  size_t composites = 0;
  for (size_t s = a.subgraphs.size(); s < b.subgraphs.size(); ++s) {
    if (after.expanded()[s] != 1) {
      return absl::FailedPreconditionError(
          absl::StrCat("subgraph ", s, " (", b.subgraphs[s]->name,
                       ") of the carved model is the decomposition of ",
                       after.expanded()[s], " composites, not of one"));
    }
    if (const size_t left = after.CountUnused(s); left > 0) {
      return absl::FailedPreconditionError(
          absl::StrCat("decomposition ", s, " (", b.subgraphs[s]->name,
                       ") holds ", left, " tensors nothing uses"));
    }
    ++composites;
  }
  *summary = absl::StrFormat(
      "%d subgraphs compute what they computed with %d new composites in the "
      "place of their decompositions (%d operators, %.1f MiB of constants "
      "compared); %d signatures, the metadata and the operator codes are as "
      "they were; %d unused tensors, none of them new",
      a.subgraphs.size(), composites, before.num_operators(),
      before.constant_bytes() / 1048576.0, a.signature_defs.size(), unused);
  return absl::OkStatus();
}

absl::Status CompareOnCpu(litert::BufferRef<uint8_t> original,
                          litert::BufferRef<uint8_t> carved,
                          const NumericCheckOptions& options,
                          std::string* summary) {
  // One model at a time: each run holds a copy of the weights that XNNPACK
  // packs.
  auto before = RunOnCpu(original, options);
  if (!before.ok()) return before.status();
  auto after = RunOnCpu(carved, options);
  if (!after.ok()) return after.status();
  if (before->size() != after->size()) {
    return absl::FailedPreconditionError(
        "the models differ in the number of signatures");
  }
  size_t outputs = 0;
  uint64_t bytes = 0;
  std::string weak;
  for (size_t s = 0; s < before->size(); ++s) {
    const auto& x = (*before)[s];
    const auto& y = (*after)[s];
    if (x.outputs.size() != y.outputs.size()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "signature '", x.key, "' has a different number of outputs"));
    }
    size_t nans = 0;
    size_t floats = 0;
    for (size_t i = 0; i < x.outputs.size(); ++i) {
      const auto& p = x.outputs[i];
      const auto& q = y.outputs[i];
      if (p.name != q.name || p.bytes != q.bytes || p.hash != q.hash) {
        return absl::FailedPreconditionError(absl::StrCat(
            "output '", p.name, "' of signature '", x.key,
            "' differs between the original and the carved model"));
      }
      ++outputs;
      bytes += p.bytes;
      nans += p.nans;
      floats += p.floats;
    }
    // Equal outputs that are not numbers say little.
    if (floats > 0 && nans == floats) {
      absl::StrAppend(&weak, " (every floating-point output of '", x.key,
                      "' is NaN for these inputs)");
    }
  }
  *summary = absl::StrFormat(
      "%d outputs of %d signatures (%d bytes) are bit-identical on the CPU%s",
      outputs, before->size(), bytes, weak);
  return absl::OkStatus();
}

}  // namespace litert::tools
