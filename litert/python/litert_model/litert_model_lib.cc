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

#include "litert/python/litert_model/litert_model_lib.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/graph_validation.h"
#include "litert/core/model/model.h"
#include "litert/core/model/subgraph_refs.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::litert_model {
namespace {

using ::litert::internal::AttachInput;
using ::litert::internal::AttachOutput;
using ::litert::internal::DCE;
using ::litert::internal::Drop;
using ::litert::internal::FindInput;
using ::litert::internal::FindOutput;
using ::litert::internal::GetTflOpCodeInd;
using ::litert::internal::GetTflOpCodes;
using ::litert::internal::GetTflOptions;
using ::litert::internal::GetTflOptions2;
using ::litert::internal::HasSubgraphRefs;
using ::litert::internal::kDispatchOpCodeTflInd;
using ::litert::internal::MarkDead;
using ::litert::internal::SetTflOpCodeInd;
using ::litert::internal::SetTflOpCodes;
using ::litert::internal::SetTflOptions;
using ::litert::internal::SetTflOptions2;
using ::litert::internal::TakeTflOpCodes;
using ::litert::internal::TakeTflOptions;
using ::litert::internal::TakeTflOptions2;
using ::litert::internal::TflOpCode;
using ::litert::internal::TflOpCodePtr;
using ::litert::internal::TflOptions;
using ::litert::internal::TflOptions2;

using OpPositions = absl::flat_hash_map<const LiteRtOpT*, size_t>;
using TensorSet = absl::flat_hash_set<const LiteRtTensorT*>;

OpPositions GetOpPositions(const LiteRtSubgraphT& subgraph) {
  OpPositions positions;
  const auto ops = subgraph.Ops();
  positions.reserve(ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    positions.emplace(ops[i], i);
  }
  return positions;
}

TensorSet GetTensorSet(const LiteRtSubgraphT& subgraph) {
  const auto tensors = subgraph.Tensors();
  return TensorSet(tensors.begin(), tensors.end());
}

bool ContainsTensor(const LiteRtSubgraphT& subgraph,
                    const LiteRtTensorT& tensor) {
  const auto tensors = subgraph.Tensors();
  return absl::c_find(tensors, &tensor) != tensors.end();
}

std::string Describe(const LiteRtTensorT& tensor) {
  if (tensor.Name().empty()) {
    return "<unnamed tensor>";
  }
  return absl::StrCat("'", tensor.Name(), "'");
}

Unexpected InvalidArgument(std::string message) {
  return Unexpected(kLiteRtStatusErrorInvalidArgument, std::move(message));
}

Unexpected GraphInvariant(std::string message) {
  return Unexpected(kLiteRtStatusErrorInvalidGraphInvariant,
                    std::move(message));
}

// Removes the use (op, input_index) from `tensor`'s use list.
void RemoveUse(LiteRtTensorT& tensor, const LiteRtOpT& op, size_t input_index) {
  for (size_t i = 0; i < tensor.NumUses(); ++i) {
    if (tensor.Users()[i] == &op && tensor.UserArgInds()[i] == input_index) {
      tensor.RemoveUse(i);
      return;
    }
  }
}

// Picks unique names for `tensors` following the previous signature names:
// a tensor that was already part of the signature keeps its name, a tensor
// replacing a removed tensor at the same position inherits that name, other
// tensors get their own name or `<prefix>_<i>`.
std::vector<std::string> SyncNames(const std::vector<std::string>& old_names,
                                   const std::vector<LiteRtTensor>& old_tensors,
                                   const std::vector<LiteRtTensor>& tensors,
                                   absl::string_view prefix) {
  std::vector<std::string> names(tensors.size());
  std::vector<bool> resolved(tensors.size(), false);
  std::vector<bool> old_used(old_tensors.size(), false);
  absl::flat_hash_set<std::string> taken;

  // Tensors that keep their previous name.
  for (size_t i = 0; i < tensors.size(); ++i) {
    for (size_t j = 0; j < old_tensors.size(); ++j) {
      if (!old_used[j] && old_tensors[j] == tensors[i]) {
        if (taken.insert(old_names[j]).second) {
          names[i] = old_names[j];
          resolved[i] = true;
        }
        old_used[j] = true;
        break;
      }
    }
  }
  const absl::flat_hash_set<LiteRtTensor> new_set(tensors.begin(),
                                                  tensors.end());
  auto unique_name = [&taken](std::string base) {
    if (taken.insert(base).second) {
      return base;
    }
    for (int suffix = 1;; ++suffix) {
      std::string candidate = absl::StrCat(base, "_", suffix);
      if (taken.insert(candidate).second) {
        return candidate;
      }
    }
  };
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (resolved[i]) {
      continue;
    }
    if (i < old_tensors.size() && !old_used[i] &&
        !new_set.contains(old_tensors[i])) {
      // Replacement at the same position inherits the name.
      old_used[i] = true;
      names[i] = unique_name(old_names[i]);
      continue;
    }
    const absl::string_view own_name = tensors[i]->Name();
    names[i] = unique_name(own_name.empty() ? absl::StrCat(prefix, "_", i)
                                            : std::string(own_name));
  }
  return names;
}

// Matches the (name, tensor) pairs of a signature against the subgraph I/O
// list `io` and returns the names in `io` order.
Expected<std::vector<std::string>> OrderSignatureNames(
    absl::Span<const NamedTensor> named, absl::Span<const LiteRtTensor> io,
    absl::string_view what) {
  if (named.size() != io.size()) {
    return InvalidArgument(
        absl::StrFormat("signature has %d %ss but the subgraph has %d",
                        named.size(), what, io.size()));
  }
  std::vector<std::string> names(io.size());
  std::vector<bool> used(named.size(), false);
  for (size_t i = 0; i < io.size(); ++i) {
    bool found = false;
    for (size_t j = 0; j < named.size(); ++j) {
      if (!used[j] && named[j].second == io[i]) {
        used[j] = true;
        names[i] = named[j].first;
        found = true;
        break;
      }
    }
    if (!found) {
      return InvalidArgument(absl::StrFormat(
          "subgraph %s %d (%s) is missing from the signature %ss", what, i,
          Describe(*io[i]), what));
    }
  }
  return names;
}

Expected<void> CheckSignatureEntries(const LiteRtSubgraphT& subgraph,
                                     absl::Span<const NamedTensor> named,
                                     absl::string_view what) {
  absl::flat_hash_set<absl::string_view> names;
  for (const auto& [name, tensor] : named) {
    if (name.empty()) {
      return InvalidArgument(
          absl::StrFormat("signature %s names must not be empty", what));
    }
    if (!names.insert(name).second) {
      return InvalidArgument(
          absl::StrFormat("duplicate signature %s name '%s'", what, name));
    }
    if (tensor == nullptr || !ContainsTensor(subgraph, *tensor)) {
      return InvalidArgument(absl::StrFormat(
          "signature %s '%s' does not belong to the subgraph", what, name));
    }
  }
  return {};
}

Expected<void> CheckSubgraphInputs(const LiteRtSubgraphT& subgraph,
                                   absl::Span<LiteRtTensorT* const> inputs) {
  absl::flat_hash_set<const LiteRtTensorT*> seen;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const LiteRtTensorT* tensor = inputs[i];
    if (tensor == nullptr || !ContainsTensor(subgraph, *tensor)) {
      return InvalidArgument(
          absl::StrFormat("input %d does not belong to the subgraph", i));
    }
    if (tensor->DefiningOp() != nullptr) {
      return InvalidArgument(absl::StrFormat(
          "input %d (%s) is produced by an op and cannot be a subgraph input",
          i, Describe(*tensor)));
    }
    if (IsConstantTensor(*tensor)) {
      return InvalidArgument(absl::StrFormat(
          "input %d (%s) is constant and cannot be a subgraph input", i,
          Describe(*tensor)));
    }
    if (!seen.insert(tensor).second) {
      return InvalidArgument(absl::StrFormat(
          "input %d (%s) is listed more than once", i, Describe(*tensor)));
    }
  }
  return {};
}

Expected<void> CheckSubgraphOutputs(const LiteRtSubgraphT& subgraph,
                                    absl::Span<LiteRtTensorT* const> outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i] == nullptr || !ContainsTensor(subgraph, *outputs[i])) {
      return InvalidArgument(
          absl::StrFormat("output %d does not belong to the subgraph", i));
    }
  }
  return {};
}

}  // namespace

//===----------------------------------------------------------------------===//
// Element types.
//===----------------------------------------------------------------------===//

int ElementTypeBitWidth(LiteRtElementType element_type) {
  switch (element_type) {
    case kLiteRtElementTypeInt2:
      return 2;
    case kLiteRtElementTypeInt4:
    case kLiteRtElementTypeUInt4:
      return 4;
    case kLiteRtElementTypeBool:
    case kLiteRtElementTypeInt8:
    case kLiteRtElementTypeUInt8:
    case kLiteRtElementTypeFloat8E4M3FN:
    case kLiteRtElementTypeFloat8E5M2:
      return 8;
    case kLiteRtElementTypeInt16:
    case kLiteRtElementTypeUInt16:
    case kLiteRtElementTypeFloat16:
    case kLiteRtElementTypeBFloat16:
      return 16;
    case kLiteRtElementTypeInt32:
    case kLiteRtElementTypeUInt32:
    case kLiteRtElementTypeFloat32:
      return 32;
    case kLiteRtElementTypeInt64:
    case kLiteRtElementTypeUInt64:
    case kLiteRtElementTypeFloat64:
    case kLiteRtElementTypeComplex64:
      return 64;
    case kLiteRtElementTypeComplex128:
      return 128;
    case kLiteRtElementTypeNone:
    case kLiteRtElementTypeTfResource:
    case kLiteRtElementTypeTfString:
    case kLiteRtElementTypeTfVariant:
      return 0;
  }
  return 0;
}

Expected<std::optional<size_t>> DenseByteSize(LiteRtElementType element_type,
                                              absl::Span<const int32_t> shape) {
  const int bits = ElementTypeBitWidth(element_type);
  if (bits == 0) {
    return std::optional<size_t>();
  }
  if (std::any_of(shape.begin(), shape.end(),
                  [](int32_t dim) { return dim < 0; })) {
    return std::optional<size_t>();
  }
  if (absl::c_find(shape, 0) != shape.end()) {
    return std::optional<size_t>(0);
  }
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  auto overflow = [] {
    return InvalidArgument("the tensor byte size overflows size_t");
  };
  size_t num_elements = 1;
  for (int32_t dim : shape) {
    if (num_elements > kMax / static_cast<size_t>(dim)) {
      return overflow();
    }
    num_elements *= static_cast<size_t>(dim);
  }
  if (num_elements > (kMax - 7) / bits) {
    return overflow();
  }
  return std::optional<size_t>((num_elements * bits + 7) / 8);
}

//===----------------------------------------------------------------------===//
// Queries.
//===----------------------------------------------------------------------===//

Expected<size_t> FindOpIndex(const LiteRtSubgraphT& subgraph,
                             const LiteRtOpT& op) {
  const auto ops = subgraph.Ops();
  const auto it = absl::c_find(ops, &op);
  if (it == ops.end()) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "op does not belong to the subgraph");
  }
  return static_cast<size_t>(it - ops.begin());
}

Expected<size_t> FindTensorIndex(const LiteRtSubgraphT& subgraph,
                                 const LiteRtTensorT& tensor) {
  const auto tensors = subgraph.Tensors();
  const auto it = absl::c_find(tensors, &tensor);
  if (it == tensors.end()) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "tensor does not belong to the subgraph");
  }
  return static_cast<size_t>(it - tensors.begin());
}

Expected<size_t> FindSubgraphIndex(const LiteRtModelT& model,
                                   const LiteRtSubgraphT& subgraph) {
  const auto subgraphs = model.Subgraphs();
  const auto it = absl::c_find(subgraphs, &subgraph);
  if (it == subgraphs.end()) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "subgraph does not belong to the model");
  }
  return static_cast<size_t>(it - subgraphs.begin());
}

LiteRtSubgraphT* FindOwningSubgraph(const LiteRtModelT& model,
                                    const LiteRtOpT& op) {
  for (LiteRtSubgraph subgraph : model.Subgraphs()) {
    if (FindOpIndex(*subgraph, op)) {
      return subgraph;
    }
  }
  return nullptr;
}

bool IsConstantTensor(const LiteRtTensorT& tensor) {
  if (tensor.Weights().Buffer().Size() > 0) {
    return true;
  }
  if (tensor.DefiningOp() != nullptr ||
      tensor.Type().first != kLiteRtRankedTensorType) {
    return false;
  }
  const auto& layout = tensor.Type().second.ranked_tensor_type.layout;
  return layout.rank == 1 && layout.dimensions[0] == 0;
}

//===----------------------------------------------------------------------===//
// Op codes.
//===----------------------------------------------------------------------===//

Expected<int32_t> GetOrAddTflOpCodeIndex(LiteRtModelT& model, LiteRtOpCode code,
                                         absl::string_view custom_code) {
  const bool is_custom = code == kLiteRtOpCodeTflCustom;
  const auto& codes = GetTflOpCodes(model);
  std::optional<int32_t> best;
  int32_t best_version = 0;
  for (size_t i = 0; i < codes.size(); ++i) {
    const TflOpCode& entry = *codes[i];
    const int32_t entry_code =
        std::max<int32_t>(entry.builtin_code, entry.deprecated_builtin_code);
    if (entry_code != code) {
      continue;
    }
    if (is_custom && entry.custom_code != custom_code) {
      continue;
    }
    if (!best.has_value() || entry.version < best_version) {
      best = static_cast<int32_t>(i);
      best_version = entry.version;
    }
  }
  if (best.has_value()) {
    return *best;
  }

  std::vector<TflOpCodePtr> new_codes = TakeTflOpCodes(model);
  auto entry = std::make_unique<TflOpCode>();
  entry->builtin_code = static_cast<tflite::BuiltinOperator>(code);
  entry->deprecated_builtin_code = static_cast<int8_t>(std::min<int32_t>(
      code, tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES));
  if (is_custom) {
    entry->custom_code = std::string(custom_code);
  }
  // TFLite's BuiltinOpResolver registers BROADCAST_TO starting at version 2.
  entry->version = code == kLiteRtOpCodeTflBroadcastTo ? 2 : 1;
  new_codes.push_back(std::move(entry));
  const auto index = static_cast<int32_t>(new_codes.size() - 1);
  SetTflOpCodes(model, std::move(new_codes));
  return index;
}

//===----------------------------------------------------------------------===//
// Builtin options.
//===----------------------------------------------------------------------===//

std::pair<int32_t, int32_t> GetBuiltinOptionsTypes(const LiteRtOpT& op) {
  return {static_cast<int32_t>(GetTflOptions(op).type),
          static_cast<int32_t>(GetTflOptions2(op).type)};
}

OwningBufferRef<uint8_t> PackBuiltinOptions(LiteRtOpT& op) {
  // Temporarily move the unions out of the op to pack them without copying.
  TflOptions options = TakeTflOptions(op);
  TflOptions2 options_2 = TakeTflOptions2(op);

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<void> options_offset;
  flatbuffers::Offset<void> options_2_offset;
  if (options.value != nullptr) {
    options_offset = options.Pack(builder);
  }
  if (options_2.value != nullptr) {
    options_2_offset = options_2.Pack(builder);
  }
  tflite::OperatorBuilder op_builder(builder);
  if (options.type != tflite::BuiltinOptions_NONE) {
    op_builder.add_builtin_options_type(options.type);
    if (!options_offset.IsNull()) {
      op_builder.add_builtin_options(options_offset);
    }
  }
  if (options_2.type != tflite::BuiltinOptions2_NONE) {
    op_builder.add_builtin_options_2_type(options_2.type);
    if (!options_2_offset.IsNull()) {
      op_builder.add_builtin_options_2(options_2_offset);
    }
  }
  builder.Finish(op_builder.Finish());

  SetTflOptions(op, std::move(options));
  SetTflOptions2(op, std::move(options_2));
  // Pass a const pointer: `OwningBufferRef(ByteT*, size_t)` would take
  // ownership of the builder's memory instead of copying it.
  return OwningBufferRef<uint8_t>(
      static_cast<const uint8_t*>(builder.GetBufferPointer()),
      builder.GetSize());
}

Expected<void> UnpackBuiltinOptions(BufferRef<uint8_t> operator_flatbuffer,
                                    LiteRtOpT& op) {
  // Copy into an owned (suitably aligned) buffer before verification.
  OwningBufferRef<uint8_t> buffer(
      static_cast<const uint8_t*>(operator_flatbuffer.Data()),
      operator_flatbuffer.Size());
  flatbuffers::Verifier verifier(buffer.Data(), buffer.Size());
  if (buffer.Size() == 0 ||
      !verifier.VerifyBuffer<tflite::Operator>(/*identifier=*/nullptr)) {
    return Unexpected(kLiteRtStatusErrorInvalidFlatbuffer,
                      "builtin options are not a valid tflite::Operator "
                      "flatbuffer");
  }
  const auto* packed = flatbuffers::GetRoot<tflite::Operator>(buffer.Data());
  std::unique_ptr<tflite::OperatorT> unpacked(packed->UnPack());
  SetTflOptions(op, std::move(unpacked->builtin_options));
  SetTflOptions2(op, std::move(unpacked->builtin_options_2));
  return {};
}

//===----------------------------------------------------------------------===//
// Mutations.
//===----------------------------------------------------------------------===//

Expected<LiteRtOpT*> AddOp(LiteRtModelT& model, LiteRtSubgraphT& subgraph,
                           LiteRtOpCode code,
                           absl::Span<LiteRtTensorT* const> inputs,
                           absl::Span<LiteRtTensorT* const> outputs,
                           const AddOpOptions& options) {
  const size_t num_ops = subgraph.Ops().size();
  const size_t position = options.index.value_or(num_ops);
  if (position > num_ops) {
    return Unexpected(kLiteRtStatusErrorIndexOOB,
                      absl::StrFormat("op index %d is out of range [0, %d]",
                                      position, num_ops));
  }
  const bool is_custom = code == kLiteRtOpCodeTflCustom;
  if (is_custom &&
      (!options.custom_code.has_value() || options.custom_code->empty())) {
    return InvalidArgument("CUSTOM ops require a non-empty custom code");
  }
  if (!is_custom && options.custom_code.has_value()) {
    return InvalidArgument("a custom code is only allowed on CUSTOM ops");
  }

  const TensorSet tensors = GetTensorSet(subgraph);
  const OpPositions positions = GetOpPositions(subgraph);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const LiteRtTensorT* input = inputs[i];
    if (input == nullptr || !tensors.contains(input)) {
      return InvalidArgument(
          absl::StrFormat("input %d does not belong to the subgraph", i));
    }
    if (input->DefiningOp() != nullptr &&
        positions.at(input->DefiningOp()) >= position) {
      return GraphInvariant(absl::StrFormat(
          "input %d (%s) is produced by an op at or after index %d; the op "
          "would break the topological order",
          i, Describe(*input), position));
    }
  }
  absl::flat_hash_set<const LiteRtTensorT*> seen_outputs;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const LiteRtTensorT* output = outputs[i];
    if (output == nullptr || !tensors.contains(output)) {
      return InvalidArgument(
          absl::StrFormat("output %d does not belong to the subgraph", i));
    }
    if (output->DefiningOp() != nullptr) {
      return InvalidArgument(
          absl::StrFormat("output %d (%s) is already produced by another op", i,
                          Describe(*output)));
    }
    if (IsConstantTensor(*output)) {
      return InvalidArgument(
          absl::StrFormat("output %d (%s) is constant", i, Describe(*output)));
    }
    if (FindInput(subgraph, *output)) {
      return InvalidArgument(absl::StrFormat(
          "output %d (%s) is a subgraph input", i, Describe(*output)));
    }
    if (!seen_outputs.insert(output).second) {
      return InvalidArgument(absl::StrFormat(
          "output %d (%s) is listed more than once", i, Describe(*output)));
    }
    if (absl::c_find(inputs, output) != inputs.end()) {
      return InvalidArgument(absl::StrFormat(
          "output %d (%s) is also an input of the op", i, Describe(*output)));
    }
    for (const LiteRtOp user : output->Users()) {
      if (positions.at(user) < position) {
        return GraphInvariant(absl::StrFormat(
            "output %d (%s) is used by an op before index %d; the op would "
            "break the topological order",
            i, Describe(*output), position));
      }
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      const int32_t tfl_code_index,
      GetOrAddTflOpCodeIndex(model, code, options.custom_code.value_or("")));

  LiteRtOpT& op = subgraph.EmplaceOpAt(static_cast<int>(position));
  op.SetOpCode(code);
  op.SetOpIndex(static_cast<uint32_t>(position));
  SetTflOpCodeInd(op, tfl_code_index);
  if (options.custom_code.has_value()) {
    op.SetCustomCode(*options.custom_code);
  }
  if (options.custom_options.has_value() && !options.custom_options->empty()) {
    op.SetCustomOptions(
        static_cast<const uint8_t*>(options.custom_options->data()),
        options.custom_options->size());
  }
  for (LiteRtTensorT* input : inputs) {
    AttachInput(input, op);
  }
  for (LiteRtTensorT* output : outputs) {
    AttachOutput(output, op);
  }
  return &op;
}

Expected<void> SetOpInput(LiteRtSubgraphT& subgraph, LiteRtOpT& op,
                          size_t input_index, LiteRtTensorT& tensor) {
  LITERT_ASSIGN_OR_RETURN(const size_t op_position, FindOpIndex(subgraph, op));
  if (input_index >= op.NumInputs()) {
    return Unexpected(
        kLiteRtStatusErrorIndexOOB,
        absl::StrFormat("input index %d is out of range for an op with %d "
                        "inputs",
                        input_index, op.NumInputs()));
  }
  if (!ContainsTensor(subgraph, tensor)) {
    return InvalidArgument("tensor does not belong to the subgraph");
  }
  LiteRtTensorT& old_tensor = op.Input(input_index);
  if (&old_tensor == &tensor) {
    return {};
  }
  if (const LiteRtOpT* producer = tensor.DefiningOp(); producer != nullptr) {
    LITERT_ASSIGN_OR_RETURN(const size_t producer_position,
                            FindOpIndex(subgraph, *producer));
    if (producer_position >= op_position) {
      return GraphInvariant(absl::StrFormat(
          "tensor %s is produced at op index %d which is not before the "
          "consumer at index %d",
          Describe(tensor), producer_position, op_position));
    }
  }
  RemoveUse(old_tensor, op, input_index);
  op.Inputs()[input_index] = &tensor;
  tensor.Users().push_back(&op);
  tensor.UserArgInds().push_back(input_index);
  return {};
}

Expected<void> SetOpOutput(LiteRtSubgraphT& subgraph, LiteRtOpT& op,
                           size_t output_index, LiteRtTensorT& tensor) {
  LITERT_ASSIGN_OR_RETURN(const size_t op_position, FindOpIndex(subgraph, op));
  if (output_index >= op.NumOutputs()) {
    return Unexpected(
        kLiteRtStatusErrorIndexOOB,
        absl::StrFormat("output index %d is out of range for an op with %d "
                        "outputs",
                        output_index, op.NumOutputs()));
  }
  if (!ContainsTensor(subgraph, tensor)) {
    return InvalidArgument("tensor does not belong to the subgraph");
  }
  LiteRtTensorT& old_tensor = op.Output(output_index);
  if (&old_tensor == &tensor) {
    return {};
  }
  if (tensor.DefiningOp() != nullptr) {
    return InvalidArgument(absl::StrFormat(
        "tensor %s is already produced by another op", Describe(tensor)));
  }
  if (IsConstantTensor(tensor)) {
    return InvalidArgument(
        absl::StrFormat("tensor %s is constant", Describe(tensor)));
  }
  if (FindInput(subgraph, tensor)) {
    return InvalidArgument(
        absl::StrFormat("tensor %s is a subgraph input", Describe(tensor)));
  }
  const OpPositions positions = GetOpPositions(subgraph);
  for (const LiteRtOp user : tensor.Users()) {
    if (positions.at(user) <= op_position) {
      return GraphInvariant(absl::StrFormat(
          "tensor %s is used by an op at or before the producer index %d",
          Describe(tensor), op_position));
    }
  }
  old_tensor.ClearDefiningOp();
  op.Outputs()[output_index] = &tensor;
  tensor.SetDefiningOp(op, output_index);
  return {};
}

Expected<void> RemoveOp(LiteRtModelT& model, LiteRtSubgraphT& subgraph,
                        LiteRtOpT& op) {
  LITERT_RETURN_IF_ERROR(FindOpIndex(subgraph, op));
  model.RemoveAssetFromOp(&op);
  Drop(op);
  subgraph.RemoveOpIf([target = &op](const LiteRtOpT& candidate) {
    return &candidate == target;
  });
  return {};
}

Expected<void> RemoveTensor(LiteRtSubgraphT& subgraph, LiteRtTensorT& tensor) {
  LITERT_RETURN_IF_ERROR(FindTensorIndex(subgraph, tensor));
  if (tensor.NumUses() != 0 || tensor.DefiningOp() != nullptr) {
    return InvalidArgument(absl::StrFormat(
        "tensor %s is still connected to ops", Describe(tensor)));
  }
  if (FindInput(subgraph, tensor) || FindOutput(subgraph, tensor)) {
    return InvalidArgument(absl::StrFormat(
        "tensor %s is a subgraph input or output", Describe(tensor)));
  }
  for (const LiteRtTensorT* other : subgraph.Tensors()) {
    if (other->Qparams().first != kLiteRtQuantizationBlockWise) {
      continue;
    }
    const auto& block_wise = other->Qparams().second.block_wise;
    if (block_wise.scales == &tensor || block_wise.zero_points == &tensor) {
      return InvalidArgument(absl::StrFormat(
          "tensor %s holds block-wise quantization parameters of %s",
          Describe(tensor), Describe(*other)));
    }
  }
  subgraph.RemoveTensorIf([target = &tensor](const LiteRtTensorT& candidate) {
    return &candidate == target;
  });
  return {};
}

Expected<void> ReplaceAllUses(
    LiteRtModelT& model, LiteRtSubgraphT& subgraph, LiteRtTensorT& old_tensor,
    LiteRtTensorT& new_tensor,
    const absl::flat_hash_set<const LiteRtOpT*>& exclude) {
  if (!ContainsTensor(subgraph, old_tensor) ||
      !ContainsTensor(subgraph, new_tensor)) {
    return InvalidArgument("tensors must belong to the subgraph");
  }
  if (&old_tensor == &new_tensor) {
    return InvalidArgument("cannot replace a tensor with itself");
  }
  std::vector<std::pair<LiteRtOp, LiteRtParamIndex>> uses;
  for (size_t i = 0; i < old_tensor.NumUses(); ++i) {
    auto use = old_tensor.GetUse(i);
    if (!exclude.contains(use.first)) {
      uses.push_back(use);
    }
  }
  if (const LiteRtOpT* producer = new_tensor.DefiningOp(); producer) {
    const OpPositions positions = GetOpPositions(subgraph);
    const size_t producer_position = positions.at(producer);
    for (const auto& [user, unused_index] : uses) {
      if (positions.at(user) <= producer_position) {
        return GraphInvariant(absl::StrFormat(
            "replacement %s is produced at op index %d which is not before "
            "the user at index %d",
            Describe(new_tensor), producer_position, positions.at(user)));
      }
    }
  }
  for (const auto& [user, input_index] : uses) {
    RemoveUse(old_tensor, *user, input_index);
    user->Inputs()[input_index] = &new_tensor;
    new_tensor.Users().push_back(user);
    new_tensor.UserArgInds().push_back(input_index);
  }
  for (LiteRtTensor& output : subgraph.Outputs()) {
    if (output == &old_tensor) {
      output = &new_tensor;
    }
  }
  for (LiteRtSignature signature : model.Signatures()) {
    if (&signature->GetSubgraph() == &subgraph) {
      signature->RemapOutputTensor(&old_tensor, &new_tensor);
    }
  }
  return {};
}

Expected<void> SetSubgraphInputs(LiteRtModelT& model, LiteRtSubgraphT& subgraph,
                                 absl::Span<LiteRtTensorT* const> inputs) {
  LITERT_RETURN_IF_ERROR(CheckSubgraphInputs(subgraph, inputs));
  subgraph.Inputs().assign(inputs.begin(), inputs.end());
  SyncSignatures(model, subgraph);
  return {};
}

Expected<void> SetSubgraphOutputs(LiteRtModelT& model,
                                  LiteRtSubgraphT& subgraph,
                                  absl::Span<LiteRtTensorT* const> outputs) {
  LITERT_RETURN_IF_ERROR(CheckSubgraphOutputs(subgraph, outputs));
  subgraph.Outputs().assign(outputs.begin(), outputs.end());
  SyncSignatures(model, subgraph);
  return {};
}

bool EliminateDeadCode(LiteRtModelT& model, LiteRtSubgraphT& subgraph) {
  bool changed = false;
  while (true) {
    // Disconnect ops whose results are all unused. Visiting ops in reverse
    // order lets producers become dead in the same sweep.
    const auto ops = subgraph.Ops();
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      LiteRtOpT& op = **it;
      if (op.Outputs().empty()) {
        continue;
      }
      const bool all_unused = std::all_of(
          op.Outputs().begin(), op.Outputs().end(),
          [&subgraph](const LiteRtTensorT* output) {
            return output->NumUses() == 0 && !FindOutput(subgraph, *output);
          });
      if (all_unused) {
        MarkDead(op);
      }
    }
    // `DCE` deallocates ops without edges or subgraph references; detach their
    // assets first.
    for (LiteRtOp op : subgraph.Ops()) {
      if (op->Inputs().empty() && op->Outputs().empty() &&
          !HasSubgraphRefs(*op)) {
        model.RemoveAssetFromOp(op);
      }
    }
    if (!DCE(subgraph)) {
      break;
    }
    changed = true;
  }
  return changed;
}

void SyncSignatures(LiteRtModelT& model, LiteRtSubgraphT& subgraph) {
  for (LiteRtSignature signature : model.Signatures()) {
    if (&signature->GetSubgraph() != &subgraph) {
      continue;
    }
    std::vector<LiteRtTensor> old_inputs;
    old_inputs.reserve(signature->InputNames().size());
    for (size_t i = 0; i < signature->InputNames().size(); ++i) {
      old_inputs.push_back(signature->GetInputTensor(i));
    }
    std::vector<LiteRtTensor> old_outputs;
    old_outputs.reserve(signature->OutputNames().size());
    for (size_t i = 0; i < signature->OutputNames().size(); ++i) {
      old_outputs.push_back(signature->GetOutputTensor(i));
    }
    std::vector<std::string> input_names = SyncNames(
        signature->InputNames(), old_inputs, subgraph.Inputs(), "input");
    std::vector<std::string> output_names = SyncNames(
        signature->OutputNames(), old_outputs, subgraph.Outputs(), "output");
    std::string key(signature->Key());
    *signature = LiteRtSignatureT(&subgraph, std::move(input_names),
                                  subgraph.Inputs(), std::move(output_names),
                                  subgraph.Outputs(), std::move(key));
  }
}

Expected<LiteRtSignatureT*> AddSignature(
    LiteRtModelT& model, absl::string_view key, LiteRtSubgraphT& subgraph,
    absl::Span<const NamedTensor> inputs,
    absl::Span<const NamedTensor> outputs) {
  if (key.empty()) {
    return InvalidArgument("signature key must not be empty");
  }
  if (model.FindSignature(key)) {
    return Unexpected(kLiteRtStatusErrorAlreadyExists,
                      absl::StrFormat("signature '%s' already exists", key));
  }
  LITERT_RETURN_IF_ERROR(FindSubgraphIndex(model, subgraph));
  LITERT_RETURN_IF_ERROR(CheckSignatureEntries(subgraph, inputs, "input"));
  LITERT_RETURN_IF_ERROR(CheckSignatureEntries(subgraph, outputs, "output"));

  std::vector<LiteRtTensorT*> input_tensors;
  for (const auto& entry : inputs) input_tensors.push_back(entry.second);
  std::vector<LiteRtTensorT*> output_tensors;
  for (const auto& entry : outputs) output_tensors.push_back(entry.second);

  // Validate everything before mutating the subgraph I/O.
  const bool set_inputs = subgraph.Inputs().empty() && !inputs.empty();
  const bool set_outputs = subgraph.Outputs().empty() && !outputs.empty();
  if (set_inputs) {
    LITERT_RETURN_IF_ERROR(CheckSubgraphInputs(subgraph, input_tensors));
  }
  if (set_outputs) {
    LITERT_RETURN_IF_ERROR(CheckSubgraphOutputs(subgraph, output_tensors));
  }
  const std::vector<LiteRtTensor> final_inputs =
      set_inputs ? std::vector<LiteRtTensor>(input_tensors.begin(),
                                             input_tensors.end())
                 : subgraph.Inputs();
  const std::vector<LiteRtTensor> final_outputs =
      set_outputs ? std::vector<LiteRtTensor>(output_tensors.begin(),
                                              output_tensors.end())
                  : subgraph.Outputs();
  LITERT_ASSIGN_OR_RETURN(std::vector<std::string> input_names,
                          OrderSignatureNames(inputs, final_inputs, "input"));
  LITERT_ASSIGN_OR_RETURN(
      std::vector<std::string> output_names,
      OrderSignatureNames(outputs, final_outputs, "output"));

  if (set_inputs) {
    LITERT_RETURN_IF_ERROR(SetSubgraphInputs(model, subgraph, input_tensors));
  }
  if (set_outputs) {
    LITERT_RETURN_IF_ERROR(SetSubgraphOutputs(model, subgraph, output_tensors));
  }
  return &model.EmplaceSignature(&subgraph, std::move(input_names),
                                 subgraph.Inputs(), std::move(output_names),
                                 subgraph.Outputs(), std::string(key));
}

Expected<const LiteRtSignatureT*> RemoveSignature(LiteRtModelT& model,
                                                  absl::string_view key) {
  LITERT_ASSIGN_OR_RETURN(auto signature, model.FindSignature(key));
  const LiteRtSignatureT* removed = &signature.get();
  model.RemoveSignaturesIf([removed](const LiteRtSignatureT& candidate) {
    return &candidate == removed;
  });
  return removed;
}

Expected<size_t> PruneUnreachableSubgraphs(LiteRtModelT& model) {
  const bool added_temp_signature = model.Signatures().empty();
  if (added_temp_signature) {
    if (model.Subgraphs().empty()) {
      return size_t{0};
    }
    model.EmplaceSignature(
        model.MainSubgraph(), std::vector<std::string>{},
        std::vector<LiteRtTensor>{}, std::vector<std::string>{},
        std::vector<LiteRtTensor>{}, std::string(kDefaultSignatureKey));
  }
  const size_t before = model.NumSubgraphs();
  std::vector<std::string> key_storage;
  key_storage.reserve(model.Signatures().size());
  for (LiteRtSignature signature : model.Signatures()) {
    key_storage.emplace_back(signature->Key());
  }
  std::vector<absl::string_view> keys(key_storage.begin(), key_storage.end());

  Expected<void> status = PruneModelToSignatures(model, keys);
  if (added_temp_signature) {
    model.RemoveSignaturesIf([](const LiteRtSignatureT&) { return true; });
  }
  if (!status) {
    if (static_cast<LiteRtStatus>(status.Error().StatusValue()) ==
        kLiteRtStatusErrorUnsupported) {
      return size_t{0};
    }
    return status.Error();
  }
  return before - model.NumSubgraphs();
}

//===----------------------------------------------------------------------===//
// Validation.
//===----------------------------------------------------------------------===//

Expected<void> ValidateModel(const LiteRtModelT& model, ValidationLevel level) {
  const bool full = level == ValidationLevel::kFull;
  std::vector<std::string> errors;
  const auto& op_codes = GetTflOpCodes(model);
  const auto subgraphs = model.Subgraphs();
  for (size_t s = 0; s < subgraphs.size(); ++s) {
    const LiteRtSubgraphT& subgraph = *subgraphs[s];
    const TensorSet tensors = GetTensorSet(subgraph);
    const OpPositions positions = GetOpPositions(subgraph);
    absl::flat_hash_map<const LiteRtTensorT*, size_t> tensor_indices;
    for (size_t t = 0; t < subgraph.Tensors().size(); ++t) {
      tensor_indices.emplace(subgraph.Tensors()[t], t);
    }
    auto label = [&](const LiteRtTensorT* tensor) {
      const auto it = tensor_indices.find(tensor);
      return absl::StrFormat(
          "tensor %s (index %s)", Describe(*tensor),
          it == tensor_indices.end() ? "?" : absl::StrCat(it->second));
    };
    auto add_error = [&](std::string message) {
      errors.push_back(absl::StrFormat("subgraph %d: %s", s, message));
    };
    if (tensors.size() != subgraph.Tensors().size() ||
        positions.size() != subgraph.Ops().size()) {
      add_error("duplicate tensor or op entries");
    }
    const absl::flat_hash_set<const LiteRtTensorT*> inputs(
        subgraph.Inputs().begin(), subgraph.Inputs().end());

    // Ops: membership, local topology, op codes and topological order.
    for (size_t p = 0; p < subgraph.Ops().size(); ++p) {
      const LiteRtOpT& op = *subgraph.Ops()[p];
      bool edges_ok = true;
      for (size_t i = 0; i < op.NumInputs(); ++i) {
        const LiteRtTensorT* input = op.Inputs()[i];
        if (input == nullptr || !tensors.contains(input)) {
          add_error(absl::StrFormat(
              "op %d input %d does not belong to the subgraph", p, i));
          edges_ok = false;
          continue;
        }
        if (const LiteRtOpT* producer = input->DefiningOp(); producer) {
          const auto it = positions.find(producer);
          if (it == positions.end()) {
            add_error(absl::StrFormat(
                "op %d input %d (%s) is produced by an op outside the "
                "subgraph",
                p, i, label(input)));
          } else if (full && it->second >= p) {
            add_error(absl::StrFormat(
                "op %d input %d (%s) is produced by op %d; ops are not in "
                "topological order",
                p, i, label(input), it->second));
          }
        } else if (full && !IsConstantTensor(*input) &&
                   !inputs.contains(input)) {
          add_error(absl::StrFormat(
              "op %d input %d (%s) is neither constant, a subgraph input nor "
              "produced by an op",
              p, i, label(input)));
        }
      }
      for (size_t i = 0; i < op.NumOutputs(); ++i) {
        const LiteRtTensorT* output = op.Outputs()[i];
        if (output == nullptr || !tensors.contains(output)) {
          add_error(absl::StrFormat(
              "op %d output %d does not belong to the subgraph", p, i));
          edges_ok = false;
        }
      }
      if (edges_ok && !internal::ValidateLocalTopology(op)) {
        add_error(absl::StrFormat(
            "op %d has inconsistent def-use links with its tensors", p));
      }
      const int32_t code_index = GetTflOpCodeInd(op);
      if (code_index == kDispatchOpCodeTflInd) {
        if (op.OpCode() != kLiteRtOpCodeTflCustom) {
          add_error(
              absl::StrFormat("op %d has no operator code table entry", p));
        }
      } else if (code_index < 0 ||
                 static_cast<size_t>(code_index) >= op_codes.size()) {
        add_error(absl::StrFormat(
            "op %d refers to operator code %d but the table has %d entries", p,
            code_index, op_codes.size()));
      } else {
        const TflOpCode& entry = *op_codes[code_index];
        const int32_t entry_code = std::max<int32_t>(
            entry.builtin_code, entry.deprecated_builtin_code);
        const auto custom_code = op.CustomCode();
        if (entry_code != op.OpCode()) {
          add_error(absl::StrFormat(
              "op %d has op code %d but its operator code entry is %d", p,
              op.OpCode(), entry_code));
        } else if (custom_code.HasValue() &&
                   *custom_code != entry.custom_code) {
          add_error(absl::StrFormat(
              "op %d has custom code '%s' but its operator code entry has "
              "'%s'",
              p, *custom_code, entry.custom_code));
        }
      }
    }

    // Tensors: use lists, defining ops, types and quantization.
    for (const LiteRtTensorT* tensor : subgraph.Tensors()) {
      for (size_t u = 0; u < tensor->NumUses(); ++u) {
        const auto [user, index] = tensor->GetUse(u);
        if (!positions.contains(user)) {
          add_error(absl::StrFormat("%s is used by an op outside the subgraph",
                                    label(tensor)));
        } else if (index >= user->NumInputs() ||
                   user->Inputs()[index] != tensor) {
          add_error(absl::StrFormat(
              "%s has a use (op %d, input %d) that the op does not have",
              label(tensor), positions.at(user), index));
        }
      }
      if (const LiteRtOpT* producer = tensor->DefiningOp(); producer) {
        if (!positions.contains(producer)) {
          add_error(absl::StrFormat(
              "%s is produced by an op outside the subgraph", label(tensor)));
        } else if (tensor->DefiningOpOutInd() >= producer->NumOutputs() ||
                   producer->Outputs()[tensor->DefiningOpOutInd()] != tensor) {
          add_error(absl::StrFormat(
              "%s has a defining op that does not list it as output",
              label(tensor)));
        }
        if (full && tensor->Weights().Buffer().Size() > 0) {
          add_error(absl::StrFormat(
              "%s has constant data but is produced by an op", label(tensor)));
        }
      }
      if (full && tensor->Type().first == kLiteRtRankedTensorType &&
          tensor->Type().second.ranked_tensor_type.layout.rank >
              LITERT_TENSOR_MAX_RANK) {
        add_error(absl::StrFormat("%s has a rank above %d", label(tensor),
                                  LITERT_TENSOR_MAX_RANK));
      }
      if (tensor->Qparams().first == kLiteRtQuantizationBlockWise) {
        const auto& block_wise = tensor->Qparams().second.block_wise;
        if (block_wise.scales == nullptr ||
            !tensors.contains(block_wise.scales) ||
            (block_wise.zero_points != nullptr &&
             !tensors.contains(block_wise.zero_points))) {
          add_error(absl::StrFormat(
              "%s has block-wise quantization tensors outside the subgraph",
              label(tensor)));
        }
      }
    }

    // Subgraph I/O.
    if (full && inputs.size() != subgraph.Inputs().size()) {
      add_error("subgraph inputs contain duplicates");
    }
    for (size_t i = 0; i < subgraph.Inputs().size(); ++i) {
      const LiteRtTensorT* input = subgraph.Inputs()[i];
      if (input == nullptr || !tensors.contains(input)) {
        add_error(absl::StrFormat(
            "subgraph input %d does not belong to the subgraph", i));
      } else if (full && input->DefiningOp() != nullptr) {
        add_error(absl::StrFormat("subgraph input %d (%s) is produced by an op",
                                  i, label(input)));
      }
    }
    for (size_t i = 0; i < subgraph.Outputs().size(); ++i) {
      const LiteRtTensorT* output = subgraph.Outputs()[i];
      if (output == nullptr || !tensors.contains(output)) {
        add_error(absl::StrFormat(
            "subgraph output %d does not belong to the subgraph", i));
      } else if (full && output->DefiningOp() == nullptr &&
                 !IsConstantTensor(*output) && !inputs.contains(output)) {
        add_error(absl::StrFormat(
            "subgraph output %d (%s) is neither constant, a subgraph input "
            "nor produced by an op",
            i, label(output)));
      }
    }
  }

  // Signatures.
  absl::flat_hash_set<absl::string_view> keys;
  for (const LiteRtSignature signature : model.Signatures()) {
    const std::string where =
        absl::StrFormat("signature '%s'", signature->Key());
    if (full && !keys.insert(signature->Key()).second) {
      errors.push_back(absl::StrCat(where, ": duplicate signature key"));
    }
    const LiteRtSubgraphT& subgraph = signature->GetSubgraph();
    if (!FindSubgraphIndex(model, subgraph)) {
      errors.push_back(
          absl::StrCat(where, ": refers to a subgraph outside the model"));
      continue;
    }
    auto check_io = [&](const std::vector<std::string>& names,
                        absl::Span<const LiteRtTensor> io, bool is_input) {
      const char* what = is_input ? "input" : "output";
      if (names.size() != io.size()) {
        errors.push_back(
            absl::StrFormat("%s: has %d %ss but its subgraph has %d", where,
                            names.size(), what, io.size()));
        return;
      }
      for (size_t i = 0; i < names.size(); ++i) {
        const LiteRtTensor tensor = is_input ? signature->GetInputTensor(i)
                                             : signature->GetOutputTensor(i);
        if (tensor != io[i]) {
          errors.push_back(
              absl::StrFormat("%s: %s '%s' does not match subgraph %s %d",
                              where, what, names[i], what, i));
        }
      }
    };
    check_io(signature->InputNames(), subgraph.Inputs(), /*is_input=*/true);
    check_io(signature->OutputNames(), subgraph.Outputs(), /*is_input=*/false);
  }

  if (!errors.empty()) {
    return GraphInvariant(absl::StrJoin(errors, "\n"));
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Dispatch ops and assets.
//===----------------------------------------------------------------------===//

Expected<LiteRtOpT*> OutlineSubgraphToDispatchOp(LiteRtModelT& model,
                                                 LiteRtSubgraphT& subgraph) {
  LITERT_RETURN_IF_ERROR(FindSubgraphIndex(model, subgraph));
  const TensorSet tensors = GetTensorSet(subgraph);
  const std::vector<LiteRtTensor> old_outputs = subgraph.Outputs();
  for (size_t i = 0; i < old_outputs.size(); ++i) {
    if (old_outputs[i] == nullptr || !tensors.contains(old_outputs[i])) {
      return InvalidArgument(absl::StrFormat(
          "subgraph output %d does not belong to the subgraph", i));
    }
  }
  // Signature outputs are remapped by position below. The loader and
  // `AddSignature` keep them in subgraph output order; reject anything else
  // before changing the model.
  for (const LiteRtSignature signature : model.Signatures()) {
    if (&signature->GetSubgraph() != &subgraph) {
      continue;
    }
    bool positional = signature->OutputNames().size() == old_outputs.size();
    for (size_t i = 0; positional && i < old_outputs.size(); ++i) {
      positional = signature->GetOutputTensor(i) == old_outputs[i];
    }
    if (!positional) {
      return GraphInvariant(absl::StrFormat(
          "signature '%s' outputs are out of sync with the subgraph outputs",
          signature->Key()));
    }
  }

  // Every output position becomes a distinct output of the dispatch op. The
  // first occurrence of a tensor produced by an op is reused; outputs that are
  // not produced by an op (constants, forwarded subgraph inputs) and repeated
  // occurrences get a fresh tensor with the same name, type and quantization.
  absl::flat_hash_set<const LiteRtTensorT*> claimed;
  std::vector<LiteRtTensor> new_outputs;
  new_outputs.reserve(old_outputs.size());
  for (LiteRtTensor output : old_outputs) {
    if (output->DefiningOp() != nullptr && claimed.insert(output).second) {
      new_outputs.push_back(output);
      continue;
    }
    LiteRtTensorT& fresh = subgraph.EmplaceTensor();
    fresh.SetName(std::string(output->Name()));
    fresh.SetType(output->Type());
    if (output->Qparams().first == kLiteRtQuantizationPerChannel) {
      const auto& per_channel = output->Qparams().second.per_channel;
      const std::vector<float> scales(
          per_channel.scales, per_channel.scales + per_channel.num_channels);
      const std::vector<int64_t> zero_points(
          per_channel.zero_points,
          per_channel.zero_points + per_channel.num_channels);
      fresh.SetQarams(MakePerChannelQuantization(
          scales, zero_points, per_channel.quantized_dimension, fresh));
    } else {
      fresh.SetQarams(output->Qparams());
    }
    new_outputs.push_back(&fresh);
  }

  for (LiteRtOp op : subgraph.Ops()) {
    model.RemoveAssetFromOp(op);
    MarkDead(*op);
  }
  LiteRtOpT& dispatch_op = subgraph.EmplaceOp();
  MakeDispatchOp(dispatch_op);
  dispatch_op.SetCustomCode(std::string(kDispatchOpCustomCode));
  dispatch_op.SetOpIndex(0);
  for (LiteRtTensor input : subgraph.Inputs()) {
    AttachInput(input, dispatch_op);
  }
  for (LiteRtTensor output : new_outputs) {
    AttachOutput(output, dispatch_op);
  }

  subgraph.Outputs() = new_outputs;
  for (LiteRtSignature signature : model.Signatures()) {
    if (&signature->GetSubgraph() != &subgraph) {
      continue;
    }
    std::vector<LiteRtTensor> inputs;
    inputs.reserve(signature->InputNames().size());
    for (size_t i = 0; i < signature->InputNames().size(); ++i) {
      inputs.push_back(signature->GetInputTensor(i));
    }
    std::vector<std::string> input_names = signature->InputNames();
    std::vector<std::string> output_names = signature->OutputNames();
    std::string key(signature->Key());
    if (key == kDefaultSignatureKey) {
      // The loader's placeholder signature names outputs after their tensors,
      // so a repeated output repeats its name. Now that every position has its
      // own tensor, rename repeats to "<name>_<position>".
      absl::flat_hash_set<std::string> taken;
      for (size_t i = 0; i < output_names.size(); ++i) {
        if (taken.insert(output_names[i]).second) continue;
        std::string candidate = absl::StrCat(output_names[i], "_", i);
        for (int suffix = 1; !taken.insert(candidate).second; ++suffix) {
          candidate = absl::StrCat(output_names[i], "_", i, "_", suffix);
        }
        output_names[i] = std::move(candidate);
      }
    }
    *signature =
        LiteRtSignatureT(&subgraph, std::move(input_names), std::move(inputs),
                         std::move(output_names), new_outputs, std::move(key));
  }

  while (DCE(subgraph)) {
  }
  return &dispatch_op;
}

bool IsDispatchOp(const LiteRtOpT& op) {
  if (op.OpCode() != kLiteRtOpCodeTflCustom) {
    return false;
  }
  if (GetTflOpCodeInd(op) == kDispatchOpCodeTflInd) {
    return true;
  }
  const auto custom_code = op.CustomCode();
  return custom_code.HasValue() && *custom_code == kDispatchOpCustomCode;
}

Expected<void> AttachAsset(LiteRtModelT& model, LiteRtOpT& op,
                           uint32_t buffer_id, absl::string_view name) {
  if (FindOwningSubgraph(model, op) == nullptr) {
    return InvalidArgument("op does not belong to the model");
  }
  if (!IsDispatchOp(op)) {
    return InvalidArgument("assets can only be attached to dispatch ops");
  }
  if (buffer_id == 0 || buffer_id >= model.Buffers()->NumBuffers()) {
    return InvalidArgument(absl::StrFormat("unknown buffer id %d", buffer_id));
  }
  if (name.empty()) {
    return InvalidArgument("asset name must not be empty");
  }
  model.AttachAssetToOp(&op, buffer_id, std::string(name));
  return {};
}

Expected<internal::DispatchOpOptions> ReadDispatchOpOptions(
    const LiteRtOpT& op) {
  const BufferRef<uint8_t> options = op.CustomOptions();
  if (!IsDispatchOp(op) || options.Size() == 0) {
    return Unexpected(kLiteRtStatusErrorNotFound, "op has no dispatch options");
  }
  if (!flexbuffers::VerifyBuffer(options.Data(), options.Size())) {
    return Unexpected(kLiteRtStatusErrorInvalidFlatbuffer,
                      "dispatch options are not a valid flexbuffer");
  }
  const auto root = flexbuffers::GetRoot(options.Data(), options.Size());
  if (!root.IsMap()) {
    return Unexpected(kLiteRtStatusErrorInvalidFlatbuffer,
                      "dispatch options are not a flexbuffer map");
  }
  const auto map = root.AsMap();
  if (!map["bytecode_size"].IsUInt() || !map["bytecode_offset"].IsUInt() ||
      !map["name"].IsString()) {
    return Unexpected(kLiteRtStatusErrorInvalidFlatbuffer,
                      "dispatch options miss bytecode_size, bytecode_offset "
                      "or name");
  }
  return internal::GetDispatchOpOptions(options);
}

}  // namespace litert::litert_model
