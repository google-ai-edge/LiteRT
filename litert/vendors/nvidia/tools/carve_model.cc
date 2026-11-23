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

#include "litert/vendors/nvidia/tools/carve_model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/build_stamp.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/vendors/nvidia/tools/carve_pattern.h"
#include "tflite/schema/schema_generated.h"

namespace litert::tools {
namespace {

// The constant data keeps its offset within a page when the flatbuffer in
// front of it changes size.
constexpr uint64_t kDataAlignment = 4096;

// Zeros behind a flatbuffer that holds its constants: SIMD kernels read past
// the end of the last one (LiteRT's serializer pads the same way).
constexpr size_t kTailPadding = 128;

// A buffer or an operator's large custom options live behind the flatbuffer
// when their offset is above 1 (the schema reserves 0 and 1).
bool IsAppended(uint64_t offset) { return offset > 1; }

// Outlines the regions of one subgraph.
class Outliner {
 public:
  Outliner(tflite::ModelT& model, size_t subgraph_index)
      : model_(model),
        subgraph_index_(subgraph_index),
        subgraph_(*model.subgraphs[subgraph_index]),
        replaced_(subgraph_.operators.size(), false),
        removed_(subgraph_.operators.size(), false),
        region_tensors_(subgraph_.tensors.size(), false) {
    for (const auto& subgraph : model.subgraphs) names_.insert(subgraph->name);
  }

  absl::Status Outline(const CarveRegionSpec& region) {
    if (auto status = Validate(region); !status.ok()) return status;

    auto decomposition = std::make_unique<tflite::SubGraphT>();
    decomposition->name = UniqueName(
        absl::StrCat(subgraph_.name, "_", region.composite_name, ".impl"));
    // The copy of each of the subgraph's tensors in the decomposition.
    absl::flat_hash_map<int32_t, int32_t> copies;

    // The inputs come first, as in the decompositions the converter writes. A
    // constant input is no constant here: the composite's input brings its
    // data.
    for (size_t i = 0; i < region.inputs.size(); ++i) {
      const int32_t copy = AddCopy(region.inputs[i], *decomposition);
      auto& tensor = *decomposition->tensors[copy];
      tensor.name = absl::StrCat(decomposition->name, "_arg", i);
      tensor.buffer = 0;
      tensor.external_buffer = 0;
      decomposition->inputs.push_back(copy);
      copies.emplace(region.inputs[i], copy);
    }
    for (const int32_t input : decomposition->inputs) {
      CopyQuantizationTensors(input, *decomposition, copies);
    }

    for (const int32_t op_index : region.ops) {
      auto op =
          std::make_unique<tflite::OperatorT>(*subgraph_.operators[op_index]);
      for (int32_t& input : op->inputs) {
        if (input < 0) continue;
        region_tensors_[input] = true;
        input = CopyOf(input, *decomposition, copies);
      }
      for (int32_t& output : op->outputs) {
        region_tensors_[output] = true;
        const int32_t copy = AddCopy(output, *decomposition);
        copies[output] = copy;
        CopyQuantizationTensors(copy, *decomposition, copies);
        output = copy;
      }
      decomposition->operators.push_back(std::move(op));
    }
    for (const int32_t output : region.outputs) {
      decomposition->outputs.push_back(copies.at(output));
    }

    auto composite = std::make_unique<tflite::OperatorT>();
    composite->opcode_index = CompositeOpCode();
    composite->inputs = region.inputs;
    composite->outputs = region.outputs;
    tflite::StableHLOCompositeOptionsT options;
    options.name = region.composite_name;
    options.decomposition_subgraph_index =
        static_cast<int32_t>(model_.subgraphs.size());
    options.composite_attributes = region.attributes;
    composite->builtin_options_2.Set(std::move(options));

    for (const int32_t op_index : region.ops) {
      replaced_[op_index] = true;
      removed_[op_index] = op_index != region.ops.back();
    }
    subgraph_.operators[region.ops.back()] = std::move(composite);
    model_.subgraphs.push_back(std::move(decomposition));
    return absl::OkStatus();
  }

  // Drops the operators the composites replaced and the tensors only they
  // used, and renumbers what refers to the tensors that remain.
  void Finish() {
    std::vector<std::unique_ptr<tflite::OperatorT>> operators;
    for (size_t i = 0; i < subgraph_.operators.size(); ++i) {
      if (!removed_[i]) operators.push_back(std::move(subgraph_.operators[i]));
    }
    subgraph_.operators = std::move(operators);

    auto& tensors = subgraph_.tensors;
    // A tensor leaves only if the region was all that used it: what the
    // subgraph had unused before stays as it was.
    std::vector<bool> candidate = region_tensors_;
    for (bool changed = true; changed;) {
      changed = false;
      for (size_t t = 0; t < tensors.size(); ++t) {
        if (!candidate[t]) continue;
        for (const int32_t* ref : QuantizationTensors(*tensors[t])) {
          if (*ref >= 0 && !candidate[*ref]) {
            candidate[*ref] = true;
            changed = true;
          }
        }
      }
    }
    std::vector<bool> used(tensors.size(), false);
    auto use = [&](const std::vector<int32_t>& indices) {
      for (const int32_t index : indices) {
        if (index >= 0) used[index] = true;
      }
    };
    for (const auto& op : subgraph_.operators) {
      use(op->inputs);
      use(op->outputs);
      use(op->intermediates);
    }
    use(subgraph_.inputs);
    use(subgraph_.outputs);
    for (bool changed = true; changed;) {
      changed = false;
      for (size_t t = 0; t < tensors.size(); ++t) {
        if (candidate[t] && !used[t]) continue;
        for (const int32_t* ref : QuantizationTensors(*tensors[t])) {
          if (*ref >= 0 && !used[*ref]) {
            used[*ref] = true;
            changed = true;
          }
        }
      }
    }

    std::vector<int32_t> renumbered(tensors.size(), -1);
    std::vector<std::unique_ptr<tflite::TensorT>> kept;
    for (size_t t = 0; t < tensors.size(); ++t) {
      if (candidate[t] && !used[t]) continue;
      renumbered[t] = static_cast<int32_t>(kept.size());
      kept.push_back(std::move(tensors[t]));
    }
    tensors = std::move(kept);
    auto renumber = [&](std::vector<int32_t>& indices) {
      for (int32_t& index : indices) {
        if (index >= 0) index = renumbered[index];
      }
    };
    for (auto& op : subgraph_.operators) {
      renumber(op->inputs);
      renumber(op->outputs);
      renumber(op->intermediates);
    }
    renumber(subgraph_.inputs);
    renumber(subgraph_.outputs);
    for (auto& tensor : tensors) {
      for (int32_t* ref : QuantizationTensors(*tensor)) {
        if (*ref >= 0) *ref = renumbered[*ref];
      }
    }
    for (auto& signature : model_.signature_defs) {
      if (signature->subgraph_index != subgraph_index_) continue;
      for (auto& input : signature->inputs) {
        input->tensor_index = renumbered[input->tensor_index];
      }
      for (auto& output : signature->outputs) {
        output->tensor_index = renumbered[output->tensor_index];
      }
    }
  }

 private:
  absl::Status Validate(const CarveRegionSpec& region) const {
    auto error = [&](absl::string_view message) {
      return absl::InvalidArgumentError(
          absl::StrCat("region ", region.composite_name, " of subgraph ",
                       subgraph_.name, ": ", message));
    };
    const auto num_tensors = static_cast<int64_t>(subgraph_.tensors.size());
    absl::flat_hash_set<int32_t> produced;
    if (region.ops.empty() || region.outputs.empty()) {
      return error("no operators or no outputs");
    }
    for (size_t i = 0; i < region.ops.size(); ++i) {
      const int32_t op_index = region.ops[i];
      if (op_index < 0 ||
          static_cast<size_t>(op_index) >= subgraph_.operators.size() ||
          (i > 0 && op_index <= region.ops[i - 1])) {
        return error("the operators are not ascending indices");
      }
      if (replaced_[op_index]) return error("overlaps another region");
      const auto& op = *subgraph_.operators[op_index];
      // LiteRT's loader refuses a model with intermediates, so a region
      // cannot hold one.
      if (!op.intermediates.empty()) return error("intermediate tensors");
      for (const int32_t input : op.inputs) {
        if (input >= num_tensors) return error("a tensor index out of range");
      }
      for (const int32_t output : op.outputs) {
        if (output < 0 || output >= num_tensors) {
          return error("a tensor index out of range");
        }
        produced.insert(output);
      }
    }
    for (const int32_t input : region.inputs) {
      if (input < 0 || input >= num_tensors) {
        return error("an input index out of range");
      }
      if (produced.contains(input)) {
        return error("an input is computed inside the region");
      }
    }
    for (const int32_t output : region.outputs) {
      if (!produced.contains(output)) {
        return error("an output is not computed inside the region");
      }
    }
    for (const auto& tensor : subgraph_.tensors) {
      for (const int32_t* ref : QuantizationTensors(*tensor)) {
        if (*ref < -1 || *ref >= num_tensors) {
          return error("a quantization tensor index out of range");
        }
      }
    }
    return absl::OkStatus();
  }

  int32_t AddCopy(int32_t index, tflite::SubGraphT& decomposition) const {
    decomposition.tensors.push_back(
        std::make_unique<tflite::TensorT>(*subgraph_.tensors[index]));
    return static_cast<int32_t>(decomposition.tensors.size() - 1);
  }

  // The copy of a tensor the region reads: an input or an earlier result if
  // it is one, otherwise a constant, which is copied with its buffer index, so
  // the data is shared with the subgraph's tensor, not duplicated.
  int32_t CopyOf(int32_t index, tflite::SubGraphT& decomposition,
                 absl::flat_hash_map<int32_t, int32_t>& copies) const {
    if (const auto it = copies.find(index); it != copies.end()) {
      return it->second;
    }
    const int32_t copy = AddCopy(index, decomposition);
    copies.emplace(index, copy);
    CopyQuantizationTensors(copy, decomposition, copies);
    return copy;
  }

  // Tensor `copy` of the decomposition still names the subgraph's scale and
  // zero point tensors: gives it copies of its own.
  void CopyQuantizationTensors(
      int32_t copy, tflite::SubGraphT& decomposition,
      absl::flat_hash_map<int32_t, int32_t>& copies) const {
    // The tensor object stays where it is while the vector grows.
    tflite::TensorT& tensor = *decomposition.tensors[copy];
    for (int32_t* ref : QuantizationTensors(tensor)) {
      if (*ref >= 0) *ref = CopyOf(*ref, decomposition, copies);
    }
  }

  uint32_t CompositeOpCode() {
    auto& codes = model_.operator_codes;
    for (size_t i = 0; i < codes.size(); ++i) {
      if (codes[i]->builtin_code ==
              tflite::BuiltinOperator_STABLEHLO_COMPOSITE &&
          codes[i]->custom_code.empty()) {
        return static_cast<uint32_t>(i);
      }
    }
    auto code = std::make_unique<tflite::OperatorCodeT>();
    code->builtin_code = tflite::BuiltinOperator_STABLEHLO_COMPOSITE;
    // Readers from before the wide code take the larger of the two fields.
    code->deprecated_builtin_code = static_cast<int8_t>(
        tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
    codes.push_back(std::move(code));
    return static_cast<uint32_t>(codes.size() - 1);
  }

  // NAME.impl, NAME.impl_0, NAME.impl_1, ... as the converter names the
  // decompositions of one composite.
  std::string UniqueName(const std::string& base) {
    int& next = next_suffix_.try_emplace(base, -1).first->second;
    for (;;) {
      std::string name = next < 0 ? base : absl::StrCat(base, "_", next);
      ++next;
      if (names_.insert(name).second) return name;
    }
  }

  tflite::ModelT& model_;
  const size_t subgraph_index_;
  tflite::SubGraphT& subgraph_;
  // The operators that belong to a region, and those of them that go away
  // (all but the one the composite takes the place of).
  std::vector<bool> replaced_;
  std::vector<bool> removed_;
  // The tensors the regions' operators read or wrote.
  std::vector<bool> region_tensors_;
  absl::flat_hash_set<std::string> names_;
  absl::flat_hash_map<std::string, int> next_suffix_;
};

}  // namespace

std::vector<int32_t*> QuantizationTensors(tflite::TensorT& tensor) {
  if (tensor.quantization == nullptr) return {};
  auto& details = tensor.quantization->details;
  if (auto* blockwise = details.AsBlockwiseQuantization()) {
    return {&blockwise->scales, &blockwise->zero_points};
  }
  if (auto* multi_axis = details.AsMultiAxisQuantization()) {
    return {&multi_axis->scales, &multi_axis->zero_points};
  }
  return {};
}

absl::Status OutlineRegions(tflite::ModelT& model, size_t subgraph_index,
                            const std::vector<CarveRegionSpec>& regions) {
  if (subgraph_index >= model.subgraphs.size()) {
    return absl::InvalidArgumentError("subgraph index out of range");
  }
  Outliner outliner(model, subgraph_index);
  for (const auto& region : regions) {
    if (auto status = outliner.Outline(region); !status.ok()) return status;
  }
  outliner.Finish();
  return absl::OkStatus();
}

absl::StatusOr<CarvedModel> SerializeCarvedModel(
    tflite::ModelT& model, litert::BufferRef<uint8_t> source) {
  // A buffer inside the flatbuffer whose data is present and empty makes its
  // tensor a constant without elements (an empty shape). The schema's objects
  // cannot say "present and empty": written back, the data would be absent
  // and the tensor no constant. Such a buffer becomes data of no bytes behind
  // the flatbuffer, which is how models with their data there hold these
  // constants.
  std::vector<bool> is_empty(model.buffers.size(), false);
  bool any_empty = false;
  if (const auto* buffers = tflite::GetModel(source.Data())->buffers()) {
    if (buffers->size() != model.buffers.size()) {
      return absl::InvalidArgumentError(
          "the model's buffers are not those of its source");
    }
    for (flatbuffers::uoffset_t i = 0; i < buffers->size(); ++i) {
      const auto* buffer = buffers->Get(i);
      is_empty[i] = buffer->data() != nullptr && buffer->data()->size() == 0 &&
                    !IsAppended(buffer->offset());
      any_empty |= is_empty[i];
    }
  }

  // Where the source's flatbuffer ends and the data behind it begins.
  uint64_t first = std::numeric_limits<uint64_t>::max();
  auto note = [&](uint64_t offset, uint64_t size) {
    if (!IsAppended(offset)) return true;
    first = std::min(first, offset);
    return offset <= source.Size() && size <= source.Size() - offset;
  };
  bool in_range = true;
  for (const auto& buffer : model.buffers) {
    in_range &= note(buffer->offset, buffer->size);
  }
  for (const auto& subgraph : model.subgraphs) {
    for (const auto& op : subgraph->operators) {
      in_range &=
          note(op->large_custom_options_offset, op->large_custom_options_size);
    }
  }
  if (!in_range) {
    return absl::InvalidArgumentError(
        "a buffer of the model lies outside the file");
  }

  // An offset is a fixed-width field that the flatbuffer stores when it is
  // above 1. The model therefore has its final size with any such offsets,
  // and is packed a second time once that size has told where the data goes.
  auto place_empty_buffers = [&](uint64_t offset) {
    for (size_t i = 0; i < model.buffers.size(); ++i) {
      if (is_empty[i]) model.buffers[i]->offset = offset;
    }
  };
  place_empty_buffers(2);

  CarvedModel carved;
  if (first == std::numeric_limits<uint64_t>::max()) {
    carved.flatbuffer =
        litert::internal::SerializeFlatbuffer(model, kTailPadding);
    if (!any_empty) return carved;
    // No data behind the flatbuffer: the empty constants point at the
    // padding.
    const uint64_t size = carved.flatbuffer.Size();
    place_empty_buffers(size - kTailPadding);
    carved.flatbuffer =
        litert::internal::SerializeFlatbuffer(model, kTailPadding);
    if (carved.flatbuffer.Size() != size) {
      return absl::InternalError(
          "placing the buffers changed the model's size");
    }
    return carved;
  }

  // The data behind the flatbuffer is copied as one block, so the flatbuffer
  // must end before it starts.
  flatbuffers::Verifier::Options verifier_options;
  verifier_options.max_size = std::numeric_limits<size_t>::max();
  flatbuffers::Verifier verifier(source.Data(), first, verifier_options);
  if (!tflite::VerifyModelBuffer(verifier)) {
    return absl::UnimplementedError(
        "the model's flatbuffer and the data behind it are interleaved");
  }

  const uint64_t size = litert::internal::SerializeFlatbuffer(model).Size();
  const uint64_t new_first =
      size <= first ? first - (first - size) / kDataAlignment * kDataAlignment
                    : first + (size - first + kDataAlignment - 1) /
                                  kDataAlignment * kDataAlignment;
  auto move = [&](uint64_t& offset) {
    if (IsAppended(offset)) offset = offset - first + new_first;
  };
  for (size_t i = 0; i < model.buffers.size(); ++i) {
    if (!is_empty[i]) move(model.buffers[i]->offset);
  }
  for (auto& subgraph : model.subgraphs) {
    for (auto& op : subgraph->operators) move(op->large_custom_options_offset);
  }
  place_empty_buffers(new_first);
  carved.flatbuffer = litert::internal::SerializeFlatbuffer(model);
  if (carved.flatbuffer.Size() != size) {
    return absl::InternalError("moving the buffers changed the model's size");
  }
  carved.gap = new_first - size;
  carved.appended =
      litert::BufferRef<uint8_t>(source.Data() + first, source.Size() - first);
  return carved;
}

absl::StatusOr<std::optional<CarvedModel>> CarveModel(
    litert::BufferRef<uint8_t> source, absl::Span<const CarvePattern> patterns,
    CarveReport* report) {
  auto loaded = litert::internal::LoadModelFromBuffer(source);
  if (!loaded) {
    return absl::InvalidArgumentError(absl::StrCat(
        "LiteRT cannot load the model: ", loaded.Error().Message()));
  }
  const LiteRtModelT& model = **loaded;

  std::vector<bool> is_decomposition(model.NumSubgraphs(), false);
  for (const auto* subgraph : model.Subgraphs()) {
    for (const auto* op : subgraph->Ops()) {
      if (op->OpCode() == kLiteRtOpCodeShloComposite) {
        const auto* options =
            litert::internal::GetTflOptions2(*op).AsStableHLOCompositeOptions();
        if (options != nullptr && options->decomposition_subgraph_index >= 0 &&
            static_cast<size_t>(options->decomposition_subgraph_index) <
                is_decomposition.size()) {
          is_decomposition[options->decomposition_subgraph_index] = true;
        }
      } else if (op->OpCode() == kLiteRtOpCodeTflCustom) {
        // The bytecode of a compiled model sits behind the flatbuffer at
        // offsets its dispatch ops hold; a rewrite would have to move them.
        if (const auto code = op->CustomCode();
            code && *code == litert::internal::kLiteRtDispatchOpCustomName) {
          return absl::UnimplementedError("the model is already compiled");
        }
      }
    }
  }

  report->num_subgraphs = model.NumSubgraphs();
  std::vector<std::vector<CarveRegionSpec>> regions(model.NumSubgraphs());
  for (size_t s = 0; s < model.NumSubgraphs(); ++s) {
    if (is_decomposition[s]) {
      ++report->num_decompositions_skipped;
      continue;
    }
    const LiteRtSubgraphT& subgraph = model.Subgraph(s);
    const CarveMatches matches = FindCarveRegions(patterns, subgraph);
    if (matches.regions.empty() && matches.near_misses.empty()) continue;

    // LiteRT keeps the file's order of tensors and operators, so a position
    // in the loaded subgraph is the index in the file.
    absl::flat_hash_map<const LiteRtTensorT*, int32_t> tensor_index;
    for (size_t i = 0; i < subgraph.Tensors().size(); ++i) {
      tensor_index[subgraph.Tensors()[i]] = static_cast<int32_t>(i);
    }
    absl::flat_hash_map<const LiteRtOpT*, int32_t> op_index;
    for (size_t i = 0; i < subgraph.Ops().size(); ++i) {
      op_index[subgraph.Ops()[i]] = static_cast<int32_t>(i);
    }
    auto entry_of = [&](const CarvePattern& pattern) -> CarveReportEntry& {
      for (auto& entry : report->entries) {
        if (entry.subgraph_index == s && entry.pattern == pattern.name) {
          return entry;
        }
      }
      auto& entry = report->entries.emplace_back();
      entry.subgraph_index = s;
      entry.subgraph_name = std::string(subgraph.Name());
      entry.pattern = pattern.name;
      return entry;
    };
    for (const auto& region : matches.regions) {
      auto& spec = regions[s].emplace_back();
      spec.composite_name = region.pattern->name;
      for (const auto* op : region.ops) spec.ops.push_back(op_index.at(op));
      for (const auto* tensor : region.inputs) {
        spec.inputs.push_back(tensor_index.at(tensor));
      }
      for (const auto* tensor : region.outputs) {
        spec.outputs.push_back(tensor_index.at(tensor));
      }
      spec.attributes = PackCarveAttributes(region.attributes);
      ++entry_of(*region.pattern).num_carved;
      ++report->num_carved;
    }
    for (const auto& near_miss : matches.near_misses) {
      auto& near_misses = entry_of(*near_miss.pattern).near_misses;
      auto it = std::find_if(near_misses.begin(), near_misses.end(),
                             [&](const CarveReportEntry::NearMiss& n) {
                               return n.reason == near_miss.reason;
                             });
      if (it == near_misses.end()) {
        it = near_misses.insert(near_misses.end(), {near_miss.reason, 0, ""});
        it->first_place =
            absl::StrCat("operator ", op_index.at(near_miss.anchor));
      }
      ++it->count;
    }
  }
  if (report->num_carved == 0) return std::nullopt;

  std::unique_ptr<tflite::ModelT> unpacked(
      tflite::GetModel(source.Data())->UnPack());
  for (size_t s = 0; s < regions.size(); ++s) {
    if (regions[s].empty()) continue;
    if (auto status = OutlineRegions(*unpacked, s, regions[s]); !status.ok()) {
      return status;
    }
  }
  auto carved = SerializeCarvedModel(*unpacked, source);
  if (!carved.ok()) return carved.status();
  return std::optional<CarvedModel>(std::move(*carved));
}

}  // namespace litert::tools
