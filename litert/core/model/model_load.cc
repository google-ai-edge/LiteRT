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

#include "litert/core/model/model_load.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/build_stamp.h"
#include "litert/core/dispatch_bytecode_manifest.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/flatbuffer_to_litert.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

// Provides a view of model-level resources when constructing litert graph.
class FlatbufferContext {
 public:
  using LiteRtBufferId = uint32_t;
  using TflBufferInd = uint32_t;
  using BufferIdMap = absl::flat_hash_map<TflBufferInd, LiteRtBufferId>;

  FlatbufferContext(const FlatbufferWrapper& tfl_flatbuffer,
                    BufferManager* buffer_manager)
      : tfl_flatbuffer_(tfl_flatbuffer), buffer_manager_(buffer_manager) {}

  Expected<void> SetOpCode(LiteRtOpT& litert_op, uint32_t ind) {
    const auto* code = PackedModel()->operator_codes()->Get(ind);
    const int32_t builtin_code = code->builtin_code();
    const int32_t dep_code = code->deprecated_builtin_code();
    litert_op.SetOpCode(
        static_cast<LiteRtOpCode>(std::max(dep_code, builtin_code)));
    if (code->custom_code()) {
      litert_op.SetCustomCode(code->custom_code()->str());
    }
    litert::internal::SetTflOpCodeInd(litert_op, ind);
    return {};
  }

  // Get the buffer at the given index in the tflite model.
  Expected<const TflPackedBuffer*> GetTflBuffer(uint32_t ind) const {
    const auto* packed_model = tfl_flatbuffer_.PackedModel();
    if (ind >= packed_model->buffers()->size()) {
      LITERT_LOG(LITERT_ERROR, "Buffer index out of range");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    return packed_model->buffers()->Get(ind);
  }

  BufferManager* GetBufferManager() { return buffer_manager_; }

  const uint8_t* AllocBase() const { return tfl_flatbuffer_.AllocBase(); }

  const TflPackedModel* PackedModel() const {
    return tfl_flatbuffer_.PackedModel();
  }

  BufferIdMap& RegisteredTflBufferIds() { return registered_tfl_buffer_ids_; }

 private:
  const FlatbufferWrapper& tfl_flatbuffer_;
  BufferManager* buffer_manager_;
  BufferIdMap registered_tfl_buffer_ids_;
};

LiteRtStatus UnpackOp(FlatbufferContext& context, LiteRtSubgraphT& parent,
                      const TflPackedOp& tfl_op, LiteRtOpT& litert_op,
                      size_t op_index) {
  // I/O TENSORS

  if (tfl_op.intermediates() && tfl_op.intermediates()->size() != 0) {
    // TODO: b/365299994 - Support intermediates.
    LITERT_LOG(LITERT_ERROR, "Intermediate tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tfl_op.mutating_variable_inputs() &&
      tfl_op.mutating_variable_inputs()->size() != 0) {
    // TODO: b/365299994 - Support mutating variable inputs.
    LITERT_LOG(LITERT_ERROR, "Mutating variable inputs not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  const auto num_inputs = tfl_op.inputs()->size();
  for (auto i = 0; i < num_inputs; ++i) {
    const auto input_ind = tfl_op.inputs()->Get(i);
    // Skipping optional input tensor.
    if (input_ind == -1) {
      continue;
    }
    AttachInput(&parent.Tensor(input_ind), litert_op);
  }

  const auto num_outputs = tfl_op.outputs()->size();
  for (auto i = 0; i < num_outputs; ++i) {
    const auto output_ind = tfl_op.outputs()->Get(i);
    AttachOutput(&parent.Tensor(output_ind), litert_op);
  }

  // OPTIONS

  if (tfl_op.large_custom_options_size() != 0) {
    // TODO: b/365299994 - Support large custom options.
    LITERT_LOG(LITERT_WARNING,
               "Large custom options not yet supported in litert::Model.");
  }

  const auto* custom_opts = tfl_op.custom_options();
  if (custom_opts) {
    litert_op.SetCustomOptions(custom_opts->data(), custom_opts->size());
  }

  // TODO figure out how to parse builtins with the packed flatbuffer api.
  TflOpPtr tfl_op_ptr(tfl_op.UnPack());
  litert::internal::SetTflOptions(litert_op,
                                  std::move(tfl_op_ptr->builtin_options));
  litert::internal::SetTflOptions2(litert_op,
                                   std::move(tfl_op_ptr->builtin_options_2));

  // OP CODE

  LITERT_RETURN_IF_ERROR(context.SetOpCode(litert_op, tfl_op.opcode_index()));
  litert_op.SetOpIndex(op_index);

  return kLiteRtStatusOk;
}

struct TflBufferContext {
  BufferRef<uint8_t> buffer;
  // Is buffer appended to the flatbuffer?
  bool is_external;
};

Expected<TflBufferContext> ReadBuffer(FlatbufferContext& context,
                                      uint32_t buffer_ind) {
  auto buffer = context.GetTflBuffer(buffer_ind);
  if (!buffer) {
    return buffer.Error();
  }

  const auto& tfl_buffer = **buffer;

  if (tfl_buffer.offset() != 0) {
    // Data is appended to the end of the flatbuffer.

    const auto* alloc_base = context.AllocBase();
    const auto offset = tfl_buffer.offset();
    const auto size = tfl_buffer.size();

    return TflBufferContext{BufferRef<uint8_t>(alloc_base + offset, size),
                            true};
  } else if (tfl_buffer.data()) {
    // Data is in the flatbuffer.

    const auto* start = tfl_buffer.data()->data();
    const auto size = tfl_buffer.data()->size();

    return TflBufferContext{BufferRef<uint8_t>(start, size), false};
  } else {
    return TflBufferContext{};
  }
}

LiteRtStatus UnpackTensor(FlatbufferContext& context,
                          const TflPackedTensor& tfl_tensor,
                          LiteRtTensorT& litert_tensor) {
  const auto buffer_ind = tfl_tensor.buffer();
  if (buffer_ind != 0) {
    auto buffer = ReadBuffer(context, buffer_ind);
    if (!buffer) {
      return buffer.Error().Status();
    }

    auto it = context.RegisteredTflBufferIds().find(buffer_ind);
    if (it != context.RegisteredTflBufferIds().end()) {
      litert_tensor.Weights().SetBufferId(it->second);
    } else {
      BufferContext lrt_buf_ctx;
      lrt_buf_ctx.should_append = buffer->is_external;
      SetWeightsFromUnownedBuffer(litert_tensor.Weights(), buffer->buffer,
                                  lrt_buf_ctx);
      context.RegisteredTflBufferIds()[buffer_ind] =
          litert_tensor.Weights().GetBufferId();
    }
  }

  // TENSOR TYPE

  TflTensorType tfl_tensor_type(tfl_tensor.type(), TflShapeInfo(tfl_tensor));
  auto tensor_type = MapTensorType(tfl_tensor_type);
  if (!tensor_type) {
    return tensor_type.Error().Status();
  }

  litert_tensor.SetType(std::move(*tensor_type));

  // QUANTIZATION

  if (tfl_tensor.quantization()) {
    auto quantization =
        MapQuantization(tfl_tensor.quantization());
    if (!quantization) {
      return quantization.Error().Status();
    }
    litert_tensor.SetQarams(std::move(*quantization));
  }

  // MISC

  if (tfl_tensor.name()) {
    litert_tensor.SetName(tfl_tensor.name()->str());
  }

  if (tfl_tensor.is_variable()) {
    // TODO: b/365299994 - Support variable tensors.
    LITERT_LOG(LITERT_ERROR, "Variable tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tfl_tensor.variant_tensors() &&
      tfl_tensor.variant_tensors()->size() != 0) {
    // TODO: b/365299994 - Support variant tensors.
    LITERT_LOG(LITERT_ERROR, "Variant tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tfl_tensor.sparsity() != nullptr) {
    // TODO: b/365299994 - Support sparsity tensors.
    LITERT_LOG(LITERT_WARNING,
               "Sparsity tensors may not yet be fully supported.");
  }

  return kLiteRtStatusOk;
}

LiteRtStatus UnpackSubgraph(FlatbufferContext& context,
                            const TflPackedSubgraph& tfl_subgraph,
                            LiteRtSubgraphT& litert_subgraph) {
  // Unpack tensors.
  const auto num_tensors = tfl_subgraph.tensors()->size();
  for (auto i = 0; i < num_tensors; ++i) {
    const auto* tfl_tensor = tfl_subgraph.tensors()->Get(i);
    auto& litert_tensor = litert_subgraph.EmplaceTensor();
    LITERT_RETURN_IF_ERROR(UnpackTensor(context, *tfl_tensor, litert_tensor));
    litert_tensor.SetTensorIndex(i);
  }

  // Unpack ops, pass litert_subgraph so they can look up the new litert
  // tensors.
  const auto num_ops = tfl_subgraph.operators()->size();
  for (auto i = 0; i < num_ops; ++i) {
    const auto* tfl_op = tfl_subgraph.operators()->Get(i);
    LITERT_RETURN_IF_ERROR(UnpackOp(context, litert_subgraph, *tfl_op,
                                    litert_subgraph.EmplaceOp(), i));
  }

  // Update subgraph I/O.
  const auto num_inputs = tfl_subgraph.inputs()->size();
  for (auto i = 0; i < num_inputs; ++i) {
    const auto tfl_input_ind = tfl_subgraph.inputs()->Get(i);
    if (tfl_input_ind < 0 ||
        static_cast<size_t>(tfl_input_ind) >= num_tensors) {
      LITERT_LOG(LITERT_ERROR,
                 "flatbuffer model has invalid input index in subgraph: %d",
                 tfl_input_ind);
      return kLiteRtStatusErrorInvalidFlatbuffer;
    }
    litert_subgraph.Inputs().push_back(&litert_subgraph.Tensor(tfl_input_ind));
  }
  const auto num_outputs = tfl_subgraph.outputs()->size();
  for (auto i = 0; i < num_outputs; ++i) {
    const auto tfl_output_ind = tfl_subgraph.outputs()->Get(i);
    if (tfl_output_ind < 0 ||
        static_cast<size_t>(tfl_output_ind) >= num_tensors) {
      LITERT_LOG(LITERT_ERROR,
                 "flatbuffer model has invalid output index in subgraph: %d",
                 tfl_output_ind);
      return kLiteRtStatusErrorInvalidFlatbuffer;
    }
    litert_subgraph.Outputs().push_back(
        &litert_subgraph.Tensor(tfl_output_ind));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus UnpackSignatures(std::vector<TflSignaturePtr>& tfl_signatures,
                              LiteRtModelT& parent) {
  for (auto& tfl_signature : tfl_signatures) {
    if (tfl_signature->subgraph_index >= parent.Subgraphs().size()) {
      LITERT_LOG(LITERT_ERROR,
                 "Signature does not refer to a valid subgraph index.");
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* litert_subgraph =
        parent.Subgraphs().at(tfl_signature->subgraph_index);

    auto& tfl_inputs = tfl_signature->inputs;
    auto& tfl_outputs = tfl_signature->outputs;

    // Tflite signatures map a tensor index to a name. The input & output
    // indexes of signatures and subgraph are not matched, but the nubmer of
    // inputs and outputs should be the same.
    if (tfl_inputs.size() != litert_subgraph->Inputs().size() ||
        tfl_outputs.size() != litert_subgraph->Outputs().size()) {
      LITERT_LOG(LITERT_ERROR,
                 "Signature has incorrect number of input/outputs");
      return kLiteRtStatusErrorInvalidFlatbuffer;
    }

    absl::flat_hash_map<LiteRtTensor, std::string> input_aliases;
    input_aliases.reserve(tfl_inputs.size());
    for (const auto& tfl_input : tfl_inputs) {
      auto* tensor = litert_subgraph->Tensors().at(tfl_input->tensor_index);
      const std::string& name = tfl_input->name;
      input_aliases[tensor] =
          !name.empty() ? name : std::string(tensor->Name());
    }

    absl::flat_hash_map<LiteRtTensor, std::string> output_aliases;
    output_aliases.reserve(tfl_outputs.size());
    for (const auto& tfl_output : tfl_outputs) {
      auto* tensor = litert_subgraph->Tensors().at(tfl_output->tensor_index);
      const std::string& name = tfl_output->name;
      output_aliases[tensor] =
          !name.empty() ? name : std::string(tensor->Name());
    }

    // Keep signature input/output names in the same order as the subgraph.
    std::vector<std::string> input_names;
    std::vector<LiteRtTensor> input_tensors;
    input_names.reserve(tfl_inputs.size());
    for (auto& tensor : litert_subgraph->Inputs()) {
      input_names.push_back(input_aliases.contains(tensor)
                                ? input_aliases.at(tensor)
                                : std::string(tensor->Name()));
      input_tensors.push_back(tensor);
    }
    std::vector<std::string> output_names;
    std::vector<LiteRtTensor> output_tensors;
    output_names.reserve(tfl_outputs.size());
    for (auto& tensor : litert_subgraph->Outputs()) {
      output_names.push_back(output_aliases.contains(tensor)
                                 ? output_aliases.at(tensor)
                                 : std::string(tensor->Name()));
      output_tensors.push_back(tensor);
    }

    parent.EmplaceSignature(litert_subgraph, std::move(input_names),
                            std::move(input_tensors), std::move(output_names),
                            std::move(output_tensors),
                            tfl_signature->signature_key);
  }

  if (tfl_signatures.empty()) {
    parent.EmplaceSignature(MakeDefaultSignature(parent.MainSubgraph()));
  }

  return kLiteRtStatusOk;
}

using DispatchManifestMap = absl::flat_hash_map<std::pair<size_t, size_t>,
                                                DispatchBytecodeManifestEntry>;

bool IsByteRangeInBounds(size_t offset, size_t size, size_t total_size) {
  return offset <= total_size && size <= total_size - offset;
}

Expected<DispatchManifestMap> ParseDispatchManifest(const LiteRtModelT& model,
                                                    size_t model_size) {
  LITERT_ASSIGN_OR_RETURN(
      auto manifest_buffer,
      model.FindMetadata(kLiteRtDispatchBytecodeManifestKey));
  LITERT_ASSIGN_OR_RETURN(auto manifest_entries,
                          ParseDispatchBytecodeManifest(manifest_buffer));

  DispatchManifestMap manifest_by_op;
  manifest_by_op.reserve(manifest_entries.size());
  for (const auto& entry : manifest_entries) {
    const auto op_key = std::make_pair(entry.subgraph_index, entry.op_index);
    if (manifest_by_op.contains(op_key)) {
      return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                   "Dispatch manifest contains duplicate op mapping");
    }
    if (entry.bytecode_offset == 0 || entry.bytecode_size == 0) {
      return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                   "Dispatch manifest contains zero offset/size");
    }
    if (!IsByteRangeInBounds(entry.bytecode_offset, entry.bytecode_size,
                             model_size)) {
      return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                   "Dispatch manifest contains out-of-bounds bytecode range");
    }
    manifest_by_op.insert_or_assign(op_key, entry);
  }

  return manifest_by_op;
}

Expected<LiteRtModelT::Ptr> UnpackModel(FlatbufferWrapper&& flatbuffer) {
  auto litert_model = std::make_unique<LiteRtModelT>(std::move(flatbuffer));

  FlatbufferContext context(litert::internal::GetTflFlatbuffer(*litert_model),
                            litert_model->Buffers());
  const auto* packed_model = context.PackedModel();

  if (packed_model->subgraphs()) {
    const auto num_subgraphs = packed_model->subgraphs()->size();
    for (auto i = 0; i < num_subgraphs; ++i) {
      const auto* tfl_subgraph = packed_model->subgraphs()->Get(i);
      LITERT_RETURN_IF_ERROR(UnpackSubgraph(context, *tfl_subgraph,
                                            litert_model->EmplaceSubgraph()));
    }
  }

  // TODO Figure out how to load signatures in packed flatbuffer.
  if (packed_model->signature_defs()) {
    std::vector<TflSignaturePtr> tfl_signatures;
    for (auto i = 0; i < packed_model->signature_defs()->size(); ++i) {
      const auto* tfl_signature = packed_model->signature_defs()->Get(i);
      tfl_signatures.push_back(TflSignaturePtr(tfl_signature->UnPack()));
    }
    LITERT_RETURN_IF_ERROR(UnpackSignatures(tfl_signatures, *litert_model));
  } else {
    litert_model->EmplaceSignature(
        MakeDefaultSignature(litert_model->MainSubgraph()));
  }

  if (packed_model->metadata()) {
    const auto num_metadata = packed_model->metadata()->size();
    for (auto i = 0; i < num_metadata; ++i) {
      const auto* tfl_metadata = packed_model->metadata()->Get(i);
      auto name = tfl_metadata->name()->str();
      const auto buf_id = tfl_metadata->buffer();
      auto buf = ReadBuffer(context, buf_id);
      if (!buf) {
        return buf.Error();
      }

      litert_model->PushMetadata(name, buf->buffer.Data(), buf->buffer.Size());
    }
  }

  if (packed_model->operator_codes()) {
    const auto num_operator_codes = packed_model->operator_codes()->size();
    std::vector<TflOpCodePtr> tfl_op_codes(num_operator_codes);
    for (auto i = 0; i < num_operator_codes; ++i) {
      const auto* tfl_op_code = packed_model->operator_codes()->Get(i);
      TflOpCodePtr tfl_op_code_ptr(tfl_op_code->UnPack());
      tfl_op_codes[i] = std::move(tfl_op_code_ptr);
    }
    litert::internal::SetTflOpCodes(*litert_model, std::move(tfl_op_codes));
  }

  return litert_model;
}

}  // namespace

Expected<LiteRtModelT::Ptr> LoadModelFromBuffer(
    OwningBufferRef<uint8_t>&& buffer) {
  auto flatbuffer = FlatbufferWrapper::CreateFromBuffer(std::move(buffer));
  if (!flatbuffer) {
    return flatbuffer.Error();
  }
  return UnpackModel(std::move(**flatbuffer));
}

Expected<LiteRtModelT::Ptr> LoadModelFromBuffer(BufferRef<uint8_t> buffer) {
  auto flatbuffer = FlatbufferWrapper::CreateFromBuffer(buffer);
  if (!flatbuffer) {
    return flatbuffer.Error();
  }
  return UnpackModel(std::move(**flatbuffer));
}

Expected<LiteRtModelT::Ptr> LoadModelFromFile(absl::string_view filename,
                                              ModelFileLoadOptions options) {
  FlatbufferWrapper::FileLoadOptions flatbuffer_options;
  flatbuffer_options.allow_modifications = options.allow_modifications;
  flatbuffer_options.load_mode =
      options.load_mode == ModelFileLoadMode::kMetadataOnlyForFileCopy
          ? FlatbufferWrapper::FileLoadMode::kMetadataOnlyForFileCopy
          : FlatbufferWrapper::FileLoadMode::kDefault;

  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(filename, flatbuffer_options);
  if (!flatbuffer) {
    return flatbuffer.Error();
  }
  LITERT_ASSIGN_OR_RETURN(auto model, UnpackModel(std::move(**flatbuffer)));
  model->SetFileLoadMode(options.load_mode ==
                                 ModelFileLoadMode::kMetadataOnlyForFileCopy
                             ? kLiteRtModelFileLoadModeMetadataOnlyForFileCopy
                             : kLiteRtModelFileLoadModeDefault);
  model->SetSourcePath(std::string(filename));

  const size_t loaded_model_size = GetTflFlatbuffer(*model).Buf().Size();
  std::optional<DispatchManifestMap> dispatch_manifest;
  if (auto parsed_manifest = ParseDispatchManifest(*model, loaded_model_size);
      parsed_manifest) {
    dispatch_manifest = std::move(*parsed_manifest);
  } else if (parsed_manifest.Error().Status() != kLiteRtStatusErrorNotFound) {
    LITERT_LOG(LITERT_WARNING,
               "Ignoring invalid dispatch bytecode manifest metadata: %s",
               parsed_manifest.Error().Message().c_str());
  }

  // Load bytecode of each dispatch op and attach it to the model.
  absl::flat_hash_map<size_t, unsigned int> buffer_id_map;
  for (size_t subgraph_index = 0; subgraph_index < model->Subgraphs().size();
       ++subgraph_index) {
    LiteRtSubgraph subgraph = &model->Subgraph(subgraph_index);
    for (size_t op_index = 0; op_index < subgraph->Ops().size(); ++op_index) {
      LiteRtOp op = subgraph->Ops()[op_index];
      if (op->OpCode() != kLiteRtOpCodeTflCustom ||
          op->CustomOptions().Size() == 0) {
        continue;
      }
      auto op_custom_code = op->CustomCode();
      if (!op_custom_code || *op_custom_code != kLiteRtDispatchOpCustomName) {
        continue;
      }

      std::optional<DispatchBytecodeManifestEntry> manifest_entry;
      if (dispatch_manifest.has_value()) {
        auto it = dispatch_manifest->find({subgraph_index, op_index});
        if (it != dispatch_manifest->end()) {
          manifest_entry = it->second;
        }
      }

      DispatchOpOptions dispatch_opts;
      bool dispatch_opts_from_manifest = false;
      if (manifest_entry.has_value()) {
        dispatch_opts.bytecode_offset = manifest_entry->bytecode_offset;
        dispatch_opts.bytecode_size = manifest_entry->bytecode_size;
        dispatch_opts.name = manifest_entry->function_name;
        dispatch_opts_from_manifest = true;
      } else {
        dispatch_opts = GetDispatchOpOptions(op->CustomOptions());
      }

      if (dispatch_opts.bytecode_offset == 0 ||
          dispatch_opts.bytecode_size == 0) {
        if (dispatch_opts_from_manifest) {
          LITERT_LOG(LITERT_WARNING,
                     "Skipping dispatch manifest entry with empty bytecode "
                     "range");
        }
        continue;
      }
      if (!IsByteRangeInBounds(dispatch_opts.bytecode_offset,
                               dispatch_opts.bytecode_size,
                               loaded_model_size)) {
        // Metadata-only loads only keep the flatbuffer root in memory.
        continue;
      }
      if (!buffer_id_map.contains(dispatch_opts.bytecode_offset)) {
        BufferRef<uint8_t> byte_code(GetTflFlatbuffer(*model).AllocBase() +
                                         dispatch_opts.bytecode_offset,
                                     dispatch_opts.bytecode_size);
        const BufferManager::BufferId buf_id =
            model->Buffers()->RegisterNonOwnedBuffer(byte_code);
        buffer_id_map.insert({dispatch_opts.bytecode_offset, buf_id});
      }

      model->AttachAssetToOp(op, buffer_id_map[dispatch_opts.bytecode_offset],
                             std::move(dispatch_opts.name));
    }
  }

  return std::move(model);
}

Expected<LiteRtModelT::Ptr> LoadModelFromFile(absl::string_view filename,
                                              bool allow_modifications) {
  ModelFileLoadOptions options;
  options.allow_modifications = allow_modifications;
  return LoadModelFromFile(filename, options);
}

}  // namespace litert::internal
