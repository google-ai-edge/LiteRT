/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "litert/vendors/arm_vulkan_ml/common/vgf/loader.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "parse_vgf.hpp"
#include "vgf/decoder.hpp"
#include "vgf/vulkan_helpers.generated.hpp"

namespace litert::arm_vulkan_ml::vgf {
namespace {

using litert::Error;
using litert::Expected;
using mlsdk::vgfutils::ModelSequence;
using mlsdk::vgfutils::Resource;

Expected<size_t> ComputeTensorBytes(const std::vector<int64_t>& shape,
                                    VkFormat format) {
  if (shape.empty()) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Unshaped VGF tensors are not supported");
  }

  size_t elements = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      return Error(
          kLiteRtStatusErrorUnsupported,
          "Dynamic or non positive VGF tensor dimensions are not supported");
    }

    if (elements >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "VGF tensor element count overflow");
    }
    elements *= static_cast<size_t>(dim);
  }

  const auto element_size =
      mlsdk::vgflib::blockSize(static_cast<mlsdk::vgflib::FormatType>(format));
  if (element_size == 0) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Unsupported VGF VkFormat block size");
  }
  if (elements > std::numeric_limits<size_t>::max() / element_size) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "VGF tensor byte size overflow");
  }
  return elements * element_size;
}

std::string SelectGraphName(
    const mlsdk::vgfutils::Segment& segment,
    const mlsdk::vgflib::ModuleTableDecoder& module_table_decoder) {
  // Different VGFs could expose the same runnable graph under different
  // metadata fields. Prefer the callable entry point name when present, then
  // fall back to the module name and finally the raw segment name from the
  // sequence table so callers get the correct identifier
  const auto entry_point = std::string(
      module_table_decoder.getModuleEntryPoint(segment.mModuleIndex));
  if (!entry_point.empty()) {
    return entry_point;
  }

  const auto module_name =
      std::string(module_table_decoder.getModuleName(segment.mModuleIndex));
  if (!module_name.empty()) {
    return module_name;
  }

  if (!segment.mName.empty()) {
    return segment.mName;
  }

  return "graph";
}

bool SegmentMatches(
    std::string_view preferred_name, const mlsdk::vgfutils::Segment& segment,
    const mlsdk::vgflib::ModuleTableDecoder& module_table_decoder) {
  if (preferred_name.empty()) {
    return true;
  }

  // The requested graph name may come from any naming layer the VGF metadata
  // exposes. Check all of them so callers can address a graph by the segment
  // name in the sequence table, the module name in the module table, or the
  // actual shader entry point name.
  return preferred_name == segment.mName ||
         preferred_name ==
             module_table_decoder.getModuleName(segment.mModuleIndex) ||
         preferred_name ==
             module_table_decoder.getModuleEntryPoint(segment.mModuleIndex);
}

std::unordered_map<uint32_t, std::string> BuildBindingNameMap(
    const std::vector<mlsdk::vgfutils::NamedBindingSlot>& named_bindings) {
  std::unordered_map<uint32_t, std::string> binding_names;
  binding_names.reserve(named_bindings.size());
  for (const auto& named_binding : named_bindings) {
    if (!named_binding.mName.empty()) {
      binding_names.try_emplace(named_binding.mBindingSlot.mBinding,
                                named_binding.mName);
    }
  }
  return binding_names;
}

Expected<TensorDesc> MakeTensorDesc(
    const mlsdk::vgfutils::BindingSlot& slot,
    const std::vector<mlsdk::vgfutils::Resource>& resources,
    const std::unordered_map<uint32_t, std::string>& binding_names,
    const char* fallback_prefix) {
  // Bindings refer into the resource table by MRT index, where the tensor
  // shape and format live.
  if (slot.mMrtIndex >= resources.size()) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                 "VGF binding MRT index is out of range");
  }

  const auto& resource = resources[slot.mMrtIndex];
  TensorDesc desc;
  desc.binding = slot.mBinding;
  desc.shape = resource.mShape;
  desc.format = static_cast<VkFormat>(resource.mVkFormat);
  if (auto name_it = binding_names.find(slot.mBinding);
      name_it != binding_names.end()) {
    desc.name = name_it->second;
  } else {
    desc.name = std::string(fallback_prefix) + std::to_string(slot.mBinding);
  }

  auto bytes = ComputeTensorBytes(desc.shape, desc.format);
  if (!bytes) {
    return bytes.Error();
  }
  desc.bytes = *bytes;
  return desc;
}

Expected<GraphDesc> LoadVgfGraph(const void* borrowed_data, size_t size,
                                 std::string_view graph_name,
                                 std::vector<char> owned_data) {
  GraphDesc graph;
  const bool use_owned_data = !owned_data.empty();

  if (use_owned_data) {
    graph.vgf_data = std::move(owned_data);
  }

  const void* data = use_owned_data
                         ? static_cast<const void*>(graph.vgf_data.data())
                         : borrowed_data;

  if (data == nullptr || size == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "VGF buffer must not be null or empty");
  }

  const size_t header_size = mlsdk::vgflib::HeaderSize();
  if (size < header_size) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                 "VGF buffer is smaller than the header");
  }

  auto header_decoder = mlsdk::vgflib::CreateHeaderDecoder(
      data, /*headerSize=*/header_size, /*fileSize=*/size);

  if (!header_decoder) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer, "Invalid VGF header");
  }

  const uint64_t module_table_offset = header_decoder->GetModuleTableOffset();
  const uint64_t module_table_size = header_decoder->GetModuleTableSize();
  const uint64_t sequence_table_offset =
      header_decoder->GetModelSequenceTableOffset();
  const uint64_t sequence_table_size =
      header_decoder->GetModelSequenceTableSize();
  const uint64_t resource_table_offset =
      header_decoder->GetModelResourceTableOffset();
  const uint64_t resource_table_size =
      header_decoder->GetModelResourceTableSize();
  const uint64_t constants_offset = header_decoder->GetConstantsOffset();
  const uint64_t constants_size = header_decoder->GetConstantsSize();

  const auto* bytes = static_cast<const uint8_t*>(data);
  const void* module_table_ptr = bytes + module_table_offset;
  const void* sequence_table_ptr = bytes + sequence_table_offset;
  const void* resource_table_ptr = bytes + resource_table_offset;
  const void* constants_ptr = bytes + constants_offset;

  auto module_table_decoder = mlsdk::vgflib::CreateModuleTableDecoder(
      module_table_ptr, module_table_size);

  if (!module_table_decoder) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                 "Invalid VGF module table");
  }

  auto constant_decoder =
      mlsdk::vgflib::CreateConstantDecoder(constants_ptr, constants_size);

  if (!constant_decoder) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                 "Invalid VGF constants section");
  }

  ModelSequence model_sequence = mlsdk::vgfutils::parseModelSequenceTable(
      sequence_table_ptr, sequence_table_size);

  std::vector<Resource> resources = mlsdk::vgfutils::parseModelResourceTable(
      resource_table_ptr, resource_table_size);

  const auto input_binding_names = BuildBindingNameMap(model_sequence.mInputs);
  const auto output_binding_names =
      BuildBindingNameMap(model_sequence.mOutputs);

  // A VGF can carry multiple modules, including non graph segments and graph
  // segments that do not contain SPIR-V. Filter down to runnable
  // GRAPH segments first, then resolve the callers name against the
  // different naming layers VGF metadata may provide for the same graph.
  std::vector<const mlsdk::vgfutils::Segment*> candidate_segments;
  candidate_segments.reserve(model_sequence.mSegments.size());

  const mlsdk::vgfutils::Segment* selected_segment = nullptr;
  size_t matching_segment_count = 0;
  for (const auto& segment : model_sequence.mSegments) {
    if (segment.mType != mlsdk::vgflib::ModuleType::GRAPH) {
      continue;
    }
    if (!module_table_decoder->hasSPIRV(segment.mModuleIndex)) {
      continue;
    }
    candidate_segments.push_back(&segment);
    if (graph_name.empty()) {
      // When no graph name is provided, only accept files with a single
      // runnable GRAPH segment. Multi graph VGFs need an explicit name so we
      // do not silently pick the wrong workload.
      if (selected_segment != nullptr) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Multiple VGF GRAPH segments found; function name is "
                     "required to be clear");
      }
      selected_segment = &segment;
      continue;
    }

    // If a graph name was provided, compare it against every identifier that
    // may refer to the same runnable segment in VGF metadata.
    if (SegmentMatches(graph_name, segment, *module_table_decoder)) {
      ++matching_segment_count;
      selected_segment = &segment;
    }
  }

  // Name aliases are allowed, but they must still resolve to a single runnable
  // segment. If the requested name matches more than one GRAPH segment, force
  // the caller to be explicit.
  if (matching_segment_count > 1) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Requested function name matched multiple VGF GRAPH segments");
  }

  if (selected_segment == nullptr) {
    if (!graph_name.empty() && !candidate_segments.empty()) {
      if (candidate_segments.size() == 1) {
        // if there is only one candidate segment and it doesn't match the
        // requested name, its safe to assume we use the single candidate
        // segment rather than throwing an error over the name mismatch
        selected_segment = candidate_segments.front();
      } else {
        return Error(
            kLiteRtStatusErrorNotFound,
            "No VGF GRAPH segment matched the requested function name");
      }
    }
  }

  if (selected_segment == nullptr) {
    return Error(kLiteRtStatusErrorNotFound,
                 "No GRAPH segment with SPIR-V found in VGF");
  }

  graph.name = SelectGraphName(*selected_segment, *module_table_decoder);

  // The module table owns the actual payload referenced by the selected graph
  // segment.
  auto spirv =
      module_table_decoder->getModuleCode(selected_segment->mModuleIndex);
  graph.spirv.assign(spirv.begin(), spirv.end());
  if (graph.spirv.empty()) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                 "Selected VGF graph has empty SPIR-V");
  }

  graph.inputs.reserve(selected_segment->mInputs.size());
  for (const auto& slot : selected_segment->mInputs) {
    auto desc = MakeTensorDesc(slot, resources, input_binding_names, "input_");
    if (!desc) {
      return desc.Error();
    }
    graph.inputs.push_back(std::move(*desc));
  }

  graph.outputs.reserve(selected_segment->mOutputs.size());
  for (const auto& slot : selected_segment->mOutputs) {
    auto desc =
        MakeTensorDesc(slot, resources, output_binding_names, "output_");
    if (!desc) {
      return desc.Error();
    }
    graph.outputs.push_back(std::move(*desc));
  }

  graph.const_inputs.reserve(selected_segment->mConstantBindings.size());
  for (const auto& binding : selected_segment->mConstantBindings) {
    const uint32_t constant_index = binding.mConstantIndex;
    if (constant_index >= constant_decoder->size()) {
      return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                   "VGF constant index is out of range");
    }

    const uint32_t mrt_index =
        constant_decoder->getConstantMrtIndex(constant_index);

    if (mrt_index >= resources.size()) {
      return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                   "VGF constant MRT index is out of range");
    }

    const auto& resource = resources[mrt_index];
    ConstantTensorDesc desc;

    // Graph IDs can differ from the payload's index in the constants table.
    // The VGF parser supplies identity bindings for legacy files.
    desc.binding = binding.mGraphConstantId;
    desc.shape = resource.mShape;
    desc.format = static_cast<VkFormat>(resource.mVkFormat);
    desc.name = "const_" + std::to_string(binding.mGraphConstantId);

    auto constant_data = constant_decoder->getConstant(constant_index);
    if (constant_data.empty()) {
      return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                   "VGF constant data is empty");
    }

    auto bytes = ComputeTensorBytes(desc.shape, desc.format);

    if (!bytes || constant_data.size() != *bytes) {
      return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                   "VGF constant data size does not match tensor metadata");
    }
    desc.bytes = *bytes;

    if (use_owned_data) {
      const uintptr_t constant_address =
          reinterpret_cast<uintptr_t>(constant_data.data());
      const uintptr_t vgf_address =
          reinterpret_cast<uintptr_t>(graph.vgf_data.data());

      if (desc.bytes > graph.vgf_data.size() ||
          constant_address < vgf_address ||
          constant_address - vgf_address > graph.vgf_data.size() - desc.bytes) {
        return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                     "VGF constant data is outside the owned buffer");
      }

      desc.vgf_data_offset =
          static_cast<size_t>(constant_address - vgf_address);
      desc.uses_vgf_data = true;
    } else {
      desc.data.assign(constant_data.begin(), constant_data.end());
    }
    graph.const_inputs.push_back(std::move(desc));
  }

  if (graph.outputs.empty() || graph.inputs.empty()) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                 "Selected VGF graph has no inputs or outputs");
  }

  return graph;
}

}  // namespace

Expected<GraphDesc> LoadVgfGraphFromBuffer(const void* data, size_t size,
                                           std::string_view graph_name) {
  return LoadVgfGraph(data, size, graph_name, {});
}

Expected<GraphDesc> LoadVgfGraphFromOwnedBuffer(std::vector<char> data,
                                                std::string_view graph_name) {
  const size_t size = data.size();
  return LoadVgfGraph(nullptr, size, graph_name, std::move(data));
}

}  // namespace litert::arm_vulkan_ml::vgf
