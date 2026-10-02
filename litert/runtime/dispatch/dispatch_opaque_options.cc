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

#include "litert/runtime/dispatch/dispatch_opaque_options.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::internal {

namespace {

struct Payload {
  // Base address of the memory allocation.
  const void* alloc_base = nullptr;
  // File descriptor for the backing file of the allocation.
  int alloc_base_fd = -1;
  absl::flat_hash_map<std::string, LiteRtJitExecutable> jit_executable_handles;
  // Byte offset of alloc_base in the backing file when alloc_base_fd is used.
  size_t alloc_base_file_offset = 0;
  // Size in bytes of the model rooted at alloc_base.
  size_t alloc_base_size = 0;
  // Indicates whether the file region metadata (offset or size) is populated.
  bool has_alloc_base_file_region = false;
  // Mapping of a function name to its input and output tensor ports. Can be
  // used to identify the tensor port index for a given signature namd + tensor
  // name from a delegate kernel.
  absl::flat_hash_map<std::string, NodeTensorPortMapping>
      node_tensor_port_mappings;
  // Mapping of a function/node name to its original TFLite signature name.
  absl::flat_hash_map<std::string, std::string> function_to_signature_map;
};

}  // namespace

Expected<DispatchDelegateOptions> DispatchDelegateOptions::Create(
    OpaqueOptions& options) {
  const auto id = options.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return DispatchDelegateOptions(options.Get(), OwnHandle::kNo);
}

Expected<DispatchDelegateOptions> DispatchDelegateOptions::Create() {
  LiteRtOpaqueOptions opaque_options;
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      Discriminator(), new Payload(),
      [](void* payload) { delete reinterpret_cast<Payload*>(payload); },
      &opaque_options));
  return DispatchDelegateOptions(opaque_options, OwnHandle::kYes);
}

// TODO LUKE document
Expected<void> DispatchDelegateOptions::SetAllocBase(const void* alloc_base) {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  payload->alloc_base = alloc_base;
  return {};
}
Expected<const void*> DispatchDelegateOptions::GetAllocBase() {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  return payload->alloc_base;
}

// TODO LUKE document
Expected<void> DispatchDelegateOptions::SetAllocBaseFd(int alloc_base_fd) {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  payload->alloc_base_fd = alloc_base_fd;
  return {};
}
Expected<int> DispatchDelegateOptions::GetAllocBaseFd() {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  return payload->alloc_base_fd;
}

Expected<void> DispatchDelegateOptions::AddExecHandle(
    absl::string_view name, LiteRtJitExecutable handle) {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  payload->jit_executable_handles[name] = handle;
  return {};
}

Expected<LiteRtJitExecutable> DispatchDelegateOptions::GetExecHandle(
    absl::string_view name) {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  auto it = payload->jit_executable_handles.find(name);
  if (it != payload->jit_executable_handles.end()) {
    return it->second;
  }
  return nullptr;
}

Expected<void> DispatchDelegateOptions::SetAllocBaseFileOffset(
    size_t alloc_base_file_offset) {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  payload->alloc_base_file_offset = alloc_base_file_offset;
  payload->has_alloc_base_file_region = true;
  return {};
}

Expected<size_t> DispatchDelegateOptions::GetAllocBaseFileOffset() {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  return payload->alloc_base_file_offset;
}

Expected<void> DispatchDelegateOptions::SetAllocBaseSize(
    size_t alloc_base_size) {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  payload->alloc_base_size = alloc_base_size;
  payload->has_alloc_base_file_region = true;
  return {};
}

Expected<size_t> DispatchDelegateOptions::GetAllocBaseSize() {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  return payload->alloc_base_size;
}

Expected<bool> DispatchDelegateOptions::HasAllocBaseFileRegion() {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  return payload->has_alloc_base_file_region;
}

Expected<void> DispatchDelegateOptions::SetNodeTensorPortMapping(
    absl::string_view function_name,
    std::vector<TensorPortMapping> input_tensor_ports,
    std::vector<TensorPortMapping> output_tensor_ports) {
  if (function_name.empty()) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  NodeTensorPortMapping mapping;
  mapping.input_tensor_ports = std::move(input_tensor_ports);
  mapping.output_tensor_ports = std::move(output_tensor_ports);
  payload->node_tensor_port_mappings[function_name] = std::move(mapping);
  return {};
}

Expected<const NodeTensorPortMapping*>
DispatchDelegateOptions::GetNodeTensorPortMapping(
    absl::string_view function_name) {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  auto it = payload->node_tensor_port_mappings.find(function_name);
  if (it == payload->node_tensor_port_mappings.end()) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound);
  }
  return &it->second;
}

Expected<void> DispatchDelegateOptions::RegisterFunctionSignature(
    absl::string_view function_name, absl::string_view signature_name) {
  if (function_name.empty() || signature_name.empty()) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  payload->function_to_signature_map[function_name] =
      std::string(signature_name);
  return {};
}

Expected<std::string_view> DispatchDelegateOptions::GetFunctionSignature(
    absl::string_view function_name) {
  LITERT_ASSIGN_OR_RETURN(Payload * payload, GetData<Payload>());
  auto it = payload->function_to_signature_map.find(function_name);
  if (it == payload->function_to_signature_map.end()) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound);
  }
  return std::string_view(it->second);
}

}  // namespace litert::internal
