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

#ifndef ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_OPAQUE_OPTIONS_H_
#define ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_OPAQUE_OPTIONS_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::internal {

// Mapping between a named model tensor on a delegated node/subgraph and its
// port index. The tensor name is not pure name since its the output of the
// TfLiteOpaqueTensorName call. Thus may contain the prefix of the signature
// name and a suffix indicating the output port index.
struct TensorPortMapping {
  absl::string_view opaque_tensor_name;
  int port_index = -1;
};

// Input and output tensor port mappings for a single delegated function/node.
struct NodeTensorPortMapping {
  std::vector<TensorPortMapping> input_tensor_ports;
  std::vector<TensorPortMapping> output_tensor_ports;
};

// The DispatchDelegateOptions is used to share information between the
// CompiledModel and the DispatchDelegate.
//
// Note: Since they're alway build together, this structure doesn't need to be
// ABI stable.
//
// This is an internal typed wrapper for a `LiteRtOpaqueOptions` node used to
// pass runtime-generated model metadata from `CompiledModel` to the dispatch
// delegate while delegates are created and applied.
//
// Unlike user-facing option builders such as `CpuOptions`, this object is
// itself an opaque transport node. `CompiledModel` creates it after model
// allocation and JIT metadata are available, temporarily appends it to
// `LiteRtOptionsT::options`, and the dispatch delegate retrieves it using
// `Discriminator()`.
//
// The producer and consumer are built together and the payload remains within
// the same runtime, so it may contain in-process pointers and handles and does
// not require a serialized or ABI-stable representation.
class DispatchDelegateOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static const char* Discriminator() { return "dispatch_delegate"; }

  // Get a non-owning view of the given opaque options if they are of the
  // correct derived type.
  static Expected<DispatchDelegateOptions> Create(OpaqueOptions& options);

  // Create a new owning view.
  static Expected<DispatchDelegateOptions> Create();

  // alloc_base ----------------------------------------------------------------

  // Alloc base is the address of the first byte of the flatbuffer model being
  // executed. This is relevant to backends with a compiled asset stored at the
  // back of the fb.

  // Set alloc base as a raw pointer.
  Expected<void> SetAllocBase(const void* alloc_base);

  // Get alloc base as a raw pointer.
  Expected<const void*> GetAllocBase();

  // alloc_base_fd -------------------------------------------------------------

  // Alloc base fd is simiilar to alloc base but it is a file descriptor to
  // assets stored externally.

  // Set alloc base fd.
  Expected<void> SetAllocBaseFd(int alloc_base_fd);

  // Get alloc base fd.
  Expected<int> GetAllocBaseFd();

  // Add an opaque executable handle for JIT
  Expected<void> AddExecHandle(absl::string_view name,
                               LiteRtJitExecutable handle);

  // Get the opaque executable handle for JIT
  Expected<LiteRtJitExecutable> GetExecHandle(absl::string_view name);

  // alloc_base_file_offset ----------------------------------------------------

  // Byte offset of alloc_base in the backing file when alloc_base_fd is used.
  Expected<void> SetAllocBaseFileOffset(size_t alloc_base_file_offset);

  // Get alloc base file offset.
  Expected<size_t> GetAllocBaseFileOffset();

  // alloc_base_size -----------------------------------------------------------

  // Size in bytes of the model rooted at alloc_base when alloc_base_fd is used.
  Expected<void> SetAllocBaseSize(size_t alloc_base_size);

  // Get alloc base size.
  Expected<size_t> GetAllocBaseSize();

  // Returns whether the file region metadata (offset or size) is populated.
  Expected<bool> HasAllocBaseFileRegion();

  // node_tensor_port_mappings ------------------------------------------------

  // Set the input and output tensor port mappings for a function/node sepcified
  // by `function_name`.
  Expected<void> SetNodeTensorPortMapping(
      absl::string_view function_name,
      std::vector<TensorPortMapping> input_tensor_ports,
      std::vector<TensorPortMapping> output_tensor_ports);

  // Get the input and output tensor port mappings for a function/node specified
  // by `function_name`.
  Expected<const NodeTensorPortMapping*> GetNodeTensorPortMapping(
      absl::string_view function_name);

  // function_to_signature_map ------------------------------------------------

  // Set the signature name associated with a specific function/node name.
  Expected<void> RegisterFunctionSignature(absl::string_view function_name,
                                           absl::string_view signature_name);

  // Get the signature name associated with a specific function/node name.
  Expected<std::string_view> GetFunctionSignature(
      absl::string_view function_name);
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_OPAQUE_OPTIONS_H_
