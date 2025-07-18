// Copyright 2025 Google LLC.
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

#ifndef LITERT_VENDORS_COMMON_VENDOR_TRAITS_H_
#define LITERT_VENDORS_COMMON_VENDOR_TRAITS_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/common/vendor_traits.h"

namespace litert::vendors {

// Forward declarations
class VendorDeviceContext;
class VendorInvocationContext;

// Tag types for each vendor
struct QualcommTag {};
struct MediaTekTag {};
struct GoogleTensorTag {};

// Base template for vendor traits - all specializations must provide these
// members
template <typename VendorTag>
struct VendorTraits {
  // Vendor identification
  static constexpr const char* kVendorId = "Unknown";

  // Supported capabilities
  static constexpr uint32_t kCapabilities = kLiteRtDispatchCapabilitiesNone;

  // Whether this vendor supports async execution
  static constexpr bool kSupportsAsync = false;

  // Whether this vendor supports graph API
  static constexpr bool kSupportsGraph = false;

  // Whether this vendor supports metrics collection
  static constexpr bool kSupportsMetrics = false;

  // Vendor-specific backend library name
  static constexpr const char* kBackendLibraryName = "";

  // Backend-specific types
  using BackendContext = void;
  using BackendBuffer = void;
  using BackendModel = void;

  // Initialize vendor backend
  static LiteRtStatus Initialize(const std::string& lib_dir) {
    return kLiteRtStatusErrorUnsupported;
  }

  // Get backend version/build ID
  static std::string GetBuildId() { return "Unknown"; }

  // Create device context
  static Expected<std::unique_ptr<VendorDeviceContext>> CreateDeviceContext(
      const LiteRtDispatchDeviceContext* device_context) {
    return Unexpected(kLiteRtStatusErrorUnsupported, "Not implemented");
  }

  // Register tensor buffer with backend
  static LiteRtStatus RegisterTensorBuffer(VendorDeviceContext* context,
                                           LiteRtTensorBuffer buffer,
                                           LiteRtTensorBufferHandle* handle) {
    return kLiteRtStatusErrorUnsupported;
  }

  // Unregister tensor buffer
  static LiteRtStatus UnregisterTensorBuffer(VendorDeviceContext* context,
                                             LiteRtTensorBufferHandle handle) {
    return kLiteRtStatusErrorUnsupported;
  }

  // Create invocation context from bytecode
  static Expected<std::unique_ptr<VendorInvocationContext>>
  CreateInvocationContext(VendorDeviceContext* device_context,
                          const void* exec_bytecode_ptr,
                          size_t exec_bytecode_size,
                          const char* function_name) {
    return Unexpected(kLiteRtStatusErrorUnsupported, "Not implemented");
  }
};

// Specialization for Qualcomm
template <>
struct VendorTraits<QualcommTag> {
  static constexpr const char* kVendorId = "Qualcomm";
  static constexpr uint32_t kCapabilities = kLiteRtDispatchCapabilitiesBasic;
  static constexpr bool kSupportsAsync = false;
  static constexpr bool kSupportsGraph = false;
  static constexpr bool kSupportsMetrics =
      true;  // Qualcomm has profiling support
  static constexpr const char* kBackendLibraryName = "libQnnSystem.so";

  // QNN-specific types
  struct QnnBackend;
  struct QnnContext;
  struct QnnMemHandle;

  using BackendContext = QnnContext;
  using BackendBuffer = QnnMemHandle;
  using BackendModel = void*;  // QNN Graph handle

  static LiteRtStatus Initialize(const std::string& lib_dir);
  static std::string GetBuildId();
  static Expected<std::unique_ptr<VendorDeviceContext>> CreateDeviceContext(
      const LiteRtDispatchDeviceContext* device_context);
  static LiteRtStatus RegisterTensorBuffer(VendorDeviceContext* context,
                                           LiteRtTensorBuffer buffer,
                                           LiteRtTensorBufferHandle* handle);
  static LiteRtStatus UnregisterTensorBuffer(VendorDeviceContext* context,
                                             LiteRtTensorBufferHandle handle);
  static Expected<std::unique_ptr<VendorInvocationContext>>
  CreateInvocationContext(VendorDeviceContext* device_context,
                          const void* exec_bytecode_ptr,
                          size_t exec_bytecode_size, const char* function_name);
};

// Specialization for MediaTek
template <>
struct VendorTraits<MediaTekTag> {
  static constexpr const char* kVendorId = "MediaTek";
  static constexpr uint32_t kCapabilities = kLiteRtDispatchCapabilitiesBasic;
  static constexpr bool kSupportsAsync = false;
  static constexpr bool kSupportsGraph = false;
  static constexpr bool kSupportsMetrics =
      false;  // MediaTek doesn't have metrics support yet
  static constexpr const char* kBackendLibraryName = "libneuron_adapter.so";

  // NeuronAdapter-specific types
  struct NeuronCompilation;
  struct NeuronExecution;
  struct NeuronMemory;

  using BackendContext = NeuronCompilation;
  using BackendBuffer = NeuronMemory;
  using BackendModel = NeuronExecution;

  static LiteRtStatus Initialize(const std::string& lib_dir);
  static std::string GetBuildId();
  static Expected<std::unique_ptr<VendorDeviceContext>> CreateDeviceContext(
      const LiteRtDispatchDeviceContext* device_context);
  static LiteRtStatus RegisterTensorBuffer(VendorDeviceContext* context,
                                           LiteRtTensorBuffer buffer,
                                           LiteRtTensorBufferHandle* handle);
  static LiteRtStatus UnregisterTensorBuffer(VendorDeviceContext* context,
                                             LiteRtTensorBufferHandle handle);
  static Expected<std::unique_ptr<VendorInvocationContext>>
  CreateInvocationContext(VendorDeviceContext* device_context,
                          const void* exec_bytecode_ptr,
                          size_t exec_bytecode_size, const char* function_name);
};

// Specialization for Google Tensor
template <>
struct VendorTraits<GoogleTensorTag> {
  static constexpr const char* kVendorId = "Google";
  static constexpr uint32_t kCapabilities = kLiteRtDispatchCapabilitiesBasic |
                                            kLiteRtDispatchCapabilitiesAsync |
                                            kLiteRtDispatchCapabilitiesGraph;
  static constexpr bool kSupportsAsync = true;
  static constexpr bool kSupportsGraph = true;
  static constexpr bool kSupportsMetrics =
      true;  // Google Tensor has full metrics support
  static constexpr const char* kBackendLibraryName = "libsouthbound.so";

  // Southbound-specific types
  struct ThrContext;
  struct ThrGraph;
  struct ThrBuffer;

  using BackendContext = ThrContext;
  using BackendBuffer = ThrBuffer;
  using BackendModel = ThrGraph;
  static LiteRtStatus Initialize(const std::string& lib_dir);
  static std::string GetBuildId();
  static Expected<std::unique_ptr<VendorDeviceContext>> CreateDeviceContext(
      const LiteRtDispatchDeviceContext* device_context);
  static LiteRtStatus RegisterTensorBuffer(VendorDeviceContext* context,
                                           LiteRtTensorBuffer buffer,
                                           LiteRtTensorBufferHandle* handle);
  static LiteRtStatus UnregisterTensorBuffer(VendorDeviceContext* context,
                                             LiteRtTensorBufferHandle handle);
  static Expected<std::unique_ptr<VendorInvocationContext>>
  CreateInvocationContext(VendorDeviceContext* device_context,
                          const void* exec_bytecode_ptr,
                          size_t exec_bytecode_size, const char* function_name);

  // Async interface methods (required since kSupportsAsync = true)
  static LiteRtStatus AttachInputEvent(
      LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
      LiteRtEvent input_event);
  static LiteRtStatus InvokeAsync(
      LiteRtDispatchInvocationContext invocation_context, int num_output_events,
      LiteRtEvent* output_events);

  // Graph interface methods (required since kSupportsGraph = true)
  static LiteRtStatus GraphCreate(LiteRtDispatchDeviceContext device_context,
                                  LiteRtDispatchGraph* graph);
  static LiteRtStatus GraphDestroy(LiteRtDispatchGraph graph);
  static LiteRtStatus AddNode(LiteRtDispatchGraph graph,
                              LiteRtDispatchNodeId node_id,
                              LiteRtDispatchNodeType node_type);
  static LiteRtStatus AddEdge(LiteRtDispatchGraph graph,
                              LiteRtDispatchEdgeId edge_id);
  static LiteRtStatus ConnectNodeInput(LiteRtDispatchGraph graph,
                                       LiteRtDispatchNodeId node_id,
                                       int input_index,
                                       LiteRtDispatchEdgeId edge_id);
  static LiteRtStatus ConnectNodeOutput(LiteRtDispatchGraph graph,
                                        LiteRtDispatchNodeId node_id,
                                        int output_index,
                                        LiteRtDispatchEdgeId edge_id);
  static LiteRtStatus ConnectGraphInput(LiteRtDispatchGraph graph,
                                        int graph_input_index,
                                        LiteRtDispatchEdgeId edge_id);
  static LiteRtStatus ConnectGraphOutput(LiteRtDispatchGraph graph,
                                         int graph_output_index,
                                         LiteRtDispatchEdgeId edge_id);
  static LiteRtStatus LoadExecutable(
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType type, const LiteRtMemBuffer* bytecode,
      LiteRtDispatchExecutableHandle* exec_handle);
  static LiteRtStatus UnloadExecutable(
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableHandle exec_handle);
  static LiteRtStatus AssignNodeFunction(
      LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
      LiteRtDispatchExecutableHandle exec_handle, const char* function_name);
  static LiteRtStatus AnnotateGraph(LiteRtDispatchGraph graph, const char* key,
                                    const char* value);
  static LiteRtStatus AnnotateNode(LiteRtDispatchGraph graph,
                                   LiteRtDispatchNodeId node_id,
                                   const char* key, const char* value);
  static LiteRtStatus AnnotateEdge(LiteRtDispatchGraph graph,
                                   LiteRtDispatchEdgeId edge_id,
                                   const char* key, const char* value);
  static LiteRtStatus InvocationContextCreateFromGraph(
      LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
      LiteRtDispatchInvocationContext* invocation_context);
};

// Helper to check vendor capabilities at compile time
template <typename VendorTag>
constexpr bool SupportsAsync() {
  return VendorTraits<VendorTag>::kSupportsAsync;
}

template <typename VendorTag>
constexpr bool SupportsGraph() {
  return VendorTraits<VendorTag>::kSupportsGraph;
}

template <typename VendorTag>
constexpr uint32_t GetCapabilities() {
  return VendorTraits<VendorTag>::kCapabilities;
}

template <typename VendorTag>
constexpr bool SupportsMetrics() {
  return VendorTraits<VendorTag>::kSupportsMetrics;
}

// Common buffer requirements structure
struct BufferRequirements {
  std::vector<LiteRtTensorBufferType> supported_types;
  size_t buffer_size;
  std::vector<uint32_t> strides;

  BufferRequirements() : buffer_size(0) {}

  // Convert to C API tensor buffer requirements
  Expected<LiteRtTensorBufferRequirements> ToLiteRtRequirements() const;
};

}  // namespace litert::vendors

#endif  // LITERT_VENDORS_COMMON_VENDOR_TRAITS_H_