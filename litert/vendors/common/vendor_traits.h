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

  // Vendor-specific backend library name
  static constexpr char* kBackendLibraryName = "";

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
  static constexpr char* kVendorId = "Qualcomm";
  static constexpr uint32_t kCapabilities = kLiteRtDispatchCapabilitiesBasic;
  static constexpr bool kSupportsAsync = false;
  static constexpr bool kSupportsGraph = false;
  static constexpr char* kBackendLibraryName = "libQnnSystem.so";

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
  static constexpr auto kVendorId = "MediaTek";
  static constexpr uint32_t kCapabilities = kLiteRtDispatchCapabilitiesBasic;
  static constexpr bool kSupportsAsync = false;
  static constexpr bool kSupportsGraph = false;
  static constexpr auto kBackendLibraryName = "libneuron_adapter.so";

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
  static constexpr auto kVendorId = "Google";
  static constexpr uint32_t kCapabilities = kLiteRtDispatchCapabilitiesBasic |
                                            kLiteRtDispatchCapabilitiesAsync |
                                            kLiteRtDispatchCapabilitiesGraph;
  static constexpr bool kSupportsAsync = true;
  static constexpr bool kSupportsGraph = true;
  static constexpr char* kBackendLibraryName = "libsouthbound.so";

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

  // Additional graph API support
  static Expected<LiteRtDispatchGraph*> CreateGraph();
  static LiteRtStatus DestroyGraph(LiteRtDispatchGraph* graph);
  static LiteRtStatus AddNode(LiteRtDispatchGraph* graph,
                              LiteRtDispatchNodeId node_id,
                              LiteRtDispatchNodeType node_type);
  static LiteRtStatus AddEdge(LiteRtDispatchGraph* graph,
                              LiteRtDispatchEdgeId edge_id);
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