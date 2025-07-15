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

// Example of how to migrate a vendor dispatch implementation to use the
// template framework

#include "litert/vendors/common/vendor_dispatch_base.h"
#include "litert/vendors/qualcomm/dispatch/qualcomm_dispatch_template.h"

// Example 1: Qualcomm dispatch using templates
namespace litert {
namespace vendors {
namespace qualcomm {

// All the common dispatch API functions are handled by the template.
// Just need to define the entry point:
DEFINE_VENDOR_DISPATCH_ENTRY_POINT(QualcommTag)

}  // namespace qualcomm
}  // namespace vendors
}  // namespace litert

// Example 2: MediaTek dispatch migration
namespace litert {
namespace vendors {
namespace mediatek {

// MediaTek-specific device context
class MediaTekDeviceContext : public VendorDeviceContext {
 public:
  // Implementation details...
};

// MediaTek-specific invocation context  
class MediaTekInvocationContext : public VendorInvocationContext {
 public:
  // Implementation details...
};

// Implement MediaTek trait methods
template <>
LiteRtStatus VendorTraits<MediaTekTag>::Initialize(const std::string& lib_dir) {
  // Load NeuronAdapter library
  return kLiteRtStatusOk;
}

template <>
std::string VendorTraits<MediaTekTag>::GetBuildId() {
  return "NeuronAdapter v4.0.0";
}

// ... other trait method implementations ...

// Define entry point
DEFINE_VENDOR_DISPATCH_ENTRY_POINT(MediaTekTag)

}  // namespace mediatek
}  // namespace vendors
}  // namespace litert

// Example 3: Google Tensor with graph support
namespace litert {
namespace vendors {
namespace google_tensor {

// Since Google Tensor supports graph API, we need to extend the base template
template <>
class VendorDispatch<GoogleTensorTag> : public VendorDispatch<GoogleTensorTag> {
 public:
  // Additional graph API functions
  static LiteRtStatus GraphCreate(LiteRtDispatchGraphHandle* graph_handle) {
    auto result = VendorTraits<GoogleTensorTag>::CreateGraph();
    if (!result.ok()) {
      return result.Error().Status();
    }
    *graph_handle = result.Value();
    return kLiteRtStatusOk;
  }
  
  static LiteRtStatus GraphDestroy(LiteRtDispatchGraphHandle graph_handle) {
    return VendorTraits<GoogleTensorTag>::DestroyGraph(graph_handle);
  }
  
  static LiteRtStatus AddNode(LiteRtDispatchGraphHandle graph_handle,
                             LiteRtDispatchNodeId node_id,
                             LiteRtDispatchNodeType node_type) {
    return VendorTraits<GoogleTensorTag>::AddNode(graph_handle, node_id, node_type);
  }
  
  // ... other graph API functions ...
  
  // Override GetGraphInterface to return the actual interface
  static const LiteRtDispatchGraphInterface* GetGraphInterface() {
    static const LiteRtDispatchGraphInterface graph_interface = {
      .graph_create = GraphCreate,
      .graph_destroy = GraphDestroy,
      .add_node = AddNode,
      // ... other functions ...
    };
    return &graph_interface;
  }
};

// Define entry point
DEFINE_VENDOR_DISPATCH_ENTRY_POINT(GoogleTensorTag)

}  // namespace google_tensor
}  // namespace vendors
}  // namespace litert

// Migration benefits:
// 1. Eliminates ~500 lines of boilerplate code per vendor
// 2. Ensures consistent error handling across vendors
// 3. Makes it easy to add new vendors
// 4. Centralizes common logic like parameter validation
// 5. Type-safe vendor-specific extensions