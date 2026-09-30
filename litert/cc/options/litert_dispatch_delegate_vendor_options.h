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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_DISPATCH_DELEGATE_VENDOR_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_DISPATCH_DELEGATE_VENDOR_OPTIONS_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert {

// Mapping between a named model tensor on a delegated node/subgraph and its
// port index.
struct TensorPortMapping {
  std::string opaque_tensor_name;
  int port_index = -1;
};

// Input and output tensor port mappings for a single delegated function/node.
struct NodeTensorPortMapping {
  std::vector<TensorPortMapping> input_tensor_ports;
  std::vector<TensorPortMapping> output_tensor_ports;
};

// Holds common model metadata constructed at the LiteRT runtime level
// passed across the boundary down to the Southbound vendor dispatch layer.
// Because it is internal between the Dispatch Delegate and vendor plugins
// (and not accessible via the Northbound client interface), it is defined
// directly as a C++ class.
class DispatchDelegateVendorOptions {
 public:
  static constexpr const char* kIdentifier = "dispatch_delegate_vendor_options";

  static const char* Discriminator() { return kIdentifier; }

  Expected<void> SetNodeTensorPortMapping(
      absl::string_view function_name,
      std::vector<TensorPortMapping> input_tensor_ports,
      std::vector<TensorPortMapping> output_tensor_ports) {
    auto [it, inserted] = node_tensor_port_mappings_.emplace(
        function_name,
        NodeTensorPortMapping{
            /*input_tensor_ports=*/std::move(input_tensor_ports),
            /*output_tensor_ports=*/std::move(output_tensor_ports),
        });
    if (!inserted) {
      return Unexpected(kLiteRtStatusErrorAlreadyExists,
                        "Tensor port mapping already set for function");
    }
    return {};
  }

  const NodeTensorPortMapping* GetNodeTensorPortMapping(
      absl::string_view function_name) const {
    auto it = node_tensor_port_mappings_.find(function_name);
    if (it == node_tensor_port_mappings_.end()) {
      return nullptr;
    }
    return &it->second;
  }

 private:
  // Mapping of a function name to its input and output tensor ports.
  absl::flat_hash_map<std::string, NodeTensorPortMapping>
      node_tensor_port_mappings_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_DISPATCH_DELEGATE_VENDOR_OPTIONS_H_
