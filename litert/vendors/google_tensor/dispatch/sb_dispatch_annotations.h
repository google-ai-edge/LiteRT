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

#ifndef ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_SB_DISPATCH_ANNOTATIONS_H_
#define ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_SB_DISPATCH_ANNOTATIONS_H_

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::google_tensor {

// Annotations that are used to annotate the graph directives. This can only be
// used to annotate a graph.
class GraphDirectiveAnnotations {
 public:
  // Set to 1 to enable CPU cache coherency for all host CPU cacheable buffers.
  static constexpr absl::string_view kPreferCoherent = "prefer_coherent";
};

// Annotations that are used to annotate the dispatch directives. This can be
// used to annotate a graph or a node.
class DispatchDirectiveAnnotations {
 public:
  // The execution priority for a node or sub-graph.
  // - For first-party clients, this is a non-negative integer ranging from 0 to
  // 11, where 0 is the highest priority and 11 is the lowest priority. The
  // recommendation is to start with lowest priority and increase it as needed.
  // - For third-party clients, the value can be "high", "medium", "low".
  static constexpr absl::string_view kPriority = "tpu_priority";
  // If numeric value is provided, the device power state to use for the Edgetpu
  // accelerator for one or more Edgetpu kernel-nodes in the scope of the
  // directive.
  static constexpr absl::string_view kEdgetpuDevicePowerState =
      "edgetpu_device_power_state";
  // If numeric value is provided, the memory power state to use for the Edgetpu
  // accelerator for one or more Edgetpu kernel-nodes in the scope of the
  // directive.
  static constexpr absl::string_view kEdgetpuMemoryPowerState =
      "edgetpu_memory_power_state";
  // If set to 1, the firmware won't allow any other requests that are
  // within the same priority and QoS band to start before this inference
  // request finishes, which ensures the ordering.
  static constexpr absl::string_view kEdgetpuAtomicInference =
      "edgetpu_atomic_inference";
};
}  // namespace litert::google_tensor

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_SB_DISPATCH_ANNOTATIONS_H_
