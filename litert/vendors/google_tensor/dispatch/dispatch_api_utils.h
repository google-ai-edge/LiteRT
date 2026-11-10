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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_UTILS_H_

#include <optional>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_dispatch_delegate_vendor_options.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

// A collection of utility functions for handling common patterns.

namespace litert::google_tensor {

// Returns the `ThrNodeId` corresponding to `node_id`.
//
// NOTE: the returned `ThrNodeId` has static storage duration.
ThrNodeId ToThrNodeId(LiteRtDispatchNodeId node_id);

// Returns the `ThrEdgeId` corresponding to `edge_id`.
//
// NOTE: the returned `ThrEdgeId` has static storage duration.
ThrEdgeId ToThrEdgeId(LiteRtDispatchEdgeId edge_id);

// Returns true if `port_index` is configured with a coherent cache policy in
// `coherency_map` for the given `function_signature`. Returns false if no
// matching entry is found or if required inputs are empty/null.
bool IsPortCoherent(int port_index,
                    absl::Span<const internal::TensorPortMapping> tensor_ports,
                    const TensorCoherencyMap* coherency_map,
                    std::optional<absl::string_view> function_signature);

// Annotates input and output graph edges with "cache_policy" according to
// the configured GoogleTensorOptions input/output coherency entries. If no
// coherency configuration is present or a port does not match, the edge is
// annotated with the default policy "disable_cmo".
LiteRtStatus AnnotateCoherencyEdges(LiteRtDispatchDeviceContext device_context,
                                    const char* absl_nullable function_name,
                                    int num_inputs, int num_outputs,
                                    LiteRtDispatchGraph graph);

// Extracts `DispatchDelegateVendorOptions` from `options` via the
// runtime context's opaque options chain and deserializes it from the TOML
// payload. Returns nullopt if any step in the options resolution chain fails,
// if no vendor options were registered, or if TOML parsing fails.
//
// Exposed for testing.
std::optional<litert::internal::DispatchDelegateVendorOptions>
FindDispatchDelegateVendorOptions(
    const LiteRtRuntimeContext* absl_nonnull runtime_context,
    LiteRtOptions absl_nonnull options);

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_UTILS_H_
