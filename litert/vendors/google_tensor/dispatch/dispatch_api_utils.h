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

#include "litert/vendors/c/litert_dispatch.h"
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

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_UTILS_H_
