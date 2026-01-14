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

#include "litert/vendors/google_tensor/dispatch/dispatch_api_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

namespace litert::google_tensor {

namespace {

// Specifies a type of Thr ID.
enum class ThrIdType {
  kNode,  // The ID identifies a Thr node.
  kEdge,  // The ID identifies a Thr edge.
};

const char* absl_nonnull GetThrId(uint64_t id, ThrIdType type) {
  // Workaround for b/369144429.
  //
  // `std::unique_ptr` is used to maintain value pointer stability.
  using ThrIdCache =
      absl::flat_hash_map<uint64_t, absl_nonnull std::unique_ptr<std::string>>;

  static absl::NoDestructor<ThrIdCache> thr_node_ids;
  static absl::NoDestructor<ThrIdCache> thr_edge_ids;

  ThrIdCache* thr_ids;
  switch (type) {
    case ThrIdType::kNode:
      thr_ids = &(*thr_node_ids);
      break;
    case ThrIdType::kEdge:
      thr_ids = &(*thr_edge_ids);
      break;
  }

  if (auto iter = thr_ids->find(id); iter != thr_ids->end()) {
    return iter->second->c_str();
  }

  std::string thr_id;
  switch (type) {
    case ThrIdType::kNode:
      thr_id = "node_" + std::to_string(id);
      break;
    case ThrIdType::kEdge:
      thr_id = "edge_" + std::to_string(id);
      break;
  }

  auto [iter, _] = thr_ids->try_emplace(
      id, std::make_unique<std::string>(std::move(thr_id)));
  return iter->second->c_str();
}

}  // namespace

ThrNodeId ToThrNodeId(LiteRtDispatchNodeId node_id) {
  return GetThrId(node_id, ThrIdType::kNode);
}

ThrEdgeId ToThrEdgeId(LiteRtDispatchEdgeId edge_id) {
  return GetThrId(edge_id, ThrIdType::kEdge);
}

}  // namespace litert::google_tensor
