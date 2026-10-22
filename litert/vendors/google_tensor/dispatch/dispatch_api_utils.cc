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

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/optional_ref.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_dispatch_delegate_vendor_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_macros.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
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

std::optional<litert::internal::DispatchDelegateVendorOptions>
FindDispatchDelegateVendorOptions(
    const LiteRtRuntimeContext* absl_nonnull runtime_context,
    LiteRtOptions absl_nonnull options) {
  if (runtime_context == nullptr || options == nullptr) {
    return std::nullopt;
  }
  LiteRtOpaqueOptions opaque_opts = nullptr;
  if (runtime_context->get_opaque_options(options, &opaque_opts) !=
          kLiteRtStatusOk ||
      opaque_opts == nullptr) {
    return std::nullopt;
  }
  void* payload_data = nullptr;
  if (runtime_context->find_opaque_options_data(
          opaque_opts,
          litert::internal::DispatchDelegateVendorOptions::Discriminator(),
          &payload_data) != kLiteRtStatusOk ||
      payload_data == nullptr) {
    return std::nullopt;
  }
  auto vendor_options =
      litert::internal::DispatchDelegateVendorOptions::CreateFromToml(
          static_cast<const char*>(payload_data));
  if (!vendor_options) {
    LITERT_LOG(LITERT_ERROR,
               "Failed to parse DispatchDelegateVendorOptions TOML: %s",
               vendor_options.Error().Message().c_str());
    return std::nullopt;
  }
  return std::move(*vendor_options);
}

ThrNodeId ToThrNodeId(LiteRtDispatchNodeId node_id) {
  return GetThrId(node_id, ThrIdType::kNode);
}

ThrEdgeId ToThrEdgeId(LiteRtDispatchEdgeId edge_id) {
  return GetThrId(edge_id, ThrIdType::kEdge);
}

bool IsPortCoherent(
    int port_index,
    absl::Span<const litert::internal::TensorPortMapping> tensor_ports,
    const TensorCoherencyMap* coherency_map,
    std::optional<absl::string_view> function_signature) {
  if (!function_signature.has_value() || function_signature->empty() ||
      tensor_ports.empty() || coherency_map == nullptr) {
    return false;
  }
  auto function_signature_str = std::string(*function_signature);
  for (const litert::internal::TensorPortMapping& port_map : tensor_ports) {
    if (port_map.port_index == port_index) {
      auto it = coherency_map->find(
          std::make_pair(function_signature_str, port_map.tensor_name));
      if (it != coherency_map->end()) {
        return it->second;
      }
      return false;
    }
  }
  return false;
}

LiteRtStatus AnnotateCoherencyEdges(LiteRtDispatchDeviceContext device_context,
                                    const char* absl_nullable function_name,
                                    int num_inputs, int num_outputs,
                                    LiteRtDispatchGraph graph) {
  GT_LOG_RETURN_IF_NULL(graph);
  GT_LOG_RETURN_IF_NULL(device_context);

  // Retrieve input and output coherency maps from the GoogleTensorOptions.
  const TensorCoherencyMap* input_coherency_map = nullptr;
  const TensorCoherencyMap* output_coherency_map = nullptr;
  const absl::optional_ref<
      const LiteRtDispatchDeviceContextT::GoogleTensorOptionsData>
      gt_options = device_context->GetGoogleTensorOptions();
  if (gt_options.has_value()) {
    if (!gt_options->input_coherency_map.empty()) {
      input_coherency_map = &gt_options->input_coherency_map;
    }
    if (!gt_options->output_coherency_map.empty()) {
      output_coherency_map = &gt_options->output_coherency_map;
    }
  }

  // Retrieve the function signature and tensor port mappings from the
  // DispatchDelegateVendorOptions if we have non empty coherency maps and a
  // function name.
  std::optional<absl::string_view> function_signature;
  absl::Span<const litert::internal::TensorPortMapping> input_tensor_ports;
  absl::Span<const litert::internal::TensorPortMapping> output_tensor_ports;
  if ((input_coherency_map != nullptr || output_coherency_map != nullptr) &&
      function_name != nullptr) {
    const absl::optional_ref<
        const litert::internal::DispatchDelegateVendorOptions>
        vendor_options = device_context->GetDispatchDelegateVendorOptions();
    if (vendor_options.has_value()) {
      for (const litert::internal::FunctionMapping& mapping :
           vendor_options->FunctionMappings()) {
        if (mapping.function_name == function_name) {
          function_signature = mapping.signature_name;
          input_tensor_ports = mapping.input_tensor_ports;
          output_tensor_ports = mapping.output_tensor_ports;
          break;
        }
      }
    }
  }

  // Annotate every single input edge: [0 .. num_inputs - 1]
  for (int input_index = 0; input_index < num_inputs; ++input_index) {
    LiteRtDispatchEdgeId edge_id;
    LITERT_RETURN_IF_ERROR(graph->GetInputEdgeId(input_index, edge_id));
    const bool is_coherent =
        IsPortCoherent(input_index, input_tensor_ports, input_coherency_map,
                       function_signature);
    const char* policy = is_coherent ? "coherent" : "disable_cmo";
    LITERT_RETURN_IF_ERROR(
        graph->AnnotateEdge(edge_id, "cache_policy", policy));
    LITERT_LOG(LITERT_DEBUG,
               "Annotated input port %d (edge %" PRIu64
               ") with cache_policy=%s",
               input_index, edge_id, policy);
  }

  // Annotate every single output edge:
  //   [num_inputs .. num_inputs + num_outputs - 1]
  for (int output_index = 0; output_index < num_outputs; ++output_index) {
    LiteRtDispatchEdgeId edge_id;
    LITERT_RETURN_IF_ERROR(graph->GetOutputEdgeId(output_index, edge_id));
    const bool is_coherent =
        IsPortCoherent(output_index, output_tensor_ports, output_coherency_map,
                       function_signature);
    const char* policy = is_coherent ? "coherent" : "disable_cmo";
    LITERT_RETURN_IF_ERROR(
        graph->AnnotateEdge(edge_id, "cache_policy", policy));
    LITERT_LOG(LITERT_DEBUG,
               "Annotated output port %d (edge %" PRIu64
               ") with cache_policy=%s",
               output_index, edge_id, policy);
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::google_tensor
