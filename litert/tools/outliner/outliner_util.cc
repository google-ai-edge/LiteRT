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

#include "litert/tools/outliner/outliner_util.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/plugin/algo.h"
#include "litert/core/model/model.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::tools {

namespace {

void ReindexModel(LiteRtModelT& model) {
  for (auto& sg : model.Subgraphs()) {
    for (size_t i = 0; i < sg->Ops().size(); ++i) {
      sg->Ops()[i]->SetOpIndex(i);
    }
    for (size_t i = 0; i < sg->Tensors().size(); ++i) {
      sg->Tensors()[i]->SetTensorIndex(i);
    }
  }
}

using ::LiteRtModelT;
using ::LiteRtOpT;
using ::LiteRtSubgraphT;
using ::LiteRtTensorT;

// Find tensors by name in a subgraph.
absl::flat_hash_set<const LiteRtTensorT*> FindTensors(
    const LiteRtSubgraphT& subgraph, const std::vector<std::string>& names) {
  absl::flat_hash_set<const LiteRtTensorT*> found;
  for (const auto& tensor : subgraph.Tensors()) {
    for (const auto& name : names) {
      if (tensor->Name() == name) {
        found.insert(tensor);
      }
    }
  }
  return found;
}

// Perform a traversal to find all ops between start and end tensors.
litert::Expected<std::vector<LiteRtOpT*>> IdentifyAndValidateSubgraphOps(
    LiteRtSubgraphT& subgraph,
    const absl::flat_hash_set<const LiteRtTensorT*>& starts,
    const absl::flat_hash_set<const LiteRtTensorT*>& ends) {
  absl::flat_hash_set<LiteRtOpT*> reachable_from_starts_ops;
  absl::flat_hash_set<const LiteRtTensorT*> reachable_tensors = starts;
  std::vector<const LiteRtTensorT*> queue(starts.begin(), starts.end());

  // 1. Forward pass: find everything reachable from starts.
  size_t head = 0;
  while (head < queue.size()) {
    const auto* tensor = queue[head++];
    if (ends.contains(tensor)) continue;

    for (auto* user_op : tensor->Users()) {
      if (reachable_from_starts_ops.insert(user_op).second) {
        for (const auto* output_tensor : user_op->Outputs()) {
          if (reachable_tensors.insert(output_tensor).second) {
            queue.push_back(output_tensor);
          }
        }
      }
    }
  }

  // 2. Backward pass: find everything that can reach ends.
  absl::flat_hash_set<LiteRtOpT*> can_reach_ends_ops;
  absl::flat_hash_set<const LiteRtTensorT*> can_reach_ends_tensors = ends;
  std::vector<const LiteRtTensorT*> back_queue(ends.begin(), ends.end());

  head = 0;
  while (head < back_queue.size()) {
    const auto* tensor = back_queue[head++];
    if (starts.contains(tensor)) continue;

    if (tensor->DefiningOp()) {
      auto* defining_op = tensor->DefiningOp();
      if (can_reach_ends_ops.insert(defining_op).second) {
        for (const auto* input_tensor : defining_op->Inputs()) {
          if (input_tensor &&
              can_reach_ends_tensors.insert(input_tensor).second) {
            back_queue.push_back(input_tensor);
          }
        }
      }
    }
  }

  // 3. Intersection: Ops that are both reachable from starts AND can reach
  // ends.
  std::vector<LiteRtOpT*> final_ops;
  for (auto* op : subgraph.Ops()) {
    if (reachable_from_starts_ops.contains(op) &&
        can_reach_ends_ops.contains(op)) {
      final_ops.push_back(op);
    }
  }

  if (final_ops.empty()) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "No ops found between boundaries");
  }

  return final_ops;
}

}  // namespace

absl::Status OutlineSubgraph(LiteRtModelT& model, size_t subgraph_index,
                             const OutlinerOptions& options) {
  if (subgraph_index >= model.NumSubgraphs()) {
    return absl::InvalidArgumentError("Invalid subgraph index");
  }

  auto& main_sg = *model.Subgraphs()[subgraph_index];
  auto starts = FindTensors(main_sg, options.start_tensors);
  auto ends = FindTensors(main_sg, options.end_tensors);

  if (starts.empty() || ends.empty()) {
    return absl::NotFoundError("Could not find start or end tensors");
  }

  auto ops_to_outline_res =
      IdentifyAndValidateSubgraphOps(main_sg, starts, ends);
  if (!ops_to_outline_res) {
    return absl::InternalError(ops_to_outline_res.Error().Message());
  }
  auto ops_to_outline_raw = std::move(*ops_to_outline_res);

  std::cout << "Identified " << ops_to_outline_raw.size()
            << " operations to outline.\n";

  std::vector<LiteRtOp> ops_to_outline;
  for (auto* op : ops_to_outline_raw) {
    ops_to_outline.push_back(op);
  }

  // 1. Create decomposition subgraph.
  auto& decomp_sg = model.EmplaceSubgraph();
  int decomp_sg_idx = model.NumSubgraphs() - 1;

  // 2. Use OutlinePartition to do the heavy lifting of slicing and re-wiring.
  LiteRtOp custom_op =
      litert::internal::OutlinePartition(main_sg, &decomp_sg, ops_to_outline);

  // 3. Transform into a StableHLO Composite Op and final state cleanup.
  custom_op->SetOpCode(kLiteRtOpCodeShloComposite);
  custom_op->ClearCustomOptions();
  custom_op->SetCustomCode("");
  litert::internal::ClearTflOptions(*custom_op);

  auto op_codes = litert::internal::TakeTflOpCodes(model);
  int32_t composite_tfl_idx = -1;
  for (int32_t i = 0; i < (int32_t)op_codes.size(); ++i) {
    if (op_codes[i]->builtin_code ==
        tflite::BuiltinOperator_STABLEHLO_COMPOSITE) {
      composite_tfl_idx = i;
      break;
    }
  }
  if (composite_tfl_idx == -1) {
    composite_tfl_idx = op_codes.size();
    auto new_op_code = std::make_unique<tflite::OperatorCodeT>();
    new_op_code->builtin_code = tflite::BuiltinOperator_STABLEHLO_COMPOSITE;
    new_op_code->deprecated_builtin_code =
        (int8_t)tflite::BuiltinOperator_STABLEHLO_COMPOSITE;
    op_codes.push_back(std::move(new_op_code));
  }
  litert::internal::SetTflOpCodes(model, std::move(op_codes));
  litert::internal::SetTflOpCodeInd(*custom_op, composite_tfl_idx);

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    for (const auto& [k, v] : options.attributes) {
      fbb.String(k.c_str(), v);
    }
  });
  fbb.Finish();

  auto native_opts = std::make_unique<tflite::StableHLOCompositeOptionsT>();
  native_opts->name = options.composite_name;
  native_opts->decomposition_subgraph_index = decomp_sg_idx;
  native_opts->composite_attributes.assign(fbb.GetBuffer().begin(),
                                           fbb.GetBuffer().end());

  tflite::BuiltinOptions2Union tfl_composite_opts;
  tfl_composite_opts.type = tflite::BuiltinOptions2_StableHLOCompositeOptions;
  tfl_composite_opts.value = native_opts.release();
  litert::internal::SetTflOptions2(*custom_op, std::move(tfl_composite_opts));

  ReindexModel(model);

  return absl::OkStatus();
}

}  // namespace litert::tools
