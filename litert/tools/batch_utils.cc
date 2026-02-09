// Copyright 2024 Google LLC.
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

#include "litert/tools/batch_utils.h"

#include <cstdint>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"

namespace litert::tools {

namespace {

inline static absl::flat_hash_set<LiteRtOpCode>
GetPotentiallyShapeAlteringOps() {
  return {
      kLiteRtOpCodeTflTranspose,      kLiteRtOpCodeTflReshape,
      kLiteRtOpCodeTflSqueeze,        kLiteRtOpCodeTflExpandDims,
      kLiteRtOpCodeTflPack,           kLiteRtOpCodeTflUnpack,
      kLiteRtOpCodeTflSplit,          kLiteRtOpCodeTflSplitV,
      kLiteRtOpCodeTflSlice,          kLiteRtOpCodeTflStridedSlice,
      kLiteRtOpCodeTflTile,           kLiteRtOpCodeTflBroadcastTo,
      kLiteRtOpCodeTflGather,         kLiteRtOpCodeTflGatherNd,
      kLiteRtOpCodeTflConcatenation,  kLiteRtOpCodeTflSpaceToDepth,
      kLiteRtOpCodeTflDepthToSpace,   kLiteRtOpCodeTflBatchToSpaceNd,
      kLiteRtOpCodeTflSpaceToBatchNd, kLiteRtOpCodeTflTransposeConv,
  };
}

bool IsDynamicAtIndex0(const LiteRtTensorT& tensor) {
  const auto& type = tensor.Type();
  if (type.first != kLiteRtRankedTensorType) return false;
  const auto& ranked = type.second.ranked_tensor_type;
  return ranked.layout.rank > 0 && ranked.layout.dimensions[0] == -1;
}

LiteRtStatus ValidateTensor(const LiteRtTensorT& tensor) {
  if (tensor.Type().first != kLiteRtRankedTensorType) return kLiteRtStatusOk;
  const auto& ranked = tensor.Type().second.ranked_tensor_type;
  for (int i = 1; i < ranked.layout.rank; ++i) {
    if (ranked.layout.dimensions[i] == -1) {
      ABSL_LOG(ERROR) << absl::StreamFormat(
          "Tensor '%s' has dynamic dimension at index %d. Only the 0-th "
          "dimension can be dynamic.",
          tensor.Name().data(), i);
      return kLiteRtStatusErrorInvalidArgument;
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ValidateOp(const LiteRtOpT& op) {
  if (!GetPotentiallyShapeAlteringOps().contains(op.OpCode()))
    return kLiteRtStatusOk;

  bool input_has_dynamic_batch = false;
  for (const auto& input : op.Inputs()) {
    if (IsDynamicAtIndex0(*input)) {
      input_has_dynamic_batch = true;
      break;
    }
  }

  if (input_has_dynamic_batch) {
    for (const auto& output : op.Outputs()) {
      if (output->Type().first == kLiteRtRankedTensorType) {
        if (!IsDynamicAtIndex0(*output)) {
          ABSL_LOG(ERROR) << absl::StreamFormat(
              "Shape-altering operation '%d' has dynamic batch input but the "
              "dynamic dimension is missing or moved in the output.",
              static_cast<int>(op.OpCode()));
          return kLiteRtStatusErrorInvalidArgument;
        }
      }
    }
  }
  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus ValidateModelForBatchFix(const LiteRtModelT& model) {
  LiteRtStatus status = kLiteRtStatusOk;
  ::ForEachIr(model, [&](LiteRtSubgraph subgraph) {
    if (status != kLiteRtStatusOk) return;

    // Validate all tensors in subgraph.
    for (const auto& tensor : subgraph->Tensors()) {
      status = ValidateTensor(*tensor);
      if (status != kLiteRtStatusOk) return;
    }

    // Validate all ops in subgraph.
    for (const auto& op : subgraph->Ops()) {
      status = ValidateOp(*op);
      if (status != kLiteRtStatusOk) return;
    }
  });
  return status;
}

void FixBatchDimension(LiteRtModelT& model, int32_t batch_size) {
  ::ForEachIr(model, [&](LiteRtSubgraph subgraph) {
    for (auto& tensor : subgraph->Tensors()) {
      if (tensor->Type().first == kLiteRtRankedTensorType) {
        auto type = tensor->Type();
        auto& ranked = type.second.ranked_tensor_type;
        if (ranked.layout.rank > 0) {
          ranked.layout.dimensions[0] = batch_size;
          tensor->SetType(type);
        }
      }
    }
  });
}

}  // namespace litert::tools
