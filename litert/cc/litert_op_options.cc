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

#include "litert/cc/litert_op_options.h"

#include <cstdint>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_macros.h"

namespace litert {

LiteRtStatus CompositeOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeShloComposite) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const char* op_name;
  LITERT_RETURN_IF_ERROR(LiteRtGetSHLOCompositeOpName(op, &op_name));
  name = op_name;

  LITERT_RETURN_IF_ERROR(
      LiteRtGetSHLOCompositeOpDecompositionSubgraphIndex(op, &subgraph));

  LITERT_RETURN_IF_ERROR(LiteRtGetSHLOCompositeOpVersion(op, &version));

  const uint8_t* impl_attributes = nullptr;
  int32_t impl_attributes_size = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetSHLOCompositeOpAttributes(
      op, &impl_attributes, &impl_attributes_size));

  if (impl_attributes_size > 0) {
    attributes_map =
        flexbuffers::GetRoot(impl_attributes, impl_attributes_size).AsMap();
  }
  this->op = op;

  return kLiteRtStatusOk;
}

LiteRtStatus RmsNormOpts::InitFromOp(LiteRtOp litert_op) {
  LITERT_RETURN_IF_ERROR(CompositeOptions::InitFromOp(litert_op));
  if (!attributes_map.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  constexpr char kEpsilonKey[] = "epsilon";
  flexbuffers::Reference raw_epsilon = attributes_map.value()[kEpsilonKey];
  if (raw_epsilon.IsNull()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  epsilon  = raw_epsilon.AsFloat();
  return kLiteRtStatusOk;
}
}  // namespace litert
