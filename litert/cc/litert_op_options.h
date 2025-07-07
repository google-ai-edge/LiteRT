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

#ifndef ODML_LITERT_LITERT_CC_LITERT_OP_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_LITERT_OP_OPTIONS_H_

#include <cstdint>
#include <optional>
#include <type_traits>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

struct OpOptions {
  virtual LiteRtStatus InitFromOp(LiteRtOp op) = 0;
  virtual ~OpOptions() = default;
};

using ActivationFunction = uint32_t;
enum ActivationFunctionType : uint32_t {
  kActivationFunctionTypeNone = 0,
  kActivationFunctionTypeRelu = 1,
  kActivationFunctionTypeReluN1To1 = 2,
  kActivationFunctionTypeRelu6 = 3,
  kActivationFunctionTypeTanh = 4,
  kActivationFunctionTypeSignBit = 5,
  kActivationFunctionTypeMin = kActivationFunctionTypeNone,
  kActivationFunctionTypeMax = kActivationFunctionTypeSignBit,
};

// Struct to hold LiteRt composite ops.
struct CompositeOptions : public OpOptions {
  // Name for special composites representing manual partitions.
  static constexpr absl::string_view kNpuCall = "odml.npu_call";
  static constexpr absl::string_view kRmsNorm = "odml.rms_norm";

  // The root op.
  LiteRtOp op;
  // Decomposition subgraph.
  int subgraph;
  // The name of the composite op (stored in model).
  absl::string_view name;
  // The version of the composite op.
  int32_t version;
  // The attributes of the composite op.
  std::optional<flexbuffers::Map> attributes_map;

  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

struct RmsNormOpts : public CompositeOptions {
  // The epsilon composite attribute of the RMS norm.
  float epsilon;

  LiteRtStatus InitFromOp(LiteRtOp litert_op) override;
};

// Struct to hold LiteRt Add op.
struct AddOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt BatchMatmul op.
struct BatchMatmulOptions : public OpOptions {
  LiteRtOp op;
  bool adj_x;
  bool adj_y;
  bool asymmetric_quantize_input;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Returns the composite info for the given op if it is a composite op.
template <typename OptionsT>
Expected<OptionsT> GetOptionsAs(LiteRtOp op) {
  if constexpr (std::is_same_v<OptionsT, CompositeOptions>) {
    CompositeOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, RmsNormOpts>) {
    RmsNormOpts options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, AddOptions>) {
    AddOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, BatchMatmulOptions>) {
    BatchMatmulOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else {
    // TODO: Add more as needed.
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_OP_OPTIONS_H_
