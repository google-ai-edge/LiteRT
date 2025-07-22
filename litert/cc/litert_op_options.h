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
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

enum TfliteTensorType : uint32_t {
  TensorType_FLOAT32 = 0,
  TensorType_FLOAT16 = 1,
  TensorType_INT32 = 2,
  TensorType_UINT8 = 3,
  TensorType_INT64 = 4,
  TensorType_STRING = 5,
  TensorType_BOOL = 6,
  TensorType_INT16 = 7,
  TensorType_COMPLEX64 = 8,
  TensorType_INT8 = 9,
  TensorType_FLOAT64 = 10,
  TensorType_COMPLEX128 = 11,
  TensorType_UINT64 = 12,
  TensorType_RESOURCE = 13,
  TensorType_VARIANT = 14,
  TensorType_UINT32 = 15,
  TensorType_UINT16 = 16,
  TensorType_INT4 = 17,
  TensorType_BFLOAT16 = 18,
  TensorType_MIN = TensorType_FLOAT32,
  TensorType_MAX = TensorType_BFLOAT16
};

inline LiteRtElementType GetElementType(uint32_t tflite_element_type) {
  switch (tflite_element_type) {
    case TensorType_FLOAT32:
      return kLiteRtElementTypeFloat32;
    case TensorType_FLOAT16:
      return kLiteRtElementTypeFloat16;
    case TensorType_INT32:
      return kLiteRtElementTypeInt32;
    case TensorType_UINT8:
      return kLiteRtElementTypeUInt8;
    case TensorType_INT64:
      return kLiteRtElementTypeInt64;
    case TensorType_STRING:
      return kLiteRtElementTypeTfString;
    case TensorType_BOOL:
      return kLiteRtElementTypeBool;
    case TensorType_INT16:
      return kLiteRtElementTypeInt16;
    case TensorType_COMPLEX64:
      return kLiteRtElementTypeComplex64;
    case TensorType_INT8:
      return kLiteRtElementTypeInt8;
    case TensorType_FLOAT64:
      return kLiteRtElementTypeFloat64;
    case TensorType_COMPLEX128:
      return kLiteRtElementTypeComplex128;
    case TensorType_UINT64:
      return kLiteRtElementTypeUInt64;
    case TensorType_RESOURCE:
      return kLiteRtElementTypeTfResource;
    case TensorType_VARIANT:
      return kLiteRtElementTypeTfVariant;
    case TensorType_UINT32:
      return kLiteRtElementTypeUInt32;
    case TensorType_UINT16:
      return kLiteRtElementTypeUInt16;
    case TensorType_INT4:
      return kLiteRtElementTypeInt4;
    case TensorType_BFLOAT16:
      return kLiteRtElementTypeBFloat16;
    default:
      return kLiteRtElementTypeNone;
  }
};

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


using FullyConnectedOptionsWeightsFormat = uint32_t;
enum FullyConnectedOptionsWeightsFormatType : uint32_t {
  kFullyConnectedOptionsWeightsFormatDefault = 0,
  kFullyConnectedOptionsWeightsFormatShuffled4x16Int8 = 1,
  kFullyConnectedOptionsWeightsFormatMin =
      kFullyConnectedOptionsWeightsFormatDefault,
  kFullyConnectedOptionsWeightsFormatMax =
      kFullyConnectedOptionsWeightsFormatShuffled4x16Int8
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

// Struct to hold LiteRt Concatenation op.
struct ConcatenationOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  int32_t axis;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt Div op.
struct DivOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt FullyConnected op.
struct FullyConnectedOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function = 1;
  FullyConnectedOptionsWeightsFormat weights_format;
  bool keep_num_dims;
  LiteRtElementType quantized_bias_type;
  bool asymmetric_quantize_input;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt Mul op.
struct MulOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt Softmax op.
struct SoftmaxOptions : public OpOptions {
  LiteRtOp op;
  float beta;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt StridedSlice op.
struct StridedSliceOptions : public OpOptions {
  LiteRtOp op;
  int32_t begin_mask;
  int32_t end_mask;
  int32_t ellipsis_mask;
  int32_t new_axis_mask;
  int32_t shrink_axis_mask;
  bool offset;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt Sub op.
struct SubOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt Reshape op.
struct ReshapeOptions : public OpOptions {
  LiteRtOp op;
  std::vector<int32_t> new_shape;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt Sum op.
struct SumOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt ReduceMax op.
struct ReduceMaxOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt Pack op.
struct PackOptions : public OpOptions {
  LiteRtOp op;
  int32_t axis;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt Gather op.
struct GatherOptions : public OpOptions {
  LiteRtOp op;
  int32_t axis;
  int32_t batch_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Struct to hold LiteRt Mean op.
struct MeanOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
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
  } else if constexpr (std::is_same_v<OptionsT, ConcatenationOptions>) {
    ConcatenationOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, DivOptions>) {
    DivOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, FullyConnectedOptions>) {
    FullyConnectedOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, MulOptions>) {
    MulOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, SoftmaxOptions>) {
    SoftmaxOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, StridedSliceOptions>) {
    StridedSliceOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, SubOptions>) {
    SubOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, ReshapeOptions>) {
    ReshapeOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, SumOptions>) {
    SumOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, ReduceMaxOptions>) {
    ReduceMaxOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, PackOptions>) {
    PackOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, GatherOptions>) {
    GatherOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, MeanOptions>) {
    MeanOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else {
    // TODO: Add more as needed.
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_OP_OPTIONS_H_
