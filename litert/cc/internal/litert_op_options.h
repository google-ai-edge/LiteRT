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

#include "flatbuffers/flexbuffers.h"
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/c/litert_builder.h"  // IWYU pragma: keep
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_common.h"

/// @file
/// @brief Defines structures for holding options for various LiteRT operators.

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

inline uint32_t GetTfliteTensorType(LiteRtElementType element_type) {
  switch (element_type) {
    case kLiteRtElementTypeFloat32:
      return TensorType_FLOAT32;
    case kLiteRtElementTypeFloat16:
      return TensorType_FLOAT16;
    case kLiteRtElementTypeInt32:
      return TensorType_INT32;
    case kLiteRtElementTypeUInt8:
      return TensorType_UINT8;
    case kLiteRtElementTypeInt64:
      return TensorType_INT64;
    case kLiteRtElementTypeTfString:
      return TensorType_STRING;
    case kLiteRtElementTypeBool:
      return TensorType_BOOL;
    case kLiteRtElementTypeInt16:
      return TensorType_INT16;
    case kLiteRtElementTypeComplex64:
      return TensorType_COMPLEX64;
    case kLiteRtElementTypeInt8:
      return TensorType_INT8;
    case kLiteRtElementTypeFloat64:
      return TensorType_FLOAT64;
    case kLiteRtElementTypeComplex128:
      return TensorType_COMPLEX128;
    case kLiteRtElementTypeUInt64:
      return TensorType_UINT64;
    case kLiteRtElementTypeTfResource:
      return TensorType_RESOURCE;
    case kLiteRtElementTypeTfVariant:
      return TensorType_VARIANT;
    case kLiteRtElementTypeUInt32:
      return TensorType_UINT32;
    case kLiteRtElementTypeUInt16:
      return TensorType_UINT16;
    case kLiteRtElementTypeInt4:
      return TensorType_INT4;
    case kLiteRtElementTypeBFloat16:
      return TensorType_BFLOAT16;
    default:
      return TensorType_FLOAT32;
  }
};

/// @brief Base struct for operator options.
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

using Padding = uint32_t;
enum PaddingType : uint32_t {
  kPaddingSame = 0,
  kPaddingValid = 1,
  kPaddingMin = kPaddingSame,
  kPaddingMax = kPaddingValid,
};

using MirrorPadMode = uint32_t;
enum MirrorPadModeType : uint32_t {
  kMirrorPadModeReflect = 0,
  kMirrorPadModeSymmetric = 1,
  kMirrorPadModeMin = kMirrorPadModeReflect,
  kMirrorPadModeMax = kMirrorPadModeSymmetric,
};

/// @brief Struct to hold options for LiteRT composite ops.
struct CompositeOptions : public OpOptions {
  /// Name for special composites representing manual partitions.
  static constexpr absl::string_view kNpuCall = "odml.npu_call";
  static constexpr absl::string_view kCpuCall = "odml.cpu_call";
  static constexpr absl::string_view kRmsNorm = "odml.rms_norm";
  static constexpr absl::string_view kL2Norm = "odml.l2_norm";
  static constexpr absl::string_view kGroupNorm = "odml.group_norm";

  /// The root op.
  LiteRtOp op;
  /// Decomposition subgraph.
  int subgraph;
  /// The name of the composite op (stored in model).
  absl::string_view name;
  /// The version of the composite op.
  int32_t version;
  /// The attributes of the composite op.
  std::optional<flexbuffers::Map> attributes_map;

  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

struct RmsNormOpts : public CompositeOptions {
  /// The epsilon composite attribute of the RMS norm.
  float epsilon;
  LiteRtStatus InitFromOp(LiteRtOp litert_op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Add op.
struct AddOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT BatchMatmul op.
struct BatchMatmulOptions : public OpOptions {
  LiteRtOp op;
  bool adj_x;
  bool adj_y;
  bool asymmetric_quantize_input;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Concatenation op.
struct ConcatenationOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  int32_t axis;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Div op.
struct DivOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT FullyConnected op.
struct FullyConnectedOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function = 1;
  FullyConnectedOptionsWeightsFormat weights_format;
  bool keep_num_dims;
  LiteRtElementType quantized_bias_type;
  bool asymmetric_quantize_input;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Mul op.
struct MulOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Softmax op.
struct SoftmaxOptions : public OpOptions {
  LiteRtOp op;
  float beta;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT StridedSlice op.
struct StridedSliceOptions : public OpOptions {
  LiteRtOp op;
  int32_t begin_mask;
  int32_t end_mask;
  int32_t ellipsis_mask;
  int32_t new_axis_mask;
  int32_t shrink_axis_mask;
  bool offset;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Sub op.
struct SubOptions : public OpOptions {
  LiteRtOp op;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Reshape op.
struct ReshapeOptions : public OpOptions {
  LiteRtOp op;
  std::vector<int32_t> new_shape;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Sum op.
struct SumOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT ReduceMax op.
struct ReduceMaxOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT ReduceMin op.
struct ReduceMinOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT ReduceAny op.
struct ReduceAnyOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT ReduceAll op.
struct ReduceAllOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Pack op.
struct PackOptions : public OpOptions {
  LiteRtOp op;
  int32_t axis;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Unpack op.
struct UnpackOptions : public OpOptions {
  LiteRtOp op;
  int32_t axis;
  int32_t num;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Gather op.
struct GatherOptions : public OpOptions {
  LiteRtOp op;
  int32_t axis;
  int32_t batch_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Mean op.
struct MeanOptions : public OpOptions {
  LiteRtOp op;
  bool keep_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Split op.
struct SplitOptions : public OpOptions {
  LiteRtOp op;
  int32_t num_splits;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Conv2d op.
struct Conv2dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t dilation_w_factor;
  int32_t dilation_h_factor;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Conv3d op.
struct Conv3dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t stride_d;
  int32_t dilation_w_factor;
  int32_t dilation_h_factor;
  int32_t dilation_d_factor;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT DepthwiseConv2d op.
struct DepthwiseConv2dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t depth_multiplier;
  ActivationFunction fused_activation_function;
  int32_t dilation_w_factor;
  int32_t dilation_h_factor;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT TransposeConv op.
struct TransposeConvOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT AveragePool2d op.
struct AveragePool2dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t filter_width;
  int32_t filter_height;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT MaxPool2d op.
struct MaxPool2dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t filter_width;
  int32_t filter_height;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT L2Pool2d op.
struct L2Pool2dOptions : public OpOptions {
  LiteRtOp op;
  Padding padding;
  int32_t stride_w;
  int32_t stride_h;
  int32_t filter_width;
  int32_t filter_height;
  ActivationFunction fused_activation_function;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT ResizeBilinear op.
struct ResizeBilinearOptions : public OpOptions {
  LiteRtOp op;
  bool align_corners;
  bool half_pixel_centers;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT LeakyRelu op.
struct LeakyReluOptions : public OpOptions {
  LiteRtOp op;
  float alpha;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT SpaceToDepth op.
struct SpaceToDepthOptions : public OpOptions {
  LiteRtOp op;
  int32_t block_size;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT DepthToSpace op.
struct DepthToSpaceOptions : public OpOptions {
  LiteRtOp op;
  int32_t block_size;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};
/// @brief Struct to hold options for the LiteRT ResizeNearestNeighbor op.
struct ResizeNearestNeighborOptions : public OpOptions {
  LiteRtOp op;
  bool align_corners;
  bool half_pixel_centers;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT CumSum op.
struct CumSumOptions : public OpOptions {
  LiteRtOp op;
  bool exclusive;
  bool reverse;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Gelu op.
struct GeluOptions : public OpOptions {
  LiteRtOp op;
  bool approximate;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT MirrorPad op.
struct MirrorPadOptions : public OpOptions {
  LiteRtOp op;
  MirrorPadMode mode;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Struct to hold options for the LiteRT Squeeze op.
struct SqueezeOptions : public OpOptions {
  LiteRtOp op;
  std::vector<int32_t> squeeze_dims;
  LiteRtStatus InitFromOp(LiteRtOp op) override;
  Expected<void> SetOpOptions(LiteRtBuilder builder);
};

/// @brief Returns the composite info for the given op if it is a composite op.
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
  } else if constexpr (std::is_same_v<OptionsT, ReduceMinOptions>) {
    ReduceMinOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, ReduceAnyOptions>) {
    ReduceAnyOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, ReduceAllOptions>) {
    ReduceAllOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, PackOptions>) {
    PackOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, UnpackOptions>) {
    UnpackOptions options;
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
  } else if constexpr (std::is_same_v<OptionsT, SplitOptions>) {
    SplitOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, Conv2dOptions>) {
    Conv2dOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, Conv3dOptions>) {
    Conv3dOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, DepthwiseConv2dOptions>) {
    DepthwiseConv2dOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, TransposeConvOptions>) {
    TransposeConvOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, AveragePool2dOptions>) {
    AveragePool2dOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, MaxPool2dOptions>) {
    MaxPool2dOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, L2Pool2dOptions>) {
    L2Pool2dOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, ResizeBilinearOptions>) {
    ResizeBilinearOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, LeakyReluOptions>) {
    LeakyReluOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, SpaceToDepthOptions>) {
    SpaceToDepthOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, DepthToSpaceOptions>) {
    DepthToSpaceOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, ResizeNearestNeighborOptions>) {
    ResizeNearestNeighborOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, CumSumOptions>) {
    CumSumOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, GeluOptions>) {
    GeluOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, MirrorPadOptions>) {
    MirrorPadOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else if constexpr (std::is_same_v<OptionsT, SqueezeOptions>) {
    SqueezeOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else {
    // TODO: Add more as needed.
    return Unexpected(Status::kErrorInvalidArgument);
  }
}

inline LiteRtStatus CompositeOptions::InitFromOp(LiteRtOp op) {
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

inline Expected<void> CompositeOptions::SetOpOptions(LiteRtBuilder builder) {
  return Unexpected(Status::kErrorUnsupported);
}

inline LiteRtStatus RmsNormOpts::InitFromOp(LiteRtOp litert_op) {
  LITERT_RETURN_IF_ERROR(CompositeOptions::InitFromOp(litert_op));
  if (!attributes_map.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  constexpr char kEpsilonKey[] = "epsilon";
  flexbuffers::Reference raw_epsilon = attributes_map.value()[kEpsilonKey];
  if (raw_epsilon.IsNull()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  epsilon = raw_epsilon.AsFloat();
  return kLiteRtStatusOk;
}

inline Expected<void> RmsNormOpts::SetOpOptions(LiteRtBuilder builder) {
  return Unexpected(Status::kErrorUnsupported);
}

inline LiteRtStatus AddOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAddFusedActivationOption(op, &fused_activation_function));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> AddOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildAddOpOption(builder, op, &fused_activation_function));
  return Expected<void>();
}

inline LiteRtStatus BatchMatmulOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetBatchMatmulAdjXOption(op, &adj_x));
  LITERT_RETURN_IF_ERROR(LiteRtGetBatchMatmulAdjYOption(op, &adj_y));
  LITERT_RETURN_IF_ERROR(LiteRtGetBatchMatmulAsymmetricQuantizeInputOption(
      op, &asymmetric_quantize_input));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> BatchMatmulOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildBatchMatmulOpOption(
      builder, op, &adj_x, &adj_y, &asymmetric_quantize_input));
  return Expected<void>();
}

inline LiteRtStatus ConcatenationOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetConcatenationAxisOption(op, &axis));
  LITERT_RETURN_IF_ERROR(LiteRtGetConcatenationFusedActivationOption(
      op, &fused_activation_function));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> ConcatenationOptions::SetOpOptions(
    LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildConcatenationOpOption(
      builder, op, &fused_activation_function, &axis));
  return Expected<void>();
}

inline LiteRtStatus DivOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflDiv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDivFusedActivationOption(op, &fused_activation_function));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> DivOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildDivOpOption(builder, op, &fused_activation_function));
  return Expected<void>();
}

inline LiteRtStatus FullyConnectedOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetFullyConnectedFusedActivationOption(
      op, &fused_activation_function));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetFullyConnectedWeightsFormatOption(op, &weights_format));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetFullyConnectedKeepNumDimsOption(op, &keep_num_dims));
  uint32_t retrieved_quantized_bias_type;
  LITERT_RETURN_IF_ERROR(LiteRtFullyConnectedGetQuantizedBiasTypeOption(
      op, &retrieved_quantized_bias_type));
  quantized_bias_type = GetElementType(retrieved_quantized_bias_type);
  LITERT_RETURN_IF_ERROR(LiteRtGetFullyConnectedAsymmetricQuantizeInputOption(
      op, &asymmetric_quantize_input));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> FullyConnectedOptions::SetOpOptions(
    LiteRtBuilder builder) {
  uint32_t quantized_bias_type_uint32 =
      GetTfliteTensorType(quantized_bias_type);
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildFullyConnectedOpOption(
      builder, op, &fused_activation_function, &weights_format, &keep_num_dims,
      &quantized_bias_type_uint32, &asymmetric_quantize_input));
  return Expected<void>();
}

inline LiteRtStatus MulOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflMul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMulFusedActivationOption(op, &fused_activation_function));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> MulOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildMulOpOption(builder, op, &fused_activation_function));
  return Expected<void>();
}

inline LiteRtStatus SoftmaxOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSoftmax) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetSoftmaxBetaOption(op, &beta));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> SoftmaxOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildSoftmaxOpOption(builder, op, &beta));
  return Expected<void>();
}

inline LiteRtStatus StridedSliceOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetStridedSliceBeginMaskOption(op, &begin_mask));
  LITERT_RETURN_IF_ERROR(LiteRtGetStridedSliceEndMaskOption(op, &end_mask));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceEllipsisMaskOption(op, &ellipsis_mask));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceNewAxisMaskOption(op, &new_axis_mask));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceShrinkAxisMaskOption(op, &shrink_axis_mask));
  LITERT_RETURN_IF_ERROR(LiteRtGetStridedSliceOffsetOption(op, &offset));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> StridedSliceOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildStridedSliceOpOption(
      builder, op, &begin_mask, &end_mask, &ellipsis_mask, &new_axis_mask,
      &shrink_axis_mask, &offset));
  return Expected<void>();
}

inline LiteRtStatus SubOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSub) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetSubFusedActivationOption(op, &fused_activation_function));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> SubOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildSubOpOption(builder, op, &fused_activation_function));
  return Expected<void>();
}

inline LiteRtStatus ReshapeOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflReshape) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const int32_t* new_shape_data;
  int32_t new_shape_size;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetReshapeNewShapeOption(op, &new_shape_data, &new_shape_size));
  new_shape.assign(new_shape_data, new_shape_data + new_shape_size);

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> ReshapeOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildReshapeOpOption(
      builder, op, new_shape.data(), new_shape.size()));
  return Expected<void>();
}

inline LiteRtStatus SumOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSum) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetSumKeepDimsOption(op, &keep_dims));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> SumOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildSumOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

inline LiteRtStatus ReduceMaxOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflReduceMax) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetReduceMaxKeepDimsOption(op, &keep_dims));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> ReduceMaxOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildReduceMaxOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

inline LiteRtStatus ReduceMinOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflReduceMin) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetReduceMinKeepDimsOption(op, &keep_dims));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> ReduceMinOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildReduceMinOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

inline LiteRtStatus ReduceAnyOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflReduceAny) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetReduceAnyKeepDimsOption(op, &keep_dims));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> ReduceAnyOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildReduceAnyOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

inline LiteRtStatus ReduceAllOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflReduceAll) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetReduceAllKeepDimsOption(op, &keep_dims));

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> ReduceAllOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildReduceAllOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

inline LiteRtStatus PackOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflPack) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetPackAxisOption(op, &axis));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> PackOptions::SetOpOptions(LiteRtBuilder builder) {
  LiteRtParamIndex num_inputs;
  LITERT_RETURN_IF_ERROR(LiteRtGetNumOpInputs(op, &num_inputs));
  int32_t values_count_int = static_cast<int32_t>(num_inputs);
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildPackOpOption(builder, op, &axis, &values_count_int));
  return Expected<void>();
}

inline LiteRtStatus UnpackOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflUnpack) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetUnpackAxisOption(op, &axis));
  LITERT_RETURN_IF_ERROR(LiteRtGetUnpackNumOption(op, &num));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> UnpackOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildUnpackOpOption(builder, op, &axis, &num));
  return Expected<void>();
}

inline LiteRtStatus GatherOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflGather) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetGatherAxisOption(op, &axis));
  LITERT_RETURN_IF_ERROR(LiteRtGetGatherBatchDimsOption(op, &batch_dims));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> GatherOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildGatherOpOption(builder, op, &axis, &batch_dims));
  return Expected<void>();
}

inline LiteRtStatus MeanOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflMean) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetMeanKeepDimsOption(op, &keep_dims));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> MeanOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildMeanOpOption(builder, op, &keep_dims));
  return Expected<void>();
}

inline LiteRtStatus SplitOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSplit) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetSplitNumSplitsOption(op, &num_splits));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> SplitOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildSplitOpOption(builder, op, &num_splits));
  return Expected<void>();
}

inline LiteRtStatus Conv2dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv2dDilationWOption(op, &dilation_w_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv2dDilationHOption(op, &dilation_h_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv2dFusedActivationOption(op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> Conv2dOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildConv2dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &dilation_w_factor,
      &dilation_h_factor, &fused_activation_function));
  return Expected<void>();
}

inline LiteRtStatus Conv3dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflConv3d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetConv3dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetConv3dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetConv3dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(LiteRtGetConv3dStrideDOption(op, &stride_d));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv3dDilationWOption(op, &dilation_w_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv3dDilationHOption(op, &dilation_h_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv3dDilationDOption(op, &dilation_d_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv3dFusedActivationOption(op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> Conv3dOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildConv3dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &stride_d,
      &dilation_w_factor, &dilation_h_factor, &dilation_d_factor,
      &fused_activation_function));
  return Expected<void>();
}

inline LiteRtStatus DepthwiseConv2dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dDepthMultiplierOption(op, &depth_multiplier));
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dFusedActivationOption(
      op, &fused_activation_function));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dDilationWOption(op, &dilation_w_factor));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dDilationHOption(op, &dilation_h_factor));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> DepthwiseConv2dOptions::SetOpOptions(
    LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildDepthwiseConv2dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &depth_multiplier,
      &fused_activation_function, &dilation_w_factor, &dilation_h_factor));
  return Expected<void>();
}

inline LiteRtStatus TransposeConvOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflTransposeConv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetTransposeConvPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetTransposeConvStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetTransposeConvStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(LiteRtGetTransposeConvFusedActivationOption(
      op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> TransposeConvOptions::SetOpOptions(
    LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildTransposeConvOpOption(
      builder, op, &padding, &stride_w, &stride_h, &fused_activation_function));
  return Expected<void>();
}

inline LiteRtStatus AveragePool2dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dFilterWidthOption(op, &filter_width));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dFilterHeightOption(op, &filter_height));
  LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dFusedActivationOption(
      op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> AveragePool2dOptions::SetOpOptions(
    LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildAveragePool2dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &filter_width,
      &filter_height, &fused_activation_function));
  return Expected<void>();
}

inline LiteRtStatus MaxPool2dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflMaxPool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFilterWidthOption(op, &filter_width));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFilterHeightOption(op, &filter_height));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFusedActivationOption(op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> MaxPool2dOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildMaxPool2dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &filter_width,
      &filter_height, &fused_activation_function));
  return Expected<void>();
}

inline LiteRtStatus L2Pool2dOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflL2Pool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetL2Pool2dPaddingOption(op, &padding));
  LITERT_RETURN_IF_ERROR(LiteRtGetL2Pool2dStrideWOption(op, &stride_w));
  LITERT_RETURN_IF_ERROR(LiteRtGetL2Pool2dStrideHOption(op, &stride_h));
  LITERT_RETURN_IF_ERROR(LiteRtGetL2Pool2dFilterWidthOption(op, &filter_width));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetL2Pool2dFilterHeightOption(op, &filter_height));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetL2Pool2dFusedActivationOption(op, &fused_activation_function));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> L2Pool2dOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildL2Pool2dOpOption(
      builder, op, &padding, &stride_w, &stride_h, &filter_width,
      &filter_height, &fused_activation_function));
  return Expected<void>();
}

inline LiteRtStatus ResizeBilinearOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflResizeBilinear) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetResizeBilinearAlignCornersOption(op, &align_corners));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetResizeBilinearHalfPixelCenterOption(op, &half_pixel_centers));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> ResizeBilinearOptions::SetOpOptions(
    LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildResizeBilinearOpOption(
      builder, op, &align_corners, &half_pixel_centers));
  return Expected<void>();
}

inline LiteRtStatus LeakyReluOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflLeakyRelu) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetLeakyReluAlphaOption(op, &alpha));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> LeakyReluOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildLeakyReluOpOption(builder, op, &alpha));
  return Expected<void>();
}

inline LiteRtStatus SpaceToDepthOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSpaceToDepth) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetSpaceToDepthBlockSizeOption(op, &block_size));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> SpaceToDepthOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildSpaceToDepthOpOption(builder, op, &block_size));
  return Expected<void>();
}

inline LiteRtStatus DepthToSpaceOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflDepthToSpace) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthToSpaceBlockSizeOption(op, &block_size));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> DepthToSpaceOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildDepthToSpaceOpOption(builder, op, &block_size));
  return Expected<void>();
}

inline LiteRtStatus ResizeNearestNeighborOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflResizeNearestNeighbor) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtGetResizeNearestNeighborAlignCornersOption(op, &align_corners));
  LITERT_RETURN_IF_ERROR(LiteRtGetResizeNearestNeighborHalfPixelCenterOption(
      op, &half_pixel_centers));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> ResizeNearestNeighborOptions::SetOpOptions(
    LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildResizeNearestNeighborOpOption(
      builder, op, &align_corners, &half_pixel_centers));
  return Expected<void>();
}

inline LiteRtStatus CumSumOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflCumsum) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetCumsumExclusiveOption(op, &exclusive));
  LITERT_RETURN_IF_ERROR(LiteRtGetCumsumReverseOption(op, &reverse));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> CumSumOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildCumsumOpOption(builder, op, &exclusive, &reverse));
  return Expected<void>();
}

inline LiteRtStatus GeluOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflGelu) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetGeluApproximateOption(op, &approximate));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> GeluOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildGeluOpOption(builder, op, &approximate));
  return Expected<void>();
}

inline LiteRtStatus MirrorPadOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflMirrorPad) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LiteRtGetMirrorPadModeOption(op, &mode));
  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> MirrorPadOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(
      LiteRtBuilderBuildMirrorPadOpOption(builder, op, &mode));
  return Expected<void>();
}

inline LiteRtStatus SqueezeOptions::InitFromOp(LiteRtOp op) {
  LiteRtOpCode opcode;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpCode(op, &opcode));
  if (opcode != kLiteRtOpCodeTflSqueeze) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const int32_t* squeeze_dims_data;
  int32_t num_squeeze_dims;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetSqueezeDimsOption(op, &squeeze_dims_data, &num_squeeze_dims));
  squeeze_dims.assign(squeeze_dims_data, squeeze_dims_data + num_squeeze_dims);

  this->op = op;

  return kLiteRtStatusOk;
}

inline Expected<void> SqueezeOptions::SetOpOptions(LiteRtBuilder builder) {
  LITERT_RETURN_IF_ERROR(LiteRtBuilderBuildSqueezeOpOption(
      builder, op, squeeze_dims.data(), squeeze_dims.size()));
  return Expected<void>();
}
}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_OP_OPTIONS_H_
