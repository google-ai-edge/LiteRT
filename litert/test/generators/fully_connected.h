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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_FULLY_CONNECTED_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_FULLY_CONNECTED_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tflite/converter/schema/schema_generated.h"
#include "tflite/kernels/internal/cppmath.h"
#include "tflite/kernels/internal/portable_tensor_utils.h"
#include "tflite/kernels/internal/quantization_util.h"
#include "tflite/kernels/internal/reference/fully_connected.h"
#include "tflite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tflite/kernels/internal/runtime_shape.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/types/fp16.h"

namespace litert::testing {

enum class FullyConnectedPreset {
  kFloatStatic,
  kFloatStaticNoBias,
  kFloatStatic1d,
  kFloatStatic1dKeepDims,
  kFloatStatic2d,
  kFloatStatic2dKeepDims,
  kFloatStatic3d,
  kFloatStaticRelu,
  kFloatStaticRelu6,
  kFloatStaticReluN1To1,
  kFloatStatic3dReshape,
  kFloatStatic3dKeepDims,
  kFloatStatic4d,
  kFloatStatic4dKeepDims,
  kFloatFp16WeightsStatic,
  kFloatFp16WeightsStaticF32Bias,
  kFloatFp16WeightsStaticNoBias,
  kFloatDynamic,
  kFloatDynamicFilter,
  kFloatDynamicFilterNoBias,
  kFloatDynamicBias,
  kHybridInt8Static,
  kHybridInt8Dynamic,
  kHybridInt8PerChannelStatic,
  kHybridInt8PerChannelDynamic,
  kHybridInt8AsymmetricStatic,
  kHybridInt8AsymmetricDynamic,
  kHybridInt8PerChannelAsymmetricStatic,
  kHybridInt8PerChannelAsymmetricDynamic,
  kUint8Static1d,
  kUint8Static1dKeepDims,
  kUint8Static2d,
  kUint8Static2dKeepDims,
  kUint8Static3d,
  kUint8Static3dReshape,
  kUint8Static3dKeepDims,
  kUint8Static4d,
  kUint8Static4dKeepDims,
  kUint8Static,
  kUint8StaticNoBias,
  kUint8StaticRelu,
  kUint8StaticRelu6,
  kUint8StaticReluN1To1,
  kUint8Int16Static,
  kUint8ShuffledStatic,
  kInt8Static1d,
  kInt8Static1dKeepDims,
  kInt8Static2d,
  kInt8Static2dKeepDims,
  kInt8Static3d,
  kInt8Static3dReshape,
  kInt8Static3dKeepDims,
  kInt8Static4d,
  kInt8Static4dKeepDims,
  kInt8Static,
  kInt8StaticRelu,
  kInt8StaticRelu6,
  kInt8StaticReluN1To1,
  kInt8StaticNoBias,
  kInt8PerChannelStatic,
  kInt8PerChannelStaticNoBias,
  kInt8Int4PerChannelStatic,
  kInt8Int2PerChannelStatic,
  kInt16Int8StaticInt32Bias,
  kInt16Int8PerChannelStaticInt32Bias,
  kInt16Int8StaticInt64Bias,
  kInt16Int8PerChannelStaticInt64Bias,
  kInt16Int16StaticInt32Bias,
  kInt16Int16PerChannelStaticInt32Bias,
  kInt16Int16StaticInt64Bias,
  kInt16Int16PerChannelStaticInt64Bias,
  kInt16Int4PerChannelStaticInt32Bias,
  kInt16Int4PerChannelStaticInt64Bias,
  kInt16Int2PerChannelStaticInt32Bias,
  kInt16Int2PerChannelStaticInt64Bias,
};

template <FullyConnectedPreset Preset>
using FullyConnectedPresetC =
    std::integral_constant<FullyConnectedPreset, Preset>;

template <FullyConnectedPreset... Presets>
using FullyConnectedPresetListC = TypeList<FullyConnectedPresetC<Presets>...>;

namespace fully_connected_internal {

struct PresetDescriptor {
  const char* name;
  LiteRtElementType input_type;
  LiteRtElementType filter_type;
  LiteRtElementType output_type;
  LiteRtElementType bias_type;
  tflite::FullyConnectedOptionsWeightsFormat weights_format;
  bool dynamic_filter;
  bool dynamic_bias;
  bool fp16_filter;
  bool fp16_bias;
  bool per_channel;
  bool bias_always_present;
  bool bias_never_present;
  bool fixed_activation;
  tflite::ActivationFunctionType activation;
  double reference_tolerance;
  int fixed_rank;
  int keep_num_dims_mode;
  bool flatten_all_non_batch_dims;
  bool asymmetric_quantize_inputs;
};

inline constexpr size_t kMaxRank = 6;
inline constexpr double kSymmetricHybridReferenceTolerance = 5e-2;
inline constexpr double kAsymmetricHybridReferenceTolerance = 5e-3;

inline constexpr std::array<tflite::ActivationFunctionType, 4> kActivations = {
    tflite::ActivationFunctionType_NONE,
    tflite::ActivationFunctionType_RELU,
    tflite::ActivationFunctionType_RELU6,
    tflite::ActivationFunctionType_RELU_N1_TO_1,
};

constexpr PresetDescriptor MakeDescriptor(
    const char* name, LiteRtElementType input_type,
    LiteRtElementType filter_type, LiteRtElementType output_type,
    LiteRtElementType bias_type,
    tflite::FullyConnectedOptionsWeightsFormat weights_format,
    bool dynamic_filter = false, bool dynamic_bias = false,
    bool fp16_filter = false, bool fp16_bias = false, bool per_channel = false,
    bool bias_always_present = false, bool bias_never_present = false,
    bool fixed_activation = false,
    tflite::ActivationFunctionType activation =
        tflite::ActivationFunctionType_NONE,
    double reference_tolerance = 0.0, int fixed_rank = 0,
    int keep_num_dims_mode = -1, bool flatten_all_non_batch_dims = false,
    bool asymmetric_quantize_inputs = false) {
  return {name,
          input_type,
          filter_type,
          output_type,
          bias_type,
          weights_format,
          dynamic_filter,
          dynamic_bias,
          fp16_filter,
          fp16_bias,
          per_channel,
          bias_always_present,
          bias_never_present,
          fixed_activation,
          activation,
          reference_tolerance,
          fixed_rank,
          keep_num_dims_mode,
          flatten_all_non_batch_dims,
          asymmetric_quantize_inputs};
}

constexpr PresetDescriptor MakeDescriptor(
    const char* name, LiteRtElementType input_type,
    LiteRtElementType filter_type, LiteRtElementType output_type,
    LiteRtElementType bias_type,
    tflite::FullyConnectedOptionsWeightsFormat weights_format,
    bool dynamic_filter, bool dynamic_bias, bool fp16_filter, bool fp16_bias,
    bool per_channel, bool bias_always_present, bool bias_never_present,
    double reference_tolerance, int fixed_rank = 0,
    int keep_num_dims_mode = -1, bool flatten_all_non_batch_dims = false,
    bool asymmetric_quantize_inputs = false) {
  return MakeDescriptor(name, input_type, filter_type, output_type, bias_type,
                        weights_format, dynamic_filter, dynamic_bias,
                        fp16_filter, fp16_bias, per_channel,
                        bias_always_present, bias_never_present,
                        /*fixed_activation=*/false,
                        /*activation=*/tflite::ActivationFunctionType_NONE,
                        reference_tolerance, fixed_rank, keep_num_dims_mode,
                        flatten_all_non_batch_dims, asymmetric_quantize_inputs);
}

constexpr PresetDescriptor MakeHybridDescriptor(
    const char* name, bool dynamic, bool per_channel,
    bool asymmetric_quantize_inputs) {
  return MakeDescriptor(
      name, kLiteRtElementTypeFloat32, kLiteRtElementTypeInt8,
      kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*dynamic_filter=*/dynamic,
      /*dynamic_bias=*/dynamic,
      /*fp16_filter=*/false,
      /*fp16_bias=*/false,
      /*per_channel=*/per_channel,
      /*bias_always_present=*/false,
      /*bias_never_present=*/false,
      /*reference_tolerance=*/
      asymmetric_quantize_inputs ? kAsymmetricHybridReferenceTolerance
                                 : kSymmetricHybridReferenceTolerance,
      /*fixed_rank=*/0,
      /*keep_num_dims_mode=*/-1,
      /*flatten_all_non_batch_dims=*/false, asymmetric_quantize_inputs);
}

constexpr bool IsHybrid(const PresetDescriptor& descriptor) {
  return descriptor.input_type == kLiteRtElementTypeFloat32 &&
         descriptor.filter_type == kLiteRtElementTypeInt8 &&
         descriptor.output_type == kLiteRtElementTypeFloat32;
}

constexpr PresetDescriptor DescribePreset(FullyConnectedPreset preset) {
  switch (preset) {
    case FullyConnectedPreset::kFloatStatic:
      return MakeDescriptor(
          "FullyConnectedFloatStatic", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatStaticNoBias:
      return MakeDescriptor(
          "FullyConnectedFloatStaticNoBias", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/true,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatStatic1d:
      return MakeDescriptor(
          "FullyConnectedFloatStatic1d", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4,
          /*fixed_rank=*/1,
          /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kFloatStatic1dKeepDims:
      return MakeDescriptor(
          "FullyConnectedFloatStatic1dKeepDims", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4,
          /*fixed_rank=*/1,
          /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kFloatStatic2d:
      return MakeDescriptor(
          "FullyConnectedFloatStatic2d", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4,
          /*fixed_rank=*/2,
          /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kFloatStatic2dKeepDims:
      return MakeDescriptor(
          "FullyConnectedFloatStatic2dKeepDims", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4,
          /*fixed_rank=*/2,
          /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kFloatStatic3d:
      return MakeDescriptor(
          "FullyConnectedFloatStatic3d", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4,
          /*fixed_rank=*/3,
          /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kFloatStaticRelu:
      return MakeDescriptor(
          "FullyConnectedFloatStaticRelu", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/true,
          /*activation=*/tflite::ActivationFunctionType_RELU,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatStaticRelu6:
      return MakeDescriptor(
          "FullyConnectedFloatStaticRelu6", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/true,
          /*activation=*/tflite::ActivationFunctionType_RELU6,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatStaticReluN1To1:
      return MakeDescriptor(
          "FullyConnectedFloatStaticReluN1To1", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/true,
          /*activation=*/tflite::ActivationFunctionType_RELU_N1_TO_1,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatStatic3dReshape:
      return MakeDescriptor(
          "FullyConnectedFloatStatic3dReshape", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4,
          /*fixed_rank=*/3,
          /*keep_num_dims_mode=*/0,
          /*flatten_all_non_batch_dims=*/true);
    case FullyConnectedPreset::kFloatStatic3dKeepDims:
      return MakeDescriptor(
          "FullyConnectedFloatStatic3dKeepDims", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4,
          /*fixed_rank=*/3,
          /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kFloatStatic4d:
      return MakeDescriptor(
          "FullyConnectedFloatStatic4d", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4,
          /*fixed_rank=*/4,
          /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kFloatStatic4dKeepDims:
      return MakeDescriptor(
          "FullyConnectedFloatStatic4dKeepDims", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4,
          /*fixed_rank=*/4,
          /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kFloatFp16WeightsStatic:
      return MakeDescriptor(
          "FullyConnectedFloatFp16WeightsStatic", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/true, /*fp16_bias=*/true,
          /*per_channel=*/false,
          /*bias_always_present=*/true, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatFp16WeightsStaticF32Bias:
      return MakeDescriptor(
          "FullyConnectedFloatFp16WeightsStaticF32Bias",
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/true, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/true, /*bias_never_present=*/false,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatFp16WeightsStaticNoBias:
      return MakeDescriptor(
          "FullyConnectedFloatFp16WeightsStaticNoBias",
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/true, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/true,
          /*fixed_activation=*/false,
          /*activation=*/tflite::ActivationFunctionType_NONE,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatDynamic:
      return MakeDescriptor(
          "FullyConnectedFloatDynamic", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/true, /*dynamic_bias=*/true,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/true, /*bias_never_present=*/false,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatDynamicFilter:
      return MakeDescriptor(
          "FullyConnectedFloatDynamicFilter", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/true, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/true, /*bias_never_present=*/false,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatDynamicFilterNoBias:
      return MakeDescriptor(
          "FullyConnectedFloatDynamicFilterNoBias", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/true, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/true,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kFloatDynamicBias:
      return MakeDescriptor(
          "FullyConnectedFloatDynamicBias", kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
          kLiteRtElementTypeFloat32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/true,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/true, /*bias_never_present=*/false,
          /*reference_tolerance=*/1e-4);
    case FullyConnectedPreset::kHybridInt8Static:
      return MakeHybridDescriptor("FullyConnectedHybridInt8Static",
                                  /*dynamic=*/false,
                                  /*per_channel=*/false,
                                  /*asymmetric_quantize_inputs=*/false);
    case FullyConnectedPreset::kHybridInt8Dynamic:
      return MakeHybridDescriptor("FullyConnectedHybridInt8Dynamic",
                                  /*dynamic=*/true,
                                  /*per_channel=*/false,
                                  /*asymmetric_quantize_inputs=*/false);
    case FullyConnectedPreset::kHybridInt8PerChannelStatic:
      return MakeHybridDescriptor("FullyConnectedHybridInt8PerChannelStatic",
                                  /*dynamic=*/false,
                                  /*per_channel=*/true,
                                  /*asymmetric_quantize_inputs=*/false);
    case FullyConnectedPreset::kHybridInt8PerChannelDynamic:
      return MakeHybridDescriptor("FullyConnectedHybridInt8PerChannelDynamic",
                                  /*dynamic=*/true,
                                  /*per_channel=*/true,
                                  /*asymmetric_quantize_inputs=*/false);
    case FullyConnectedPreset::kHybridInt8AsymmetricStatic:
      return MakeHybridDescriptor("FullyConnectedHybridInt8AsymmetricStatic",
                                  /*dynamic=*/false,
                                  /*per_channel=*/false,
                                  /*asymmetric_quantize_inputs=*/true);
    case FullyConnectedPreset::kHybridInt8AsymmetricDynamic:
      return MakeHybridDescriptor("FullyConnectedHybridInt8AsymmetricDynamic",
                                  /*dynamic=*/true,
                                  /*per_channel=*/false,
                                  /*asymmetric_quantize_inputs=*/true);
    case FullyConnectedPreset::kHybridInt8PerChannelAsymmetricStatic:
      return MakeHybridDescriptor(
          "FullyConnectedHybridInt8PerChannelAsymmetricStatic",
          /*dynamic=*/false,
          /*per_channel=*/true,
          /*asymmetric_quantize_inputs=*/true);
    case FullyConnectedPreset::kHybridInt8PerChannelAsymmetricDynamic:
      return MakeHybridDescriptor(
          "FullyConnectedHybridInt8PerChannelAsymmetricDynamic",
          /*dynamic=*/true,
          /*per_channel=*/true,
          /*asymmetric_quantize_inputs=*/true);
    case FullyConnectedPreset::kUint8Static1d:
      return MakeDescriptor("FullyConnectedUint8Static1d",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/1,
                            /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kUint8Static1dKeepDims:
      return MakeDescriptor("FullyConnectedUint8Static1dKeepDims",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/1,
                            /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kUint8Static2d:
      return MakeDescriptor("FullyConnectedUint8Static2d",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/2,
                            /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kUint8Static2dKeepDims:
      return MakeDescriptor("FullyConnectedUint8Static2dKeepDims",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/2,
                            /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kUint8Static3d:
      return MakeDescriptor("FullyConnectedUint8Static3d",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/3,
                            /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kUint8Static3dReshape:
      return MakeDescriptor("FullyConnectedUint8Static3dReshape",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/3,
                            /*keep_num_dims_mode=*/0,
                            /*flatten_all_non_batch_dims=*/true);
    case FullyConnectedPreset::kUint8Static3dKeepDims:
      return MakeDescriptor("FullyConnectedUint8Static3dKeepDims",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/3,
                            /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kUint8Static4d:
      return MakeDescriptor("FullyConnectedUint8Static4d",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/4,
                            /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kUint8Static4dKeepDims:
      return MakeDescriptor("FullyConnectedUint8Static4dKeepDims",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/4,
                            /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kUint8Static:
      return MakeDescriptor("FullyConnectedUint8Static",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kUint8StaticNoBias:
      return MakeDescriptor("FullyConnectedUint8StaticNoBias",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/true,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kUint8StaticRelu:
      return MakeDescriptor("FullyConnectedUint8StaticRelu",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*fixed_activation=*/true,
                            /*activation=*/tflite::ActivationFunctionType_RELU,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kUint8StaticRelu6:
      return MakeDescriptor("FullyConnectedUint8StaticRelu6",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*fixed_activation=*/true,
                            /*activation=*/tflite::ActivationFunctionType_RELU6,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kUint8StaticReluN1To1:
      return MakeDescriptor(
          "FullyConnectedUint8StaticReluN1To1", kLiteRtElementTypeUInt8,
          kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
          kLiteRtElementTypeInt32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/true,
          /*activation=*/tflite::ActivationFunctionType_RELU_N1_TO_1,
          /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kUint8Int16Static:
      return MakeDescriptor("FullyConnectedUint8Int16Static",
                            kLiteRtElementTypeUInt8, kLiteRtElementTypeUInt8,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/true,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kUint8ShuffledStatic:
      return MakeDescriptor(
          "FullyConnectedUint8ShuffledStatic", kLiteRtElementTypeUInt8,
          kLiteRtElementTypeUInt8, kLiteRtElementTypeInt16,
          kLiteRtElementTypeInt32,
          tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/true, /*bias_never_present=*/false,
          /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt8Static1d:
      return MakeDescriptor("FullyConnectedInt8Static1d",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/1,
                            /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kInt8Static1dKeepDims:
      return MakeDescriptor("FullyConnectedInt8Static1dKeepDims",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/1,
                            /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kInt8Static2d:
      return MakeDescriptor("FullyConnectedInt8Static2d",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/2,
                            /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kInt8Static2dKeepDims:
      return MakeDescriptor("FullyConnectedInt8Static2dKeepDims",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/2,
                            /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kInt8Static3d:
      return MakeDescriptor("FullyConnectedInt8Static3d",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/3,
                            /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kInt8Static3dReshape:
      return MakeDescriptor("FullyConnectedInt8Static3dReshape",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/3,
                            /*keep_num_dims_mode=*/0,
                            /*flatten_all_non_batch_dims=*/true);
    case FullyConnectedPreset::kInt8Static3dKeepDims:
      return MakeDescriptor("FullyConnectedInt8Static3dKeepDims",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/3,
                            /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kInt8Static4d:
      return MakeDescriptor("FullyConnectedInt8Static4d",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/4,
                            /*keep_num_dims_mode=*/0);
    case FullyConnectedPreset::kInt8Static4dKeepDims:
      return MakeDescriptor("FullyConnectedInt8Static4dKeepDims",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0,
                            /*fixed_rank=*/4,
                            /*keep_num_dims_mode=*/1);
    case FullyConnectedPreset::kInt8Static:
      return MakeDescriptor("FullyConnectedInt8Static", kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt8StaticRelu:
      return MakeDescriptor("FullyConnectedInt8StaticRelu",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*fixed_activation=*/true,
                            /*activation=*/tflite::ActivationFunctionType_RELU,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt8StaticRelu6:
      return MakeDescriptor("FullyConnectedInt8StaticRelu6",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*fixed_activation=*/true,
                            /*activation=*/tflite::ActivationFunctionType_RELU6,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt8StaticReluN1To1:
      return MakeDescriptor(
          "FullyConnectedInt8StaticReluN1To1", kLiteRtElementTypeInt8,
          kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
          kLiteRtElementTypeInt32,
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
          /*dynamic_filter=*/false, /*dynamic_bias=*/false,
          /*fp16_filter=*/false, /*fp16_bias=*/false,
          /*per_channel=*/false,
          /*bias_always_present=*/false, /*bias_never_present=*/false,
          /*fixed_activation=*/true,
          /*activation=*/tflite::ActivationFunctionType_RELU_N1_TO_1,
          /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt8StaticNoBias:
      return MakeDescriptor("FullyConnectedInt8StaticNoBias",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/true,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt8PerChannelStatic:
      return MakeDescriptor("FullyConnectedInt8PerChannelStatic",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt8PerChannelStaticNoBias:
      return MakeDescriptor("FullyConnectedInt8PerChannelStaticNoBias",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/true,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt8Int4PerChannelStatic:
      return MakeDescriptor("FullyConnectedInt8Int4PerChannelStatic",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt4,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt8Int2PerChannelStatic:
      return MakeDescriptor("FullyConnectedInt8Int2PerChannelStatic",
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt2,
                            kLiteRtElementTypeInt8, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int8StaticInt32Bias:
      return MakeDescriptor("FullyConnectedInt16Int8StaticInt32Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int8PerChannelStaticInt32Bias:
      return MakeDescriptor("FullyConnectedInt16Int8PerChannelStaticInt32Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int8StaticInt64Bias:
      return MakeDescriptor("FullyConnectedInt16Int8StaticInt64Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt64,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/true,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int8PerChannelStaticInt64Bias:
      return MakeDescriptor("FullyConnectedInt16Int8PerChannelStaticInt64Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt8,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt64,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/true,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int16StaticInt32Bias:
      return MakeDescriptor("FullyConnectedInt16Int16StaticInt32Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt16,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int16PerChannelStaticInt32Bias:
      return MakeDescriptor("FullyConnectedInt16Int16PerChannelStaticInt32Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt16,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int16StaticInt64Bias:
      return MakeDescriptor("FullyConnectedInt16Int16StaticInt64Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt16,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt64,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/false,
                            /*bias_always_present=*/true,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int16PerChannelStaticInt64Bias:
      return MakeDescriptor("FullyConnectedInt16Int16PerChannelStaticInt64Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt16,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt64,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/true,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int4PerChannelStaticInt32Bias:
      return MakeDescriptor("FullyConnectedInt16Int4PerChannelStaticInt32Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt4,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int4PerChannelStaticInt64Bias:
      return MakeDescriptor("FullyConnectedInt16Int4PerChannelStaticInt64Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt4,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt64,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/true,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int2PerChannelStaticInt32Bias:
      return MakeDescriptor("FullyConnectedInt16Int2PerChannelStaticInt32Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt2,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt32,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/false,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
    case FullyConnectedPreset::kInt16Int2PerChannelStaticInt64Bias:
      return MakeDescriptor("FullyConnectedInt16Int2PerChannelStaticInt64Bias",
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt2,
                            kLiteRtElementTypeInt16, kLiteRtElementTypeInt64,
                            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                            /*dynamic_filter=*/false, /*dynamic_bias=*/false,
                            /*fp16_filter=*/false, /*fp16_bias=*/false,
                            /*per_channel=*/true,
                            /*bias_always_present=*/true,
                            /*bias_never_present=*/false,
                            /*reference_tolerance=*/0.0);
  }
  return MakeDescriptor("FullyConnectedInvalid", kLiteRtElementTypeNone,
                        kLiteRtElementTypeNone, kLiteRtElementTypeNone,
                        kLiteRtElementTypeNone,
                        tflite::FullyConnectedOptionsWeightsFormat_DEFAULT);
}

inline size_t NumElements(absl::Span<const int32_t> dims) {
  size_t num_elements = 1;
  for (const int32_t dim : dims) {
    num_elements *= static_cast<size_t>(dim);
  }
  return num_elements;
}

inline size_t RoundUp(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

inline std::vector<uint16_t> Float32ToFp16(absl::Span<const float> values) {
  std::vector<uint16_t> fp16_values(values.size());
  absl::c_transform(values, fp16_values.begin(), fp16_ieee_from_fp32_value);
  return fp16_values;
}

inline std::vector<float> Fp16ToFloat32(absl::Span<const uint16_t> values) {
  std::vector<float> fp32_values(values.size());
  absl::c_transform(values, fp32_values.begin(), fp16_ieee_to_fp32_value);
  return fp32_values;
}

template <typename T>
std::vector<uint8_t> CopyToBytes(absl::Span<const T> values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

template <typename T>
std::vector<uint8_t> CopyToBytes(const std::vector<T>& values) {
  return CopyToBytes(absl::MakeConstSpan(values));
}

template <typename T>
std::vector<T> BytesToVector(absl::Span<const uint8_t> bytes) {
  std::vector<T> values(bytes.size() / sizeof(T));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

inline OwningBufferRef<uint8_t> MakeOwningBytes(
    absl::Span<const uint8_t> values) {
  return OwningBufferRef<uint8_t>(values.data(), values.size());
}

inline Expected<SimpleBuffer> MakeBufferFromDetails(
    const TensorDetails& tensor) {
  Dimensions dims(tensor.dims.begin(), tensor.dims.end());
  return SimpleBuffer::Create(RankedTensorType(
      static_cast<ElementType>(tensor.element_type), Layout(std::move(dims))));
}

inline absl::Span<const uint8_t> ConstBytes(
    const OwningBufferRef<uint8_t>& bytes) {
  return static_cast<const BufferRef<uint8_t>&>(bytes).Span();
}

inline void CopyBytesIntoBuffer(absl::Span<const uint8_t> bytes,
                                SimpleBuffer& buffer) {
  std::memcpy(buffer.MutableData().Data(), bytes.data(), bytes.size());
}

template <typename Rng>
bool RandomBool(Rng& rng) {
  std::bernoulli_distribution dist(0.5);
  return dist(rng);
}

template <typename Rng>
int RandomInt(Rng& rng, int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(rng);
}

template <typename Rng>
float RandomFloat(Rng& rng, float min, float max) {
  std::uniform_real_distribution<float> dist(min, max);
  return dist(rng);
}

template <typename T, typename Rng>
std::vector<T> GenerateIntegralValues(Rng& rng, size_t count, int min,
                                      int max) {
  std::vector<T> values(count);
  std::uniform_int_distribution<int> dist(min, max);
  std::generate(values.begin(), values.end(),
                [&]() { return static_cast<T>(dist(rng)); });
  return values;
}

template <typename Rng>
std::vector<float> GenerateFloatValues(Rng& rng, size_t count, float min,
                                       float max) {
  std::vector<float> values(count);
  std::uniform_real_distribution<float> dist(min, max);
  std::generate(values.begin(), values.end(), [&]() { return dist(rng); });
  return values;
}

template <typename Rng>
std::vector<uint8_t> ShuffleAndXorUint8Weights(Rng& rng, int input_channels,
                                               int output_channels,
                                               int zero_point) {
  std::vector<uint8_t> weights = GenerateIntegralValues<uint8_t>(
      rng, static_cast<size_t>(input_channels) * output_channels,
      std::max(0, zero_point - 8), std::min(255, zero_point + 8));
  std::vector<uint8_t> shuffled(weights.size());
  uint8_t* shuffled_ptr = shuffled.data();
  for (int block_o = 0; block_o < output_channels; block_o += 4) {
    for (int block_i = 0; block_i < input_channels; block_i += 16) {
      for (int o = 0; o < 4; ++o) {
        for (int i = 0; i < 16; ++i) {
          *shuffled_ptr++ =
              weights[(block_o + o) * input_channels + block_i + i] ^ 0x80;
        }
      }
    }
  }
  return shuffled;
}

inline std::vector<uint8_t> PackSignedData(absl::Span<const int8_t> values,
                                           int bit_width) {
  const size_t packed_elements = bit_width == 4 ? RoundUp(values.size(), 2) / 2
                                                : RoundUp(values.size(), 4) / 4;
  std::vector<int8_t> packed(packed_elements);
  tflite::tensor_utils::PackInt8IntoDenseInt(values.data(), values.size(),
                                             bit_width, packed.data());
  return CopyToBytes(packed);
}

inline std::vector<int8_t> UnpackSignedData(absl::Span<const uint8_t> values,
                                            int num_elements, int bit_width) {
  std::vector<int8_t> unpacked(num_elements);
  tflite::tensor_utils::UnpackPackedIntToInt8(
      reinterpret_cast<const int8_t*>(values.data()), num_elements, bit_width,
      unpacked.data());
  return unpacked;
}

inline tflite::RuntimeShape MakeRuntimeShape(absl::Span<const int32_t> dims) {
  return tflite::RuntimeShape(static_cast<int>(dims.size()), dims.data());
}

inline void CalculateFloatActivationRange(
    tflite::ActivationFunctionType activation, float* min, float* max) {
  switch (activation) {
    case tflite::ActivationFunctionType_RELU:
      *min = 0.0f;
      *max = std::numeric_limits<float>::max();
      return;
    case tflite::ActivationFunctionType_RELU6:
      *min = 0.0f;
      *max = 6.0f;
      return;
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
      *min = -1.0f;
      *max = 1.0f;
      return;
    default:
      *min = std::numeric_limits<float>::lowest();
      *max = std::numeric_limits<float>::max();
      return;
  }
}

inline int32_t QuantizeFloatToInt(float scale, int32_t zero_point,
                                  float value) {
  return zero_point + static_cast<int32_t>(std::round(value / scale));
}

inline void CalculateQuantizedActivationRange(
    tflite::ActivationFunctionType activation, LiteRtElementType output_type,
    const TensorDetails::QuantizationDetails& quantization, int32_t* act_min,
    int32_t* act_max) {
  int32_t qmin = 0;
  int32_t qmax = 0;
  switch (output_type) {
    case kLiteRtElementTypeUInt8:
      qmin = std::numeric_limits<uint8_t>::min();
      qmax = std::numeric_limits<uint8_t>::max();
      break;
    case kLiteRtElementTypeInt8:
      qmin = std::numeric_limits<int8_t>::min();
      qmax = std::numeric_limits<int8_t>::max();
      break;
    case kLiteRtElementTypeInt16:
      qmin = std::numeric_limits<int16_t>::min();
      qmax = std::numeric_limits<int16_t>::max();
      break;
    default:
      *act_min = 0;
      *act_max = 0;
      return;
  }
  if (activation == tflite::ActivationFunctionType_RELU) {
    *act_min = std::max(
        qmin,
        QuantizeFloatToInt(quantization.scale, quantization.zero_point, 0.0f));
    *act_max = qmax;
  } else if (activation == tflite::ActivationFunctionType_RELU6) {
    *act_min = std::max(
        qmin,
        QuantizeFloatToInt(quantization.scale, quantization.zero_point, 0.0f));
    *act_max = std::min(
        qmax,
        QuantizeFloatToInt(quantization.scale, quantization.zero_point, 6.0f));
  } else if (activation == tflite::ActivationFunctionType_RELU_N1_TO_1) {
    *act_min = std::max(
        qmin,
        QuantizeFloatToInt(quantization.scale, quantization.zero_point, -1.0f));
    *act_max = std::min(
        qmax,
        QuantizeFloatToInt(quantization.scale, quantization.zero_point, 1.0f));
  } else {
    *act_min = qmin;
    *act_max = qmax;
  }
}

inline tflite::FullyConnectedParams MakeFloatReferenceParams(
    tflite::ActivationFunctionType activation) {
  tflite::FullyConnectedParams params;
  CalculateFloatActivationRange(activation, &params.float_activation_min,
                                &params.float_activation_max);
  return params;
}

inline tflite::FullyConnectedParams MakeQuantizedReferenceParams(
    tflite::ActivationFunctionType activation, const TensorDetails& input,
    const TensorDetails& filter, const TensorDetails& output,
    bool per_channel) {
  tflite::FullyConnectedParams params;
  params.input_offset = -static_cast<int32_t>(input.quantization->zero_point);
  params.weights_offset =
      per_channel ? 0 : -static_cast<int32_t>(filter.quantization->zero_point);
  params.output_offset = static_cast<int32_t>(output.quantization->zero_point);
  CalculateQuantizedActivationRange(
      activation, output.element_type, *output.quantization,
      &params.quantized_activation_min, &params.quantized_activation_max);
  return params;
}

inline void PopulateScalarOutputMultiplier(
    const TensorDetails& input, const TensorDetails& filter,
    const TensorDetails& output, tflite::FullyConnectedParams* params) {
  const auto scalar_scale = [](const auto& quantization) {
    return quantization->type == kLiteRtQuantizationPerChannel
               ? quantization->scales.front()
               : quantization->scale;
  };
  const double real_multiplier =
      static_cast<double>(scalar_scale(input.quantization)) *
      static_cast<double>(scalar_scale(filter.quantization)) /
      static_cast<double>(scalar_scale(output.quantization));
  tflite::QuantizeMultiplier(real_multiplier, &params->output_multiplier,
                             &params->output_shift);
}

inline void PopulatePerChannelOutputMultipliers(
    const TensorDetails& input, const TensorDetails& filter,
    const TensorDetails& output, std::vector<int32_t>* multipliers,
    std::vector<int>* shifts) {
  const auto scalar_scale = [](const auto& quantization) {
    return quantization->type == kLiteRtQuantizationPerChannel
               ? quantization->scales.front()
               : quantization->scale;
  };
  const auto filter_scales =
      filter.quantization->type == kLiteRtQuantizationPerChannel
          ? filter.quantization->scales
          : std::vector<float>(filter.dims.front(), filter.quantization->scale);
  multipliers->resize(filter_scales.size());
  shifts->resize(filter_scales.size());
  const double input_scale =
      static_cast<double>(scalar_scale(input.quantization));
  const double output_scale =
      static_cast<double>(scalar_scale(output.quantization));
  for (size_t i = 0; i < filter_scales.size(); ++i) {
    const double real_multiplier =
        input_scale * static_cast<double>(filter_scales[i]) / output_scale;
    tflite::QuantizeMultiplier(real_multiplier, &(*multipliers)[i],
                               &(*shifts)[i]);
  }
}

inline float SingleScale(
    const std::optional<TensorDetails::QuantizationDetails>& quantization) {
  return quantization->type == kLiteRtQuantizationPerChannel
             ? quantization->scales.front()
             : quantization->scale;
}

inline std::vector<float> ChannelScales(
    const TensorDetails::QuantizationDetails& quantization, int channels) {
  if (quantization.type == kLiteRtQuantizationPerChannel) {
    return quantization.scales;
  }
  return std::vector<float>(channels, quantization.scale);
}

inline std::vector<float> DequantizeFilterToFloat(
    const PresetDescriptor& descriptor, const TensorDetails& filter,
    absl::Span<const uint8_t> raw_bytes, int output_channels,
    int input_channels) {
  std::vector<float> result(static_cast<size_t>(output_channels) *
                            input_channels);
  const auto scales = ChannelScales(*filter.quantization, output_channels);
  switch (filter.element_type) {
    case kLiteRtElementTypeFloat32: {
      auto values = BytesToVector<float>(raw_bytes);
      return values;
    }
    case kLiteRtElementTypeFloat16: {
      auto values = Fp16ToFloat32(BytesToVector<uint16_t>(raw_bytes));
      return values;
    }
    case kLiteRtElementTypeInt8: {
      auto values = BytesToVector<int8_t>(raw_bytes);
      for (int oc = 0; oc < output_channels; ++oc) {
        for (int ic = 0; ic < input_channels; ++ic) {
          result[oc * input_channels + ic] =
              scales[oc] * static_cast<float>(values[oc * input_channels + ic]);
        }
      }
      return result;
    }
    default:
      return result;
  }
}

inline double ApplyActivation(tflite::ActivationFunctionType activation,
                              double value) {
  switch (activation) {
    case tflite::ActivationFunctionType_RELU:
      return std::max(0.0, value);
    case tflite::ActivationFunctionType_RELU6:
      return std::clamp(value, 0.0, 6.0);
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
      return std::clamp(value, -1.0, 1.0);
    default:
      return value;
  }
}

inline double QuantizationScaleForChannel(
    const TensorDetails::QuantizationDetails& quantization, int channel) {
  return quantization.type == kLiteRtQuantizationPerChannel
             ? static_cast<double>(quantization.scales[channel])
             : static_cast<double>(quantization.scale);
}

inline int64_t QuantizationZeroPointForChannel(
    const TensorDetails::QuantizationDetails& quantization, int channel) {
  return quantization.type == kLiteRtQuantizationPerChannel
             ? quantization.zero_points[channel]
             : quantization.zero_point;
}

inline std::vector<uint8_t> UnshuffleAndXorUint8Weights(
    absl::Span<const uint8_t> shuffled, int input_channels,
    int output_channels) {
  std::vector<uint8_t> weights(static_cast<size_t>(input_channels) *
                               output_channels);
  const uint8_t* shuffled_ptr = shuffled.data();
  for (int block_o = 0; block_o < output_channels; block_o += 4) {
    for (int block_i = 0; block_i < input_channels; block_i += 16) {
      for (int o = 0; o < 4; ++o) {
        for (int i = 0; i < 16; ++i) {
          weights[(block_o + o) * input_channels + block_i + i] =
              *shuffled_ptr++ ^ 0x80;
        }
      }
    }
  }
  return weights;
}

template <typename T>
std::vector<double> DequantizeVectorValues(
    absl::Span<const T> values,
    const TensorDetails::QuantizationDetails& quantization) {
  std::vector<double> result(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const double scale =
        QuantizationScaleForChannel(quantization, static_cast<int>(i));
    const int64_t zero_point =
        QuantizationZeroPointForChannel(quantization, static_cast<int>(i));
    result[i] =
        (static_cast<double>(values[i]) - static_cast<double>(zero_point)) *
        scale;
  }
  return result;
}

template <typename T>
std::vector<double> DequantizeFilterValues(
    absl::Span<const T> values,
    const TensorDetails::QuantizationDetails& quantization, int output_channels,
    int input_channels) {
  std::vector<double> result(static_cast<size_t>(output_channels) *
                             input_channels);
  for (int oc = 0; oc < output_channels; ++oc) {
    const double scale = QuantizationScaleForChannel(quantization, oc);
    const int64_t zero_point =
        QuantizationZeroPointForChannel(quantization, oc);
    for (int ic = 0; ic < input_channels; ++ic) {
      const size_t index = static_cast<size_t>(oc) * input_channels + ic;
      result[index] = (static_cast<double>(values[index]) -
                       static_cast<double>(zero_point)) *
                      scale;
    }
  }
  return result;
}

inline std::vector<double> DecodeInputToDouble(const TensorDetails& input,
                                               const SimpleBuffer& buffer) {
  switch (input.element_type) {
    case kLiteRtElementTypeFloat32: {
      const auto input_view = buffer.AsView<float>();
      return std::vector<double>(input_view.data.begin(),
                                 input_view.data.end());
    }
    case kLiteRtElementTypeUInt8:
      return DequantizeVectorValues(buffer.AsView<uint8_t>().data,
                                    *input.quantization);
    case kLiteRtElementTypeInt8:
      return DequantizeVectorValues(buffer.AsView<int8_t>().data,
                                    *input.quantization);
    case kLiteRtElementTypeInt16:
      return DequantizeVectorValues(buffer.AsView<int16_t>().data,
                                    *input.quantization);
    default:
      return {};
  }
}

inline std::vector<double> DecodeFilterToDouble(
    const PresetDescriptor& descriptor, const TensorDetails& filter,
    absl::Span<const uint8_t> raw_bytes, int output_channels,
    int input_channels) {
  if (filter.element_type == kLiteRtElementTypeFloat32) {
    auto values = BytesToVector<float>(raw_bytes);
    return std::vector<double>(values.begin(), values.end());
  }
  if (filter.element_type == kLiteRtElementTypeFloat16) {
    auto values = Fp16ToFloat32(BytesToVector<uint16_t>(raw_bytes));
    return std::vector<double>(values.begin(), values.end());
  }

  if (filter.element_type == kLiteRtElementTypeUInt8 &&
      descriptor.weights_format ==
          tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8) {
    auto values =
        UnshuffleAndXorUint8Weights(raw_bytes, input_channels, output_channels);
    return DequantizeFilterValues<uint8_t>(values, *filter.quantization,
                                           output_channels, input_channels);
  }

  switch (filter.element_type) {
    case kLiteRtElementTypeUInt8: {
      auto values = BytesToVector<uint8_t>(raw_bytes);
      return DequantizeFilterValues<uint8_t>(values, *filter.quantization,
                                             output_channels, input_channels);
    }
    case kLiteRtElementTypeInt8: {
      auto values = BytesToVector<int8_t>(raw_bytes);
      return DequantizeFilterValues<int8_t>(values, *filter.quantization,
                                            output_channels, input_channels);
    }
    case kLiteRtElementTypeInt16: {
      auto values = BytesToVector<int16_t>(raw_bytes);
      return DequantizeFilterValues<int16_t>(values, *filter.quantization,
                                             output_channels, input_channels);
    }
    case kLiteRtElementTypeInt4: {
      auto values =
          UnpackSignedData(raw_bytes, output_channels * input_channels, 4);
      return DequantizeFilterValues<int8_t>(values, *filter.quantization,
                                            output_channels, input_channels);
    }
    case kLiteRtElementTypeInt2: {
      auto values =
          UnpackSignedData(raw_bytes, output_channels * input_channels, 2);
      return DequantizeFilterValues<int8_t>(values, *filter.quantization,
                                            output_channels, input_channels);
    }
    default:
      return {};
  }
}

inline std::vector<double> DecodeBiasToDouble(
    const std::optional<TensorDetails>& bias,
    const std::optional<absl::Span<const uint8_t>>& raw_bytes,
    int output_channels) {
  if (!bias.has_value() || !raw_bytes.has_value()) {
    return std::vector<double>(output_channels, 0.0);
  }
  switch (bias->element_type) {
    case kLiteRtElementTypeFloat32: {
      auto values = BytesToVector<float>(*raw_bytes);
      return std::vector<double>(values.begin(), values.end());
    }
    case kLiteRtElementTypeFloat16: {
      auto values = Fp16ToFloat32(BytesToVector<uint16_t>(*raw_bytes));
      return std::vector<double>(values.begin(), values.end());
    }
    case kLiteRtElementTypeInt32: {
      auto values = BytesToVector<int32_t>(*raw_bytes);
      return DequantizeVectorValues<int32_t>(values, *bias->quantization);
    }
    case kLiteRtElementTypeInt64: {
      auto values = BytesToVector<int64_t>(*raw_bytes);
      return DequantizeVectorValues<int64_t>(values, *bias->quantization);
    }
    default:
      return std::vector<double>(output_channels, 0.0);
  }
}

struct HybridBatchQuantization {
  std::vector<int8_t> values;
  std::vector<double> scales;
  std::vector<int32_t> zero_points;
};

inline HybridBatchQuantization QuantizeHybridInputToInt8(
    absl::Span<const float> values, size_t num_batches, int input_channels,
    bool asymmetric_quantize_inputs) {
  HybridBatchQuantization quantization;
  quantization.values.resize(values.size());
  quantization.scales.resize(num_batches);
  quantization.zero_points.resize(num_batches);

  for (size_t batch = 0; batch < num_batches; ++batch) {
    const size_t batch_offset = batch * input_channels;
    const float* batch_values = values.data() + batch_offset;
    int8_t* batch_quantized = quantization.values.data() + batch_offset;

    if (asymmetric_quantize_inputs) {
      constexpr int32_t kMinScale = -128;
      constexpr int32_t kMaxScale = 127;
      const auto minmax =
          std::minmax_element(batch_values, batch_values + input_channels);
      const double rmin = static_cast<double>(std::min(0.0f, *minmax.first));
      const double rmax = static_cast<double>(std::max(0.0f, *minmax.second));
      if (rmin == rmax) {
        std::fill_n(batch_quantized, input_channels, 0);
        quantization.scales[batch] = 1.0;
        quantization.zero_points[batch] = 0;
        continue;
      }

      const double scale =
          (rmax - rmin) / static_cast<double>(kMaxScale - kMinScale);
      const double zero_point_from_min = kMinScale - rmin / scale;
      const double zero_point_from_max = kMaxScale - rmax / scale;
      const double zero_point_from_min_error =
          std::abs(static_cast<double>(kMinScale)) + std::abs(rmin / scale);
      const double zero_point_from_max_error =
          std::abs(static_cast<double>(kMaxScale)) + std::abs(rmax / scale);
      const double zero_point_double =
          zero_point_from_min_error < zero_point_from_max_error
              ? zero_point_from_min
              : zero_point_from_max;
      const int32_t zero_point =
          zero_point_double <= kMinScale ? kMinScale
          : zero_point_double >= kMaxScale
              ? kMaxScale
              : static_cast<int32_t>(std::round(zero_point_double));

      quantization.scales[batch] = scale;
      quantization.zero_points[batch] = zero_point;
      const double inv_scale = 1.0 / scale;
      for (int ic = 0; ic < input_channels; ++ic) {
        const int32_t quantized_value =
            static_cast<int32_t>(tflite::TfLiteRound(
                zero_point +
                static_cast<double>(batch_values[ic]) * inv_scale));
        batch_quantized[ic] = static_cast<int8_t>(
            std::clamp(quantized_value, kMinScale, kMaxScale));
      }
      continue;
    }

    constexpr int32_t kScale = 127;
    const auto minmax =
        std::minmax_element(batch_values, batch_values + input_channels);
    const double range =
        std::max(std::abs(static_cast<double>(*minmax.first)),
                 std::abs(static_cast<double>(*minmax.second)));
    if (range == 0.0) {
      std::fill_n(batch_quantized, input_channels, 0);
      quantization.scales[batch] = 1.0;
      quantization.zero_points[batch] = 0;
      continue;
    }

    const double scale = range / static_cast<double>(kScale);
    const double inv_scale = static_cast<double>(kScale) / range;
    quantization.scales[batch] = scale;
    quantization.zero_points[batch] = 0;
    for (int ic = 0; ic < input_channels; ++ic) {
      const int32_t quantized_value = static_cast<int32_t>(tflite::TfLiteRound(
          static_cast<double>(batch_values[ic]) * inv_scale));
      batch_quantized[ic] =
          static_cast<int8_t>(std::clamp(quantized_value, -kScale, kScale));
    }
  }

  return quantization;
}

inline Expected<std::vector<int8_t>> DecodeFilterToInt8(
    const PresetDescriptor& descriptor, const TensorDetails& filter,
    absl::Span<const uint8_t> raw_bytes, int output_channels,
    int input_channels) {
  switch (filter.element_type) {
    case kLiteRtElementTypeInt8:
      return BytesToVector<int8_t>(raw_bytes);
    case kLiteRtElementTypeInt4:
      return UnpackSignedData(raw_bytes, output_channels * input_channels, 4);
    case kLiteRtElementTypeInt2:
      return UnpackSignedData(raw_bytes, output_channels * input_channels, 2);
    default:
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unsupported FC hybrid filter type");
  }
}

template <typename T>
T QuantizeRealValue(double value,
                    const TensorDetails::QuantizationDetails& quantization) {
  const double scaled = value / static_cast<double>(quantization.scale) +
                        static_cast<double>(quantization.zero_point);
  const int64_t quantized = static_cast<int64_t>(std::llround(scaled));
  const int64_t clamped =
      std::clamp(quantized, static_cast<int64_t>(std::numeric_limits<T>::min()),
                 static_cast<int64_t>(std::numeric_limits<T>::max()));
  return static_cast<T>(clamped);
}

inline Expected<void> WriteSemanticOutput(const TensorDetails& output,
                                          absl::Span<const double> real_values,
                                          SimpleBuffer& buffer) {
  switch (output.element_type) {
    case kLiteRtElementTypeFloat32: {
      auto output_view = buffer.AsView<float>();
      for (size_t i = 0; i < real_values.size(); ++i) {
        output_view.data[i] = static_cast<float>(real_values[i]);
      }
      return {};
    }
    case kLiteRtElementTypeUInt8: {
      auto output_view = buffer.AsView<uint8_t>();
      for (size_t i = 0; i < real_values.size(); ++i) {
        output_view.data[i] =
            QuantizeRealValue<uint8_t>(real_values[i], *output.quantization);
      }
      return {};
    }
    case kLiteRtElementTypeInt8: {
      auto output_view = buffer.AsView<int8_t>();
      for (size_t i = 0; i < real_values.size(); ++i) {
        output_view.data[i] =
            QuantizeRealValue<int8_t>(real_values[i], *output.quantization);
      }
      return {};
    }
    case kLiteRtElementTypeInt16: {
      auto output_view = buffer.AsView<int16_t>();
      for (size_t i = 0; i < real_values.size(); ++i) {
        output_view.data[i] =
            QuantizeRealValue<int16_t>(real_values[i], *output.quantization);
      }
      return {};
    }
    default:
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unsupported FC semantic output type");
  }
}

template <typename InputT, typename FilterT, typename BiasT, typename OutputT>
void ReferencePerTensorInt(
    const tflite::FullyConnectedParams& op_params,
    const tflite::RuntimeShape& input_shape, const InputT* input_data,
    const tflite::RuntimeShape& filter_shape, const FilterT* filter_data,
    const tflite::RuntimeShape& bias_shape, const BiasT* bias_data,
    const tflite::RuntimeShape& output_shape, OutputT* output_data) {
  tflite::reference_integer_ops::FullyConnected(
      op_params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data);
}

template <typename InputT, typename FilterT, typename BiasT, typename OutputT>
void ReferencePerChannelInt(
    const tflite::FullyConnectedParams& op_params,
    const int32_t* output_multiplier, const int* output_shift,
    const tflite::RuntimeShape& input_shape, const InputT* input_data,
    const tflite::RuntimeShape& filter_shape, const FilterT* filter_data,
    const tflite::RuntimeShape& bias_shape, const BiasT* bias_data,
    const tflite::RuntimeShape& output_shape, OutputT* output_data) {
  tflite::reference_integer_ops::FullyConnectedPerChannel(
      op_params, output_multiplier, output_shift, input_shape, input_data,
      filter_shape, filter_data, bias_shape, bias_data, output_shape,
      output_data);
}

inline LiteRtElementType QuantizedBiasType(const PresetDescriptor& descriptor) {
  return IsHybrid(descriptor) ? kLiteRtElementTypeFloat32
                              : descriptor.bias_type;
}

inline tflite::TensorType ToTensorType(LiteRtElementType element_type) {
  switch (element_type) {
    case kLiteRtElementTypeFloat32:
      return tflite::TensorType_FLOAT32;
    case kLiteRtElementTypeInt32:
      return tflite::TensorType_INT32;
    case kLiteRtElementTypeInt64:
      return tflite::TensorType_INT64;
    default:
      return tflite::TensorType_FLOAT32;
  }
}

}  // namespace fully_connected_internal

namespace Internal = fully_connected_internal;

template <typename Preset>
class FullyConnected : public TestGraph {
 private:
  static_assert(
      std::is_same_v<typename Preset::value_type, FullyConnectedPreset>);
  static constexpr FullyConnectedPreset kPreset = Preset::value;
  static constexpr auto kDescriptor = Internal::DescribePreset(kPreset);

 public:
  struct Params {
    TensorDetails input;
    TensorDetails filter;
    std::optional<TensorDetails> stored_filter;
    std::optional<TensorDetails> bias;
    std::optional<TensorDetails> stored_bias;
    TensorDetails output;
    std::vector<TensorDetails> internal_outputs;
    tflite::ActivationFunctionType activation;
    bool keep_num_dims;
  };

  using Ptr = std::unique_ptr<FullyConnected>;

  static constexpr absl::string_view Name() { return kDescriptor.name; }

  template <typename Rng>
  static Expected<Ptr> Create(Rng& rng) {
    LITERT_ASSIGN_OR_RETURN(auto params, GenerateParams(rng));
    return Create(std::move(params));
  }

  static Expected<Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<FullyConnected>(std::move(params),
                                            std::move(model));
  }

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;

    if (kDescriptor.output_type == kLiteRtElementTypeFloat32) {
      spec.comparator_kind = ConformanceComparatorKind::kFloatAccumulationAware;
      spec.absolute_tolerance = std::max(ComputeFloatConformanceTolerance(),
                                         kDescriptor.reference_tolerance);
      spec.relative_tolerance = 1.0e-6;
      return spec;
    }
    // Currently all other cases are quantized kernels, which might get changed
    // in the future.
    spec.comparator_kind = ConformanceComparatorKind::kQuantizedBucket;
    spec.bucket_tolerance = 1;
    return spec;
  }

  std::optional<double> ReferenceTolerance() const override {
    if (kDescriptor.reference_tolerance > 0.0) {
      return kDescriptor.reference_tolerance;
    }
    if (kDescriptor.output_type == kLiteRtElementTypeUInt8 ||
        kDescriptor.output_type == kLiteRtElementTypeInt8) {
      return 1e-2;
    }
    if (kDescriptor.output_type == kLiteRtElementTypeInt16) {
      return kDescriptor.input_type == kLiteRtElementTypeUInt8 ? 1e-2 : 2.5e-1;
    }
    return kDescriptor.reference_tolerance;
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    (void)data_builder;
    VarBuffers inputs;
    LITERT_ASSIGN_OR_RETURN(auto input_buffer,
                            Internal::MakeBufferFromDetails(params_.input));
    Internal::CopyBytesIntoBuffer(
        GenerateTensorBytes(device, params_.input, /*is_filter=*/false,
                            /*is_bias=*/false),
        input_buffer);
    inputs.push_back(std::move(input_buffer));

    if (kDescriptor.dynamic_filter) {
      LITERT_ASSIGN_OR_RETURN(auto filter_buffer,
                              Internal::MakeBufferFromDetails(params_.filter));
      Internal::CopyBytesIntoBuffer(
          GenerateTensorBytes(device, params_.filter, /*is_filter=*/true,
                              /*is_bias=*/false),
          filter_buffer);
      inputs.push_back(std::move(filter_buffer));
    }

    if (kDescriptor.dynamic_bias && params_.bias.has_value()) {
      LITERT_ASSIGN_OR_RETURN(auto bias_buffer,
                              Internal::MakeBufferFromDetails(*params_.bias));
      Internal::CopyBytesIntoBuffer(
          GenerateTensorBytes(device, *params_.bias, /*is_filter=*/false,
                              /*is_bias=*/true),
          bias_buffer);
      inputs.push_back(std::move(bias_buffer));
    }

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    if (outputs.size() != 1) {
      return Error(kLiteRtStatusErrorInvalidArgument, "Expected one output");
    }
    if (inputs.empty()) {
      return Error(kLiteRtStatusErrorInvalidArgument, "Expected one input");
    }

    size_t next_dynamic_input = 1;
    const absl::Span<const uint8_t> filter_bytes =
        kDescriptor.dynamic_filter
            ? inputs[next_dynamic_input++].Span<uint8_t>()
            : Internal::ConstBytes(*(params_.stored_filter.has_value()
                                         ? params_.stored_filter->data
                                         : params_.filter.data));
    const std::optional<absl::Span<const uint8_t>> bias_bytes =
        (params_.stored_bias.has_value() || params_.bias.has_value())
            ? std::make_optional(
                  kDescriptor.dynamic_bias
                      ? inputs[next_dynamic_input++].Span<uint8_t>()
                      : Internal::ConstBytes(*(params_.stored_bias.has_value()
                                                   ? params_.stored_bias->data
                                                   : params_.bias->data)))
            : std::nullopt;

    if (kDescriptor.output_type == kLiteRtElementTypeFloat32) {
      return ReferenceSemantic(inputs[0], filter_bytes, bias_bytes, outputs[0]);
    }
    if (kDescriptor.input_type == kLiteRtElementTypeUInt8 ||
        kDescriptor.output_type == kLiteRtElementTypeInt8 ||
        kDescriptor.output_type == kLiteRtElementTypeInt16) {
      return ReferenceSemantic(inputs[0], filter_bytes, bias_bytes, outputs[0]);
    }
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Unsupported FC semantic preset");
  }

  FullyConnected(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static double QuantizedMaxScale(
      const std::optional<TensorDetails::QuantizationDetails>& quantization) {
    if (!quantization.has_value()) {
      return 1.0;
    }
    if (quantization->type == kLiteRtQuantizationPerChannel) {
      return static_cast<double>(*std::max_element(quantization->scales.begin(),
                                                   quantization->scales.end()));
    }
    return static_cast<double>(quantization->scale);
  }

  static double MaxGeneratedMagnitude(const TensorDetails& tensor,
                                      bool is_filter, bool is_bias) {
    switch (tensor.element_type) {
      case kLiteRtElementTypeFloat32:
        return 1.0;
      case kLiteRtElementTypeUInt8:
        return 8.0 * QuantizedMaxScale(tensor.quantization);
      case kLiteRtElementTypeInt8:
        return (is_filter ? 16.0 : 32.0) *
               QuantizedMaxScale(tensor.quantization);
      case kLiteRtElementTypeInt16:
        return (is_filter ? 128.0 : 256.0) *
               QuantizedMaxScale(tensor.quantization);
      case kLiteRtElementTypeFloat16:
        return 1.0;
      case kLiteRtElementTypeInt32:
        return (is_bias ? 256.0 : 64.0) *
               QuantizedMaxScale(tensor.quantization);
      case kLiteRtElementTypeInt64:
        return 4096.0 * QuantizedMaxScale(tensor.quantization);
      case kLiteRtElementTypeInt4:
        return 8.0 * QuantizedMaxScale(tensor.quantization);
      case kLiteRtElementTypeInt2:
        return 2.0 * QuantizedMaxScale(tensor.quantization);
      default:
        return 1.0;
    }
  }

  // Computes a dynamic tolerance for floating-point accumulation errors based
  // on reduction size and value ranges. Note that this formula uses float
  // epsilon as its base error and does not account for the significantly
  // larger noise introduced by quantization in hybrid or fixed-point cases.
  double ComputeFloatConformanceTolerance() const {
    const double reduction_size =
        static_cast<double>(params_.filter.dims.back());
    const double max_input = MaxGeneratedMagnitude(params_.input,
                                                   /*is_filter=*/false,
                                                   /*is_bias=*/false);
    const double max_filter = MaxGeneratedMagnitude(params_.filter,
                                                    /*is_filter=*/true,
                                                    /*is_bias=*/false);
    const double max_bias =
        params_.bias.has_value()
            ? MaxGeneratedMagnitude(*params_.bias, /*is_filter=*/false,
                                    /*is_bias=*/true)
            : 0.0;
    const double tolerance =
        static_cast<double>(std::numeric_limits<float>::epsilon()) *
        (reduction_size * max_input * max_filter + max_bias) * 8.0;
    return std::max(1.0e-5, tolerance);
  }

  Expected<void> ReferenceSemantic(
      const SimpleBuffer& input, absl::Span<const uint8_t> filter_bytes,
      const std::optional<absl::Span<const uint8_t>>& bias_bytes,
      SimpleBuffer& output) const {
    const int input_channels = params_.filter.dims.back();
    const int output_channels = params_.filter.dims.front();
    const size_t num_batches =
        Internal::NumElements(params_.input.dims) / input_channels;
    const TensorDetails& reference_filter = params_.stored_filter.has_value()
                                                ? *params_.stored_filter
                                                : params_.filter;
    const std::optional<TensorDetails> reference_bias =
        params_.stored_bias.has_value() ? params_.stored_bias : params_.bias;

    if (Internal::IsHybrid(kDescriptor)) {
      const auto input_view = input.AsView<float>();
      LITERT_ASSIGN_OR_RETURN(const auto filter_values,
                              Internal::DecodeFilterToInt8(
                                  kDescriptor, reference_filter, filter_bytes,
                                  output_channels, input_channels));
      const std::vector<double> bias_values = Internal::DecodeBiasToDouble(
          reference_bias, bias_bytes, output_channels);
      const auto batch_quantization = Internal::QuantizeHybridInputToInt8(
          absl::MakeConstSpan(input_view.data.data(), input_view.NumElements()),
          num_batches, input_channels,
          /*asymmetric_quantize_inputs=*/
          kDescriptor.asymmetric_quantize_inputs);

      std::vector<int32_t> row_sums(output_channels, 0);
      for (int oc = 0; oc < output_channels; ++oc) {
        int32_t row_sum = 0;
        for (int ic = 0; ic < input_channels; ++ic) {
          row_sum +=
              static_cast<int32_t>(filter_values[oc * input_channels + ic]);
        }
        row_sums[oc] = row_sum;
      }

      std::vector<double> real_outputs(num_batches * output_channels);
      for (size_t batch = 0; batch < num_batches; ++batch) {
        for (int oc = 0; oc < output_channels; ++oc) {
          int32_t dotprod = 0;
          for (int ic = 0; ic < input_channels; ++ic) {
            dotprod +=
                static_cast<int32_t>(filter_values[oc * input_channels + ic]) *
                static_cast<int32_t>(
                    batch_quantization.values[batch * input_channels + ic]);
          }
          dotprod -= row_sums[oc] * batch_quantization.zero_points[batch];
          const double scale = batch_quantization.scales[batch] *
                               Internal::QuantizationScaleForChannel(
                                   *params_.filter.quantization, oc);
          const double acc =
              bias_values[oc] + static_cast<double>(dotprod) * scale;
          real_outputs[batch * output_channels + oc] =
              Internal::ApplyActivation(params_.activation, acc);
        }
      }
      return Internal::WriteSemanticOutput(params_.output, real_outputs,
                                           output);
    }

    const std::vector<double> input_values =
        Internal::DecodeInputToDouble(params_.input, input);
    const std::vector<double> filter_values = Internal::DecodeFilterToDouble(
        kDescriptor, reference_filter, filter_bytes, output_channels,
        input_channels);
    const std::vector<double> bias_values = Internal::DecodeBiasToDouble(
        reference_bias, bias_bytes, output_channels);

    std::vector<double> real_outputs(num_batches * output_channels);
    for (size_t batch = 0; batch < num_batches; ++batch) {
      for (int oc = 0; oc < output_channels; ++oc) {
        double acc = bias_values[oc];
        for (int ic = 0; ic < input_channels; ++ic) {
          acc += input_values[batch * input_channels + ic] *
                 filter_values[oc * input_channels + ic];
        }
        real_outputs[batch * output_channels + oc] =
            Internal::ApplyActivation(params_.activation, acc);
      }
    }

    return Internal::WriteSemanticOutput(params_.output, real_outputs, output);
  }

  template <typename Rng>
  static Expected<Params> GenerateParams(Rng& rng) {
    Params params;
    if (kDescriptor.fixed_activation) {
      params.activation = kDescriptor.activation;
    } else if (kDescriptor.weights_format ==
               tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8) {
      params.activation = tflite::ActivationFunctionType_NONE;
    } else {
      params.activation = Internal::kActivations[Internal::RandomInt(
          rng, 0, static_cast<int>(Internal::kActivations.size() - 1))];
    }
    params.keep_num_dims =
        kDescriptor.keep_num_dims_mode >= 0
            ? static_cast<bool>(kDescriptor.keep_num_dims_mode)
            : Internal::RandomBool(rng);

    int rank =
        kDescriptor.fixed_rank > 0
            ? kDescriptor.fixed_rank
            : Internal::RandomInt(rng, 1, static_cast<int>(Internal::kMaxRank));
    int input_channels = Internal::RandomInt(rng, 1, 64);
    int output_channels = Internal::RandomInt(rng, 1, 64);
    if (kDescriptor.filter_type == kLiteRtElementTypeInt4) {
      input_channels = Internal::RoundUp(input_channels, 2);
    } else if (kDescriptor.filter_type == kLiteRtElementTypeInt2) {
      input_channels = Internal::RoundUp(input_channels, 4);
    } else if (kDescriptor.weights_format ==
               tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8) {
      input_channels = Internal::RoundUp(input_channels, 16);
      output_channels = Internal::RoundUp(output_channels, 4);
    }

    Dimensions input_shape(rank, 1);
    int fc_input_channels = input_channels;
    if (kDescriptor.weights_format ==
        tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8) {
      const int batch_size = (rank == 1 || !Internal::RandomBool(rng)) ? 1 : 4;
      if (rank > 1) {
        input_shape[0] = batch_size;
      }
    } else {
      for (int i = 0; i < rank - 1; ++i) {
        input_shape[i] = Internal::RandomInt(rng, 1, 4);
      }
    }
    input_shape.back() = input_channels;
    if (kDescriptor.flatten_all_non_batch_dims) {
      fc_input_channels = 1;
      for (int i = 1; i < rank; ++i) {
        fc_input_channels *= input_shape[i];
      }
    }

    Dimensions output_shape;
    if (params.keep_num_dims) {
      output_shape = input_shape;
      output_shape.back() = output_channels;
    } else {
      output_shape = {
          static_cast<int32_t>(
              Internal::NumElements(absl::MakeConstSpan(input_shape)) /
              fc_input_channels),
          output_channels};
    }

    const bool has_bias =
        kDescriptor.bias_always_present
            ? true
            : (kDescriptor.bias_never_present ? false
                                              : Internal::RandomBool(rng));
    const float input_scale =
        kDescriptor.input_type == kLiteRtElementTypeInt16
            ? Internal::RandomFloat(rng, 1.0f / 2048.0f, 1.0f / 512.0f)
            : Internal::RandomFloat(rng, 1.0f / 128.0f, 1.0f / 32.0f);
    const float base_filter_scale =
        kDescriptor.filter_type == kLiteRtElementTypeInt16
            ? Internal::RandomFloat(rng, 1.0f / 2048.0f, 1.0f / 256.0f)
            : Internal::RandomFloat(rng, 1.0f / 128.0f, 1.0f / 16.0f);
    const auto filter_scales =
        kDescriptor.per_channel
            ? Internal::GenerateFloatValues(rng, output_channels,
                                            base_filter_scale * 0.5f,
                                            base_filter_scale * 1.5f)
            : std::vector<float>{base_filter_scale};
    const float max_filter_scale =
        *std::max_element(filter_scales.begin(), filter_scales.end());
    const float output_scale =
        kDescriptor.output_type == kLiteRtElementTypeInt16
            ? std::max(input_scale * max_filter_scale, 1e-6f)
            : std::max(
                  input_scale * max_filter_scale * fc_input_channels * 4.0f,
                  1e-6f);

    params.input = TensorDetails{{input_shape.begin(), input_shape.end()},
                                 kDescriptor.input_type,
                                 "input"};
    params.filter = TensorDetails{{output_channels, fc_input_channels},
                                  kDescriptor.filter_type,
                                  "filter"};
    params.output = TensorDetails{{output_shape.begin(), output_shape.end()},
                                  kDescriptor.output_type,
                                  "output"};
    if (kDescriptor.fp16_filter) {
      params.stored_filter = TensorDetails{{output_channels, fc_input_channels},
                                           kLiteRtElementTypeFloat16,
                                           "filter_fp16"};
    }

    if (kDescriptor.input_type != kLiteRtElementTypeFloat32) {
      int64_t input_zero_point =
          kDescriptor.input_type == kLiteRtElementTypeUInt8 ? 128 : 0;
      int64_t output_zero_point =
          kDescriptor.output_type == kLiteRtElementTypeUInt8 ? 128 : 0;
      int64_t filter_zero_point =
          kDescriptor.filter_type == kLiteRtElementTypeUInt8 ? 128 : 0;
      params.input.quantization = TensorDetails::QuantizationDetails::PerTensor(
          input_scale, input_zero_point);
      params.output.quantization =
          TensorDetails::QuantizationDetails::PerTensor(output_scale,
                                                        output_zero_point);
      if (kDescriptor.per_channel) {
        params.filter.quantization =
            TensorDetails::QuantizationDetails::PerChannel(
                /*quantized_dimension=*/0, filter_scales,
                std::vector<int64_t>(output_channels, 0));
      } else {
        params.filter.quantization =
            TensorDetails::QuantizationDetails::PerTensor(filter_scales.front(),
                                                          filter_zero_point);
      }
      if (has_bias) {
        if (kDescriptor.per_channel) {
          std::vector<float> bias_scales(output_channels);
          for (int i = 0; i < output_channels; ++i) {
            bias_scales[i] = input_scale * filter_scales[i];
          }
          params.bias = TensorDetails{
              {output_channels},
              kDescriptor.bias_type,
              "bias",
              std::nullopt,
              TensorDetails::QuantizationDetails::PerChannel(
                  /*quantized_dimension=*/0, std::move(bias_scales),
                  std::vector<int64_t>(output_channels, 0))};
        } else {
          params.bias =
              TensorDetails{{output_channels},
                            kDescriptor.bias_type,
                            "bias",
                            std::nullopt,
                            TensorDetails::QuantizationDetails::PerTensor(
                                input_scale * filter_scales.front(), 0)};
        }
      }
    } else {
      if (Internal::IsHybrid(kDescriptor)) {
        if (kDescriptor.per_channel) {
          params.filter.quantization =
              TensorDetails::QuantizationDetails::PerChannel(
                  /*quantized_dimension=*/0, filter_scales,
                  std::vector<int64_t>(output_channels, 0));
        } else {
          params.filter.quantization =
              TensorDetails::QuantizationDetails::PerTensor(
                  filter_scales.front(), 0);
        }
      }
      if (has_bias) {
        params.bias =
            TensorDetails{{output_channels}, kDescriptor.bias_type, "bias"};
        if (kDescriptor.fp16_bias) {
          params.stored_bias = TensorDetails{
              {output_channels}, kLiteRtElementTypeFloat16, "bias_fp16"};
        }
      }
    }

    if (params.stored_filter.has_value()) {
      const auto filter_bytes =
          GenerateTensorBytes(rng, *params.stored_filter,
                              /*is_filter=*/true,
                              /*is_bias=*/false, kDescriptor.weights_format);
      params.stored_filter->data = Internal::MakeOwningBytes(filter_bytes);
    } else if (!kDescriptor.dynamic_filter) {
      const auto filter_bytes =
          GenerateTensorBytes(rng, params.filter, /*is_filter=*/true,
                              /*is_bias=*/false, kDescriptor.weights_format);
      params.filter.data = Internal::MakeOwningBytes(filter_bytes);
    }
    if (params.bias.has_value() && !kDescriptor.dynamic_bias) {
      if (params.stored_bias.has_value()) {
        const auto bias_bytes =
            GenerateTensorBytes(rng, *params.stored_bias, /*is_filter=*/false,
                                /*is_bias=*/true);
        params.stored_bias->data = Internal::MakeOwningBytes(bias_bytes);
      } else {
        const auto bias_bytes =
            GenerateTensorBytes(rng, *params.bias, /*is_filter=*/false,
                                /*is_bias=*/true);
        params.bias->data = Internal::MakeOwningBytes(bias_bytes);
      }
    }

    if (kDescriptor.weights_format ==
        tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8) {
      params.internal_outputs.push_back(TensorDetails{
          {static_cast<int32_t>(
              Internal::NumElements(absl::MakeConstSpan(input_shape)))},
          kLiteRtElementTypeUInt8,
          "shuffled_workspace"});
    }

    return params;
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    if (params.stored_filter.has_value() || params.stored_bias.has_value()) {
      return BuildDequantizeFullyConnectedGraph(params);
    }
    std::vector<TensorDetails> inputs = {params.input, params.filter};
    if (params.bias.has_value()) {
      inputs.push_back(*params.bias);
    }
    const auto quantized_bias_type =
        Internal::ToTensorType(Internal::QuantizedBiasType(kDescriptor));
    if (params.internal_outputs.empty()) {
      return SingleOpModel<kLiteRtOpCodeTflFullyConnected>(
          inputs, {params.output}, params.activation,
          kDescriptor.weights_format, params.keep_num_dims,
          /*asymmetric_quantize_inputs=*/kDescriptor.asymmetric_quantize_inputs,
          quantized_bias_type);
    }
    return SingleOpModelWithInternalOutputs<kLiteRtOpCodeTflFullyConnected>(
        inputs, {params.output}, params.internal_outputs, params.activation,
        kDescriptor.weights_format, params.keep_num_dims,
        /*asymmetric_quantize_inputs=*/kDescriptor.asymmetric_quantize_inputs,
        quantized_bias_type);
  }

  static Expected<LiteRtModelT::Ptr> BuildDequantizeFullyConnectedGraph(
      const Params& params) {
    if (!params.internal_outputs.empty()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "FC internal outputs are unsupported for fp16-weight graph");
    }

    using DequantizeOpDetails = OpDetails<kLiteRtOpCodeTflDequantize>;
    using FullyConnectedOpDetails = OpDetails<kLiteRtOpCodeTflFullyConnected>;

    LiteRtModelT model;
    std::vector<TflOpCodePtr> tfl_codes;
    auto& sg = model.EmplaceSubgraph();

    DequantizeOpDetails dequantize_details;
    const int dequantize_opcode_index = 0;
    tfl_codes.push_back(
        dequantize_details.MakeTflCode(dequantize_details.Version({}, {})));

    FullyConnectedOpDetails fully_connected_details(
        params.activation, kDescriptor.weights_format, params.keep_num_dims,
        /*asymmetric_quantize_inputs=*/kDescriptor.asymmetric_quantize_inputs,
        Internal::ToTensorType(Internal::QuantizedBiasType(kDescriptor)));
    std::vector<TensorDetails> fully_connected_inputs = {params.input,
                                                         params.filter};
    if (params.bias.has_value()) {
      fully_connected_inputs.push_back(*params.bias);
    }
    const int fully_connected_opcode_index = 1;
    tfl_codes.push_back(
        fully_connected_details.MakeTflCode(fully_connected_details.Version(
            fully_connected_inputs, {params.output})));

    std::vector<std::string> input_names;
    std::vector<LiteRtTensor> input_tensors;
    std::vector<std::string> output_names;
    std::vector<LiteRtTensor> output_tensors;

    auto add_tensor = [&](const TensorDetails& tensor_details,
                          bool is_subgraph_input,
                          bool is_subgraph_output) -> LiteRtTensor {
      auto& tensor = sg.EmplaceTensor();
      tensor.SetType(::MakeRankedTensorType(tensor_details.element_type,
                                            tensor_details.dims));
      tensor.SetName(tensor_details.name);
      SetTensorQuantization(tensor, tensor_details.quantization);
      if (tensor_details.data.has_value()) {
        ::SetWeightsFromUnownedBuffer(tensor.Weights(), *tensor_details.data);
      } else if (is_subgraph_input) {
        sg.Inputs().push_back(&tensor);
        input_names.push_back(tensor_details.name);
        input_tensors.push_back(&tensor);
      }
      if (is_subgraph_output) {
        sg.Outputs().push_back(&tensor);
        output_names.push_back(tensor_details.name);
        output_tensors.push_back(&tensor);
      }
      return &tensor;
    };

    LiteRtTensor input_tensor =
        add_tensor(params.input, /*is_subgraph_input=*/true,
                   /*is_subgraph_output=*/false);
    LiteRtTensor filter_tensor =
        add_tensor(params.filter, /*is_subgraph_input=*/false,
                   /*is_subgraph_output=*/false);
    LiteRtTensor stored_filter_tensor = nullptr;
    if (params.stored_filter.has_value()) {
      stored_filter_tensor = add_tensor(*params.stored_filter,
                                        /*is_subgraph_input=*/false,
                                        /*is_subgraph_output=*/false);
    }

    LiteRtTensor bias_tensor = nullptr;
    LiteRtTensor stored_bias_tensor = nullptr;
    if (params.bias.has_value()) {
      bias_tensor = add_tensor(*params.bias, /*is_subgraph_input=*/false,
                               /*is_subgraph_output=*/false);
      if (params.stored_bias.has_value()) {
        stored_bias_tensor = add_tensor(*params.stored_bias,
                                        /*is_subgraph_input=*/false,
                                        /*is_subgraph_output=*/false);
      }
    }

    LiteRtTensor output_tensor = add_tensor(params.output,
                                            /*is_subgraph_input=*/false,
                                            /*is_subgraph_output=*/true);

    auto add_dequantize_op = [&](LiteRtTensor source_tensor,
                                 LiteRtTensor destination_tensor) {
      auto& op = sg.EmplaceOp();
      op.SetOpCode(kLiteRtOpCodeTflDequantize);
      SetTflOptions(op, dequantize_details.MakeTflOptions());
      SetTflOpCodeInd(op, dequantize_opcode_index);
      AttachInput(source_tensor, op);
      AttachOutput(destination_tensor, op);
    };

    if (stored_filter_tensor != nullptr) {
      add_dequantize_op(stored_filter_tensor, filter_tensor);
    }
    if (stored_bias_tensor != nullptr) {
      add_dequantize_op(stored_bias_tensor, bias_tensor);
    }

    auto& fully_connected_op = sg.EmplaceOp();
    fully_connected_op.SetOpCode(kLiteRtOpCodeTflFullyConnected);
    SetTflOptions(fully_connected_op, fully_connected_details.MakeTflOptions());
    SetTflOpCodeInd(fully_connected_op, fully_connected_opcode_index);
    AttachInput(input_tensor, fully_connected_op);
    AttachInput(filter_tensor, fully_connected_op);
    if (bias_tensor != nullptr) {
      AttachInput(bias_tensor, fully_connected_op);
    }
    AttachOutput(output_tensor, fully_connected_op);

    model.EmplaceSignature(&sg, std::move(input_names),
                           std::move(input_tensors), std::move(output_names),
                           std::move(output_tensors), "default");
    SetTflOpCodes(model, std::move(tfl_codes));
    LITERT_ASSIGN_OR_RETURN(auto serialized, SerializeModel(std::move(model)));
    return LoadModelFromBuffer(std::move(serialized));
  }

  template <typename Rng>
  static std::vector<uint8_t> GenerateTensorBytes(
      Rng& rng, const TensorDetails& tensor, bool is_filter, bool is_bias,
      tflite::FullyConnectedOptionsWeightsFormat weights_format =
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT) {
    const size_t num_elements = Internal::NumElements(tensor.dims);
    switch (tensor.element_type) {
      case kLiteRtElementTypeFloat32:
        return Internal::CopyToBytes(
            Internal::GenerateFloatValues(rng, num_elements, -1.0f, 1.0f));
      case kLiteRtElementTypeFloat16: {
        const auto fp32_values =
            Internal::GenerateFloatValues(rng, num_elements, -1.0f, 1.0f);
        const auto fp16_values =
            Internal::Float32ToFp16(absl::MakeConstSpan(fp32_values));
        return Internal::CopyToBytes(fp16_values);
      }
      case kLiteRtElementTypeUInt8:
        if (is_filter &&
            weights_format ==
                tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8) {
          return Internal::ShuffleAndXorUint8Weights(
              rng, tensor.dims[1], tensor.dims[0],
              static_cast<int>(tensor.quantization->zero_point));
        }
        return Internal::CopyToBytes(Internal::GenerateIntegralValues<uint8_t>(
            rng, num_elements,
            std::max(0, static_cast<int>(tensor.quantization->zero_point) - 8),
            std::min(255,
                     static_cast<int>(tensor.quantization->zero_point) + 8)));
      case kLiteRtElementTypeInt8:
        return Internal::CopyToBytes(Internal::GenerateIntegralValues<int8_t>(
            rng, num_elements, is_filter ? -16 : -32, is_filter ? 15 : 31));
      case kLiteRtElementTypeInt16:
        return Internal::CopyToBytes(Internal::GenerateIntegralValues<int16_t>(
            rng, num_elements, is_filter ? -128 : -256, is_filter ? 127 : 255));
      case kLiteRtElementTypeInt32:
        return Internal::CopyToBytes(Internal::GenerateIntegralValues<int32_t>(
            rng, num_elements, is_bias ? -256 : -64, is_bias ? 256 : 64));
      case kLiteRtElementTypeInt64:
        return Internal::CopyToBytes(Internal::GenerateIntegralValues<int64_t>(
            rng, num_elements, -4096, 4096));
      case kLiteRtElementTypeInt4: {
        const auto unpacked =
            Internal::GenerateIntegralValues<int8_t>(rng, num_elements, -8, 7);
        return Internal::PackSignedData(unpacked, /*bit_width=*/4);
      }
      case kLiteRtElementTypeInt2: {
        const auto unpacked =
            Internal::GenerateIntegralValues<int8_t>(rng, num_elements, -2, 1);
        return Internal::PackSignedData(unpacked, /*bit_width=*/2);
      }
      default:
        return {};
    }
  }

  Expected<void> ReferenceFloat(
      const SimpleBuffer& input, absl::Span<const uint8_t> filter_bytes,
      const std::optional<absl::Span<const uint8_t>>& bias_bytes,
      SimpleBuffer& output) const {
    const auto input_view = input.AsView<float>();
    auto output_view = output.AsView<float>();
    const auto filter_values = Internal::DequantizeFilterToFloat(
        kDescriptor, params_.filter, filter_bytes, params_.filter.dims[0],
        params_.filter.dims[1]);
    const std::vector<float> bias_values =
        bias_bytes.has_value() ? Internal::BytesToVector<float>(*bias_bytes)
                               : std::vector<float>();
    const auto op_params =
        Internal::MakeFloatReferenceParams(params_.activation);
    const auto input_shape = Internal::MakeRuntimeShape(params_.input.dims);
    const auto filter_shape = Internal::MakeRuntimeShape(params_.filter.dims);
    const auto bias_shape = params_.bias.has_value()
                                ? Internal::MakeRuntimeShape(params_.bias->dims)
                                : tflite::RuntimeShape();
    const auto output_shape = Internal::MakeRuntimeShape(params_.output.dims);
    tflite::reference_ops::FullyConnected(
        op_params, input_shape, input_view.data.data(), filter_shape,
        filter_values.data(), bias_shape,
        params_.bias.has_value() ? bias_values.data() : nullptr, output_shape,
        output_view.data.data());
    return {};
  }

  Expected<void> ReferenceUint8(
      const SimpleBuffer& input, absl::Span<const uint8_t> filter_bytes,
      const std::optional<absl::Span<const uint8_t>>& bias_bytes,
      SimpleBuffer& output) const {
    const auto input_view = input.AsView<uint8_t>();
    auto op_params = Internal::MakeQuantizedReferenceParams(
        params_.activation, params_.input, params_.filter, params_.output,
        /*per_channel=*/false);
    Internal::PopulateScalarOutputMultiplier(params_.input, params_.filter,
                                             params_.output, &op_params);
    const auto input_shape = Internal::MakeRuntimeShape(params_.input.dims);
    const auto filter_shape = Internal::MakeRuntimeShape(params_.filter.dims);
    const auto bias_shape = params_.bias.has_value()
                                ? Internal::MakeRuntimeShape(params_.bias->dims)
                                : tflite::RuntimeShape();
    const auto output_shape = Internal::MakeRuntimeShape(params_.output.dims);
    const std::vector<int32_t> bias_values =
        bias_bytes.has_value() ? Internal::BytesToVector<int32_t>(*bias_bytes)
                               : std::vector<int32_t>();

    if (kDescriptor.weights_format ==
        tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8) {
      auto output_view = output.AsView<int16_t>();
      std::vector<uint8_t> workspace(input_view.NumElements());
      tflite::reference_ops::ShuffledFullyConnected(
          op_params, input_shape, input_view.data.data(), filter_shape,
          filter_bytes.data(), bias_shape,
          params_.bias.has_value() ? bias_values.data() : nullptr, output_shape,
          output_view.data.data(), workspace.data());
      return {};
    }

    if (kDescriptor.output_type == kLiteRtElementTypeUInt8) {
      auto output_view = output.AsView<uint8_t>();
      tflite::reference_ops::FullyConnected(
          op_params, input_shape, input_view.data.data(), filter_shape,
          filter_bytes.data(), bias_shape,
          params_.bias.has_value() ? bias_values.data() : nullptr, output_shape,
          output_view.data.data());
      return {};
    }

    if (!params_.bias.has_value()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Uint8 to int16 fully connected requires bias");
    }
    auto output_view = output.AsView<int16_t>();
    tflite::reference_ops::FullyConnected(
        op_params, input_shape, input_view.data.data(), filter_shape,
        filter_bytes.data(), bias_shape, bias_values.data(), output_shape,
        output_view.data.data());
    return {};
  }

  Expected<void> ReferenceInt8(
      const SimpleBuffer& input, absl::Span<const uint8_t> filter_bytes,
      const std::optional<absl::Span<const uint8_t>>& bias_bytes,
      SimpleBuffer& output) const {
    const auto input_view = input.AsView<int8_t>();
    auto output_view = output.AsView<int8_t>();
    auto op_params = Internal::MakeQuantizedReferenceParams(
        params_.activation, params_.input, params_.filter, params_.output,
        kDescriptor.per_channel);
    const auto input_shape = Internal::MakeRuntimeShape(params_.input.dims);
    const auto filter_shape = Internal::MakeRuntimeShape(params_.filter.dims);
    const auto bias_shape = params_.bias.has_value()
                                ? Internal::MakeRuntimeShape(params_.bias->dims)
                                : tflite::RuntimeShape();
    const auto output_shape = Internal::MakeRuntimeShape(params_.output.dims);
    const auto filter_values =
        params_.filter.element_type == kLiteRtElementTypeInt8
            ? Internal::BytesToVector<int8_t>(filter_bytes)
            : Internal::UnpackSignedData(
                  filter_bytes,
                  static_cast<int>(Internal::NumElements(params_.filter.dims)),
                  params_.filter.element_type == kLiteRtElementTypeInt4 ? 4
                                                                        : 2);
    const auto bias_values = bias_bytes.has_value()
                                 ? Internal::BytesToVector<int32_t>(*bias_bytes)
                                 : std::vector<int32_t>();

    if (kDescriptor.per_channel) {
      std::vector<int32_t> output_multiplier;
      std::vector<int> output_shift;
      Internal::PopulatePerChannelOutputMultipliers(
          params_.input, params_.filter, params_.output, &output_multiplier,
          &output_shift);
      Internal::ReferencePerChannelInt<int8_t, int8_t, int32_t, int8_t>(
          op_params, output_multiplier.data(), output_shift.data(), input_shape,
          input_view.data.data(), filter_shape, filter_values.data(),
          bias_shape, params_.bias.has_value() ? bias_values.data() : nullptr,
          output_shape, output_view.data.data());
      return {};
    }

    Internal::PopulateScalarOutputMultiplier(params_.input, params_.filter,
                                             params_.output, &op_params);
    Internal::ReferencePerTensorInt<int8_t, int8_t, int32_t, int8_t>(
        op_params, input_shape, input_view.data.data(), filter_shape,
        filter_values.data(), bias_shape,
        params_.bias.has_value() ? bias_values.data() : nullptr, output_shape,
        output_view.data.data());
    return {};
  }

  template <typename FilterT, typename BiasT>
  Expected<void> ReferenceInt16Impl(
      const SimpleBuffer& input, absl::Span<const FilterT> filter_values,
      const std::optional<absl::Span<const uint8_t>>& bias_bytes,
      SimpleBuffer& output) const {
    const auto input_view = input.AsView<int16_t>();
    auto output_view = output.AsView<int16_t>();
    auto op_params = Internal::MakeQuantizedReferenceParams(
        params_.activation, params_.input, params_.filter, params_.output,
        kDescriptor.per_channel);
    const auto input_shape = Internal::MakeRuntimeShape(params_.input.dims);
    const auto filter_shape = Internal::MakeRuntimeShape(params_.filter.dims);
    const auto bias_shape = params_.bias.has_value()
                                ? Internal::MakeRuntimeShape(params_.bias->dims)
                                : tflite::RuntimeShape();
    const auto output_shape = Internal::MakeRuntimeShape(params_.output.dims);
    const auto bias_values = bias_bytes.has_value()
                                 ? Internal::BytesToVector<BiasT>(*bias_bytes)
                                 : std::vector<BiasT>();

    if (kDescriptor.per_channel) {
      std::vector<int32_t> output_multiplier;
      std::vector<int> output_shift;
      Internal::PopulatePerChannelOutputMultipliers(
          params_.input, params_.filter, params_.output, &output_multiplier,
          &output_shift);
      Internal::ReferencePerChannelInt<int16_t, FilterT, BiasT, int16_t>(
          op_params, output_multiplier.data(), output_shift.data(), input_shape,
          input_view.data.data(), filter_shape, filter_values.data(),
          bias_shape, params_.bias.has_value() ? bias_values.data() : nullptr,
          output_shape, output_view.data.data());
      return {};
    }

    Internal::PopulateScalarOutputMultiplier(params_.input, params_.filter,
                                             params_.output, &op_params);
    Internal::ReferencePerTensorInt<int16_t, FilterT, BiasT, int16_t>(
        op_params, input_shape, input_view.data.data(), filter_shape,
        filter_values.data(), bias_shape,
        params_.bias.has_value() ? bias_values.data() : nullptr, output_shape,
        output_view.data.data());
    return {};
  }

  Expected<void> ReferenceInt16(
      const SimpleBuffer& input, absl::Span<const uint8_t> filter_bytes,
      const std::optional<absl::Span<const uint8_t>>& bias_bytes,
      SimpleBuffer& output) const {
    if (kDescriptor.filter_type == kLiteRtElementTypeInt16) {
      const auto filter_values = Internal::BytesToVector<int16_t>(filter_bytes);
      if (kDescriptor.bias_type == kLiteRtElementTypeInt64) {
        return ReferenceInt16Impl<int16_t, int64_t>(
            input, absl::MakeConstSpan(filter_values), bias_bytes, output);
      }
      return ReferenceInt16Impl<int16_t, int32_t>(
          input, absl::MakeConstSpan(filter_values), bias_bytes, output);
    }

    const auto filter_values =
        params_.filter.element_type == kLiteRtElementTypeInt8
            ? Internal::BytesToVector<int8_t>(filter_bytes)
            : Internal::UnpackSignedData(
                  filter_bytes,
                  static_cast<int>(Internal::NumElements(params_.filter.dims)),
                  params_.filter.element_type == kLiteRtElementTypeInt4 ? 4
                                                                        : 2);
    if (kDescriptor.bias_type == kLiteRtElementTypeInt64) {
      return ReferenceInt16Impl<int8_t, int64_t>(
          input, absl::MakeConstSpan(filter_values), bias_bytes, output);
    }
    return ReferenceInt16Impl<int8_t, int32_t>(
        input, absl::MakeConstSpan(filter_values), bias_bytes, output);
  }

  Params params_;
};

using FullyConnectedPresets = FullyConnectedPresetListC<
    FullyConnectedPreset::kFloatStatic,
    FullyConnectedPreset::kFloatStaticNoBias,
    FullyConnectedPreset::kFloatStatic1d,
    FullyConnectedPreset::kFloatStatic1dKeepDims,
    FullyConnectedPreset::kFloatStatic2d,
    FullyConnectedPreset::kFloatStatic2dKeepDims,
    FullyConnectedPreset::kFloatStatic3d,
    FullyConnectedPreset::kFloatStaticRelu,
    FullyConnectedPreset::kFloatStaticRelu6,
    FullyConnectedPreset::kFloatStaticReluN1To1,
    FullyConnectedPreset::kFloatStatic3dReshape,
    FullyConnectedPreset::kFloatStatic3dKeepDims,
    FullyConnectedPreset::kFloatStatic4d,
    FullyConnectedPreset::kFloatStatic4dKeepDims,
    FullyConnectedPreset::kFloatFp16WeightsStatic,
    FullyConnectedPreset::kFloatFp16WeightsStaticF32Bias,
    FullyConnectedPreset::kFloatFp16WeightsStaticNoBias,
    FullyConnectedPreset::kFloatDynamic,
    FullyConnectedPreset::kFloatDynamicFilter,
    FullyConnectedPreset::kFloatDynamicFilterNoBias,
    FullyConnectedPreset::kFloatDynamicBias,
    FullyConnectedPreset::kHybridInt8Static,
    FullyConnectedPreset::kHybridInt8Dynamic,
    FullyConnectedPreset::kHybridInt8PerChannelStatic,
    FullyConnectedPreset::kHybridInt8PerChannelDynamic,
    FullyConnectedPreset::kHybridInt8AsymmetricStatic,
    FullyConnectedPreset::kHybridInt8AsymmetricDynamic,
    FullyConnectedPreset::kHybridInt8PerChannelAsymmetricStatic,
    FullyConnectedPreset::kHybridInt8PerChannelAsymmetricDynamic,
    FullyConnectedPreset::kUint8Static1d,
    FullyConnectedPreset::kUint8Static1dKeepDims,
    FullyConnectedPreset::kUint8Static2d,
    FullyConnectedPreset::kUint8Static2dKeepDims,
    FullyConnectedPreset::kUint8Static3d,
    FullyConnectedPreset::kUint8Static3dReshape,
    FullyConnectedPreset::kUint8Static3dKeepDims,
    FullyConnectedPreset::kUint8Static4d,
    FullyConnectedPreset::kUint8Static4dKeepDims,
    FullyConnectedPreset::kUint8Static,
    FullyConnectedPreset::kUint8StaticNoBias,
    FullyConnectedPreset::kUint8StaticRelu,
    FullyConnectedPreset::kUint8StaticRelu6,
    FullyConnectedPreset::kUint8StaticReluN1To1,
    FullyConnectedPreset::kUint8Int16Static,
    FullyConnectedPreset::kUint8ShuffledStatic,
    FullyConnectedPreset::kInt8Static1d,
    FullyConnectedPreset::kInt8Static1dKeepDims,
    FullyConnectedPreset::kInt8Static2d,
    FullyConnectedPreset::kInt8Static2dKeepDims,
    FullyConnectedPreset::kInt8Static3d,
    FullyConnectedPreset::kInt8Static3dReshape,
    FullyConnectedPreset::kInt8Static3dKeepDims,
    FullyConnectedPreset::kInt8Static4d,
    FullyConnectedPreset::kInt8Static4dKeepDims,
    FullyConnectedPreset::kInt8Static, FullyConnectedPreset::kInt8StaticRelu,
    FullyConnectedPreset::kInt8StaticRelu6,
    FullyConnectedPreset::kInt8StaticReluN1To1,
    FullyConnectedPreset::kInt8StaticNoBias,
    FullyConnectedPreset::kInt8PerChannelStatic,
    FullyConnectedPreset::kInt8PerChannelStaticNoBias,
    FullyConnectedPreset::kInt8Int4PerChannelStatic,
    FullyConnectedPreset::kInt8Int2PerChannelStatic,
    FullyConnectedPreset::kInt16Int8StaticInt32Bias,
    FullyConnectedPreset::kInt16Int8PerChannelStaticInt32Bias,
    FullyConnectedPreset::kInt16Int8StaticInt64Bias,
    FullyConnectedPreset::kInt16Int8PerChannelStaticInt64Bias,
    FullyConnectedPreset::kInt16Int16StaticInt32Bias,
    FullyConnectedPreset::kInt16Int16PerChannelStaticInt32Bias,
    FullyConnectedPreset::kInt16Int16StaticInt64Bias,
    FullyConnectedPreset::kInt16Int16PerChannelStaticInt64Bias,
    FullyConnectedPreset::kInt16Int4PerChannelStaticInt32Bias,
    FullyConnectedPreset::kInt16Int4PerChannelStaticInt64Bias,
    FullyConnectedPreset::kInt16Int2PerChannelStaticInt32Bias,
    FullyConnectedPreset::kInt16Int2PerChannelStaticInt64Bias>;

using FullyConnectedAtsCpuPresets = FullyConnectedPresets;

using FullyConnectedAtsCompilePresets =
    FullyConnectedPresetListC<FullyConnectedPreset::kFloatStatic2d,
                              FullyConnectedPreset::kFloatStatic3d,
                              FullyConnectedPreset::kFloatStatic4d>;

using FullyConnectedAtsGpuPresets = FullyConnectedAtsCompilePresets;

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_FULLY_CONNECTED_H_
