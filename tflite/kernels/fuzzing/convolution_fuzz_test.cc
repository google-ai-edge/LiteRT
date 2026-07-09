/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"
#include "tflite/kernels/fuzzing/fuzzing_util.h"
#include "tflite/kernels/fuzzing/one_op_fuzz_model.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace builtin {

TfLiteRegistration* Register_CONVOLUTION_GENERIC_OPT();
TfLiteRegistration* Register_CONVOLUTION_MULTITHREADED_OPT();
TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_GENERIC_OPT();
TfLiteRegistration* Register_TRANSPOSECONV_GENERIC_OPT();
TfLiteRegistration* Register_CONV_3D_GENERIC_OPT();
TfLiteRegistration* Register_CONV_3D_TRANSPOSE();

}  // namespace builtin
}  // namespace ops

namespace {

using fuzzing::RunResult;

constexpr size_t kMaxTensorBytesToMaterialize = 4096;
constexpr size_t kMaxFuzzerLiveAllocationBytes = 64 * 1024 * 1024;
constexpr int64_t kMaxInt32Value = std::numeric_limits<int32_t>::max();

enum class Conv2DKernel { kGenericOptimized, kMultithreadedOptimized };

struct Conv2DCase {
  std::vector<int32_t> input_shape;
  std::vector<int32_t> filter_shape;
  TensorType input_type;
  TensorType filter_type;
  TensorType output_type;
  Conv2DKernel kernel;
  Padding padding;
  int32_t stride_width;
  int32_t stride_height;
  int32_t dilation_width;
  int32_t dilation_height;
  bool force_persistent_filter;
  bool invoke;
};

struct DepthwiseConvCase {
  std::vector<int32_t> input_shape;
  std::vector<int32_t> filter_shape;
  TensorType input_type;
  TensorType filter_type;
  TensorType output_type;
  Padding padding;
  int32_t stride_width;
  int32_t stride_height;
  int32_t dilation_width;
  int32_t dilation_height;
  bool invoke;
};

struct TransposeConvCase {
  std::vector<int32_t> output_shape;
  std::vector<int32_t> filter_shape;
  std::vector<int32_t> input_shape;
  TensorType input_type;
  TensorType filter_type;
  TensorType output_type;
  Padding padding;
  int32_t stride_width;
  int32_t stride_height;
  bool invoke;
};

struct Conv3DCase {
  std::vector<int32_t> input_shape;
  std::vector<int32_t> filter_shape;
  Padding padding;
  int32_t stride_depth;
  int32_t stride_width;
  int32_t stride_height;
  int32_t dilation_depth;
  int32_t dilation_width;
  int32_t dilation_height;
  bool invoke;
};

struct Conv3DTransposeCase {
  std::vector<int32_t> output_shape;
  std::vector<int32_t> filter_shape;
  std::vector<int32_t> input_shape;
  Padding padding;
  int32_t stride_depth;
  int32_t stride_width;
  int32_t stride_height;
  int32_t dilation_depth;
  int32_t dilation_width;
  int32_t dilation_height;
  bool invoke;
};

int32_t DimOr(const std::vector<int32_t>& shape, size_t index,
              int32_t fallback) {
  return index < shape.size() ? shape[index] : fallback;
}

int32_t PositiveDimOr(const std::vector<int32_t>& shape, size_t index,
                      int32_t fallback) {
  return std::max<int32_t>(1, DimOr(shape, index, fallback));
}

int32_t BiasChannelsOrOne(const std::vector<int32_t>& filter_shape,
                          size_t index) {
  const int32_t channels = DimOr(filter_shape, index, 1);
  if (channels < 0) {
    return 1;
  }
  return channels;
}

bool SmallEnoughToMaterialize(TensorType type,
                              const std::vector<int32_t>& shape,
                              size_t* element_count) {
  size_t elements = 0;
  if (!fuzzing::CheckedShapeElementCount(shape, &elements)) {
    return false;
  }
  const size_t type_size = fuzzing::TypeSize(type);
  size_t bytes = 0;
  if (type_size == 0 ||
      !fuzzing::StorageBytesForElements(type, elements, &bytes) ||
      bytes > kMaxTensorBytesToMaterialize) {
    return false;
  }
  *element_count = elements;
  return true;
}

std::vector<uint8_t> MakeTensorBytes(TensorType type,
                                     const std::vector<int32_t>& shape,
                                     int64_t seed) {
  size_t elements = 0;
  if (!SmallEnoughToMaterialize(type, shape, &elements)) {
    return {};
  }
  return fuzzing::MakeValues(type, elements, seed);
}

std::vector<uint8_t> MakeInt32Bytes(const std::vector<int32_t>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(int32_t), 0);
  for (size_t i = 0; i < values.size(); ++i) {
    std::memcpy(bytes.data() + i * sizeof(int32_t), &values[i],
                sizeof(int32_t));
  }
  return bytes;
}

flatbuffers::Offset<QuantizationParameters> MakeQuantization(
    flatbuffers::FlatBufferBuilder* builder, TensorType type,
    int32_t quantized_dimension, int32_t channels) {
  if (type != TensorType_UINT8 && type != TensorType_INT8 &&
      type != TensorType_INT4 && type != TensorType_INT16) {
    return 0;
  }
  channels = std::max<int32_t>(1, std::min<int32_t>(channels, 8));
  std::vector<float> scales(channels, 0.25f);
  std::vector<int64_t> zero_points(channels, 0);
  return CreateQuantizationParameters(
      *builder, 0, 0, builder->CreateVector(scales),
      builder->CreateVector(zero_points), QuantizationDetails_NONE, 0,
      quantized_dimension);
}

flatbuffers::Offset<Tensor> MakeTensor(
    flatbuffers::FlatBufferBuilder* builder,
    const std::vector<int32_t>& shape, TensorType type, uint32_t buffer,
    flatbuffers::Offset<QuantizationParameters> quantization = 0) {
  return CreateTensor(*builder, builder->CreateVector(shape), type, buffer, 0,
                      quantization);
}

RunResult BuildAndRun(flatbuffers::FlatBufferBuilder* builder,
                      BuiltinOperator builtin_operator,
                      BuiltinOptions builtin_options_type,
                      flatbuffers::Offset<void> builtin_options,
                      TfLiteRegistration* registration, int min_version,
                      int max_version,
                      const std::vector<flatbuffers::Offset<Tensor>>& tensors,
                      const std::vector<flatbuffers::Offset<Buffer>>& buffers,
                      const std::vector<int32_t>& model_inputs,
                      const std::vector<int32_t>& model_outputs,
                      const std::vector<int32_t>& op_inputs,
                      const std::vector<int32_t>& op_outputs,
                      const std::vector<std::vector<int32_t>>& input_shapes,
                      const std::vector<TensorType>& input_types,
                      bool force_persistent_filter = false,
                      int persistent_filter_tensor_index = -1,
                      bool invoke = false,
                      const char* description = "convolution_fuzz") {
  if (model_inputs.size() != input_shapes.size() ||
      (invoke && input_shapes.size() != input_types.size())) {
    return RunResult::kHarnessFailure;
  }

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = description;
  model_spec.builtin_operator = builtin_operator;
  model_spec.version = max_version;
  model_spec.builtin_options_type = builtin_options_type;
  model_spec.builtin_options = builtin_options;
  model_spec.tensors = tensors;
  model_spec.buffers = buffers;
  model_spec.model_inputs = model_inputs;
  model_spec.model_outputs = model_outputs;
  model_spec.op_inputs = op_inputs;
  model_spec.op_outputs = op_outputs;

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = registration;
  run_spec.min_version = min_version;
  run_spec.max_version = max_version;
  run_spec.max_live_allocation_bytes = kMaxFuzzerLiveAllocationBytes;
  run_spec.invoke = invoke;
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    if (invoke && fuzzing::TypeSize(input_types[i]) == 0) {
      return RunResult::kRejected;
    }
    fuzzing::RuntimeTensor runtime_tensor;
    runtime_tensor.tensor_index = model_inputs[i];
    runtime_tensor.shape = input_shapes[i];
    if (invoke) {
      runtime_tensor.data = MakeTensorBytes(
          input_types[i], input_shapes[i], static_cast<int64_t>(i));
    }
    run_spec.runtime_tensors.push_back(std::move(runtime_tensor));
  }
  if (force_persistent_filter && persistent_filter_tensor_index >= 0) {
    run_spec.persistent_ro_tensors.push_back(persistent_filter_tensor_index);
  }
  return fuzzing::BuildAndRunOneOpModel(builder, model_spec, run_spec);
}

RunResult RunConv2D(const Conv2DCase& test_case) {
  flatbuffers::FlatBufferBuilder builder;
  const TensorType bias_type = test_case.input_type == TensorType_FLOAT32
                                   ? TensorType_FLOAT32
                                   : TensorType_INT32;
  const int32_t output_channels = BiasChannelsOrOne(test_case.filter_shape, 0);
  const int32_t input_channels = PositiveDimOr(test_case.filter_shape, 3, 1);
  const auto input_quantization =
      MakeQuantization(&builder, test_case.input_type, 0, 1);
  const auto filter_quantization = MakeQuantization(
      &builder, test_case.filter_type, 0, output_channels);
  const auto output_quantization =
      MakeQuantization(&builder, test_case.output_type, 0, 1);
  const auto bias_quantization = MakeQuantization(&builder, bias_type, 0, 1);

  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(MakeTensor(&builder, test_case.input_shape,
                               test_case.input_type, 0, input_quantization));
  tensors.push_back(MakeTensor(&builder, test_case.filter_shape,
                               test_case.filter_type, 0, filter_quantization));
  tensors.push_back(MakeTensor(&builder, {output_channels}, bias_type, 0,
                               bias_quantization));
  tensors.push_back(MakeTensor(&builder, {}, test_case.output_type, 0,
                               output_quantization));

  const auto options =
      CreateConv2DOptions(builder, test_case.padding, test_case.stride_width,
                          test_case.stride_height, ActivationFunctionType_NONE,
                          test_case.dilation_width, test_case.dilation_height,
                          TensorType_FLOAT32)
          .Union();
  TfLiteRegistration* registration =
      test_case.kernel == Conv2DKernel::kMultithreadedOptimized
          ? ops::builtin::Register_CONVOLUTION_MULTITHREADED_OPT()
          : ops::builtin::Register_CONVOLUTION_GENERIC_OPT();
  return BuildAndRun(
      &builder, BuiltinOperator_CONV_2D, BuiltinOptions_Conv2DOptions, options,
      registration, /*min_version=*/1, /*max_version=*/8, tensors,
      /*buffers=*/{}, /*model_inputs=*/{0, 1, 2}, /*model_outputs=*/{3},
      /*op_inputs=*/{0, 1, 2}, /*op_outputs=*/{3},
      /*input_shapes=*/{test_case.input_shape, test_case.filter_shape,
                        {output_channels}},
      /*input_types=*/{test_case.input_type, test_case.filter_type, bias_type},
      test_case.force_persistent_filter, /*persistent_filter_tensor_index=*/1,
      test_case.invoke && input_channels > 0);
}

RunResult RunDepthwiseConv(const DepthwiseConvCase& test_case) {
  flatbuffers::FlatBufferBuilder builder;
  const TensorType bias_type = test_case.input_type == TensorType_FLOAT32
                                   ? TensorType_FLOAT32
                                   : TensorType_INT32;
  const int32_t input_channels = PositiveDimOr(test_case.input_shape, 3, 1);
  const int32_t output_channels = BiasChannelsOrOne(test_case.filter_shape, 3);
  const int32_t depth_multiplier =
      std::max<int32_t>(1, output_channels / input_channels);
  const auto input_quantization =
      MakeQuantization(&builder, test_case.input_type, 0, 1);
  const auto filter_quantization = MakeQuantization(
      &builder, test_case.filter_type, 3, output_channels);
  const auto output_quantization =
      MakeQuantization(&builder, test_case.output_type, 0, 1);
  const auto bias_quantization = MakeQuantization(&builder, bias_type, 0, 1);

  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(MakeTensor(&builder, test_case.input_shape,
                               test_case.input_type, 0, input_quantization));
  tensors.push_back(MakeTensor(&builder, test_case.filter_shape,
                               test_case.filter_type, 0, filter_quantization));
  tensors.push_back(MakeTensor(&builder, {output_channels}, bias_type, 0,
                               bias_quantization));
  tensors.push_back(MakeTensor(&builder, {}, test_case.output_type, 0,
                               output_quantization));

  const auto options = CreateDepthwiseConv2DOptions(
                           builder, test_case.padding, test_case.stride_width,
                           test_case.stride_height, depth_multiplier,
                           ActivationFunctionType_NONE,
                           test_case.dilation_width, test_case.dilation_height)
                           .Union();
  return BuildAndRun(
      &builder, BuiltinOperator_DEPTHWISE_CONV_2D,
      BuiltinOptions_DepthwiseConv2DOptions, options,
      ops::builtin::Register_DEPTHWISE_CONVOLUTION_GENERIC_OPT(),
      /*min_version=*/1, /*max_version=*/7, tensors,
      /*buffers=*/{}, /*model_inputs=*/{0, 1, 2}, /*model_outputs=*/{3},
      /*op_inputs=*/{0, 1, 2}, /*op_outputs=*/{3},
      /*input_shapes=*/{test_case.input_shape, test_case.filter_shape,
                        {output_channels}},
      /*input_types=*/{test_case.input_type, test_case.filter_type, bias_type},
      /*force_persistent_filter=*/false, /*persistent_filter_tensor_index=*/-1,
      test_case.invoke);
}

RunResult RunTransposeConv(const TransposeConvCase& test_case) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{}),
      fuzzing::CreateAlignedBuffer(&builder,
                                   MakeInt32Bytes(test_case.output_shape))};
  const auto input_quantization =
      MakeQuantization(&builder, test_case.input_type, 0, 1);
  const int32_t output_channels = BiasChannelsOrOne(test_case.filter_shape, 0);
  const auto filter_quantization = MakeQuantization(
      &builder, test_case.filter_type, 0, output_channels);
  const auto output_quantization =
      MakeQuantization(&builder, test_case.output_type, 0, 1);

  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(MakeTensor(&builder, {static_cast<int32_t>(
                                             test_case.output_shape.size())},
                               TensorType_INT32, 1));
  tensors.push_back(MakeTensor(&builder, test_case.filter_shape,
                               test_case.filter_type, 0, filter_quantization));
  tensors.push_back(MakeTensor(&builder, test_case.input_shape,
                               test_case.input_type, 0, input_quantization));
  tensors.push_back(MakeTensor(&builder, {}, test_case.output_type, 0,
                               output_quantization));

  const auto options =
      CreateTransposeConvOptions(builder, test_case.padding,
                                 test_case.stride_width,
                                 test_case.stride_height,
                                 ActivationFunctionType_NONE,
                                 TensorType_FLOAT32)
          .Union();
  return BuildAndRun(
      &builder, BuiltinOperator_TRANSPOSE_CONV,
      BuiltinOptions_TransposeConvOptions, options,
      ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT(), /*min_version=*/1,
      /*max_version=*/5, tensors, buffers, /*model_inputs=*/{1, 2},
      /*model_outputs=*/{3}, /*op_inputs=*/{0, 1, 2}, /*op_outputs=*/{3},
      /*input_shapes=*/{test_case.filter_shape, test_case.input_shape},
      /*input_types=*/{test_case.filter_type, test_case.input_type},
      /*force_persistent_filter=*/false, /*persistent_filter_tensor_index=*/-1,
      test_case.invoke, "transpose_conv_fuzz");
}

RunResult RunConv3D(const Conv3DCase& test_case) {
  flatbuffers::FlatBufferBuilder builder;
  const int32_t output_channels = BiasChannelsOrOne(test_case.filter_shape, 4);
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(
      MakeTensor(&builder, test_case.input_shape, TensorType_FLOAT32, 0));
  tensors.push_back(
      MakeTensor(&builder, test_case.filter_shape, TensorType_FLOAT32, 0));
  tensors.push_back(MakeTensor(&builder, {output_channels}, TensorType_FLOAT32,
                               0));
  tensors.push_back(MakeTensor(&builder, {}, TensorType_FLOAT32, 0));

  const auto options =
      CreateConv3DOptions(builder, test_case.padding, test_case.stride_depth,
                          test_case.stride_width, test_case.stride_height,
                          ActivationFunctionType_NONE,
                          test_case.dilation_depth, test_case.dilation_width,
                          test_case.dilation_height)
          .Union();
  return BuildAndRun(
      &builder, BuiltinOperator_CONV_3D, BuiltinOptions_Conv3DOptions, options,
      ops::builtin::Register_CONV_3D_GENERIC_OPT(), /*min_version=*/1,
      /*max_version=*/1, tensors, /*buffers=*/{}, /*model_inputs=*/{0, 1, 2},
      /*model_outputs=*/{3}, /*op_inputs=*/{0, 1, 2}, /*op_outputs=*/{3},
      /*input_shapes=*/{test_case.input_shape, test_case.filter_shape,
                        {output_channels}},
      /*input_types=*/{TensorType_FLOAT32, TensorType_FLOAT32,
                       TensorType_FLOAT32},
      /*force_persistent_filter=*/false, /*persistent_filter_tensor_index=*/-1,
      test_case.invoke);
}

RunResult RunConv3DTranspose(const Conv3DTransposeCase& test_case) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{}),
      fuzzing::CreateAlignedBuffer(&builder,
                                   MakeInt32Bytes(test_case.output_shape))};
  const int32_t bias_channels = BiasChannelsOrOne(test_case.filter_shape, 3);
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(MakeTensor(
      &builder, {static_cast<int32_t>(test_case.output_shape.size())},
      TensorType_INT32, 1));
  tensors.push_back(
      MakeTensor(&builder, test_case.filter_shape, TensorType_FLOAT32, 0));
  tensors.push_back(
      MakeTensor(&builder, test_case.input_shape, TensorType_FLOAT32, 0));
  tensors.push_back(MakeTensor(&builder, {bias_channels}, TensorType_FLOAT32,
                               0));
  tensors.push_back(MakeTensor(&builder, {}, TensorType_FLOAT32, 0));

  const auto options =
      CreateConv3DOptions(builder, test_case.padding, test_case.stride_depth,
                          test_case.stride_width, test_case.stride_height,
                          ActivationFunctionType_NONE,
                          test_case.dilation_depth, test_case.dilation_width,
                          test_case.dilation_height)
          .Union();
  return BuildAndRun(
      &builder, BuiltinOperator_CONV_3D_TRANSPOSE,
      BuiltinOptions_Conv3DOptions, options,
      ops::builtin::Register_CONV_3D_TRANSPOSE(), /*min_version=*/1,
      /*max_version=*/1, tensors, buffers, /*model_inputs=*/{1, 2, 3},
      /*model_outputs=*/{4}, /*op_inputs=*/{0, 1, 2, 3}, /*op_outputs=*/{4},
      /*input_shapes=*/{test_case.filter_shape, test_case.input_shape,
                        {bias_channels}},
      /*input_types=*/{TensorType_FLOAT32, TensorType_FLOAT32,
                       TensorType_FLOAT32},
      /*force_persistent_filter=*/false, /*persistent_filter_tensor_index=*/-1,
      test_case.invoke, "conv3d_transpose_fuzz");
}

bool HasRank(const std::vector<int32_t>& shape, size_t rank) {
  return shape.size() == rank;
}

bool HasNegativeDim(const std::vector<int32_t>& shape) {
  return std::any_of(shape.begin(), shape.end(),
                     [](int32_t dim) { return dim < 0; });
}

bool HasNonPositiveDim(const std::vector<int32_t>& shape,
                       std::initializer_list<size_t> indexes) {
  for (const size_t index : indexes) {
    if (index >= shape.size() || shape[index] <= 0) {
      return true;
    }
  }
  return false;
}

bool ProductOverflowsInt32(std::initializer_list<int64_t> dims) {
  int64_t product = 1;
  for (const int64_t dim : dims) {
    if (dim < 0) {
      return true;
    }
    if (dim == 0 || product == 0) {
      product = 0;
      continue;
    }
    if (product > kMaxInt32Value / dim) {
      return true;
    }
    product *= dim;
  }
  return false;
}

bool GetConvolutionOutputDim(Padding padding, int32_t input_size,
                             int32_t filter_size, int32_t stride,
                             int32_t dilation, int32_t* output_size) {
  if (output_size == nullptr || input_size < 0 || filter_size <= 0 ||
      stride <= 0 || dilation <= 0) {
    return false;
  }
  const int64_t effective_filter_size =
      (static_cast<int64_t>(filter_size) - 1) * dilation + 1;
  int64_t value = 0;
  switch (padding) {
    case Padding_SAME:
      value = (static_cast<int64_t>(input_size) + stride - 1) / stride;
      break;
    case Padding_VALID:
      value = (static_cast<int64_t>(input_size) + stride -
               effective_filter_size) /
              stride;
      break;
    default:
      return false;
  }
  if (value < 0 || value > kMaxInt32Value) {
    return false;
  }
  *output_size = static_cast<int32_t>(value);
  return true;
}

bool TransposeConvTypesUnsupported(const TransposeConvCase& test_case) {
  if (test_case.output_type != test_case.input_type) {
    return true;
  }
  switch (test_case.input_type) {
    case TensorType_FLOAT32:
      return test_case.filter_type != TensorType_FLOAT32 &&
             test_case.filter_type != TensorType_INT8;
    case TensorType_UINT8:
      return test_case.filter_type != TensorType_UINT8;
    case TensorType_INT8:
    case TensorType_INT16:
      return test_case.filter_type != TensorType_INT8;
    default:
      return true;
  }
}

bool Conv2DMustRejectIfInvoked(const Conv2DCase& test_case) {
  if (!HasRank(test_case.input_shape, 4) ||
      !HasRank(test_case.filter_shape, 4) ||
      HasNegativeDim(test_case.input_shape) ||
      HasNegativeDim(test_case.filter_shape) ||
      HasNonPositiveDim(test_case.input_shape, {1, 2, 3}) ||
      HasNonPositiveDim(test_case.filter_shape, {0, 1, 2, 3}) ||
      test_case.stride_width <= 0 || test_case.stride_height <= 0 ||
      test_case.dilation_width <= 0 || test_case.dilation_height <= 0) {
    return true;
  }

  const int32_t input_channels = test_case.input_shape[3];
  const int32_t filter_input_channels = test_case.filter_shape[3];
  if (input_channels % filter_input_channels != 0) {
    return true;
  }
  const int32_t groups = input_channels / filter_input_channels;
  if (test_case.filter_shape[0] % groups != 0) {
    return true;
  }

  int32_t output_height = 0;
  int32_t output_width = 0;
  if (!GetConvolutionOutputDim(test_case.padding, test_case.input_shape[1],
                               test_case.filter_shape[1],
                               test_case.stride_height,
                               test_case.dilation_height, &output_height) ||
      !GetConvolutionOutputDim(test_case.padding, test_case.input_shape[2],
                               test_case.filter_shape[2],
                               test_case.stride_width,
                               test_case.dilation_width, &output_width)) {
    return true;
  }
  return ProductOverflowsInt32(
      {test_case.input_shape[0], output_height, output_width,
       test_case.filter_shape[0]});
}

bool DepthwiseConvMustRejectIfInvoked(const DepthwiseConvCase& test_case) {
  if (!HasRank(test_case.input_shape, 4) ||
      !HasRank(test_case.filter_shape, 4) ||
      HasNegativeDim(test_case.input_shape) ||
      HasNegativeDim(test_case.filter_shape) ||
      HasNonPositiveDim(test_case.input_shape, {1, 2, 3}) ||
      HasNonPositiveDim(test_case.filter_shape, {1, 2, 3}) ||
      test_case.filter_shape[0] != 1 || test_case.stride_width <= 0 ||
      test_case.stride_height <= 0 ||
      test_case.stride_width > std::numeric_limits<int16_t>::max() ||
      test_case.stride_height > std::numeric_limits<int16_t>::max() ||
      test_case.dilation_width <= 0 || test_case.dilation_height <= 0 ||
      test_case.dilation_width > std::numeric_limits<int16_t>::max() ||
      test_case.dilation_height > std::numeric_limits<int16_t>::max()) {
    return true;
  }

  const int32_t input_channels = test_case.input_shape[3];
  const int32_t output_channels = test_case.filter_shape[3];
  if (output_channels % input_channels != 0) {
    return true;
  }

  int32_t output_height = 0;
  int32_t output_width = 0;
  if (!GetConvolutionOutputDim(test_case.padding, test_case.input_shape[1],
                               test_case.filter_shape[1],
                               test_case.stride_height,
                               test_case.dilation_height, &output_height) ||
      !GetConvolutionOutputDim(test_case.padding, test_case.input_shape[2],
                               test_case.filter_shape[2],
                               test_case.stride_width,
                               test_case.dilation_width, &output_width)) {
    return true;
  }
  return ProductOverflowsInt32(
      {test_case.input_shape[0], output_height, output_width,
       output_channels});
}

bool TransposeConvMustRejectIfInvoked(const TransposeConvCase& test_case) {
  if (!HasRank(test_case.output_shape, 4) ||
      !HasRank(test_case.input_shape, 4) ||
      HasNegativeDim(test_case.output_shape) ||
      HasNegativeDim(test_case.input_shape) ||
      HasNonPositiveDim(test_case.input_shape, {1, 2, 3}) ||
      TransposeConvTypesUnsupported(test_case) ||
      test_case.stride_width <= 0 || test_case.stride_height <= 0 ||
      test_case.stride_width > std::numeric_limits<int16_t>::max() ||
      test_case.stride_height > std::numeric_limits<int16_t>::max()) {
    return true;
  }
  return ProductOverflowsInt32({test_case.output_shape[0],
                                test_case.output_shape[1],
                                test_case.output_shape[2],
                                test_case.output_shape[3]});
}

bool Conv3DMustRejectIfInvoked(const Conv3DCase& test_case) {
  if (!HasRank(test_case.input_shape, 5) ||
      !HasRank(test_case.filter_shape, 5) ||
      HasNegativeDim(test_case.input_shape) ||
      HasNegativeDim(test_case.filter_shape) ||
      HasNonPositiveDim(test_case.input_shape, {1, 2, 3, 4}) ||
      HasNonPositiveDim(test_case.filter_shape, {0, 1, 2, 3, 4}) ||
      test_case.input_shape[4] != test_case.filter_shape[3] ||
      test_case.stride_depth <= 0 || test_case.stride_width <= 0 ||
      test_case.stride_height <= 0 || test_case.dilation_depth <= 0 ||
      test_case.dilation_width <= 0 || test_case.dilation_height <= 0) {
    return true;
  }

  int32_t output_depth = 0;
  int32_t output_height = 0;
  int32_t output_width = 0;
  if (!GetConvolutionOutputDim(test_case.padding, test_case.input_shape[1],
                               test_case.filter_shape[0],
                               test_case.stride_depth,
                               test_case.dilation_depth, &output_depth) ||
      !GetConvolutionOutputDim(test_case.padding, test_case.input_shape[2],
                               test_case.filter_shape[1],
                               test_case.stride_height,
                               test_case.dilation_height, &output_height) ||
      !GetConvolutionOutputDim(test_case.padding, test_case.input_shape[3],
                               test_case.filter_shape[2],
                               test_case.stride_width,
                               test_case.dilation_width, &output_width)) {
    return true;
  }
  return ProductOverflowsInt32({test_case.input_shape[0], output_depth,
                                output_height, output_width,
                                test_case.filter_shape[4]});
}

bool Conv3DTransposeMustRejectIfInvoked(
    const Conv3DTransposeCase& test_case) {
  if (!HasRank(test_case.output_shape, 5) ||
      !HasRank(test_case.input_shape, 5) ||
      !HasRank(test_case.filter_shape, 5) ||
      HasNegativeDim(test_case.output_shape) ||
      HasNegativeDim(test_case.input_shape) ||
      HasNegativeDim(test_case.filter_shape) ||
      HasNonPositiveDim(test_case.input_shape, {1, 2, 3, 4}) ||
      HasNonPositiveDim(test_case.filter_shape, {0, 1, 2, 3, 4}) ||
      test_case.input_shape[4] != test_case.filter_shape[4] ||
      test_case.output_shape[0] != test_case.input_shape[0] ||
      test_case.output_shape[4] != test_case.filter_shape[3]) {
    return true;
  }
  return ProductOverflowsInt32({test_case.output_shape[0],
                                test_case.output_shape[1],
                                test_case.output_shape[2],
                                test_case.output_shape[3],
                                test_case.output_shape[4]});
}

void ExpectSafeAndContractCompliant(RunResult result, bool must_reject) {
  EXPECT_NE(result, RunResult::kHarnessFailure);
  // Some kernels historically accept empty tensors or defer structural checks
  // until Eval. Treat those as part of the safety fuzzing space unless this
  // specific case was actually invoked and should fail by the op contract.
  if (must_reject) {
    EXPECT_EQ(result, RunResult::kRejected);
  }
}

std::vector<std::tuple<Conv2DCase>> Conv2DSeeds() {
  return {
      std::tuple<Conv2DCase>{Conv2DCase{{1, 32768, 32768, 1},
                                        {1, 1, 1, 1},
                                        TensorType_FLOAT32,
                                        TensorType_FLOAT32,
                                        TensorType_FLOAT32,
                                        Conv2DKernel::kGenericOptimized,
                                        Padding_VALID,
                                        1,
                                        1,
                                        1,
                                        1,
                                        false,
                                        false}},
      std::tuple<Conv2DCase>{Conv2DCase{{1, 1, 1, 1},
                                        {1, 46341, 46341, 1},
                                        TensorType_FLOAT32,
                                        TensorType_FLOAT32,
                                        TensorType_FLOAT32,
                                        Conv2DKernel::kMultithreadedOptimized,
                                        Padding_VALID,
                                        1,
                                        1,
                                        1,
                                        1,
                                        true,
                                        false}},
      std::tuple<Conv2DCase>{Conv2DCase{{46341, 2, 46341, 1},
                                        {1, 1, 1, 1},
                                        TensorType_FLOAT32,
                                        TensorType_INT8,
                                        TensorType_FLOAT32,
                                        Conv2DKernel::kGenericOptimized,
                                        Padding_VALID,
                                        1,
                                        1,
                                        1,
                                        1,
                                        false,
                                        false}},
      std::tuple<Conv2DCase>{Conv2DCase{{1, 46341, 1, 46341},
                                        {1, 1, 1, 46341},
                                        TensorType_FLOAT32,
                                        TensorType_INT8,
                                        TensorType_FLOAT32,
                                        Conv2DKernel::kGenericOptimized,
                                        Padding_VALID,
                                        1,
                                        1,
                                        1,
                                        1,
                                        false,
                                        false}},
      std::tuple<Conv2DCase>{Conv2DCase{{1, 1, 1, 46341},
                                        {46341, 1, 1, 46341},
                                        TensorType_FLOAT32,
                                        TensorType_INT4,
                                        TensorType_FLOAT32,
                                        Conv2DKernel::kGenericOptimized,
                                        Padding_VALID,
                                        1,
                                        1,
                                        1,
                                        1,
                                        false,
                                        false}},
  };
}

std::vector<std::tuple<DepthwiseConvCase>> DepthwiseConvSeeds() {
  return {
      std::tuple<DepthwiseConvCase>{DepthwiseConvCase{{46341, 46341, 1, 1},
                                                      {1, 1, 1, 1},
                                                      TensorType_FLOAT32,
                                                      TensorType_INT8,
                                                      TensorType_FLOAT32,
                                                      Padding_VALID,
                                                      1,
                                                      1,
                                                      1,
                                                      1,
                                                      false}},
      std::tuple<DepthwiseConvCase>{DepthwiseConvCase{{1, 46341, 1, 1},
                                                      {1, 46341, 1, 46341},
                                                      TensorType_FLOAT32,
                                                      TensorType_INT4,
                                                      TensorType_FLOAT32,
                                                      Padding_VALID,
                                                      1,
                                                      1,
                                                      1,
                                                      1,
                                                      false}},
      std::tuple<DepthwiseConvCase>{DepthwiseConvCase{{1, 1, 1, 0},
                                                      {1, 1, 1, 1},
                                                      TensorType_FLOAT32,
                                                      TensorType_FLOAT32,
                                                      TensorType_FLOAT32,
                                                      Padding_VALID,
                                                      1,
                                                      1,
                                                      1,
                                                      1,
                                                      true}},
  };
}

std::vector<std::tuple<TransposeConvCase>> TransposeConvSeeds() {
  return {
      std::tuple<TransposeConvCase>{TransposeConvCase{{1, 1, 1, 0},
                                                      {1, 1, 1, 1},
                                                      {1, 46341, 46341, 1},
                                                      TensorType_FLOAT32,
                                                      TensorType_FLOAT32,
                                                      TensorType_FLOAT32,
                                                      Padding_SAME,
                                                      1,
                                                      1,
                                                      false}},
      std::tuple<TransposeConvCase>{TransposeConvCase{{1, 1, 1, 1},
                                                      {1, 1, 1, 1},
                                                      {46341, 46341, 1, 1},
                                                      TensorType_FLOAT32,
                                                      TensorType_INT8,
                                                      TensorType_FLOAT32,
                                                      Padding_SAME,
                                                      1,
                                                      1,
                                                      false}},
      std::tuple<TransposeConvCase>{TransposeConvCase{{1, -1, 1, 1},
                                                      {1, 1, 1, 1},
                                                      {1, 1, 1, 1},
                                                      TensorType_FLOAT32,
                                                      TensorType_FLOAT32,
                                                      TensorType_FLOAT32,
                                                      Padding_SAME,
                                                      1,
                                                      1,
                                                      false}},
  };
}

std::vector<std::tuple<Conv3DCase>> Conv3DSeeds() {
  return {
      std::tuple<Conv3DCase>{Conv3DCase{{1, 46341, 46341, 1, 1},
                                        {1, 1, 1, 1, 1},
                                        Padding_SAME,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        false}},
      std::tuple<Conv3DCase>{Conv3DCase{{1, 1, 46341, 46341, 1},
                                        {1, 1, 1, 1, 1},
                                        Padding_SAME,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        false}},
      std::tuple<Conv3DCase>{Conv3DCase{{1, 1, 1, 1, 46341},
                                        {46341, 1, 1, 46341, 1},
                                        Padding_VALID,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        false}},
  };
}

std::vector<std::tuple<Conv3DTransposeCase>> Conv3DTransposeSeeds() {
  return {
      std::tuple<Conv3DTransposeCase>{Conv3DTransposeCase{{1, 1, 1, 1, 0},
                                                          {1, 46341, 46341, 1,
                                                           0},
                                                          {1, 1, 1, 1, 0},
                                                          Padding_SAME,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          false}},
      std::tuple<Conv3DTransposeCase>{Conv3DTransposeCase{{1, 1, 1, 1, 1},
                                                          {1, 1, 1, 0, 1},
                                                          {1, 1, 1, 1, 1},
                                                          Padding_SAME,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          false}},
      std::tuple<Conv3DTransposeCase>{Conv3DTransposeCase{{1, -1, 1, 1, 1},
                                                          {1, 1, 1, 1, 1},
                                                          {1, 1, 1, 1, 1},
                                                          Padding_SAME,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          false}},
      std::tuple<Conv3DTransposeCase>{Conv3DTransposeCase{{1, 46341, 46341, 1,
                                                           1},
                                                          {1, 1, 1, 1, 1},
                                                          {1, 1, 1, 1, 1},
                                                          Padding_SAME,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          1,
                                                          false}},
  };
}

auto SmallDimDomain() {
  return fuzztest::OneOf(fuzztest::InRange<int32_t>(0, 4),
                         fuzztest::Just<int32_t>(-1),
                         fuzztest::Just<int32_t>(8),
                         fuzztest::Just<int32_t>(32768),
                         fuzztest::Just<int32_t>(46340),
                         fuzztest::Just<int32_t>(46341));
}

auto PositiveOptionDomain() {
  return fuzztest::OneOf(fuzztest::InRange<int32_t>(1, 4),
                         fuzztest::Just<int32_t>(0),
                         fuzztest::Just<int32_t>(-1),
                         fuzztest::Just<int32_t>(65536));
}

auto ShapeDomain(int min_rank, int max_rank) {
  return fuzztest::VectorOf(SmallDimDomain())
      .WithMinSize(min_rank)
      .WithMaxSize(max_rank);
}

auto Conv2DCaseDomain() {
  return fuzztest::StructOf<Conv2DCase>(
      ShapeDomain(0, 5), ShapeDomain(0, 5),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_UINT8}),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_UINT8, TensorType_INT4}),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_UINT8}),
      fuzztest::ElementOf<Conv2DKernel>(
          {Conv2DKernel::kGenericOptimized,
           Conv2DKernel::kMultithreadedOptimized}),
      fuzztest::ElementOf<Padding>({Padding_VALID, Padding_SAME}),
      PositiveOptionDomain(), PositiveOptionDomain(), PositiveOptionDomain(),
      PositiveOptionDomain(), fuzztest::Arbitrary<bool>(),
      fuzztest::Arbitrary<bool>());
}

auto DepthwiseConvCaseDomain() {
  return fuzztest::StructOf<DepthwiseConvCase>(
      ShapeDomain(0, 5), ShapeDomain(0, 5),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_UINT8}),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_UINT8, TensorType_INT4}),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_UINT8}),
      fuzztest::ElementOf<Padding>({Padding_VALID, Padding_SAME}),
      PositiveOptionDomain(), PositiveOptionDomain(), PositiveOptionDomain(),
      PositiveOptionDomain(), fuzztest::Arbitrary<bool>());
}

auto TransposeConvCaseDomain() {
  return fuzztest::StructOf<TransposeConvCase>(
      ShapeDomain(0, 5), ShapeDomain(0, 5), ShapeDomain(0, 5),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_UINT8}),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_UINT8}),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_UINT8}),
      fuzztest::ElementOf<Padding>({Padding_VALID, Padding_SAME}),
      PositiveOptionDomain(), PositiveOptionDomain(),
      fuzztest::Arbitrary<bool>());
}

auto Conv3DCaseDomain() {
  return fuzztest::StructOf<Conv3DCase>(
      ShapeDomain(0, 6), ShapeDomain(0, 6),
      fuzztest::ElementOf<Padding>({Padding_VALID, Padding_SAME}),
      PositiveOptionDomain(), PositiveOptionDomain(), PositiveOptionDomain(),
      PositiveOptionDomain(), PositiveOptionDomain(), PositiveOptionDomain(),
      fuzztest::Arbitrary<bool>());
}

auto Conv3DTransposeCaseDomain() {
  return fuzztest::StructOf<Conv3DTransposeCase>(
      ShapeDomain(0, 6), ShapeDomain(0, 6), ShapeDomain(0, 6),
      fuzztest::ElementOf<Padding>({Padding_VALID, Padding_SAME}),
      PositiveOptionDomain(), PositiveOptionDomain(), PositiveOptionDomain(),
      PositiveOptionDomain(), PositiveOptionDomain(), PositiveOptionDomain(),
      fuzztest::Arbitrary<bool>());
}

void Conv2DIsSafeAndContractCompliant(const Conv2DCase& test_case) {
  ExpectSafeAndContractCompliant(RunConv2D(test_case),
                                 test_case.invoke &&
                                     Conv2DMustRejectIfInvoked(test_case));
}

void DepthwiseConvIsSafeAndContractCompliant(
    const DepthwiseConvCase& test_case) {
  ExpectSafeAndContractCompliant(RunDepthwiseConv(test_case),
                                 test_case.invoke &&
                                     DepthwiseConvMustRejectIfInvoked(
                                         test_case));
}

void TransposeConvIsSafeAndContractCompliant(
    const TransposeConvCase& test_case) {
  ExpectSafeAndContractCompliant(RunTransposeConv(test_case),
                                 test_case.invoke &&
                                     TransposeConvMustRejectIfInvoked(
                                         test_case));
}

void Conv3DIsSafeAndContractCompliant(const Conv3DCase& test_case) {
  ExpectSafeAndContractCompliant(RunConv3D(test_case),
                                 test_case.invoke &&
                                     Conv3DMustRejectIfInvoked(test_case));
}

void Conv3DTransposeIsSafeAndContractCompliant(
    const Conv3DTransposeCase& test_case) {
  ExpectSafeAndContractCompliant(
      RunConv3DTranspose(test_case),
      test_case.invoke && Conv3DTransposeMustRejectIfInvoked(test_case));
}

TEST(ConvolutionFuzzTest, SmokeInvokes) {
  EXPECT_NE(RunConv2D({{1, 3, 3, 1},
                       {1, 1, 1, 1},
                       TensorType_FLOAT32,
                       TensorType_FLOAT32,
                       TensorType_FLOAT32,
                       Conv2DKernel::kGenericOptimized,
                       Padding_VALID,
                       1,
                       1,
                       1,
                       1,
                       false,
                       true}),
            RunResult::kHarnessFailure);
  EXPECT_NE(RunDepthwiseConv({{1, 3, 3, 1},
                              {1, 1, 1, 1},
                              TensorType_FLOAT32,
                              TensorType_FLOAT32,
                              TensorType_FLOAT32,
                              Padding_VALID,
                              1,
                              1,
                              1,
                              1,
                              true}),
            RunResult::kHarnessFailure);
  EXPECT_EQ(RunTransposeConv({{1, 3, 3, 1},
                              {1, 1, 1, 1},
                              {1, 3, 3, 1},
                              TensorType_FLOAT32,
                              TensorType_FLOAT32,
                              TensorType_FLOAT32,
                              Padding_SAME,
                              1,
                              1,
                              true}),
            RunResult::kSuccess);
  EXPECT_NE(RunConv3D({{1, 2, 2, 2, 1},
                       {1, 1, 1, 1, 1},
                       Padding_VALID,
                       1,
                       1,
                       1,
                       1,
                       1,
                       1,
                       true}),
            RunResult::kHarnessFailure);
  EXPECT_EQ(RunConv3DTranspose({{1, 2, 2, 2, 1},
                                {1, 1, 1, 1, 1},
                                {1, 2, 2, 2, 1},
                                Padding_SAME,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                true}),
            RunResult::kSuccess);
}

TEST(ConvolutionFuzzTest, PersistentFilterInvokesMultithreadedKernel) {
  EXPECT_EQ(RunConv2D({{1, 3, 3, 1},
                       {1, 1, 1, 1},
                       TensorType_FLOAT32,
                       TensorType_FLOAT32,
                       TensorType_FLOAT32,
                       Conv2DKernel::kMultithreadedOptimized,
                       Padding_VALID,
                       1,
                       1,
                       1,
                       1,
                       /*force_persistent_filter=*/true,
                       /*invoke=*/true}),
            RunResult::kSuccess);
}

FUZZ_TEST(ConvolutionFuzzTest, Conv2DIsSafeAndContractCompliant)
    .WithDomains(Conv2DCaseDomain())
    .WithSeeds(Conv2DSeeds());
FUZZ_TEST(ConvolutionFuzzTest, DepthwiseConvIsSafeAndContractCompliant)
    .WithDomains(DepthwiseConvCaseDomain())
    .WithSeeds(DepthwiseConvSeeds());
FUZZ_TEST(ConvolutionFuzzTest, TransposeConvIsSafeAndContractCompliant)
    .WithDomains(TransposeConvCaseDomain())
    .WithSeeds(TransposeConvSeeds());
FUZZ_TEST(ConvolutionFuzzTest, Conv3DIsSafeAndContractCompliant)
    .WithDomains(Conv3DCaseDomain())
    .WithSeeds(Conv3DSeeds());
FUZZ_TEST(ConvolutionFuzzTest, Conv3DTransposeIsSafeAndContractCompliant)
    .WithDomains(Conv3DTransposeCaseDomain())
    .WithSeeds(Conv3DTransposeSeeds());

}  // namespace
}  // namespace tflite
