/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"
#include "litert/core/model/verifier.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "tflite/core/api/error_reporter.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/version.h"

namespace tflite {
namespace testing {
namespace {

// Avoid logging overhead with a no-op error reporter.
class NullErrorReporter : public ErrorReporter {
  int Report(const char* format, va_list args) override { return 0; }
};

std::vector<std::tuple<std::string>> GetSeeds() {
  std::string path = litert::testing::GetTfliteFilePath("testdata/add.bin");
  std::ifstream file(path, std::ios::binary);
  if (!file) return {};
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  return {std::make_tuple(content)};
}

LiteRtElementType MapElementType(int32_t raw_type) {
  static constexpr LiteRtElementType kTypes[] = {
      kLiteRtElementTypeNone,
      kLiteRtElementTypeFloat32,       // FLOAT32 = 0
      kLiteRtElementTypeFloat16,       // FLOAT16 = 1
      kLiteRtElementTypeInt32,         // INT32 = 2
      kLiteRtElementTypeUInt8,         // UINT8 = 3
      kLiteRtElementTypeInt64,         // INT64 = 4
      kLiteRtElementTypeTfString,      // STRING = 5
      kLiteRtElementTypeBool,          // BOOL = 6
      kLiteRtElementTypeInt16,         // INT16 = 7
      kLiteRtElementTypeComplex64,     // COMPLEX64 = 8
      kLiteRtElementTypeInt8,          // INT8 = 9
      kLiteRtElementTypeFloat64,       // FLOAT64 = 10
      kLiteRtElementTypeComplex128,    // COMPLEX128 = 11
      kLiteRtElementTypeUInt64,        // UINT64 = 12
      kLiteRtElementTypeTfResource,    // RESOURCE = 13
      kLiteRtElementTypeTfVariant,     // VARIANT = 14
      kLiteRtElementTypeUInt32,        // UINT32 = 15
      kLiteRtElementTypeUInt16,        // UINT16 = 16
      kLiteRtElementTypeInt4,          // INT4 = 17
      kLiteRtElementTypeBFloat16,      // BFLOAT16 = 18
      kLiteRtElementTypeInt2,          // INT2 = 19
      kLiteRtElementTypeUInt4,         // UINT4 = 20
      kLiteRtElementTypeFloat8E4M3FN,  // FLOAT8_E4M3FN = 21
      kLiteRtElementTypeFloat8E5M2,    // FLOAT8_E5M2 = 22
  };
  constexpr size_t kNumTypes = sizeof(kTypes) / sizeof(kTypes[0]);
  const size_t idx = static_cast<uint32_t>(raw_type) % kNumTypes;
  return kTypes[idx];
}

::TensorType MakeUnrankedTensor(LiteRtElementType element_type) {
  ::TensorType type;
  type.first = kLiteRtUnrankedTensorType;
  type.second.unranked_tensor_type.element_type = element_type;
  return type;
}

void PopulateOpOptions(int32_t option_selector, int32_t param_a,
                       int32_t param_b, int32_t param_c, int32_t param_d,
                       const std::vector<int32_t>& vec_param, LiteRtOpT& op) {
  static constexpr tflite::BuiltinOptions kFuzzedOptionTypes[] = {
      tflite::BuiltinOptions_NONE,
      tflite::BuiltinOptions_GatherOptions,
      tflite::BuiltinOptions_DepthToSpaceOptions,
      tflite::BuiltinOptions_SpaceToDepthOptions,
      tflite::BuiltinOptions_SplitOptions,
      tflite::BuiltinOptions_UnpackOptions,
      tflite::BuiltinOptions_ConcatenationOptions,
      tflite::BuiltinOptions_OneHotOptions,
      tflite::BuiltinOptions_SqueezeOptions,
      tflite::BuiltinOptions_ReshapeOptions,
      tflite::BuiltinOptions_StridedSliceOptions,
      tflite::BuiltinOptions_Conv2DOptions,
      tflite::BuiltinOptions_DepthwiseConv2DOptions,
      tflite::BuiltinOptions_TransposeConvOptions,
      tflite::BuiltinOptions_Pool2DOptions,
      tflite::BuiltinOptions_BatchMatMulOptions,
      tflite::BuiltinOptions_FullyConnectedOptions,
      tflite::BuiltinOptions_ReducerOptions,
      tflite::BuiltinOptions_PackOptions,
  };
  constexpr size_t kNumOptionTypes =
      sizeof(kFuzzedOptionTypes) / sizeof(kFuzzedOptionTypes[0]);
  const tflite::BuiltinOptions option_type =
      kFuzzedOptionTypes[static_cast<uint32_t>(option_selector) %
                         kNumOptionTypes];

  litert::internal::TflOptions tfl_options;
  tfl_options.type = option_type;
  switch (option_type) {
    case tflite::BuiltinOptions_GatherOptions: {
      auto opts = std::make_unique<tflite::GatherOptionsT>();
      opts->axis = param_a;
      opts->batch_dims = param_b;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_DepthToSpaceOptions: {
      auto opts = std::make_unique<tflite::DepthToSpaceOptionsT>();
      opts->block_size = param_a;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_SpaceToDepthOptions: {
      auto opts = std::make_unique<tflite::SpaceToDepthOptionsT>();
      opts->block_size = param_a;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_SplitOptions: {
      auto opts = std::make_unique<tflite::SplitOptionsT>();
      opts->num_splits = param_a;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_UnpackOptions: {
      auto opts = std::make_unique<tflite::UnpackOptionsT>();
      opts->num = param_a;
      opts->axis = param_b;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_ConcatenationOptions: {
      auto opts = std::make_unique<tflite::ConcatenationOptionsT>();
      opts->axis = param_a;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_OneHotOptions: {
      auto opts = std::make_unique<tflite::OneHotOptionsT>();
      opts->axis = param_a;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_SqueezeOptions: {
      auto opts = std::make_unique<tflite::SqueezeOptionsT>();
      opts->squeeze_dims = vec_param;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_ReshapeOptions: {
      auto opts = std::make_unique<tflite::ReshapeOptionsT>();
      opts->new_shape = vec_param;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_StridedSliceOptions: {
      auto opts = std::make_unique<tflite::StridedSliceOptionsT>();
      opts->begin_mask = param_a;
      opts->end_mask = param_b;
      opts->shrink_axis_mask = param_c;
      opts->new_axis_mask = param_d;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_Conv2DOptions: {
      auto opts = std::make_unique<tflite::Conv2DOptionsT>();
      opts->padding =
          (param_a & 1) ? tflite::Padding_SAME : tflite::Padding_VALID;
      opts->stride_w = param_b;
      opts->stride_h = param_c;
      opts->dilation_w_factor = param_d;
      opts->dilation_h_factor = param_a;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_DepthwiseConv2DOptions: {
      auto opts = std::make_unique<tflite::DepthwiseConv2DOptionsT>();
      opts->padding =
          (param_a & 1) ? tflite::Padding_SAME : tflite::Padding_VALID;
      opts->stride_w = param_b;
      opts->stride_h = param_c;
      opts->depth_multiplier = param_d;
      opts->dilation_w_factor = 1;
      opts->dilation_h_factor = 1;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_TransposeConvOptions: {
      auto opts = std::make_unique<tflite::TransposeConvOptionsT>();
      opts->padding =
          (param_a & 1) ? tflite::Padding_SAME : tflite::Padding_VALID;
      opts->stride_w = param_b;
      opts->stride_h = param_c;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_Pool2DOptions: {
      auto opts = std::make_unique<tflite::Pool2DOptionsT>();
      opts->padding =
          (param_a & 1) ? tflite::Padding_SAME : tflite::Padding_VALID;
      opts->stride_w = param_b;
      opts->stride_h = param_c;
      opts->filter_width = param_d;
      opts->filter_height = param_a;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_BatchMatMulOptions: {
      auto opts = std::make_unique<tflite::BatchMatMulOptionsT>();
      opts->adj_x = (param_a & 1) != 0;
      opts->adj_y = (param_b & 1) != 0;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_FullyConnectedOptions: {
      auto opts = std::make_unique<tflite::FullyConnectedOptionsT>();
      opts->keep_num_dims = (param_a & 1) != 0;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_ReducerOptions: {
      auto opts = std::make_unique<tflite::ReducerOptionsT>();
      opts->keep_dims = (param_a & 1) != 0;
      tfl_options.value = opts.release();
      break;
    }
    case tflite::BuiltinOptions_PackOptions: {
      auto opts = std::make_unique<tflite::PackOptionsT>();
      opts->values_count = param_a;
      opts->axis = param_b;
      tfl_options.value = opts.release();
      break;
    }
    default:
      break;
  }
  litert::internal::SetTflOptions(op, std::move(tfl_options));
}

}  // namespace

// Simple test which fuzzes the actual TFLite flatbuffer model used to create
// the interpreter.
void ModelFuzzerTest(const std::string& data) {
  // Loading while using the model verifier should never crash (though it may
  // return a null model).
  litert::LiteRtVerifier verifier;
  NullErrorReporter error_reporter;
  auto model = FlatBufferModel::VerifyAndBuildFromBuffer(
      data.data(), data.size(), &verifier, &error_reporter);

  if (model) {
    // If we get a valid model, interpreter creation should never crash (though
    // it may fail with an error code).
    std::unique_ptr<Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder(model->GetModel(), resolver,
                       &error_reporter)(&interpreter);
  }
}
FUZZ_TEST(ModelFuzzer, ModelFuzzerTest).WithSeeds(GetSeeds());

// Directly fuzzes ValidateFlatBufferOperatorCodes with arbitrary operator code
// combinations.
void ValidateOperatorCodesFuzzerTest(int8_t deprecated_builtin_code,
                                     int32_t builtin_code, int32_t version) {
  flatbuffers::FlatBufferBuilder builder;
  auto opcode = tflite::CreateOperatorCode(
      builder, deprecated_builtin_code, /*custom_code=*/0, version,
      static_cast<tflite::BuiltinOperator>(builtin_code));
  auto model = tflite::CreateModel(builder, TFLITE_SCHEMA_VERSION,
                                   builder.CreateVector(&opcode, 1));
  tflite::FinishModelBuffer(builder, model);

  const tflite::Model* tfl_model = tflite::GetModel(builder.GetBufferPointer());
  (void)litert::internal::ValidateFlatBufferOperatorCodes(tfl_model);
}
FUZZ_TEST(ModelFuzzer, ValidateOperatorCodesFuzzerTest);

// Directly fuzzes ValidateFlatBufferTensors and
// ValidateFlatBufferTensorQuantization on constructed FlatBuffer tensors.
void ValidateTensorFuzzerTest(const std::vector<int32_t>& shape,
                              const std::vector<int32_t>& shape_signature,
                              bool has_rank, int8_t raw_tensor_type,
                              const std::vector<float>& scales,
                              const std::vector<int64_t>& zero_points,
                              int32_t quantized_dimension,
                              uint8_t quant_details_mode, int32_t bw_scales,
                              int32_t bw_zero_points, int32_t bw_block_size,
                              const std::vector<int32_t>& bw_dims) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder),
      tflite::CreateBuffer(builder),
  };

  flatbuffers::Offset<void> details_offset = 0;
  const auto details_type =
      static_cast<tflite::QuantizationDetails>(quant_details_mode);
  switch (details_type) {
    case tflite::QuantizationDetails_BlockwiseQuantization:
      details_offset = tflite::CreateBlockwiseQuantization(
                           builder, bw_scales, bw_zero_points, bw_block_size,
                           bw_dims.empty() ? 0 : builder.CreateVector(bw_dims))
                           .Union();
      break;
    case tflite::QuantizationDetails_MultiAxisQuantization:
      details_offset = tflite::CreateMultiAxisQuantization(
                           builder, bw_scales, bw_zero_points, bw_block_size,
                           bw_dims.empty() ? 0 : builder.CreateVector(bw_dims))
                           .Union();
      break;
    default:
      break;
  }

  auto quant = tflite::CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0,
      scales.empty() ? 0 : builder.CreateVector(scales),
      zero_points.empty() ? 0 : builder.CreateVector(zero_points), details_type,
      details_offset, quantized_dimension);

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      tflite::CreateTensor(
          builder, shape.empty() && !has_rank ? 0 : builder.CreateVector(shape),
          static_cast<tflite::TensorType>(raw_tensor_type), /*buffer=*/0,
          builder.CreateString("t0"), quant, /*is_variable=*/false,
          /*sparsity=*/0,
          shape_signature.empty() ? 0 : builder.CreateVector(shape_signature),
          has_rank),
      tflite::CreateTensor(
          builder, builder.CreateVector(std::vector<int32_t>{2, 1}),
          tflite::TensorType_FLOAT16, /*buffer=*/1, builder.CreateString("t1")),
  };

  auto subgraph =
      tflite::CreateSubGraph(builder, builder.CreateVector(tensors));
  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, /*operator_codes=*/0,
      builder.CreateVector(&subgraph, 1), builder.CreateString("tensor_fuzz"),
      builder.CreateVector(buffers));
  tflite::FinishModelBuffer(builder, model);

  const tflite::Model* tfl_model = tflite::GetModel(builder.GetBufferPointer());
  const tflite::SubGraph* tfl_subgraph = tfl_model->subgraphs()->Get(0);
  const tflite::Tensor* tfl_tensor = tfl_subgraph->tensors()->Get(0);
  (void)litert::internal::ValidateFlatBufferTensorQuantization(*tfl_subgraph,
                                                               *tfl_tensor);
  (void)litert::internal::ValidateFlatBufferTensors(tfl_model);
}
FUZZ_TEST(ModelFuzzer, ValidateTensorFuzzerTest);

// Directly fuzzes ShapeInferenceEngine::InferOpShapes across all opcodes with
// structured tensor shapes, constant weight buffers, output counts (including
// zero outputs), and operator options.
void InferOpShapesFuzzerTest(
    int32_t raw_op_code, const std::vector<std::vector<int32_t>>& input_shapes,
    const std::vector<int32_t>& input_elem_types,
    const std::vector<std::string>& input_weights, int32_t num_outputs,
    int32_t option_selector, int32_t param_a, int32_t param_b, int32_t param_c,
    int32_t param_d, const std::vector<int32_t>& vec_param) {
  constexpr int32_t kNumOpCodes =
      static_cast<int32_t>(kLiteRtOpCodeShloComposite) + 1;
  const LiteRtOpCode op_code = static_cast<LiteRtOpCode>(
      static_cast<uint32_t>(raw_op_code) % kNumOpCodes);

  LiteRtModelT model;
  LiteRtOpT op;
  op.SetOpCode(op_code);

  constexpr size_t kMaxFuzzedInputs = 6;
  const size_t clamped_inputs = input_shapes.size() > kMaxFuzzedInputs
                                    ? kMaxFuzzedInputs
                                    : input_shapes.size();
  std::vector<std::unique_ptr<LiteRtTensorT>> input_tensors;
  input_tensors.reserve(clamped_inputs);

  for (size_t i = 0; i < clamped_inputs; ++i) {
    auto tensor = std::make_unique<LiteRtTensorT>();
    LiteRtElementType elem_type = i < input_elem_types.size()
                                      ? MapElementType(input_elem_types[i])
                                      : kLiteRtElementTypeFloat32;
    const auto& dims = input_shapes[i];
    if (dims.size() <= LITERT_TENSOR_MAX_RANK) {
      tensor->SetType(MakeRankedTensorType(elem_type, dims));
    } else {
      tensor->SetType(MakeUnrankedTensor(elem_type));
    }
    if (i < input_weights.size() && !input_weights[i].empty()) {
      const std::string& w = input_weights[i];
      SetWeightsFromOwnedBuffer(
          tensor->Weights(),
          litert::OwningBufferRef<uint8_t>(
              reinterpret_cast<const uint8_t*>(w.data()), w.size()));
    }
    op.Inputs().push_back(tensor.get());
    input_tensors.push_back(std::move(tensor));
  }

  constexpr size_t kMaxFuzzedOutputs = 4;
  const size_t clamped_outputs =
      static_cast<uint32_t>(num_outputs) % kMaxFuzzedOutputs;
  std::vector<std::unique_ptr<LiteRtTensorT>> output_tensors;
  output_tensors.reserve(clamped_outputs);
  for (size_t i = 0; i < clamped_outputs; ++i) {
    auto tensor = std::make_unique<LiteRtTensorT>();
    tensor->SetType(MakeUnrankedTensor(kLiteRtElementTypeFloat32));
    op.Outputs().push_back(tensor.get());
    output_tensors.push_back(std::move(tensor));
  }

  PopulateOpOptions(option_selector, param_a, param_b, param_c, param_d,
                    vec_param, op);

  litert::internal::ShapeInferenceEngine engine(&model);
  (void)engine.InferOpShapes(&op);
}
FUZZ_TEST(ModelFuzzer, InferOpShapesFuzzerTest);

}  // namespace testing
}  // namespace tflite
