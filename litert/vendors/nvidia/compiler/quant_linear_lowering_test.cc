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

// The lowering of a statically quantized model's products (FindQuantRegions in
// tensorrt_graph_builder.cc): graphs with the op sequences of one of its
// blocks, at small dims, are compiled and run against the host references of
// the kernels composed the same way.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "cuda_runtime_api.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"
#include "litert/vendors/nvidia/tensorrt_logger.h"
#include "NvInfer.h"
#include "litert/vendors/nvidia/trtllm/quant_linear.h"
#include "litert/vendors/nvidia/trtllm/quant_linear_reference.h"
#include "tflite/schema/schema_generated.h"

namespace {

namespace ref = litert::nvidia::quant_reference;

class Random {
 public:
  explicit Random(uint64_t seed) : state_(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint32_t Next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return static_cast<uint32_t>(state_ >> 32);
  }
  float Float(float low, float high) {
    return low + (high - low) * (Next() >> 8) / 16777216.0f;
  }
  std::vector<float> Floats(size_t count, float low, float high) {
    std::vector<float> values(count);
    for (auto& value : values) value = Float(low, high);
    return values;
  }
  std::vector<uint8_t> Bytes(size_t count) {
    std::vector<uint8_t> bytes(count);
    for (auto& byte : bytes) byte = static_cast<uint8_t>(Next());
    return bytes;
  }

 private:
  uint64_t state_;
};

// A LiteRT subgraph under construction: tensors, constants and the ops of such
// a model's blocks.
class Graph {
 public:
  Graph() : graph_(model_.EmplaceSubgraph()) {}

  LiteRtSubgraphT& subgraph() { return graph_; }

  LiteRtTensorT& Tensor(LiteRtElementType type,
                        const std::vector<int32_t>& dims) {
    auto& tensor = graph_.EmplaceTensor();
    tensor.SetName("t" + std::to_string(next_name_++));
    tensor.SetType(MakeRankedTensorType(type, dims));
    return tensor;
  }

  LiteRtTensorT& Input(LiteRtElementType type, const std::vector<int32_t>& dims,
                       const char* name) {
    auto& tensor = Tensor(type, dims);
    tensor.SetName(name);
    graph_.Inputs().push_back(&tensor);
    return tensor;
  }

  void Output(LiteRtTensorT& tensor, const char* name) {
    tensor.SetName(name);
    graph_.Outputs().push_back(&tensor);
  }

  // An INT8 tensor with a scale and zero point 0.
  LiteRtTensorT& Int8(const std::vector<int32_t>& dims, float scale) {
    auto& tensor = Tensor(kLiteRtElementTypeInt8, dims);
    tensor.SetQarams(MakePerTensorQuantization(scale, 0));
    return tensor;
  }

  template <typename T>
  LiteRtTensorT& Constant(LiteRtElementType type,
                          const std::vector<int32_t>& dims,
                          const std::vector<T>& values) {
    auto& storage = storage_.emplace_back(values.size() * sizeof(T));
    std::memcpy(storage.data(), values.data(), storage.size());
    auto& tensor = Tensor(type, dims);
    SetWeightsFromUnownedBuffer(
        tensor.Weights(),
        litert::BufferRef<uint8_t>(storage.data(), storage.size()));
    return tensor;
  }

  LiteRtTensorT& Ints(const std::vector<int32_t>& values) {
    return Constant(kLiteRtElementTypeInt32,
                    {static_cast<int32_t>(values.size())}, values);
  }

  // A FLOAT16 scalar, as the model's residual and query multipliers are.
  LiteRtTensorT& Scalar16(float value) {
    return Constant(kLiteRtElementTypeFloat16, {},
                    std::vector<uint16_t>{ref::HalfBits(value)});
  }

  LiteRtTensorT& Scalar32(float value) {
    return Constant(kLiteRtElementTypeFloat32, {}, std::vector<float>{value});
  }

  // INT4 weights [rows, columns] with a scale per row.
  LiteRtTensorT& Int4Weights(int32_t rows, int32_t columns,
                             const std::vector<uint8_t>& packed,
                             const std::vector<float>& scales) {
    auto& weights = Constant(kLiteRtElementTypeInt4, {rows, columns}, packed);
    const std::vector<int64_t> zero_points(rows, 0);
    weights.SetQarams(MakePerChannelQuantization(scales, zero_points,
                                                 /*quantized_dim=*/0, weights));
    return weights;
  }

  // Centered INT2 weights of `dims` with a scale per output: the scale and
  // zero point tensors have 1 at `input_axis`.
  LiteRtTensorT& CenteredInt2Weights(const std::vector<int32_t>& dims,
                                     size_t input_axis,
                                     const std::vector<uint8_t>& packed,
                                     const std::vector<float>& scales) {
    auto& weights = Constant(kLiteRtElementTypeInt2, dims, packed);
    std::vector<int32_t> scale_dims = dims;
    scale_dims[input_axis] = 1;
    auto& scale_tensor =
        Constant(kLiteRtElementTypeFloat32, scale_dims, scales);
    auto& zero_points = Constant(kLiteRtElementTypeFloat32, scale_dims,
                                 std::vector<float>(scales.size(), -0.5f));
    weights.SetQarams(MakeBlockWiseQuantization(&scale_tensor, &zero_points,
                                                /*block_size=*/0));
    return weights;
  }

  LiteRtOpT& Op(LiteRtOpCode code, const std::vector<LiteRtTensorT*>& inputs,
                LiteRtTensorT& output) {
    auto& op = graph_.EmplaceOp();
    op.SetOpCode(code);
    for (auto* input : inputs) litert::internal::AttachInput(input, op);
    litert::internal::AttachOutput(&output, op);
    return op;
  }

  LiteRtTensorT& Cast(LiteRtTensorT& input, LiteRtElementType type) {
    auto& output = Tensor(type, Dims(input));
    Op(kLiteRtOpCodeTflCast, {&input}, output);
    return output;
  }

  LiteRtTensorT& Reshape(LiteRtTensorT& input,
                         const std::vector<int32_t>& dims) {
    auto& output = Like(input, dims);
    auto& op = Op(kLiteRtOpCodeTflReshape, {&input, &Ints(dims)}, output);
    tflite::ReshapeOptionsT reshape;
    reshape.new_shape = dims;
    tflite::BuiltinOptionsUnion options;
    options.Set(std::move(reshape));
    litert::internal::SetTflOptions(op, std::move(options));
    return output;
  }

  LiteRtTensorT& Transpose(LiteRtTensorT& input,
                           const std::vector<int32_t>& perm) {
    const auto dims = Dims(input);
    std::vector<int32_t> out_dims(dims.size());
    for (size_t i = 0; i < perm.size(); ++i) out_dims[i] = dims[perm[i]];
    auto& output = Like(input, out_dims);
    Op(kLiteRtOpCodeTflTranspose, {&input, &Ints(perm)}, output);
    return output;
  }

  LiteRtTensorT& Slice(LiteRtTensorT& input, const std::vector<int32_t>& begin,
                       const std::vector<int32_t>& size) {
    auto& output = Like(input, size);
    Op(kLiteRtOpCodeTflSlice, {&input, &Ints(begin), &Ints(size)}, output);
    return output;
  }

  LiteRtTensorT& Concat(LiteRtTensorT& a, LiteRtTensorT& b, int32_t axis) {
    auto dims = Dims(a);
    dims[axis] += Dims(b)[axis];
    auto& output = Like(a, dims);
    auto& op = Op(kLiteRtOpCodeTflConcatenation, {&a, &b}, output);
    tflite::ConcatenationOptionsT concatenation;
    concatenation.axis = axis;
    tflite::BuiltinOptionsUnion options;
    options.Set(std::move(concatenation));
    litert::internal::SetTflOptions(op, std::move(options));
    return output;
  }

  LiteRtTensorT& Quantize(LiteRtTensorT& input, float scale) {
    auto& output = Int8(Dims(input), scale);
    Op(kLiteRtOpCodeTflQuantize, {&input}, output);
    return output;
  }

  LiteRtTensorT& Dequantize(
      LiteRtTensorT& input,
      LiteRtElementType type = kLiteRtElementTypeFloat16) {
    auto& output = Tensor(type, Dims(input));
    Op(kLiteRtOpCodeTflDequantize, {&input}, output);
    return output;
  }

  LiteRtTensorT& Binary(LiteRtOpCode code, LiteRtTensorT& a, LiteRtTensorT& b) {
    auto& output = Tensor(ElementType(a), Dims(a));
    auto& op = Op(code, {&a, &b}, output);
    tflite::BuiltinOptionsUnion options;
    if (code == kLiteRtOpCodeTflMul) {
      options.Set(tflite::MulOptionsT());
    } else {
      options.Set(tflite::AddOptionsT());
    }
    litert::internal::SetTflOptions(op, std::move(options));
    return output;
  }

  LiteRtTensorT& Unary(LiteRtOpCode code, LiteRtTensorT& input) {
    auto& output = Tensor(ElementType(input), Dims(input));
    Op(code, {&input}, output);
    return output;
  }

  // CAST -> odml.rms_norm -> CAST around a FLOAT16 tensor.
  LiteRtTensorT& Norm(LiteRtTensorT& input, const std::vector<float>& weights) {
    auto& wide = Cast(input, kLiteRtElementTypeFloat32);
    auto& gain = Constant(kLiteRtElementTypeFloat32,
                          {static_cast<int32_t>(weights.size())}, weights);
    auto& normed = Tensor(kLiteRtElementTypeFloat32, Dims(input));
    auto& op = Op(kLiteRtOpCodeShloComposite, {&wide, &gain}, normed);
    flexbuffers::Builder attributes;
    const auto map = attributes.StartMap();
    attributes.Float("epsilon", 1e-6f);
    attributes.EndMap(map);
    attributes.Finish();
    SetComposite(op, "odml.rms_norm", attributes.GetBuffer());
    return Cast(normed, kLiteRtElementTypeFloat16);
  }

  LiteRtTensorT& Rope(LiteRtTensorT& input, LiteRtTensorT& positions,
                      float max_timescale, float proportion) {
    auto& output = Tensor(kLiteRtElementTypeFloat16, Dims(input));
    auto& op = Op(kLiteRtOpCodeShloComposite, {&input, &positions}, output);
    flexbuffers::Builder attributes;
    const auto map = attributes.StartMap();
    attributes.Float("max_timescale", max_timescale);
    attributes.Float("min_timescale", 1.0f);
    attributes.Float("proportion", proportion);
    attributes.EndMap(map);
    attributes.Finish();
    SetComposite(op, "odml.rope", attributes.GetBuffer());
    return output;
  }

  // FULLY_CONNECTED that keeps the leading dims.
  LiteRtTensorT& FullyConnected(LiteRtTensorT& input, LiteRtTensorT& weights,
                                LiteRtTensorT& output) {
    auto& op = Op(kLiteRtOpCodeTflFullyConnected, {&input, &weights}, output);
    tflite::FullyConnectedOptionsT fc;
    fc.keep_num_dims = true;
    tflite::BuiltinOptionsUnion options;
    options.Set(std::move(fc));
    litert::internal::SetTflOptions(op, std::move(options));
    return output;
  }

  LiteRtTensorT& BatchMatmul(LiteRtTensorT& lhs, LiteRtTensorT& rhs,
                             LiteRtTensorT& output) {
    auto& op = Op(kLiteRtOpCodeTflBatchMatmul, {&lhs, &rhs}, output);
    tflite::BatchMatMulOptionsT matmul;
    tflite::BuiltinOptionsUnion options;
    options.Set(std::move(matmul));
    litert::internal::SetTflOptions(op, std::move(options));
    return output;
  }

  // odml.factorized_mlp(x, six weights) -> y FLOAT16 with the six INT8 scales
  // as attributes, as the carve tool writes it.
  LiteRtTensorT& FactorizedMlpComposite(
      LiteRtTensorT& x, const std::vector<LiteRtTensorT*>& weights,
      const float* scales) {
    auto& output = Tensor(kLiteRtElementTypeFloat16, Dims(x));
    std::vector<LiteRtTensorT*> inputs = {&x};
    inputs.insert(inputs.end(), weights.begin(), weights.end());
    auto& op = Op(kLiteRtOpCodeShloComposite, inputs, output);
    flexbuffers::Builder attributes;
    const auto map = attributes.StartMap();
    attributes.Int("version", 1);
    attributes.String("activation", "silu");
    const char* names[6] = {"scale_in",  "scale_up1",   "scale_up2",
                            "scale_act", "scale_down1", "scale_out"};
    for (int i = 0; i < 6; ++i) attributes.Float(names[i], scales[i]);
    attributes.EndMap(map);
    attributes.Finish();
    SetComposite(op, "odml.factorized_mlp", attributes.GetBuffer());
    return output;
  }

  static std::vector<int32_t> Dims(const LiteRtTensorT& tensor) {
    const auto& layout = tensor.Type().second.ranked_tensor_type.layout;
    return std::vector<int32_t>(layout.dimensions,
                                layout.dimensions + layout.rank);
  }

  static LiteRtElementType ElementType(const LiteRtTensorT& tensor) {
    return tensor.Type().second.ranked_tensor_type.element_type;
  }

 private:
  // A tensor of `dims` with the element type and quantization of `like`.
  LiteRtTensorT& Like(LiteRtTensorT& like, const std::vector<int32_t>& dims) {
    auto& tensor = Tensor(ElementType(like), dims);
    if (like.Qparams().first == kLiteRtQuantizationPerTensor) {
      tensor.SetQarams(
          MakePerTensorQuantization(like.Qparams().second.per_tensor.scale, 0));
    }
    return tensor;
  }

  static void SetComposite(LiteRtOpT& op, const char* name,
                           const std::vector<uint8_t>& attributes) {
    tflite::StableHLOCompositeOptionsT composite;
    composite.name = name;
    composite.composite_attributes = attributes;
    tflite::BuiltinOptions2Union options;
    options.Set(std::move(composite));
    litert::internal::SetTflOptions2(op, std::move(options));
  }

  LiteRtModelT model_;
  LiteRtSubgraphT& graph_;
  std::deque<std::vector<uint8_t>> storage_;
  int next_name_ = 0;
};

// The outputs of a compiled graph by name, as raw bytes, and its layers.
struct RunResult {
  std::map<std::string, std::vector<uint8_t>> outputs;
  std::string layers;
};

// Compiles `graph` and runs it once on `inputs` (raw bytes by tensor name).
void RunGraph(Graph& graph,
              const std::map<std::string, std::vector<uint8_t>>& inputs,
              RunResult* result) {
  const auto* context = LrtGetCompilerContext();
  for (auto* op : graph.subgraph().Ops()) {
    ASSERT_TRUE(litert::nvidia::IsTensorRtOpSupported(
        litert::compiler::Op(context, op)))
        << "op code " << op->OpCode();
  }
  auto built = litert::nvidia::BuildTensorRtEngine(
      litert::compiler::Subgraph(context, &graph.subgraph()));
  ASSERT_TRUE(built.HasValue()) << built.Error().Message();
  litert::nvidia::TensorRtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(logger));
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
      built->engine.data(), built->engine.size()));
  ASSERT_NE(engine, nullptr);
  std::unique_ptr<nvinfer1::IEngineInspector> inspector(
      engine->createEngineInspector());
  ASSERT_NE(inspector, nullptr);
  result->layers = inspector->getEngineInformation(
      nvinfer1::LayerInformationFormat::kONELINE);
  std::unique_ptr<nvinfer1::IExecutionContext> execution(
      engine->createExecutionContext());
  ASSERT_NE(execution, nullptr);
  std::vector<void*> buffers;
  const auto bytes_of = [&](const std::string& name) {
    const auto dims = engine->getTensorShape(name.c_str());
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) count *= dims.d[i];
    switch (engine->getTensorDataType(name.c_str())) {
      case nvinfer1::DataType::kFLOAT:
      case nvinfer1::DataType::kINT32:
        return count * 4;
      case nvinfer1::DataType::kHALF:
      case nvinfer1::DataType::kBF16:
        return count * 2;
      default:
        return count;
    }
  };
  const auto& graph_inputs = graph.subgraph().Inputs();
  ASSERT_EQ(built->input_names.size(), graph_inputs.size());
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    const std::string& name = built->input_names[i];
    const auto found = inputs.find(std::string(graph_inputs[i]->Name()));
    ASSERT_NE(found, inputs.end()) << graph_inputs[i]->Name();
    ASSERT_EQ(found->second.size(), bytes_of(name)) << name;
    void* device = nullptr;
    ASSERT_EQ(cudaMalloc(&device, found->second.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device, found->second.data(), found->second.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_TRUE(execution->setTensorAddress(name.c_str(), device));
    buffers.push_back(device);
  }
  const auto& graph_outputs = graph.subgraph().Outputs();
  ASSERT_EQ(built->output_names.size(), graph_outputs.size());
  std::vector<std::pair<std::string, void*>> device_outputs;
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    const std::string& name = built->output_names[i];
    void* device = nullptr;
    ASSERT_EQ(cudaMalloc(&device, bytes_of(name)), cudaSuccess);
    ASSERT_TRUE(execution->setTensorAddress(name.c_str(), device));
    buffers.push_back(device);
    device_outputs.emplace_back(std::string(graph_outputs[i]->Name()), device);
    result->outputs[device_outputs.back().first].resize(bytes_of(name));
  }
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_TRUE(execution->enqueueV3(stream));
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  for (const auto& [name, device] : device_outputs) {
    auto& bytes = result->outputs[name];
    ASSERT_EQ(
        cudaMemcpy(bytes.data(), device, bytes.size(), cudaMemcpyDeviceToHost),
        cudaSuccess);
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  for (void* buffer : buffers) EXPECT_EQ(cudaFree(buffer), cudaSuccess);
}

template <typename T>
std::vector<uint8_t> RawBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

std::vector<uint8_t> HalfBytes(const std::vector<float>& values) {
  std::vector<uint16_t> halves(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    halves[i] = ref::HalfBits(values[i]);
  return RawBytes(halves);
}

// FLOAT16 output bytes as floats.
std::vector<float> HalfFloats(const std::vector<uint8_t>& bytes) {
  std::vector<float> values(bytes.size() / 2);
  for (size_t i = 0; i < values.size(); ++i) {
    uint16_t half = 0;
    std::memcpy(&half, bytes.data() + 2 * i, sizeof(half));
    values[i] = ref::HalfValue(half);
  }
  return values;
}

std::vector<float> RoundedToFp16(std::vector<float> values) {
  for (auto& value : values) value = ref::RoundToFp16(value);
  return values;
}

size_t CountOf(const std::string& text, const std::string& word) {
  size_t count = 0;
  for (size_t at = text.find(word); at != std::string::npos;
       at = text.find(word, at + word.size())) {
    ++count;
  }
  return count;
}

// Every element equal: the FLOAT16 values the reference computes.
void ExpectSame(const std::vector<float>& actual,
                const std::vector<float>& expected, const char* what) {
  ASSERT_EQ(actual.size(), expected.size()) << what;
  size_t different = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i] && ++different <= 5) {
      ADD_FAILURE() << what << " " << i << ": " << actual[i] << " expected "
                    << expected[i];
    }
  }
  EXPECT_EQ(different, 0u) << what << " of " << actual.size();
}

// Rotated values: equal but where the device's sine or cosine rounds to the
// other FLOAT16 neighbour.
void ExpectRotated(const std::vector<float>& actual,
                   const std::vector<float>& expected, const char* what) {
  ASSERT_EQ(actual.size(), expected.size()) << what;
  size_t different = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] == expected[i]) continue;
    ++different;
    ASSERT_NEAR(actual[i], expected[i],
                4e-3f * std::max(1.0f, std::fabs(expected[i])))
        << what << " " << i;
  }
  EXPECT_LE(different, actual.size() / 50 + 1) << what;
}

// [1, tokens, heads, depth] values as [1, heads, tokens, depth].
std::vector<float> HeadMajor(const std::vector<float>& values, int tokens,
                             int heads, int depth) {
  std::vector<float> result(values.size());
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < heads; ++h) {
      std::copy_n(
          values.begin() + (static_cast<size_t>(t) * heads + h) * depth, depth,
          result.begin() + (static_cast<size_t>(h) * tokens + t) * depth);
    }
  }
  return result;
}

LiteRtNvidiaQuantLinearParams Int4Params(int rows, int columns) {
  LiteRtNvidiaQuantLinearParams params = {};
  params.layout = kLiteRtNvidiaQuantLayoutRows;
  params.bit_width = 4;
  params.rows = rows;
  params.columns = columns;
  params.segments = 1;
  params.swap0 = 1;
  params.swap1 = 1;
  params.activation = kLiteRtNvidiaQuantInt8;
  params.requant = kLiteRtNvidiaQuantRequantFp32;
  return params;
}

// XNNPACK's per-row requantization scale.
std::vector<float> Int4Multipliers(float input_scale,
                                   const std::vector<float>& weight_scales,
                                   float output_scale) {
  std::vector<float> multipliers;
  for (const float scale : weight_scales) {
    const float product = input_scale * scale;
    multipliers.push_back(product / output_scale);
  }
  return multipliers;
}

constexpr int kWidth = 64;  // the residual stream
constexpr int kHeads = 4;   // query heads
constexpr int kKvHeads = 2;
constexpr int kDepth = 16;  // head size

// The front of a block that writes a cache (prefill and decode blocks 0 to
// 19): the stream's norm and QUANTIZE, the q, k and v projections, their
// head norms, the rotation of q and k and the query's scalar. One glue, one
// projections and one qkv rope launch.
void RunQkvFront(int tokens, bool rotate_through_reshape, float proportion) {
  SCOPED_TRACE(testing::Message() << tokens << " tokens");
  Random random(11 + tokens);
  const float in_scale = 0.06697f;
  const float q_scale = 0.1162f;
  const float kv_scale = 0.1423f;
  const float q_multiplier = 0.125f;
  const float max_timescale = proportion < 1.0f ? 1000000.0f : 10000.0f;
  const int q_rows = kHeads * kDepth;
  const int kv_rows = kKvHeads * kDepth;
  const auto stream_norm = random.Floats(kWidth, 0.5f, 2.0f);
  const auto q_norm = random.Floats(kDepth, 0.5f, 2.0f);
  const auto k_norm = random.Floats(kDepth, 0.5f, 2.0f);
  const auto v_norm = random.Floats(kDepth, 0.5f, 2.0f);
  const auto q_weights = random.Bytes(q_rows * kWidth / 2);
  const auto k_weights = random.Bytes(kv_rows * kWidth / 2);
  const auto v_weights = random.Bytes(kv_rows * kWidth / 2);
  const auto q_scales = random.Floats(q_rows, 0.004f, 0.008f);
  const auto k_scales = random.Floats(kv_rows, 0.004f, 0.008f);
  const auto v_scales = random.Floats(kv_rows, 0.004f, 0.008f);

  Graph g;
  auto& embeddings =
      g.Input(kLiteRtElementTypeFloat32, {1, tokens, kWidth}, "embeddings");
  auto& input_pos = g.Input(kLiteRtElementTypeInt32, {tokens}, "input_pos");
  auto& x = g.Cast(embeddings, kLiteRtElementTypeFloat16);
  auto& positions = g.Reshape(input_pos, {1, tokens});
  auto& xq = g.Quantize(g.Norm(x, stream_norm), in_scale);
  const auto head = [&](const std::vector<uint8_t>& packed,
                        const std::vector<float>& scales, int heads,
                        float scale,
                        const std::vector<float>& norm) -> LiteRtTensorT& {
    auto& weights = g.Int4Weights(heads * kDepth, kWidth, packed, scales);
    auto& projected = g.FullyConnected(
        xq, weights, g.Int8({1, tokens, heads * kDepth}, scale));
    return g.Norm(
        g.Dequantize(g.Reshape(projected, {1, tokens, heads, kDepth})), norm);
  };
  const auto rotate = [&](LiteRtTensorT& normed, int heads) -> LiteRtTensorT& {
    auto& turned = rotate_through_reshape
                       ? g.Reshape(normed, {1, heads, tokens, kDepth})
                       : g.Transpose(normed, {0, 2, 1, 3});
    return g.Rope(turned, positions, max_timescale, proportion);
  };
  auto& q = g.Binary(
      kLiteRtOpCodeTflMul,
      rotate(head(q_weights, q_scales, kHeads, q_scale, q_norm), kHeads),
      g.Scalar16(q_multiplier));
  auto& k =
      rotate(head(k_weights, k_scales, kKvHeads, kv_scale, k_norm), kKvHeads);
  auto& v = head(v_weights, v_scales, kKvHeads, kv_scale, v_norm);
  g.Output(q, "q");
  g.Output(k, "k");
  g.Output(v, "v");

  const auto values =
      random.Floats(static_cast<size_t>(tokens) * kWidth, -3.0f, 3.0f);
  std::vector<int32_t> position_values(tokens);
  for (int t = 0; t < tokens; ++t) position_values[t] = 4090 + 7 * t;
  RunResult result;
  RunGraph(g,
           {{"embeddings", RawBytes(values)},
            {"input_pos", RawBytes(position_values)}},
           &result);
  if (testing::Test::HasFatalFailure()) return;
  EXPECT_EQ(CountOf(result.layers, "cuda_quant_glue"), 1u) << result.layers;
  EXPECT_EQ(CountOf(result.layers, "cuda_quant_projections"), 1u);
  EXPECT_EQ(CountOf(result.layers, "cuda_quant_qkv_rope"), 1u);

  LiteRtNvidiaQuantGlueParams glue = {};
  glue.width = kWidth;
  glue.x_type = kLiteRtNvidiaQuantFp16;
  glue.norm = 1;
  glue.norm_eps = 1e-6f;
  glue.quantize = kLiteRtNvidiaQuantGlueStatic;
  glue.quantize_scale = ref::RoundToFp16(1.0f / in_scale);
  const auto quantized =
      ref::Glue(glue, tokens, RoundedToFp16(values), {}, {}, stream_norm);
  std::vector<uint8_t> weights = q_weights;
  weights.insert(weights.end(), k_weights.begin(), k_weights.end());
  weights.insert(weights.end(), v_weights.begin(), v_weights.end());
  auto multipliers = Int4Multipliers(in_scale, q_scales, q_scale);
  for (const auto* scales : {&k_scales, &v_scales}) {
    const auto more = Int4Multipliers(in_scale, *scales, kv_scale);
    multipliers.insert(multipliers.end(), more.begin(), more.end());
  }
  const auto projected =
      ref::Linear(Int4Params(q_rows + 2 * kv_rows, kWidth), tokens,
                  std::vector<float>(quantized.q.begin(), quantized.q.end()),
                  weights, multipliers, {}, {});
  LiteRtNvidiaQuantQkvRopeParams rope = {};
  rope.tokens = tokens;
  rope.q_heads = kHeads;
  rope.k_heads = kKvHeads;
  rope.v_heads = kKvHeads;
  rope.depth = kDepth;
  rope.q_scale = ref::RoundToFp16(q_scale);
  rope.k_scale = ref::RoundToFp16(kv_scale);
  rope.v_scale = ref::RoundToFp16(kv_scale);
  rope.epsilon = 1e-6f;
  rope.q_multiplier = q_multiplier;
  auto inverse = ref::InverseTimescales(kDepth, 1.0f, max_timescale);
  const int rotated = static_cast<int>(std::lround(proportion * kDepth / 2));
  std::fill(inverse.begin() + rotated, inverse.end(), 0.0f);
  const auto expected = ref::QkvRope(
      rope, std::vector<int8_t>(projected.begin(), projected.end()),
      position_values, q_norm, k_norm, v_norm, inverse);
  ExpectRotated(HalfFloats(result.outputs["q"]), expected.q, "q");
  ExpectRotated(HalfFloats(result.outputs["k"]), expected.k, "k");
  ExpectSame(HalfFloats(result.outputs["v"]), expected.v, "v");
}

TEST(QuantLinearLoweringTest, QkvFrontOfADecodeBlock) {
  RunQkvFront(/*tokens=*/1, /*rotate_through_reshape=*/true,
              /*proportion=*/1.0f);
}

TEST(QuantLinearLoweringTest, QkvFrontOfAPrefillBlock) {
  RunQkvFront(/*tokens=*/5, /*rotate_through_reshape=*/false,
              /*proportion=*/1.0f);
  // A global layer rotates a quarter of the head.
  RunQkvFront(/*tokens=*/3, /*rotate_through_reshape=*/false,
              /*proportion=*/0.25f);
}

// The factorized MLP's weights and scales at small dims: the stream's halves
// [C] -> [R1] each, regrouped into B blocks of 2 R1 / B -> [N] per block ->
// SiLU -> [2 D] per block -> its halves [B D] -> [R4] each.
struct FactorizedMlp {
  static constexpr int kC = kWidth / 2;
  static constexpr int kR1 = 64;
  static constexpr int kB = 4;
  static constexpr int kK2 = 2 * kR1 / kB;
  static constexpr int kN = 48;
  static constexpr int kD = 16;
  static constexpr int kR4 = kWidth / 2;

  explicit FactorizedMlp(Random& random)
      : up1{random.Bytes(kR1 * kC / 4), random.Bytes(kR1 * kC / 4)},
        up2(random.Bytes(kB * kK2 * kN / 4)),
        down1(random.Bytes(kB * kN * 2 * kD / 4)),
        down2{random.Bytes(kR4 * kB * kD / 4), random.Bytes(kR4 * kB * kD / 4)},
        up1_scales{random.Floats(kR1, 0.02f, 0.04f),
                   random.Floats(kR1, 0.02f, 0.04f)},
        up2_scales(random.Floats(kB * kN, 0.05f, 0.08f)),
        down1_scales(random.Floats(kB * 2 * kD, 0.02f, 0.04f)),
        down2_scales{random.Floats(kR4, 0.01f, 0.03f),
                     random.Floats(kR4, 0.01f, 0.03f)} {}

  // The ops of the model's graph from the stream's INT8 tensor `xq`
  // [m, 2 C] to the INT8 result [m, 2 R4].
  LiteRtTensorT& Build(Graph& g, LiteRtTensorT& xq, int m, bool rank4) const {
    LiteRtTensorT* halves[2];
    for (int half = 0; half < 2; ++half) {
      auto& weights =
          g.CenteredInt2Weights({kR1, kC}, 1, up1[half], up1_scales[half]);
      auto& product = g.FullyConnected(g.Slice(xq, {0, half * kC}, {m, kC}),
                                       weights, g.Int8({m, kR1}, scales[1]));
      halves[half] = &g.Reshape(product, {m, kB, 1, kR1 / kB});
    }
    auto& blocks =
        g.Reshape(g.Concat(*halves[0], *halves[1], 2), {m, kB, 1, kK2});
    // The decode graph's weights have a leading 1, the prefill's do not.
    const auto block_dims = [&](int inputs, int outputs) {
      return rank4 ? std::vector<int32_t>{1, kB, inputs, outputs}
                   : std::vector<int32_t>{kB, inputs, outputs};
    };
    auto& up2_weights = g.CenteredInt2Weights(block_dims(kK2, kN),
                                              rank4 ? 2 : 1, up2, up2_scales);
    auto& wide =
        g.BatchMatmul(blocks, up2_weights, g.Int8({m, kB, 1, kN}, scales[2]));
    auto& real = g.Dequantize(g.Reshape(wide, {m, kB * kN}));
    auto& silu = g.Binary(kLiteRtOpCodeTflMul, real,
                          g.Unary(kLiteRtOpCodeTflLogistic, real));
    auto& activated = g.Reshape(g.Quantize(silu, scales[3]), {m, kB, 1, kN});
    auto& down1_weights = g.CenteredInt2Weights(
        block_dims(kN, 2 * kD), rank4 ? 2 : 1, down1, down1_scales);
    auto& narrow =
        g.Reshape(g.BatchMatmul(activated, down1_weights,
                                g.Int8({m, kB, 1, 2 * kD}, scales[4])),
                  {m, kB, 2, kD});
    LiteRtTensorT* results[2];
    for (int half = 0; half < 2; ++half) {
      auto& weights = g.CenteredInt2Weights({kR4, kB * kD}, 1, down2[half],
                                            down2_scales[half]);
      auto& rows = g.Reshape(g.Slice(narrow, {0, 0, half, 0}, {m, kB, 1, kD}),
                             {m, kB * kD});
      results[half] =
          &g.FullyConnected(rows, weights, g.Int8({m, kR4}, scales[5]));
    }
    return g.Concat(*results[0], *results[1], 1);
  }

  // The same MLP as the composite the carve tool makes of those ops, from
  // the FLOAT16 tensor `x` [1, m, 2 C] in front of its QUANTIZE.
  LiteRtTensorT& BuildComposite(Graph& g, LiteRtTensorT& x, bool rank4) const {
    const auto block_dims = [&](int inputs, int outputs) {
      return rank4 ? std::vector<int32_t>{1, kB, inputs, outputs}
                   : std::vector<int32_t>{kB, inputs, outputs};
    };
    const size_t block_axis = rank4 ? 2 : 1;
    return g.FactorizedMlpComposite(
        x,
        {&g.CenteredInt2Weights({kR1, kC}, 1, up1[0], up1_scales[0]),
         &g.CenteredInt2Weights({kR1, kC}, 1, up1[1], up1_scales[1]),
         &g.CenteredInt2Weights(block_dims(kK2, kN), block_axis, up2,
                                up2_scales),
         &g.CenteredInt2Weights(block_dims(kN, 2 * kD), block_axis, down1,
                                down1_scales),
         &g.CenteredInt2Weights({kR4, kB * kD}, 1, down2[0], down2_scales[0]),
         &g.CenteredInt2Weights({kR4, kB * kD}, 1, down2[1], down2_scales[1])},
        scales);
  }

  // The reference: the four products on the INT8 rows `xq`.
  std::vector<float> Reference(const std::vector<float>& xq, int m) const {
    const auto join = [](const std::vector<uint8_t>& a,
                         const std::vector<uint8_t>& b) {
      std::vector<uint8_t> all(a);
      all.insert(all.end(), b.begin(), b.end());
      return all;
    };
    const auto join_scales = [](const std::vector<float>& a,
                                const std::vector<float>& b) {
      std::vector<float> all(a);
      all.insert(all.end(), b.begin(), b.end());
      return all;
    };
    const auto ratio = [&](int from, int to) {
      return static_cast<double>(scales[from]) /
             static_cast<double>(scales[to]);
    };
    LiteRtNvidiaQuantLinearParams params = {};
    params.layout = kLiteRtNvidiaQuantLayoutRows;
    params.bit_width = 2;
    params.rows = 2 * kR1;
    params.columns = kC;
    params.segments = 2;
    params.swap0 = 2;
    params.swap1 = kB;
    params.requant = kLiteRtNvidiaQuantRequantDouble;
    params.ratio = ratio(0, 1);
    auto stage = ref::Linear(params, m, xq, join(up1[0], up1[1]),
                             join_scales(up1_scales[0], up1_scales[1]), {}, {});
    // The SiLU between the INT8 tensors, per INT8 value.
    std::vector<int8_t> silu(256);
    for (int q = -128; q <= 127; ++q) {
      const float x = ref::RoundToFp16(q * ref::RoundToFp16(scales[2]));
      const float logistic = ref::RoundToFp16(
          static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(x)))));
      silu[q + 128] = static_cast<int8_t>(ref::Quantize(
          ref::RoundToFp16(x * logistic), ref::RoundToFp16(1.0f / scales[3])));
    }
    params = {};
    params.layout = kLiteRtNvidiaQuantLayoutBlocks;
    params.bit_width = 2;
    params.rows = kB * kN;
    params.columns = kK2;
    params.segments = kB;
    params.swap0 = 1;
    params.swap1 = 1;
    params.requant = kLiteRtNvidiaQuantRequantDouble;
    params.lut = 1;
    params.ratio = ratio(1, 2);
    stage = ref::Linear(params, m, stage, up2, up2_scales, silu, {});
    params.rows = kB * 2 * kD;
    params.columns = kN;
    params.swap0 = kB;
    params.swap1 = 2;
    params.lut = 0;
    params.ratio = ratio(3, 4);
    stage = ref::Linear(params, m, stage, down1, down1_scales, {}, {});
    params = {};
    params.layout = kLiteRtNvidiaQuantLayoutRows;
    params.bit_width = 2;
    params.rows = 2 * kR4;
    params.columns = kB * kD;
    params.segments = 2;
    params.swap0 = 1;
    params.swap1 = 1;
    params.requant = kLiteRtNvidiaQuantRequantDouble;
    params.ratio = ratio(4, 5);
    return ref::Linear(params, m, stage, join(down2[0], down2[1]),
                       join_scales(down2_scales[0], down2_scales[1]), {}, {});
  }

  std::vector<uint8_t> up1[2];
  std::vector<uint8_t> up2;
  std::vector<uint8_t> down1;
  std::vector<uint8_t> down2[2];
  std::vector<float> up1_scales[2];
  std::vector<float> up2_scales;
  std::vector<float> down1_scales;
  std::vector<float> down2_scales[2];
  // Of the input, up-1, up-2, the SiLU, down-1 and down-2.
  float scales[6] = {0.02716f, 0.06822f, 0.114f, 0.05113f, 0.02303f, 0.02726f};
};

// The back of a block and the front of the next one that reads another block's
// cache (decode blocks 20 to 49): the attention output's QUANTIZE and output
// projection, the residual update, the norm and QUANTIZE in front of the
// factorized MLP, the MLP, the second residual update, the next block's norm,
// QUANTIZE and query projection with its head norm, rotation and scalar.
// Launches: the output projection (it quantizes its float input), a glue, the
// four of the factorized MLP, a glue, the query projection, a qkv rope.
void RunBlockBack(int tokens, bool prefill, bool carved = false) {
  SCOPED_TRACE(testing::Message() << tokens << " tokens, carved " << carved);
  Random random(23 + tokens);
  const float attention_scale = 0.04118f;
  const float projected_scale = 0.05521f;
  const float next_scale = 0.03644f;
  const float q_scale = 0.131f;
  const float q_multiplier = 0.125f;
  const float c_x[2] = {0.796f, 0.5376f};
  const float c_y[2] = {0.6055f, 0.8433f};
  const FactorizedMlp factorized_mlp(random);
  const auto out_weights = random.Bytes(kWidth * kWidth / 2);
  const auto out_scales = random.Floats(kWidth, 0.006f, 0.009f);
  const auto q_weights = random.Bytes(kWidth * kWidth / 2);
  const auto q_scales = random.Floats(kWidth, 0.006f, 0.009f);
  const auto norm_after_attention = random.Floats(kWidth, 0.5f, 2.0f);
  const auto norm_before_mlp = random.Floats(kWidth, 0.5f, 2.0f);
  const auto norm_after_mlp = random.Floats(kWidth, 0.5f, 2.0f);
  const auto norm_next = random.Floats(kWidth, 0.5f, 2.0f);
  const auto q_norm = random.Floats(kDepth, 0.5f, 2.0f);

  Graph g;
  auto& x = g.Input(kLiteRtElementTypeFloat16, {1, tokens, kWidth}, "x");
  auto& attention = g.Input(kLiteRtElementTypeFloat16,
                            {1, kHeads, tokens, kDepth}, "attention");
  auto& input_pos = g.Input(kLiteRtElementTypeInt32, {tokens}, "input_pos");
  auto& positions = g.Reshape(input_pos, {1, tokens});
  // The prefill quantizes the attention output head-major and transposes
  // the INT8 tensor; the decode reshapes its one token first.
  LiteRtTensorT* rows = nullptr;
  if (prefill) {
    rows = &g.Reshape(
        g.Transpose(g.Quantize(attention, attention_scale), {0, 2, 1, 3}),
        {1, tokens, kWidth});
  } else {
    rows =
        &g.Reshape(g.Quantize(g.Reshape(attention, {1, tokens, kHeads, kDepth}),
                              attention_scale),
                   {1, tokens, kWidth});
  }
  auto& projected = g.FullyConnected(
      *rows, g.Int4Weights(kWidth, kWidth, out_weights, out_scales),
      g.Int8({1, tokens, kWidth}, projected_scale));
  const auto residual = [&](LiteRtTensorT& stream, LiteRtTensorT& branch,
                            const std::vector<float>& norm,
                            int i) -> LiteRtTensorT& {
    return g.Binary(
        kLiteRtOpCodeTflAdd,
        g.Binary(kLiteRtOpCodeTflMul, stream, g.Scalar16(c_x[i])),
        g.Binary(kLiteRtOpCodeTflMul,
                 g.Norm(Graph::ElementType(branch) == kLiteRtElementTypeInt8
                            ? g.Dequantize(branch)
                            : branch,
                        norm),
                 g.Scalar16(c_y[i])));
  };
  auto& x1 = residual(x, projected, norm_after_attention, 0);
  // The carved model has the MLP as one composite between the FLOAT16
  // tensors around it; the model as exported has its ops.
  LiteRtTensorT* mlp_out = nullptr;
  if (carved) {
    mlp_out = &factorized_mlp.BuildComposite(g, g.Norm(x1, norm_before_mlp),
                                             /*rank4=*/!prefill);
  } else {
    auto& mlp_in =
        g.Quantize(g.Reshape(g.Norm(x1, norm_before_mlp), {tokens, kWidth}),
                   factorized_mlp.scales[0]);
    mlp_out =
        &g.Reshape(factorized_mlp.Build(g, mlp_in, tokens, /*rank4=*/!prefill),
                   {1, tokens, kWidth});
  }
  auto& x2 = residual(x1, *mlp_out, norm_after_mlp, 1);
  auto& next = g.Quantize(g.Norm(x2, norm_next), next_scale);
  auto& q_projected =
      g.FullyConnected(next, g.Int4Weights(kWidth, kWidth, q_weights, q_scales),
                       g.Int8({1, tokens, kWidth}, q_scale));
  auto& q_normed =
      g.Norm(g.Dequantize(g.Reshape(q_projected, {1, tokens, kHeads, kDepth})),
             q_norm);
  auto& q = g.Binary(
      kLiteRtOpCodeTflMul,
      g.Rope(prefill ? g.Transpose(q_normed, {0, 2, 1, 3})
                     : g.Reshape(q_normed, {1, kHeads, tokens, kDepth}),
             positions, 10000.0f, 1.0f),
      g.Scalar16(q_multiplier));
  g.Output(x2, "x2");
  g.Output(q, "q");

  const auto x_values = RoundedToFp16(
      random.Floats(static_cast<size_t>(tokens) * kWidth, -3.0f, 3.0f));
  const auto attention_values = RoundedToFp16(
      random.Floats(static_cast<size_t>(tokens) * kWidth, -5.5f, 5.5f));
  std::vector<int32_t> position_values(tokens);
  for (int t = 0; t < tokens; ++t) position_values[t] = 700 + t;
  RunResult result;
  RunGraph(g,
           {{"x", HalfBytes(x_values)},
            {"attention", HalfBytes(attention_values)},
            {"input_pos", RawBytes(position_values)}},
           &result);
  if (testing::Test::HasFatalFailure()) return;
  EXPECT_EQ(CountOf(result.layers, "cuda_quant_projections"), 2u)
      << result.layers;
  EXPECT_EQ(CountOf(result.layers, "cuda_quant_glue"), 2u);
  EXPECT_EQ(CountOf(result.layers, "cuda_quant_factorized_mlp"), 4u);
  EXPECT_EQ(CountOf(result.layers, "cuda_quant_qkv_rope"), 1u);

  // The attention output arrives [1, heads, tokens, depth]; the projection
  // reads [tokens, heads * depth].
  std::vector<float> attention_rows(attention_values.size());
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < kHeads; ++h) {
      // The decode's reshape keeps the order of its one token.
      const size_t from = prefill || tokens == 1
                              ? (static_cast<size_t>(h) * tokens + t) * kDepth
                              : (static_cast<size_t>(t) * kHeads + h) * kDepth;
      std::copy_n(attention_values.begin() + from, kDepth,
                  attention_rows.begin() +
                      (static_cast<size_t>(t) * kHeads + h) * kDepth);
    }
  }
  auto out_params = Int4Params(kWidth, kWidth);
  out_params.activation = kLiteRtNvidiaQuantFp16;
  out_params.quantize_scale = ref::RoundToFp16(1.0f / attention_scale);
  const auto projected_q = ref::Linear(
      out_params, tokens, attention_rows, out_weights,
      Int4Multipliers(attention_scale, out_scales, projected_scale), {}, {});
  LiteRtNvidiaQuantGlueParams glue = {};
  glue.width = kWidth;
  glue.x_type = kLiteRtNvidiaQuantFp16;
  glue.branch = 1;
  glue.norm = 1;
  glue.quantize = kLiteRtNvidiaQuantGlueStatic;
  glue.branch_scale = ref::RoundToFp16(projected_scale);
  glue.branch_eps = 1e-6f;
  glue.c_x = ref::RoundToFp16(c_x[0]);
  glue.c_y = ref::RoundToFp16(c_y[0]);
  glue.norm_eps = 1e-6f;
  glue.quantize_scale = ref::RoundToFp16(1.0f / factorized_mlp.scales[0]);
  const auto before_mlp =
      ref::Glue(glue, tokens, x_values,
                std::vector<int8_t>(projected_q.begin(), projected_q.end()),
                norm_after_attention, norm_before_mlp);
  const auto mlp = factorized_mlp.Reference(
      std::vector<float>(before_mlp.q.begin(), before_mlp.q.end()), tokens);
  glue.branch_scale = ref::RoundToFp16(factorized_mlp.scales[5]);
  glue.c_x = ref::RoundToFp16(c_x[1]);
  glue.c_y = ref::RoundToFp16(c_y[1]);
  glue.quantize_scale = ref::RoundToFp16(1.0f / next_scale);
  const auto after_mlp = ref::Glue(glue, tokens, before_mlp.x,
                                   std::vector<int8_t>(mlp.begin(), mlp.end()),
                                   norm_after_mlp, norm_next);
  ExpectSame(HalfFloats(result.outputs["x2"]), after_mlp.x, "x2");
  const auto q_rows = ref::Linear(
      Int4Params(kWidth, kWidth), tokens,
      std::vector<float>(after_mlp.q.begin(), after_mlp.q.end()), q_weights,
      Int4Multipliers(next_scale, q_scales, q_scale), {}, {});
  LiteRtNvidiaQuantQkvRopeParams rope = {};
  rope.tokens = tokens;
  rope.q_heads = kHeads;
  rope.depth = kDepth;
  rope.q_scale = ref::RoundToFp16(q_scale);
  rope.epsilon = 1e-6f;
  rope.q_multiplier = q_multiplier;
  const auto expected = ref::QkvRope(
      rope, std::vector<int8_t>(q_rows.begin(), q_rows.end()), position_values,
      q_norm, {}, {}, ref::InverseTimescales(kDepth, 1.0f, 10000.0f));
  ExpectRotated(HalfFloats(result.outputs["q"]), expected.q, "q");
}

TEST(QuantLinearLoweringTest, BackOfADecodeBlock) {
  RunBlockBack(/*tokens=*/1, /*prefill=*/false);
}

TEST(QuantLinearLoweringTest, BackOfAPrefillBlock) {
  RunBlockBack(/*tokens=*/6, /*prefill=*/true);
}

// The same blocks of the carved model: odml.factorized_mlp is lowered as the
// four launches of its ops, behind the same glue, with the same results.
TEST(QuantLinearLoweringTest, BackOfACarvedBlock) {
  RunBlockBack(/*tokens=*/1, /*prefill=*/false, /*carved=*/true);
  RunBlockBack(/*tokens=*/6, /*prefill=*/true, /*carved=*/true);
}

// The tail of the decode graph: the final norm, whose FLOAT16 result is the
// model's hidden state output (through a CAST) and the head's input, the
// vocabulary product on centered INT2 weights and the soft cap. A glue (the
// row's dynamic quantization) and the head's product, with the cap in it.
// With `residual` the last residual update is in front of the final norm, as
// in the model, and the glue takes it too.
void RunHead(bool residual) {
  SCOPED_TRACE(residual);
  Random random(31);
  constexpr int kVocab = 512;
  const float cap = 30.0f;
  const float branch_in_scale = 0.04118f;
  const float branch_scale = 0.05521f;
  const float c_x = 0.5376f;
  const float c_y = 0.8433f;
  const auto final_norm = random.Floats(kWidth, 0.5f, 2.0f);
  const auto branch_norm = random.Floats(kWidth, 0.5f, 2.0f);
  const auto branch_weights = random.Bytes(kWidth * kWidth / 2);
  const auto branch_scales = random.Floats(kWidth, 0.006f, 0.009f);
  const auto weights = random.Bytes(kVocab * kWidth / 4);
  const auto scales = random.Floats(kVocab, 0.02f, 0.025f);

  Graph g;
  auto& x = g.Input(kLiteRtElementTypeFloat16, {1, 1, kWidth}, "x");
  LiteRtTensorT* stream = &x;
  if (residual) {
    auto& branch_in =
        g.Input(kLiteRtElementTypeFloat16, {1, 1, kWidth}, "branch");
    auto& branch = g.FullyConnected(
        g.Quantize(branch_in, branch_in_scale),
        g.Int4Weights(kWidth, kWidth, branch_weights, branch_scales),
        g.Int8({1, 1, kWidth}, branch_scale));
    stream = &g.Binary(
        kLiteRtOpCodeTflAdd, g.Binary(kLiteRtOpCodeTflMul, x, g.Scalar16(c_x)),
        g.Binary(kLiteRtOpCodeTflMul, g.Norm(g.Dequantize(branch), branch_norm),
                 g.Scalar16(c_y)));
  }
  auto& hidden = g.Norm(*stream, final_norm);
  g.Output(g.Cast(hidden, kLiteRtElementTypeFloat32), "activations");
  auto& product = g.FullyConnected(
      hidden, g.CenteredInt2Weights({kVocab, kWidth}, 1, weights, scales),
      g.Tensor(kLiteRtElementTypeFloat16, {1, 1, kVocab}));
  auto& capped =
      g.Binary(kLiteRtOpCodeTflMul,
               g.Unary(kLiteRtOpCodeTflTanh,
                       g.Binary(kLiteRtOpCodeTflMul,
                                g.Cast(product, kLiteRtElementTypeFloat32),
                                g.Scalar32(1.0f / cap))),
               g.Scalar32(cap));
  g.Output(capped, "logits");

  const auto x_values = RoundedToFp16(random.Floats(kWidth, -3.0f, 3.0f));
  const auto branch_values = RoundedToFp16(random.Floats(kWidth, -5.0f, 5.0f));
  std::map<std::string, std::vector<uint8_t>> inputs = {
      {"x", HalfBytes(x_values)}};
  if (residual) inputs["branch"] = HalfBytes(branch_values);
  RunResult result;
  RunGraph(g, inputs, &result);
  if (testing::Test::HasFatalFailure()) return;
  EXPECT_EQ(CountOf(result.layers, "cuda_quant_head"), 2u) << result.layers;
  EXPECT_EQ(CountOf(result.layers, "cuda_quant_projections"),
            residual ? 1u : 0u);

  LiteRtNvidiaQuantGlueParams glue = {};
  glue.width = kWidth;
  glue.x_type = kLiteRtNvidiaQuantFp16;
  glue.norm = 1;
  glue.norm_eps = 1e-6f;
  glue.quantize = kLiteRtNvidiaQuantGlueDynamic;
  std::vector<int8_t> branch_q;
  if (residual) {
    auto branch_params = Int4Params(kWidth, kWidth);
    branch_params.activation = kLiteRtNvidiaQuantFp16;
    branch_params.quantize_scale = ref::RoundToFp16(1.0f / branch_in_scale);
    const auto projected = ref::Linear(
        branch_params, 1, branch_values, branch_weights,
        Int4Multipliers(branch_in_scale, branch_scales, branch_scale), {}, {});
    branch_q.assign(projected.begin(), projected.end());
    glue.branch = 1;
    glue.branch_scale = ref::RoundToFp16(branch_scale);
    glue.branch_eps = 1e-6f;
    glue.c_x = ref::RoundToFp16(c_x);
    glue.c_y = ref::RoundToFp16(c_y);
  }
  const auto quantized =
      ref::Glue(glue, 1, x_values, branch_q, branch_norm, final_norm);
  LiteRtNvidiaQuantLinearParams params = {};
  params.layout = kLiteRtNvidiaQuantLayoutRows;
  params.bit_width = 2;
  params.rows = kVocab;
  params.columns = kWidth;
  params.segments = 1;
  params.swap0 = 1;
  params.swap1 = 1;
  params.requant = kLiteRtNvidiaQuantRequantLogits;
  params.cap_in = 1.0f / cap;
  params.cap_out = cap;
  const auto expected = ref::Linear(
      params, 1, std::vector<float>(quantized.q.begin(), quantized.q.end()),
      weights, scales, {}, quantized.row_scale);
  std::vector<float> logits(kVocab);
  std::memcpy(logits.data(), result.outputs["logits"].data(),
              logits.size() * sizeof(float));
  for (int i = 0; i < kVocab; ++i) {
    // The device's tanh and the host's differ in the last bits.
    ASSERT_NEAR(logits[i], expected[i], 1e-5f * std::fabs(expected[i]) + 1e-6f)
        << i;
  }
  // The hidden state output keeps its FLOAT16 values.
  std::vector<float> hidden_values(kWidth);
  std::memcpy(hidden_values.data(), result.outputs["activations"].data(),
              hidden_values.size() * sizeof(float));
  ExpectSame(hidden_values, quantized.z, "activations");
}

TEST(QuantLinearLoweringTest, HeadMatchesTheDynamicRangeKernel) {
  RunHead(/*residual=*/false);
  RunHead(/*residual=*/true);
}

// Projections whose results ops outside the quant regions read: three
// FULLY_CONNECTED ops on one QUANTIZE are one launch, and each DEQUANTIZE
// takes the real view of its columns of the INT8 result, exact at scales
// that the activation type represents.
TEST(QuantLinearLoweringTest, ProjectionResultsFeedOpsOutsideRegions) {
  Random random(41);
  constexpr int kTokens = 2;
  const float in_scale = 0.05f;
  const int rows[3] = {kWidth, kWidth / 2, kWidth / 2};
  // Powers of two: exact in every float type.
  const float out_scales[3] = {0.25f, 0.5f, 0.125f};
  Graph g;
  auto& x = g.Input(kLiteRtElementTypeFloat16, {1, kTokens, kWidth}, "x");
  auto& xq = g.Quantize(x, in_scale);
  std::vector<uint8_t> weights[3];
  std::vector<float> scales[3];
  const char* names[3] = {"a", "b", "c"};
  for (int i = 0; i < 3; ++i) {
    weights[i] = random.Bytes(rows[i] * kWidth / 2);
    scales[i] = random.Floats(rows[i], 0.004f, 0.008f);
    auto& projected = g.FullyConnected(
        xq, g.Int4Weights(rows[i], kWidth, weights[i], scales[i]),
        g.Int8({1, kTokens, rows[i]}, out_scales[i]));
    g.Output(g.Dequantize(projected), names[i]);
  }
  const auto x_values =
      RoundedToFp16(random.Floats(kTokens * kWidth, -5.0f, 5.0f));
  RunResult result;
  RunGraph(g, {{"x", HalfBytes(x_values)}}, &result);
  if (testing::Test::HasFatalFailure()) return;
  EXPECT_EQ(CountOf(result.layers, "cuda_quant_projections"), 1u)
      << result.layers;
  for (int i = 0; i < 3; ++i) {
    auto params = Int4Params(rows[i], kWidth);
    params.activation = kLiteRtNvidiaQuantFp16;
    params.quantize_scale = ref::RoundToFp16(1.0f / in_scale);
    auto expected = ref::Linear(
        params, kTokens, x_values, weights[i],
        Int4Multipliers(in_scale, scales[i], out_scales[i]), {}, {});
    for (auto& value : expected) value *= out_scales[i];
    ExpectSame(HalfFloats(result.outputs[names[i]]), expected, names[i]);
  }
}

// The Gemma 4 E2B and E4B have the same INT4 products between FLOAT32
// tensors. They are not quant regions: they keep the FP16-activation
// lowering, which carries the INT8 tensors as reals (no rounding), so the
// result is the float product to the activation type's precision.
TEST(QuantLinearLoweringTest, ProductsBetweenFloat32TensorsKeepTheirLowering) {
  Random random(43);
  constexpr int kTokens = 1;
  const float in_scale = 0.05f;
  const float out_scale = 0.25f;
  const auto weights = random.Bytes(kWidth * kWidth / 2);
  const auto scales = random.Floats(kWidth, 0.004f, 0.008f);
  Graph g;
  auto& x = g.Input(kLiteRtElementTypeFloat32, {1, kTokens, kWidth}, "x");
  auto& projected = g.FullyConnected(
      g.Quantize(x, in_scale), g.Int4Weights(kWidth, kWidth, weights, scales),
      g.Int8({1, kTokens, kWidth}, out_scale));
  g.Output(g.Dequantize(projected, kLiteRtElementTypeFloat32), "y");
  // Multiples of 1/8: exact in the activation type.
  std::vector<float> x_values(kTokens * kWidth);
  for (auto& value : x_values) {
    value = 0.125f * (static_cast<int>(random.Next() % 33) - 16);
  }
  RunResult result;
  RunGraph(g, {{"x", RawBytes(x_values)}}, &result);
  if (testing::Test::HasFatalFailure()) return;
  EXPECT_EQ(CountOf(result.layers, "cuda_quant"), 0u) << result.layers;
  std::vector<float> y(kTokens * kWidth);
  std::memcpy(y.data(), result.outputs["y"].data(), y.size() * sizeof(float));
  for (int row = 0; row < kWidth; ++row) {
    float expected = 0.0f;
    for (int k = 0; k < kWidth; ++k) {
      expected += x_values[k] *
                  ref::PackedWeight(weights.data(), 4,
                                    static_cast<size_t>(row) * kWidth + k);
    }
    expected *= scales[row];
    EXPECT_NEAR(y[row], expected, 0.02f * std::fabs(expected) + 1e-2f) << row;
  }
}

}  // namespace
