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

// The attention composites over KV caches in the forms the general plugin
// takes (LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION=1), as graphs hold them,
// lowered by the graph builder and run step by step with the caches in place
// against the host references of trtllm/general_attention_reference.h:
//  - a decode block: float keys and values into packed INT4 caches
//    (odml.cache_update), then odml.sdpa_transposed with a float query, an
//    additive float mask, a scale and a soft cap, at depth 128;
//  - a prefill block: INT8 keys and values of another scale into UINT8
//    caches, and an INT8 query with a bool mask;
//  - odml.fused_sdpa_cache_update over packed INT4 ring caches with a float
//    chunk, a float mask and a soft cap, across the ring's end, and its
//    one-output form;
//  - without the switch, none of these composites is taken.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include "litert/compiler/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/vendors/nvidia/compiler/quant_attention_test_graph.h"
#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"
#include "litert/vendors/nvidia/tensorrt_logger.h"
#include "NvInfer.h"
#include "litert/vendors/nvidia/trtllm/general_attention_reference.h"
#include "litert/vendors/nvidia/trtllm/qsa_reference.h"

namespace {

namespace ref = litert::nvidia::general_reference;
namespace qsa_ref = litert::nvidia::qsa_reference;
using litert::nvidia::quant_test::Graph;

constexpr int kKvHeads = 2;
constexpr int kQueryHeads = 8;
constexpr int kGroup = kQueryHeads / kKvHeads;

class Random {
 public:
  explicit Random(uint32_t seed) : state_(seed * 2654435761u + 7u) {}
  float Next() {  // [-1, 1)
    state_ = state_ * 1664525u + 1013904223u;
    return static_cast<float>(state_ >> 8) / 8388608.0f - 1.0f;
  }
  std::vector<float> Values(size_t count, float range) {
    std::vector<float> values(count);
    for (auto& value : values) value = Next() * range;
    return values;
  }
  int Int(int low, int high) {  // [low, high]
    const float u = (Next() + 1.0f) * 0.5f;
    return std::min(high, low + static_cast<int>(u * (high - low + 1)));
  }

 private:
  uint32_t state_;
};

// The value a FLOAT32 input has inside the engine (BF16 activations).
float Bf16(float value) {
  return qsa_ref::Bf16BitsToFloat(qsa_ref::FloatToBf16Bits(value));
}

template <typename T>
std::vector<uint8_t> Bytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

// A cache's unscaled int values as the model stores them (INT4 packed, UINT8
// with its zero point), and back.
std::vector<uint8_t> PackCache(const std::vector<int>& values, int bits,
                               bool unsigned_bytes) {
  std::vector<uint8_t> bytes;
  if (bits == 4) {
    for (size_t i = 0; i < values.size(); i += 2) {
      bytes.push_back(static_cast<uint8_t>((values[i] & 0xF) |
                                           ((values[i + 1] & 0xF) << 4)));
    }
    return bytes;
  }
  for (int v : values) {
    bytes.push_back(static_cast<uint8_t>(unsigned_bytes ? v + 128 : v));
  }
  return bytes;
}

std::vector<int> UnpackCache(const std::vector<uint8_t>& bytes, int bits,
                             bool unsigned_bytes) {
  std::vector<int> values;
  if (bits == 4) {
    for (uint8_t byte : bytes) {
      for (int nibble : {byte & 0xF, byte >> 4}) {
        values.push_back(nibble >= 8 ? nibble - 16 : nibble);
      }
    }
    return values;
  }
  for (uint8_t byte : bytes) {
    values.push_back(unsigned_bytes
                         ? static_cast<int>(byte) - 128
                         : static_cast<int>(static_cast<int8_t>(byte)));
  }
  return values;
}

ref::Caches HostCaches(const std::vector<int>& k, const std::vector<int>& v,
                       int len, int depth, int bits, float k_scale,
                       float v_scale) {
  ref::Caches caches;
  caches.heads = kKvHeads;
  caches.len = len;
  caches.depth = depth;
  caches.bits = bits;
  caches.k_scale = k_scale;
  caches.v_scale = v_scale;
  caches.k.assign(k.begin(), k.end());
  caches.v.assign(v.begin(), v.end());
  return caches;
}

// The engine of a graph with a device buffer per input and output; the
// outputs `aliases` names share an input's buffer, as LiteRT-LM binds caches.
class Session {
 public:
  void Build(Graph* graph, const std::vector<std::pair<int, int>>& aliases) {
    const litert::compiler::Subgraph subgraph(LrtGetCompilerContext(),
                                              &graph->Get());
    for (const auto& op : subgraph.Ops()) {
      ASSERT_TRUE(litert::nvidia::IsTensorRtOpSupported(op))
          << "op code " << op.Code();
    }
    auto built = litert::nvidia::BuildTensorRtEngine(subgraph);
    ASSERT_TRUE(built.HasValue()) << built.Error().Message();
    input_names_ = built->input_names;
    output_names_ = built->output_names;
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(built->engine.data(),
                                                  built->engine.size()));
    ASSERT_NE(engine_, nullptr);
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(
        engine_->createEngineInspector());
    layers_ = inspector->getEngineInformation(
        nvinfer1::LayerInformationFormat::kJSON);
    context_.reset(engine_->createExecutionContext());
    ASSERT_NE(context_, nullptr);
    for (const auto& name : input_names_) {
      input_bytes_.push_back(TensorBytes(name));
      void* pointer = nullptr;
      ASSERT_EQ(cudaMalloc(&pointer, input_bytes_.back()), cudaSuccess);
      inputs_.push_back(pointer);
      ASSERT_TRUE(context_->setTensorAddress(name.c_str(), pointer));
    }
    outputs_.assign(output_names_.size(), nullptr);
    for (const auto& [output, input] : aliases) {
      outputs_[output] = inputs_[input];
    }
    for (size_t i = 0; i < output_names_.size(); ++i) {
      output_bytes_.push_back(TensorBytes(output_names_[i]));
      if (outputs_[i] == nullptr) {
        ASSERT_EQ(cudaMalloc(&outputs_[i], output_bytes_[i]), cudaSuccess);
        owned_.push_back(outputs_[i]);
      }
      ASSERT_TRUE(
          context_->setTensorAddress(output_names_[i].c_str(), outputs_[i]));
    }
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  ~Session() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    for (void* pointer : inputs_) cudaFree(pointer);
    for (void* pointer : owned_) cudaFree(pointer);
  }

  void SetInput(int index, const std::vector<uint8_t>& bytes) {
    ASSERT_EQ(bytes.size(), input_bytes_[index]) << input_names_[index];
    ASSERT_EQ(cudaMemcpy(inputs_[index], bytes.data(), bytes.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  void Run() {
    ASSERT_TRUE(context_->enqueueV3(stream_));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  }
  std::vector<uint8_t> Output(int index) {
    std::vector<uint8_t> host(output_bytes_[index]);
    EXPECT_EQ(cudaMemcpy(host.data(), outputs_[index], host.size(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return host;
  }
  std::vector<float> FloatOutput(int index) {
    const auto bytes = Output(index);
    std::vector<float> values(bytes.size() / 4);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
  }
  bool LayersMention(const std::string& text) const {
    return layers_.find(text) != std::string::npos;
  }

 private:
  size_t TensorBytes(const std::string& name) const {
    const auto dims = engine_->getTensorShape(name.c_str());
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) count *= dims.d[i];
    switch (engine_->getTensorDataType(name.c_str())) {
      case nvinfer1::DataType::kINT8:
      case nvinfer1::DataType::kBOOL:
      case nvinfer1::DataType::kUINT8:
        return count;
      case nvinfer1::DataType::kHALF:
      case nvinfer1::DataType::kBF16:
        return count * 2;
      default:
        return count * 4;
    }
  }

  litert::nvidia::TensorRtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<std::string> input_names_, output_names_;
  std::vector<void*> inputs_, outputs_, owned_;
  std::vector<size_t> input_bytes_, output_bytes_;
  std::string layers_;
  cudaStream_t stream_ = nullptr;
};

void ExpectNear(const std::vector<float>& got, const std::vector<double>& want,
                double tolerance) {
  ASSERT_EQ(got.size(), want.size());
  double max_error = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    // The engine's attention output is BF16 inside, FP32 at its boundary.
    ASSERT_NEAR(got[i], want[i], tolerance + std::fabs(want[i]) / 128.0)
        << "element " << i;
    max_error = std::max(max_error, std::fabs(got[i] - want[i]));
  }
  ::testing::Test::RecordProperty("max_error", std::to_string(max_error));
}

void SetGeneralAttention(bool on) {
#ifdef _WIN32
  _putenv_s("LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION", on ? "1" : "");
#else
  if (on) {
    setenv("LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION", "1", 1);
  } else {
    unsetenv("LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION");
  }
#endif
}

class GeneralAttentionGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "no CUDA device";
    }
    SetGeneralAttention(true);
  }
  void TearDown() override { SetGeneralAttention(false); }
};

// ---------------------------------------------------------------------------
// A decode block over packed INT4 caches with a float query.
// ---------------------------------------------------------------------------

constexpr int kDecodeDepth = 128;
constexpr int kDecodeLen = 64;
constexpr float kDecodeKScale = 0.25f;
constexpr float kDecodeVScale = 0.2f;
constexpr float kDecodeScale = 0.0625f;
constexpr float kDecodeSoftcap = 6.0f;

// Inputs: q, k, v, params, k cache, v cache, k start, v start, mask.
// Outputs: attention, k cache', v cache'.
void BuildDecodeBlock(Graph* graph) {
  const auto f32 = kLiteRtElementTypeFloat32;
  const auto i32 = kLiteRtElementTypeInt32;
  const auto i4 = kLiteRtElementTypeInt4;
  auto& q = graph->Input("q", f32, {1, kQueryHeads, 1, kDecodeDepth});
  auto& k = graph->Input("k", f32, {1, kKvHeads, 1, kDecodeDepth});
  auto& v = graph->Input("v", f32, {1, kKvHeads, 1, kDecodeDepth});
  auto& params = graph->Input("param_tensor", i32, {1, 1, 1, 7});
  auto& k_cache =
      graph->Input("kv_cache_k_0", i4, {1, kKvHeads, kDecodeLen, kDecodeDepth},
                   kDecodeKScale);
  auto& v_cache =
      graph->Input("kv_cache_v_0", i4, {1, kKvHeads, kDecodeDepth, kDecodeLen},
                   kDecodeVScale);
  auto& k_start = graph->Input("k_start", i32, {4});
  auto& v_start = graph->Input("v_start", i32, {4});
  auto& mask = graph->Input("mask", f32, {1, 1, 1, kDecodeLen});
  auto& k_out =
      graph->Tensor("kv_cache_k_0_out", i4,
                    {1, kKvHeads, kDecodeLen, kDecodeDepth}, kDecodeKScale);
  auto& v_out =
      graph->Tensor("kv_cache_v_0_out", i4,
                    {1, kKvHeads, kDecodeDepth, kDecodeLen}, kDecodeVScale);
  graph->Composite("odml.cache_update",
                   [](flexbuffers::Builder* attributes) {
                     attributes->Int("cache_size", kDecodeLen);
                     attributes->Int("head_size", kDecodeDepth);
                     attributes->Int("kv_cache_batch_size", kKvHeads);
                     attributes->Float("scale_k", kDecodeKScale);
                     attributes->Float("scale_v", kDecodeVScale);
                   },
                   {&k, &v, &params, &k_cache, &v_cache, &k_start, &v_start},
                   {&k_out, &v_out});
  auto& attention =
      graph->Tensor("attention", f32, {1, kQueryHeads, 1, kDecodeDepth});
  graph->Composite("odml.sdpa_transposed",
                   [](flexbuffers::Builder* attributes) {
                     attributes->Int("k_ts_idx", 2);
                     attributes->Float("scale", kDecodeScale);
                     attributes->Float("softcap", kDecodeSoftcap);
                     attributes->Int("v_ts_idx", 3);
                   },
                   {&q, &k_out, &v_out, &mask, &params}, {&attention});
  graph->Output(attention);
  graph->Output(k_out);
  graph->Output(v_out);
}

TEST_F(GeneralAttentionGraphTest, DecodeFloatQueryFloatMaskInt4Depth128) {
  Graph graph;
  BuildDecodeBlock(&graph);
  Session session;
  ASSERT_NO_FATAL_FAILURE(session.Build(&graph, {{1, 4}, {2, 5}}));
  EXPECT_TRUE(session.LayersMention("cuda_general_attention"));
  EXPECT_TRUE(session.LayersMention("cuda_general_cache_update"));

  Random random(11);
  const size_t count =
      static_cast<size_t>(kKvHeads) * kDecodeLen * kDecodeDepth;
  std::vector<int> k_cache(count), v_cache(count);
  for (auto& value : k_cache) value = random.Int(-8, 7);
  for (auto& value : v_cache) value = random.Int(-8, 7);
  session.SetInput(4, PackCache(k_cache, 4, false));
  session.SetInput(5, PackCache(v_cache, 4, false));
  for (int position = 20; position < 23; ++position) {
    SCOPED_TRACE(::testing::Message() << "position " << position);
    auto q =
        random.Values(static_cast<size_t>(kQueryHeads) * kDecodeDepth, 0.5f);
    auto k = random.Values(static_cast<size_t>(kKvHeads) * kDecodeDepth, 1.8f);
    auto v = random.Values(static_cast<size_t>(kKvHeads) * kDecodeDepth, 1.5f);
    std::vector<float> mask(kDecodeLen);
    for (int j = 0; j < kDecodeLen; ++j) {
      mask[j] = j > position ? -1e9f : (j % 5 == 0 ? -1.5f : 0.0f);
    }
    session.SetInput(0, Bytes(q));
    session.SetInput(1, Bytes(k));
    session.SetInput(2, Bytes(v));
    session.SetInput(
        3, Bytes(std::vector<int32_t>{position, position + 1, 0, 0, 0, 0, 0}));
    session.SetInput(6, Bytes(std::vector<int32_t>{0, 0, position, 0}));
    session.SetInput(7, Bytes(std::vector<int32_t>{0, 0, 0, position}));
    session.SetInput(8, Bytes(mask));
    ASSERT_NO_FATAL_FAILURE(session.Run());
    // The token's keys and values on the INT4 grid.
    for (int h = 0; h < kKvHeads; ++h) {
      for (int d = 0; d < kDecodeDepth; ++d) {
        k_cache[(static_cast<size_t>(h) * kDecodeLen + position) *
                    kDecodeDepth +
                d] =
            ref::QuantizeInt4(Bf16(k[h * kDecodeDepth + d]), kDecodeKScale);
        v_cache[(static_cast<size_t>(h) * kDecodeDepth + d) * kDecodeLen +
                position] =
            ref::QuantizeInt4(Bf16(v[h * kDecodeDepth + d]), kDecodeVScale);
      }
    }
    ASSERT_EQ(UnpackCache(session.Output(1), 4, false), k_cache);
    ASSERT_EQ(UnpackCache(session.Output(2), 4, false), v_cache);
    ref::AttentionCase c;
    c.kv_heads = kKvHeads;
    c.rows = kGroup;
    c.depth = kDecodeDepth;
    c.mask_rows = 1;
    c.bool_mask = false;
    c.scale = kDecodeScale;
    c.softcap = kDecodeSoftcap;
    for (auto& value : q) value = Bf16(value);
    const auto want =
        ref::Attention(c,
                       HostCaches(k_cache, v_cache, kDecodeLen, kDecodeDepth, 4,
                                  kDecodeKScale, kDecodeVScale),
                       q, {}, {}, mask);
    ExpectNear(session.FloatOutput(0), want, 2e-3);
  }
}

// ---------------------------------------------------------------------------
// A prefill block over UINT8 caches with an INT8 query.
// ---------------------------------------------------------------------------

constexpr int kPrefillDepth = 64;
constexpr int kPrefillLen = 48;
constexpr int kPrefillTokens = 4;
constexpr float kPrefillQScale = 0.01f;
constexpr float kPrefillRowScale = 0.013f;  // the keys and values the step adds
constexpr float kPrefillKScale = 0.02f;
constexpr float kPrefillVScale = 0.03f;

// Inputs: q, k, v, params, k cache, v cache, k start, v start, mask.
// Outputs: attention, k cache', v cache'.
void BuildPrefillBlock(Graph* graph) {
  const auto f32 = kLiteRtElementTypeFloat32;
  const auto i32 = kLiteRtElementTypeInt32;
  const auto i8 = kLiteRtElementTypeInt8;
  const auto u8 = kLiteRtElementTypeUInt8;
  const int n = kPrefillTokens;
  auto& q = graph->Input("q", f32, {1, kQueryHeads, n, kPrefillDepth});
  auto& k = graph->Input("k", f32, {1, kKvHeads, n, kPrefillDepth});
  auto& v = graph->Input("v", f32, {1, kKvHeads, n, kPrefillDepth});
  auto& params = graph->Input("param_tensor", i32, {1, 1, 1, 7});
  auto& k_cache = graph->Input("kv_cache_k_0", u8,
                               {1, kKvHeads, kPrefillLen, kPrefillDepth},
                               kPrefillKScale, 128);
  auto& v_cache = graph->Input("kv_cache_v_0", u8,
                               {1, kKvHeads, kPrefillDepth, kPrefillLen},
                               kPrefillVScale, 128);
  auto& k_start = graph->Input("k_start", i32, {4});
  auto& v_start = graph->Input("v_start", i32, {4});
  auto& mask =
      graph->Input("mask", kLiteRtElementTypeBool, {1, 1, n, kPrefillLen});
  auto& q8 = graph->Quantize(
      q, "q_quantized", 8, {1, kQueryHeads, n, kPrefillDepth}, kPrefillQScale);
  auto& k8 = graph->Quantize(k, "k_quantized", 8,
                             {1, kKvHeads, n, kPrefillDepth}, kPrefillRowScale);
  auto& v8 = graph->Quantize(v, "v_quantized", 8,
                             {1, kKvHeads, n, kPrefillDepth}, kPrefillRowScale);
  static_cast<void>(i8);
  auto& k_out = graph->Tensor("kv_cache_k_0_out", u8,
                              {1, kKvHeads, kPrefillLen, kPrefillDepth},
                              kPrefillKScale, 128);
  auto& v_out = graph->Tensor("kv_cache_v_0_out", u8,
                              {1, kKvHeads, kPrefillDepth, kPrefillLen},
                              kPrefillVScale, 128);
  graph->Composite("odml.cache_update",
                   [](flexbuffers::Builder* attributes) {
                     attributes->Int("cache_size", kPrefillLen);
                     attributes->Int("head_size", kPrefillDepth);
                     attributes->Int("kv_cache_batch_size", kKvHeads);
                   },
                   {&k8, &v8, &params, &k_cache, &v_cache, &k_start, &v_start},
                   {&k_out, &v_out});
  auto& attention =
      graph->Tensor("attention", f32, {1, kQueryHeads, n, kPrefillDepth});
  graph->Composite("odml.sdpa_transposed",
                   [](flexbuffers::Builder* attributes) {
                     attributes->Int("k_ts_idx", 2);
                     attributes->Int("v_ts_idx", 3);
                   },
                   {&q8, &k_out, &v_out, &mask, &params}, {&attention});
  graph->Output(attention);
  graph->Output(k_out);
  graph->Output(v_out);
}

TEST_F(GeneralAttentionGraphTest, PrefillInt8QueryUint8CachesOtherRowScale) {
  Graph graph;
  BuildPrefillBlock(&graph);
  Session session;
  ASSERT_NO_FATAL_FAILURE(session.Build(&graph, {{1, 4}, {2, 5}}));
  EXPECT_TRUE(session.LayersMention("cuda_general_attention"));
  EXPECT_TRUE(session.LayersMention("cuda_general_cache_update"));

  Random random(23);
  const int n = kPrefillTokens;
  const size_t count =
      static_cast<size_t>(kKvHeads) * kPrefillLen * kPrefillDepth;
  std::vector<int> k_cache(count), v_cache(count);
  for (auto& value : k_cache) value = random.Int(-128, 127);
  for (auto& value : v_cache) value = random.Int(-128, 127);
  session.SetInput(4, PackCache(k_cache, 8, true));
  session.SetInput(5, PackCache(v_cache, 8, true));
  const int start = 10;
  auto q =
      random.Values(static_cast<size_t>(kQueryHeads) * n * kPrefillDepth, 0.3f);
  auto k =
      random.Values(static_cast<size_t>(kKvHeads) * n * kPrefillDepth, 1.5f);
  auto v =
      random.Values(static_cast<size_t>(kKvHeads) * n * kPrefillDepth, 1.5f);
  // A causal mask; the last row is padding and sees nothing.
  std::vector<uint8_t> mask(static_cast<size_t>(n) * kPrefillLen, 0);
  std::vector<float> mask_values(mask.size(), 0.0f);
  for (int t = 0; t + 1 < n; ++t) {
    for (int j = 0; j <= start + t; ++j) {
      mask[t * kPrefillLen + j] = 1;
      mask_values[t * kPrefillLen + j] = 1.0f;
    }
  }
  session.SetInput(0, Bytes(q));
  session.SetInput(1, Bytes(k));
  session.SetInput(2, Bytes(v));
  session.SetInput(
      3, Bytes(std::vector<int32_t>{start, start + n, 0, 0, 0, 0, 0}));
  session.SetInput(6, Bytes(std::vector<int32_t>{0, 0, start, 0}));
  session.SetInput(7, Bytes(std::vector<int32_t>{0, 0, 0, start}));
  session.SetInput(8, mask);
  ASSERT_NO_FATAL_FAILURE(session.Run());
  // The rows on their own INT8 grid, then on the caches'.
  auto twice = [](float x, float cache_scale) {
    const float row =
        ref::QuantizeInt8(Bf16(x), kPrefillRowScale) * kPrefillRowScale;
    return ref::QuantizeInt8(row, cache_scale);
  };
  for (int h = 0; h < kKvHeads; ++h) {
    for (int t = 0; t < n; ++t) {
      for (int d = 0; d < kPrefillDepth; ++d) {
        const size_t i = (static_cast<size_t>(h) * n + t) * kPrefillDepth + d;
        k_cache[(static_cast<size_t>(h) * kPrefillLen + start + t) *
                    kPrefillDepth +
                d] = twice(k[i], kPrefillKScale);
        v_cache[(static_cast<size_t>(h) * kPrefillDepth + d) * kPrefillLen +
                start + t] = twice(v[i], kPrefillVScale);
      }
    }
  }
  ASSERT_EQ(UnpackCache(session.Output(1), 8, true), k_cache);
  ASSERT_EQ(UnpackCache(session.Output(2), 8, true), v_cache);
  ref::AttentionCase c;
  c.kv_heads = kKvHeads;
  c.rows = kGroup * n;
  c.depth = kPrefillDepth;
  c.mask_rows = n;
  c.bool_mask = true;
  c.q_int8_scale = kPrefillQScale;
  for (auto& value : q) value = Bf16(value);
  const auto want =
      ref::Attention(c,
                     HostCaches(k_cache, v_cache, kPrefillLen, kPrefillDepth, 8,
                                kPrefillKScale, kPrefillVScale),
                     q, {}, {}, mask_values);
  ExpectNear(session.FloatOutput(0), want, 3e-3);
}

// ---------------------------------------------------------------------------
// odml.fused_sdpa_cache_update over packed INT4 ring caches.
// ---------------------------------------------------------------------------

constexpr int kFusedDepth = 64;
constexpr int kRing = 32;
constexpr int kChunk = 2;
constexpr int kFusedRows = kGroup * kChunk;
constexpr float kFusedKScale = 0.3f;
constexpr float kFusedVScale = 0.25f;
constexpr float kFusedSoftcap = 8.0f;

// Inputs: q, k cache, v cache, k, v, mask, params. Outputs: attention and,
// with `update`, the caches.
void BuildFusedBlock(Graph* graph, bool update) {
  const auto f32 = kLiteRtElementTypeFloat32;
  const auto i4 = kLiteRtElementTypeInt4;
  auto& q = graph->Input("q", f32, {1, kKvHeads, kFusedRows, kFusedDepth});
  auto& k_cache = graph->Input("kv_cache_k_0", i4,
                               {1, kKvHeads, kRing, kFusedDepth}, kFusedKScale);
  auto& v_cache = graph->Input("kv_cache_v_0", i4,
                               {1, kKvHeads, kFusedDepth, kRing}, kFusedVScale);
  auto& k = graph->Input("k", f32, {1, kKvHeads, kChunk, kFusedDepth});
  auto& v = graph->Input("v", f32, {1, kKvHeads, kFusedDepth, kChunk});
  auto& mask = graph->Input("mask", f32, {1, 1, kChunk, kRing + kChunk});
  auto& params =
      graph->Input("param_tensor", kLiteRtElementTypeInt32, {1, 1, 1, 7});
  auto& attention =
      graph->Tensor("attention", f32, {1, kKvHeads, kFusedRows, kFusedDepth});
  std::vector<LiteRtTensorT*> outputs = {&attention};
  if (update) {
    outputs.push_back(&graph->Tensor("kv_cache_k_0_out", i4,
                                     {1, kKvHeads, kRing, kFusedDepth},
                                     kFusedKScale));
    outputs.push_back(&graph->Tensor("kv_cache_v_0_out", i4,
                                     {1, kKvHeads, kFusedDepth, kRing},
                                     kFusedVScale));
  }
  graph->Composite(
      "odml.fused_sdpa_cache_update",
      [](flexbuffers::Builder* attributes) {
        attributes->Float("softcap", kFusedSoftcap);
      },
      {&q, &k_cache, &v_cache, &k, &v, &mask, &params}, outputs);
  for (auto* output : outputs) graph->Output(*output);
}

void RunFused(bool update) {
  Graph graph;
  BuildFusedBlock(&graph, update);
  Session session;
  if (update) {
    ASSERT_NO_FATAL_FAILURE(session.Build(&graph, {{1, 1}, {2, 2}}));
  } else {
    ASSERT_NO_FATAL_FAILURE(session.Build(&graph, {}));
  }
  EXPECT_TRUE(session.LayersMention("cuda_general_fused_attention"));

  Random random(37);
  const size_t count = static_cast<size_t>(kKvHeads) * kRing * kFusedDepth;
  std::vector<int> k_cache(count), v_cache(count);
  for (auto& value : k_cache) value = random.Int(-8, 7);
  for (auto& value : v_cache) value = random.Int(-8, 7);
  session.SetInput(1, PackCache(k_cache, 4, false));
  session.SetInput(2, PackCache(v_cache, 4, false));
  // Two steps: the second chunk wraps around the ring's end.
  for (int p0 : {30, 32}) {
    SCOPED_TRACE(::testing::Message() << "p0 " << p0);
    auto q = random.Values(
        static_cast<size_t>(kKvHeads) * kFusedRows * kFusedDepth, 0.4f);
    auto k = random.Values(static_cast<size_t>(kKvHeads) * kChunk * kFusedDepth,
                           2.0f);
    // The values as the composite takes them, [H, D, T].
    auto v_columns = random.Values(
        static_cast<size_t>(kKvHeads) * kFusedDepth * kChunk, 1.8f);
    std::vector<float> mask(static_cast<size_t>(kChunk) * (kRing + kChunk));
    for (int t = 0; t < kChunk; ++t) {
      for (int j = 0; j < kRing + kChunk; ++j) {
        const bool visible =
            j < kRing ? (j < p0 || j % 3 != 0) : j - kRing <= t;
        mask[t * (kRing + kChunk) + j] =
            visible ? (j % 7 == 0 ? -0.75f : 0.0f) : -1e9f;
      }
    }
    session.SetInput(0, Bytes(q));
    session.SetInput(3, Bytes(k));
    session.SetInput(4, Bytes(v_columns));
    session.SetInput(5, Bytes(mask));
    session.SetInput(
        6, Bytes(std::vector<int32_t>{p0, p0 + kChunk, 0, 0, 0, 0, 0}));
    ASSERT_NO_FATAL_FAILURE(session.Run());

    for (auto& value : q) value = Bf16(value);
    for (auto& value : k) value = Bf16(value);
    std::vector<float> v_rows(v_columns.size());
    for (int h = 0; h < kKvHeads; ++h) {
      for (int d = 0; d < kFusedDepth; ++d) {
        for (int t = 0; t < kChunk; ++t) {
          v_rows[(static_cast<size_t>(h) * kChunk + t) * kFusedDepth + d] =
              Bf16(v_columns[(static_cast<size_t>(h) * kFusedDepth + d) *
                                 kChunk +
                             t]);
        }
      }
    }
    ref::AttentionCase c;
    c.kv_heads = kKvHeads;
    c.rows = kFusedRows;
    c.depth = kFusedDepth;
    c.new_len = kChunk;
    c.mask_rows = kChunk;
    c.bool_mask = false;
    c.softcap = kFusedSoftcap;
    const auto want =
        ref::Attention(c,
                       HostCaches(k_cache, v_cache, kRing, kFusedDepth, 4,
                                  kFusedKScale, kFusedVScale),
                       q, k, v_rows, mask);
    ExpectNear(session.FloatOutput(0), want, 3e-3);
    if (!update) continue;
    for (int t = 0; t < kChunk; ++t) {
      const int slot = (p0 + t) % kRing;
      for (int h = 0; h < kKvHeads; ++h) {
        for (int d = 0; d < kFusedDepth; ++d) {
          const size_t i =
              (static_cast<size_t>(h) * kChunk + t) * kFusedDepth + d;
          k_cache[(static_cast<size_t>(h) * kRing + slot) * kFusedDepth + d] =
              ref::QuantizeInt4(k[i], kFusedKScale);
          v_cache[(static_cast<size_t>(h) * kFusedDepth + d) * kRing + slot] =
              ref::QuantizeInt4(v_rows[i], kFusedVScale);
        }
      }
    }
    ASSERT_EQ(UnpackCache(session.Output(1), 4, false), k_cache);
    ASSERT_EQ(UnpackCache(session.Output(2), 4, false), v_cache);
  }
}

TEST_F(GeneralAttentionGraphTest, FusedInt4FloatChunkAcrossTheRingsEnd) {
  RunFused(/*update=*/true);
}

TEST_F(GeneralAttentionGraphTest, FusedOneOutputLeavesTheCaches) {
  RunFused(/*update=*/false);
}

TEST_F(GeneralAttentionGraphTest, NoneWithoutTheSwitch) {
  SetGeneralAttention(false);
  Graph decode, prefill, fused;
  BuildDecodeBlock(&decode);
  BuildPrefillBlock(&prefill);
  BuildFusedBlock(&fused, /*update=*/true);
  for (Graph* graph : {&decode, &prefill, &fused}) {
    const litert::compiler::Subgraph subgraph(LrtGetCompilerContext(),
                                              &graph->Get());
    for (const auto& op : subgraph.Ops()) {
      if (op.Code() == kLiteRtOpCodeShloComposite) {
        EXPECT_FALSE(litert::nvidia::IsTensorRtOpSupported(op));
      }
    }
  }
}

}  // namespace
