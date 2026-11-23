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

#include "litert/vendors/nvidia/compiler/slot_plugins.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/compiler/decode_attention_plugin.h"
#include "litert/vendors/nvidia/compiler/int8_cache_write_plugin.h"
#include "litert/vendors/nvidia/compiler/qkv_norm_rope_plugin.h"
#include "litert/vendors/nvidia/compiler/tensorrt_rtx_plugin_compat.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/slot_test_util.h"

namespace litert::nvidia {
namespace {

using slot_test::Bytes;
using slot_test::Device;
using slot_test::Download;
using slot_test::Upload;

class TestLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[SlotPluginsTest][%d] %s\n",
                   static_cast<int>(severity), message);
    }
  }
};

// TensorRT keeps the first logger it is given.
TestLogger& Logger() {
  static auto* const logger = new TestLogger();
  return *logger;
}

struct Tensor {
  nvinfer1::Dims dims;
  nvinfer1::DataType type;
};

nvinfer1::Dims Dims1(int64_t a) {
  nvinfer1::Dims dims{};
  dims.nbDims = 1;
  dims.d[0] = a;
  return dims;
}

size_t BytesOf(const Tensor& tensor) {
  size_t count = 1;
  for (int i = 0; i < tensor.dims.nbDims; ++i) count *= tensor.dims.d[i];
  switch (tensor.type) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return count * 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
      return count * 2;
    default:
      return count;
  }
}

nvinfer1::PluginTensorDesc DescOf(const Tensor& tensor) {
  nvinfer1::PluginTensorDesc desc{};
  desc.dims = tensor.dims;
  desc.type = tensor.type;
  desc.format = nvinfer1::TensorFormat::kLINEAR;
  return desc;
}

nvinfer1::IPluginV3OneRuntime* Runtime(nvinfer1::IPluginV3* plugin) {
  return static_cast<nvinfer1::IPluginV3OneRuntime*>(
      plugin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
}

// A plugin layer of a network over inputs of the given shapes and types.
struct Layer {
  std::unique_ptr<nvinfer1::IBuilder> builder;
  std::unique_ptr<nvinfer1::INetworkDefinition> network;
  std::unique_ptr<nvinfer1::IPluginV3> plugin;
  nvinfer1::IPluginV3Layer* layer = nullptr;
  std::vector<Tensor> inputs;
};

std::unique_ptr<Layer> MakeLayer(nvinfer1::IPluginV3* plugin,
                                 const std::vector<Tensor>& inputs) {
  auto result = std::make_unique<Layer>();
  result->plugin.reset(plugin);
  if (plugin == nullptr) return nullptr;
  result->inputs = inputs;
  result->builder.reset(nvinfer1::createInferBuilder(Logger()));
  if (!result->builder) return nullptr;
  result->network.reset(result->builder->createNetworkV2(0));
  if (!result->network) return nullptr;
  std::vector<nvinfer1::ITensor*> tensors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    tensors.push_back(result->network->addInput(
        ("input" + std::to_string(i)).c_str(), inputs[i].type, inputs[i].dims));
    if (tensors.back() == nullptr) return nullptr;
  }
  result->layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
      *result->network, tensors.data(), static_cast<int32_t>(tensors.size()),
      *result->plugin);
  return result->layer != nullptr ? std::move(result) : nullptr;
}

// The twin as an engine holds it: made again from its serialized fields.
std::unique_ptr<nvinfer1::IPluginV3> Restored(nvinfer1::IPluginV3* twin) {
  auto* core = static_cast<nvinfer1::IPluginV3OneCore*>(
      twin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kCORE));
  auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(
      getPluginRegistry()->getCreator(core->getPluginName(),
                                      core->getPluginVersion(),
                                      core->getPluginNamespace()));
  if (creator == nullptr) return nullptr;
  return std::unique_ptr<nvinfer1::IPluginV3>(
      creator->createPlugin("twin", Runtime(twin)->getFieldsToSerialize(),
                            nvinfer1::TensorRTPhase::kRUNTIME));
}

// The descriptors of a twin over `slots` slots of a layer.
std::vector<nvinfer1::PluginTensorDesc> Repeated(
    const std::vector<Tensor>& tensors, int slots) {
  std::vector<nvinfer1::PluginTensorDesc> descs;
  for (int s = 0; s < slots; ++s) {
    for (const Tensor& tensor : tensors) descs.push_back(DescOf(tensor));
  }
  return descs;
}

// The outputs of `plugin` as the one layer of an engine: built (TensorRT
// asks the plugin for its formats, shapes and workspace), deserialized (which
// makes the plugin again from its serialized fields) and run on `input_bytes`.
// Output o is bound to the buffer of input in_place[o] where that is not -1,
// as LiteRT-LM binds a cache. Empty after a failure.
std::vector<std::vector<uint8_t>> RunInEngine(
    nvinfer1::IPluginV3& plugin, const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& outputs,
    const std::vector<std::vector<uint8_t>>& input_bytes,
    const std::vector<int>& in_place) {
  std::unique_ptr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(Logger()));
  std::unique_ptr<nvinfer1::INetworkDefinition> network(
      builder != nullptr ? builder->createNetworkV2(0) : nullptr);
  if (network == nullptr) {
    ADD_FAILURE() << "no network";
    return {};
  }
  std::vector<nvinfer1::ITensor*> tensors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    tensors.push_back(network->addInput(("in" + std::to_string(i)).c_str(),
                                        inputs[i].type, inputs[i].dims));
  }
  nvinfer1::IPluginV3Layer* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
      *network, tensors.data(), static_cast<int32_t>(tensors.size()), plugin);
  if (layer == nullptr ||
      layer->getNbOutputs() != static_cast<int32_t>(outputs.size())) {
    ADD_FAILURE() << "no plugin layer with " << outputs.size() << " outputs";
    return {};
  }
  for (size_t o = 0; o < outputs.size(); ++o) {
    layer->getOutput(static_cast<int32_t>(o))
        ->setName(("out" + std::to_string(o)).c_str());
    network->markOutput(*layer->getOutput(static_cast<int32_t>(o)));
  }
  std::unique_ptr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig());
  std::unique_ptr<nvinfer1::IHostMemory> plan(
      config != nullptr ? builder->buildSerializedNetwork(*network, *config)
                        : nullptr);
  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(Logger()));
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      plan != nullptr && runtime != nullptr
          ? runtime->deserializeCudaEngine(plan->data(), plan->size())
          : nullptr);
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine != nullptr ? engine->createExecutionContext() : nullptr);
  if (context == nullptr) {
    ADD_FAILURE() << "no engine";
    return {};
  }
  std::vector<Device> buffers;
  bool ok = true;
  for (size_t i = 0; i < inputs.size(); ++i) {
    buffers.push_back(Upload(input_bytes[i]));
    ok = ok && buffers.back() != nullptr &&
         context->setTensorAddress(("in" + std::to_string(i)).c_str(),
                                   buffers.back().get());
  }
  std::vector<void*> results;
  for (size_t o = 0; o < outputs.size(); ++o) {
    if (in_place[o] >= 0) {
      results.push_back(buffers[in_place[o]].get());
    } else {
      buffers.push_back(Upload(nullptr, BytesOf(outputs[o])));
      ok = ok && buffers.back() != nullptr;
      results.push_back(buffers.back().get());
    }
    ok = ok && context->setTensorAddress(("out" + std::to_string(o)).c_str(),
                                         results.back());
  }
  cudaStream_t stream = nullptr;
  ok = ok && cudaStreamCreate(&stream) == cudaSuccess &&
       context->enqueueV3(stream) &&
       cudaStreamSynchronize(stream) == cudaSuccess;
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (!ok) {
    ADD_FAILURE() << "the engine did not run";
    return {};
  }
  std::vector<std::vector<uint8_t>> bytes;
  for (size_t o = 0; o < outputs.size(); ++o) {
    bytes.push_back(Download(results[o], BytesOf(outputs[o])));
  }
  return bytes;
}

// `tensors` once per slot.
std::vector<Tensor> RepeatedTensors(const std::vector<Tensor>& tensors,
                                    int slots) {
  std::vector<Tensor> repeated;
  for (int s = 0; s < slots; ++s) {
    repeated.insert(repeated.end(), tensors.begin(), tensors.end());
  }
  return repeated;
}

// The twin's outputs for `slots` slots against the single-sequence plugin's
// for each slot alone, through the plugins' own enqueue.
void CheckAttentionTwin(int slots, int heads, int rows, int depth,
                        int cache_len, int int8_cache, int window) {
  SCOPED_TRACE(::testing::Message()
               << slots << " slots, heads " << heads << ", rows " << rows
               << ", depth " << depth << ", " << cache_len << " keys, int8 "
               << int8_cache << ", window " << window);
  const LiteRtNvidiaInt8CacheParams int8 = {0.0125f, -3, 0.02f, 5,
                                            int8_cache == 2};
  const nvinfer1::DataType cache_type =
      int8_cache != 0 ? nvinfer1::DataType::kINT8 : nvinfer1::DataType::kHALF;
  std::vector<Tensor> inputs = {
      {nvinfer1::Dims4{1, heads, rows, depth}, nvinfer1::DataType::kBF16},
      {nvinfer1::Dims4{1, heads, cache_len, depth}, cache_type},
      {int8_cache == 2 ? nvinfer1::Dims4{1, heads, depth, cache_len}
                       : nvinfer1::Dims4{1, heads, cache_len, depth},
       cache_type},
      {nvinfer1::Dims4{1, 1, 1, cache_len}, nvinfer1::DataType::kBOOL}};
  if (window != 0) inputs.push_back({Dims1(1), nvinfer1::DataType::kINT32});
  const std::vector<Tensor> outputs = {inputs[0]};
  auto layer = MakeLayer(
      CreateDecodeAttentionPlugin(-10000.0f, /*new_keys=*/false,
                                  int8_cache != 0 ? &int8 : nullptr, window),
      inputs);
  ASSERT_NE(layer, nullptr);
  std::unique_ptr<nvinfer1::IPluginV3> twin(
      CreateSlotsTwin(*layer->layer, slots));
  ASSERT_NE(twin, nullptr);
  const auto restored = Restored(twin.get());
  ASSERT_NE(restored, nullptr);

  const int n = static_cast<int>(inputs.size());
  auto single_in = Repeated(inputs, 1), single_out = Repeated(outputs, 1);
  auto twin_in = Repeated(inputs, slots), twin_out = Repeated(outputs, slots);
  auto* single_runtime = Runtime(layer->plugin.get());
  auto* twin_runtime = Runtime(restored.get());
  ASSERT_EQ(
      single_runtime->onShapeChange(single_in.data(), n, single_out.data(), 1),
      0);
  ASSERT_EQ(twin_runtime->onShapeChange(twin_in.data(), slots * n,
                                        twin_out.data(), slots),
            0);
  const int read = window != 0 ? window : cache_len;
  const size_t single_ws =
      LiteRtNvidiaDecodeAttentionWorkspaceBytes(heads, rows, read, depth);
  auto* twin_build = static_cast<nvinfer1::IPluginV3OneBuild*>(
      restored->getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
  std::vector<nvinfer1::DynamicPluginTensorDesc> dynamic(slots * n),
      dynamic_out(slots);
  for (int i = 0; i < slots * n; ++i) {
    dynamic[i].desc = twin_in[i];
    dynamic[i].min = dynamic[i].opt = dynamic[i].max = twin_in[i].dims;
  }
  for (int s = 0; s < slots; ++s) dynamic_out[s].desc = twin_out[s];
  ASSERT_EQ(twin_build->getWorkspaceSize(dynamic.data(), slots * n,
                                         dynamic_out.data(), slots),
            slots * single_ws);

  const size_t q_bytes = BytesOf(inputs[0]);
  std::vector<Device> buffers;
  std::vector<const void*> twin_inputs;
  std::vector<void*> twin_outputs;
  std::vector<std::vector<uint8_t>> expected(slots);
  std::vector<std::vector<uint8_t>> twin_bytes;
  Device single_workspace = Upload(nullptr, single_ws);
  Device twin_workspace = Upload(nullptr, slots * single_ws);
  ASSERT_TRUE(single_workspace && twin_workspace);
  for (int s = 0; s < slots; ++s) {
    // Slot s sees the first keys up to its own position.
    const int visible = cache_len * (s + 1) / slots;
    std::vector<uint8_t> mask(cache_len, 0);
    std::fill_n(mask.begin(), visible, 1);
    const int32_t start =
        window != 0 ? std::max(0, visible - window) / 16 * 16 : 0;
    std::vector<uint8_t> start_bytes(sizeof(start));
    std::memcpy(start_bytes.data(), &start, sizeof(start));
    const std::vector<std::vector<uint8_t>> hosts = {
        Bytes(q_bytes, 10 * s + 1, true),
        Bytes(BytesOf(inputs[1]), 10 * s + 2, int8_cache == 0),
        Bytes(BytesOf(inputs[2]), 10 * s + 3, int8_cache == 0), mask,
        start_bytes};
    std::vector<const void*> slot_inputs;
    for (int i = 0; i < n; ++i) {
      buffers.push_back(Upload(hosts[i]));
      ASSERT_NE(buffers.back(), nullptr);
      slot_inputs.push_back(buffers.back().get());
      twin_bytes.push_back(hosts[i]);
    }
    buffers.push_back(Upload(nullptr, q_bytes));
    void* single_output = buffers.back().get();
    buffers.push_back(Upload(nullptr, q_bytes));
    twin_outputs.push_back(buffers.back().get());
    twin_inputs.insert(twin_inputs.end(), slot_inputs.begin(),
                       slot_inputs.end());
    ASSERT_EQ(single_runtime->enqueue(single_in.data(), single_out.data(),
                                      slot_inputs.data(), &single_output,
                                      single_workspace.get(), nullptr),
              0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expected[s] = Download(single_output, q_bytes);
  }
  ASSERT_EQ(
      twin_runtime->enqueue(twin_in.data(), twin_out.data(), twin_inputs.data(),
                            twin_outputs.data(), twin_workspace.get(), nullptr),
      0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (int s = 0; s < slots; ++s) {
    const std::vector<uint8_t> actual = Download(twin_outputs[s], q_bytes);
    EXPECT_NE(std::count(actual.begin(), actual.end(), 0),
              static_cast<ptrdiff_t>(actual.size()));
    EXPECT_EQ(actual, expected[s]) << "slot " << s;
  }
  if (slots != 3) return;
  // And as a layer of an engine.
  const std::vector<std::vector<uint8_t>> engine = RunInEngine(
      *twin, RepeatedTensors(inputs, slots), RepeatedTensors(outputs, slots),
      twin_bytes, std::vector<int>(slots, -1));
  ASSERT_EQ(engine.size(), static_cast<size_t>(slots));
  for (int s = 0; s < slots; ++s) {
    EXPECT_EQ(engine[s], expected[s]) << "slot " << s << " in an engine";
  }
}

TEST(SlotPluginsTest, DecodeAttentionTwinMatchesTheLayerOnEachSlot) {
  // Gemma 4 26B-A4B's local and global layers, E4B's local layer through a
  // window and its global layer.
  for (const int slots : {2, 3, 8}) {
    CheckAttentionTwin(slots, 8, 2, 256, 1024, /*int8_cache=*/0, /*window=*/0);
    CheckAttentionTwin(slots, 2, 8, 512, 2048, 0, 0);
    CheckAttentionTwin(slots, 2, 4, 256, 2048, 2, 512);
    CheckAttentionTwin(slots, 1, 8, 512, 2048, 2, 0);
    CheckAttentionTwin(slots, 2, 4, 256, 1024, 1, 0);
  }
}

std::vector<Tensor> AttentionInputs(int heads, int rows, int depth,
                                    int cache_len) {
  return {
      {nvinfer1::Dims4{1, heads, rows, depth}, nvinfer1::DataType::kBF16},
      {nvinfer1::Dims4{1, heads, cache_len, depth}, nvinfer1::DataType::kHALF},
      {nvinfer1::Dims4{1, heads, cache_len, depth}, nvinfer1::DataType::kHALF},
      {nvinfer1::Dims4{1, 1, 1, cache_len}, nvinfer1::DataType::kBOOL}};
}

TEST(SlotPluginsTest, DeclinesLayersWithoutATwin) {
  // One slot is the layer itself; 16 rows over a cache of depth 512 run on
  // the tensor-core kernel; new keys and row tokens are prefill and verify
  // forms.
  auto decode = MakeLayer(CreateDecodeAttentionPlugin(-10000.0f),
                          AttentionInputs(8, 2, 256, 1024));
  ASSERT_NE(decode, nullptr);
  EXPECT_EQ(CreateSlotsTwin(*decode->layer, 1), nullptr);
  EXPECT_EQ(CreateSlotsTwin(*decode->layer, 9), nullptr);
  EXPECT_EQ(CreateSlotRowsPlugin(*decode->plugin), nullptr);
  auto streaming = MakeLayer(CreateDecodeAttentionPlugin(-10000.0f),
                             AttentionInputs(1, 16, 512, 2048));
  ASSERT_NE(streaming, nullptr);
  EXPECT_EQ(CreateSlotsTwin(*streaming->layer, 2), nullptr);
  auto tokens =
      MakeLayer(CreateDecodeAttentionPlugin(-10000.0f, false, nullptr, 0,
                                            /*row_tokens=*/4),
                AttentionInputs(2, 8, 512, 2048));
  ASSERT_NE(tokens, nullptr);
  EXPECT_EQ(CreateSlotsTwin(*tokens->layer, 2), nullptr);
  EXPECT_EQ(CreateCacheWriteSlotsPlugin(0, false), nullptr);
  EXPECT_EQ(CreateCacheWriteSlotsPlugin(17, false), nullptr);
}

// A layer's plugin on each slot alone against its twin on all of them,
// through the plugins' enqueue: `inputs` and `outputs` are the layer's
// tensors, `per_slot[s][i]` slot s's bytes of input i. An output listed in
// `in_place` (output o is input in_place[o], or -1) is bound to that input's
// buffer in both runs, as LiteRT-LM binds a cache.
void CheckTwin(Layer& layer, int slots, const std::vector<Tensor>& outputs,
               const std::vector<std::vector<std::vector<uint8_t>>>& per_slot,
               const std::vector<int>& in_place) {
  std::unique_ptr<nvinfer1::IPluginV3> twin(
      CreateSlotsTwin(*layer.layer, slots));
  ASSERT_NE(twin, nullptr);
  const auto restored = Restored(twin.get());
  ASSERT_NE(restored, nullptr);
  const int n = static_cast<int>(layer.inputs.size());
  const int m = static_cast<int>(outputs.size());
  auto single_in = Repeated(layer.inputs, 1), single_out = Repeated(outputs, 1);
  auto twin_in = Repeated(layer.inputs, slots),
       twin_out = Repeated(outputs, slots);
  auto* single_runtime = Runtime(layer.plugin.get());
  auto* twin_runtime = Runtime(restored.get());
  ASSERT_EQ(
      single_runtime->onShapeChange(single_in.data(), n, single_out.data(), m),
      0);
  ASSERT_EQ(twin_runtime->onShapeChange(twin_in.data(), slots * n,
                                        twin_out.data(), slots * m),
            0);
  std::vector<Device> buffers;
  const auto keep = [&](Device device) {
    EXPECT_NE(device, nullptr);
    buffers.push_back(std::move(device));
    return buffers.back().get();
  };
  std::vector<const void*> twin_inputs;
  std::vector<void*> twin_outputs;
  std::vector<void*> expected;
  for (int s = 0; s < slots; ++s) {
    // Two sets of the slot's inputs: an in-place output changes its input.
    std::vector<const void*> single_inputs;
    std::vector<void*> single_outputs;
    const size_t first = twin_inputs.size();
    for (int i = 0; i < n; ++i) {
      single_inputs.push_back(keep(Upload(per_slot[s][i])));
      twin_inputs.push_back(keep(Upload(per_slot[s][i])));
    }
    for (int o = 0; o < m; ++o) {
      if (in_place[o] >= 0) {
        single_outputs.push_back(const_cast<void*>(single_inputs[in_place[o]]));
        twin_outputs.push_back(
            const_cast<void*>(twin_inputs[first + in_place[o]]));
      } else {
        single_outputs.push_back(keep(Upload(nullptr, BytesOf(outputs[o]))));
        twin_outputs.push_back(keep(Upload(nullptr, BytesOf(outputs[o]))));
      }
    }
    ASSERT_EQ(single_runtime->enqueue(single_in.data(), single_out.data(),
                                      single_inputs.data(),
                                      single_outputs.data(), nullptr, nullptr),
              0);
    expected.insert(expected.end(), single_outputs.begin(),
                    single_outputs.end());
  }
  ASSERT_EQ(
      twin_runtime->enqueue(twin_in.data(), twin_out.data(), twin_inputs.data(),
                            twin_outputs.data(), nullptr, nullptr),
      0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (int s = 0; s < slots; ++s) {
    for (int o = 0; o < m; ++o) {
      const size_t bytes = BytesOf(outputs[o]);
      const std::vector<uint8_t> actual =
          Download(twin_outputs[s * m + o], bytes);
      EXPECT_NE(std::count(actual.begin(), actual.end(), 0),
                static_cast<ptrdiff_t>(bytes));
      EXPECT_EQ(actual, Download(expected[s * m + o], bytes))
          << "slot " << s << " output " << o;
      if (in_place[o] >= 0) {
        EXPECT_NE(actual, per_slot[s][in_place[o]]) << "slot " << s;
      }
    }
  }
  if (slots != 3) return;
  // And as a layer of an engine, an in-place output bound to its input.
  std::vector<std::vector<uint8_t>> twin_bytes;
  std::vector<int> twin_in_place;
  for (int s = 0; s < slots; ++s) {
    twin_bytes.insert(twin_bytes.end(), per_slot[s].begin(), per_slot[s].end());
    for (int o = 0; o < m; ++o) {
      twin_in_place.push_back(in_place[o] >= 0 ? s * n + in_place[o] : -1);
    }
  }
  const std::vector<std::vector<uint8_t>> engine =
      RunInEngine(*twin, RepeatedTensors(layer.inputs, slots),
                  RepeatedTensors(outputs, slots), twin_bytes, twin_in_place);
  ASSERT_EQ(engine.size(), static_cast<size_t>(slots * m));
  for (int s = 0; s < slots; ++s) {
    for (int o = 0; o < m; ++o) {
      EXPECT_EQ(engine[s * m + o],
                Download(expected[s * m + o], BytesOf(outputs[o])))
          << "slot " << s << " output " << o << " in an engine";
    }
  }
}

template <typename T>
std::vector<uint8_t> AsBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

TEST(SlotPluginsTest, QkvNormRopeTwinMatchesTheLayerOnEachSlot) {
  struct Shape {
    int heads, kv_heads, depth, tokens;
  };
  // Gemma 4 26B-A4B's local and global layers, and a verify step's rows.
  for (const Shape& shape :
       {Shape{16, 8, 256, 1}, Shape{16, 2, 512, 1}, Shape{8, 2, 256, 5}}) {
    const int width = (shape.heads + 2 * shape.kv_heads) * shape.depth;
    nvinfer1::Dims qkv_dims{};
    qkv_dims.nbDims = 3;
    qkv_dims.d[0] = 1;
    qkv_dims.d[1] = shape.tokens;
    qkv_dims.d[2] = width;
    const std::vector<Tensor> inputs = {
        {qkv_dims, nvinfer1::DataType::kBF16},
        {Dims1(shape.tokens), nvinfer1::DataType::kINT32},
        {Dims1(shape.depth), nvinfer1::DataType::kFLOAT},
        {Dims1(shape.depth), nvinfer1::DataType::kFLOAT}};
    const std::vector<Tensor> outputs = {
        {nvinfer1::Dims4{1, shape.heads, shape.tokens, shape.depth},
         nvinfer1::DataType::kBF16},
        {nvinfer1::Dims4{1, shape.kv_heads, shape.tokens, shape.depth},
         nvinfer1::DataType::kBF16},
        {nvinfer1::Dims4{1, shape.kv_heads, shape.tokens, shape.depth},
         nvinfer1::DataType::kBF16}};
    std::vector<float> q_scale(shape.depth), k_scale(shape.depth);
    for (int i = 0; i < shape.depth; ++i) {
      q_scale[i] = 0.5f + 0.003f * static_cast<float>(i % 97);
      k_scale[i] = 1.25f - 0.002f * static_cast<float>(i % 89);
    }
    for (const int slots : {2, 3, 8}) {
      SCOPED_TRACE(::testing::Message()
                   << slots << " slots, heads " << shape.heads << "+2x"
                   << shape.kv_heads << ", depth " << shape.depth);
      auto layer = MakeLayer(
          CreateQkvNormRopePlugin(shape.heads, shape.kv_heads, shape.depth,
                                  /*rotary=*/shape.depth, /*has_v_norm=*/true,
                                  1e-6f, 1.0f, 10000.0f),
          inputs);
      ASSERT_NE(layer, nullptr);
      std::vector<std::vector<std::vector<uint8_t>>> per_slot;
      for (int s = 0; s < slots; ++s) {
        // Slots at positions of their own, short and long contexts.
        std::vector<int32_t> positions(shape.tokens);
        for (int t = 0; t < shape.tokens; ++t) {
          positions[t] = (s % 2 == 0 ? 3 : 8000) + 257 * s + t;
        }
        per_slot.push_back({Bytes(BytesOf(inputs[0]), 20 * s + 1, true),
                            AsBytes(positions), AsBytes(q_scale),
                            AsBytes(k_scale)});
      }
      CheckTwin(*layer, slots, outputs, per_slot, {-1, -1, -1});
    }
  }
}

TEST(SlotPluginsTest, Int8CacheWriteTwinMatchesTheLayerOnEachSlot) {
  // Gemma 4 E4B's key and value caches: two KV heads of depth 256.
  constexpr int kHeads = 2, kDepth = 256, kCacheLen = 2048;
  for (const bool columns : {false, true}) {
    const std::vector<Tensor> inputs = {
        {columns ? nvinfer1::Dims4{1, kHeads, kDepth, kCacheLen}
                 : nvinfer1::Dims4{1, kHeads, kCacheLen, kDepth},
         nvinfer1::DataType::kINT8},
        {nvinfer1::Dims4{1, kHeads, 1, kDepth}, nvinfer1::DataType::kBF16},
        {Dims1(1), nvinfer1::DataType::kINT32}};
    const std::vector<Tensor> outputs = {inputs[0]};
    for (const int slots : {2, 3, 8}) {
      for (const bool in_place : {true, false}) {
        SCOPED_TRACE(::testing::Message()
                     << slots << " slots, columns " << columns << ", in place "
                     << in_place);
        auto layer =
            MakeLayer(CreateInt8CacheWritePlugin(columns, 0.03f, -4), inputs);
        ASSERT_NE(layer, nullptr);
        std::vector<std::vector<std::vector<uint8_t>>> per_slot;
        for (int s = 0; s < slots; ++s) {
          // The start, the end and positions the write clamps.
          const int32_t positions[] = {0,  kCacheLen - 1, 517, kCacheLen + 5,
                                       -3, 1024,          7,   kCacheLen - 2};
          per_slot.push_back({Bytes(BytesOf(inputs[0]), 30 * s + 1, false),
                              Bytes(BytesOf(inputs[1]), 30 * s + 2, true),
                              AsBytes(std::vector<int32_t>{positions[s]})});
        }
        CheckTwin(*layer, slots, outputs, per_slot, {in_place ? 0 : -1});
      }
    }
  }
}

// FP32 to FP16 bits by the conversion every x86-64 and ARM64 host has, for
// the values the test writes (finite, in FP16's normal range or zero).
uint16_t HalfBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  if ((bits & 0x7fffffffu) == 0) return sign;
  const int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t half =
      (static_cast<uint32_t>(exponent) << 10) | ((bits & 0x7fffffu) >> 13);
  const uint32_t rest = bits & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ++half;
  return static_cast<uint16_t>(sign | half);
}

// The FP16 cache write plugin against what the single-sequence step runs
// today: a cast of the step's BF16 rows to the cache's type and TensorRT's
// own KV cache update, built into an engine and run on each slot's cache.
TEST(SlotPluginsTest, CacheWritePluginMatchesTheNativeUpdate) {
  constexpr int kHeads = 2, kDepth = 512, kCacheLen = 96, kSlots = 4;
  const Tensor cache = {nvinfer1::Dims4{1, kHeads, kCacheLen, kDepth},
                        nvinfer1::DataType::kHALF};
  const Tensor update = {nvinfer1::Dims4{1, kHeads, 1, kDepth},
                         nvinfer1::DataType::kBF16};
  const Tensor position = {Dims1(1), nvinfer1::DataType::kINT32};
  std::unique_ptr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(Logger()));
  ASSERT_NE(builder, nullptr);
  std::unique_ptr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(0));
  ASSERT_NE(network, nullptr);
  auto* cache_in = network->addInput("cache", cache.type, cache.dims);
  auto* update_in = network->addInput("update", update.type, update.dims);
  auto* position_in =
      network->addInput("position", position.type, position.dims);
  ASSERT_TRUE(cache_in != nullptr && update_in != nullptr &&
              position_in != nullptr);
  auto* cast = network->addCast(*update_in, nvinfer1::DataType::kHALF);
  ASSERT_NE(cast, nullptr);
  auto* writer =
      network->addKVCacheUpdate(*cache_in, *cast->getOutput(0), *position_in,
                                nvinfer1::KVCacheMode::kLINEAR);
  ASSERT_NE(writer, nullptr);
  writer->getOutput(0)->setName("updated");
  network->markOutput(*writer->getOutput(0));
  std::unique_ptr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig());
  ASSERT_NE(config, nullptr);
  std::unique_ptr<nvinfer1::IHostMemory> plan(
      builder->buildSerializedNetwork(*network, *config));
  ASSERT_NE(plan, nullptr);
  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(Logger()));
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(plan->data(), plan->size()));
  ASSERT_NE(engine, nullptr);
  ASSERT_STREQ(engine->getAliasedInputTensor("updated"), "cache");
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  ASSERT_NE(context, nullptr);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  std::unique_ptr<nvinfer1::IPluginV3> plugin(
      CreateCacheWriteSlotsPlugin(kSlots, /*columns=*/false));
  ASSERT_NE(plugin, nullptr);
  const std::vector<Tensor> inputs = {cache, update, position};
  auto plugin_in = Repeated(inputs, kSlots),
       plugin_out = Repeated({cache}, kSlots);
  ASSERT_EQ(Runtime(plugin.get())
                ->onShapeChange(plugin_in.data(), 3 * kSlots, plugin_out.data(),
                                kSlots),
            0);
  std::vector<Device> buffers;
  std::vector<const void*> plugin_inputs;
  std::vector<void*> plugin_outputs, native_caches;
  std::vector<std::vector<uint8_t>> expected;
  const int32_t positions[kSlots] = {0, 41, kCacheLen - 1, 7};
  for (int s = 0; s < kSlots; ++s) {
    const std::vector<uint8_t> cache_bytes =
        Bytes(BytesOf(cache), 40 * s + 1, true);
    const std::vector<uint8_t> update_bytes =
        Bytes(BytesOf(update), 40 * s + 2, true);
    // What the write leaves in the cache.
    expected.push_back(cache_bytes);
    for (int h = 0; h < kHeads; ++h) {
      for (int d = 0; d < kDepth; ++d) {
        uint16_t bf16;
        std::memcpy(&bf16, update_bytes.data() + 2 * (h * kDepth + d), 2);
        const uint32_t wide = static_cast<uint32_t>(bf16) << 16;
        float value;
        std::memcpy(&value, &wide, 4);
        const uint16_t half = HalfBits(value);
        std::memcpy(
            expected[s].data() +
                2 * ((static_cast<size_t>(h) * kCacheLen + positions[s]) *
                         kDepth +
                     d),
            &half, 2);
      }
    }
    for (int way = 0; way < 2; ++way) {
      buffers.push_back(Upload(cache_bytes));
      buffers.push_back(Upload(update_bytes));
      buffers.push_back(Upload(&positions[s], sizeof(int32_t)));
      for (int i = 0; i < 3; ++i) {
        ASSERT_NE(buffers[buffers.size() - 3 + i], nullptr);
      }
      void* const slot_cache = buffers[buffers.size() - 3].get();
      void* const slot_update = buffers[buffers.size() - 2].get();
      void* const slot_position = buffers[buffers.size() - 1].get();
      if (way == 0) {
        ASSERT_TRUE(context->setTensorAddress("cache", slot_cache));
        ASSERT_TRUE(context->setTensorAddress("updated", slot_cache));
        ASSERT_TRUE(context->setTensorAddress("update", slot_update));
        ASSERT_TRUE(context->setTensorAddress("position", slot_position));
        ASSERT_TRUE(context->enqueueV3(stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        native_caches.push_back(slot_cache);
      } else {
        plugin_inputs.insert(plugin_inputs.end(),
                             {slot_cache, slot_update, slot_position});
        plugin_outputs.push_back(slot_cache);
      }
    }
  }
  ASSERT_EQ(
      Runtime(plugin.get())
          ->enqueue(plugin_in.data(), plugin_out.data(), plugin_inputs.data(),
                    plugin_outputs.data(), nullptr, stream),
      0);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  for (int s = 0; s < kSlots; ++s) {
    const std::vector<uint8_t> written =
        Download(plugin_outputs[s], BytesOf(cache));
    EXPECT_EQ(written, Download(native_caches[s], BytesOf(cache)))
        << "slot " << s;
    EXPECT_EQ(written, expected[s]) << "slot " << s;
  }
  cudaStreamDestroy(stream);
}

}  // namespace
}  // namespace litert::nvidia
