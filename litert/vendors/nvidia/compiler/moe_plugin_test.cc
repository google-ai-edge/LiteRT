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

#include "litert/vendors/nvidia/compiler/moe_plugin.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"
#include "litert/vendors/nvidia/windows_compat.h"

namespace litert::nvidia {
namespace {

// The scalar conversions the loader's results are held to.
float HalfBitsToFloat(uint16_t bits) {
  const uint32_t sign = (bits & 0x8000u) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1f;
  const uint32_t mantissa = bits & 0x3ff;
  uint32_t word;
  if (exponent == 0) {
    if (mantissa == 0) {
      word = sign;
    } else {
      int shift = 0;
      uint32_t m = mantissa;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++shift;
      }
      word = sign | ((127 - 15 - shift + 1) << 23) | ((m & 0x3ffu) << 13);
    }
  } else if (exponent == 0x1f) {
    word = sign | 0x7f800000u | (mantissa << 13);
  } else {
    word = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 112;
  uint32_t mantissa = bits & 0x7fffffu;
  if (((bits >> 23) & 0xff) == 0xff) {
    return static_cast<uint16_t>(sign | 0x7c00u | (mantissa != 0 ? 0x200u : 0));
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000u;
    const int shift = 14 - exponent;
    uint32_t half = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half & 1u))) {
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
    ++half;
  }
  return static_cast<uint16_t>(sign | half);
}

struct Random {
  uint64_t state;
  uint64_t Next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
  void Fill(std::vector<uint8_t>* bytes) {
    size_t at = 0;
    for (; at + 8 <= bytes->size(); at += 8) {
      const uint64_t word = Next();
      std::memcpy(bytes->data() + at, &word, 8);
    }
    for (; at < bytes->size(); ++at)
      (*bytes)[at] = static_cast<uint8_t>(Next());
  }
};

struct Shape {
  int experts;
  int model_dim;
  int hidden_dim;
  int dense_hidden;
  int scale_type;  // the experts', the router's and the dense rows': 0 FP32
};

// Block scales as a model file holds them, FP32 or FP16: ordinary positive
// scales, and among them rows of zeros and rows of arbitrary bit patterns
// (denormals, infinities, NaNs, negative values).
std::vector<uint8_t> MakeScales(size_t rows, size_t blocks, int scale_type,
                                Random* random) {
  const size_t width = scale_type == 0 ? 4 : 2;
  std::vector<uint8_t> bytes(rows * blocks * width);
  for (size_t r = 0; r < rows; ++r) {
    const int kind = r % 53 == 7 ? 1 : r % 61 == 11 ? 2 : 0;
    for (size_t b = 0; b < blocks; ++b) {
      uint8_t* out = bytes.data() + (r * blocks + b) * width;
      const uint64_t bits = random->Next();
      if (scale_type == 0) {
        float value =
            1e-3f + 0.05f * static_cast<float>(bits >> 40) / 16777216.0f;
        if (kind == 1) value = 0.0f;
        if (kind == 2) {
          const uint32_t raw = static_cast<uint32_t>(bits);
          std::memcpy(&value, &raw, 4);
        }
        std::memcpy(out, &value, 4);
      } else {
        // FP16 in [2^-10, 2^-4) for ordinary rows.
        uint16_t half = static_cast<uint16_t>(0x1400u + (bits >> 40) % 0x1800u);
        if (kind == 1) half = 0;
        if (kind == 2) half = static_cast<uint16_t>(bits);
        std::memcpy(out, &half, 2);
      }
    }
  }
  return bytes;
}

// The scales as the arena holds them when they are not compact: FP16.
std::vector<uint8_t> AsHalves(const std::vector<uint8_t>& file,
                              int scale_type) {
  if (scale_type != 0) return file;
  std::vector<uint8_t> halves(file.size() / 2);
  for (size_t v = 0; v < file.size() / 4; ++v) {
    float value;
    std::memcpy(&value, file.data() + v * 4, 4);
    const uint16_t half = FloatToHalfBits(value);
    std::memcpy(halves.data() + v * 2, &half, 2);
  }
  return halves;
}

// A layer of a model file and what its arena has to hold.
struct Layer {
  MoeExpertSource source;
  Shape shape;
  std::vector<uint8_t> gate, up, down;
  std::vector<uint8_t> gate_fractions, up_fractions, down_fractions;
  std::vector<uint16_t> gate_max, up_max, down_max;
  std::vector<uint8_t> router, router_scales;
  std::vector<uint8_t> dense_gate_up, dense_gate_up_scales;
  std::vector<uint8_t> dense_down, dense_down_scales;
};

// The compact form of a piece of block scales: per row the FP16 maximum and
// a byte per block.
void Compact(const std::vector<uint8_t>& file, size_t rows, size_t blocks,
             int scale_type, std::vector<uint8_t>* fractions,
             std::vector<uint16_t>* maxima) {
  fractions->resize(rows * blocks);
  maxima->resize(rows);
  std::vector<float> row(blocks);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t b = 0; b < blocks; ++b) {
      if (scale_type == 0) {
        std::memcpy(&row[b], file.data() + (r * blocks + b) * 4, 4);
      } else {
        uint16_t half;
        std::memcpy(&half, file.data() + (r * blocks + b) * 2, 2);
        row[b] = HalfBitsToFloat(half);
      }
    }
    LiteRtNvidiaCompactScaleRow(row.data(), static_cast<int32_t>(blocks),
                                fractions->data() + r * blocks,
                                maxima->data() + r);
  }
}

// Writes a model file of `count` layers of `shape`, their pieces apart from
// each other and out of order.
std::vector<Layer> WriteModel(const std::string& path, const Shape& shape,
                              int count, uint64_t seed) {
  Random random{seed};
  const size_t gate_rows =
      static_cast<size_t>(shape.hidden_dim) * shape.experts;
  const size_t down_rows = static_cast<size_t>(shape.model_dim) * shape.experts;
  const size_t model_blocks = shape.model_dim / 32;
  const size_t hidden_blocks = shape.hidden_dim / 32;
  const size_t dense_blocks = shape.dense_hidden / 32;
  const size_t dense = shape.dense_hidden;
  std::vector<Layer> layers(count);
  std::FILE* file = std::fopen(path.c_str(), "wb");
  EXPECT_NE(file, nullptr);
  if (file == nullptr) return {};
  uint64_t offset = 0;
  std::vector<uint8_t> padding;
  const auto write = [&](const std::vector<uint8_t>& bytes, Layer* layer,
                         int piece) {
    padding.assign(1 + random.Next() % 4099, 0xa5);
    std::fwrite(padding.data(), 1, padding.size(), file);
    offset += padding.size();
    layer->source.offsets[piece] = offset;
    layer->source.sizes[piece] = bytes.size();
    std::fwrite(bytes.data(), 1, bytes.size(), file);
    offset += bytes.size();
  };
  for (Layer& layer : layers) {
    layer.shape = shape;
    layer.source.model_path = path;
    layer.source.scale_type = shape.scale_type;
    layer.source.router_scale_type = shape.scale_type;
    layer.source.dense_scale_type = shape.scale_type;
    layer.source.dense_hidden = shape.dense_hidden;
    layer.gate.resize(gate_rows * shape.model_dim / 2);
    layer.up.resize(layer.gate.size());
    layer.down.resize(down_rows * shape.hidden_dim / 2);
    layer.router.resize(static_cast<size_t>(shape.experts) * shape.model_dim /
                        2);
    std::vector<uint8_t> dense_gate(dense * shape.model_dim / 2);
    std::vector<uint8_t> dense_up(dense_gate.size());
    layer.dense_down.resize(static_cast<size_t>(shape.model_dim) * dense / 2);
    for (auto* bytes : {&layer.gate, &layer.up, &layer.down, &layer.router,
                        &dense_gate, &dense_up, &layer.dense_down}) {
      random.Fill(bytes);
    }
    const auto gate_s =
        MakeScales(gate_rows, model_blocks, shape.scale_type, &random);
    const auto up_s =
        MakeScales(gate_rows, model_blocks, shape.scale_type, &random);
    const auto down_s =
        MakeScales(down_rows, hidden_blocks, shape.scale_type, &random);
    const auto router_s =
        MakeScales(shape.experts, model_blocks, shape.scale_type, &random);
    const auto dense_gate_s =
        MakeScales(dense, model_blocks, shape.scale_type, &random);
    const auto dense_up_s =
        MakeScales(dense, model_blocks, shape.scale_type, &random);
    const auto dense_down_s =
        MakeScales(shape.model_dim, dense_blocks, shape.scale_type, &random);
    // Out of the pieces' order, as a model file may hold them.
    write(layer.down, &layer, 2);
    write(dense_up_s, &layer, 11);
    write(gate_s, &layer, 3);
    write(layer.gate, &layer, 0);
    write(router_s, &layer, 7);
    write(dense_gate, &layer, 8);
    write(down_s, &layer, 5);
    write(layer.up, &layer, 1);
    write(layer.dense_down, &layer, 12);
    write(up_s, &layer, 4);
    write(layer.router, &layer, 6);
    write(dense_gate_s, &layer, 10);
    write(dense_up, &layer, 9);
    write(dense_down_s, &layer, 13);
    Compact(gate_s, gate_rows, model_blocks, shape.scale_type,
            &layer.gate_fractions, &layer.gate_max);
    Compact(up_s, gate_rows, model_blocks, shape.scale_type,
            &layer.up_fractions, &layer.up_max);
    Compact(down_s, down_rows, hidden_blocks, shape.scale_type,
            &layer.down_fractions, &layer.down_max);
    layer.router_scales = AsHalves(router_s, shape.scale_type);
    layer.dense_gate_up = dense_gate;
    layer.dense_gate_up.insert(layer.dense_gate_up.end(), dense_up.begin(),
                               dense_up.end());
    layer.dense_gate_up_scales = AsHalves(dense_gate_s, shape.scale_type);
    const auto up_halves = AsHalves(dense_up_s, shape.scale_type);
    layer.dense_gate_up_scales.insert(layer.dense_gate_up_scales.end(),
                                      up_halves.begin(), up_halves.end());
    layer.dense_down_scales = AsHalves(dense_down_s, shape.scale_type);
  }
  padding.assign(777, 0xa5);
  std::fwrite(padding.data(), 1, padding.size(), file);
  offset += padding.size();
  EXPECT_EQ(std::fclose(file), 0);
  for (Layer& layer : layers) layer.source.model_size = offset;
  return layers;
}

std::unique_ptr<nvinfer1::IPluginV3> MakePlugin(const Layer& layer) {
  return std::unique_ptr<nvinfer1::IPluginV3>(CreateMoePlugin(
      layer.shape.experts, /*top_k=*/2, layer.shape.model_dim,
      layer.shape.hidden_dim, /*gate=*/1, layer.source, /*routed=*/true,
      /*norm_eps=*/1e-6f, /*router_multiplier=*/1.0f, /*block=*/true,
      /*dense_gate=*/1, /*residual_multiplier=*/1.0f));
}

// The plugin as an engine being deserialized creates it.
std::unique_ptr<nvinfer1::IPluginV3> Deserialize(const Layer& layer) {
  const auto built = MakePlugin(layer);
  if (built == nullptr) return nullptr;
  auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(
      getPluginRegistry()->getCreator("LiteRtNvidiaMoe", "1", ""));
  if (creator == nullptr) return nullptr;
  auto* core = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      built->getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
  return std::unique_ptr<nvinfer1::IPluginV3>(creator->createPlugin(
      "moe", core->getFieldsToSerialize(), nvinfer1::TensorRTPhase::kRUNTIME));
}

// The plugin attached to an execution context: its arena is loaded.
std::unique_ptr<nvinfer1::IPluginV3> Attach(nvinfer1::IPluginV3* plugin) {
  auto* runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      plugin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
  return std::unique_ptr<nvinfer1::IPluginV3>(
      runtime->attachToContext(nullptr));
}

// The rows of an expert piece as an arena holds them: in the model's order
// (row channel * experts + expert) or expert by expert.
template <typename T>
std::vector<T> ArenaOrder(const std::vector<T>& model, size_t experts,
                          size_t channels, bool expert_major) {
  if (!expert_major) return model;
  const size_t row = model.size() / (experts * channels);
  std::vector<T> arena(model.size());
  for (size_t c = 0; c < channels; ++c) {
    for (size_t e = 0; e < experts; ++e) {
      std::copy_n(model.begin() + (c * experts + e) * row, row,
                  arena.begin() + (e * channels + c) * row);
    }
  }
  return arena;
}

void ExpectBytes(const void* arena, const void* expected, size_t bytes,
                 const char* what) {
  SCOPED_TRACE(what);
  ASSERT_NE(arena, nullptr);
  std::vector<uint8_t> actual(bytes);
  ASSERT_EQ(cudaMemcpy(actual.data(), arena, bytes, cudaMemcpyDefault),
            cudaSuccess);
  if (std::memcmp(actual.data(), expected, bytes) == 0) return;
  const auto* want = static_cast<const uint8_t*>(expected);
  size_t first = 0;
  while (actual[first] == want[first]) ++first;
  ADD_FAILURE() << "differs from byte " << first << " of " << bytes << ": "
                << static_cast<int>(actual[first]) << " for "
                << static_cast<int>(want[first]);
}

// The arena of `layer` holds what the scalar conversions make of the file.
void ExpectArena(const Layer& layer, bool expect_device,
                 bool expect_expert_major) {
  SCOPED_TRACE(::testing::Message()
               << "experts at " << layer.source.offsets[0]);
  LiteRtNvidiaMoeArena arena{};
  bool on_device = false;
  ASSERT_TRUE(LoadMoeArena(layer.source, layer.shape.experts,
                           layer.shape.model_dim, layer.shape.hidden_dim,
                           &arena, &on_device));
  EXPECT_EQ(on_device, expect_device);
  EXPECT_EQ(arena.compact_scales, 1);
  EXPECT_EQ(arena.expert_major != 0, expect_expert_major);
  const size_t experts = layer.shape.experts;
  const size_t hidden = layer.shape.hidden_dim;
  const size_t model = layer.shape.model_dim;
  const bool by_expert = arena.expert_major != 0;
  const auto expect_rows = [&](const void* pointer, const auto& rows,
                               size_t channels, const char* what) {
    const auto expected = ArenaOrder(rows, experts, channels, by_expert);
    ExpectBytes(pointer, expected.data(), expected.size() * sizeof(expected[0]),
                what);
  };
  expect_rows(arena.gate, layer.gate, hidden, "gate");
  expect_rows(arena.up, layer.up, hidden, "up");
  expect_rows(arena.down, layer.down, model, "down");
  expect_rows(arena.gate_scales, layer.gate_fractions, hidden,
              "gate fractions");
  expect_rows(arena.up_scales, layer.up_fractions, hidden, "up fractions");
  expect_rows(arena.down_scales, layer.down_fractions, model, "down fractions");
  expect_rows(arena.gate_scale_max, layer.gate_max, hidden, "gate maxima");
  expect_rows(arena.up_scale_max, layer.up_max, hidden, "up maxima");
  expect_rows(arena.down_scale_max, layer.down_max, model, "down maxima");
  ExpectBytes(arena.router, layer.router.data(), layer.router.size(), "router");
  ExpectBytes(arena.router_scales, layer.router_scales.data(),
              layer.router_scales.size(), "router scales");
  ExpectBytes(arena.dense_gate_up, layer.dense_gate_up.data(),
              layer.dense_gate_up.size(), "dense gate and up");
  ExpectBytes(arena.dense_gate_up_scales, layer.dense_gate_up_scales.data(),
              layer.dense_gate_up_scales.size(), "dense gate and up scales");
  ExpectBytes(arena.dense_down, layer.dense_down.data(),
              layer.dense_down.size(), "dense down");
  ExpectBytes(arena.dense_down_scales, layer.dense_down_scales.data(),
              layer.dense_down_scales.size(), "dense down scales");
}

size_t LoadedBytes() {
  size_t device = 0;
  size_t host = 0;
  MoeArenaBytes(&device, &host);
  return device + host;
}

// The parameter: LITERT_NVIDIA_MOE_EXPERT_MAJOR, empty for the default.
class MoePluginTest : public testing::TestWithParam<const char*> {
 protected:
  void SetUp() override {
    SetMoeArenaLoadingEnabled(true);
    // Empty: the defaults.
    setenv("LITERT_NVIDIA_MOE_PLACEMENT", "", 1);
    setenv("LITERT_NVIDIA_MOE_LOAD_THREADS", "", 1);
    setenv("LITERT_NVIDIA_MOE_EXPERT_MAJOR", GetParam(), 1);
  }
  void TearDown() override {
    for (const std::string& path : paths_) std::remove(path.c_str());
  }
  // A file of this test and layout alone: arenas are kept by file.
  std::string Path(const std::string& name) {
    paths_.push_back(::testing::TempDir() + "/moe_" + name + "_" + GetParam() +
                     ".bin");
    return paths_.back();
  }
  bool ExpertMajor() const { return std::strcmp(GetParam(), "0") != 0; }

  std::vector<std::string> paths_;
};

INSTANTIATE_TEST_SUITE_P(Layouts, MoePluginTest, testing::Values("", "0", "1"));

// Gemma 4 26B-A4B's layer: every expert piece spans several staging buffers.
TEST_P(MoePluginTest, AttachLoadsTheBytesOfTheFile) {
  const std::vector<Layer> layers =
      WriteModel(Path("26b_layer"), {128, 2816, 704, 2112, 1}, 1, 1);
  ASSERT_EQ(layers.size(), 1u);
  const auto plugin = MakePlugin(layers[0]);
  ASSERT_NE(plugin, nullptr);
  const size_t before = LoadedBytes();
  const auto attached = Attach(plugin.get());
  ASSERT_NE(attached, nullptr);
  EXPECT_GT(LoadedBytes(), before);
  ExpectArena(layers[0], /*expect_device=*/true, ExpertMajor());
}

// The layers of a deserialized engine load together when the first of them
// is attached, by one thread or several, FP32 scales converted on the way.
TEST_P(MoePluginTest, AttachLoadsTheAnnouncedArenas) {
  int seed = 100;
  for (const char* threads : {"1", "3", "16"}) {
    SCOPED_TRACE(threads);
    setenv("LITERT_NVIDIA_MOE_LOAD_THREADS", threads, 1);
    const std::vector<Layer> layers =
        WriteModel(Path(std::string("layers_") + threads), {12, 256, 96, 64, 0},
                   7, ++seed);
    ASSERT_EQ(layers.size(), 7u);
    std::vector<std::unique_ptr<nvinfer1::IPluginV3>> plugins;
    for (const Layer& layer : layers) {
      plugins.push_back(Deserialize(layer));
      ASSERT_NE(plugins.back(), nullptr);
    }
    const size_t before = LoadedBytes();
    const auto attached = Attach(plugins[0].get());
    ASSERT_NE(attached, nullptr);
    const size_t after = LoadedBytes();
    EXPECT_GT(after, before);
    for (const Layer& layer : layers) {
      ExpectArena(layer, /*expect_device=*/true, ExpertMajor());
    }
    // Checking them loaded nothing more.
    EXPECT_EQ(LoadedBytes(), after);
  }
}

// Arenas in page-locked host memory (a device short of memory) are filled
// in place.
TEST_P(MoePluginTest, LoadsHostArenas) {
  setenv("LITERT_NVIDIA_MOE_PLACEMENT", "host", 1);
  for (const int scale_type : {0, 1}) {
    const std::vector<Layer> layers =
        WriteModel(Path("host_" + std::to_string(scale_type)),
                   {16, 704, 128, 96, scale_type}, 3, 7 + scale_type);
    ASSERT_EQ(layers.size(), 3u);
    std::vector<std::unique_ptr<nvinfer1::IPluginV3>> plugins;
    for (const Layer& layer : layers) plugins.push_back(Deserialize(layer));
    for (const Layer& layer : layers) {
      ExpectArena(layer, /*expect_device=*/false, ExpertMajor());
    }
  }
}

// Experts whose rows of one output channel are larger than the loader's
// staging buffers (1,024 experts of 4,112-byte rows: 4.2 MB) are loaded all
// the same, in the model's row order.
TEST_P(MoePluginTest, KeepsTheModelsOrderForChannelsLargerThanTheStaging) {
  if (std::strcmp(GetParam(), "1") != 0) {
    GTEST_SKIP() << "Runs once, with expert-major arenas asked for.";
  }
  const std::vector<Layer> layers =
      WriteModel(Path("wide_channels"), {1024, 8224, 32, 64, 1}, 1, 11);
  ASSERT_EQ(layers.size(), 1u);
  ExpectArena(layers[0], /*expect_device=*/true, /*expect_expert_major=*/false);
}

// A file that is not the one the plan was built from is not loaded.
TEST_P(MoePluginTest, RejectsAFileOfAnotherSize) {
  std::vector<Layer> layers =
      WriteModel(Path("resized"), {12, 256, 96, 64, 1}, 1, 3);
  ASSERT_EQ(layers.size(), 1u);
  layers[0].source.model_size += 1;
  LiteRtNvidiaMoeArena arena{};
  bool on_device = false;
  EXPECT_FALSE(LoadMoeArena(layers[0].source, 12, 256, 96, &arena, &on_device));
}

}  // namespace
}  // namespace litert::nvidia
