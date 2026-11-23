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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"
#include "litert/vendors/nvidia/trtllm/slot_test_util.h"

// LiteRtNvidiaLaunchMoeLayerBlockSlotRows: the mixture-of-experts layer
// block over the rows of several sequence slots against
// LiteRtNvidiaLaunchMoeLayerBlock on each row alone.

namespace {

using litert::nvidia::slot_test::Bytes;
using litert::nvidia::slot_test::Device;
using litert::nvidia::slot_test::Download;
using litert::nvidia::slot_test::TimeTwoWays;
using litert::nvidia::slot_test::Upload;

// FP16 bits of a positive normal value.
uint16_t HalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t exponent = ((bits >> 23) & 0xff) - 127 + 15;
  return static_cast<uint16_t>((exponent << 10) | ((bits & 0x7fffffu) >> 13));
}

// `count` FP16 scales in the range of Gemma 4 26B's blockwise scales.
std::vector<uint16_t> Scales(size_t count, uint32_t seed) {
  std::vector<uint16_t> scales(count);
  uint32_t state = seed * 2654435761u + 99u;
  for (uint16_t& scale : scales) {
    state = state * 1664525u + 1013904223u;
    scale = HalfBits(0.003f + 0.0002f * static_cast<float>((state >> 16) % 97));
  }
  return scales;
}

struct LayerShape {
  int experts, top_k, dim, hidden, dense_hidden, gate, dense_gate;
};

// A layer's weights on the device, with FP16 expert scales in the model's
// row order or (compact) the plugin's arena: compact scales, expert-major
// rows and a verify step's rows grouped by expert.
struct Layer {
  LayerShape shape;
  std::vector<Device> memory;
  LiteRtNvidiaMoeArena arena = {};
  Device expert_scale, router_norm, expert_norm, dense_norm, dense_post,
      moe_post, sum_post;
  cudaStream_t side_stream = nullptr;
  cudaEvent_t fork = nullptr, join = nullptr;
  size_t dense_bytes = 0;   // what a launch reads of the router and the dense
                            // feed-forward
  size_t expert_bytes = 0;  // and of one routed expert

  ~Layer() {
    if (join != nullptr) cudaEventDestroy(join);
    if (fork != nullptr) cudaEventDestroy(fork);
    if (side_stream != nullptr) cudaStreamDestroy(side_stream);
  }

  LiteRtNvidiaMoeBlockParams Block() const {
    return {shape.dense_hidden,
            shape.dense_gate,
            /*residual_multiplier=*/1.25f,
            static_cast<const float*>(dense_norm.get()),
            static_cast<const float*>(dense_post.get()),
            static_cast<const float*>(moe_post.get()),
            static_cast<const float*>(sum_post.get()),
            side_stream,
            fork,
            join};
  }
};

std::unique_ptr<Layer> MakeLayer(const LayerShape& s, bool compact,
                                 uint32_t seed) {
  auto layer = std::make_unique<Layer>();
  layer->shape = s;
  bool ok = true;
  const auto keep = [&](Device device) {
    ok = ok && device != nullptr;
    layer->memory.push_back(std::move(device));
    return layer->memory.back().get();
  };
  const auto weights = [&](size_t rows, int inputs, uint32_t salt) {
    return static_cast<const uint8_t*>(
        keep(Upload(Bytes(rows * inputs / 2, seed + salt, /*floats=*/false))));
  };
  const size_t hidden_rows = static_cast<size_t>(s.hidden) * s.experts;
  const size_t down_rows = static_cast<size_t>(s.dim) * s.experts;
  const int dim_blocks = s.dim / 32, hidden_blocks = s.hidden / 32;
  LiteRtNvidiaMoeArena& arena = layer->arena;
  arena.gate = weights(hidden_rows, s.dim, 1);
  arena.up = weights(hidden_rows, s.dim, 2);
  arena.down = weights(down_rows, s.hidden, 3);
  if (compact) {
    arena.gate_scales =
        keep(Upload(Bytes(hidden_rows * dim_blocks, seed + 4, false)));
    arena.up_scales =
        keep(Upload(Bytes(hidden_rows * dim_blocks, seed + 5, false)));
    arena.down_scales =
        keep(Upload(Bytes(down_rows * hidden_blocks, seed + 6, false)));
    arena.gate_scale_max = keep(Upload(Scales(hidden_rows, seed + 7)));
    arena.up_scale_max = keep(Upload(Scales(hidden_rows, seed + 8)));
    arena.down_scale_max = keep(Upload(Scales(down_rows, seed + 9)));
    arena.compact_scales = 1;
    arena.expert_major = 1;
    arena.group_rows = 1;
  } else {
    arena.gate_scales =
        keep(Upload(Scales(hidden_rows * dim_blocks, seed + 4)));
    arena.up_scales = keep(Upload(Scales(hidden_rows * dim_blocks, seed + 5)));
    arena.down_scales =
        keep(Upload(Scales(down_rows * hidden_blocks, seed + 6)));
  }
  arena.router = weights(s.experts, s.dim, 10);
  arena.router_scales = keep(
      Upload(Scales(static_cast<size_t>(s.experts) * dim_blocks, seed + 11)));
  arena.dense_gate_up =
      weights(2 * static_cast<size_t>(s.dense_hidden), s.dim, 12);
  arena.dense_gate_up_scales = keep(Upload(
      Scales(2 * static_cast<size_t>(s.dense_hidden) * dim_blocks, seed + 13)));
  arena.dense_down = weights(s.dim, s.dense_hidden, 14);
  arena.dense_down_scales = keep(Upload(
      Scales(static_cast<size_t>(s.dim) * (s.dense_hidden / 32), seed + 15)));
  const auto vector_of = [&](int count, float base, float step, int period) {
    std::vector<float> values(count);
    for (int i = 0; i < count; ++i) {
      values[i] = base + step * static_cast<float>(i % period);
    }
    Device device = Upload(values);
    ok = ok && device != nullptr;
    return device;
  };
  layer->expert_scale = vector_of(s.experts, 0.98f, 0.0003f, 1 << 20);
  layer->router_norm = vector_of(s.dim, 0.7f, 0.0005f, 61);
  layer->expert_norm = vector_of(s.dim, 1.1f, -0.0004f, 53);
  layer->dense_norm = vector_of(s.dim, 0.9f, 0.0003f, 47);
  layer->dense_post = vector_of(s.dim, 1.2f, -0.0006f, 43);
  layer->moe_post = vector_of(s.dim, 0.8f, 0.0004f, 41);
  layer->sum_post = vector_of(s.dim, 1.05f, -0.0002f, 37);
  ok = ok &&
       cudaStreamCreateWithFlags(&layer->side_stream, cudaStreamNonBlocking) ==
           cudaSuccess &&
       cudaEventCreateWithFlags(&layer->fork, cudaEventDisableTiming) ==
           cudaSuccess &&
       cudaEventCreateWithFlags(&layer->join, cudaEventDisableTiming) ==
           cudaSuccess;
  // Weights (half a byte a value) and scales (FP16, or a byte a block and
  // an FP16 maximum a row).
  const auto piece = [&](size_t rows, int inputs, bool compact_scales) {
    return rows * inputs / 2 + (compact_scales ? rows * (inputs / 32) + rows * 2
                                               : rows * (inputs / 32) * 2);
  };
  layer->dense_bytes =
      piece(s.experts, s.dim, false) +
      piece(2 * static_cast<size_t>(s.dense_hidden), s.dim, false) +
      piece(s.dim, s.dense_hidden, false);
  layer->expert_bytes =
      2 * piece(s.hidden, s.dim, compact) + piece(s.dim, s.hidden, compact);
  return ok ? std::move(layer) : nullptr;
}

LiteRtNvidiaMoeShape Shape(const LayerShape& s, int rows) {
  return {rows, s.top_k, s.experts, s.dim, s.hidden, s.gate};
}

size_t WorkspaceBytes(const LayerShape& s, int rows) {
  const LiteRtNvidiaMoeShape shape = Shape(s, rows);
  return LiteRtNvidiaMoeBlockWorkspaceBytes(&shape, s.dense_hidden,
                                            s.dense_gate);
}

constexpr float kEps = 1e-6f;
constexpr float kRouterMultiplier = 0.6f;

// The layer block over `rows` rows of x in one launch.
cudaError_t LaunchSlotRows(const Layer& layer, int rows, const void* x,
                           void* workspace, void* y, cudaStream_t stream) {
  const LiteRtNvidiaMoeShape shape = Shape(layer.shape, rows);
  const LiteRtNvidiaMoeBlockParams block = layer.Block();
  return LiteRtNvidiaLaunchMoeLayerBlockSlotRows(
      &shape, &layer.arena, x,
      static_cast<const float*>(layer.expert_scale.get()),
      static_cast<const float*>(layer.router_norm.get()), kRouterMultiplier,
      static_cast<const float*>(layer.expert_norm.get()), kEps,
      /*renormalize=*/1, &block, workspace, y, stream);
}

// The layer block on each of the rows alone, as a decode step runs it.
cudaError_t LaunchEachRow(const Layer& layer, int rows, const void* x,
                          void* workspace, void* y, cudaStream_t stream) {
  const LiteRtNvidiaMoeShape shape = Shape(layer.shape, 1);
  const LiteRtNvidiaMoeBlockParams block = layer.Block();
  const size_t row_bytes = static_cast<size_t>(layer.shape.dim) * 2;
  for (int r = 0; r < rows; ++r) {
    const cudaError_t status = LiteRtNvidiaLaunchMoeLayerBlock(
        &shape, &layer.arena, static_cast<const char*>(x) + r * row_bytes,
        static_cast<const float*>(layer.expert_scale.get()),
        static_cast<const float*>(layer.router_norm.get()), kRouterMultiplier,
        static_cast<const float*>(layer.expert_norm.get()), kEps,
        /*renormalize=*/1, &block, workspace,
        static_cast<char*>(y) + r * row_bytes, stream);
    if (status != cudaSuccess) return status;
  }
  return cudaSuccess;
}

// Rows of BF16 values of both signs near one; row r repeats row `same_as[r]`
// when that is not r (two slots at the same token route alike).
std::vector<uint8_t> Rows(int rows, int dim, uint32_t seed,
                          const std::vector<int>& same_as = {}) {
  const size_t row_bytes = static_cast<size_t>(dim) * 2;
  std::vector<uint8_t> x(rows * row_bytes);
  for (int r = 0; r < rows; ++r) {
    const int source = r < static_cast<int>(same_as.size()) ? same_as[r] : r;
    const std::vector<uint8_t> row = Bytes(row_bytes, seed + 31 * source, true);
    std::copy(row.begin(), row.end(), x.begin() + r * row_bytes);
  }
  return x;
}

// The experts the launch routed its rows to: the routing is the first thing
// in the workspace (int32 [rows * top_k]).
std::vector<int32_t> RoutedExperts(const void* workspace, int pairs) {
  std::vector<int32_t> experts(pairs);
  cudaMemcpy(experts.data(), workspace, pairs * sizeof(int32_t),
             cudaMemcpyDeviceToHost);
  return experts;
}

TEST(MoeSlotRowsTest, EveryRowMatchesItsOwnLaunch) {
  struct Case {
    const char* name;
    LayerShape shape;
    bool compact;
  };
  const Case cases[] = {
      {"FP16 scales, the model's order", {64, 4, 704, 128, 96, 1, 1}, false},
      {"the plugin's arena", {64, 4, 704, 128, 96, 1, 1}, true},
      {"erf GELU, SiLU experts", {64, 4, 704, 128, 96, 3, 2}, true},
      {"the 26B's layer", {128, 8, 2816, 704, 2112, 1, 1}, true},
  };
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  for (const Case& c : cases) {
    const auto layer = MakeLayer(c.shape, c.compact, 5);
    ASSERT_NE(layer, nullptr);
    const size_t row_bytes = static_cast<size_t>(c.shape.dim) * 2;
    for (int rows = 1; rows <= kLiteRtNvidiaInt4BlockGemvMaxRows; ++rows) {
      SCOPED_TRACE(::testing::Message() << c.name << ", " << rows << " rows");
      // The last two rows repeat rows 0 and 1: the same experts twice.
      std::vector<int> same_as(rows);
      for (int r = 0; r < rows; ++r) same_as[r] = r;
      if (rows >= 4) {
        same_as[rows - 2] = 0;
        same_as[rows - 1] = 1;
      }
      const Device x = Upload(Rows(rows, c.shape.dim, 100 * rows, same_as));
      const Device y = Upload(nullptr, rows * row_bytes);
      const Device expected = Upload(nullptr, rows * row_bytes);
      const Device workspace = Upload(nullptr, WorkspaceBytes(c.shape, rows));
      const Device row_workspace = Upload(nullptr, WorkspaceBytes(c.shape, 1));
      ASSERT_TRUE(x && y && expected && workspace && row_workspace);
      ASSERT_EQ(LaunchEachRow(*layer, rows, x.get(), row_workspace.get(),
                              expected.get(), stream),
                cudaSuccess);
      ASSERT_EQ(LaunchSlotRows(*layer, rows, x.get(), workspace.get(), y.get(),
                               stream),
                cudaSuccess);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      const std::vector<uint8_t> got = Download(y.get(), rows * row_bytes);
      const std::vector<uint8_t> want =
          Download(expected.get(), rows * row_bytes);
      ASSERT_EQ(got.size(), rows * row_bytes);
      for (int r = 0; r < rows; ++r) {
        EXPECT_EQ(std::memcmp(got.data() + r * row_bytes,
                              want.data() + r * row_bytes, row_bytes),
                  0)
            << "row " << r;
        // A row of finite values that the block changed.
        int finite = 0;
        for (int i = 0; i < c.shape.dim; ++i) {
          uint16_t bits = 0;
          std::memcpy(&bits, got.data() + r * row_bytes + 2 * i, 2);
          finite += (bits & 0x7f80) != 0x7f80;
        }
        EXPECT_EQ(finite, c.shape.dim) << "row " << r;
      }
      if (rows >= 2) {
        EXPECT_NE(std::memcmp(got.data(), got.data() + row_bytes, row_bytes),
                  0);
        // Different rows choose different experts, the repeated rows the
        // same ones.
        const std::vector<int32_t> routed =
            RoutedExperts(workspace.get(), rows * c.shape.top_k);
        const std::set<int32_t> first(routed.begin(),
                                      routed.begin() + c.shape.top_k);
        const std::set<int32_t> second(routed.begin() + c.shape.top_k,
                                       routed.begin() + 2 * c.shape.top_k);
        EXPECT_EQ(first.size(), static_cast<size_t>(c.shape.top_k));
        EXPECT_NE(first, second);
        if (rows >= 4) {
          EXPECT_TRUE(std::equal(routed.begin(), routed.begin() + c.shape.top_k,
                                 routed.end() - 2 * c.shape.top_k));
        }
      }
    }
  }
  cudaStreamDestroy(stream);
}

TEST(MoeSlotRowsTest, RejectsInvalidArguments) {
  const LayerShape s = {64, 4, 704, 128, 96, 1, 1};
  const auto layer = MakeLayer(s, /*compact=*/true, 5);
  ASSERT_NE(layer, nullptr);
  const int rows = 3;
  const Device x = Upload(Rows(rows, s.dim, 1));
  const Device y = Upload(nullptr, static_cast<size_t>(rows) * s.dim * 2);
  const Device workspace =
      Upload(nullptr, WorkspaceBytes(s, kLiteRtNvidiaInt4BlockGemvMaxRows + 1));
  ASSERT_TRUE(x && y && workspace);
  EXPECT_EQ(
      LaunchSlotRows(*layer, rows, x.get(), workspace.get(), y.get(), nullptr),
      cudaSuccess);
  // More rows than slots: a prefill's rows are the layer block's.
  EXPECT_EQ(LaunchSlotRows(*layer, kLiteRtNvidiaInt4BlockGemvMaxRows + 1,
                           x.get(), workspace.get(), y.get(), nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(
      LaunchSlotRows(*layer, 0, x.get(), workspace.get(), y.get(), nullptr),
      cudaErrorInvalidValue);
  EXPECT_EQ(
      LaunchSlotRows(*layer, rows, nullptr, workspace.get(), y.get(), nullptr),
      cudaErrorInvalidValue);
  EXPECT_EQ(LaunchSlotRows(*layer, rows, x.get(), nullptr, y.get(), nullptr),
            cudaErrorInvalidValue);
  const LiteRtNvidiaMoeShape shape = Shape(s, rows);
  EXPECT_EQ(LiteRtNvidiaLaunchMoeLayerBlockSlotRows(
                &shape, &layer->arena, x.get(),
                static_cast<const float*>(layer->expert_scale.get()),
                static_cast<const float*>(layer->router_norm.get()),
                kRouterMultiplier,
                static_cast<const float*>(layer->expert_norm.get()), kEps, 1,
                /*block=*/nullptr, workspace.get(), y.get(), nullptr),
            cudaErrorInvalidValue);
  LiteRtNvidiaMoeArena no_dense = layer->arena;
  no_dense.dense_down = nullptr;
  const LiteRtNvidiaMoeBlockParams block = layer->Block();
  EXPECT_EQ(LiteRtNvidiaLaunchMoeLayerBlockSlotRows(
                &shape, &no_dense, x.get(),
                static_cast<const float*>(layer->expert_scale.get()),
                static_cast<const float*>(layer->router_norm.get()),
                kRouterMultiplier,
                static_cast<const float*>(layer->expert_norm.get()), kEps, 1,
                &block, workspace.get(), y.get(), nullptr),
            cudaErrorInvalidValue);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

// One launch for the rows of every slot against a launch per slot, on Gemma
// 4 26B-A4B's layer. A decode step runs its 30 layers one after the other,
// so a layer's weights are not in the device's cache when its turn comes:
// four layers with weights of their own are replayed in turn here, from
// CUDA graphs as a step replays them, and the time is a layer's.
TEST(MoeSlotRowsTest, TimesThe26bLayer) {
  const LayerShape s = {128, 8, 2816, 704, 2112, 1, 1};
  constexpr int kLayers = 4;
  std::vector<std::unique_ptr<Layer>> layers;
  for (int l = 0; l < kLayers; ++l) {
    layers.push_back(MakeLayer(s, /*compact=*/true, 1000 * (l + 1)));
    ASSERT_NE(layers.back(), nullptr);
  }
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  const size_t row_bytes = static_cast<size_t>(s.dim) * 2;
  const double mb = 1.0 / 1048576.0;
  std::printf(
      "a layer's router and dense feed-forward: %.1f MB; one expert: "
      "%.2f MB, %d a row\n",
      layers[0]->dense_bytes * mb, layers[0]->expert_bytes * mb, s.top_k);
  for (const bool alike : {false, true}) {
    for (const int rows : {1, 2, 3, 4, 8}) {
      if (alike && rows == 1) continue;
      // Every slot at a token of its own, or (alike) all at the same one:
      // the same experts for every row.
      const Device x =
          Upload(Rows(rows, s.dim, 7 * rows,
                      alike ? std::vector<int>(rows, 0) : std::vector<int>()));
      std::vector<Device> y, expected, workspace, row_workspace;
      for (int l = 0; l < kLayers; ++l) {
        y.push_back(Upload(nullptr, rows * row_bytes));
        expected.push_back(Upload(nullptr, rows * row_bytes));
        workspace.push_back(Upload(nullptr, WorkspaceBytes(s, rows)));
        row_workspace.push_back(Upload(nullptr, WorkspaceBytes(s, 1)));
        ASSERT_TRUE(y[l] && expected[l] && workspace[l] && row_workspace[l]);
      }
      ASSERT_NE(x, nullptr);
      float separate = 0.0f, one = 0.0f;
      ASSERT_TRUE(TimeTwoWays(
          stream,
          [&] {
            for (int l = 0; l < kLayers; ++l) {
              const cudaError_t status = LaunchEachRow(
                  *layers[l], rows, x.get(), row_workspace[l].get(),
                  expected[l].get(), stream);
              if (status != cudaSuccess) return status;
            }
            return cudaSuccess;
          },
          [&] {
            for (int l = 0; l < kLayers; ++l) {
              const cudaError_t status =
                  LaunchSlotRows(*layers[l], rows, x.get(), workspace[l].get(),
                                 y[l].get(), stream);
              if (status != cudaSuccess) return status;
            }
            return cudaSuccess;
          },
          100, &separate, &one));
      int distinct = 0;
      for (int l = 0; l < kLayers; ++l) {
        EXPECT_EQ(Download(y[l].get(), rows * row_bytes),
                  Download(expected[l].get(), rows * row_bytes))
            << "layer " << l;
        const std::vector<int32_t> routed =
            RoutedExperts(workspace[l].get(), rows * s.top_k);
        distinct += static_cast<int>(
            std::set<int32_t>(routed.begin(), routed.end()).size());
      }
      const double dense = layers[0]->dense_bytes * mb;
      const double expert = layers[0]->expert_bytes * mb;
      std::printf(
          "%d rows%s: %d launches %.1f us a layer, one launch %.1f us "
          "(%.2fx); read %.1f MB against %.1f MB (%.1f distinct experts a "
          "layer: %.1f MB if each were read once)\n",
          rows, alike ? " at the same token" : "", rows, separate / kLayers,
          one / kLayers, separate / one, dense + rows * s.top_k * expert,
          rows * (dense + s.top_k * expert),
          static_cast<double>(distinct) / kLayers,
          dense + expert * distinct / kLayers);
    }
  }
  cudaStreamDestroy(stream);
}

}  // namespace
