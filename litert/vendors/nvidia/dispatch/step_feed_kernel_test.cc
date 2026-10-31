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

// The step feed kernels against LiteRT-LM's CPU arithmetic: the embedder's
// dequantize-then-multiply in float32, the ring-buffer decode mask, the
// position and cache parameter.

#include "litert/vendors/nvidia/dispatch/step_feed_kernel.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"

namespace {

struct CudaDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};
using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation AllocateCuda(size_t bytes) {
  void* pointer = nullptr;
  if (cudaMalloc(&pointer, bytes) != cudaSuccess) {
    ADD_FAILURE() << "cudaMalloc(" << bytes << ") failed";
  }
  return CudaAllocation(pointer);
}

struct HostDeleter {
  void operator()(void* pointer) const { cudaFreeHost(pointer); }
};
using HostAllocation = std::unique_ptr<void, HostDeleter>;

HostAllocation AllocateMapped(size_t bytes) {
  void* pointer = nullptr;
  if (cudaHostAlloc(&pointer, bytes, cudaHostAllocMapped) != cudaSuccess) {
    ADD_FAILURE() << "cudaHostAlloc(" << bytes << ") failed";
  }
  std::memset(pointer, 0, bytes);
  return HostAllocation(pointer);
}

// LiteRT-LM's FillRingBufferDecodeAttentionMask for one step.
std::vector<uint8_t> RingMask(int ring, int query_step, int window) {
  std::vector<uint8_t> mask(ring, 0);
  for (int col = 0; col < ring; ++col) {
    const int diff = query_step - col;
    const int distance = ((diff % ring) + ring) % ring;
    if (distance <= window - 1 && distance <= query_step &&
        distance <= ring - 1) {
      mask[col] = 1;
    }
  }
  return mask;
}

int DecodeInt4(const std::vector<uint8_t>& table, size_t index) {
  const int nibble = (table[index / 2] >> (4 * (index % 2))) & 15;
  return (nibble ^ 8) - 8;
}

// An FP16 value (a normal number) as float32: exact, as the CPU embedder's
// blockwise lookup converts a block's scale.
float HalfToFloat(uint16_t bits) {
  const int exponent = (bits >> 10) & 31;
  const float mantissa = 1.0f + static_cast<float>(bits & 1023) / 1024.0f;
  const float value = std::ldexp(mantissa, exponent - 15);
  return (bits & 0x8000) ? -value : value;
}

TEST(StepFeedKernelTest, FeedsTheNextStepLikeTheHost) {
  constexpr int kVocab = 300;
  constexpr int kDim = 3840;
  constexpr int kRing = 1152;
  constexpr int kWindow = 1024;
  constexpr int kGlobalKeys = 4096;
  const float multiplier = static_cast<float>(std::sqrt(3840.0));
  std::vector<uint8_t> table(static_cast<size_t>(kVocab) * kDim / 2);
  uint32_t state = 12345u;
  for (auto& byte : table) {
    state = state * 1664525u + 1013904223u;
    byte = static_cast<uint8_t>(state >> 24);
  }
  std::vector<float> scales(kVocab);
  for (int v = 0; v < kVocab; ++v) {
    scales[v] = 0.004f + 0.00002f * static_cast<float>(v % 97);
  }
  auto device_table = AllocateCuda(table.size());
  auto device_scales = AllocateCuda(scales.size() * sizeof(float));
  auto device_token = AllocateCuda(sizeof(int32_t));
  auto device_embeddings = AllocateCuda(kDim * sizeof(float));
  auto device_positions = AllocateCuda(sizeof(int32_t));
  auto device_param = AllocateCuda(7 * sizeof(int32_t));
  auto device_local = AllocateCuda(kRing);
  auto device_global = AllocateCuda(kGlobalKeys);
  auto device_counter = AllocateCuda(sizeof(unsigned int));
  ASSERT_EQ(cudaMemset(device_counter.get(), 0, sizeof(unsigned int)),
            cudaSuccess);
  auto mirror = AllocateMapped(sizeof(int32_t) + kDim * sizeof(float) +
                               sizeof(int32_t) + 7 * sizeof(int32_t) + kRing);
  auto* token_mirror = static_cast<int32_t*>(mirror.get());
  auto* embeddings_mirror = reinterpret_cast<float*>(token_mirror + 1);
  auto* positions_mirror = reinterpret_cast<int32_t*>(embeddings_mirror + kDim);
  auto* param_mirror = positions_mirror + 1;
  auto* local_mirror = reinterpret_cast<uint8_t*>(param_mirror + 7);
  ASSERT_EQ(cudaMemcpy(device_table.get(), table.data(), table.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_scales.get(), scales.data(),
                       scales.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);

  // A few positions: before the window fills, at its edge, past the ring.
  for (const int position : {0, 5, 1022, 1023, 1151, 1152, 3000}) {
    for (const int32_t token : {0, 7, kVocab - 1}) {
      const int32_t param[7] = {position, position + 1, position + 1, 0, 0, 0,
                                0};
      std::vector<uint8_t> global(kGlobalKeys, 0);
      for (int k = 0; k <= position && k < kGlobalKeys; ++k) global[k] = 1;
      ASSERT_EQ(cudaMemcpy(device_token.get(), &token, sizeof(token),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(device_positions.get(), &position, sizeof(position),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(device_param.get(), param, sizeof(param),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemset(device_local.get(), 0xAA, kRing), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(device_global.get(), global.data(), kGlobalKeys,
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      LiteRtNvidiaStepFeedParams params{};
      params.token = static_cast<const int32_t*>(device_token.get());
      params.table = static_cast<const uint8_t*>(device_table.get());
      params.bits = 4;
      params.row_scales = static_cast<const float*>(device_scales.get());
      params.multiplier = multiplier;
      params.vocab = kVocab;
      params.dim = kDim;
      params.embeddings = static_cast<float*>(device_embeddings.get());
      params.positions = static_cast<int32_t*>(device_positions.get());
      params.param = static_cast<int32_t*>(device_param.get());
      params.local_mask = static_cast<uint8_t*>(device_local.get());
      params.ring = kRing;
      params.window = kWindow;
      params.global_mask = static_cast<uint8_t*>(device_global.get());
      params.global_keys = kGlobalKeys;
      params.blocks_done = static_cast<unsigned int*>(device_counter.get());
      params.token_mirror = token_mirror;
      params.embeddings_mirror = embeddings_mirror;
      params.positions_mirror = positions_mirror;
      params.param_mirror = param_mirror;
      params.param_count = 7;
      params.local_mask_mirror = local_mirror;
      ASSERT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

      std::vector<float> embeddings(kDim);
      std::vector<uint8_t> local(kRing);
      std::vector<uint8_t> global_after(kGlobalKeys);
      int32_t positions_after = -1;
      int32_t param_after[7];
      ASSERT_EQ(cudaMemcpy(embeddings.data(), device_embeddings.get(),
                           kDim * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(local.data(), device_local.get(), kRing,
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(global_after.data(), device_global.get(),
                           kGlobalKeys, cudaMemcpyDeviceToHost),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(&positions_after, device_positions.get(),
                           sizeof(int32_t), cudaMemcpyDeviceToHost),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(param_after, device_param.get(), sizeof(param_after),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      // The embedding: the CPU embedder's two float32 roundings, bit for bit.
      for (int k = 0; k < kDim; ++k) {
        const float dequantized =
            static_cast<float>(
                DecodeInt4(table, static_cast<size_t>(token) * kDim + k)) *
            scales[token];
        const float expected = dequantized * multiplier;
        ASSERT_EQ(embeddings[k], expected) << "token " << token << " dim " << k;
        ASSERT_EQ(embeddings_mirror[k], expected);
      }
      EXPECT_EQ(token_mirror[0], token);
      // The next position's inputs.
      const int next = position + 1;
      EXPECT_EQ(positions_after, next);
      EXPECT_EQ(positions_mirror[0], next);
      EXPECT_EQ(param_after[0], next);
      EXPECT_EQ(param_after[1], next + 1);
      EXPECT_EQ(param_after[2], next + 1);
      for (int i = 3; i < 7; ++i) EXPECT_EQ(param_after[i], 0);
      for (int i = 0; i < 7; ++i) EXPECT_EQ(param_mirror[i], param_after[i]);
      const std::vector<uint8_t> expected_local =
          RingMask(kRing, next, kWindow);
      EXPECT_EQ(local, expected_local) << "position " << position;
      EXPECT_EQ(std::vector<uint8_t>(local_mirror, local_mirror + kRing),
                expected_local);
      for (int k = 0; k < kGlobalKeys; ++k) {
        EXPECT_EQ(global_after[k], k <= next ? 1 : 0) << "key " << k;
      }
    }
  }
}

TEST(StepFeedKernelTest, OutOfRangeTokenGivesZeros) {
  constexpr int kVocab = 4;
  constexpr int kDim = 8;
  std::vector<uint8_t> table(kVocab * kDim / 2, 0x75);
  std::vector<float> scales(kVocab, 0.5f);
  auto device_table = AllocateCuda(table.size());
  auto device_scales = AllocateCuda(scales.size() * sizeof(float));
  auto device_token = AllocateCuda(sizeof(int32_t));
  auto device_embeddings = AllocateCuda(kDim * sizeof(float));
  auto device_positions = AllocateCuda(sizeof(int32_t));
  auto device_counter = AllocateCuda(sizeof(unsigned int));
  ASSERT_EQ(cudaMemset(device_counter.get(), 0, sizeof(unsigned int)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_table.get(), table.data(), table.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_scales.get(), scales.data(),
                       scales.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  const int32_t token = kVocab;
  const int32_t position = 3;
  ASSERT_EQ(cudaMemcpy(device_token.get(), &token, sizeof(token),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_positions.get(), &position, sizeof(position),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(device_embeddings.get(), 0xFF, kDim * sizeof(float)),
            cudaSuccess);
  LiteRtNvidiaStepFeedParams params{};
  params.token = static_cast<const int32_t*>(device_token.get());
  params.table = static_cast<const uint8_t*>(device_table.get());
  params.bits = 4;
  params.row_scales = static_cast<const float*>(device_scales.get());
  params.multiplier = 2.0f;
  params.vocab = kVocab;
  params.dim = kDim;
  params.embeddings = static_cast<float*>(device_embeddings.get());
  params.positions = static_cast<int32_t*>(device_positions.get());
  params.blocks_done = static_cast<unsigned int*>(device_counter.get());
  ASSERT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> embeddings(kDim);
  int32_t positions_after = 0;
  ASSERT_EQ(cudaMemcpy(embeddings.data(), device_embeddings.get(),
                       kDim * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&positions_after, device_positions.get(),
                       sizeof(int32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (float value : embeddings) EXPECT_EQ(value, 0.0f);
  EXPECT_EQ(positions_after, 4);
  // Invalid parameters are refused.
  params.dim = 7;
  EXPECT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr),
            cudaErrorInvalidValue);
  params.dim = kDim;
  params.local_mask = static_cast<uint8_t*>(device_embeddings.get());
  params.ring = 8;
  params.window = 9;
  EXPECT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr),
            cudaErrorInvalidValue);
  params.local_mask = nullptr;
  // One kind of scales, and blocks that divide the row.
  params.block_scales = reinterpret_cast<const uint16_t*>(device_scales.get());
  params.scale_block = 4;
  EXPECT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr),
            cudaErrorInvalidValue);
  params.row_scales = nullptr;
  params.scale_block = 3;
  EXPECT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr),
            cudaErrorInvalidValue);
  params.block_scales = nullptr;
  EXPECT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr),
            cudaErrorInvalidValue);
  params.row_scales = static_cast<const float*>(device_scales.get());
  params.bits = 3;  // 4 or 2
  EXPECT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr),
            cudaErrorInvalidValue);
  params.bits = 4;
  params.per_layer =
      static_cast<float*>(device_embeddings.get());  // without its tables
  EXPECT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr),
            cudaErrorInvalidValue);
}

int DecodeInt2(const std::vector<uint8_t>& table, size_t index) {
  const int pair = (table[index / 4] >> (2 * (index % 4))) & 3;
  return (pair ^ 2) - 2;
}

TEST(StepFeedKernelTest, FeedsInt2AndPerLayerEmbeddingsLikeTheHost) {
  // The Gemma E models: an INT2 text table, and per-layer tables with their
  // own row scales, concatenated, no multiplier.
  constexpr int kVocab = 40;
  constexpr int kDim = 16;
  constexpr int kLayers = 3;
  constexpr int kWidth = 8;
  const float multiplier = 4.0f;
  std::vector<uint8_t> table(static_cast<size_t>(kVocab) * kDim / 4);
  std::vector<uint8_t> per_layer_table(static_cast<size_t>(kLayers) * kVocab *
                                       kWidth / 4);
  uint32_t state = 99u;
  for (auto* bytes : {&table, &per_layer_table}) {
    for (auto& byte : *bytes) {
      state = state * 1664525u + 1013904223u;
      byte = static_cast<uint8_t>(state >> 24);
    }
  }
  std::vector<float> scales(kVocab);
  std::vector<float> per_layer_scales(static_cast<size_t>(kLayers) * kVocab);
  for (int v = 0; v < kVocab; ++v)
    scales[v] = 0.01f + 0.001f * static_cast<float>(v);
  for (size_t i = 0; i < per_layer_scales.size(); ++i) {
    per_layer_scales[i] = 0.02f + 0.0005f * static_cast<float>(i % 13);
  }
  auto device_table = AllocateCuda(table.size());
  auto device_scales = AllocateCuda(scales.size() * sizeof(float));
  auto device_per_layer_table = AllocateCuda(per_layer_table.size());
  auto device_per_layer_scales =
      AllocateCuda(per_layer_scales.size() * sizeof(float));
  auto device_token = AllocateCuda(sizeof(int32_t));
  auto device_embeddings = AllocateCuda(kDim * sizeof(float));
  auto device_per_layer = AllocateCuda(kLayers * kWidth * sizeof(float));
  auto device_positions = AllocateCuda(sizeof(int32_t));
  auto device_counter = AllocateCuda(sizeof(unsigned int));
  auto mirror = AllocateMapped(kLayers * kWidth * sizeof(float));
  auto* per_layer_mirror = static_cast<float*>(mirror.get());
  ASSERT_EQ(cudaMemset(device_counter.get(), 0, sizeof(unsigned int)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_table.get(), table.data(), table.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_scales.get(), scales.data(),
                       scales.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_per_layer_table.get(), per_layer_table.data(),
                       per_layer_table.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_per_layer_scales.get(), per_layer_scales.data(),
                       per_layer_scales.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  for (const int32_t token : {0, 11, kVocab - 1}) {
    const int32_t position = 7;
    ASSERT_EQ(cudaMemcpy(device_token.get(), &token, sizeof(token),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_positions.get(), &position, sizeof(position),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    LiteRtNvidiaStepFeedParams params{};
    params.token = static_cast<const int32_t*>(device_token.get());
    params.table = static_cast<const uint8_t*>(device_table.get());
    params.bits = 2;
    params.row_scales = static_cast<const float*>(device_scales.get());
    params.multiplier = multiplier;
    params.vocab = kVocab;
    params.dim = kDim;
    params.embeddings = static_cast<float*>(device_embeddings.get());
    params.positions = static_cast<int32_t*>(device_positions.get());
    params.per_layer = static_cast<float*>(device_per_layer.get());
    params.per_layer_table =
        static_cast<const uint8_t*>(device_per_layer_table.get());
    params.per_layer_scales =
        static_cast<const float*>(device_per_layer_scales.get());
    params.per_layer_count = kLayers;
    params.per_layer_width = kWidth;
    params.per_layer_bits = 2;
    params.per_layer_multiplier = 1.0f;
    params.per_layer_mirror = per_layer_mirror;
    params.blocks_done = static_cast<unsigned int*>(device_counter.get());
    ASSERT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> embeddings(kDim);
    std::vector<float> per_layer(kLayers * kWidth);
    ASSERT_EQ(cudaMemcpy(embeddings.data(), device_embeddings.get(),
                         kDim * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(per_layer.data(), device_per_layer.get(),
                   per_layer.size() * sizeof(float), cudaMemcpyDeviceToHost),
        cudaSuccess);
    // The INT2 unpack of the CPU embedder: (q * 64) * (scale / 64), which is
    // q * scale rounded once, then the multiplier.
    for (int k = 0; k < kDim; ++k) {
      const float dequantized =
          static_cast<float>(
              DecodeInt2(table, static_cast<size_t>(token) * kDim + k)) *
          scales[token];
      ASSERT_EQ(embeddings[k], dequantized * multiplier) << "dim " << k;
    }
    for (int layer = 0; layer < kLayers; ++layer) {
      for (int c = 0; c < kWidth; ++c) {
        const size_t index =
            (static_cast<size_t>(layer) * kVocab + token) * kWidth + c;
        const float expected =
            static_cast<float>(DecodeInt2(per_layer_table, index)) *
            per_layer_scales[static_cast<size_t>(layer) * kVocab + token];
        ASSERT_EQ(per_layer[layer * kWidth + c], expected)
            << "layer " << layer << " column " << c;
        ASSERT_EQ(per_layer_mirror[layer * kWidth + c], expected);
      }
    }
  }
}

TEST(StepFeedKernelTest, BlockScalesFeedLikeTheHostsBlockwiseLookup) {
  constexpr int kVocab = 50;
  constexpr int kDim = 2816;
  constexpr int kBlock = 32;
  constexpr int kBlocks = kDim / kBlock;
  const float multiplier = static_cast<float>(std::sqrt(2816.0));
  std::vector<uint8_t> table(static_cast<size_t>(kVocab) * kDim / 2);
  uint32_t state = 777u;
  for (auto& byte : table) {
    state = state * 1664525u + 1013904223u;
    byte = static_cast<uint8_t>(state >> 24);
  }
  // Normal FP16 scales between 2^-7 and 2^-4, by their bits.
  std::vector<uint16_t> scales(static_cast<size_t>(kVocab) * kBlocks);
  for (auto& bits : scales) {
    state = state * 1664525u + 1013904223u;
    bits = static_cast<uint16_t>(0x2000 | ((state >> 8) & 0x0FFF));
  }
  auto device_table = AllocateCuda(table.size());
  auto device_scales = AllocateCuda(scales.size() * sizeof(uint16_t));
  auto device_token = AllocateCuda(sizeof(int32_t));
  auto device_embeddings = AllocateCuda(kDim * sizeof(float));
  auto device_positions = AllocateCuda(sizeof(int32_t));
  auto device_counter = AllocateCuda(sizeof(unsigned int));
  ASSERT_EQ(cudaMemset(device_counter.get(), 0, sizeof(unsigned int)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_table.get(), table.data(), table.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(device_scales.get(), scales.data(),
                 scales.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
      cudaSuccess);
  for (const int32_t token : {0, 17, kVocab - 1}) {
    const int32_t position = 40;
    ASSERT_EQ(cudaMemcpy(device_token.get(), &token, sizeof(token),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_positions.get(), &position, sizeof(position),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    LiteRtNvidiaStepFeedParams params{};
    params.token = static_cast<const int32_t*>(device_token.get());
    params.table = static_cast<const uint8_t*>(device_table.get());
    params.bits = 4;
    params.block_scales = static_cast<const uint16_t*>(device_scales.get());
    params.scale_block = kBlock;
    params.multiplier = multiplier;
    params.vocab = kVocab;
    params.dim = kDim;
    params.embeddings = static_cast<float*>(device_embeddings.get());
    params.positions = static_cast<int32_t*>(device_positions.get());
    params.blocks_done = static_cast<unsigned int*>(device_counter.get());
    ASSERT_EQ(LiteRtNvidiaLaunchStepFeed(&params, nullptr), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> embeddings(kDim);
    ASSERT_EQ(cudaMemcpy(embeddings.data(), device_embeddings.get(),
                         kDim * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    // The block's scale as float32, the dequantize, the multiply: the CPU
    // embedder's roundings, bit for bit.
    for (int k = 0; k < kDim; ++k) {
      const float scale = HalfToFloat(
          scales[static_cast<size_t>(token) * kBlocks + k / kBlock]);
      const float dequantized =
          static_cast<float>(
              DecodeInt4(table, static_cast<size_t>(token) * kDim + k)) *
          scale;
      const float expected = dequantized * multiplier;
      ASSERT_EQ(embeddings[k], expected) << "token " << token << " dim " << k;
    }
    int32_t positions_after = 0;
    ASSERT_EQ(cudaMemcpy(&positions_after, device_positions.get(),
                         sizeof(int32_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(positions_after, position + 1);
  }
}

}  // namespace
