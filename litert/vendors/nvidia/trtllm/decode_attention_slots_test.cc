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

// The decode attention over several sequence slots in one launch
// (decode_attention_slots.h) against the single-sequence launches
// (decode_attention.h) run on each slot alone: every slot's output must be
// theirs byte for byte, for 1 to 8 slots at different positions, over the
// shapes and cache types of Gemma 4's decode steps. A second test times one
// launch against the slots' separate launches on the 26B's shapes.

#include "litert/vendors/nvidia/trtllm/decode_attention_slots.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"

namespace {

struct CudaDeleter {
  void operator()(void* p) const { cudaFree(p); }
};
using Device = std::unique_ptr<void, CudaDeleter>;

Device Upload(const void* host, size_t bytes) {
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
  if (host != nullptr) {
    cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice);
  } else {
    cudaMemset(p, 0, bytes);
  }
  return Device(p);
}

// A reproducible stream of bits.
class Random {
 public:
  explicit Random(uint64_t seed) : state_(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint32_t Next() {
    state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(state_ >> 33);
  }

 private:
  uint64_t state_;
};

// Finite FP16 values of magnitude 2^-5 to 2, either sign.
std::vector<uint16_t> RandomHalves(size_t count, uint64_t seed) {
  Random random(seed);
  std::vector<uint16_t> values(count);
  for (uint16_t& v : values) {
    const uint32_t bits = random.Next();
    v = static_cast<uint16_t>(((bits & 1) << 15) |
                              ((10 + (bits >> 1) % 6) << 10) |
                              ((bits >> 8) & 0x3ff));
  }
  return values;
}

// Finite BF16 values of magnitude 2^-5 to 2, either sign.
std::vector<uint16_t> RandomBf16(size_t count, uint64_t seed) {
  Random random(seed);
  std::vector<uint16_t> values(count);
  for (uint16_t& v : values) {
    const uint32_t bits = random.Next();
    v = static_cast<uint16_t>(((bits & 1) << 15) |
                              ((122 + (bits >> 1) % 6) << 7) |
                              ((bits >> 8) & 0x7f));
  }
  return values;
}

std::vector<int8_t> RandomBytes(size_t count, uint64_t seed) {
  Random random(seed);
  std::vector<int8_t> values(count);
  for (int8_t& v : values) v = static_cast<int8_t>(random.Next() >> 3);
  return values;
}

struct Config {
  const char* name;
  bool q_bf16;
  bool int8;     // INT8 caches (per-tensor quantized), else FP16
  bool columns;  // INT8 value caches stored [heads, depth, cache_len]
  int heads, rows, depth, cache_len;
  int window;     // INT8 only: keys read from each slot's own window start
  int mask_rows;  // 1 or rows
  bool ring;      // a ring cache: slots past the wrap see every key
};

// One slot's buffers on the device, and where the sequence stands.
struct Slot {
  Device q, k, v, mask, window_start, single, batched;
  int position = 0;
};

constexpr float kFill = -10000.0f;
const LiteRtNvidiaInt8CacheParams kInt8Rows = {0.0125f, -3, 0.02f, 5, false};
const LiteRtNvidiaInt8CacheParams kInt8Columns = {0.0125f, -3, 0.02f, 5, true};

// The slots of a step: slot i at its own position (so many keys visible to
// its rows; a ring cache past its wrap shows every key; with mask_rows ==
// rows, row r sees r keys fewer, as a verify step's rows do), its own
// window start where the launch reads a window.
std::vector<Slot> MakeSlots(const Config& c, int slots, uint64_t seed) {
  std::vector<Slot> result(slots);
  const size_t q_elems = static_cast<size_t>(c.heads) * c.rows * c.depth;
  const size_t cache_elems =
      static_cast<size_t>(c.heads) * c.cache_len * c.depth;
  for (int i = 0; i < slots; ++i) {
    Slot& slot = result[i];
    const uint64_t s = seed * 1000 + i * 10;
    const auto q = c.q_bf16 ? RandomBf16(q_elems, s) : RandomHalves(q_elems, s);
    slot.q = Upload(q.data(), q_elems * 2);
    if (c.int8) {
      const auto k = RandomBytes(cache_elems, s + 1);
      const auto v = RandomBytes(cache_elems, s + 2);
      slot.k = Upload(k.data(), cache_elems);
      slot.v = Upload(v.data(), cache_elems);
    } else {
      const auto k = RandomHalves(cache_elems, s + 1);
      const auto v = RandomHalves(cache_elems, s + 2);
      slot.k = Upload(k.data(), cache_elems * 2);
      slot.v = Upload(v.data(), cache_elems * 2);
    }
    // Slot 0 is a few tokens in, the last slot at the end of the cache (a
    // ring: wrapped), the others in between.
    slot.position = i == 0
                        ? std::min(5, c.cache_len)
                        : static_cast<int>(static_cast<int64_t>(c.cache_len) *
                                           (i + 1) / slots);
    std::vector<uint8_t> mask(static_cast<size_t>(c.mask_rows) * c.cache_len,
                              0);
    for (int r = 0; r < c.mask_rows; ++r) {
      const bool wrapped = c.ring && slot.position >= c.cache_len;
      const int visible =
          wrapped ? c.cache_len
                  : std::max(1, slot.position - (c.mask_rows - 1 - r));
      std::fill_n(mask.begin() + static_cast<size_t>(r) * c.cache_len,
                  std::min(visible, c.cache_len), 1);
    }
    slot.mask = Upload(mask.data(), mask.size());
    if (c.window != c.cache_len) {
      // A window that ends at the slot's position, on a multiple of 16.
      const int32_t start = std::max(0, std::min(c.cache_len - c.window,
                                                 slot.position - c.window)) /
                            16 * 16;
      slot.window_start = Upload(&start, sizeof(start));
    }
    slot.single = Upload(nullptr, q_elems * 2);
    slot.batched = Upload(nullptr, q_elems * 2);
    EXPECT_TRUE(slot.q && slot.k && slot.v && slot.mask && slot.single &&
                slot.batched);
  }
  return result;
}

cudaError_t LaunchSingle(const Config& c, const Slot& slot, void* workspace,
                         cudaStream_t stream) {
  const bool* mask = static_cast<const bool*>(slot.mask.get());
  if (c.int8) {
    return LiteRtNvidiaLaunchDecodeAttentionInt8Window(
        slot.q.get(), c.q_bf16, slot.k.get(), slot.v.get(),
        c.columns ? &kInt8Columns : &kInt8Rows, mask, c.mask_rows, c.heads,
        c.rows, c.cache_len, c.window,
        static_cast<const int32_t*>(slot.window_start.get()), c.depth, kFill,
        slot.single.get(), workspace, stream);
  }
  return LiteRtNvidiaLaunchDecodeAttention(
      slot.q.get(), c.q_bf16, slot.k.get(), slot.v.get(), mask, c.mask_rows,
      c.heads, c.rows, c.cache_len, c.depth, kFill, slot.single.get(),
      workspace, stream);
}

cudaError_t LaunchSlots(const Config& c, const std::vector<Slot>& slots,
                        void* workspace, cudaStream_t stream) {
  std::vector<LiteRtNvidiaDecodeAttentionSlot> table(slots.size());
  for (size_t i = 0; i < slots.size(); ++i) {
    table[i] = {slots[i].q.get(),
                slots[i].k.get(),
                slots[i].v.get(),
                static_cast<const bool*>(slots[i].mask.get()),
                static_cast<const int32_t*>(slots[i].window_start.get()),
                slots[i].batched.get()};
  }
  return LiteRtNvidiaLaunchDecodeAttentionSlots(
      table.data(), static_cast<int32_t>(table.size()), c.q_bf16,
      c.int8 ? (c.columns ? &kInt8Columns : &kInt8Rows) : nullptr, c.mask_rows,
      c.heads, c.rows, c.cache_len, c.window, c.depth, kFill, workspace,
      stream);
}

// The decode steps' attention shapes: Gemma 4 26B-A4B and 12B (FP16 caches;
// sliding layers over a ring of 1,024 keys, global layers over the whole
// context, short and long), E4B and E2B (INT8 caches, the value cache as
// the model stores it or as rows, the local layers through a window), and
// a verify step's rows with a mask row each.
const Config kConfigs[] = {
    {"26b local", true, false, false, 8, 2, 256, 1024, 1024, 1, true},
    {"26b global 2k", true, false, false, 2, 8, 512, 2048, 2048, 1, false},
    {"26b global 9k", true, false, false, 2, 8, 512, 9216, 9216, 1, false},
    // Either side of the length where many rows' blocks go from 32 to 64 keys.
    {"global 2,688", true, false, false, 2, 8, 512, 2688, 2688, 1, false},
    {"global 2,720", true, false, false, 2, 8, 512, 2720, 2720, 1, false},
    {"12b local", true, false, false, 4, 2, 256, 1024, 1024, 1, true},
    {"12b global 9k", true, false, false, 1, 8, 512, 9216, 9216, 1, false},
    {"fp16 queries", false, false, false, 4, 4, 128, 512, 512, 1, true},
    {"verify rows", true, false, false, 8, 10, 256, 1024, 1024, 10, false},
    {"e4b local window", true, true, true, 2, 4, 256, 2048, 512, 1, false},
    {"e4b global", true, true, true, 1, 8, 512, 2048, 2048, 1, false},
    {"e2b global 9k", true, true, true, 1, 8, 512, 9216, 9216, 1, false},
    {"int8 value rows", true, true, false, 2, 4, 256, 1024, 1024, 1, true},
    {"int8 rows window", false, true, false, 4, 16, 128, 1024, 256, 16, false},
};

TEST(DecodeAttentionSlotsTest, EverySlotMatchesItsOwnLaunch) {
  for (const Config& c : kConfigs) {
    const size_t out_bytes =
        static_cast<size_t>(c.heads) * c.rows * c.depth * 2;
    for (int slots = 1; slots <= kLiteRtNvidiaDecodeAttentionMaxSlots;
         ++slots) {
      SCOPED_TRACE(std::string(c.name) + ", " + std::to_string(slots) +
                   " slots");
      const std::vector<Slot> step = MakeSlots(c, slots, 7 + slots);
      Device single_ws =
          Upload(nullptr, LiteRtNvidiaDecodeAttentionWorkspaceBytes(
                              c.heads, c.rows, c.window, c.depth));
      Device slots_ws =
          Upload(nullptr, LiteRtNvidiaDecodeAttentionSlotsWorkspaceBytes(
                              slots, c.heads, c.rows, c.window, c.depth));
      ASSERT_TRUE(single_ws && slots_ws);
      for (const Slot& slot : step) {
        ASSERT_EQ(LaunchSingle(c, slot, single_ws.get(), nullptr), cudaSuccess);
      }
      ASSERT_EQ(LaunchSlots(c, step, slots_ws.get(), nullptr), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::vector<uint8_t> single(out_bytes), batched(out_bytes);
      for (int i = 0; i < slots; ++i) {
        ASSERT_EQ(cudaMemcpy(single.data(), step[i].single.get(), out_bytes,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(batched.data(), step[i].batched.get(), out_bytes,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        // An output that is all zeros would compare equal without a launch.
        EXPECT_TRUE(std::any_of(single.begin(), single.end(),
                                [](uint8_t b) { return b != 0; }));
        EXPECT_EQ(std::memcmp(single.data(), batched.data(), out_bytes), 0)
            << "slot " << i << " at position " << step[i].position;
      }
    }
  }
}

TEST(DecodeAttentionSlotsTest, RejectsInvalidArguments) {
  const Config& c = kConfigs[0];
  const std::vector<Slot> step = MakeSlots(c, 2, 3);
  Device ws = Upload(nullptr, LiteRtNvidiaDecodeAttentionSlotsWorkspaceBytes(
                                  2, c.heads, c.rows, c.window, c.depth));
  ASSERT_TRUE(ws);
  LiteRtNvidiaDecodeAttentionSlot table[9] = {};
  for (int i = 0; i < 9; ++i) {
    const Slot& slot = step[i % 2];
    table[i] = {slot.q.get(), slot.k.get(),
                slot.v.get(), static_cast<const bool*>(slot.mask.get()),
                nullptr,      slot.batched.get()};
  }
  auto launch = [&](int32_t slots, int32_t rows, int32_t window,
                    int32_t depth) {
    return LiteRtNvidiaLaunchDecodeAttentionSlots(
        table, slots, true, nullptr, 1, c.heads, rows, c.cache_len, window,
        depth, kFill, ws.get(), nullptr);
  };
  EXPECT_EQ(launch(0, c.rows, c.cache_len, c.depth), cudaErrorInvalidValue);
  EXPECT_EQ(launch(9, c.rows, c.cache_len, c.depth), cudaErrorInvalidValue);
  EXPECT_EQ(launch(2, 17, c.cache_len, c.depth), cudaErrorInvalidValue);
  EXPECT_EQ(launch(2, c.rows, c.cache_len, 384), cudaErrorInvalidValue);
  // FP16 caches have no window.
  EXPECT_EQ(launch(2, c.rows, c.cache_len / 2, c.depth), cudaErrorInvalidValue);
  table[1].out = nullptr;
  EXPECT_EQ(launch(2, c.rows, c.cache_len, c.depth), cudaErrorInvalidValue);
}

// One launch over R slots against the slots' R launches, each way captured
// as a CUDA graph (as a decode step runs them) and replayed in turn.
TEST(DecodeAttentionSlotsTest, TimesThe26bShapes) {
  struct Shape {
    const char* name;
    int heads, rows, depth, cache_len, visible;
    bool ring;
  };
  const Shape shapes[] = {
      {"local, ring of 1,024 keys", 8, 2, 256, 1024, 1024, true},
      {"global, 1,024 of 2,048 keys", 2, 8, 512, 2048, 1024, false},
      {"global, 8,192 of 9,216 keys", 2, 8, 512, 9216, 8192, false},
  };
  constexpr int kReplays = 200;
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  cudaEvent_t begin = nullptr, end = nullptr;
  ASSERT_EQ(cudaEventCreate(&begin), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);
  for (const Shape& shape : shapes) {
    const Config c = {shape.name,      true,       false,       false,
                      shape.heads,     shape.rows, shape.depth, shape.cache_len,
                      shape.cache_len, 1,          shape.ring};
    const double slot_mb =
        2.0 * shape.heads * shape.visible * shape.depth * 2 / 1048576.0;
    for (const int slots : {1, 2, 3, 4, 8}) {
      std::vector<Slot> step = MakeSlots(c, slots, 11);
      // Every slot at the shape's context.
      std::vector<uint8_t> mask(shape.cache_len, 0);
      std::fill_n(mask.begin(), shape.visible, 1);
      for (Slot& slot : step) {
        ASSERT_EQ(cudaMemcpy(slot.mask.get(), mask.data(), mask.size(),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
      }
      Device single_ws =
          Upload(nullptr, LiteRtNvidiaDecodeAttentionWorkspaceBytes(
                              c.heads, c.rows, c.window, c.depth));
      Device slots_ws =
          Upload(nullptr, LiteRtNvidiaDecodeAttentionSlotsWorkspaceBytes(
                              slots, c.heads, c.rows, c.window, c.depth));
      ASSERT_TRUE(single_ws && slots_ws);
      // Warm launches outside the captures (the kernels' attributes).
      for (const Slot& slot : step) {
        ASSERT_EQ(LaunchSingle(c, slot, single_ws.get(), stream), cudaSuccess);
      }
      ASSERT_EQ(LaunchSlots(c, step, slots_ws.get(), stream), cudaSuccess);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      cudaGraphExec_t graphs[2] = {};
      for (int way = 0; way < 2; ++way) {
        ASSERT_EQ(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
        if (way == 0) {
          for (const Slot& slot : step) {
            ASSERT_EQ(LaunchSingle(c, slot, single_ws.get(), stream),
                      cudaSuccess);
          }
        } else {
          ASSERT_EQ(LaunchSlots(c, step, slots_ws.get(), stream), cudaSuccess);
        }
        cudaGraph_t graph = nullptr;
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
        ASSERT_EQ(cudaGraphInstantiate(&graphs[way], graph, 0), cudaSuccess);
        cudaGraphDestroy(graph);
      }
      std::vector<float> times[2];
      // The first replays warm the device up and are not counted.
      for (int i = -kReplays / 4; i < kReplays; ++i) {
        for (int way = 0; way < 2; ++way) {
          ASSERT_EQ(cudaEventRecord(begin, stream), cudaSuccess);
          ASSERT_EQ(cudaGraphLaunch(graphs[way], stream), cudaSuccess);
          ASSERT_EQ(cudaEventRecord(end, stream), cudaSuccess);
          ASSERT_EQ(cudaEventSynchronize(end), cudaSuccess);
          float ms = 0.0f;
          ASSERT_EQ(cudaEventElapsedTime(&ms, begin, end), cudaSuccess);
          if (i >= 0) times[way].push_back(ms * 1000.0f);
        }
      }
      for (auto& t : times) std::sort(t.begin(), t.end());
      const float separate = times[0][kReplays / 2];
      const float one = times[1][kReplays / 2];
      std::printf(
          "%s, %d slots: %d launches %.1f us, one launch %.1f us (%.2fx); "
          "%.1f MB of keys and values read either way\n",
          shape.name, slots, slots, separate, one, separate / one,
          slots * slot_mb);
      for (cudaGraphExec_t graph : graphs) cudaGraphExecDestroy(graph);
    }
  }
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaStreamDestroy(stream);
}

}  // namespace
