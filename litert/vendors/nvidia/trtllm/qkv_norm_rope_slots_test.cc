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

#include "litert/vendors/nvidia/trtllm/qkv_norm_rope_slots.h"

#include <cstdint>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/qkv_norm_rope.h"
#include "litert/vendors/nvidia/trtllm/slot_test_util.h"

namespace {

using litert::nvidia::slot_test::Bytes;
using litert::nvidia::slot_test::Device;
using litert::nvidia::slot_test::Download;
using litert::nvidia::slot_test::TimeTwoWays;
using litert::nvidia::slot_test::Upload;

// One slot's inputs and the outputs of both ways.
struct Slot {
  Device qkv, positions, q, k, v, single_q, single_k, single_v;
};

struct Step {
  LiteRtNvidiaQkvNormRopeParams params;
  Device q_scale, k_scale;
  std::vector<Slot> slots;
  size_t q_bytes, kv_bytes;
};

// `slots` slots of the shape, slot s at the positions first[s], first[s] + 1,
// ... of its own sequence.
Step MakeStep(const LiteRtNvidiaQkvNormRopeParams& params, int slots,
              const std::vector<int32_t>& first, uint32_t seed) {
  Step step;
  step.params = params;
  std::vector<float> q_scale(params.depth), k_scale(params.depth);
  for (int i = 0; i < params.depth; ++i) {
    q_scale[i] = 0.5f + 0.003f * static_cast<float>(i % 97);
    k_scale[i] = 1.25f - 0.002f * static_cast<float>(i % 89);
  }
  step.q_scale = Upload(q_scale);
  step.k_scale = Upload(k_scale);
  step.q_bytes =
      static_cast<size_t>(params.heads) * params.tokens * params.depth * 2;
  step.kv_bytes =
      static_cast<size_t>(params.kv_heads) * params.tokens * params.depth * 2;
  const size_t qkv_bytes = step.q_bytes + 2 * step.kv_bytes;
  for (int s = 0; s < slots; ++s) {
    Slot slot;
    slot.qkv = Upload(Bytes(qkv_bytes, seed + 7 * s, /*floats=*/true));
    std::vector<int32_t> positions(params.tokens);
    for (int t = 0; t < params.tokens; ++t) positions[t] = first[s] + t;
    slot.positions = Upload(positions);
    for (Device* out : {&slot.q, &slot.single_q}) {
      *out = Upload(nullptr, step.q_bytes);
    }
    for (Device* out : {&slot.k, &slot.v, &slot.single_k, &slot.single_v}) {
      *out = Upload(nullptr, step.kv_bytes);
    }
    step.slots.push_back(std::move(slot));
  }
  return step;
}

cudaError_t LaunchSingle(const Step& step, const Slot& slot,
                         cudaStream_t stream) {
  return LiteRtNvidiaLaunchQkvNormRope(
      &step.params, slot.qkv.get(),
      static_cast<const int32_t*>(slot.positions.get()),
      static_cast<const float*>(step.q_scale.get()),
      static_cast<const float*>(step.k_scale.get()), slot.single_q.get(),
      slot.single_k.get(), slot.single_v.get(), stream);
}

cudaError_t LaunchSlots(const Step& step, cudaStream_t stream) {
  std::vector<LiteRtNvidiaQkvNormRopeSlot> table;
  for (const Slot& slot : step.slots) {
    table.push_back({slot.qkv.get(),
                     static_cast<const int32_t*>(slot.positions.get()),
                     slot.q.get(), slot.k.get(), slot.v.get()});
  }
  return LiteRtNvidiaLaunchQkvNormRopeSlots(
      &step.params, table.data(), static_cast<int32_t>(table.size()),
      static_cast<const float*>(step.q_scale.get()),
      static_cast<const float*>(step.k_scale.get()), stream);
}

TEST(QkvNormRopeSlotsTest, EverySlotMatchesItsOwnLaunch) {
  struct Shape {
    const char* name;
    LiteRtNvidiaQkvNormRopeParams params;
  };
  // {tokens, heads, kv_heads, depth, rotary, has_v_norm, epsilon, min, max}
  const Shape shapes[] = {
      {"26b local", {1, 16, 8, 256, 256, 1, 1e-6f, 1.0f, 10000.0f}},
      {"26b global", {1, 16, 2, 512, 512, 1, 1e-6f, 1.0f, 10000.0f}},
      {"12b", {1, 8, 4, 256, 256, 1, 1e-6f, 1.0f, 10000.0f}},
      {"e4b", {1, 8, 2, 256, 256, 1, 1e-6f, 1.0f, 10000.0f}},
      {"e2b", {1, 8, 1, 256, 256, 0, 1e-6f, 1.0f, 10000.0f}},
      {"a verify step's rows", {5, 16, 2, 512, 512, 1, 1e-6f, 1.0f, 10000.0f}},
      {"proportional rotation", {1, 4, 2, 128, 32, 0, 1e-5f, 1.0f, 1.0e6f}},
  };
  // Short and long contexts, the same position twice, positions past a
  // local layer's ring.
  const std::vector<int32_t> first = {5, 8191, 0, 1024, 31999, 5, 2047, 300};
  for (const Shape& shape : shapes) {
    ASSERT_TRUE(LiteRtNvidiaQkvNormRopeSupports(&shape.params));
    for (int slots = 1; slots <= kLiteRtNvidiaQkvNormRopeMaxSlots; ++slots) {
      SCOPED_TRACE(::testing::Message()
                   << shape.name << ", " << slots << " slots");
      Step step = MakeStep(shape.params, slots, first, 100 * slots + 3);
      for (const Slot& slot : step.slots) {
        ASSERT_EQ(LaunchSingle(step, slot, nullptr), cudaSuccess);
      }
      ASSERT_EQ(LaunchSlots(step, nullptr), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      for (int s = 0; s < slots; ++s) {
        const Slot& slot = step.slots[s];
        const auto q = Download(slot.q.get(), step.q_bytes);
        ASSERT_FALSE(q.empty());
        EXPECT_NE(std::count(q.begin(), q.end(), 0),
                  static_cast<ptrdiff_t>(q.size()));
        EXPECT_EQ(q, Download(slot.single_q.get(), step.q_bytes))
            << "q of slot " << s;
        EXPECT_EQ(Download(slot.k.get(), step.kv_bytes),
                  Download(slot.single_k.get(), step.kv_bytes))
            << "k of slot " << s;
        EXPECT_EQ(Download(slot.v.get(), step.kv_bytes),
                  Download(slot.single_v.get(), step.kv_bytes))
            << "v of slot " << s;
      }
    }
  }
}

TEST(QkvNormRopeSlotsTest, RejectsInvalidArguments) {
  const LiteRtNvidiaQkvNormRopeParams params = {1, 16,    2,    512,     512,
                                                1, 1e-6f, 1.0f, 10000.0f};
  Step step = MakeStep(params, 2, {3, 9}, 1);
  std::vector<LiteRtNvidiaQkvNormRopeSlot> table;
  for (const Slot& slot : step.slots) {
    table.push_back({slot.qkv.get(),
                     static_cast<const int32_t*>(slot.positions.get()),
                     slot.q.get(), slot.k.get(), slot.v.get()});
  }
  const auto* q_scale = static_cast<const float*>(step.q_scale.get());
  const auto* k_scale = static_cast<const float*>(step.k_scale.get());
  EXPECT_EQ(LiteRtNvidiaLaunchQkvNormRopeSlots(&params, table.data(), 2,
                                               q_scale, k_scale, nullptr),
            cudaSuccess);
  EXPECT_EQ(LiteRtNvidiaLaunchQkvNormRopeSlots(&params, table.data(), 0,
                                               q_scale, k_scale, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchQkvNormRopeSlots(
                &params, table.data(), kLiteRtNvidiaQkvNormRopeMaxSlots + 1,
                q_scale, k_scale, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchQkvNormRopeSlots(&params, nullptr, 2, q_scale,
                                               k_scale, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchQkvNormRopeSlots(&params, table.data(), 2,
                                               nullptr, k_scale, nullptr),
            cudaErrorInvalidValue);
  table[1].positions = nullptr;
  EXPECT_EQ(LiteRtNvidiaLaunchQkvNormRopeSlots(&params, table.data(), 2,
                                               q_scale, k_scale, nullptr),
            cudaErrorInvalidValue);
  LiteRtNvidiaQkvNormRopeParams odd = params;
  odd.depth = 511;
  table[1].positions = table[0].positions;
  EXPECT_EQ(LiteRtNvidiaLaunchQkvNormRopeSlots(&odd, table.data(), 2, q_scale,
                                               k_scale, nullptr),
            cudaErrorInvalidValue);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

// One launch for every slot against a launch per slot, on the 26B's two
// attention shapes, replayed from CUDA graphs as a decode step replays them.
TEST(QkvNormRopeSlotsTest, TimesThe26bShapes) {
  struct Shape {
    const char* name;
    LiteRtNvidiaQkvNormRopeParams params;
  };
  const Shape shapes[] = {
      {"26b local", {1, 16, 8, 256, 256, 1, 1e-6f, 1.0f, 10000.0f}},
      {"26b global", {1, 16, 2, 512, 512, 1, 1e-6f, 1.0f, 10000.0f}},
  };
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  const std::vector<int32_t> first = {5, 8191, 0, 1024, 31999, 5, 2047, 300};
  for (const Shape& shape : shapes) {
    for (const int slots : {1, 2, 3, 4, 8}) {
      Step step = MakeStep(shape.params, slots, first, 17);
      float separate = 0.0f, one = 0.0f;
      ASSERT_TRUE(TimeTwoWays(
          stream,
          [&] {
            for (const Slot& slot : step.slots) {
              const cudaError_t status = LaunchSingle(step, slot, stream);
              if (status != cudaSuccess) return status;
            }
            return cudaSuccess;
          },
          [&] { return LaunchSlots(step, stream); }, 200, &separate, &one));
      std::printf(
          "%s, %d slots: %d launches %.1f us, one launch %.1f us "
          "(%.2fx)\n",
          shape.name, slots, slots, separate, one, separate / one);
    }
  }
  cudaStreamDestroy(stream);
}

}  // namespace
