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

#include "litert/vendors/nvidia/trtllm/cache_write_slots.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/compiler/int8_cache_write_kernel.h"
#include "litert/vendors/nvidia/trtllm/slot_test_util.h"

namespace {

using litert::nvidia::Int8CacheWriteSource;
using litert::nvidia::LaunchInt8CacheWrite;
using litert::nvidia::slot_test::Bytes;
using litert::nvidia::slot_test::Device;
using litert::nvidia::slot_test::Download;
using litert::nvidia::slot_test::TimeTwoWays;
using litert::nvidia::slot_test::Upload;

size_t SourceBytes(int32_t source) {
  return source == kLiteRtNvidiaCacheWriteFloat ? 4 : 2;
}

// Element i of an update as FP32.
float SourceValue(const std::vector<uint8_t>& update, int32_t source,
                  size_t i) {
  if (source == kLiteRtNvidiaCacheWriteFloat) {
    float value;
    std::memcpy(&value, update.data() + 4 * i, 4);
    return value;
  }
  uint16_t bits;
  std::memcpy(&bits, update.data() + 2 * i, 2);
  if (source == kLiteRtNvidiaCacheWriteBf16) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &wide, 4);
    return value;
  }
  const int exponent = (bits >> 10) & 0x1f;
  const int mantissa = bits & 0x3ff;
  const float magnitude =
      exponent == 0
          ? std::ldexp(static_cast<float>(mantissa), -24)
          : std::ldexp(static_cast<float>(mantissa + 1024), exponent - 25);
  return (bits & 0x8000) ? -magnitude : magnitude;
}

// FP32 to FP16 bits, round to nearest even (finite values).
uint16_t HalfBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const float magnitude = std::fabs(value);
  if (magnitude == 0.0f) return sign;
  int exponent;
  const float fraction = std::frexp(magnitude, &exponent);  // [0.5, 1)
  // The value in units of the FP16 spacing at its exponent (2^-24 for the
  // subnormals).
  const int unit = std::max(exponent - 11, -24);
  const float scaled = std::ldexp(fraction, exponent - unit);
  float rounded = std::nearbyint(scaled);  // the default mode: to even
  const uint32_t steps = static_cast<uint32_t>(rounded);
  if (unit == -24 && steps < 1024) return sign | static_cast<uint16_t>(steps);
  // steps in [1024, 2048]: 2048 carries into the next exponent.
  const int biased = unit + 25 + (steps == 2048 ? 1 : 0);
  if (biased >= 31) return sign | 0x7c00;
  return sign | static_cast<uint16_t>((biased << 10) |
                                      (steps == 2048 ? 0 : steps - 1024));
}

// Updates of the source type: values near one, and for the float sources a
// few past FP16's range and below its normals.
std::vector<uint8_t> MakeUpdate(int32_t source, size_t elements,
                                uint32_t seed) {
  if (source != kLiteRtNvidiaCacheWriteFloat) {
    std::vector<uint8_t> update = Bytes(elements * 2, seed, /*floats=*/true);
    if (source == kLiteRtNvidiaCacheWriteBf16) {
      const uint16_t odd[] = {0x4780, 0xc7ff, 0x3300, 0x3080,
                              0xb2c0, 0x0000, 0x8000, 0x477f};
      for (size_t i = 0; i < 8 && 5 * i + 1 < elements; ++i) {
        std::memcpy(update.data() + 2 * (5 * i + 1), &odd[i], 2);
      }
    }
    return update;
  }
  std::vector<uint8_t> update(elements * 4);
  uint32_t state = seed * 2654435761u + 1u;
  for (size_t i = 0; i < elements; ++i) {
    state = state * 1664525u + 1013904223u;
    float value = (static_cast<float>(state >> 8) / 8388608.0f - 1.0f) * 3.0f;
    if (i % 37 == 5) value *= 1.0e5f;
    if (i % 41 == 7) value *= 1.0e-6f;
    std::memcpy(update.data() + 4 * i, &value, 4);
  }
  return update;
}

struct Shape {
  const char* name;
  int32_t source;
  bool int8, columns;
  int heads, cache_len, depth, tokens;
};

// The writes of `count` caches, each with a cache, an update and a position
// of its own, and the caches a write per launch gives.
struct Step {
  Shape shape;
  std::vector<Device> cache, expected, update, position;
  std::vector<std::vector<uint8_t>> host_cache, host_update;
  std::vector<int32_t> positions;
  std::vector<LiteRtNvidiaCacheWrite> writes;
  size_t cache_bytes;
};

constexpr float kScale = 0.0625f;
constexpr int32_t kZeroPoint = 3;

Step MakeStep(const Shape& shape, int count, uint32_t seed) {
  Step step;
  step.shape = shape;
  step.cache_bytes = static_cast<size_t>(shape.heads) * shape.cache_len *
                     shape.depth * (shape.int8 ? 1 : 2);
  const size_t elements =
      static_cast<size_t>(shape.heads) * shape.tokens * shape.depth;
  // The start, the middle and the end of the cache, and positions before
  // and past it that the write clamps.
  const int last = shape.cache_len - shape.tokens;
  const int32_t places[] = {0,  last / 2, last, last + 3,
                            -2, 5,        last, shape.cache_len + 100};
  for (int w = 0; w < count; ++w) {
    step.host_cache.push_back(
        Bytes(step.cache_bytes, seed + 13 * w, /*floats=*/!shape.int8));
    step.host_update.push_back(
        MakeUpdate(shape.source, elements, seed + 17 * w + 1));
    step.positions.push_back(places[w % 8]);
    step.cache.push_back(Upload(step.host_cache[w]));
    step.expected.push_back(Upload(step.host_cache[w]));
    step.update.push_back(Upload(step.host_update[w]));
    step.position.push_back(Upload(&step.positions[w], sizeof(int32_t)));
    step.writes.push_back({step.cache[w].get(), step.update[w].get(),
                           static_cast<const int32_t*>(step.position[w].get()),
                           shape.columns ? 1 : 0, shape.int8 ? 1 : 0,
                           kScale + 0.001f * w, kZeroPoint - w});
  }
  return step;
}

cudaError_t LaunchWrites(const Step& step, cudaStream_t stream) {
  return LiteRtNvidiaLaunchCacheWrites(
      step.writes.data(), static_cast<int32_t>(step.writes.size()),
      step.shape.source, step.shape.heads, step.shape.cache_len,
      step.shape.depth, step.shape.tokens, stream);
}

// One write a launch, into the `expected` caches: an INT8 cache through the
// single-sequence kernel, an FP16 cache through this launch with one write.
cudaError_t LaunchEach(const Step& step, cudaStream_t stream) {
  const Shape& s = step.shape;
  for (size_t w = 0; w < step.writes.size(); ++w) {
    LiteRtNvidiaCacheWrite write = step.writes[w];
    write.cache = step.expected[w].get();
    const cudaError_t status =
        s.int8 ? LaunchInt8CacheWrite(
                     write.cache, write.cache, step.cache_bytes, write.update,
                     static_cast<Int8CacheWriteSource>(s.source),
                     write.position, s.heads, s.cache_len, s.depth, s.tokens,
                     s.columns, write.scale, write.zero_point, stream)
               : LiteRtNvidiaLaunchCacheWrites(&write, 1, s.source, s.heads,
                                               s.cache_len, s.depth, s.tokens,
                                               stream);
    if (status != cudaSuccess) return status;
  }
  return cudaSuccess;
}

// What an FP16 cache holds after its write.
std::vector<uint8_t> ExpectedHalfCache(const Step& step, int w) {
  const Shape& s = step.shape;
  std::vector<uint8_t> cache = step.host_cache[w];
  const int pos =
      std::min(std::max(step.positions[w], 0), s.cache_len - s.tokens);
  for (int h = 0; h < s.heads; ++h) {
    for (int t = 0; t < s.tokens; ++t) {
      for (int d = 0; d < s.depth; ++d) {
        const size_t from =
            (static_cast<size_t>(h) * s.tokens + t) * s.depth + d;
        const size_t to =
            s.columns
                ? (static_cast<size_t>(h) * s.depth + d) * s.cache_len + pos + t
                : (static_cast<size_t>(h) * s.cache_len + pos + t) * s.depth +
                      d;
        uint16_t bits;
        if (s.source == kLiteRtNvidiaCacheWriteHalf) {
          std::memcpy(&bits, step.host_update[w].data() + 2 * from, 2);
        } else {
          bits = HalfBits(SourceValue(step.host_update[w], s.source, from));
        }
        std::memcpy(cache.data() + 2 * to, &bits, 2);
      }
    }
  }
  return cache;
}

TEST(CacheWriteSlotsTest, EveryCacheMatchesItsOwnWrite) {
  const Shape shapes[] = {
      {"26b local keys", kLiteRtNvidiaCacheWriteBf16, false, false, 8, 1024,
       256, 1},
      {"26b global values", kLiteRtNvidiaCacheWriteBf16, false, false, 2, 2048,
       512, 1},
      {"FP16 rows as columns", kLiteRtNvidiaCacheWriteHalf, false, true, 2, 640,
       256, 1},
      {"FP32 rows of a verify step", kLiteRtNvidiaCacheWriteFloat, false, false,
       4, 512, 128, 5},
      {"e4b keys", kLiteRtNvidiaCacheWriteBf16, true, false, 2, 2048, 256, 1},
      {"e4b values", kLiteRtNvidiaCacheWriteBf16, true, true, 2, 2048, 256, 1},
      {"e2b values, FP16 rows", kLiteRtNvidiaCacheWriteHalf, true, true, 1,
       1280, 256, 1},
      {"INT8 rows of a verify step", kLiteRtNvidiaCacheWriteFloat, true, true,
       2, 512, 256, 5},
  };
  for (const Shape& shape : shapes) {
    for (const int count : {1, 2, 3, 4, 8, 16}) {
      SCOPED_TRACE(::testing::Message()
                   << shape.name << ", " << count << " writes");
      Step step = MakeStep(shape, count, 10 * count + 1);
      ASSERT_EQ(LaunchEach(step, nullptr), cudaSuccess);
      ASSERT_EQ(LaunchWrites(step, nullptr), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      for (int w = 0; w < count; ++w) {
        const std::vector<uint8_t> got =
            Download(step.cache[w].get(), step.cache_bytes);
        ASSERT_EQ(got.size(), step.cache_bytes);
        EXPECT_NE(got, step.host_cache[w]) << "write " << w;
        EXPECT_EQ(got, Download(step.expected[w].get(), step.cache_bytes))
            << "write " << w;
        if (!shape.int8) {
          EXPECT_EQ(got, ExpectedHalfCache(step, w)) << "write " << w;
        }
      }
    }
  }
}

TEST(CacheWriteSlotsTest, RejectsInvalidArguments) {
  const Shape shape = {"", kLiteRtNvidiaCacheWriteBf16, true, false, 2, 64, 32,
                       1};
  Step step = MakeStep(shape, 2, 1);
  const auto launch = [&](int32_t count, int32_t source, int32_t tokens) {
    return LiteRtNvidiaLaunchCacheWrites(step.writes.data(), count, source,
                                         shape.heads, shape.cache_len,
                                         shape.depth, tokens, nullptr);
  };
  EXPECT_EQ(launch(2, shape.source, 1), cudaSuccess);
  EXPECT_EQ(launch(0, shape.source, 1), cudaErrorInvalidValue);
  EXPECT_EQ(launch(kLiteRtNvidiaCacheWriteMaxWrites + 1, shape.source, 1),
            cudaErrorInvalidValue);
  EXPECT_EQ(launch(2, 3, 1), cudaErrorInvalidValue);
  EXPECT_EQ(launch(2, shape.source, shape.cache_len + 1),
            cudaErrorInvalidValue);
  EXPECT_EQ(
      LiteRtNvidiaLaunchCacheWrites(nullptr, 2, shape.source, shape.heads,
                                    shape.cache_len, shape.depth, 1, nullptr),
      cudaErrorInvalidValue);
  step.writes[1].scale = 0.0f;
  EXPECT_EQ(launch(2, shape.source, 1), cudaErrorInvalidValue);
  step.writes[1].scale = kScale;
  step.writes[1].position = nullptr;
  EXPECT_EQ(launch(2, shape.source, 1), cudaErrorInvalidValue);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

// A layer's key and value writes of every slot in one launch against a
// launch per write, replayed from CUDA graphs as a decode step replays them.
TEST(CacheWriteSlotsTest, TimesALayersWrites) {
  const Shape shapes[] = {
      {"26b local layer (FP16 caches)", kLiteRtNvidiaCacheWriteBf16, false,
       false, 8, 1024, 256, 1},
      {"26b global layer (FP16 caches)", kLiteRtNvidiaCacheWriteBf16, false,
       false, 2, 2048, 512, 1},
      {"e4b layer (INT8 caches)", kLiteRtNvidiaCacheWriteBf16, true, false, 2,
       2048, 256, 1},
  };
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  for (const Shape& shape : shapes) {
    for (const int slots : {1, 2, 3, 4, 8}) {
      Step step = MakeStep(shape, 2 * slots, 3);
      float separate = 0.0f, one = 0.0f;
      ASSERT_TRUE(TimeTwoWays(
          stream, [&] { return LaunchEach(step, stream); },
          [&] { return LaunchWrites(step, stream); }, 200, &separate, &one));
      std::printf(
          "%s, %d slots: %d launches %.1f us, one launch %.1f us "
          "(%.2fx)\n",
          shape.name, slots, 2 * slots, separate, one, separate / one);
    }
  }
  cudaStreamDestroy(stream);
}

}  // namespace
