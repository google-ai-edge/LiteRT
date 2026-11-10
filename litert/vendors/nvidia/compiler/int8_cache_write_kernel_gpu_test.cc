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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/compiler/int8_cache_write_kernel.h"

namespace litert::nvidia {
namespace {

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation() {
    if (data_ != nullptr) cudaFree(data_);
  }
  cudaError_t Allocate(size_t bytes) { return cudaMalloc(&data_, bytes); }
  void* get() const { return data_; }

 private:
  void* data_ = nullptr;
};

// Round-to-nearest-even conversions for normal values.
uint16_t FloatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = x & 0x7fffffu;
  uint32_t half_mantissa = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u))) {
    if (++half_mantissa == 0x400u) {
      half_mantissa = 0;
      ++exponent;
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               half_mantissa);
}

float HalfToFloat(uint16_t h) {
  const int sign = (h >> 15) & 1;
  const int exponent = (h >> 10) & 0x1f;
  const int mantissa = h & 0x3ff;
  const float value = std::ldexp(1.0f + mantissa / 1024.0f, exponent - 15);
  return sign ? -value : value;
}

uint16_t FloatToBf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t lsb = (x >> 16) & 1u;
  x += 0x7fffu + lsb;
  return static_cast<uint16_t>(x >> 16);
}

float Bf16ToFloat(uint16_t b) {
  const uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

int8_t QuantizeReference(float x, float scale, int32_t zero_point) {
  x = std::fmin(std::fmax(x, -128.0f * scale), 127.0f * scale);
  int32_t q = static_cast<int32_t>(std::nearbyint(x / scale)) + zero_point;
  q = q < -128 ? -128 : q > 127 ? 127 : q;
  return static_cast<int8_t>(q);
}

struct Case {
  Int8CacheWriteSource source;
  bool columns;
  int tokens;
  int position;
  bool in_place;
};

TEST(Int8CacheWriteKernelTest, WritesQuantizedRowsAndKeepsTheRest) {
  constexpr int kHeads = 2, kCacheLen = 2048, kDepth = 256;
  constexpr float kScale = 0.0625f;
  constexpr int32_t kZero = -3;
  const Case cases[] = {
      {Int8CacheWriteSource::kBf16, false, 1, 300, true},
      {Int8CacheWriteSource::kBf16, true, 1, 300, true},
      {Int8CacheWriteSource::kHalf, false, 1024, 1000, true},
      {Int8CacheWriteSource::kHalf, true, 1024, 1000, true},
      {Int8CacheWriteSource::kFloat, false, 4, 2044, true},
      {Int8CacheWriteSource::kFloat, true, 4, 2044, true},
      {Int8CacheWriteSource::kBf16, false, 1, 300, false},
      {Int8CacheWriteSource::kBf16, true, 1024, 1024, false},
      // Positions past the end clamp to the last fitting row.
      {Int8CacheWriteSource::kBf16, false, 4, 5000, true},
  };
  const size_t cache_bytes = static_cast<size_t>(kHeads) * kCacheLen * kDepth;
  for (const Case& c : cases) {
    SCOPED_TRACE(testing::Message()
                 << "source=" << static_cast<int>(c.source)
                 << " columns=" << c.columns << " tokens=" << c.tokens
                 << " position=" << c.position << " in_place=" << c.in_place);
    std::vector<int8_t> cache(cache_bytes);
    for (size_t i = 0; i < cache_bytes; ++i) {
      cache[i] = static_cast<int8_t>((i * 7 + 13) % 251 - 125);
    }
    const size_t elements = static_cast<size_t>(kHeads) * c.tokens * kDepth;
    std::vector<float> values(elements);
    std::vector<uint16_t> bits16(elements);
    std::vector<float> bits32(elements);
    for (size_t i = 0; i < elements; ++i) {
      const float raw = std::sin(0.37f * static_cast<float>(i)) * 9.0f +
                        ((i % 97) - 48) * 0.0314f;
      switch (c.source) {
        case Int8CacheWriteSource::kHalf:
          bits16[i] = FloatToHalf(raw);
          values[i] = HalfToFloat(bits16[i]);
          break;
        case Int8CacheWriteSource::kBf16:
          bits16[i] = FloatToBf16(raw);
          values[i] = Bf16ToFloat(bits16[i]);
          break;
        case Int8CacheWriteSource::kFloat:
          bits32[i] = raw;
          values[i] = raw;
          break;
      }
    }
    DeviceAllocation device_in, device_out, device_update, device_position;
    ASSERT_EQ(device_in.Allocate(cache_bytes), cudaSuccess);
    ASSERT_EQ(device_update.Allocate(elements * sizeof(float)), cudaSuccess);
    ASSERT_EQ(device_position.Allocate(sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_in.get(), cache.data(), cache_bytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    if (c.source == Int8CacheWriteSource::kFloat) {
      ASSERT_EQ(cudaMemcpy(device_update.get(), bits32.data(),
                           elements * sizeof(float), cudaMemcpyHostToDevice),
                cudaSuccess);
    } else {
      ASSERT_EQ(cudaMemcpy(device_update.get(), bits16.data(),
                           elements * sizeof(uint16_t), cudaMemcpyHostToDevice),
                cudaSuccess);
    }
    const int32_t position = c.position;
    ASSERT_EQ(cudaMemcpy(device_position.get(), &position, sizeof(position),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    void* out = device_in.get();
    if (!c.in_place) {
      ASSERT_EQ(device_out.Allocate(cache_bytes), cudaSuccess);
      ASSERT_EQ(cudaMemset(device_out.get(), 0x55, cache_bytes), cudaSuccess);
      out = device_out.get();
    }
    ASSERT_EQ(LaunchInt8CacheWrite(
                  out, device_in.get(), cache_bytes, device_update.get(),
                  c.source, static_cast<const int32_t*>(device_position.get()),
                  kHeads, kCacheLen, kDepth, c.tokens, c.columns, kScale, kZero,
                  /*stream=*/nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<int8_t> actual(cache_bytes);
    ASSERT_EQ(
        cudaMemcpy(actual.data(), out, cache_bytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    // The reference: the old cache with the quantized rows written in.
    std::vector<int8_t> expected = cache;
    const int pos = std::min(std::max(c.position, 0), kCacheLen - c.tokens);
    for (int h = 0; h < kHeads; ++h) {
      for (int t = 0; t < c.tokens; ++t) {
        for (int d = 0; d < kDepth; ++d) {
          const size_t source =
              (static_cast<size_t>(h) * c.tokens + t) * kDepth + d;
          const size_t index =
              c.columns
                  ? (static_cast<size_t>(h) * kDepth + d) * kCacheLen + pos + t
                  : (static_cast<size_t>(h) * kCacheLen + pos + t) * kDepth + d;
          expected[index] = QuantizeReference(values[source], kScale, kZero);
        }
      }
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < cache_bytes; ++i) {
      if (actual[i] != expected[i] && ++mismatches <= 5) {
        ADD_FAILURE() << "byte " << i << ": " << static_cast<int>(actual[i])
                      << " != " << static_cast<int>(expected[i]);
      }
    }
    EXPECT_EQ(mismatches, 0u);
  }
}

}  // namespace
}  // namespace litert::nvidia
