// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/vendors/intel_openvino/hfa_history.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/tensor.hpp"

namespace litert {
namespace openvino {
namespace {

constexpr float kMasked = -std::numeric_limits<float>::infinity();

// A [1, 1, q, past + q] mask; |visible(row, column)| decides each entry.
template <typename T = float>
ov::Tensor MakeMask(size_t q, size_t past,
                    const std::function<bool(size_t, size_t)>& visible,
                    float masked = kMasked) {
  const auto type = sizeof(T) == sizeof(float) ? ov::element::f32
                    : std::is_same_v<T, ov::float16> ? ov::element::f16
                                                     : ov::element::bf16;
  ov::Tensor mask(type, ov::Shape{1, 1, q, past + q});
  T* data = mask.data<T>();
  for (size_t r = 0; r < q; ++r) {
    for (size_t c = 0; c < past + q; ++c) {
      data[r * (past + q) + c] = T(visible(r, c) ? 0.0f : masked);
    }
  }
  return mask;
}

// The global-attention mask of a chunk of |q| tokens after |history| cached
// ones: the first |history| cache columns and the causal part of the chunk.
std::function<bool(size_t, size_t)> Causal(size_t past, size_t history) {
  return [=](size_t r, size_t c) {
    return c < history || (c >= past && c - past <= r);
  };
}

int64_t History(const ov::Tensor& mask) {
  auto history = HistorySizeFromMask(mask);
  EXPECT_TRUE(history.HasValue()) << history.Error().Message();
  return history.HasValue() ? *history : -1;
}

TEST(HfaHistoryTest, ReadsHistoryOfCausalChunk) {
  EXPECT_EQ(History(MakeMask(4, 8, Causal(8, 5))), 5);
  EXPECT_EQ(History(MakeMask(4, 8, Causal(8, 8))), 8);
}

TEST(HfaHistoryTest, FirstChunkHasNoHistory) {
  EXPECT_EQ(History(MakeMask(4, 8, Causal(8, 0))), 0);
}

TEST(HfaHistoryTest, UnmaskedChunkSeesWholeCache) {
  EXPECT_EQ(History(MakeMask(4, 8, [](size_t, size_t) { return true; })), 8);
}

// Padding rows of a short last chunk are masked end to end and add nothing.
TEST(HfaHistoryTest, IgnoresFullyMaskedRows) {
  auto causal = Causal(8, 3);
  EXPECT_EQ(History(MakeMask(4, 8,
                             [&](size_t r, size_t c) {
                               return r < 2 && causal(r, c);
                             })),
            3);
  EXPECT_EQ(History(MakeMask(4, 8, [](size_t, size_t) { return false; })), 0);
}

// Visible cache entries that are not a prefix (a ring buffer, a sliding
// window) are all covered.
TEST(HfaHistoryTest, CoversLastVisibleCacheColumn) {
  EXPECT_EQ(History(MakeMask(4, 8,
                             [](size_t r, size_t c) {
                               return (c >= 2 && c < 6) ||
                                      (c >= 8 && c - 8 <= r);
                             })),
            6);
}

// Rows whose cache columns differ are each read: here only the last row sees
// the end of the window.
TEST(HfaHistoryTest, ReadsEveryDistinctRow) {
  EXPECT_EQ(History(MakeMask(4, 8,
                             [](size_t r, size_t c) {
                               return (c >= r && c < r + 4) ||
                                      (c >= 8 && c - 8 <= r);
                             })),
            7);
}

TEST(HfaHistoryTest, AcceptsAnyMaskedValueAndHalfTypes) {
  EXPECT_EQ(History(MakeMask(4, 8, Causal(8, 5), -1e9f)), 5);
  EXPECT_EQ(History(MakeMask(4, 8, Causal(8, 5), -100.0f)), 5);
  EXPECT_EQ(History(MakeMask<ov::float16>(4, 8, Causal(8, 6), -65504.0f)), 6);
  EXPECT_EQ(History(MakeMask<ov::bfloat16>(4, 8, Causal(8, 7))), 7);
}

TEST(HfaHistoryTest, RejectsUnsupportedMasks) {
  EXPECT_FALSE(HistorySizeFromMask(ov::Tensor(ov::element::i32,
                                              ov::Shape{1, 1, 4, 12}))
                   .HasValue());
  EXPECT_FALSE(
      HistorySizeFromMask(ov::Tensor(ov::element::f32, ov::Shape{4, 12}))
          .HasValue());
  EXPECT_FALSE(HistorySizeFromMask(ov::Tensor(ov::element::f32,
                                              ov::Shape{2, 1, 4, 12}))
                   .HasValue());
  // No cache columns: nothing to read the history from.
  EXPECT_FALSE(
      HistorySizeFromMask(ov::Tensor(ov::element::f32, ov::Shape{1, 1, 4, 4}))
          .HasValue());
  EXPECT_TRUE(CheckHistoryMask(ov::element::f16, ov::Shape{1, 1, 1024, 17408})
                  .HasValue());
}

}  // namespace
}  // namespace openvino
}  // namespace litert
