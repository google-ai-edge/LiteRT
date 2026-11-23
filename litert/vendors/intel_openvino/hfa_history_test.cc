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
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>

#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/core.hpp"
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

// A CPU model with a [1, 1, q, past + q] mask input, tagged as the HFA
// history mask when |tagged|, and, when |state_name| is not empty, an i64 [1]
// variable of that name, which the CPU plugin exposes as a VariableState the
// way NPUW exposes npuw_history_size.
ov::CompiledModel CompileMaskModel(size_t q, size_t past, bool tagged,
                                   const std::string& state_name) {
  auto mask = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::Shape{1, 1, q, past + q});
  mask->output(0).get_tensor().set_names(
      tagged ? std::unordered_set<std::string>{"mask",
                                               kHfaHistoryMaskTensorName}
             : std::unordered_set<std::string>{"mask"});
  ov::ResultVector results{std::make_shared<ov::op::v0::Result>(
      std::make_shared<ov::op::v0::Relu>(mask))};
  ov::SinkVector sinks;
  if (!state_name.empty()) {
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{1}, ov::element::i64,
                                   state_name});
    auto read = std::make_shared<ov::op::v6::ReadValue>(
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
        variable);
    sinks.push_back(std::make_shared<ov::op::v6::Assign>(read, variable));
    results.push_back(std::make_shared<ov::op::v0::Result>(read));
  }
  auto model = std::make_shared<ov::Model>(results, sinks,
                                           ov::ParameterVector{mask}, "mask");
  return ov::Core().compile_model(model, "CPU");
}

int64_t StateValue(ov::InferRequest& request, const std::string& name) {
  for (auto& state : request.query_state()) {
    if (state.get_name() == name) return state.get_state().data<int64_t>()[0];
  }
  ADD_FAILURE() << "no state " << name;
  return -1;
}

TEST(HfaHistoryReporterTest, WritesHistoryOfBoundMaskToState) {
  auto compiled = CompileMaskModel(4, 8, /*tagged=*/true,
                                   kNpuwHistorySizeStateName);
  auto request = compiled.create_infer_request();
  auto reporter = HfaHistoryReporter::Find(compiled, request, "prefill");
  ASSERT_TRUE(reporter.has_value());

  for (const size_t history : {0, 5, 8}) {
    request.set_input_tensor(0, MakeMask(4, 8, Causal(8, history)));
    auto reported = reporter->Update(request);
    ASSERT_TRUE(reported.HasValue()) << reported.Error().Message();
    EXPECT_EQ(*reported, static_cast<int64_t>(history));
    EXPECT_EQ(StateValue(request, kNpuwHistorySizeStateName),
              static_cast<int64_t>(history));
  }
}

TEST(HfaHistoryReporterTest, FindsNothingWithoutTagOrState) {
  // No tag: not a graph enable_hfa decomposed.
  auto untagged = CompileMaskModel(4, 8, /*tagged=*/false,
                                   kNpuwHistorySizeStateName);
  auto untagged_request = untagged.create_infer_request();
  EXPECT_FALSE(
      HfaHistoryReporter::Find(untagged, untagged_request, "decode"));

  // A plugin without the npuw_history_size state (e.g. before
  // openvinotoolkit/openvino#39050).
  auto stateless = CompileMaskModel(4, 8, /*tagged=*/true, "other_state");
  auto stateless_request = stateless.create_infer_request();
  EXPECT_FALSE(
      HfaHistoryReporter::Find(stateless, stateless_request, "prefill"));
}

}  // namespace
}  // namespace openvino
}  // namespace litert
