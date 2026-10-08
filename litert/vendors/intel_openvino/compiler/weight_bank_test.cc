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

#include "litert/vendors/intel_openvino/compiler/weight_bank.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/load_test_model.h"

namespace litert {
namespace openvino {
namespace {

// add_cst.tflite has a single constant operand (a 16-byte add constant), so the
// bank should record exactly that one buffer.
TEST(WeightBankTest, RecordsConstantWeightBuffers) {
  auto cc_model = testing::LoadTestFileModel("add_cst.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto graph = model.Subgraph(0);
  ASSERT_TRUE(graph.HasValue());

  WeightBank bank;
  bank.AddSubgraph(graph.Value());

  EXPECT_EQ(bank.NumBuffers(), 1u);
  EXPECT_EQ(bank.TotalBytes(), 16u);
}

// cst_multi_subgraph.tflite has two subgraphs that reference the SAME constant
// buffer. Accumulating both subgraphs must keep a single entry: this is the
// cross-partition sharing case (e.g. prefill + decode sharing a weight).
TEST(WeightBankTest, DeduplicatesBufferSharedAcrossSubgraphs) {
  auto cc_model = testing::LoadTestFileModel("cst_multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  ASSERT_EQ(model.NumSubgraphs(), 2u);
  auto subgraph_0 = model.Subgraph(0);
  auto subgraph_1 = model.Subgraph(1);
  ASSERT_TRUE(subgraph_0.HasValue());
  ASSERT_TRUE(subgraph_1.HasValue());

  WeightBank bank;
  bank.AddSubgraph(subgraph_0.Value());
  const size_t buffers_after_first = bank.NumBuffers();
  bank.AddSubgraph(subgraph_1.Value());

  // The shared buffer is recorded once, not once per subgraph.
  EXPECT_EQ(buffers_after_first, 1u);
  EXPECT_EQ(bank.NumBuffers(), 1u);
  EXPECT_EQ(bank.TotalBytes(), 16u);
}

// A freshly constructed bank is empty.
TEST(WeightBankTest, EmptyByDefault) {
  WeightBank bank;
  EXPECT_EQ(bank.NumBuffers(), 0u);
  EXPECT_EQ(bank.TotalBytes(), 0u);
}

// BufferIdOfName resolves each recorded weight tensor's name to its BufferId,
// names that share a buffer resolve to the same id, and unknown names return
// nullopt. This is the lookup that builds the GlobalGraph const_map.
TEST(WeightBankTest, BufferIdOfNameResolvesWeightTensors) {
  auto cc_model = testing::LoadTestFileModel("multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  WeightBank bank;
  for (size_t s = 0; s < model.NumSubgraphs(); ++s) {
    auto graph = model.Subgraph(s);
    ASSERT_TRUE(graph.HasValue());
    bank.AddSubgraph(graph.Value());
  }

  size_t named_weights = 0;
  for (size_t s = 0; s < model.NumSubgraphs(); ++s) {
    auto graph = model.Subgraph(s);
    for (const auto& op : graph.Value().Ops()) {
      for (const auto& input : op.Inputs()) {
        if (!input.HasWeights()) {
          continue;
        }
        ++named_weights;
        const auto id = bank.BufferIdOfName(input.Name());
        ASSERT_TRUE(id.has_value());
        EXPECT_EQ(*id, input.Weights().BufferId());
      }
    }
  }
  EXPECT_GT(named_weights, 0u);
  EXPECT_EQ(bank.BufferIdOfName("no_such_tensor"), std::nullopt);
}

// A generated constant's BufferId is unresolvable until
// FinalizeDerivedBuffers() runs, and afterwards lands past every real id.
TEST(WeightBankTest, FinalizeDerivedBuffersAssignsIdPastRealOnes) {
  auto cc_model = testing::LoadTestFileModel("add_cst.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto graph = model.Subgraph(0);
  ASSERT_TRUE(graph.HasValue());

  WeightBank bank;
  bank.AddSubgraph(graph.Value());
  ASSERT_EQ(bank.NumBuffers(), 1u);

  bank.RegisterGeneratedConstant("stacked_expert_weights",
                                 std::vector<uint8_t>{1, 2, 3, 4});
  EXPECT_EQ(bank.BufferIdOfName("stacked_expert_weights"), std::nullopt);

  bank.FinalizeDerivedBuffers();

  const auto id = bank.BufferIdOfName("stacked_expert_weights");
  ASSERT_TRUE(id.has_value());
  for (const auto& [real_id, bytes] : bank.Buffers()) {
    if (real_id != *id) EXPECT_LT(real_id, *id);
  }
  EXPECT_EQ(bank.NumBuffers(), 2u);
}

// Registering the same key twice with identical bytes is a no-op; the second
// registration must not create a second buffer.
TEST(WeightBankTest, RegisterGeneratedConstantDedupesIdenticalBytes) {
  WeightBank bank;
  bank.RegisterGeneratedConstant("key", std::vector<uint8_t>{1, 2, 3});
  bank.RegisterGeneratedConstant("key", std::vector<uint8_t>{1, 2, 3});
  bank.FinalizeDerivedBuffers();

  EXPECT_EQ(bank.NumBuffers(), 1u);
}

// Registering the same key twice with DIFFERENT bytes keeps the first
// registration rather than silently overwriting it.
TEST(WeightBankTest, RegisterGeneratedConstantKeepsFirstOnMismatch) {
  WeightBank bank;
  bank.RegisterGeneratedConstant("key", std::vector<uint8_t>{1, 2, 3});
  bank.RegisterGeneratedConstant("key", std::vector<uint8_t>{9, 9});
  bank.FinalizeDerivedBuffers();

  const auto id = bank.BufferIdOfName("key");
  ASSERT_TRUE(id.has_value());
  const auto bytes = bank.Buffers().at(*id);
  ASSERT_EQ(bytes.size(), 3u);
  EXPECT_EQ(bytes[0], 1);
}

// FinalizeDerivedBuffers() must only assign ids once: a second call is
// ignored rather than reassigning (and potentially colliding) ids.
TEST(WeightBankTest, FinalizeDerivedBuffersIgnoresSecondCall) {
  WeightBank bank;
  bank.RegisterGeneratedConstant("key", std::vector<uint8_t>{1, 2, 3});
  bank.FinalizeDerivedBuffers();
  const auto id_before = bank.BufferIdOfName("key");
  ASSERT_TRUE(id_before.has_value());

  bank.RegisterGeneratedConstant("late_key", std::vector<uint8_t>{4, 5});
  bank.FinalizeDerivedBuffers();

  EXPECT_EQ(bank.BufferIdOfName("key"), id_before);
  // The post-finalize registration never gets an id assigned.
  EXPECT_EQ(bank.BufferIdOfName("late_key"), std::nullopt);
}

// Registering after finalization is a phase-ordering bug: the key is dropped,
// and the ids and bytes already handed out are left untouched.
TEST(WeightBankTest, RegisterGeneratedConstantAfterFinalizeIsDropped) {
  WeightBank bank;
  bank.RegisterGeneratedConstant("early", std::vector<uint8_t>{1, 2, 3});
  bank.FinalizeDerivedBuffers();

  const auto early_id = bank.BufferIdOfName("early");
  ASSERT_TRUE(early_id.has_value());
  const size_t num_before = bank.NumBuffers();

  bank.RegisterGeneratedConstant("late", std::vector<uint8_t>{4, 5});

  EXPECT_EQ(bank.BufferIdOfName("late"), std::nullopt);
  EXPECT_EQ(bank.BufferIdOfName("early"), early_id);
  EXPECT_EQ(bank.NumBuffers(), num_before);
  const auto bytes = bank.Buffers().at(*early_id);
  ASSERT_EQ(bytes.size(), 3u);
  EXPECT_EQ(bytes[0], 1);
}

}  // namespace
}  // namespace openvino
}  // namespace litert
