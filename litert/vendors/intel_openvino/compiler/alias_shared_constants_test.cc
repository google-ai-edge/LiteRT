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

#include "litert/vendors/intel_openvino/compiler/alias_shared_constants.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/load_test_model.h"
#include "litert/vendors/intel_openvino/compiler/graph_iterator.h"
#include "litert/vendors/intel_openvino/compiler/npu_optimizer.h"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"
#include "litert/vendors/intel_openvino/compiler/weight_bank_attributes.h"
#include "litert/vendors/intel_openvino/compiler/weightless_caching_attributes.hpp"

namespace litert {
namespace openvino {
namespace {

// Builds a finalized bank from |model| and converts subgraph 0 through the same
// frontend path the plugin uses, returning the OpenVINO model.
std::shared_ptr<ov::Model> BuildBankAndModel(litert::compiler::Model& model,
                                             const LiteRtCompilerContext* ctx,
                                             WeightBank& bank) {
  for (size_t s = 0; s < model.NumSubgraphs(); ++s) {
    auto graph = model.Subgraph(s);
    if (graph.HasValue()) bank.AddSubgraph(graph.Value());
  }

  auto fe = std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
  auto subgraph = model.Subgraph(0);
  std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> delegate =
      std::make_shared<litert::openvino::GraphIteratorDelegate>(
          ctx, &subgraph.Value());
  return fe->convert(fe->load(delegate));
}

// Lays out the shared pool the same way the plugin does: ascending BufferId
// order, each buffer's offset running after the previous one's bytes.
std::map<int32_t, size_t> BuildPoolOffsets(const WeightBank& bank) {
  std::map<uint32_t, absl::Span<const uint8_t>> ordered(bank.Buffers().begin(),
                                                        bank.Buffers().end());
  std::map<int32_t, size_t> pool_offset_of;
  size_t running_offset = 0;
  for (const auto& [buffer_id, bytes] : ordered) {
    pool_offset_of[static_cast<int32_t>(buffer_id)] = running_offset;
    running_offset += bytes.size();
  }
  return pool_offset_of;
}

// With no shared pool and an empty bank, nothing can be aliased or tagged. This
// invariant holds regardless of what the frontend does to the weights.
TEST(AliasSharedConstantsTest, EmptyPoolAliasesNothing) {
  auto cc_model = testing::LoadTestFileModel("multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());

  auto fe = std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
  auto subgraph = model.Subgraph(0);
  std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> delegate =
      std::make_shared<litert::openvino::GraphIteratorDelegate>(
          ctx, &subgraph.Value());
  auto ov_model = fe->convert(fe->load(delegate));

  WeightBank empty_bank;
  std::map<int32_t, size_t> empty_pool;
  auto aliased =
      AliasAndTagSharedConstants(ov_model, empty_bank, empty_pool, /*idx=*/0);
  ASSERT_TRUE(aliased.HasValue());
  EXPECT_EQ(*aliased, 0u);

  for (const auto& node : ov_model->get_ordered_ops()) {
    auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!cnst) continue;
    EXPECT_FALSE(cnst->get_rt_info().count(
        ov::WeightlessCacheAttribute::get_type_info_static()))
        << cnst->get_friendly_name();
  }
}

// Every Constant the pass tags with a WeightlessCacheAttribute must carry the
// authoritative pool offset for its BufferId and its own byte size. This is the
// load-bearing correctness invariant (a wrong bin_offset makes NPUW mmap the
// wrong weight at runtime), and it holds for both aliased and left-baked
// constants -- so the test is robust to how many weights the frontend happens
// to leave byte-identical to the pool.
TEST(AliasSharedConstantsTest, TaggedConstantsCarryPoolOffsets) {
  auto cc_model = testing::LoadTestFileModel("multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  WeightBank bank;
  auto ov_model = BuildBankAndModel(model, ctx, bank);
  const auto pool_offset_of = BuildPoolOffsets(bank);

  auto aliased_or =
      AliasAndTagSharedConstants(ov_model, bank, pool_offset_of, /*idx=*/0);
  ASSERT_TRUE(aliased_or.HasValue());
  const size_t aliased = *aliased_or;

  size_t tagged = 0;
  for (const auto& node : ov_model->get_ordered_ops()) {
    auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!cnst) continue;
    const auto& rt = cnst->get_rt_info();
    const auto it =
        rt.find(ov::WeightlessCacheAttribute::get_type_info_static());
    if (it == rt.end()) continue;
    ++tagged;

    const auto& attr = it->second.as<ov::WeightlessCacheAttribute>();
    const auto buffer_id = bank.BufferIdOfName(cnst->get_friendly_name());
    ASSERT_TRUE(buffer_id.has_value()) << cnst->get_friendly_name();
    const auto off_it = pool_offset_of.find(*buffer_id);
    ASSERT_NE(off_it, pool_offset_of.end()) << cnst->get_friendly_name();
    EXPECT_EQ(attr.bin_offset, off_it->second) << cnst->get_friendly_name();
    EXPECT_EQ(attr.original_size, cnst->get_byte_size())
        << cnst->get_friendly_name();
  }

  // Every aliased constant is tagged; byte-mismatched ones are tagged in place
  // without being aliased, so tagged >= aliased.
  EXPECT_GE(tagged, aliased);
}

// Builds a one-Constant model (Constant -> Result), tagging the Constant as
// LiteRt-generated with |source_key| if non-empty.
std::shared_ptr<ov::Model> BuildSingleConstantModel(
    const std::string& friendly_name, const std::vector<uint8_t>& bytes,
    const std::string& source_key = "") {
  auto cnst = ov::op::v0::Constant::create(ov::element::u8,
                                           ov::Shape{bytes.size()}, bytes);
  cnst->set_friendly_name(friendly_name);
  if (!source_key.empty()) {
    cnst->get_rt_info()
        [LiteRtGeneratedConstantAttribute::get_type_info_static()] =
        LiteRtGeneratedConstantAttribute(source_key);
  }
  auto result = std::make_shared<ov::op::v0::Result>(cnst);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{}, "single_constant");
}

TEST(HarvestSharedConstantsTest, RegistersGeneratedConstantByKey) {
  auto model = BuildSingleConstantModel(
      /*friendly_name=*/"stacked_gate_up", /*bytes=*/{1, 2, 3, 4},
      /*source_key=*/"stacked_gate_up");

  WeightBank bank;
  std::unordered_set<std::string> live_names;
  HarvestSharedConstants(model, bank, live_names);
  bank.FinalizeDerivedBuffers();

  const auto id = bank.BufferIdOfName("stacked_gate_up");
  ASSERT_TRUE(id.has_value());
  const auto bytes = bank.Buffers().at(*id);
  ASSERT_EQ(bytes.size(), 4u);
  EXPECT_EQ(bytes[0], 1);
  EXPECT_EQ(live_names.count("stacked_gate_up"), 1u);
}

// A generated constant whose friendly_name doesn't match its source_key can't
// be resolved back by name later, so HarvestSharedConstants must skip it
// rather than registering it under a name nothing will ever look up.
TEST(HarvestSharedConstantsTest, SkipsNameKeyMismatch) {
  auto model = BuildSingleConstantModel(
      /*friendly_name=*/"renamed_after_rewrite", /*bytes=*/{5, 6, 7},
      /*source_key=*/"original_key");

  WeightBank bank;
  std::unordered_set<std::string> live_names;
  HarvestSharedConstants(model, bank, live_names);
  bank.FinalizeDerivedBuffers();

  EXPECT_EQ(bank.BufferIdOfName("original_key"), std::nullopt);
  EXPECT_EQ(bank.BufferIdOfName("renamed_after_rewrite"), std::nullopt);
  // Skipped for registration, but it IS still in the graph.
  EXPECT_EQ(live_names.count("renamed_after_rewrite"), 1u);
}

// An untagged Constant (an ordinary baked weight) must not be registered, but
// must still count as live -- the name is recorded before the registration
// skips. Reverse that and BuildPool(prune_dead=true) empties the pool.
TEST(HarvestSharedConstantsTest, SkipsConstantWithoutAttributeButMarksItLive) {
  auto model =
      BuildSingleConstantModel(/*friendly_name=*/"plain", /*bytes=*/{9});

  WeightBank bank;
  std::unordered_set<std::string> live_names;
  HarvestSharedConstants(model, bank, live_names);
  bank.FinalizeDerivedBuffers();

  EXPECT_EQ(bank.BufferIdOfName("plain"), std::nullopt);
  EXPECT_EQ(live_names.count("plain"), 1u);
}

// BuildPool with prune_dead=true must drop a registered buffer that no
// partition's Constants reference anymore (superseded by a stacked
// constant), keeping only buffers whose name was collected as live.
TEST(BuildPoolTest, PrunesBufferNotReferencedByAnyModel) {
  WeightBank bank;
  bank.RegisterGeneratedConstant("live", {1, 2});
  bank.RegisterGeneratedConstant("dead", {3, 4, 5});
  auto model =
      BuildSingleConstantModel(/*friendly_name=*/"live", /*bytes=*/{1, 2});

  std::unordered_set<std::string> live_names;
  HarvestSharedConstants(model, bank, live_names);
  const auto layout = BuildPool(live_names, bank, /*prune_dead=*/true);

  const auto live_id = bank.BufferIdOfName("live");
  const auto dead_id = bank.BufferIdOfName("dead");
  ASSERT_TRUE(live_id.has_value());
  ASSERT_TRUE(dead_id.has_value());
  const auto offsets = layout.OffsetMap();
  EXPECT_EQ(offsets.count(*live_id), 1u);
  EXPECT_EQ(offsets.count(*dead_id), 0u);
  ASSERT_EQ(layout.buffers.size(), 1u);
  EXPECT_EQ(layout.buffers[0].buffer_id, *live_id);
}

// Derived BufferIds must be assigned in ascending key order, so the pool
// layout (and every WLCA bin_offset derived from it) is reproducible.
TEST(BuildPoolTest, DerivedBufferIdsFollowAscendingKeyOrder) {
  WeightBank bank;
  // Registered out of order on purpose.
  bank.RegisterGeneratedConstant("zebra", {1});
  bank.RegisterGeneratedConstant("alpha", {2});
  bank.RegisterGeneratedConstant("middle", {3});

  BuildPool(/*live_names=*/{}, bank, /*prune_dead=*/false);

  const auto alpha = bank.BufferIdOfName("alpha");
  const auto middle = bank.BufferIdOfName("middle");
  const auto zebra = bank.BufferIdOfName("zebra");
  ASSERT_TRUE(alpha.has_value());
  ASSERT_TRUE(middle.has_value());
  ASSERT_TRUE(zebra.has_value());
  EXPECT_LT(*alpha, *middle);
  EXPECT_LT(*middle, *zebra);
}

// BuildPool with prune_dead=false (the GPU path) must keep every buffer
// regardless of whether any model still references it by name.
TEST(BuildPoolTest, KeepsEveryBufferWhenNotPruning) {
  WeightBank bank;
  bank.RegisterGeneratedConstant("a", {1});
  bank.RegisterGeneratedConstant("b", {2, 3});
  auto model = BuildSingleConstantModel(/*friendly_name=*/"a", /*bytes=*/{1});

  std::unordered_set<std::string> live_names;
  HarvestSharedConstants(model, bank, live_names);
  const auto layout = BuildPool(live_names, bank, /*prune_dead=*/false);

  EXPECT_EQ(layout.buffers.size(), 2u);
}

TEST(BuildPoolTest, WeightsBankNameIsDeterministicAndContentDerived) {
  WeightBank bank_a1;
  bank_a1.RegisterGeneratedConstant("w0", {1, 2, 3, 4});
  bank_a1.RegisterGeneratedConstant("w1", {5, 6, 7, 8});
  const auto layout_a1 =
      BuildPool(/*live_names=*/{}, bank_a1, /*prune_dead=*/false);

  WeightBank bank_a2;
  bank_a2.RegisterGeneratedConstant("w0", {1, 2, 3, 4});
  bank_a2.RegisterGeneratedConstant("w1", {5, 6, 7, 8});
  const auto layout_a2 =
      BuildPool(/*live_names=*/{}, bank_a2, /*prune_dead=*/false);

  WeightBank bank_b;
  bank_b.RegisterGeneratedConstant("w0", {1, 2, 3, 9});
  bank_b.RegisterGeneratedConstant("w1", {5, 6, 7, 8});
  const auto layout_b =
      BuildPool(/*live_names=*/{}, bank_b, /*prune_dead=*/false);

  const std::string name_a1 = layout_a1.WeightsBankName();
  const std::string name_a2 = layout_a2.WeightsBankName();
  const std::string name_b = layout_b.WeightsBankName();

  EXPECT_EQ(name_a1.size(), 28u);
  EXPECT_EQ(name_a1.substr(0, 12), "litert_npuw_");
  EXPECT_EQ(name_a1, name_a2);
  EXPECT_NE(name_a1, name_b);
}

// End-to-end integration test for DecomposeBooleanConstant + WeightBank +
// AliasAndTagSharedConstants: because DecomposeBooleanConstant preserves the
// original friendly_name on the replacement u8 Constant (and names the Convert
// node "<name>_to_bool" instead), HarvestSharedConstants and
// AliasAndTagSharedConstants resolve the decomposed u8 Constant back to its
// WeightBank BufferId, zero-copy alias it into the shared pool, and tag it with
// WeightlessCacheAttribute.
TEST(AliasSharedConstantsTest,
     DecomposedBooleanConstantIsHarvestedAndAliasedByOriginalName) {
  constexpr size_t kNumElems = 32;
  std::vector<uint8_t> raw_bytes(kNumElems);
  std::vector<char> bool_bytes(kNumElems);
  for (size_t i = 0; i < kNumElems; ++i) {
    raw_bytes[i] = static_cast<uint8_t>(i % 2 == 0 ? 1 : 0);
    bool_bytes[i] = static_cast<char>(raw_bytes[i]);
  }

  auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{kNumElems});
  auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{kNumElems});
  auto bool_cnst = ov::op::v0::Constant::create(
      ov::element::boolean, ov::Shape{kNumElems}, bool_bytes);
  bool_cnst->set_friendly_name("shared_bool_mask");
  bool_cnst->get_rt_info()
      [LiteRtGeneratedConstantAttribute::get_type_info_static()] =
      LiteRtGeneratedConstantAttribute("shared_bool_mask");
  auto sel = std::make_shared<ov::op::v1::Select>(bool_cnst, x, y);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(sel)},
      ov::ParameterVector{x, y}, "decomposed_bool_sharing");

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetDecomposeBooleanConstants(true)
      .Run(model);

  WeightBank bank;
  std::unordered_set<std::string> live_names;
  HarvestSharedConstants(model, bank, live_names);
  const auto layout = BuildPool(live_names, bank, /*prune_dead=*/true);
  ASSERT_EQ(layout.buffers.size(), 1u);

  auto aliased = AliasAndTagSharedConstants(model, bank, layout.OffsetMap(),
                                            /*partition_idx=*/0);
  ASSERT_TRUE(aliased.HasValue());
  EXPECT_EQ(*aliased, 1u);

  std::shared_ptr<ov::op::v0::Constant> u8_cnst;
  for (const auto& node : model->get_ordered_ops()) {
    if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
      u8_cnst = c;
    }
  }
  ASSERT_NE(u8_cnst, nullptr);
  EXPECT_EQ(u8_cnst->get_element_type(), ov::element::u8);
  EXPECT_EQ(u8_cnst->get_friendly_name(), "shared_bool_mask");
  EXPECT_EQ(u8_cnst->get_data_ptr<uint8_t>(), layout.buffers[0].bytes.data());
  const auto& rt = u8_cnst->get_rt_info();
  const auto it = rt.find(ov::WeightlessCacheAttribute::get_type_info_static());
  ASSERT_NE(it, rt.end());
  const auto& wlca = it->second.as<ov::WeightlessCacheAttribute>();
  EXPECT_EQ(wlca.bin_offset, 0u);
  EXPECT_EQ(wlca.original_size, kNumElems);
  EXPECT_EQ(wlca.original_dtype, ov::element::u8);
}

// A boolean Constant of 16+ elements must never reach aliasing:
// DecomposeBooleanConstant rewrites all of them to u8 first. If one does (here,
// because the decomposition is deliberately skipped),
// AliasAndTagSharedConstants must fail the compile rather than hand NPUW a
// boolean weight it cannot import.
TEST(AliasSharedConstantsTest, FailsOnUndecomposedBooleanConstant) {
  constexpr size_t kNumElems = 32;
  std::vector<char> bool_bytes(kNumElems, 1);

  auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{kNumElems});
  auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{kNumElems});
  auto bool_cnst = ov::op::v0::Constant::create(
      ov::element::boolean, ov::Shape{kNumElems}, bool_bytes);
  bool_cnst->set_friendly_name("undecomposed_bool_mask");
  auto sel = std::make_shared<ov::op::v1::Select>(bool_cnst, x, y);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(sel)},
      ov::ParameterVector{x, y}, "undecomposed_bool");

  WeightBank empty_bank;
  std::map<int32_t, size_t> empty_pool;
  auto aliased = AliasAndTagSharedConstants(model, empty_bank, empty_pool,
                                            /*partition_idx=*/0);
  ASSERT_FALSE(aliased.HasValue());
  EXPECT_EQ(aliased.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}
}  // namespace
}  // namespace openvino
}  // namespace litert
