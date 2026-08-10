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

#include "openvino/core/model.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "openvino/op/constant.hpp"
#include "absl/types/span.h"  // from @com_google_absl
#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/load_test_model.h"
#include "litert/vendors/intel_openvino/compiler/graph_iterator.h"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"
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
  EXPECT_EQ(
      AliasAndTagSharedConstants(ov_model, empty_bank, empty_pool, /*idx=*/0),
      0u);

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
// constants -- so the test is robust to how many weights the frontend happens to
// leave byte-identical to the pool.
TEST(AliasSharedConstantsTest, TaggedConstantsCarryPoolOffsets) {
  auto cc_model = testing::LoadTestFileModel("multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  WeightBank bank;
  auto ov_model = BuildBankAndModel(model, ctx, bank);
  const auto pool_offset_of = BuildPoolOffsets(bank);

  const size_t aliased =
      AliasAndTagSharedConstants(ov_model, bank, pool_offset_of, /*idx=*/0);

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

  // Every aliased Constant is tagged; left-baked (byte-mismatched) ones may be
  // tagged too, so tagged >= aliased.
  EXPECT_GE(tagged, aliased);
}

}  // namespace
}  // namespace openvino
}  // namespace litert
