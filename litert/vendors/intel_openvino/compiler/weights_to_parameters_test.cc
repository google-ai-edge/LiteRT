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

#include "litert/vendors/intel_openvino/compiler/weights_to_parameters.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/load_test_model.h"
#include "litert/vendors/intel_openvino/compiler/graph_iterator.h"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"

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

// Minimum size a weight Constant must exceed to be converted (mirrors
// kMinConvertBytes in weights_to_parameters.cc; small control constants stay).
constexpr size_t kMinConvertBytes = 256;

// Weight constants the bank knows (and above the size threshold) become
// Parameter graph inputs; const_map maps each converted weight's final input
// index to its BufferId; small control constants stay Constant. The number of
// conversions equals the number of eligible (>threshold, bank-known) constants
// present in the model -- so the test is correct regardless of model size.
TEST(WeightsToParametersTest, ConvertsWeightConstantsToParameters) {
  auto cc_model = testing::LoadTestFileModel("multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  WeightBank bank;
  auto ov_model = BuildBankAndModel(model, ctx, bank);

  // Count eligible weight constants BEFORE conversion: bank-known and larger
  // than the threshold.
  size_t eligible = 0;
  for (const auto& node : ov_model->get_ops()) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) continue;
    if (bank.BufferIdOfName(constant->get_friendly_name()).has_value() &&
        constant->get_byte_size() > kMinConvertBytes) {
      ++eligible;
    }
  }

  const size_t params_before = ov_model->inputs().size();
  std::map<std::string, uint32_t> const_map;
  const size_t converted =
      ConvertWeightsToParameters(ov_model, bank, &const_map);

  // Exactly the eligible constants are converted.
  EXPECT_EQ(converted, eligible);
  // Each conversion adds one Parameter input; const_map has one entry each.
  EXPECT_EQ(ov_model->inputs().size(), params_before + converted);
  EXPECT_EQ(const_map.size(), converted);

  // Invariant: any bank-known Constant still present must be below the
  // threshold (all large ones were converted).
  for (const auto& node : ov_model->get_ops()) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) continue;
    if (bank.BufferIdOfName(constant->get_friendly_name()).has_value()) {
      EXPECT_LE(constant->get_byte_size(), kMinConvertBytes)
          << constant->get_friendly_name();
    }
  }

  // Every const_map entry is keyed by a friendly_name that (a) resolves in the
  // bank to the recorded BufferId, and (b) names a Parameter input on the model.
  const auto& inputs = ov_model->inputs();
  for (const auto& [name, buffer_id] : const_map) {
    auto resolved_id = bank.BufferIdOfName(name);
    ASSERT_TRUE(resolved_id.has_value()) << name;
    EXPECT_EQ(static_cast<uint32_t>(*resolved_id), buffer_id);

    bool found_param = false;
    for (const auto& input : inputs) {
      auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(
          input.get_node_shared_ptr());
      if (parameter && parameter->get_friendly_name() == name) {
        found_param = true;
        break;
      }
    }
    EXPECT_TRUE(found_param) << "no Parameter input named " << name;
  }
}

// Converting against an empty bank converts nothing and adds no inputs.
TEST(WeightsToParametersTest, EmptyBankConvertsNothing) {
  auto cc_model = testing::LoadTestFileModel("multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());

  auto fe = std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
  auto subgraph = model.Subgraph(0);
  std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> delegate =
      std::make_shared<litert::openvino::GraphIteratorDelegate>(
          ctx, &subgraph.Value());
  auto ov_model = fe->convert(fe->load(delegate));
  const size_t inputs_before = ov_model->inputs().size();

  WeightBank empty_bank;
  std::map<std::string, uint32_t> const_map;
  EXPECT_EQ(ConvertWeightsToParameters(ov_model, empty_bank, &const_map), 0u);
  EXPECT_TRUE(const_map.empty());
  EXPECT_EQ(ov_model->inputs().size(), inputs_before);
}

}  // namespace
}  // namespace openvino
}  // namespace litert
