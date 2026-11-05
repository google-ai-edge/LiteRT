// Copyright (C) 2025 Intel Corporation
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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/core.hpp"
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"
#include "litert/vendors/intel_openvino/compiler/npu_optimizer.h"
#include "litert/vendors/intel_openvino/compiler/openvino_compile_context.h"
#include "litert/vendors/intel_openvino/compiler/openvino_soc_config.h"

namespace litert {
namespace {

using ::testing::Values;

const auto kSupportedOps = Values("add_simple.tflite");
const auto kSupportedSocModels = Values("LNL", "PTL");

TEST(TestOVPlugin, PartitionAddGraph) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  auto model = testing::LoadTestFileModel("add_simple.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_partition(
      plugin.Get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflAdd);
}

TEST(TestOVPlugin, CompileAddSubgraph) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  auto model = testing::LoadTestFileModel("add_simple.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_compile(
      plugin.Get(), "PTL", model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(plugin.Api()->get_compiled_result_byte_code(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  LITERT_ASSERT_OK(plugin.Api()->get_compiled_result_call_info(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);

  plugin.Api()->destroy_compiled_result(compiled);
}

// TOPK_V2 with a constant K (input 1 carries a weights buffer) yields a static
// output shape and must be selected for the OpenVINO partition.
TEST(TestOVPlugin, PartitionTopKConstK) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  auto model = testing::LoadTestFileModel("simple_topk_op.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_partition(
      plugin.Get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflTopkV2);
}

// TOPK_V2 with a runtime K (input 1 is a graph input, no weights buffer) leaves
// the sorted-axis dimension dynamic, which NPU compilation cannot handle. The
// op must be left unselected so it falls back to CPU.
TEST(TestOVPlugin, PartitionTopKRuntimeKNotSelected) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  auto model = testing::LoadTestFileModel("simple_topk_op_runtime_k.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_partition(
      plugin.Get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  EXPECT_EQ(selected_ops.size(), 0);
}

// A constant-K TOPK_V2 subgraph compiles to non-empty NPU bytecode, confirming
// the folded static shape is accepted end-to-end.
TEST(TestOVPlugin, CompileTopKConstK) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  auto model = testing::LoadTestFileModel("simple_topk_op.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_compile(
      plugin.Get(), "PTL", model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;
  LITERT_ASSERT_OK(plugin.Api()->get_compiled_result_byte_code(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  plugin.Api()->destroy_compiled_result(compiled);
}

// Tests for GetSocModelConfig
TEST(TestOVPlugin, GetSocModelConfigLNL) {
  ov::AnyMap config_map;
  EXPECT_EQ(litert::openvino::GetSocModelConfig("LNL", config_map),
            kLiteRtStatusOk);
  ASSERT_GT(config_map.count("NPU_PLATFORM"), 0u);
  EXPECT_EQ(config_map.at("NPU_PLATFORM").as<std::string>(), "NPU4000");
}

TEST(TestOVPlugin, GetSocModelConfigPTL) {
  ov::AnyMap config_map;
  EXPECT_EQ(litert::openvino::GetSocModelConfig("PTL", config_map),
            kLiteRtStatusOk);
  ASSERT_GT(config_map.count("NPU_PLATFORM"), 0u);
  EXPECT_EQ(config_map.at("NPU_PLATFORM").as<std::string>(), "NPU5010");
}

TEST(TestOVPlugin, GetSocModelConfigUnknownReturnsError) {
  ov::AnyMap config_map;
  EXPECT_EQ(litert::openvino::GetSocModelConfig("UNKNOWN_SOC", config_map),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(config_map.count("NPU_PLATFORM"), 0u);
}

TEST(TestOVPlugin,
     OptimizeModelFoldsOneHotMatMulByDefaultAndHonorsDisableConfig) {
  auto build_model = []() {
    auto idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32,
                                                       ov::Shape{1, 4});
    auto depth =
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {6});
    auto on_v =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto off_v =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto oh = std::make_shared<ov::op::v1::OneHot>(idx, depth, on_v, off_v, -1);
    std::vector<float> w(3 * 6, 0.5f);
    auto weights =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{3, 6}, w);
    auto mm = std::make_shared<ov::op::v0::MatMul>(oh, weights, false, true);
    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(mm)},
        ov::ParameterVector{idx}, "one_hot_matmul");
  };

  auto default_opts = litert::intel_openvino::IntelOpenVinoOptions::Create();
  ASSERT_TRUE(default_opts.HasValue());
  auto default_ctx_or =
      litert::openvino::OpenVinoCompileContext::Create(default_opts);
  ASSERT_TRUE(default_ctx_or.HasValue());
  auto folded_model = build_model();
  default_ctx_or->OptimizeModel(folded_model);
  size_t gathers_default = 0;
  size_t matmuls_default = 0;
  for (const auto& node : folded_model->get_ops()) {
    if (std::dynamic_pointer_cast<ov::op::v8::Gather>(node)) ++gathers_default;
    if (std::dynamic_pointer_cast<ov::op::v0::MatMul>(node)) ++matmuls_default;
  }
  EXPECT_EQ(gathers_default, 1u);
  EXPECT_EQ(matmuls_default, 0u);

  auto opts = litert::intel_openvino::IntelOpenVinoOptions::Create();
  ASSERT_TRUE(opts.HasValue());
  opts->SetConfigsMapOption("fold_one_hot_matmul_to_gather", "false");
  auto disabled_ctx_or = litert::openvino::OpenVinoCompileContext::Create(opts);
  ASSERT_TRUE(disabled_ctx_or.HasValue());
  auto unfolded_model = build_model();
  disabled_ctx_or->OptimizeModel(unfolded_model);
  size_t gathers_disabled = 0;
  size_t matmuls_disabled = 0;
  for (const auto& node : unfolded_model->get_ops()) {
    if (std::dynamic_pointer_cast<ov::op::v8::Gather>(node)) ++gathers_disabled;
    if (std::dynamic_pointer_cast<ov::op::v0::MatMul>(node)) ++matmuls_disabled;
  }
  EXPECT_EQ(gathers_disabled, 0u);
  EXPECT_EQ(matmuls_disabled, 1u);
}

TEST(TestOVPlugin, CompileFoldedMaskedOneHotLookupForLnlAndPtl) {
  using ov::op::v0::Constant;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v0::Unsqueeze;
  using ov::op::v1::GreaterEqual;
  using ov::op::v1::Less;
  using ov::op::v1::LogicalAnd;
  using ov::op::v1::LogicalOr;
  using ov::op::v1::NotEqual;
  using ov::op::v1::OneHot;
  using ov::op::v1::Reshape;
  using ov::op::v1::Select;

  constexpr int64_t kSeqLen = 16;
  constexpr int64_t kDepth = 64;
  constexpr int64_t kEmbDim = 32;

  auto coord = std::make_shared<Parameter>(ov::element::i32,
                                           ov::Shape{1, kSeqLen});
  auto depth_c = Constant::create(ov::element::i32, ov::Shape{}, {kDepth});
  auto on_c = Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
  auto off_c = Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
  auto oh =
      std::make_shared<OneHot>(coord, depth_c, on_c, off_c, /*axis=*/-1);

  auto zero_i32 = Constant::create(ov::element::i32, ov::Shape{}, {0});
  auto neg1_i32 = Constant::create(ov::element::i32, ov::Shape{}, {-1});
  auto oob = std::make_shared<LogicalOr>(
      std::make_shared<Less>(coord, zero_i32),
      std::make_shared<GreaterEqual>(coord, depth_c));
  auto is_nan = std::make_shared<LogicalAnd>(
      oob, std::make_shared<NotEqual>(coord, neg1_i32));
  auto nan_mask = std::make_shared<Unsqueeze>(
      is_nan, Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
  auto nan_fill = Constant::create(
      ov::element::f32, ov::Shape{}, {std::numeric_limits<float>::quiet_NaN()});
  auto sel = std::make_shared<Select>(nan_mask, nan_fill, oh);

  auto flat = std::make_shared<Reshape>(
      sel,
      Constant::create(ov::element::i64, ov::Shape{2},
                       std::vector<int64_t>{-1, kDepth}),
      /*special_zero=*/false);
  std::vector<float> weights(kEmbDim * kDepth, 0.25f);
  auto w_const = Constant::create(
      ov::element::f32, ov::Shape{kEmbDim, kDepth}, weights);
  auto mm = std::make_shared<MatMul>(flat, w_const, /*transpose_a=*/false,
                                     /*transpose_b=*/true);
  auto out = std::make_shared<Reshape>(
      mm,
      Constant::create(ov::element::i64, ov::Shape{3},
                       std::vector<int64_t>{1, kSeqLen, kEmbDim}),
      /*special_zero=*/false);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(out)},
      ov::ParameterVector{coord}, "masked_one_hot_lookup_npu");

  litert::openvino::NpuOptimizer().SetConstantFold(true).Run(model);

  size_t gather_count = 0;
  size_t matmul_count = 0;
  for (const auto& node : model->get_ops()) {
    if (std::dynamic_pointer_cast<ov::op::v8::Gather>(node)) ++gather_count;
    if (std::dynamic_pointer_cast<ov::op::v0::MatMul>(node)) ++matmul_count;
  }
  EXPECT_EQ(gather_count, 1u);
  EXPECT_EQ(matmul_count, 0u);

  ov::Core core;
  for (const char* soc : {"LNL", "PTL"}) {
    ov::AnyMap config_map;
    ASSERT_EQ(litert::openvino::GetSocModelConfig(soc, config_map),
              kLiteRtStatusOk);
    auto compiled = core.compile_model(model, "NPU", config_map);
    std::ostringstream blob;
    compiled.export_model(blob);
    EXPECT_FALSE(blob.str().empty()) << "empty NPU blob for " << soc;
  }
}

}  // namespace
}  // namespace litert
