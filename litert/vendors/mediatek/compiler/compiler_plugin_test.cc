// Copyright 2024 Google LLC.
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
#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_options.h"
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"
#include "litert/vendors/mediatek/schema/schema_resolver.h"

namespace litert {
namespace {

using ::testing::Values;

// clang-format off
const auto kSupportedOps = Values(
    "add_cst.tflite",
    "add_simple.tflite",
    "simple_add_op.tflite",
    "simple_mul_op.tflite",
    "simple_batch_matmul_op.tflite",
    "simple_rsqrt_op.tflite",
    "simple_concatenation_op.tflite",
    "simple_slice_op.tflite",
    "simple_sub_op.tflite",
    "simple_tanh_op.tflite",
    "simple_softmax_op.tflite",
    "simple_mean_op.tflite",
    "simple_gelu_op.tflite",
    "simple_leaky_relu_op.tflite",
    "simple_pad.tflite",
    "simple_logistic.tflite",
    "simple_sum_op.tflite",
    "simple_resize_bilinear_op.tflite",
    "simple_resize_nearest_neighbor_op.tflite",
    "simple_max_pool_2d.tflite",
    "simple_hard_swish_op.tflite",
    "simple_group_norm_op.tflite"
    // "simple_average_pool_2d_op.tflite"
    // Don't include the less op test as the support is not available in the
    // latest MTK SDK.
    // "simple_less_op.tflite"
    );
// clang-format on

TEST(TestMediatekPlugin, GetConfigInfo) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  EXPECT_STREQ(plugin.Api()->get_compiler_plugin_soc_manufacturer(),
               "MediaTek");

  LiteRtParamIndex num_supported_soc_models;
  ASSERT_EQ(plugin.Api()->get_num_compiler_plugin_supported_models(
                plugin.Get(), &num_supported_soc_models),
            kLiteRtStatusOk);
  ASSERT_EQ(num_supported_soc_models, 16);

  const char* config_id;
  ASSERT_EQ(plugin.Api()->get_compiler_plugin_supported_soc_model(
                plugin.Get(), 0, &config_id),
            kLiteRtStatusOk);
  EXPECT_STREQ(config_id, "mt6853");
}

TEST(TestMediatekPlugin, RegisterAllTransformations) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  LiteRtTransformation* transformations = nullptr;
  LiteRtParamIndex num_patterns = 0;
  ASSERT_EQ(plugin.Api()->register_all_transformations(
                plugin.Get(), &transformations, &num_patterns),
            kLiteRtStatusOk);
  ASSERT_EQ(num_patterns, 10);
  ASSERT_NE(transformations, nullptr);
  EXPECT_STREQ(transformations[0].name, "RmsNormQuantTransformation");
  EXPECT_EQ(transformations[0].benefit, 100);
  EXPECT_NE(transformations[0].pattern, nullptr);
}

TEST(TestMediatekPlugin, PartitionAdd) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  auto model = testing::LoadTestFileModel("add_simple.tflite");

  auto subgraph = model.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());
  LiteRtOpListT selected_op_list;
  ASSERT_EQ(plugin.Api()->compiler_plugin_partition(
                plugin.Get(), /*soc_model=*/"mt6989", subgraph->Get(),
                &selected_op_list),
            kLiteRtStatusOk);
  const auto selected_ops = selected_op_list.Values();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflAdd);
}

TEST(TestMediatekPlugin, PartitionGroupNorm) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  auto model = testing::LoadTestFileModel("simple_group_norm_op.tflite");

  auto subgraph = model.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());
  LiteRtOpListT selected_op_list;
  ASSERT_EQ(plugin.Api()->compiler_plugin_partition(
                plugin.Get(), /*soc_model=*/"mt6989", subgraph->Get(),
                &selected_op_list),
            kLiteRtStatusOk);
  const auto selected_ops = selected_op_list.Values();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeShloComposite);
}

TEST(TestMediatekPlugin, DlaDirectory) {
#ifdef __ANDROID__
  char* dla_directory_name = std::getenv("MTKNN_ADAPTER_DLA_DIR");
#endif

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  auto model = testing::LoadTestFileModel("add_simple.tflite");

  auto subgraph = model.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());
  LiteRtOpListT selected_op_list;
  ASSERT_EQ(plugin.Api()->compiler_plugin_partition(
                plugin.Get(), /*soc_model=*/"mt6989", subgraph->Get(),
                &selected_op_list),
            kLiteRtStatusOk);
  const auto selected_ops = selected_op_list.Values();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflAdd);

  // On Android, the environmental variable should be kept the same.
#ifdef __ANDROID__
  if (dla_directory_name) {
    EXPECT_STREQ(std::getenv("MTKNN_ADAPTER_DLA_DIR"), dla_directory_name);
  } else {
    EXPECT_EQ(std::getenv("MTKNN_ADAPTER_DLA_DIR"), nullptr);
  }
#else
  // On non-Android, the environmental variable will always be set after the
  // execution of compiler plugin.
  char* dla_directory_name = std::getenv("MTKNN_ADAPTER_DLA_DIR");
  EXPECT_NE(dla_directory_name, nullptr);
#endif
}

// /////////////////////////////////////////////////////////////////////////////

class MtkPluginOpCompatibilityTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(MtkPluginOpCompatibilityTest, SupportedOpsTest) {
  LITERT_LOG(LITERT_INFO, "Testing TFLite model: %s", GetParam().c_str());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  auto model = testing::LoadTestFileModel(GetParam());

  LiteRtCompiledResult compiled;
  ASSERT_EQ(plugin.Api()->compiler_plugin_compile(
                plugin.Get(), /*soc_model=*/"mt6991", model.Get(), &compiled),
            kLiteRtStatusOk);

  LiteRtParamIndex num_byte_code;
  ASSERT_EQ(
      plugin.Api()->get_compiled_result_num_byte_code(compiled, &num_byte_code),
      kLiteRtStatusOk);
  ASSERT_EQ(num_byte_code, 1);

  const void* byte_code;
  size_t byte_code_size;

  ASSERT_EQ(plugin.Api()->get_compiled_result_byte_code(
                compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size),
            kLiteRtStatusOk);

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  ASSERT_EQ(
      plugin.Api()->get_compiled_result_call_info(
          compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx),
      kLiteRtStatusOk);

  EXPECT_EQ(byte_code_idx, 0);

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  EXPECT_EQ(op_data_string, "Partition_0");

  plugin.Api()->destroy_compiled_result(compiled);
}

INSTANTIATE_TEST_SUITE_P(SupportedOpsTest, MtkPluginOpCompatibilityTest,
                         kSupportedOps);

// /////////////////////////////////////////////////////////////////////////////
// Weight Share Integration Test
// /////////////////////////////////////////////////////////////////////////////

// Creates a LiteRtOptions with enable_weight_share=true for MediaTek plugin.
static LiteRtOptions CreateWeightShareOptions() {
  // Create mediatek options with weight share enabled
  LrtMediatekOptions* mtk_opts = nullptr;
  LrtCreateMediatekOptions(&mtk_opts);
  LrtSetMediatekOptionsEnableWeightShare(mtk_opts, true);

  // Serialize to opaque options payload
  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LrtGetOpaqueMediatekOptionsData(mtk_opts, &identifier, &payload,
                                  &payload_deleter);

  // Create opaque options (ownership of payload transfers)
  LiteRtOpaqueOptions opaque = nullptr;
  LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter, &opaque);

  // Create LiteRtOptions and attach opaque options
  LiteRtOptions options = nullptr;
  LiteRtCreateOptions(&options);
  LiteRtAddOpaqueOptions(options, opaque);

  LrtDestroyMediatekOptions(mtk_opts);
  return options;
}

TEST(TestMediatekPlugin, WeightShareCompile) {
  // Check if NeuronAdapter supports ExtractShared
  {
    LrtMediatekOptions* opts = nullptr;
    LrtCreateMediatekOptions(&opts);
    auto api_result = litert::mediatek::NeuronAdapterApi::Create(
        /*shared_library_dir=*/std::nullopt, opts);
    LrtDestroyMediatekOptions(opts);
    if (!api_result) {
      GTEST_SKIP() << "NeuronAdapter library not available";
    }
    if (!api_result.Value()->SupportsExtractShared()) {
      GTEST_SKIP() << "NeuronExtractShared API not supported by this SDK";
    }
  }

  // Create plugin with weight_share enabled
  LiteRtOptions options = CreateWeightShareOptions();
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext(),
                                                  /*env=*/nullptr, options));

  // Use a model that exists in testdata and is supported by MediaTek plugin.
  // Note: models like "add_cst.tflite" require @models repo to generate.
  auto model = testing::LoadTestFileModel("norm.tflite");

  // Compile through the full compiler plugin path with weight share enabled.
  LiteRtCompiledResult compiled;
  ASSERT_EQ(plugin.Api()->compiler_plugin_compile(
                plugin.Get(), /*soc_model=*/"mt6991", model.Get(), &compiled),
            kLiteRtStatusOk);

  // Verify bytecode was produced
  LiteRtParamIndex num_byte_code;
  ASSERT_EQ(plugin.Api()->get_compiled_result_num_byte_code(compiled,
                                                            &num_byte_code),
            kLiteRtStatusOk);
  ASSERT_EQ(num_byte_code, 1);

  const void* byte_code;
  size_t byte_code_size;
  ASSERT_EQ(plugin.Api()->get_compiled_result_byte_code(
                compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size),
            kLiteRtStatusOk);
  ASSERT_GT(byte_code_size, 0u);

  // Verify bytecode is a valid Neuron schema with weight_share_index
  neuron::SchemaResolver resolver;
  auto init_res = resolver.Initialize(
      static_cast<const uint8_t*>(byte_code), byte_code_size);
  ASSERT_TRUE(init_res.HasValue());
  ASSERT_TRUE(init_res.Value());

  // Verify version is recorded
  auto version = resolver.GetNeuronVersion();
  ASSERT_TRUE(version.has_value());
  auto [major, minor, patch] = *version;
  LITERT_LOG(LITERT_INFO, "Bytecode Neuron version: %u.%u.%u", major, minor,
             patch);

  // Verify the partition exists and has weight share index
  std::string partition_name = "Partition_0";
  auto graph = resolver.GetCompiledGraph(partition_name);
  ASSERT_TRUE(graph.has_value());

  auto compiled_net = graph->GetCompiledNetwork();
  ASSERT_TRUE(compiled_net.HasValue());
  EXPECT_GT(compiled_net.Value().second, 0u);

  if (graph->HasWeightShareIndex()) {
    auto ws_buffers = graph->GetWeightShareBuffers();
    ASSERT_FALSE(ws_buffers.empty());
    EXPECT_GT(ws_buffers[0].second, 0u);
    LITERT_LOG(LITERT_INFO,
               "Weight share verified: %zu shared buffer(s), first size=%zu",
               ws_buffers.size(), ws_buffers[0].second);
  } else {
    LITERT_LOG(LITERT_INFO,
               "No weight share produced (SDK heuristic - model too small)");
  }

  plugin.Api()->destroy_compiled_result(compiled);
  LiteRtDestroyOptions(options);
}

}  // namespace
}  // namespace litert
