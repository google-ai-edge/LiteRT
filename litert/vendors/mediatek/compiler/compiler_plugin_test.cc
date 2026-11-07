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
#include <cstring>
#include <fstream>
#include <ios>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"
#include "litert/vendors/mediatek/compiler/compile_model.h"
#include "litert/vendors/mediatek/schema/neuron_schema_generated.h"
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

TEST(TestMediatekPlugin, SchemaWeightShareResolutionAndDeduplication) {
  neuron::BytecodeBuilder builder;
  const int8_t dummy_dla[] = {1, 2, 3, 4};
  const int8_t weight_a[] = {10, 20, 30, 40, 50, 60};
  const int8_t weight_b[] = {70, 80, 90};

  int32_t dla_idx_0 =
      builder.AddBuffer("Partition_0", dummy_dla, sizeof(dummy_dla));
  int32_t w_idx_a0 =
      builder.AddSharedWeightBuffer("w_a0", weight_a, sizeof(weight_a));
  int32_t w_idx_b0 =
      builder.AddSharedWeightBuffer("w_b0", weight_b, sizeof(weight_b));
  builder.AddCompiledNetwork("Partition_0",
                             NeuronSchema::CompiledType_AdapterCache, dla_idx_0,
                             {w_idx_a0, w_idx_b0});

  int32_t dla_idx_1 =
      builder.AddBuffer("Partition_1", dummy_dla, sizeof(dummy_dla));
  int32_t w_idx_a1 =
      builder.AddSharedWeightBuffer("w_a1", weight_a, sizeof(weight_a));
  EXPECT_EQ(w_idx_a0, w_idx_a1);
  builder.AddCompiledNetwork("Partition_1",
                             NeuronSchema::CompiledType_AdapterCache, dla_idx_1,
                             {w_idx_a1});

  ASSERT_TRUE(builder.Finish());
  auto [raw_buf, raw_size] = builder.GetBytecode();

  neuron::SchemaResolver resolver;
  auto init_res = resolver.Initialize(raw_buf, raw_size);
  ASSERT_TRUE(init_res.HasValue());
  ASSERT_TRUE(init_res.Value());

  auto graph_0 = resolver.GetCompiledGraph("Partition_0");
  ASSERT_TRUE(graph_0.has_value());
  auto weights_0 = graph_0->GetWeightShareBuffers();
  ASSERT_TRUE(weights_0.HasValue());
  ASSERT_EQ(weights_0->size(), 2);
  EXPECT_EQ((*weights_0)[0].second, sizeof(weight_a));
  EXPECT_EQ((*weights_0)[1].second, sizeof(weight_b));

  auto graph_1 = resolver.GetCompiledGraph("Partition_1");
  ASSERT_TRUE(graph_1.has_value());
  auto weights_1 = graph_1->GetWeightShareBuffers();
  ASSERT_TRUE(weights_1.HasValue());
  ASSERT_EQ(weights_1->size(), 1);
  EXPECT_EQ((*weights_1)[0].first, (*weights_0)[0].first);
  EXPECT_EQ((*weights_1)[0].second, sizeof(weight_a));
}

TEST(TestMediatekPlugin, CompileModelWithStaticSharing) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, StaticallyLinkedPlugin::Create(LrtGetCompilerContext()));
  auto model = testing::LoadTestFileModel("add_cst.tflite");

  LiteRtCompiledResult compiled;
  ASSERT_EQ(plugin.Api()->compiler_plugin_compile(
                plugin.Get(), /*soc_model=*/"mt6991", model.Get(), &compiled),
            kLiteRtStatusOk);

  const void* byte_code = nullptr;
  size_t byte_code_size = 0;
  ASSERT_EQ(plugin.Api()->get_compiled_result_byte_code(
                compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size),
            kLiteRtStatusOk);

  neuron::SchemaResolver resolver;
  auto init_res = resolver.Initialize(
      reinterpret_cast<const uint8_t*>(byte_code), byte_code_size);
  ASSERT_TRUE(init_res.HasValue());
  ASSERT_TRUE(init_res.Value());

  auto graph = resolver.GetCompiledGraph("Partition_0");
  ASSERT_TRUE(graph.has_value());

  auto weight_buffers = graph->GetWeightShareBuffers();
  ASSERT_TRUE(weight_buffers.HasValue());

  auto dla_bytecode = graph->GetCompiledNetwork();
  ASSERT_TRUE(dla_bytecode.HasValue());
  EXPECT_GT(dla_bytecode->second, 0);

  plugin.Api()->destroy_compiled_result(compiled);
}

TEST(TestMediatekPlugin, ParseExtractedStaticWeightsNonExistentFile) {
  auto parsed =
      mediatek::ParseExtractedStaticWeights("/tmp/non_existent_weights.bin");
  ASSERT_TRUE(parsed.HasValue());
  EXPECT_TRUE(parsed->empty());
}

TEST(TestMediatekPlugin, ParseExtractedStaticWeightsValidFile) {
  auto temp_dir = testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir.HasValue());
  std::string file_path = std::string(temp_dir->Str()) + "/weights.bin";

  // Create synthetic weight file with 2 entries:
  // Payload: weight_1 = 4 bytes {1, 2, 3, 4}, weight_2 = 3 bytes {5, 6, 7}
  // Offset 0: {1, 2, 3, 4} (size 4)
  // Offset 4: {5, 6, 7} (size 3)
  // Offset 7: JSON metadata: {"2": ["4", "3"], "1": ["0", "4"]}
  // json_offset = 7
  // footer: uint64_t(7), uint32_t(0xbbced1ec)
  std::ofstream out(file_path, std::ios::binary);
  const char payload[] = {1, 2, 3, 4, 5, 6, 7};
  out.write(payload, sizeof(payload));
  std::string json_str = "{\"2\": [\"4\", \"3\"], \"1\": [\"0\", \"4\"]}";
  out.write(json_str.data(), json_str.size());
  uint64_t json_offset = sizeof(payload);
  uint32_t magic = 0xbbced1ecU;
  out.write(reinterpret_cast<const char*>(&json_offset), sizeof(json_offset));
  out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  out.close();

  auto parsed = mediatek::ParseExtractedStaticWeights(file_path);
  ASSERT_TRUE(parsed.HasValue());
  ASSERT_EQ(parsed->size(), 2);
  EXPECT_EQ((*parsed)[0].size(), 4);
  EXPECT_EQ((*parsed)[0][0], 1);
  EXPECT_EQ((*parsed)[0][3], 4);
  EXPECT_EQ((*parsed)[1].size(), 3);
  EXPECT_EQ((*parsed)[1][0], 5);
  EXPECT_EQ((*parsed)[1][2], 7);
}

TEST(TestMediatekPlugin, ParseExtractedStaticWeightsInvalidMagic) {
  auto temp_dir = testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir.HasValue());
  std::string file_path = std::string(temp_dir->Str()) + "/bad_weights.bin";

  std::ofstream out(file_path, std::ios::binary);
  const char payload[] = {1, 2, 3, 4};
  out.write(payload, sizeof(payload));
  std::string json_str = "{\"1\": [\"0\", \"4\"]}";
  out.write(json_str.data(), json_str.size());
  uint64_t json_offset = sizeof(payload);
  uint32_t bad_magic = 0x12345678U;
  out.write(reinterpret_cast<const char*>(&json_offset), sizeof(json_offset));
  out.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
  out.close();

  auto parsed = mediatek::ParseExtractedStaticWeights(file_path);
  EXPECT_FALSE(parsed.HasValue());
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

}  // namespace
}  // namespace litert
