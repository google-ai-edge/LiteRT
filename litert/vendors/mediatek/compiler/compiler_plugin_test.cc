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
#include <cstdlib>
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
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"

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
    "simple_hard_swish_op.tflite"
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

}  // namespace
}  // namespace litert
