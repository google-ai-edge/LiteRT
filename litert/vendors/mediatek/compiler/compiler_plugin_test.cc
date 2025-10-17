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
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/test_models.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
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
    "simple_pad.tflite",
    "simple_logistic.tflite",
    "simple_sum_op.tflite",
    "simple_resize_bilinear_op.tflite",
    "simple_resize_nearest_neighbor_op.tflite",
    "simple_max_pool_2d.tflite",
    "simple_hard_swish_op.tflite"
    // "simple_average_pool_2d_op.tflite"
    );
// clang-format on

TEST(TestMediatekPlugin, GetConfigInfo) {
  EXPECT_STREQ(LiteRtGetCompilerPluginSocManufacturer(), "MediaTek");

  auto plugin = CreatePlugin();

  LiteRtParamIndex num_supported_soc_models;
  ASSERT_EQ(LiteRtGetNumCompilerPluginSupportedSocModels(
                plugin.get(), &num_supported_soc_models),
            kLiteRtStatusOk);
  ASSERT_EQ(num_supported_soc_models, 16);

  const char* config_id;
  ASSERT_EQ(
      LiteRtGetCompilerPluginSupportedSocModel(plugin.get(), 0, &config_id),
      kLiteRtStatusOk);
  EXPECT_STREQ(config_id, "mt6853");
}

TEST(TestMediatekPlugin, PartitionAdd) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("add_simple.tflite");

  auto subgraph = model.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());
  LiteRtOpListT selected_op_list;
  ASSERT_EQ(LiteRtCompilerPluginPartition(plugin.get(), /*soc_model=*/"mt6989",
                                          subgraph->Get(), &selected_op_list),
            kLiteRtStatusOk);
  const auto selected_ops = selected_op_list.Values();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflAdd);
}

// /////////////////////////////////////////////////////////////////////////////

class MtkPluginOpCompatibilityTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(MtkPluginOpCompatibilityTest, SupportedOpsTest) {
  LITERT_LOG(LITERT_INFO, "Testing TFLite model: %s", GetParam().c_str());
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel(GetParam());

  LiteRtCompiledResult compiled;
  ASSERT_EQ(LiteRtCompilerPluginCompile(plugin.get(), /*soc_model=*/"mt6991",
                                        model.Get(), &compiled),
            kLiteRtStatusOk);

  LiteRtParamIndex num_byte_code;
  ASSERT_EQ(LiteRtCompiledResultNumByteCodeModules(compiled, &num_byte_code),
            kLiteRtStatusOk);
  ASSERT_EQ(num_byte_code, 1);

  const void* byte_code;
  size_t byte_code_size;

  ASSERT_EQ(LiteRtGetCompiledResultByteCode(compiled, /*byte_code_idx=*/0,
                                            &byte_code, &byte_code_size),
            kLiteRtStatusOk);

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  ASSERT_EQ(LiteRtGetCompiledResultCallInfo(compiled, /*call_idx=*/0, &op_data,
                                            &op_data_size, &byte_code_idx),
            kLiteRtStatusOk);

  EXPECT_EQ(byte_code_idx, 0);

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  EXPECT_EQ(op_data_string, "Partition_0");

  LiteRtDestroyCompiledResult(compiled);
}

INSTANTIATE_TEST_SUITE_P(SupportedOpsTest, MtkPluginOpCompatibilityTest,
                         kSupportedOps);

}  // namespace
}  // namespace litert
