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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"

namespace litert {
namespace {

using ::testing::Values;

// clang-format off
// TODO: Add support and uncomment these models.
const auto kSupportedOps =
                  Values(
                    "fully_connected_3d.tflite",
                    "rms_norm.tflite",
                    "rms_norm_composite.tflite",
                    "simple_abs_op.tflite",
                    "simple_add_fused_relu_n1_1_op.tflite",
                    "simple_add_op.tflite",
                    "simple_arg_max_op.tflite",
                    "simple_arg_min_op.tflite",
                    "simple_average_poll_2d.tflite",
                    "simple_average_pool_2d_fused_relu.tflite",
                    "simple_batch_matmul_op.tflite",
                    "simple_cast_op.tflite",
                    "simple_ceil_op.tflite",
                    "simple_concatenation_fused_relu6_op.tflite",
                    "simple_concatenation_op.tflite",
                    "simple_conv_2d_fused_relu_op.tflite",
                    "simple_conv_2d_op.tflite",
                    "simple_conv_3d_op.tflite",
                    "simple_cos_op.tflite",
                    "simple_cumsum.tflite",
                    "simple_depth_to_space_op.tflite",
                    "simple_depthwise_conv_2d_fused_relu.tflite",
                    "simple_depthwise_conv_2d_op.tflite",
                    "simple_div_fused_tanh.tflite",
                    "simple_div_op.tflite",
                    "simple_dynamic_update_slice_op.tflite",
                    "simple_elu_op.tflite",
                    "simple_embedding_lookup_op.tflite",
                    "simple_equal_op.tflite",
                    "simple_exp_op.tflite",
                    "simple_floor_op.tflite",
                    "simple_floor_div.tflite",
                    "simple_fully_connected_fused_relu6_op.tflite",
                    "simple_fully_connected_op.tflite",
                    "simple_gather_nd.tflite",
                    "simple_gather_op.tflite",
                    "simple_gelu_op.tflite",
                    "simple_greater_op.tflite",
                    "simple_hard_swish_op.tflite",
                    "simple_leaky_relu_op.tflite",
                    "simple_less_op.tflite",
                    "simple_log_op.tflite",
                    "simple_logical_and_op.tflite",
                    "simple_logical_or_op.tflite",
                    "simple_logistic.tflite",
                    "simple_max_pool_2d.tflite",
                    "simple_max_pool_2d_fused_relu.tflite",
                    "simple_mean_op.tflite",
                    "simple_mirror_pad_reflect_op.tflite",
                    "simple_mirror_pad_symmetric_op.tflite",
                    "simple_mul_fused_relu.tflite",
                    "simple_mul_op.tflite",
                    "simple_neg_op.tflite",
                    "simple_not_equal.tflite",
                    "simple_pack_op.tflite",
                    "simple_pad.tflite",
                    "simple_pad_v2.tflite",
                    "simple_reducemax_op.tflite",
                    "simple_reducemin_op.tflite",
                    "simple_reduceall_op.tflite",
                    "simple_reduceany_op.tflite",
                    "simple_relu_op.tflite",
                    "simple_relu1_op.tflite",
                    "simple_relu0to1_op.tflite",
                    "simple_reshape_op.tflite",
                    "simple_resize_bilinear_op.tflite",
                    "simple_resize_nearest_neighbor_op.tflite",
                    "simple_reverse_op.tflite",
                    "simple_round_op.tflite",
                    "simple_rsqrt_op.tflite",
                    "simple_select_op.tflite",
                    "simple_select_v2_op.tflite",
                    "simple_sign_op.tflite",
                    "simple_sin_op.tflite",
                    "simple_slice_op.tflite",
                    "simple_softmax_op.tflite",
                    "simple_space_to_depth_op.tflite",
                    "simple_split_op.tflite",
                    "simple_strided_slice_op.tflite",
                    "simple_sqrt_op.tflite",
                    "simple_sub_fused_relu_N1_1_op.tflite",
                    "simple_sub_op.tflite",
                    "simple_sum_op.tflite",
                    "simple_tanh_op.tflite",
                    "simple_tile_op.tflite",
                    "simple_transpose_conv_fused_tanh.tflite",
                    "simple_transpose_conv_op.tflite",
                    "simple_transpose_op.tflite",
                    "simple_unpack_op.tflite",
                    "simple_prelu_op.tflite",
                    "simple_l2_norm.tflite",
                    "l2_norm_composite.tflite",
                    kFeedForwardModel,
                    kKeyEinsumModel,
                    kQueryEinsumModel,
                    kValueEinsumModel,
                    kAttnVecEinsumModel,
                    kROPEModel,
                    kLookUpROPEModel,
                    kRMSNormModel,
                    kSDPAModel,
                    kAttentionModel,
                    kTransformerBlockModel,
                    kQSimpleMul16x16Model,
                    kQMulAdd16x16Model,
                    kQQueryEinsum16x8Model,
                    kQKeyEinsum16x8Model,
                    kQVauleEinsum16x8Model,
                    kQAttnVecEinsum16x8Model
                    );

const auto kSupportedSocModels = Values(
    "SA8295",
    "SA8255",
    "SM8350",
    "SM8450",
    "SM8475",
    "SM8550",
    "SM8650",
    "SM8750"
);
// clang-format on

TEST(TestQnnPlugin, GetConfigInfo) {
  EXPECT_STREQ(LiteRtGetCompilerPluginSocManufacturer(), "Qualcomm");

  auto plugin = CreatePlugin();

  LiteRtParamIndex num_supported_soc_models;
  LITERT_ASSERT_OK(LiteRtGetNumCompilerPluginSupportedSocModels(
      plugin.get(), &num_supported_soc_models));
  ASSERT_EQ(num_supported_soc_models, 9);

  const char* config_id;
  LITERT_ASSERT_OK(
      LiteRtGetCompilerPluginSupportedSocModel(plugin.get(), 0, &config_id));
  EXPECT_STREQ(config_id, "UNKNOWN_SDM");
}

TEST(TestQnnPlugin, PartitionMulOps) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflMul);
}

TEST(TestQnnPlugin, CompileMulSubgraph) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), "SM8650",
                                               model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtDestroyCompiledResult(compiled);
}

TEST(TestQnnPlugin, CompileMulSubgraphWithOptions) {
  auto opts = Options::Create();
  ASSERT_TRUE(opts);

  auto qnn_opts = qualcomm::QualcommOptions::Create();
  ASSERT_TRUE(qnn_opts);
  qnn_opts->SetLogLevel(kLiteRtQualcommLogLevelError);
  qnn_opts->SetEnableWeightSharing(false);

  ASSERT_TRUE(opts->AddOpaqueOptions(std::move(*qnn_opts)));

  auto plugin = CreatePlugin(/*env=*/nullptr, opts->Get());
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), "SM8650",
                                               model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtDestroyCompiledResult(compiled);
}

TEST(TestQnnPlugin, ShareContextBinary) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("cst_multi_subgraph.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), "SM8650",
                                               model.Get(), &compiled));
  uint64_t num_byte_code;
  LITERT_ASSERT_OK(
      LiteRtCompiledResultNumByteCodeModules(compiled, &num_byte_code));
  ASSERT_EQ(num_byte_code, 1);

  LiteRtDestroyCompiledResult(compiled);
}

TEST(TestQnnPlugin, NotShareContextBinary) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("multi_subgraph.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), "SM8650",
                                               model.Get(), &compiled));
  uint64_t num_byte_code;
  LITERT_ASSERT_OK(
      LiteRtCompiledResultNumByteCodeModules(compiled, &num_byte_code));
  ASSERT_EQ(num_byte_code, 3);

  LiteRtDestroyCompiledResult(compiled);
}

class QnnPlyginSupportedSocCompilationTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(QnnPlyginSupportedSocCompilationTest, CompileMulSubgraph) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("one_mul.tflite");
  auto soc_model = GetParam();
#ifdef __ANDROID__
  if (soc_model != "V75") {
    // TODO: Make this dynamic when device cloud testing has more devices.
    GTEST_SKIP() << "On device tests only support V75s.";
  }
#endif

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), soc_model.c_str(),
                                               model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtDestroyCompiledResult(compiled);
}

INSTANTIATE_TEST_SUITE_P(SupportedOpsTest, QnnPlyginSupportedSocCompilationTest,
                         kSupportedSocModels);

class QnnPluginOpValidationTest : public ::testing::TestWithParam<std::string> {
};

TEST_P(QnnPluginOpValidationTest, SupportedOpsTest) {
  LITERT_LOG(LITERT_INFO, "Validating TFLite model: %s", GetParam().c_str());
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel(GetParam());

  const auto subgraph = model.MainSubgraph();
  LiteRtSubgraph litert_subgraph = subgraph->Get();

  LiteRtOpListT selected_ops;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, litert_subgraph, &selected_ops));

  EXPECT_EQ(selected_ops.Values().size(), litert_subgraph->Ops().size());
}

INSTANTIATE_TEST_SUITE_P(SupportedOpsTest, QnnPluginOpValidationTest,
                         kSupportedOps);

class QnnPluginOpCompatibilityTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(QnnPluginOpCompatibilityTest, SupportedOpsTest) {
  LITERT_LOG(LITERT_INFO, "Testing TFLite model: %s", GetParam().c_str());
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel(GetParam());

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), "SM8650",
                                               model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtDestroyCompiledResult(compiled);
}

INSTANTIATE_TEST_SUITE_P(SupportedOpsTest, QnnPluginOpCompatibilityTest,
                         kSupportedOps);

}  // namespace
}  // namespace litert
