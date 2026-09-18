// Copyright 2025 Google LLC.
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

#include "litert/cc/options/litert_gpu_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/test/matchers.h"

using ::testing::Eq;
using ::testing::StrEq;

namespace litert::ml_drift {
namespace {

TEST(GpuOptions, EnableConstantTensorSharingWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // An unset option resolves to the caller-supplied default.
  bool constant_tensor_sharing = true;
  LITERT_ASSERT_OK(LrtGetGpuOptionsConstantTensorsSharing(
      &constant_tensor_sharing, /*default_value=*/false, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(false));

  options.EnableConstantTensorSharing(true);

  LITERT_ASSERT_OK(LrtGetGpuOptionsConstantTensorsSharing(
      &constant_tensor_sharing, /*default_value=*/false, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(true));

  options.EnableConstantTensorSharing(false);

  LITERT_ASSERT_OK(LrtGetGpuOptionsConstantTensorsSharing(
      &constant_tensor_sharing, /*default_value=*/false, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(false));
}

TEST(GpuOptions, EnableInfiniteFloatCappingWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // An unset option resolves to the caller-supplied default.
  bool infinite_float_capping = true;
  LITERT_ASSERT_OK(LrtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, /*default_value=*/false, payload));
  EXPECT_THAT(infinite_float_capping, Eq(false));

  options.EnableInfiniteFloatCapping(true);

  LITERT_ASSERT_OK(LrtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, /*default_value=*/false, payload));
  EXPECT_THAT(infinite_float_capping, Eq(true));

  options.EnableInfiniteFloatCapping(false);

  LITERT_ASSERT_OK(LrtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, /*default_value=*/false, payload));
  EXPECT_THAT(infinite_float_capping, Eq(false));
}

TEST(GpuOptions, EnableBenchmarkModeWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // An unset option resolves to the caller-supplied default.
  bool benchmark_mode = true;
  LITERT_ASSERT_OK(LrtGetGpuOptionsBenchmarkMode(
      &benchmark_mode, /*default_value=*/false, payload));
  EXPECT_THAT(benchmark_mode, Eq(false));

  options.EnableBenchmarkMode(true);

  LITERT_ASSERT_OK(LrtGetGpuOptionsBenchmarkMode(
      &benchmark_mode, /*default_value=*/false, payload));
  EXPECT_THAT(benchmark_mode, Eq(true));

  options.EnableBenchmarkMode(false);

  LITERT_ASSERT_OK(LrtGetGpuOptionsBenchmarkMode(
      &benchmark_mode, /*default_value=*/false, payload));
  EXPECT_THAT(benchmark_mode, Eq(false));
}

TEST(GpuAcceleratorCompilationOptions,
     EnableAllowSrcQuantizedFcConvOpsCheckTrue) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  bool allow_src_quantized_fc_conv_ops = false;
  LITERT_EXPECT_OK(options.EnableAllowSrcQuantizedFcConvOps(true));

  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &allow_src_quantized_fc_conv_ops, /*default_value=*/false, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(true));
}

TEST(GpuAcceleratorCompilationOptions,
     EnableAllowSrcQuantizedFcConvOpsCheckDefaultValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // Check the default value.
  bool allow_src_quantized_fc_conv_ops = true;

  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &allow_src_quantized_fc_conv_ops, /*default_value=*/false, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(false));
}

TEST(GpuAcceleratorCompilationOptions,
     EnableAllowSrcQuantizedFcConvOpsCheckFalseValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // The default value is false, set it to true before resetting to false.
  LITERT_EXPECT_OK(options.EnableAllowSrcQuantizedFcConvOps(true));
  LITERT_EXPECT_OK(options.EnableAllowSrcQuantizedFcConvOps(false));
  bool allow_src_quantized_fc_conv_ops = true;

  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &allow_src_quantized_fc_conv_ops, /*default_value=*/false, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(false));
}

TEST(GpuAcceleratorCompilationOptions, CheckDelegatePrecisionDefaultPrecision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // An unset option resolves to the caller-supplied default.
  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionFp16;
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsPrecision(
      &precision, kLiteRtDelegatePrecisionDefault, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionDefault));
}

TEST(GpuAcceleratorCompilationOptions, SetDelegatePrecisionFp16Precision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;
  options.SetPrecision(GpuOptions::Precision::kFp16);

  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsPrecision(
      &precision, kLiteRtDelegatePrecisionDefault, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionFp16));
}

TEST(GpuAcceleratorCompilationOptions, SetDelegatePrecisionFp32Precision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;

  options.SetPrecision(GpuOptions::Precision::kFp32);

  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsPrecision(
      &precision, kLiteRtDelegatePrecisionDefault, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionFp32));
}

TEST(GpuAcceleratorCompilationOptions, SetSerializationDir) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // An unset option resolves to the caller-supplied default.
  const char* serialization_dir = nullptr;
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
      &serialization_dir, /*default_value=*/nullptr, payload));
  EXPECT_EQ(serialization_dir, nullptr);

  options.SetSerializationDir("/data/local/tmp");
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
      &serialization_dir, /*default_value=*/nullptr, payload));
  EXPECT_THAT(serialization_dir, StrEq("/data/local/tmp"));
}

TEST(GpuAcceleratorCompilationOptions, SetModelCacheKey) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // An unset option resolves to the caller-supplied default.
  const char* model_cache_key = nullptr;
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
      &model_cache_key, /*default_value=*/nullptr, payload));
  EXPECT_EQ(model_cache_key, nullptr);

  options.SetModelCacheKey("model_cache");
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
      &model_cache_key, /*default_value=*/nullptr, payload));
  EXPECT_THAT(model_cache_key, StrEq("model_cache"));
}

TEST(GpuAcceleratorCompilationOptions, SetProgramCacheFd) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // An unset option resolves to the caller-supplied default.
  int program_cache_fd = -1;
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
      &program_cache_fd, /*default_value=*/-1, payload));
  EXPECT_EQ(program_cache_fd, -1);

  options.SetProgramCacheFd(123);
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
      &program_cache_fd, /*default_value=*/-1, payload));
  EXPECT_EQ(program_cache_fd, 123);
}

TEST(GpuAcceleratorCompilationOptions, SetSerializeProgramCache) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // An unset option resolves to the caller-supplied default.
  bool serialize_program_cache = false;
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
      &serialize_program_cache, /*default_value=*/true, payload));
  EXPECT_EQ(serialize_program_cache, true);

  options.SetSerializeProgramCache(false);
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
      &serialize_program_cache, /*default_value=*/true, payload));
  EXPECT_EQ(serialize_program_cache, false);
}

TEST(GpuAcceleratorCompilationOptions, SetSerializeExternalTensors) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // Check the default value.
  bool serialize_external_tensors = false;
  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
          &serialize_external_tensors, /*default_value=*/false, payload));
  EXPECT_EQ(serialize_external_tensors, false);

  options.SetSerializeExternalTensors(true);
  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
          &serialize_external_tensors, /*default_value=*/false, payload));
  EXPECT_EQ(serialize_external_tensors, true);
}

TEST(GpuAcceleratorCompilationOptions, SetPreferTextureWeights) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // Check the default value.
  bool prefer_texture_weights = false;
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
      &prefer_texture_weights, /*default_value=*/false, payload));
  EXPECT_EQ(prefer_texture_weights, false);

  options.SetPreferTextureWeights(true);
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
      &prefer_texture_weights, /*default_value=*/false, payload));
  EXPECT_EQ(prefer_texture_weights, true);
}

#ifdef __APPLE__
TEST(GpuOptions, SetUseMetalArgumentBuffersWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // Check the default value.
  bool use_metal_argument_buffers = true;
  LITERT_ASSERT_OK(LrtGetGpuOptionsUseMetalArgumentBuffers(
      &use_metal_argument_buffers, /*default_value=*/false, payload));
  EXPECT_THAT(use_metal_argument_buffers, Eq(false));

  options.SetUseMetalArgumentBuffers(true);

  LITERT_ASSERT_OK(LrtGetGpuOptionsUseMetalArgumentBuffers(
      &use_metal_argument_buffers, /*default_value=*/false, payload));
  EXPECT_THAT(use_metal_argument_buffers, Eq(true));

  options.SetUseMetalArgumentBuffers(false);

  LITERT_ASSERT_OK(LrtGetGpuOptionsUseMetalArgumentBuffers(
      &use_metal_argument_buffers, /*default_value=*/false, payload));
  EXPECT_THAT(use_metal_argument_buffers, Eq(false));
}
#endif  // __APPLE__

TEST(GpuAcceleratorCompilationOptions, SetHintFullyDelegatedToSingleDelegate) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // Check the default value.
  bool hint_fully_delegated_to_single_delegate = false;
  LITERT_ASSERT_OK(LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
      &hint_fully_delegated_to_single_delegate, /*default_value=*/false,
      payload));
  EXPECT_EQ(hint_fully_delegated_to_single_delegate, false);

  options.SetHintFullyDelegatedToSingleDelegate(true);
  LITERT_ASSERT_OK(LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
      &hint_fully_delegated_to_single_delegate, /*default_value=*/false,
      payload));
  EXPECT_EQ(hint_fully_delegated_to_single_delegate, true);
}

TEST(GpuOptions, SetKernelBatchSizeWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // An unset option resolves to the caller-supplied default.
  int kernel_batch_size = 0;
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorRuntimeOptionsKernelBatchSize(
      &kernel_batch_size, /*default_value=*/-1, payload));
  EXPECT_THAT(kernel_batch_size, Eq(-1));

  options.SetKernelBatchSize(10);

  LITERT_ASSERT_OK(LrtGetGpuAcceleratorRuntimeOptionsKernelBatchSize(
      &kernel_batch_size, /*default_value=*/-1, payload));
  EXPECT_THAT(kernel_batch_size, Eq(10));
}

}  // namespace
}  // namespace litert::ml_drift
