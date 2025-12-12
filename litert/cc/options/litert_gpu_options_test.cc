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

namespace litert::ml_drift {
namespace {

TEST(GpuOptions, EnableConstantTensorSharingWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  bool constant_tensor_sharing = true;
  LITERT_ASSERT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(false));

  options.EnableConstantTensorSharing(true);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(true));

  options.EnableConstantTensorSharing(false);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(false));
}

TEST(GpuOptions, EnableInfiniteFloatCappingWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  bool infinite_float_capping = true;
  LITERT_ASSERT_OK(LiteRtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(false));

  options.EnableInfiniteFloatCapping(true);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(true));

  options.EnableInfiniteFloatCapping(false);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(false));
}

TEST(GpuOptions, EnableBenchmarkModeWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  bool benchmark_mode = true;
  LITERT_ASSERT_OK(LiteRtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(false));

  options.EnableBenchmarkMode(true);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(true));

  options.EnableBenchmarkMode(false);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(false));
}

TEST(GpuAcceleratorCompilationOptions,
     EnableAllowSrcQuantizedFcConvOpsCheckTrue) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  bool allow_src_quantized_fc_conv_ops = false;
  LITERT_EXPECT_OK(options.EnableAllowSrcQuantizedFcConvOps(true));

  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &allow_src_quantized_fc_conv_ops, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(true));
}

TEST(GpuAcceleratorCompilationOptions,
     EnableAllowSrcQuantizedFcConvOpsCheckDefaultValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  bool allow_src_quantized_fc_conv_ops = true;

  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &allow_src_quantized_fc_conv_ops, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(false));
}

TEST(GpuAcceleratorCompilationOptions,
     EnableAllowSrcQuantizedFcConvOpsCheckFalseValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // The default value is false, set it to true before resetting to false.
  LITERT_EXPECT_OK(options.EnableAllowSrcQuantizedFcConvOps(true));
  LITERT_EXPECT_OK(options.EnableAllowSrcQuantizedFcConvOps(false));
  bool allow_src_quantized_fc_conv_ops = true;

  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &allow_src_quantized_fc_conv_ops, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(false));
}

TEST(GpuAcceleratorCompilationOptions, CheckDelegatePrecisionDefaultPrecision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionFp16;
  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionDefault));
}

TEST(GpuAcceleratorCompilationOptions, SetDelegatePrecisionFp16Precision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;
  options.SetPrecision(GpuOptions::Precision::kFp16);

  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionFp16));
}

TEST(GpuAcceleratorCompilationOptions, SetDelegatePrecisionFp32Precision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;

  options.SetPrecision(GpuOptions::Precision::kFp32);

  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionFp32));
}

TEST(GpuAcceleratorCompilationOptions, SetSerializationDir) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());
  // Check the default value.
  const char* serialization_dir = nullptr;
  LITERT_ASSERT_OK(LiteRtGetGpuAcceleratorCompilationOptionsSerializationDir(
      &serialization_dir, payload));
  EXPECT_EQ(serialization_dir, nullptr);

  options.SetSerializationDir("/data/local/tmp");
  LITERT_ASSERT_OK(LiteRtGetGpuAcceleratorCompilationOptionsSerializationDir(
      &serialization_dir, payload));
  EXPECT_EQ(serialization_dir, "/data/local/tmp");
}

TEST(GpuAcceleratorCompilationOptions, SetModelCacheKey) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());
  // Check the default value.
  const char* model_cache_key = nullptr;
  LITERT_ASSERT_OK(LiteRtGetGpuAcceleratorCompilationOptionsModelCacheKey(
      &model_cache_key, payload));
  EXPECT_EQ(model_cache_key, nullptr);

  options.SetModelCacheKey("model_cache");
  LITERT_ASSERT_OK(LiteRtGetGpuAcceleratorCompilationOptionsModelCacheKey(
      &model_cache_key, payload));
  EXPECT_EQ(model_cache_key, "model_cache");
}

TEST(GpuAcceleratorCompilationOptions, SetSerializeProgramCache) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());
  // Check the default value.
  bool serialize_program_cache = false;
  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
          &serialize_program_cache, payload));
  EXPECT_EQ(serialize_program_cache, true);

  options.SetSerializeProgramCache(false);
  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
          &serialize_program_cache, payload));
  EXPECT_EQ(serialize_program_cache, false);
}

TEST(GpuAcceleratorCompilationOptions, SetSerializeExternalTensors) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());
  // Check the default value.
  bool serialize_external_tensors = false;
  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
          &serialize_external_tensors, payload));
  EXPECT_EQ(serialize_external_tensors, false);

  options.SetSerializeExternalTensors(true);
  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
          &serialize_external_tensors, payload));
  EXPECT_EQ(serialize_external_tensors, true);
}

TEST(GpuAcceleratorCompilationOptions, SetPreferTextureWeights) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());
  // Check the default value.
  bool prefer_texture_weights = false;
  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
          &prefer_texture_weights, payload));
  EXPECT_EQ(prefer_texture_weights, false);

  options.SetPreferTextureWeights(true);
  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
          &prefer_texture_weights, payload));
  EXPECT_EQ(prefer_texture_weights, true);
}

#ifdef __APPLE__
TEST(GpuOptions, SetUseMetalArgumentBuffersWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  bool use_metal_argument_buffers = true;
  LITERT_ASSERT_OK(LiteRtGetGpuOptionsUseMetalArgumentBuffers(
      payload, &use_metal_argument_buffers));
  EXPECT_THAT(use_metal_argument_buffers, Eq(false));

  options.SetUseMetalArgumentBuffers(true);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsUseMetalArgumentBuffers(
      payload, &use_metal_argument_buffers));
  EXPECT_THAT(use_metal_argument_buffers, Eq(true));

  options.SetUseMetalArgumentBuffers(false);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsUseMetalArgumentBuffers(
      payload, &use_metal_argument_buffers));
  EXPECT_THAT(use_metal_argument_buffers, Eq(false));
}
#endif  // __APPLE__

TEST(GpuAcceleratorCompilationOptions, SetHintFullyDelegatedToSingleDelegate) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());
  // Check the default value.
  bool hint_fully_delegated_to_single_delegate = false;
  LITERT_ASSERT_OK(LiteRtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
      &hint_fully_delegated_to_single_delegate, payload));
  EXPECT_EQ(hint_fully_delegated_to_single_delegate, false);

  options.SetHintFullyDelegatedToSingleDelegate(true);
  LITERT_ASSERT_OK(LiteRtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
      &hint_fully_delegated_to_single_delegate, payload));
  EXPECT_EQ(hint_fully_delegated_to_single_delegate, true);
}

}  // namespace
}  // namespace litert::ml_drift
