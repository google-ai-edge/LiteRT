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

namespace litert::test {

class GpuOptionsTestPeer {
 public:
  static LiteRtStatus SetUseIrModel(GpuOptions& options, bool use_ir_model) {
    return options.SetUseIrModel(use_ir_model);
  }
};

}  // namespace litert::test

namespace litert::ml_drift {
namespace {

TEST(GpuOptions, EnableConstantTensorSharingWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // An unset option returns kLiteRtStatusErrorNotFound.
  bool constant_tensor_sharing = true;
  EXPECT_THAT(
      LrtGetGpuOptionsConstantTensorsSharing(&constant_tensor_sharing, payload),
      ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.EnableConstantTensorSharing(true);

  LITERT_ASSERT_OK(LrtGetGpuOptionsConstantTensorsSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(true));

  options.EnableConstantTensorSharing(false);

  LITERT_ASSERT_OK(LrtGetGpuOptionsConstantTensorsSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(false));
}

TEST(GpuOptions, EnableInfiniteFloatCappingWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // An unset option returns kLiteRtStatusErrorNotFound.
  bool infinite_float_capping = true;
  EXPECT_THAT(
      LrtGetGpuOptionsInfiniteFloatCapping(&infinite_float_capping, payload),
      ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.EnableInfiniteFloatCapping(true);

  LITERT_ASSERT_OK(
      LrtGetGpuOptionsInfiniteFloatCapping(&infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(true));

  options.EnableInfiniteFloatCapping(false);

  LITERT_ASSERT_OK(
      LrtGetGpuOptionsInfiniteFloatCapping(&infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(false));
}

TEST(GpuOptions, EnableBenchmarkModeWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // An unset option returns kLiteRtStatusErrorNotFound.
  bool benchmark_mode = true;
  EXPECT_THAT(LrtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.EnableBenchmarkMode(true);

  LITERT_ASSERT_OK(LrtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(true));

  options.EnableBenchmarkMode(false);

  LITERT_ASSERT_OK(LrtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
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
          &allow_src_quantized_fc_conv_ops, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(true));
}

TEST(GpuAcceleratorCompilationOptions,
     EnableAllowSrcQuantizedFcConvOpsCheckDefaultValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // Check the unset option returns kLiteRtStatusErrorNotFound.
  bool allow_src_quantized_fc_conv_ops = true;

  EXPECT_THAT(LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
                  &allow_src_quantized_fc_conv_ops, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));
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
          &allow_src_quantized_fc_conv_ops, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(false));
}

TEST(GpuAcceleratorCompilationOptions, CheckDelegatePrecisionDefaultPrecision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // An unset option returns kLiteRtStatusErrorNotFound.
  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionFp16;
  EXPECT_THAT(
      LrtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload),
      ::testing::litert::IsError(kLiteRtStatusErrorNotFound));
}

TEST(GpuAcceleratorCompilationOptions, SetDelegatePrecisionFp16Precision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;
  options.SetPrecision(GpuOptions::Precision::kFp16);

  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionFp16));
}

TEST(GpuAcceleratorCompilationOptions, SetDelegatePrecisionFp32Precision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;

  options.SetPrecision(GpuOptions::Precision::kFp32);

  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionFp32));
}

TEST(GpuAcceleratorCompilationOptions, SetSerializationDir) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // An unset option returns kLiteRtStatusErrorNotFound.
  const char* serialization_dir = nullptr;
  EXPECT_THAT(LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
                  &serialization_dir, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.SetSerializationDir("/data/local/tmp");
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
      &serialization_dir, payload));
  EXPECT_THAT(serialization_dir, StrEq("/data/local/tmp"));
}

TEST(GpuAcceleratorCompilationOptions, SetModelCacheKey) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // An unset option returns kLiteRtStatusErrorNotFound.
  const char* model_cache_key = nullptr;
  EXPECT_THAT(LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
                  &model_cache_key, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.SetModelCacheKey("model_cache");
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
      &model_cache_key, payload));
  EXPECT_THAT(model_cache_key, StrEq("model_cache"));
}

TEST(GpuAcceleratorCompilationOptions, SetProgramCacheFd) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // An unset option returns kLiteRtStatusErrorNotFound.
  int program_cache_fd = -1;
  EXPECT_THAT(LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
                  &program_cache_fd, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.SetProgramCacheFd(123);
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
      &program_cache_fd, payload));
  EXPECT_EQ(program_cache_fd, 123);
}

TEST(GpuAcceleratorCompilationOptions, SetSerializeProgramCache) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // An unset option returns kLiteRtStatusErrorNotFound.
  bool serialize_program_cache = false;
  EXPECT_THAT(LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
                  &serialize_program_cache, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.SetSerializeProgramCache(false);
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
      &serialize_program_cache, payload));
  EXPECT_EQ(serialize_program_cache, false);
}

TEST(GpuAcceleratorCompilationOptions, SetSerializeExternalTensors) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // Check the unset option returns kLiteRtStatusErrorNotFound.
  bool serialize_external_tensors = false;
  EXPECT_THAT(LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
                  &serialize_external_tensors, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.SetSerializeExternalTensors(true);
  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
          &serialize_external_tensors, payload));
  EXPECT_EQ(serialize_external_tensors, true);
}

TEST(GpuAcceleratorCompilationOptions, SetPreferTextureWeights) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // Check the unset option returns kLiteRtStatusErrorNotFound.
  bool prefer_texture_weights = false;
  EXPECT_THAT(LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
                  &prefer_texture_weights, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.SetPreferTextureWeights(true);
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
      &prefer_texture_weights, payload));
  EXPECT_EQ(prefer_texture_weights, true);
}

#ifdef __APPLE__
TEST(GpuOptions, SetUseMetalArgumentBuffersWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // Check the unset option returns kLiteRtStatusErrorNotFound.
  bool use_metal_argument_buffers = true;
  EXPECT_THAT(LrtGetGpuOptionsUseMetalArgumentBuffers(
                  &use_metal_argument_buffers, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.SetUseMetalArgumentBuffers(true);

  LITERT_ASSERT_OK(LrtGetGpuOptionsUseMetalArgumentBuffers(
      &use_metal_argument_buffers, payload));
  EXPECT_THAT(use_metal_argument_buffers, Eq(true));

  options.SetUseMetalArgumentBuffers(false);

  LITERT_ASSERT_OK(LrtGetGpuOptionsUseMetalArgumentBuffers(
      &use_metal_argument_buffers, payload));
  EXPECT_THAT(use_metal_argument_buffers, Eq(false));
}
#endif  // __APPLE__

TEST(GpuAcceleratorCompilationOptions, SetHintFullyDelegatedToSingleDelegate) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();
  // Check the unset option returns kLiteRtStatusErrorNotFound.
  bool hint_fully_delegated_to_single_delegate = false;
  EXPECT_THAT(LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
                  &hint_fully_delegated_to_single_delegate, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.SetHintFullyDelegatedToSingleDelegate(true);
  LITERT_ASSERT_OK(LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
      &hint_fully_delegated_to_single_delegate, payload));
  EXPECT_EQ(hint_fully_delegated_to_single_delegate, true);
}

TEST(GpuOptions, SetKernelBatchSizeWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  // An unset option returns kLiteRtStatusErrorNotFound.
  int kernel_batch_size = 0;
  EXPECT_THAT(LrtGetGpuAcceleratorRuntimeOptionsKernelBatchSize(
                  &kernel_batch_size, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));

  options.SetKernelBatchSize(10);

  LITERT_ASSERT_OK(LrtGetGpuAcceleratorRuntimeOptionsKernelBatchSize(
      &kernel_batch_size, payload));
  EXPECT_THAT(kernel_batch_size, Eq(10));
}

TEST(GpuOptions, SetUseIrModelWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LrtGpuOptions* payload = options.Get();

  bool use_ir_model = true;
#if defined(LITERT_GPU_USE_IR_MODEL)
  LITERT_ASSERT_OK(LrtGetGpuOptionsUseIrModel(&use_ir_model, payload));
  EXPECT_THAT(use_ir_model, Eq(static_cast<bool>(LITERT_GPU_USE_IR_MODEL)));
#else
  // Check that it returns kLiteRtStatusErrorNotFound when not set.
  EXPECT_THAT(LrtGetGpuOptionsUseIrModel(&use_ir_model, payload),
              ::testing::litert::IsError(kLiteRtStatusErrorNotFound));
#endif

  LITERT_EXPECT_OK(test::GpuOptionsTestPeer::SetUseIrModel(options, true));

  LITERT_ASSERT_OK(LrtGetGpuOptionsUseIrModel(&use_ir_model, payload));
  EXPECT_THAT(use_ir_model, Eq(true));

  LITERT_EXPECT_OK(test::GpuOptionsTestPeer::SetUseIrModel(options, false));

  LITERT_ASSERT_OK(LrtGetGpuOptionsUseIrModel(&use_ir_model, payload));
  EXPECT_THAT(use_ir_model, Eq(false));
}

}  // namespace
}  // namespace litert::ml_drift
