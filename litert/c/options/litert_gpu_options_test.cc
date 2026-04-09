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

#include "litert/c/options/litert_gpu_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace {

void SerializeAndParse(LrtGpuOptions* payload,
                       LrtGpuOptions** payload_from_toml) {
  const char* identifier = nullptr;
  void* opaque_payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;

  LITERT_ASSERT_OK(LrtGetOpaqueGpuOptionsData(
      payload, &identifier, &opaque_payload, &payload_deleter));

  LITERT_ASSERT_OK(LrtCreateGpuOptionsFromToml(
      static_cast<const char*>(opaque_payload), payload_from_toml));

  if (payload_deleter && opaque_payload) {
    payload_deleter(opaque_payload);
  }
}

using ::testing::Eq;
using ::testing::NotNull;
using ::testing::StrEq;
using ::testing::litert::IsError;

TEST(GpuAcceleratorPayload, CreationWorks) {
  EXPECT_THAT(LrtCreateGpuOptions(nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));
  EXPECT_THAT(payload, NotNull());

  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetPrecision) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  LiteRtDelegatePrecision precision;
  // Check the default value.
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionDefault));

  LITERT_EXPECT_OK(LrtSetGpuAcceleratorCompilationOptionsPrecision(
      payload, kLiteRtDelegatePrecisionFp16));
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_EQ(precision, kLiteRtDelegatePrecisionFp16);

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  LiteRtDelegatePrecision precision_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsPrecision(
      &precision_from_toml, payload_from_toml));
  EXPECT_THAT(precision_from_toml, Eq(kLiteRtDelegatePrecisionFp16));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetBackend) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  LiteRtGpuBackend backend;
  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuOptionsGpuBackend(&backend, payload));
  EXPECT_THAT(backend, Eq(kLiteRtGpuBackendAutomatic));

  LITERT_EXPECT_OK(
      LrtSetGpuOptionsGpuBackend(payload, kLiteRtGpuBackendOpenCl));
  LITERT_EXPECT_OK(LrtGetGpuOptionsGpuBackend(&backend, payload));
  EXPECT_EQ(backend, kLiteRtGpuBackendOpenCl);

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  LiteRtGpuBackend backend_from_toml;
  LITERT_EXPECT_OK(
      LrtGetGpuOptionsGpuBackend(&backend_from_toml, payload_from_toml));
  EXPECT_THAT(backend_from_toml, Eq(kLiteRtGpuBackendOpenCl));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetConstantTensorSharing) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  bool constant_tensor_sharing = true;

  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuOptionsConstantTensorsSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(false));

  EXPECT_THAT(LrtGetGpuOptionsConstantTensorsSharing(nullptr, payload),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LITERT_EXPECT_OK(LrtSetGpuOptionsConstantTensorsSharing(payload, true));
  LITERT_EXPECT_OK(LrtGetGpuOptionsConstantTensorsSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(true));

  EXPECT_THAT(LrtSetGpuOptionsConstantTensorsSharing(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  bool constant_tensor_sharing_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuOptionsConstantTensorsSharing(
      &constant_tensor_sharing_from_toml, payload_from_toml));
  EXPECT_THAT(constant_tensor_sharing_from_toml, Eq(true));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetInfiniteFloatCapping) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  bool infinite_float_capping = true;

  // Check the default value.
  LITERT_EXPECT_OK(
      LrtGetGpuOptionsInfiniteFloatCapping(&infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(false));

  EXPECT_THAT(LrtGetGpuOptionsInfiniteFloatCapping(nullptr, payload),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LITERT_EXPECT_OK(LrtSetGpuOptionsInfiniteFloatCapping(payload, true));
  LITERT_EXPECT_OK(
      LrtGetGpuOptionsInfiniteFloatCapping(&infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(true));

  EXPECT_THAT(LrtSetGpuOptionsInfiniteFloatCapping(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  bool infinite_float_capping_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping_from_toml, payload_from_toml));
  EXPECT_THAT(infinite_float_capping_from_toml, Eq(true));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetBenchmarkMode) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  bool benchmark_mode = true;

  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(false));

  LITERT_EXPECT_OK(LrtSetGpuOptionsBenchmarkMode(payload, true));
  LITERT_EXPECT_OK(LrtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(true));

  EXPECT_THAT(LrtSetGpuOptionsBenchmarkMode(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  bool benchmark_mode_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuOptionsBenchmarkMode(&benchmark_mode_from_toml,
                                                 payload_from_toml));
  EXPECT_THAT(benchmark_mode_from_toml, Eq(true));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetUseBufferStorageType) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  LiteRtDelegateBufferStorageType use_buffer_storage_type =
      kLiteRtDelegateBufferStorageTypeDefault;

  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsBufferStorageType(
      &use_buffer_storage_type, payload));
  EXPECT_EQ(use_buffer_storage_type, kLiteRtDelegateBufferStorageTypeDefault);

  LITERT_EXPECT_OK(LrtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
      payload, kLiteRtDelegateBufferStorageTypeBuffer));
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsBufferStorageType(
      &use_buffer_storage_type, payload));
  EXPECT_EQ(use_buffer_storage_type, kLiteRtDelegateBufferStorageTypeBuffer);

  EXPECT_THAT(LrtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
                  nullptr, kLiteRtDelegateBufferStorageTypeBuffer),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  LiteRtDelegateBufferStorageType use_buffer_storage_type_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsBufferStorageType(
      &use_buffer_storage_type_from_toml, payload_from_toml));
  EXPECT_EQ(use_buffer_storage_type_from_toml,
            kLiteRtDelegateBufferStorageTypeBuffer);

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetPreferTextureWeights) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  bool prefer_texture_weights = true;

  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
      &prefer_texture_weights, payload));
  EXPECT_EQ(prefer_texture_weights, false);

  LITERT_EXPECT_OK(LrtSetGpuAcceleratorCompilationOptionsPreferTextureWeights(
      payload, true));
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
      &prefer_texture_weights, payload));
  EXPECT_EQ(prefer_texture_weights, true);

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  bool prefer_texture_weights_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
      &prefer_texture_weights_from_toml, payload_from_toml));
  EXPECT_THAT(prefer_texture_weights_from_toml, Eq(true));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetSerializationDir) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  const char* serialization_dir = nullptr;

  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
      &serialization_dir, payload));
  EXPECT_EQ(serialization_dir, nullptr);

  LITERT_EXPECT_OK(LrtSetGpuAcceleratorCompilationOptionsSerializationDir(
      payload, "/data/local/tmp"));
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
      &serialization_dir, payload));
  EXPECT_THAT(serialization_dir, StrEq("/data/local/tmp"));

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  const char* serialization_dir_from_toml = nullptr;
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
      &serialization_dir_from_toml, payload_from_toml));
  EXPECT_THAT(serialization_dir_from_toml, StrEq("/data/local/tmp"));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetModelToken) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  const char* model_cache_key = nullptr;

  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
      &model_cache_key, payload));
  EXPECT_EQ(model_cache_key, nullptr);

  LITERT_EXPECT_OK(LrtSetGpuAcceleratorCompilationOptionsModelCacheKey(
      payload, "model_cache"));
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
      &model_cache_key, payload));
  EXPECT_THAT(model_cache_key, StrEq("model_cache"));

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  const char* model_cache_key_from_toml = nullptr;
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
      &model_cache_key_from_toml, payload_from_toml));
  EXPECT_THAT(model_cache_key_from_toml, StrEq("model_cache"));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetProgramCacheFd) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  int program_cache_fd = -1;

  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
      &program_cache_fd, payload));
  EXPECT_EQ(program_cache_fd, -1);

  LITERT_EXPECT_OK(
      LrtSetGpuAcceleratorCompilationOptionsProgramCacheFd(payload, 123));
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
      &program_cache_fd, payload));
  EXPECT_EQ(program_cache_fd, 123);

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  int program_cache_fd_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
      &program_cache_fd_from_toml, payload_from_toml));
  EXPECT_THAT(program_cache_fd_from_toml, Eq(123));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetSerializeProgramCache) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  bool serialize_program_cache = false;

  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
      &serialize_program_cache, payload));
  EXPECT_EQ(serialize_program_cache, true);

  LITERT_EXPECT_OK(LrtSetGpuAcceleratorCompilationOptionsSerializeProgramCache(
      payload, false));
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
      &serialize_program_cache, payload));
  EXPECT_EQ(serialize_program_cache, false);

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  bool serialize_program_cache_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
      &serialize_program_cache_from_toml, payload_from_toml));
  EXPECT_THAT(serialize_program_cache_from_toml, Eq(false));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetSerializeExternalTensors) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  bool serialize_external_tensors = true;

  // Check the default value.
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
          &serialize_external_tensors, payload));
  EXPECT_EQ(serialize_external_tensors, false);

  LITERT_EXPECT_OK(
      LrtSetGpuAcceleratorCompilationOptionsSerializeExternalTensors(payload,
                                                                     true));
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
          &serialize_external_tensors, payload));
  EXPECT_EQ(serialize_external_tensors, true);

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  bool serialize_external_tensors_from_toml;
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
          &serialize_external_tensors_from_toml, payload_from_toml));
  EXPECT_THAT(serialize_external_tensors_from_toml, Eq(true));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetUseMetalArgumentBuffers) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  bool use_metal_argument_buffers = true;

  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuOptionsUseMetalArgumentBuffers(
      payload, &use_metal_argument_buffers));
  EXPECT_THAT(use_metal_argument_buffers, Eq(false));

  LITERT_EXPECT_OK(LrtSetGpuOptionsUseMetalArgumentBuffers(payload, true));
  LITERT_EXPECT_OK(LrtGetGpuOptionsUseMetalArgumentBuffers(
      payload, &use_metal_argument_buffers));
  EXPECT_THAT(use_metal_argument_buffers, Eq(true));

  EXPECT_THAT(LrtSetGpuOptionsUseMetalArgumentBuffers(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  bool use_metal_argument_buffers_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuOptionsUseMetalArgumentBuffers(
      payload_from_toml, &use_metal_argument_buffers_from_toml));
  EXPECT_THAT(use_metal_argument_buffers_from_toml, Eq(true));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetHintFullyDelegatedToSingleDelegate) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  bool hint_fully_delegated_to_single_delegate = true;

  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
      &hint_fully_delegated_to_single_delegate, payload));
  EXPECT_EQ(hint_fully_delegated_to_single_delegate, false);

  LITERT_EXPECT_OK(
      LrtSetGpuOptionsHintFullyDelegatedToSingleDelegate(payload, true));
  LITERT_EXPECT_OK(LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
      &hint_fully_delegated_to_single_delegate, payload));
  EXPECT_EQ(hint_fully_delegated_to_single_delegate, true);

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  bool hint_fully_delegated_to_single_delegate_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
      &hint_fully_delegated_to_single_delegate_from_toml, payload_from_toml));
  EXPECT_THAT(hint_fully_delegated_to_single_delegate_from_toml, Eq(true));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, TomlSerializationWithPatterns) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  LITERT_EXPECT_OK(LrtSetGpuOptionsBenchmarkMode(payload, true));
  LITERT_EXPECT_OK(
      LrtAddGpuOptionsBufferStorageTensorPattern(payload, "pattern1"));
  LITERT_EXPECT_OK(
      LrtAddGpuOptionsBufferStorageTensorPattern(payload, "pattern2"));

  const char* identifier = nullptr;
  void* opaque_payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;

  LITERT_EXPECT_OK(LrtGetOpaqueGpuOptionsData(
      payload, &identifier, &opaque_payload, &payload_deleter));
  EXPECT_THAT(identifier, StrEq("gpu_options"));
  EXPECT_THAT(opaque_payload, NotNull());
  EXPECT_THAT(static_cast<const char*>(opaque_payload),
              StrEq("benchmark_mode = true\n"
                    "buffer_storage_tensor_patterns = [\"pattern1\", "
                    "\"pattern2\"]\n"));

  LrtGpuOptions* payload_from_toml = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptionsFromToml(
      static_cast<const char*>(opaque_payload), &payload_from_toml));

  bool benchmark_mode = false;
  LITERT_EXPECT_OK(
      LrtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload_from_toml));
  EXPECT_THAT(benchmark_mode, Eq(true));

  int num_patterns = 0;
  LITERT_EXPECT_OK(
      LrtGetNumGpuAcceleratorCompilationOptionsBufferStorageTensorPatterns(
          &num_patterns, payload_from_toml));
  EXPECT_THAT(num_patterns, Eq(2));

  const char* pattern = nullptr;
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorCompilationOptionsBufferStorageTensorPattern(
          &pattern, 0, payload_from_toml));
  EXPECT_THAT(pattern, StrEq("pattern1"));

  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorCompilationOptionsBufferStorageTensorPattern(
          &pattern, 1, payload_from_toml));
  EXPECT_THAT(pattern, StrEq("pattern2"));

  payload_deleter(opaque_payload);
  LrtDestroyGpuOptions(payload);
  LrtDestroyGpuOptions(payload_from_toml);
}

TEST(GpuAcceleratorPayload, SetAndGetSyncExecutionModeWaitType) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  LiteRtGpuWaitType wait_type;
  // Check the default value.
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorRuntimeOptionsWaitType(&wait_type, payload));
  EXPECT_EQ(wait_type, kLiteRtGpuWaitTypeDefault);

  LITERT_EXPECT_OK(LrtSetGpuAcceleratorRuntimeOptionsWaitType(
      payload, kLiteRtGpuWaitTypePassive));
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorRuntimeOptionsWaitType(&wait_type, payload));
  EXPECT_EQ(wait_type, kLiteRtGpuWaitTypePassive);

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  LiteRtGpuWaitType wait_type_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorRuntimeOptionsWaitType(
      &wait_type_from_toml, payload_from_toml));
  EXPECT_THAT(wait_type_from_toml, Eq(kLiteRtGpuWaitTypePassive));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetGpuFlushPeriod) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  int gpu_flush_period;
  // Check the default value.
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorRuntimeOptionsGpuFlushPeriod(
      &gpu_flush_period, payload));
  EXPECT_THAT(gpu_flush_period, Eq(-1));

  LITERT_EXPECT_OK(
      LrtSetGpuAcceleratorRuntimeOptionsGpuFlushPeriod(payload, 10));
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorRuntimeOptionsGpuFlushPeriod(
      &gpu_flush_period, payload));
  EXPECT_EQ(gpu_flush_period, 10);

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  int gpu_flush_period_from_toml;
  LITERT_EXPECT_OK(LrtGetGpuAcceleratorRuntimeOptionsGpuFlushPeriod(
      &gpu_flush_period_from_toml, payload_from_toml));
  EXPECT_THAT(gpu_flush_period_from_toml, Eq(10));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

TEST(GpuAcceleratorPayload, SetAndGetAllowSrcQuantizedFcConvOps) {
  LrtGpuOptions* payload = nullptr;
  LITERT_ASSERT_OK(LrtCreateGpuOptions(&payload));

  bool enabled;
  // Check the default value.
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &enabled, payload));
  EXPECT_EQ(enabled, false);

  LITERT_EXPECT_OK(
      LrtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(payload,
                                                                       true));
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &enabled, payload));
  EXPECT_EQ(enabled, true);

  LrtGpuOptions* payload_from_toml = nullptr;
  SerializeAndParse(payload, &payload_from_toml);

  bool enabled_from_toml;
  LITERT_EXPECT_OK(
      LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &enabled_from_toml, payload_from_toml));
  EXPECT_THAT(enabled_from_toml, Eq(true));

  LrtDestroyGpuOptions(payload_from_toml);
  LrtDestroyGpuOptions(payload);
}

}  // namespace
