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

using ::testing::Eq;
using ::testing::NotNull;
using ::testing::StrEq;
using ::testing::litert::IsError;

TEST(GpuAcceleratorPayload, CreationWorks) {
  EXPECT_THAT(LiteRtCreateGpuOptions(nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LiteRtOpaqueOptions compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateGpuOptions(&compilation_options));

  const char* identifier = nullptr;
  LITERT_ASSERT_OK(
      LiteRtGetOpaqueOptionsIdentifier(compilation_options, &identifier));
  EXPECT_THAT(identifier, StrEq(LiteRtGetGpuOptionsPayloadIdentifier()));

  LiteRtGpuOptionsPayload payload = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(
      compilation_options, reinterpret_cast<void**>(&payload)));
  EXPECT_THAT(payload, NotNull());

  LiteRtDestroyOpaqueOptions(compilation_options);
}

TEST(GpuAcceleratorPayload, SetAndGetConstantTensorSharing) {
  LiteRtOpaqueOptions compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateGpuOptions(&compilation_options));

  LiteRtGpuOptionsPayload payload = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(
      compilation_options, reinterpret_cast<void**>(&payload)));

  bool constant_tensor_sharing = true;

  // Check the default value.
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(false));

  EXPECT_THAT(LiteRtGetGpuOptionsConstantTensorSharing(nullptr, payload),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LITERT_EXPECT_OK(
      LiteRtSetGpuOptionsConstantTensorSharing(compilation_options, true));
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(true));

  EXPECT_THAT(LiteRtSetGpuOptionsConstantTensorSharing(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LiteRtDestroyOpaqueOptions(compilation_options);
}

TEST(GpuAcceleratorPayload, SetAndGetInfiniteFloatCapping) {
  LiteRtOpaqueOptions compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateGpuOptions(&compilation_options));

  LiteRtGpuOptionsPayload payload = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(
      compilation_options, reinterpret_cast<void**>(&payload)));

  bool infinite_float_capping = true;

  // Check the default value.
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(false));

  EXPECT_THAT(LiteRtGetGpuOptionsInfiniteFloatCapping(nullptr, payload),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LITERT_EXPECT_OK(
      LiteRtSetGpuOptionsInfiniteFloatCapping(compilation_options, true));
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(true));

  EXPECT_THAT(LiteRtSetGpuOptionsInfiniteFloatCapping(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LiteRtDestroyOpaqueOptions(compilation_options);
}

TEST(GpuAcceleratorPayload, SetAndGetBenchmarkMode) {
  LiteRtOpaqueOptions compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateGpuOptions(&compilation_options));

  LiteRtGpuOptionsPayload payload = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(
      compilation_options, reinterpret_cast<void**>(&payload)));

  bool benchmark_mode = true;

  // Check the default value.
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(false));

  LITERT_EXPECT_OK(LiteRtSetGpuOptionsBenchmarkMode(compilation_options, true));
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(true));

  EXPECT_THAT(LiteRtSetGpuOptionsBenchmarkMode(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LiteRtDestroyOpaqueOptions(compilation_options);
}

TEST(GpuAcceleratorPayload, SetAndGetUseBufferStorageType) {
  LiteRtOpaqueOptions compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateGpuOptions(&compilation_options));

  LiteRtGpuOptionsPayload payload = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(
      compilation_options, reinterpret_cast<void**>(&payload)));

  LiteRtDelegateBufferStorageType use_buffer_storage_type =
      kLiteRtDelegateBufferStorageTypeDefault;

  // Check the default value.
  LITERT_EXPECT_OK(LiteRtGetGpuAcceleratorCompilationOptionsBufferStorageType(
      &use_buffer_storage_type, payload));
  EXPECT_EQ(use_buffer_storage_type, kLiteRtDelegateBufferStorageTypeDefault);

  LITERT_EXPECT_OK(
      LiteRtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
          compilation_options, kLiteRtDelegateBufferStorageTypeBuffer));
  LITERT_EXPECT_OK(LiteRtGetGpuAcceleratorCompilationOptionsBufferStorageType(
      &use_buffer_storage_type, payload));
  EXPECT_EQ(use_buffer_storage_type, kLiteRtDelegateBufferStorageTypeBuffer);

  EXPECT_THAT(LiteRtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
                  nullptr, kLiteRtDelegateBufferStorageTypeBuffer),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LiteRtDestroyOpaqueOptions(compilation_options);
}

}  // namespace
