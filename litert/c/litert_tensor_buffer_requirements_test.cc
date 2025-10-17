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

#include "litert/c/litert_tensor_buffer_requirements.h"

#include <array>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace {

constexpr const LiteRtTensorBufferType kSupportedTensorBufferTypes[] = {
    kLiteRtTensorBufferTypeHostMemory,
    kLiteRtTensorBufferTypeAhwb,
    kLiteRtTensorBufferTypeIon,
    kLiteRtTensorBufferTypeFastRpc,
};

constexpr const size_t kNumSupportedTensorBufferTypes =
    sizeof(kSupportedTensorBufferTypes) /
    sizeof(kSupportedTensorBufferTypes[0]);

constexpr const size_t kBufferSize = 1234;

}  // namespace

TEST(TensorBufferRequirements, NoStrides) {
  LiteRtTensorBufferRequirements requirements;
  ASSERT_EQ(LiteRtCreateTensorBufferRequirements(
                kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                kBufferSize,
                /*num_strides=*/0, /*strides=*/nullptr, &requirements),
            kLiteRtStatusOk);

  int num_types;
  ASSERT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                requirements, &num_types),
            kLiteRtStatusOk);
  ASSERT_EQ(num_types, kNumSupportedTensorBufferTypes);

  for (auto i = 0; i < num_types; ++i) {
    LiteRtTensorBufferType type;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                  requirements, i, &type),
              kLiteRtStatusOk);
    ASSERT_EQ(type, kSupportedTensorBufferTypes[i]);
  }

  size_t size;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(requirements, &size),
            kLiteRtStatusOk);
  ASSERT_EQ(size, kBufferSize);

  // Test default alignment
  size_t alignment;
  ASSERT_EQ(
      LiteRtGetTensorBufferRequirementsAlignment(requirements, &alignment),
      kLiteRtStatusOk);
  ASSERT_EQ(alignment, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);

  LiteRtDestroyTensorBufferRequirements(requirements);
}

TEST(TensorBufferRequirements, WithStrides) {
  constexpr std::array<uint32_t, 3> kStrides = {1, 2, 3};

  LiteRtTensorBufferRequirements requirements;
  ASSERT_EQ(LiteRtCreateTensorBufferRequirements(
                kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                kBufferSize, kStrides.size(), kStrides.data(), &requirements),
            kLiteRtStatusOk);

  int num_strides;
  const uint32_t* strides;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsStrides(requirements, &num_strides,
                                                     &strides),
            kLiteRtStatusOk);
  ASSERT_EQ(num_strides, kStrides.size());
  for (auto i = 0; i < kStrides.size(); ++i) {
    ASSERT_EQ(strides[i], kStrides[i]);
  }

  LiteRtDestroyTensorBufferRequirements(requirements);
}

TEST(TensorBufferRequirements, CustomAlignment) {
  constexpr size_t kCustomAlignment = 256;
  constexpr std::array<uint32_t, 2> kStrides = {100, 4};

  LiteRtTensorBufferRequirements requirements;
  ASSERT_EQ(LiteRtCreateTensorBufferRequirementsWithAlignment(
                kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                kBufferSize, kStrides.size(), kStrides.data(), kCustomAlignment,
                &requirements),
            kLiteRtStatusOk);

  // Verify custom alignment was set
  size_t alignment;
  ASSERT_EQ(
      LiteRtGetTensorBufferRequirementsAlignment(requirements, &alignment),
      kLiteRtStatusOk);
  ASSERT_EQ(alignment, kCustomAlignment);

  // Verify other fields are still correct
  size_t size;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(requirements, &size),
            kLiteRtStatusOk);
  ASSERT_EQ(size, kBufferSize);

  int num_types;
  ASSERT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                requirements, &num_types),
            kLiteRtStatusOk);
  ASSERT_EQ(num_types, kNumSupportedTensorBufferTypes);

  LiteRtDestroyTensorBufferRequirements(requirements);
}

TEST(TensorBufferRequirements, InvalidAlignment) {
  constexpr std::array<uint32_t, 2> kStrides = {100, 4};
  LiteRtTensorBufferRequirements requirements;

  // Test non-power-of-2 alignment (should fail)
  ASSERT_EQ(LiteRtCreateTensorBufferRequirementsWithAlignment(
                kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                kBufferSize, kStrides.size(), kStrides.data(),
                100,  // Not a power of 2
                &requirements),
            kLiteRtStatusErrorInvalidArgument);

  // Test zero alignment (should fail)
  ASSERT_EQ(LiteRtCreateTensorBufferRequirementsWithAlignment(
                kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                kBufferSize, kStrides.size(), kStrides.data(),
                0,  // Zero is invalid
                &requirements),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(TensorBufferRequirements, JoinWithDifferentAlignments) {
  constexpr std::array<uint32_t, 2> kStrides = {100, 4};

  // Create first requirements with alignment 64
  LiteRtTensorBufferRequirements req1;
  ASSERT_EQ(LiteRtCreateTensorBufferRequirements(
                kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                1000, kStrides.size(), kStrides.data(), &req1),
            kLiteRtStatusOk);

  // Create second requirements with alignment 256
  LiteRtTensorBufferRequirements req2;
  ASSERT_EQ(LiteRtCreateTensorBufferRequirementsWithAlignment(
                kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                2000, kStrides.size(), kStrides.data(), 256, &req2),
            kLiteRtStatusOk);

  // Join them
  LiteRtTensorBufferRequirements joined;
  ASSERT_EQ(LiteRtJoinTensorBufferRequirements(req1, req2, &joined),
            kLiteRtStatusOk);

  // Verify joined requirements has max alignment (256)
  size_t alignment;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsAlignment(joined, &alignment),
            kLiteRtStatusOk);
  ASSERT_EQ(alignment, 256);

  // Verify joined requirements has max buffer size (2000)
  size_t buffer_size;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(joined, &buffer_size),
            kLiteRtStatusOk);
  ASSERT_EQ(buffer_size, 2000);

  LiteRtDestroyTensorBufferRequirements(req1);
  LiteRtDestroyTensorBufferRequirements(req2);
  LiteRtDestroyTensorBufferRequirements(joined);
}

TEST(TensorBufferRequirements, MultipleAlignmentValues) {
  // Test various power-of-2 alignment values
  constexpr size_t alignments[] = {32, 64, 128, 256, 512, 1024};
  constexpr std::array<uint32_t, 2> kStrides = {100, 4};

  for (size_t alignment : alignments) {
    LiteRtTensorBufferRequirements requirements;
    ASSERT_EQ(LiteRtCreateTensorBufferRequirementsWithAlignment(
                  kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                  kBufferSize, kStrides.size(), kStrides.data(), alignment,
                  &requirements),
              kLiteRtStatusOk);

    // Verify alignment was set correctly
    size_t actual_alignment;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsAlignment(requirements,
                                                         &actual_alignment),
              kLiteRtStatusOk);
    ASSERT_EQ(actual_alignment, alignment);

    LiteRtDestroyTensorBufferRequirements(requirements);
  }
}
