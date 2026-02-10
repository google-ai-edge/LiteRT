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
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"

namespace {

constexpr const litert::TensorBufferType kSupportedTensorBufferTypes[] = {
    litert::TensorBufferType::kHostMemory,
    litert::TensorBufferType::kAhwb,
    litert::TensorBufferType::kIon,
    litert::TensorBufferType::kFastRpc,
};

constexpr const size_t kNumSupportedTensorBufferTypes =
    sizeof(kSupportedTensorBufferTypes) /
    sizeof(kSupportedTensorBufferTypes[0]);

constexpr const size_t kBufferSize = 1234;

}  // namespace

TEST(TensorBufferRequirements, Owned) {
  auto requirements = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes,
                     kNumSupportedTensorBufferTypes),
      kBufferSize);
  ASSERT_TRUE(requirements);

  auto supported_types = requirements->SupportedTypes();
  ASSERT_TRUE(supported_types);
  ASSERT_EQ(supported_types->size(), kNumSupportedTensorBufferTypes);
  for (auto i = 0; i < supported_types->size(); ++i) {
    ASSERT_EQ((*supported_types)[i], kSupportedTensorBufferTypes[i]);
  }

  auto size = requirements->BufferSize();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, kBufferSize);
}

TEST(TensorBufferRequirements, NotOwned) {
  LiteRtTensorBufferRequirements litert_requirements;
  std::vector<LiteRtTensorBufferType> litert_buffer_types;
  litert_buffer_types.reserve(kNumSupportedTensorBufferTypes);
  for (const auto& buffer_type : kSupportedTensorBufferTypes) {
    litert_buffer_types.push_back(
        static_cast<LiteRtTensorBufferType>(buffer_type));
  }
  ASSERT_EQ(
      LiteRtCreateTensorBufferRequirements(
          litert_buffer_types.size(), litert_buffer_types.data(), kBufferSize,
          /*num_strides=*/0, /*strides=*/nullptr, &litert_requirements),
      kLiteRtStatusOk);

  auto requirements = litert::TensorBufferRequirements::WrapCObject(
      litert_requirements, litert::OwnHandle::kNo);

  auto supported_types = requirements.SupportedTypes();
  ASSERT_TRUE(supported_types);
  ASSERT_EQ(supported_types->size(), kNumSupportedTensorBufferTypes);
  for (auto i = 0; i < supported_types->size(); ++i) {
    ASSERT_EQ((*supported_types)[i], kSupportedTensorBufferTypes[i]);
  }

  auto size = requirements.BufferSize();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, kBufferSize);

  ASSERT_EQ(requirements.Get(), litert_requirements);

  LiteRtDestroyTensorBufferRequirements(litert_requirements);
}

TEST(TensorBufferRequirements, WithStrides) {
  constexpr std::array<uint32_t, 3> kStrides = {1, 2, 3};

  auto requirements = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes,
                     kNumSupportedTensorBufferTypes),
      kBufferSize, absl::MakeSpan(kStrides.data(), kStrides.size()));
  ASSERT_TRUE(requirements);

  auto strides = requirements->Strides();
  ASSERT_TRUE(strides);
  ASSERT_EQ(strides->size(), kStrides.size());
  for (auto i = 0; i < kStrides.size(); ++i) {
    ASSERT_EQ((*strides)[i], kStrides[i]);
  }
}

TEST(TensorBufferRequirements, JoinSuccess) {
  constexpr const std::array kSupportedTensorBufferTypes1 = {
      litert::TensorBufferType::kHostMemory,
      litert::TensorBufferType::kAhwb,
      litert::TensorBufferType::kIon,
      litert::TensorBufferType::kFastRpc,
  };
  constexpr const size_t kBufferSize1 = 1234;

  constexpr const std::array kSupportedTensorBufferTypes2 = {
      litert::TensorBufferType::kAhwb,
      litert::TensorBufferType::kFastRpc,
  };
  constexpr const size_t kBufferSize2 = 1238;

  auto src_requirements_1 = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes1.data(),
                     kSupportedTensorBufferTypes1.size()),
      kBufferSize1);
  ASSERT_TRUE(src_requirements_1);

  auto src_requirements_2 = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes2.data(),
                     kSupportedTensorBufferTypes2.size()),
      kBufferSize2);
  ASSERT_TRUE(src_requirements_2);

  auto joint_requirements =
      litert::Join(*src_requirements_1, *src_requirements_2);
  ASSERT_TRUE(joint_requirements);

  auto supported_types = joint_requirements->SupportedTypes();
  ASSERT_TRUE(supported_types);
  ASSERT_EQ(supported_types->size(), 2);
  ASSERT_EQ((*supported_types)[0], litert::TensorBufferType::kAhwb);
  ASSERT_EQ((*supported_types)[1], litert::TensorBufferType::kFastRpc);

  auto size = joint_requirements->BufferSize();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, kBufferSize2);
}

TEST(TensorBufferRequirements, JoinFailure) {
  constexpr const std::array kSupportedTensorBufferTypes1 = {
      litert::TensorBufferType::kHostMemory,
  };
  constexpr const size_t kBufferSize1 = 1234;

  constexpr const std::array kSupportedTensorBufferTypes2 = {
      litert::TensorBufferType::kAhwb,
  };
  constexpr const size_t kBufferSize2 = 1238;

  auto src_requirements_1 = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes1.data(),
                     kSupportedTensorBufferTypes1.size()),
      kBufferSize1);
  ASSERT_TRUE(src_requirements_1);

  auto src_requirements_2 = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes2.data(),
                     kSupportedTensorBufferTypes2.size()),
      kBufferSize2);
  ASSERT_TRUE(src_requirements_2);

  auto joint_requirements =
      litert::Join(*src_requirements_1, *src_requirements_2);
  ASSERT_FALSE(joint_requirements);
}

TEST(TensorBufferRequirements, CreateWithAlignment) {
  constexpr size_t kCustomAlignment = 256;
  constexpr std::array<uint32_t, 2> kStrides = {100, 4};

  auto requirements = litert::TensorBufferRequirements::CreateWithAlignment(
      absl::MakeSpan(kSupportedTensorBufferTypes,
                     kNumSupportedTensorBufferTypes),
      kBufferSize, kCustomAlignment,
      absl::MakeSpan(kStrides.data(), kStrides.size()));
  ASSERT_TRUE(requirements);

  // Verify alignment
  auto alignment = requirements->Alignment();
  ASSERT_TRUE(alignment);
  ASSERT_EQ(*alignment, kCustomAlignment);

  // Verify other fields are still correct
  auto supported_types = requirements->SupportedTypes();
  ASSERT_TRUE(supported_types);
  ASSERT_EQ(supported_types->size(), kNumSupportedTensorBufferTypes);

  auto size = requirements->BufferSize();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, kBufferSize);

  auto strides = requirements->Strides();
  ASSERT_TRUE(strides);
  ASSERT_EQ(strides->size(), kStrides.size());
}

TEST(TensorBufferRequirements, DefaultAlignment) {
  auto requirements = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes,
                     kNumSupportedTensorBufferTypes),
      kBufferSize);
  ASSERT_TRUE(requirements);

  // Verify default alignment
  auto alignment = requirements->Alignment();
  ASSERT_TRUE(alignment);
  ASSERT_EQ(*alignment, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
}

TEST(TensorBufferRequirements, JoinWithDifferentAlignments) {
  constexpr size_t kAlignment = 256;
  constexpr std::array<uint32_t, 2> kStrides = {100, 4};

  // Create first requirements with alignment 64 (default)
  auto req1 = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes,
                     kNumSupportedTensorBufferTypes),
      1000, absl::MakeSpan(kStrides.data(), kStrides.size()));
  ASSERT_TRUE(req1);

  // Create second requirements with alignment 256
  auto req2 = litert::TensorBufferRequirements::CreateWithAlignment(
      absl::MakeSpan(kSupportedTensorBufferTypes,
                     kNumSupportedTensorBufferTypes),
      2000, kAlignment, absl::MakeSpan(kStrides.data(), kStrides.size()));
  ASSERT_TRUE(req2);

  // Join them
  auto joined = litert::Join(*req1, *req2);
  ASSERT_TRUE(joined);

  // Verify joined requirements has max alignment (256)
  auto alignment = joined->Alignment();
  ASSERT_TRUE(alignment);
  ASSERT_EQ(*alignment, kAlignment);

  // Verify joined requirements has max buffer size (2000)
  auto buffer_size = joined->BufferSize();
  ASSERT_TRUE(buffer_size);
  ASSERT_EQ(*buffer_size, 2000);
}

TEST(TensorBufferRequirements, InvalidAlignment) {
  // Test that invalid alignment is handled properly
  // Note: The C API validates alignment, so we test indirectly
  constexpr std::array<uint32_t, 2> kStrides = {100, 4};

  // Try to create with non-power-of-2 alignment via C API directly
  LiteRtTensorBufferRequirements litert_requirements;
  std::vector<LiteRtTensorBufferType> litert_buffer_types;
  litert_buffer_types.reserve(kNumSupportedTensorBufferTypes);
  for (const auto& buffer_type : kSupportedTensorBufferTypes) {
    litert_buffer_types.push_back(
        static_cast<LiteRtTensorBufferType>(buffer_type));
  }
  ASSERT_EQ(LiteRtCreateTensorBufferRequirementsWithAlignment(
                kNumSupportedTensorBufferTypes, litert_buffer_types.data(),
                kBufferSize, kStrides.size(), kStrides.data(),
                100,  // Not a power of 2
                &litert_requirements),
            kLiteRtStatusErrorInvalidArgument);

  // Try with zero alignment
  ASSERT_EQ(LiteRtCreateTensorBufferRequirementsWithAlignment(
                kNumSupportedTensorBufferTypes, litert_buffer_types.data(),
                kBufferSize, kStrides.size(), kStrides.data(),
                0,  // Zero
                &litert_requirements),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(TensorBufferRequirements, SupportedType) {
  auto requirements = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes,
                     kNumSupportedTensorBufferTypes),
      kBufferSize);
  ASSERT_TRUE(requirements);

  for (auto i = 0; i < kNumSupportedTensorBufferTypes; ++i) {
    auto type = requirements->SupportedType(i);
    ASSERT_TRUE(type);
    EXPECT_EQ(*type, kSupportedTensorBufferTypes[i]);
  }
}

TEST(TensorBufferRequirements, EmptyStrides) {
  auto requirements = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes,
                     kNumSupportedTensorBufferTypes),
      kBufferSize);
  ASSERT_TRUE(requirements);

  auto strides = requirements->Strides();
  ASSERT_TRUE(strides);
  EXPECT_EQ(strides->size(), 0);
}
