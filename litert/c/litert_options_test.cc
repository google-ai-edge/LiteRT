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

#include "litert/c/litert_options.h"

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace {

TEST(LiteRtCompiledModelOptionsTest, CreateAndDestroyDontLeak) {
  LiteRtOptions options;
  ASSERT_EQ(LiteRtCreateOptions(&options), kLiteRtStatusOk);
  LiteRtDestroyOptions(options);
}

TEST(LiteRtCompiledModelOptionsTest, CreateWithANullPointerErrors) {
  EXPECT_EQ(LiteRtCreateOptions(nullptr), kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtCompiledModelOptionsTest, SetAndGetHardwareAcceleratorsWorks) {
  LiteRtOptions options;
  ASSERT_EQ(LiteRtCreateOptions(&options), kLiteRtStatusOk);

  LiteRtHwAcceleratorSet hardware_accelerators;

  EXPECT_EQ(
      LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorNone),
      kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtGetOptionsHardwareAccelerators(options, &hardware_accelerators),
      kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAcceleratorNone);

  EXPECT_EQ(
      LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu),
      kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtGetOptionsHardwareAccelerators(options, &hardware_accelerators),
      kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAcceleratorCpu);

  EXPECT_EQ(
      LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorGpu),
      kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtGetOptionsHardwareAccelerators(options, &hardware_accelerators),
      kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAcceleratorGpu);

  EXPECT_EQ(
      LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorNpu),
      kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtGetOptionsHardwareAccelerators(options, &hardware_accelerators),
      kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAcceleratorNpu);

  EXPECT_EQ(LiteRtSetOptionsHardwareAccelerators(
                options, (kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu |
                          kLiteRtHwAcceleratorNpu) +
                             1),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LiteRtSetOptionsHardwareAccelerators(nullptr, kLiteRtHwAcceleratorNone),
      kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyOptions(options);
}

struct DummyAcceleratorCompilationOptions {
  static constexpr const char* const kIdentifier = "dummy-accelerator";

  // Allocates and sets the basic structure for the accelerator options.
  static litert::Expected<LiteRtOpaqueOptions> CreateOptions() {
    LiteRtOpaqueOptions options;
    auto* payload = new DummyAcceleratorCompilationOptions;
    auto payload_destructor = [](void* payload) {
      delete reinterpret_cast<DummyAcceleratorCompilationOptions*>(payload);
    };
    LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
        kIdentifier, payload, payload_destructor, &options));
    return options;
  }
};

TEST(LiteRtCompiledModelOptionsTest, AddAcceleratorCompilationOptionsWorks) {
  LiteRtOptions options;
  ASSERT_EQ(LiteRtCreateOptions(&options), kLiteRtStatusOk);

  auto accelerator_compilation_options1 =
      DummyAcceleratorCompilationOptions::CreateOptions();
  EXPECT_TRUE(accelerator_compilation_options1);
  auto accelerator_compilation_options2 =
      DummyAcceleratorCompilationOptions::CreateOptions();
  EXPECT_TRUE(accelerator_compilation_options2);

  EXPECT_EQ(LiteRtAddOpaqueOptions(nullptr, *accelerator_compilation_options1),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtAddOpaqueOptions(options, nullptr),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtAddOpaqueOptions(options, *accelerator_compilation_options1),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtAddOpaqueOptions(options, *accelerator_compilation_options2),
            kLiteRtStatusOk);

  LiteRtOpaqueOptions options_it = nullptr;
  EXPECT_EQ(LiteRtGetOpaqueOptions(options, &options_it), kLiteRtStatusOk);
  EXPECT_EQ(options_it, *accelerator_compilation_options1);

  EXPECT_EQ(LiteRtGetNextOpaqueOptions(&options_it), kLiteRtStatusOk);
  EXPECT_EQ(options_it, *accelerator_compilation_options2);

  LiteRtDestroyOptions(options);
}

}  // namespace
