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

#include <array>
#include <filesystem>  // NOLINT
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/filesystem.h"
#include "litert/runtime/dispatch/litert_dispatch_test_helper.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/simple_buffer.h"
#include "litert/vendors/c/litert_dispatch.h"

namespace litert::compatibility {
namespace {

using ::litert::internal::Join;
using ::litert::internal::ResetDispatchForTest;
using ::litert::testing::GetLiteRtPath;
using ::litert::testing::SimpleBuffer;
using ::testing::ElementsAre;

class DispatchCompatibilityTest : public ::testing::Test {
 protected:
  void SetUp() override { ResetDispatchForTest(); }

  void TearDown() override {
    if (invocation_context_) {
      LiteRtDispatchInvocationContextDestroy(invocation_context_);
      invocation_context_ = nullptr;
    }
    if (device_context_) {
      LiteRtDispatchDeviceContextDestroy(device_context_);
      device_context_ = nullptr;
    }
    if (options_) {
      LiteRtDestroyOptions(options_);
      options_ = nullptr;
    }
    if (env_) {
      LiteRtDestroyEnvironment(env_);
      env_ = nullptr;
    }
    ResetDispatchForTest();
  }

  LiteRtStatus InitDispatch(absl::string_view subdir,
                            absl::string_view so_name) {
    std::filesystem::path dir =
        std::filesystem::path(::testing::TempDir()) / std::string(subdir);
    std::filesystem::create_directories(dir);
    std::string src =
        Join({GetLiteRtPath("vendors/examples/compatibility_test"), so_name});
    std::string dst = (dir / std::string(so_name)).string();
    std::filesystem::copy_file(
        src, dst, std::filesystem::copy_options::overwrite_existing);

    dispatch_dir_ = dir.string();
    std::array env_options_for_create = {
        LiteRtEnvOption{
            kLiteRtEnvOptionTagDispatchLibraryDir,
            LiteRtAny{.type = kLiteRtAnyTypeString,
                      .str_value = dispatch_dir_.c_str()},
        },
    };
    LITERT_RETURN_IF_ERROR(LiteRtCreateEnvironment(
        env_options_for_create.size(), env_options_for_create.data(), &env_));

    LITERT_RETURN_IF_ERROR(LiteRtCreateOptions(&options_));

    return LiteRtDispatchInitialize(LrtGetRuntimeContext(), env_, options_);
  }

  void ExecuteAndVerifyMultiply() {
    LITERT_ASSERT_OK(LiteRtDispatchDeviceContextCreate(
        LrtGetRuntimeContext(), options_, &device_context_));

    LiteRtMemBuffer bytecode_buffer = {-1, nullptr, 0, 0};
    LITERT_ASSERT_OK(LiteRtDispatchInvocationContextCreate(
        LrtGetRuntimeContext(), device_context_,
        kLiteRtDispatchExecutableTypeJitHandle, &bytecode_buffer,
        /*function_name=*/"mul", /*num_inputs=*/2, /*num_outputs=*/1,
        &invocation_context_));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input1,
        SimpleBuffer::Create<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
    LITERT_ASSERT_OK_AND_ASSIGN(auto input_tb1, input1.SpawnTensorBuffer());
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input2,
        SimpleBuffer::Create<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
    LITERT_ASSERT_OK_AND_ASSIGN(auto input_tb2, input2.SpawnTensorBuffer());
    LITERT_ASSERT_OK_AND_ASSIGN(auto output,
                                SimpleBuffer::Create<float>({2, 2}));
    LITERT_ASSERT_OK_AND_ASSIGN(auto output_tb, output.SpawnTensorBuffer());

    LiteRtTensorBufferHandle handle1;
    LITERT_ASSERT_OK(LiteRtDispatchRegisterTensorBuffer(
        device_context_, input_tb1.Get(), &handle1));
    LiteRtTensorBufferHandle handle2;
    LITERT_ASSERT_OK(LiteRtDispatchRegisterTensorBuffer(
        device_context_, input_tb2.Get(), &handle2));
    LiteRtTensorBufferHandle handle3;
    LITERT_ASSERT_OK(LiteRtDispatchRegisterTensorBuffer(
        device_context_, output_tb.Get(), &handle3));

    LITERT_ASSERT_OK(
        LiteRtDispatchAttachInput(invocation_context_, 0, handle1));
    LITERT_ASSERT_OK(
        LiteRtDispatchAttachInput(invocation_context_, 1, handle2));
    LITERT_ASSERT_OK(
        LiteRtDispatchAttachOutput(invocation_context_, 0, handle3));

    LITERT_ASSERT_OK(LiteRtDispatchInvoke(invocation_context_));

    std::vector<float> out(4);
    LITERT_ASSERT_OK(output_tb.Read(absl::MakeSpan(out)));
    EXPECT_THAT(out, ElementsAre(1.0f, 4.0f, 9.0f, 16.0f));
  }

  std::string dispatch_dir_;
  LiteRtEnvironment env_ = nullptr;
  LiteRtOptions options_ = nullptr;
  LiteRtDispatchDeviceContext device_context_ = nullptr;
  LiteRtDispatchInvocationContext invocation_context_ = nullptr;
};

// -----------------------------------------------------------------------------
// Scenario 1: Older Vendor and New LiteRT
// -----------------------------------------------------------------------------

// 1a: Older vendor returns supported interface (V1.0 with truncated
// struct_size).
// Verifies runtime negotiates V1.0, detects newer APIs as absent via
// LITERT_ABI_HAS_API, and successfully executes tensor execution via Invoke.
TEST_F(DispatchCompatibilityTest, Scenario1a_OlderVendor_SupportedInterface) {
  LITERT_ASSERT_OK(
      InitDispatch("older_supported", "libLiteRtDispatch_OlderSupported.so"));

  LiteRtApiVersion api_version;
  LITERT_ASSERT_OK(LiteRtDispatchGetApiVersion(&api_version));
  EXPECT_EQ(api_version.major, 1);
  EXPECT_EQ(api_version.minor, 0);

  const char* vendor_id = nullptr;
  LITERT_ASSERT_OK(LiteRtDispatchGetVendorId(&vendor_id));
  EXPECT_STREQ(vendor_id, "OlderSupportedDispatchVendor");

  // Realistic float matrix multiply execution works cleanly on older vendor.
  ExecuteAndVerifyMultiply();

  // Optional API beyond struct_size (invocation_context_set_options) is safely
  // detected as absent and returns kLiteRtStatusErrorUnsupported.
  LiteRtStatus opt_status =
      LiteRtDispatchInvocationContextSetOptions(invocation_context_, options_);
  EXPECT_EQ(opt_status, kLiteRtStatusErrorUnsupported);
}

// 1b: Older vendor returns non-supported interface (query returns unsupported).
TEST_F(DispatchCompatibilityTest, Scenario1b_OlderVendor_UnsupportedInterface) {
  LiteRtStatus status = InitDispatch("older_unsupported",
                                     "libLiteRtDispatch_OlderUnsupported.so");
  EXPECT_EQ(status, kLiteRtStatusErrorWrongVersion);
}

// -----------------------------------------------------------------------------
// Scenario 2: Newer Vendor and Old LiteRT
// -----------------------------------------------------------------------------

// 2a: Newer vendor returns supported interface (V1.2 with appended extension).
// Verifies runtime negotiates V1.2, safely ignores appended extension methods,
// and executes tensor operations successfully.
TEST_F(DispatchCompatibilityTest, Scenario2a_NewerVendor_SupportedInterface) {
  LITERT_ASSERT_OK(
      InitDispatch("newer_supported", "libLiteRtDispatch_NewerSupported.so"));

  LiteRtApiVersion api_version;
  LITERT_ASSERT_OK(LiteRtDispatchGetApiVersion(&api_version));
  EXPECT_EQ(api_version.major, 1);
  EXPECT_EQ(api_version.minor, 2);

  const char* vendor_id = nullptr;
  LITERT_ASSERT_OK(LiteRtDispatchGetVendorId(&vendor_id));
  EXPECT_STREQ(vendor_id, "NewerSupportedDispatchVendor");

  // Realistic float matrix multiply execution works cleanly on newer vendor.
  ExecuteAndVerifyMultiply();
}

// 2b: Newer vendor returns non-supported interface (future V2-only vendor).
TEST_F(DispatchCompatibilityTest, Scenario2b_NewerVendor_UnsupportedInterface) {
  LiteRtStatus status = InitDispatch("newer_unsupported",
                                     "libLiteRtDispatch_NewerUnsupported.so");
  EXPECT_EQ(status, kLiteRtStatusErrorWrongVersion);
}

// -----------------------------------------------------------------------------
// Scenario 3: Same Version Vendor and LiteRT
// -----------------------------------------------------------------------------

// 3a: Same version vendor returns supported interface (V1.1 full matching).
TEST_F(DispatchCompatibilityTest, Scenario3a_SameVersion_SupportedInterface) {
  LITERT_ASSERT_OK(
      InitDispatch("same_supported", "libLiteRtDispatch_SameSupported.so"));

  LiteRtApiVersion api_version;
  LITERT_ASSERT_OK(LiteRtDispatchGetApiVersion(&api_version));
  EXPECT_EQ(api_version.major, 1);
  EXPECT_EQ(api_version.minor, 0);

  const char* vendor_id = nullptr;
  LITERT_ASSERT_OK(LiteRtDispatchGetVendorId(&vendor_id));
  EXPECT_STREQ(vendor_id, "SameSupportedDispatchVendor");

  // Real execution works cleanly.
  ExecuteAndVerifyMultiply();

  // Full interface supports invocation_context_set_options.
  LITERT_ASSERT_OK(
      LiteRtDispatchInvocationContextSetOptions(invocation_context_, options_));
}

// 3b: Vendor returns non-supported interface (returns OK but null interface).
TEST_F(DispatchCompatibilityTest,
       Scenario3b_SameVersion_NullInterfaceRejected) {
  LiteRtStatus status =
      InitDispatch("null_interface", "libLiteRtDispatch_NullInterface.so");
  EXPECT_EQ(status, kLiteRtStatusErrorWrongVersion);
}

// -----------------------------------------------------------------------------
// Scenario 4: Completely Incompatible Vendor and LiteRT
// -----------------------------------------------------------------------------

// 4a: Incompatible major version (major_version = 99).
TEST_F(DispatchCompatibilityTest, Scenario4a_IncompatibleMajorVersionRejected) {
  LiteRtStatus status =
      InitDispatch("incompatible_major", "libLiteRtDispatch_Incompatible.so");
  EXPECT_EQ(status, kLiteRtStatusErrorWrongVersion);
}

// 4b: Corrupted header (struct_size = 4 < sizeof(LiteRtAbiHeader)).
TEST_F(DispatchCompatibilityTest, Scenario4b_CorruptedHeaderRejected) {
  LiteRtStatus status =
      InitDispatch("corrupted_header", "libLiteRtDispatch_CorruptedHeader.so");
  EXPECT_EQ(status, kLiteRtStatusErrorWrongVersion);
}

}  // namespace
}  // namespace litert::compatibility
