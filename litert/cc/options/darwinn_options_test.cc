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

#include "litert/cc/options/darwinn_options_direct.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {
using testing::StrEq;
using testing::litert::IsOk;
using testing::litert::IsOkAndHolds;

TEST(DarwinnDeviceOptionsTest, CreateAndFindDeviceOptions) {
  // Create device options
  LITERT_ASSERT_OK_AND_ASSIGN(auto device_options, 
                              DarwinnDeviceOptions::Create());
  
  // Set some values
  LITERT_EXPECT_OK(device_options.SetDeviceType("usb"));
  LITERT_EXPECT_OK(device_options.SetUseAsyncApi(true));
  LITERT_EXPECT_OK(device_options.SetNumInterpreters(4));
  
  // Verify the values
  EXPECT_THAT(device_options.GetDeviceType(), IsOkAndHolds(StrEq("usb")));
  EXPECT_THAT(device_options.GetUseAsyncApi(), IsOkAndHolds(true));
  EXPECT_THAT(device_options.GetNumInterpreters(), IsOkAndHolds(4));
  
  // Find options in the list
  OpaqueOptions opaque_options = std::move(device_options);
  LITERT_ASSERT_OK_AND_ASSIGN(auto found_options,
                              FindOpaqueOptions<DarwinnDeviceOptions>(opaque_options));
  
  // Verify found options have the same values
  EXPECT_THAT(found_options.GetDeviceType(), IsOkAndHolds(StrEq("usb")));
  EXPECT_THAT(found_options.GetUseAsyncApi(), IsOkAndHolds(true));
}

TEST(DarwinnRuntimeOptionsTest, CreateAndFindRuntimeOptions) {
  // Create runtime options
  LITERT_ASSERT_OK_AND_ASSIGN(auto runtime_options,
                              DarwinnRuntimeOptions::Create());
  
  // Set some values
  LITERT_EXPECT_OK(runtime_options.SetInferencePowerState(3));
  LITERT_EXPECT_OK(runtime_options.SetInferencePriority(5));
  LITERT_EXPECT_OK(runtime_options.SetAtomicInference(true));
  
  // Verify the values
  EXPECT_THAT(runtime_options.GetInferencePowerState(), IsOkAndHolds(3));
  EXPECT_THAT(runtime_options.GetInferencePriority(), IsOkAndHolds(5));
  EXPECT_THAT(runtime_options.GetAtomicInference(), IsOkAndHolds(true));
  
  // Find options in the list
  OpaqueOptions opaque_options = std::move(runtime_options);
  LITERT_ASSERT_OK_AND_ASSIGN(auto found_options,
                              FindOpaqueOptions<DarwinnRuntimeOptions>(opaque_options));
  
  // Verify found options have the same values
  EXPECT_THAT(found_options.GetInferencePowerState(), IsOkAndHolds(3));
}

TEST(DarwinnOptionsTest, MultipleDarwinnOptions) {
  // Create both device and runtime options
  LITERT_ASSERT_OK_AND_ASSIGN(auto device_options,
                              DarwinnDeviceOptions::Create());
  LITERT_EXPECT_OK(device_options.SetDeviceType("pci"));
  
  LITERT_ASSERT_OK_AND_ASSIGN(auto runtime_options,
                              DarwinnRuntimeOptions::Create());
  LITERT_EXPECT_OK(runtime_options.SetInferencePriority(10));
  
  // Chain them together
  LITERT_EXPECT_OK(device_options.Append(std::move(runtime_options)));
  
  // Find each type
  OpaqueOptions opaque_options = std::move(device_options);
  LITERT_ASSERT_OK_AND_ASSIGN(auto found_device,
                              FindOpaqueOptions<DarwinnDeviceOptions>(opaque_options));
  EXPECT_THAT(found_device.GetDeviceType(), IsOkAndHolds(StrEq("pci")));
  
  LITERT_ASSERT_OK_AND_ASSIGN(auto found_runtime,
                              FindOpaqueOptions<DarwinnRuntimeOptions>(opaque_options));
  EXPECT_THAT(found_runtime.GetInferencePriority(), IsOkAndHolds(10));
}

TEST(DarwinnDeviceOptionsTest, CheckDefaultValues) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, DarwinnDeviceOptions::Create());
  
  // Check default values
  EXPECT_THAT(options.GetDeviceType(), IsOkAndHolds(StrEq("")));
  EXPECT_THAT(options.GetDevicePath(), IsOkAndHolds(StrEq("")));
  EXPECT_THAT(options.GetEnableMultipleSubgraphs(), IsOkAndHolds(false));
  EXPECT_THAT(options.GetCompileIfResize(), IsOkAndHolds(false));
  EXPECT_THAT(options.GetAllowCpuFallback(), IsOkAndHolds(false));
  EXPECT_THAT(options.GetSkipOpFilter(), IsOkAndHolds(false));
  EXPECT_THAT(options.GetNumInterpreters(), IsOkAndHolds(1));
  EXPECT_THAT(options.GetAvoidBounceBuffer(), IsOkAndHolds(false));
  EXPECT_THAT(options.GetUseAsyncApi(), IsOkAndHolds(false));
  EXPECT_THAT(options.GetUseTachyon(), IsOkAndHolds(false));
  EXPECT_THAT(options.GetDisableLogInfo(), IsOkAndHolds(false));
}

TEST(DarwinnRuntimeOptionsTest, CheckDefaultValues) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, DarwinnRuntimeOptions::Create());
  
  // Check default values
  EXPECT_THAT(options.GetInferencePowerState(), IsOkAndHolds(0));
  EXPECT_THAT(options.GetInferenceMemoryPowerState(), IsOkAndHolds(0));
  EXPECT_THAT(options.GetInferencePriority(), IsOkAndHolds(-1));
  EXPECT_THAT(options.GetAtomicInference(), IsOkAndHolds(false));
  EXPECT_THAT(options.GetInactivePowerState(), IsOkAndHolds(0));
  EXPECT_THAT(options.GetInactiveMemoryPowerState(), IsOkAndHolds(0));
  EXPECT_THAT(options.GetInactiveTimeoutUs(), IsOkAndHolds(1000000));
}

}  // namespace
}  // namespace litert