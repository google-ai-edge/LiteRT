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

#include "litert/cc/options/darwinn_options.h"

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

TEST(DarwinnDeviceOptionsTest, CreateAndSetValues) {
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
}

TEST(DarwinnRuntimeOptionsTest, CreateAndSetValues) {
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
}

}  // namespace
}  // namespace litert