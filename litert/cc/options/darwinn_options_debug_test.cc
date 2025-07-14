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

#include <iostream>

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

TEST(DarwinnDeviceOptionsDebugTest, CreateOnly) {
  // Create device options
  auto device_options = DarwinnDeviceOptions::Create();
  
  std::cout << "Create status: " << (device_options.HasValue() ? "OK" : "ERROR") << std::endl;
  
  if (!device_options.HasValue()) {
    std::cout << "Error message: " << device_options.Error().Message() << std::endl;
    FAIL() << "Failed to create DarwinnDeviceOptions";
  }
  
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, std::move(device_options));
  
  // Try to get a value
  auto device_type_result = options.GetDeviceType();
  std::cout << "GetDeviceType status: " << (device_type_result.HasValue() ? "OK" : "ERROR") << std::endl;
  
  if (!device_type_result.HasValue()) {
    std::cout << "GetDeviceType error: " << device_type_result.Error().Message() << std::endl;
  } else {
    std::cout << "Device type: '" << device_type_result.Value() << "'" << std::endl;
  }
}

}  // namespace
}  // namespace litert