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

#include <gtest/gtest.h>
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"

namespace litert::example {
namespace {

class ExampleDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    LITERT_ASSERT_OK(LiteRtDispatchGetApi(&api_));
    ASSERT_NE(api_.interface, nullptr);
  }

  LiteRtDispatchInterface& Api() { return *api_.interface; }

 private:
  LiteRtDispatchApi api_;
};

TEST_F(ExampleDispatchTest, GetVendorId) {
  const char* vendor_id;
  LITERT_ASSERT_OK(Api().get_vendor_id(&vendor_id));
  EXPECT_STREQ(vendor_id, "Example");
}

TEST_F(ExampleDispatchTest, GetBuildId) {
  const char* build_id;
  LITERT_ASSERT_OK(Api().get_build_id(&build_id));
  EXPECT_STREQ(build_id, "ExampleBuild");
}

TEST_F(ExampleDispatchTest, GetCapabilities) {
  int capabilities;
  LITERT_ASSERT_OK(Api().get_capabilities(&capabilities));
  EXPECT_EQ(capabilities, kLiteRtDispatchCapabilitiesBasic);
}

}  // namespace
}  // namespace litert::example
