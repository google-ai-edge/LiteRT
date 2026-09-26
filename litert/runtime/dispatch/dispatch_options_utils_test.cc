// Copyright 2026 Google LLC.
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

#include "litert/runtime/dispatch/dispatch_options_utils.h"

#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/options/litert_dispatch_delegate_vendor_options.h"

namespace litert::internal {
namespace {

TEST(DispatchOptionsUtilsTest, GetOrCreateVendorOptionsCreatesAndRetrieves) {
  LiteRtOpaqueOptions options = nullptr;

  auto vendor_options_or = GetOrCreateDispatchDelegateVendorOptions(options);
  ASSERT_TRUE(vendor_options_or.HasValue());
  DispatchDelegateVendorOptions* vendor_options = vendor_options_or.Value();
  ASSERT_NE(vendor_options, nullptr);

  // Set a port mapping.
  ASSERT_TRUE(vendor_options
                  ->SetNodeTensorPortMapping("subgraph_0", {{"input_0", 0}}, {})
                  .HasValue());

  // Calling it again on the same options chain should retrieve the existing
  // instance.
  auto second_or = GetOrCreateDispatchDelegateVendorOptions(options);
  ASSERT_TRUE(second_or.HasValue());
  EXPECT_EQ(second_or.Value(), vendor_options);

  const auto* mapping =
      second_or.Value()->GetNodeTensorPortMapping("subgraph_0");
  ASSERT_NE(mapping, nullptr);
  ASSERT_EQ(mapping->input_tensor_ports.size(), 1);
  EXPECT_EQ(mapping->input_tensor_ports[0].opaque_tensor_name, "input_0");

  LiteRtDestroyOpaqueOptions(options);
}

}  // namespace
}  // namespace litert::internal
