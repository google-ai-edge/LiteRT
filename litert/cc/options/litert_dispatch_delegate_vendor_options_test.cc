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

#include "litert/cc/options/litert_dispatch_delegate_vendor_options.h"

#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"

namespace litert {
namespace {

TEST(DispatchDelegateVendorOptionsTest, Discriminator) {
  EXPECT_STREQ(DispatchDelegateVendorOptions::Discriminator(),
               "dispatch_delegate_vendor_options");
}

TEST(DispatchDelegateVendorOptionsTest, SetAndGetNodeTensorPortMapping) {
  DispatchDelegateVendorOptions options;

  std::vector<TensorPortMapping> input_ports = {
      {"input_0", 0},
      {"input_1", 1},
  };
  std::vector<TensorPortMapping> output_ports = {
      {"output_0", 0},
  };

  ASSERT_TRUE(
      options.SetNodeTensorPortMapping("subgraph_0", input_ports, output_ports)
          .HasValue());

  // Setting mapping again for the same function should fail with already
  // exists.
  auto duplicate =
      options.SetNodeTensorPortMapping("subgraph_0", input_ports, output_ports);
  EXPECT_FALSE(duplicate.HasValue());
  EXPECT_EQ(duplicate.Error().Status(), kLiteRtStatusErrorAlreadyExists);

  const auto* mapping = options.GetNodeTensorPortMapping("subgraph_0");
  ASSERT_NE(mapping, nullptr);

  ASSERT_EQ(mapping->input_tensor_ports.size(), 2);
  EXPECT_EQ(mapping->input_tensor_ports[0].opaque_tensor_name, "input_0");
  EXPECT_EQ(mapping->input_tensor_ports[0].port_index, 0);
  EXPECT_EQ(mapping->input_tensor_ports[1].opaque_tensor_name, "input_1");
  EXPECT_EQ(mapping->input_tensor_ports[1].port_index, 1);

  ASSERT_EQ(mapping->output_tensor_ports.size(), 1);
  EXPECT_EQ(mapping->output_tensor_ports[0].opaque_tensor_name, "output_0");
  EXPECT_EQ(mapping->output_tensor_ports[0].port_index, 0);
}

TEST(DispatchDelegateVendorOptionsTest, GetNodeTensorPortMappingNotFound) {
  DispatchDelegateVendorOptions options;
  EXPECT_EQ(options.GetNodeTensorPortMapping("nonexistent"), nullptr);
}

}  // namespace
}  // namespace litert
