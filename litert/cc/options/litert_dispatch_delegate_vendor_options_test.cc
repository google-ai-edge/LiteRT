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

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

TEST(DispatchDelegateVendorOptionsTest, Discriminator) {
  EXPECT_STREQ(DispatchDelegateVendorOptions::Discriminator(),
               "dispatch_delegate_vendor_options");
}

TEST(DispatchDelegateVendorOptionsTest, AddAndGetFunctionMappings) {
  DispatchDelegateVendorOptions options;
  EXPECT_TRUE(options.FunctionMappings().empty());

  std::vector<TensorPortMapping> input_ports = {
      {"input_0", 0},
      {"input_1", 1},
  };
  std::vector<TensorPortMapping> output_ports = {
      {"output_0", 0},
  };

  options.AddFunctionMapping({
      "subgraph_0",
      "serving_default",
      std::move(input_ports),
      std::move(output_ports),
  });

  absl::Span<const FunctionMapping> mappings = options.FunctionMappings();
  ASSERT_EQ(mappings.size(), 1);
  EXPECT_EQ(mappings[0].function_name, "subgraph_0");
  EXPECT_EQ(mappings[0].signature_name, "serving_default");

  ASSERT_EQ(mappings[0].input_tensor_ports.size(), 2);
  EXPECT_EQ(mappings[0].input_tensor_ports[0].tensor_name, "input_0");
  EXPECT_EQ(mappings[0].input_tensor_ports[0].port_index, 0);
  EXPECT_EQ(mappings[0].input_tensor_ports[1].tensor_name, "input_1");
  EXPECT_EQ(mappings[0].input_tensor_ports[1].port_index, 1);

  ASSERT_EQ(mappings[0].output_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[0].output_tensor_ports[0].tensor_name, "output_0");
  EXPECT_EQ(mappings[0].output_tensor_ports[0].port_index, 0);

  // Add a second function mapping.
  options.AddFunctionMapping(FunctionMapping{
      /*function_name=*/"subgraph_1",
      /*signature_name=*/"decode",
      /*input_tensor_ports=*/{{"tokens", 0}},
      /*output_tensor_ports=*/{{"logits", 0}},
  });

  mappings = options.FunctionMappings();
  ASSERT_EQ(mappings.size(), 2);
  EXPECT_EQ(mappings[1].function_name, "subgraph_1");
  EXPECT_EQ(mappings[1].signature_name, "decode");
}

TEST(DispatchDelegateVendorOptionsTest, TomlRoundTripEmpty) {
  DispatchDelegateVendorOptions options;
  std::string toml = options.ToToml();
  EXPECT_TRUE(toml.empty());

  LITERT_ASSERT_OK_AND_ASSIGN(
      DispatchDelegateVendorOptions parsed,
      DispatchDelegateVendorOptions::CreateFromToml(toml));
  EXPECT_TRUE(parsed.FunctionMappings().empty());
}

TEST(DispatchDelegateVendorOptionsTest, TomlRoundTripWithData) {
  DispatchDelegateVendorOptions options;
  options.AddFunctionMapping({
      "DispatchOp_0",
      "ple_proj_1x4",
      /*input_tensor_ports=*/{{"embeddings", 0}, {"per_layer_embedding", 1}},
      /*output_tensor_ports=*/
      {{"activations", 0}, {"projected_per_layer_embedding0", 1}},
  });

  options.AddFunctionMapping({
      "DispatchOp_1",
      "ple_proj_1x8",
      /*input_tensor_ports=*/{{"tokens", 0}},
      /*output_tensor_ports=*/{{"logits", 0}},
  });

  std::string toml = options.ToToml();
  constexpr absl::string_view kExpectedToml =
      "func.DispatchOp_0.signature = \"ple_proj_1x4\"\n"
      "func.DispatchOp_0.in.0 = \"embeddings\"\n"
      "func.DispatchOp_0.in.1 = \"per_layer_embedding\"\n"
      "func.DispatchOp_0.out.0 = \"activations\"\n"
      "func.DispatchOp_0.out.1 = \"projected_per_layer_embedding0\"\n"
      "func.DispatchOp_1.signature = \"ple_proj_1x8\"\n"
      "func.DispatchOp_1.in.0 = \"tokens\"\n"
      "func.DispatchOp_1.out.0 = \"logits\"\n";
  EXPECT_EQ(toml, kExpectedToml);

  LITERT_ASSERT_OK_AND_ASSIGN(
      DispatchDelegateVendorOptions parsed,
      DispatchDelegateVendorOptions::CreateFromToml(toml));

  absl::Span<const FunctionMapping> mappings = parsed.FunctionMappings();
  ASSERT_EQ(mappings.size(), 2);

  EXPECT_EQ(mappings[0].function_name, "DispatchOp_0");
  EXPECT_EQ(mappings[0].signature_name, "ple_proj_1x4");
  ASSERT_EQ(mappings[0].input_tensor_ports.size(), 2);
  EXPECT_EQ(mappings[0].input_tensor_ports[0].tensor_name, "embeddings");
  EXPECT_EQ(mappings[0].input_tensor_ports[0].port_index, 0);
  EXPECT_EQ(mappings[0].input_tensor_ports[1].tensor_name,
            "per_layer_embedding");
  EXPECT_EQ(mappings[0].input_tensor_ports[1].port_index, 1);
  ASSERT_EQ(mappings[0].output_tensor_ports.size(), 2);
  EXPECT_EQ(mappings[0].output_tensor_ports[0].tensor_name, "activations");
  EXPECT_EQ(mappings[0].output_tensor_ports[0].port_index, 0);
  EXPECT_EQ(mappings[0].output_tensor_ports[1].tensor_name,
            "projected_per_layer_embedding0");
  EXPECT_EQ(mappings[0].output_tensor_ports[1].port_index, 1);

  EXPECT_EQ(mappings[1].function_name, "DispatchOp_1");
  EXPECT_EQ(mappings[1].signature_name, "ple_proj_1x8");
  ASSERT_EQ(mappings[1].input_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[1].input_tensor_ports[0].tensor_name, "tokens");
  EXPECT_EQ(mappings[1].input_tensor_ports[0].port_index, 0);
  ASSERT_EQ(mappings[1].output_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[1].output_tensor_ports[0].tensor_name, "logits");
  EXPECT_EQ(mappings[1].output_tensor_ports[0].port_index, 0);
}

TEST(DispatchDelegateVendorOptionsTest, CreateFromTomlParsesProtocolString) {
  constexpr absl::string_view kToml =
      "func.DispatchOp_0.signature = \"serving_default\"\n"
      "func.DispatchOp_0.in.0 = \"input_0\"\n"
      "func.DispatchOp_0.out.0 = \"output_0\"\n"
      "func.DispatchOp_0.future_extension_field = \"ignored\"\n";

  LITERT_ASSERT_OK_AND_ASSIGN(
      DispatchDelegateVendorOptions parsed,
      DispatchDelegateVendorOptions::CreateFromToml(kToml));

  absl::Span<const FunctionMapping> mappings = parsed.FunctionMappings();
  ASSERT_EQ(mappings.size(), 1);
  EXPECT_EQ(mappings[0].function_name, "DispatchOp_0");
  EXPECT_EQ(mappings[0].signature_name, "serving_default");
  ASSERT_EQ(mappings[0].input_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[0].input_tensor_ports[0].tensor_name, "input_0");
  EXPECT_EQ(mappings[0].input_tensor_ports[0].port_index, 0);
  ASSERT_EQ(mappings[0].output_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[0].output_tensor_ports[0].tensor_name, "output_0");
  EXPECT_EQ(mappings[0].output_tensor_ports[0].port_index, 0);
}

TEST(DispatchDelegateVendorOptionsTest, OpaqueOptionsRoundTrip) {
  DispatchDelegateVendorOptions options;
  options.AddFunctionMapping({
      "DispatchOp_0",
      "serving_default",
      /*input_tensor_ports=*/{{"input_tensor", 0}},
      /*output_tensor_ports=*/{{"output_tensor", 0}},
  });

  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(options.GetOpaqueOptionsData(&identifier, &payload,
                                               &payload_deleter));
  EXPECT_STREQ(identifier, DispatchDelegateVendorOptions::kIdentifier);
  EXPECT_NE(payload, nullptr);
  EXPECT_NE(payload_deleter, nullptr);

  LITERT_ASSERT_OK_AND_ASSIGN(
      OpaqueOptions opaque_node,
      OpaqueOptions::Create(identifier, payload, payload_deleter));

  LITERT_ASSERT_OK_AND_ASSIGN(
      DispatchDelegateVendorOptions restored,
      DispatchDelegateVendorOptions::FromOpaqueOptions(opaque_node));

  absl::Span<const FunctionMapping> mappings = restored.FunctionMappings();
  ASSERT_EQ(mappings.size(), 1);
  EXPECT_EQ(mappings[0].function_name, "DispatchOp_0");
  EXPECT_EQ(mappings[0].signature_name, "serving_default");
  ASSERT_EQ(mappings[0].input_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[0].input_tensor_ports[0].tensor_name, "input_tensor");
  EXPECT_EQ(mappings[0].input_tensor_ports[0].port_index, 0);
  ASSERT_EQ(mappings[0].output_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[0].output_tensor_ports[0].tensor_name, "output_tensor");
  EXPECT_EQ(mappings[0].output_tensor_ports[0].port_index, 0);
}

}  // namespace
}  // namespace litert
