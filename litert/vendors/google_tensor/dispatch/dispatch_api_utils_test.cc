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

#include "litert/vendors/google_tensor/dispatch/dispatch_api_utils.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/optional_ref.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_options.h"
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/internal/litert_dispatch_delegate_vendor_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_test_fixtures.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

namespace litert::google_tensor {
namespace {

TEST(DispatchApiUtilsTest, ToThrNodeId) {
  ThrNodeId id0 = ToThrNodeId(0);
  EXPECT_STREQ(id0, "node_0");

  ThrNodeId id10 = ToThrNodeId(10);
  EXPECT_STREQ(id10, "node_10");

  // Verify pointer stability (same pointer returned on repeated calls).
  EXPECT_EQ(ToThrNodeId(10), id10);
}

TEST(DispatchApiUtilsTest, ToThrEdgeId) {
  ThrEdgeId id0 = ToThrEdgeId(0);
  EXPECT_STREQ(id0, "edge_0");

  ThrEdgeId id42 = ToThrEdgeId(42);
  EXPECT_STREQ(id42, "edge_42");

  // Verify pointer stability.
  EXPECT_EQ(ToThrEdgeId(42), id42);
}

TEST(DispatchApiUtilsTest, IsPortCoherent) {
  std::vector<internal::TensorPortMapping> ports = {
      {"input_0", 0},
      {"input_1", 1},
      {"intermediate", 2},
  };

  TensorCoherencyMap coherency_map = {
      {{"serving_default", "input_0"}, true},
      {{"serving_default", "input_1"}, false},
  };

  EXPECT_TRUE(IsPortCoherent(/*port_index=*/0, ports, &coherency_map,
                             "serving_default"));
  EXPECT_FALSE(IsPortCoherent(/*port_index=*/1, ports, &coherency_map,
                              "serving_default"));
  EXPECT_FALSE(IsPortCoherent(/*port_index=*/2, ports, &coherency_map,
                              "serving_default"));
  EXPECT_FALSE(IsPortCoherent(/*port_index=*/99, ports, &coherency_map,
                              "serving_default"));
  EXPECT_FALSE(IsPortCoherent(/*port_index=*/0, ports, &coherency_map,
                              "other_signature"));
  EXPECT_FALSE(
      IsPortCoherent(/*port_index=*/0, ports, &coherency_map, std::nullopt));
  EXPECT_FALSE(IsPortCoherent(/*port_index=*/0, ports, &coherency_map, ""));
  EXPECT_FALSE(IsPortCoherent(/*port_index=*/0, ports,
                              /*coherency_map=*/nullptr, "serving_default"));
  EXPECT_FALSE(IsPortCoherent(/*port_index=*/0, /*tensor_ports=*/{},
                              &coherency_map, "serving_default"));
}

using DispatchApiUtilsFixtureTest = testing::DispatchApiTest;

TEST_F(DispatchApiUtilsFixtureTest, FindDispatchDelegateVendorOptionsSuccess) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto cc_options, Options::Create());

  internal::DispatchDelegateVendorOptions vendor_options;
  internal::FunctionMapping mapping;
  mapping.function_name = "test_function";
  mapping.signature_name = "serving_default";
  mapping.input_tensor_ports.push_back({"input_tensor_0", 0});
  mapping.output_tensor_ports.push_back({"output_tensor_0", 0});
  vendor_options.AddFunctionMapping(std::move(mapping));

  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(vendor_options.GetOpaqueOptionsData(&identifier, &payload,
                                                       &payload_deleter));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto opaque_options,
      OpaqueOptions::Create(identifier, payload, payload_deleter));
  EXPECT_TRUE(cc_options.AddOpaqueOptions(std::move(opaque_options)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto litert_c_options,
      internal::LiteRtOptionsPtrBuilder::Build(cc_options, env.GetHolder()));

  LiteRtDispatchDeviceContext custom_context;
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextCreate(
      LrtGetRuntimeContext(), litert_c_options.get(), &custom_context));

  // Test direct lookup with runtime context and options.
  std::optional<internal::DispatchDelegateVendorOptions> direct_result =
      FindDispatchDelegateVendorOptions(LrtGetRuntimeContext(),
                                        litert_c_options.get());
  ASSERT_TRUE(direct_result.has_value());
  EXPECT_EQ(direct_result->FunctionMappings().size(), 1);

  // Test options parsed and stored on device context.
  absl::optional_ref<const internal::DispatchDelegateVendorOptions> result =
      custom_context->GetDispatchDelegateVendorOptions();
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->FunctionMappings().size(), 1);
  EXPECT_EQ(result->FunctionMappings()[0].function_name, "test_function");
  EXPECT_EQ(result->FunctionMappings()[0].signature_name, "serving_default");
  ASSERT_EQ(result->FunctionMappings()[0].input_tensor_ports.size(), 1);
  EXPECT_EQ(result->FunctionMappings()[0].input_tensor_ports[0].tensor_name,
            "input_tensor_0");
  EXPECT_EQ(result->FunctionMappings()[0].input_tensor_ports[0].port_index, 0);
  ASSERT_EQ(result->FunctionMappings()[0].output_tensor_ports.size(), 1);
  EXPECT_EQ(result->FunctionMappings()[0].output_tensor_ports[0].tensor_name,
            "output_tensor_0");
  EXPECT_EQ(result->FunctionMappings()[0].output_tensor_ports[0].port_index, 0);

  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextDestroy(custom_context));
}

TEST_F(DispatchApiUtilsFixtureTest,
       FindDispatchDelegateVendorOptionsMissingDiscriminator) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto cc_options, Options::Create());

  // Add an opaque option with an unrelated discriminator.
  const char* different_identifier = "unrelated_vendor_options";
  static char dummy_payload[] = "foo = 1";
  auto noop_deleter = [](void*) {};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto opaque_options,
      OpaqueOptions::Create(different_identifier, dummy_payload, noop_deleter));
  EXPECT_TRUE(cc_options.AddOpaqueOptions(std::move(opaque_options)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto litert_c_options,
      internal::LiteRtOptionsPtrBuilder::Build(cc_options, env.GetHolder()));

  LiteRtDispatchDeviceContext custom_context;
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextCreate(
      LrtGetRuntimeContext(), litert_c_options.get(), &custom_context));

  auto result = custom_context->GetDispatchDelegateVendorOptions();
  EXPECT_FALSE(result.has_value());

  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextDestroy(custom_context));
}

TEST_F(DispatchApiUtilsFixtureTest, AnnotateCoherencyEdgesSuccess) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto cc_options, Options::Create());

  // Configure GoogleTensorOptions coherency preference.
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_options,
                              GoogleTensorOptions::Create());
  tensor_options.SetInputCoherency("serving_default", "in_coherent", true);
  tensor_options.SetInputCoherency("serving_default", "in_non_coherent", false);
  tensor_options.SetOutputCoherency("serving_default", "out_coherent", true);

  const char* tensor_identifier = nullptr;
  void* tensor_payload = nullptr;
  void (*tensor_payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(tensor_options.GetOpaqueOptionsData(
      &tensor_identifier, &tensor_payload, &tensor_payload_deleter));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_opaque_options,
      OpaqueOptions::Create(tensor_identifier, tensor_payload,
                            tensor_payload_deleter));
  EXPECT_TRUE(cc_options.AddOpaqueOptions(std::move(tensor_opaque_options)));

  // Configure DispatchDelegateVendorOptions mapping.
  internal::DispatchDelegateVendorOptions vendor_options;
  internal::FunctionMapping mapping;
  mapping.function_name = "test_func";
  mapping.signature_name = "serving_default";
  mapping.input_tensor_ports.push_back({"in_coherent", 0});
  mapping.input_tensor_ports.push_back({"in_non_coherent", 1});
  mapping.output_tensor_ports.push_back({"out_coherent", 0});
  vendor_options.AddFunctionMapping(std::move(mapping));

  const char* vendor_options_identifier = nullptr;
  void* vendor_options_payload = nullptr;
  void (*vendor_options_payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(vendor_options.GetOpaqueOptionsData(
      &vendor_options_identifier, &vendor_options_payload,
      &vendor_options_payload_deleter));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vendor_opaque_options,
      OpaqueOptions::Create(vendor_options_identifier, vendor_options_payload,
                            vendor_options_payload_deleter));
  EXPECT_TRUE(cc_options.AddOpaqueOptions(std::move(vendor_opaque_options)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto litert_c_options,
      internal::LiteRtOptionsPtrBuilder::Build(cc_options, env.GetHolder()));

  LiteRtDispatchDeviceContext custom_context;
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextCreate(
      LrtGetRuntimeContext(), litert_c_options.get(), &custom_context));

  LiteRtDispatchGraph graph;
  LITERT_ASSERT_OK(LiteRtDispatchGraphCreate(custom_context, &graph));

  // Set up 2 inputs and 1 output on the graph.
  LITERT_ASSERT_OK(graph->AddEdge(0));
  LITERT_ASSERT_OK(graph->ConnectGraphInput(0, 0));
  LITERT_ASSERT_OK(graph->AddEdge(1));
  LITERT_ASSERT_OK(graph->ConnectGraphInput(1, 1));
  LITERT_ASSERT_OK(graph->AddEdge(2));
  LITERT_ASSERT_OK(graph->ConnectGraphOutput(0, 2));

  // Note: SouthBound does not expose an API to read back edge annotations,
  // but we verify that the annotation pipeline executes successfully
  // end-to-end.
  LITERT_ASSERT_OK(AnnotateCoherencyEdges(custom_context, "test_func",
                                          /*num_inputs=*/2, /*num_outputs=*/1,
                                          graph));

  LITERT_ASSERT_OK(LiteRtDispatchGraphDestroy(graph));
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextDestroy(custom_context));
}

using DispatchApiRunOptionsTest = testing::SimpleModelTest;

LiteRtStatus BuildLiteRtOptionsFromGoogleTensorOptions(
    LiteRtEnvironment env, GoogleTensorOptions& gt_options,
    internal::LiteRtOptionsPtr& out_options) {
  LITERT_ASSIGN_OR_RETURN(auto cc_options, Options::Create());
  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_RETURN_IF_ERROR(
      gt_options.GetOpaqueOptionsData(&identifier, &payload, &payload_deleter));
  LITERT_ASSIGN_OR_RETURN(
      auto opaque_options,
      OpaqueOptions::Create(identifier, payload, payload_deleter));
  LITERT_RETURN_IF_ERROR(
      cc_options.AddOpaqueOptions(std::move(opaque_options)));
  auto env_wrap = Environment::WrapCObject(env, OwnHandle::kNo);
  LITERT_ASSIGN_OR_RETURN(
      out_options,
      internal::LiteRtOptionsPtrBuilder::Build(cc_options,
                                               env_wrap.GetHolder()));
  return kLiteRtStatusOk;
}

TEST_F(DispatchApiRunOptionsTest,
       PerInvokePerformanceModeLifecycleAndValidation) {
  std::optional<LiteRtDispatchDeviceContextT::GoogleTensorOptionsData> parsed;
  EXPECT_EQ(ParseGoogleTensorOptions(LrtGetRuntimeContext(),
                                     /*options=*/nullptr, parsed),
            kLiteRtStatusOk);
  EXPECT_FALSE(parsed.has_value());

  LITERT_ASSERT_OK_AND_ASSIGN(auto compile_gt_opts,
                              GoogleTensorOptions::Create());
  compile_gt_opts.SetPerformanceMode(
      GoogleTensorOptions::PerformanceMode::kPowerSaver);
  internal::LiteRtOptionsPtr compile_c_opts;
  LITERT_ASSERT_OK(BuildLiteRtOptionsFromGoogleTensorOptions(
      env(), compile_gt_opts, compile_c_opts));

  LiteRtDispatchDeviceContext custom_context = nullptr;
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextCreate(
      LrtGetRuntimeContext(), compile_c_opts.get(), &custom_context));

  LiteRtDispatchInvocationContext icontext = nullptr;
  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextCreate(
      LrtGetRuntimeContext(), custom_context,
      kLiteRtDispatchExecutableTypeMlModel, &model_bytecode(),
      /*function_name=*/nullptr, /*num_inputs=*/2, /*num_outputs=*/1,
      &icontext));

  EXPECT_EQ(icontext->graph()->GetAppliedAnnotation("edgetpu_performance_mode"),
            "1");

  LITERT_ASSERT_OK_AND_ASSIGN(auto burst_gt_opts,
                              GoogleTensorOptions::Create());
  burst_gt_opts.SetPerformanceMode(
      GoogleTensorOptions::PerformanceMode::kBurst);
  internal::LiteRtOptionsPtr burst_c_opts;
  LITERT_ASSERT_OK(BuildLiteRtOptionsFromGoogleTensorOptions(
      env(), burst_gt_opts, burst_c_opts));
  LITERT_ASSERT_OK(
      LiteRtDispatchInvocationContextSetOptions(icontext, burst_c_opts.get()));
  EXPECT_EQ(icontext->graph()->GetAppliedAnnotation("edgetpu_performance_mode"),
            "5");

  LiteRtOptions empty_opts = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&empty_opts));
  LITERT_ASSERT_OK(
      LiteRtDispatchInvocationContextSetOptions(icontext, empty_opts));
  LiteRtDestroyOptions(empty_opts);
  EXPECT_EQ(icontext->graph()->GetAppliedAnnotation("edgetpu_performance_mode"),
            "1");

  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextDestroy(icontext));
  LITERT_ASSERT_OK(LiteRtDispatchDeviceContextDestroy(custom_context));

  // Default graph: reject invalid payloads, override, then reset to kBalanced.
  LiteRtDispatchInvocationContext default_icontext = nullptr;
  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextCreate(
      LrtGetRuntimeContext(), device_context(),
      kLiteRtDispatchExecutableTypeMlModel, &model_bytecode(),
      /*function_name=*/nullptr, /*num_inputs=*/2, /*num_outputs=*/1,
      &default_icontext));

  auto env_wrap = Environment::WrapCObject(env(), OwnHandle::kNo);
  for (std::string bad_toml :
       {"performance_mode = invalid_int", "performance_mode = 99"}) {
    SCOPED_TRACE(bad_toml);
    LITERT_ASSERT_OK_AND_ASSIGN(auto cc_options, Options::Create());
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto opaque_options,
        OpaqueOptions::Create(LrtGoogleTensorOptionsGetIdentifier(),
                              bad_toml.data(), [](void*) {}));
    LITERT_ASSERT_OK(cc_options.AddOpaqueOptions(std::move(opaque_options)));
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto bad_c_opts, internal::LiteRtOptionsPtrBuilder::Build(
                             cc_options, env_wrap.GetHolder()));
    EXPECT_EQ(LiteRtDispatchInvocationContextSetOptions(default_icontext,
                                                        bad_c_opts.get()),
              kLiteRtStatusErrorInvalidArgument);
    LiteRtDispatchDeviceContext rejected_context = nullptr;
    EXPECT_EQ(LiteRtDispatchDeviceContextCreate(
                  LrtGetRuntimeContext(), bad_c_opts.get(), &rejected_context),
              kLiteRtStatusErrorInvalidArgument);
    EXPECT_EQ(rejected_context, nullptr);
  }
  EXPECT_EQ(default_icontext->graph()->ApplyRunPerformanceMode(
                static_cast<LiteRtGoogleTensorOptionsPerformanceMode>(6)),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(default_icontext->graph()->GetAppliedAnnotation(
                "edgetpu_performance_mode"),
            std::nullopt);

  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextSetOptions(
      default_icontext, burst_c_opts.get()));
  EXPECT_EQ(default_icontext->graph()->GetAppliedAnnotation(
                "edgetpu_performance_mode"),
            "5");

  LITERT_ASSERT_OK(
      LiteRtDispatchInvocationContextSetOptions(default_icontext, nullptr));
  EXPECT_EQ(default_icontext->graph()->GetAppliedAnnotation(
                "edgetpu_performance_mode"),
            "2");

  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextDestroy(default_icontext));
}

TEST_F(DispatchApiRunOptionsTest,
       MultiInvocationContextSharedGraphAppliesContextSpecificRunOptions) {
  LiteRtDispatchInvocationContext icontext_burst = nullptr;
  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextCreate(
      LrtGetRuntimeContext(), device_context(),
      kLiteRtDispatchExecutableTypeMlModel, &model_bytecode(),
      /*function_name=*/nullptr, /*num_inputs=*/2, /*num_outputs=*/1,
      &icontext_burst));
  auto* graph = icontext_burst->graph();

  LiteRtDispatchInvocationContext icontext_saver = nullptr;
  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextCreateFromGraph(
      device_context(), graph, &icontext_saver));

  LITERT_ASSERT_OK_AND_ASSIGN(auto burst_gt, GoogleTensorOptions::Create());
  burst_gt.SetPerformanceMode(GoogleTensorOptions::PerformanceMode::kBurst);
  internal::LiteRtOptionsPtr burst_c_opts;
  LITERT_ASSERT_OK(BuildLiteRtOptionsFromGoogleTensorOptions(
      env(), burst_gt, burst_c_opts));

  LITERT_ASSERT_OK_AND_ASSIGN(auto saver_gt, GoogleTensorOptions::Create());
  saver_gt.SetPerformanceMode(
      GoogleTensorOptions::PerformanceMode::kPowerSaver);
  internal::LiteRtOptionsPtr saver_c_opts;
  LITERT_ASSERT_OK(
      BuildLiteRtOptionsFromGoogleTensorOptions(env(), saver_gt, saver_c_opts));

  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextSetOptions(
      icontext_burst, burst_c_opts.get()));
  EXPECT_EQ(graph->GetAppliedAnnotation("edgetpu_performance_mode"), "5");

  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextSetOptions(
      icontext_saver, saver_c_opts.get()));
  EXPECT_EQ(graph->GetAppliedAnnotation("edgetpu_performance_mode"), "1");

  (void)icontext_burst->Invoke();
  EXPECT_EQ(graph->GetAppliedAnnotation("edgetpu_performance_mode"), "5");

  (void)icontext_saver->Invoke();
  EXPECT_EQ(graph->GetAppliedAnnotation("edgetpu_performance_mode"), "1");

  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextDestroy(icontext_saver));
  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextDestroy(icontext_burst));
}

}  // namespace
}  // namespace litert::google_tensor
