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

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "platforms/darwinn/tachyon/core/fence/fence.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/darwinn/driver_shared/fence/fence_test_util.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_test_fixtures.h"
#include "thread/thread_manager.h"

namespace fence_util = ::platforms::darwinn::fence_util;
namespace tachyon = ::platforms::darwinn::tachyon;

using ::litert::google_tensor::testing::SimpleModelTest;
using ::testing::Combine;
using ::testing::Pointwise;
using ::testing::Values;

// Specifies how to create an invocation context.
enum class IContextCreateMode {
  kInterface,       // Use the standard interface.
  kGraphInterface,  // Use the graph interface.
};

template <typename Sink>
void AbslStringify(Sink& sink, IContextCreateMode mode) {
  switch (mode) {
    case IContextCreateMode::kInterface:
      sink.Append("kInterface");
      break;
    case IContextCreateMode::kGraphInterface:
      sink.Append("kGraphInterface");
      break;
  }
}

// Specifies how to invoke an invocation context.
enum class IContextInvokeMode {
  kSync,   // Invoke synchronously.
  kAsync,  // Invoke asynchronously.
};

template <typename Sink>
void AbslStringify(Sink& sink, IContextInvokeMode mode) {
  switch (mode) {
    case IContextInvokeMode::kSync:
      sink.Append("kSync");
      break;
    case IContextInvokeMode::kAsync:
      sink.Append("kAsync");
      break;
  }
}

// A helper type alias that comprises the parameters for testing the
// "simple model" end-to-end:
//
//  0. How to create the invocation context.
//  1. Whether to attach input events or not.
//  2. How to invoke the invocation context.
using SimpleModelEndToEndTestParams =
    std::tuple<IContextCreateMode, bool, IContextInvokeMode>;

std::string SimpleModelEndToEndTestParamsToStr(
    const ::testing::TestParamInfo<SimpleModelEndToEndTestParams>& info) {
  return absl::StrFormat(
      "iContextCreateMode_%v__withInputEvents_%d__iContextInvokeMode_%v",
      std::get<0>(info.param), std::get<1>(info.param),
      std::get<2>(info.param));
}

// A helper test fixture that parameterizes the `SimpleModelTest` fixture
// for end-to-end testing.
class SimpleModelEndToEndTest
    : public SimpleModelTest,
      public ::testing::WithParamInterface<SimpleModelEndToEndTestParams> {};

TEST_P(SimpleModelEndToEndTest, Succeeds) {
  IContextCreateMode icontext_create_mode = std::get<0>(GetParam());
  bool attach_input_events = std::get<1>(GetParam());
  IContextInvokeMode icontext_invoke_mode = std::get<2>(GetParam());

  // A helper struct to hold state that is exclusive to the Graph interface.
  struct GraphInterfaceExclusiveState {
    LiteRtDispatchExecutableHandle exec_handle;
    LiteRtDispatchGraph graph;
  };

  // A helper struct to hold state that is exclusive to attaching input events.
  struct AttachInputEventsExclusiveState {
    std::shared_ptr<tachyon::Fence> input_fence_0;
    LiteRtEvent input_event_0;
    std::shared_ptr<tachyon::Fence> input_fence_1;
    LiteRtEvent input_event_1;
  };

  int capabilities;
  LITERT_ASSERT_OK(LiteRtDispatchGetCapabilities(&capabilities));

  std::optional<GraphInterfaceExclusiveState> graph_interface_state;
  LiteRtDispatchInvocationContext invocation_context;

  switch (icontext_create_mode) {
    case IContextCreateMode::kInterface:
      LITERT_ASSERT_OK(LiteRtDispatchInvocationContextCreate(
          device_context(), kLiteRtDispatchExecutableTypeMlModel,
          &model_bytecode(), /*function_name=*/nullptr,
          /*num_inputs=*/2, /*num_outputs=*/1, &invocation_context));
      break;
    case IContextCreateMode::kGraphInterface:
      if ((capabilities & kLiteRtDispatchCapabilitiesGraph) == 0) {
        GTEST_SKIP() << "Graph API is not supported";
      }

      LiteRtDispatchExecutableHandle exec_handle;
      LITERT_ASSERT_OK(LiteRtDispatchLoadExecutable(
          device_context(), kLiteRtDispatchExecutableTypeMlModel,
          &model_bytecode(), &exec_handle));

      LiteRtDispatchGraph graph;
      LITERT_ASSERT_OK(LiteRtDispatchGraphCreate(device_context(), &graph));

      LITERT_ASSERT_OK(LiteRtDispatchAddNode(graph, /*node_id=*/0,
                                             kLiteRtDispatchNodeTypeNpu));
      LITERT_ASSERT_OK(LiteRtDispatchAddEdge(graph, /*edge_id=*/0));
      LITERT_ASSERT_OK(LiteRtDispatchAddEdge(graph, /*edge_id=*/1));
      LITERT_ASSERT_OK(LiteRtDispatchAddEdge(graph, /*edge_id=*/2));

      LITERT_ASSERT_OK(LiteRtDispatchAssignNodeFunction(
          graph, /*node_id=*/0, exec_handle, /*function_name=*/nullptr));

      LITERT_ASSERT_OK(LiteRtDispatchConnectNodeInput(
          graph, /*node_id=*/0, /*input_index=*/0, /*edge_id=*/0));
      LITERT_ASSERT_OK(LiteRtDispatchConnectNodeInput(
          graph, /*node_id=*/0, /*input_index=*/1, /*edge_id=*/1));
      LITERT_ASSERT_OK(LiteRtDispatchConnectNodeOutput(
          graph, /*node_id=*/0, /*output_index=*/0, /*edge_id=*/2));

      LITERT_ASSERT_OK(LiteRtDispatchConnectGraphInput(graph, /*input_index=*/0,
                                                       /*edge_id=*/0));
      LITERT_ASSERT_OK(LiteRtDispatchConnectGraphInput(graph, /*input_index=*/1,
                                                       /*edge_id=*/1));
      LITERT_ASSERT_OK(LiteRtDispatchConnectGraphOutput(
          graph, /*output_index=*/0, /*edge_id=*/2));

      LITERT_ASSERT_OK(LiteRtDispatchInvocationContextCreateFromGraph(
          device_context(), graph, &invocation_context));

      graph_interface_state.emplace(exec_handle, graph);
      break;
  }

  LiteRtTensorBufferRequirements input_0_requirements;
  LITERT_ASSERT_OK(LiteRtDispatchGetInputRequirements(
      invocation_context, /*input_index=*/0, &kInput0TensorType,
      &input_0_requirements));

  LiteRtTensorBufferType input_0_type;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
      input_0_requirements, /*type_index=*/0, &input_0_type));

  size_t input_0_size;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsBufferSize(
      input_0_requirements, &input_0_size));
  ASSERT_GE(input_0_size, sizeof(kTestInput0Tensor));

  LiteRtTensorBufferRequirements input_1_requirements;
  LITERT_ASSERT_OK(LiteRtDispatchGetInputRequirements(
      invocation_context, /*input_index=*/1, &kInput1TensorType,
      &input_1_requirements));

  LiteRtTensorBufferType input_1_type;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
      input_1_requirements, /*type_index=*/0, &input_1_type));

  size_t input_1_size;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsBufferSize(
      input_1_requirements, &input_1_size));
  ASSERT_GE(input_1_size, sizeof(kTestInput1Tensor));

  LiteRtTensorBufferRequirements output_requirements;
  LITERT_ASSERT_OK(LiteRtDispatchGetOutputRequirements(
      invocation_context, /*output_index=*/0, &kOutputTensorType,
      &output_requirements));

  LiteRtTensorBufferType output_type;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
      output_requirements, /*type_index=*/0, &output_type));

  size_t output_size;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsBufferSize(
      output_requirements, &output_size));
  ASSERT_GE(output_size, sizeof(kTestOutputTensor));

  LiteRtDestroyTensorBufferRequirements(input_0_requirements);
  LiteRtDestroyTensorBufferRequirements(input_1_requirements);
  LiteRtDestroyTensorBufferRequirements(output_requirements);

  LiteRtTensorBuffer input_0_tensor_buffer;
  LITERT_ASSERT_OK(
      LiteRtCreateManagedTensorBuffer(env(), input_0_type, &kInput0TensorType,
                                      input_0_size, &input_0_tensor_buffer));

  LiteRtTensorBuffer input_1_tensor_buffer;
  LITERT_ASSERT_OK(
      LiteRtCreateManagedTensorBuffer(env(), input_1_type, &kInput1TensorType,
                                      input_1_size, &input_1_tensor_buffer));

  LiteRtTensorBuffer output_tensor_buffer;
  LITERT_ASSERT_OK(
      LiteRtCreateManagedTensorBuffer(env(), output_type, &kOutputTensorType,
                                      output_size, &output_tensor_buffer));

  LiteRtTensorBufferHandle input_1_handle;
  LITERT_ASSERT_OK(LiteRtDispatchRegisterTensorBuffer(
      device_context(), input_1_tensor_buffer, &input_1_handle));

  LiteRtTensorBufferHandle input_0_handle;
  LITERT_ASSERT_OK(LiteRtDispatchRegisterTensorBuffer(
      device_context(), input_0_tensor_buffer, &input_0_handle));

  LiteRtTensorBufferHandle output_handle;
  LITERT_ASSERT_OK(LiteRtDispatchRegisterTensorBuffer(
      device_context(), output_tensor_buffer, &output_handle));

  LITERT_ASSERT_OK(LiteRtDispatchAttachInput(invocation_context,
                                             /*graph_input_index=*/0,
                                             input_0_handle));
  LITERT_ASSERT_OK(LiteRtDispatchAttachInput(invocation_context,
                                             /*graph_input_index=*/1,
                                             input_1_handle));
  LITERT_ASSERT_OK(LiteRtDispatchAttachOutput(invocation_context,
                                              /*graph_output_index=*/0,
                                              output_handle));

  {
    void* host_mem_addr;

    LITERT_ASSERT_OK(LiteRtLockTensorBuffer(input_0_tensor_buffer,
                                            &host_mem_addr,
                                            kLiteRtTensorBufferLockModeWrite));
    std::memcpy(host_mem_addr, kTestInput0Tensor, sizeof(kTestInput0Tensor));
    LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(input_0_tensor_buffer));

    LITERT_ASSERT_OK(LiteRtLockTensorBuffer(input_1_tensor_buffer,
                                            &host_mem_addr,
                                            kLiteRtTensorBufferLockModeWrite));
    std::memcpy(host_mem_addr, kTestInput1Tensor, sizeof(kTestInput1Tensor));
    LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(input_1_tensor_buffer));
  }

  std::optional<AttachInputEventsExclusiveState> attach_input_events_state;
  if (attach_input_events) {
    if ((capabilities & kLiteRtDispatchCapabilitiesAsync) == 0) {
      GTEST_SKIP() << "Async API is not supported";
    }

    std::shared_ptr<tachyon::Fence> input_fence_0 = fence_util::CreateFence();
    LiteRtEvent input_event_0;
    LITERT_ASSERT_OK(
        LiteRtCreateEventFromSyncFenceFd(env(), input_fence_0->GetFd(),
                                         /*owns_fd=*/false, &input_event_0));

    std::shared_ptr<tachyon::Fence> input_fence_1 = fence_util::CreateFence();
    LiteRtEvent input_event_1;
    LITERT_ASSERT_OK(
        LiteRtCreateEventFromSyncFenceFd(env(), input_fence_1->GetFd(),
                                         /*owns_fd=*/false, &input_event_1));

    LITERT_ASSERT_OK(LiteRtDispatchAttachInputEvent(
        invocation_context, /*graph_input_index=*/0, input_event_0));
    LITERT_ASSERT_OK(LiteRtDispatchAttachInputEvent(
        invocation_context, /*graph_input_index=*/1, input_event_1));

    thread::DefaultQueue()->Schedule([=] {
      absl::SleepFor(absl::Milliseconds(100));
      ASSERT_OK(input_fence_1->Signal(/*success=*/true));
    });
    thread::DefaultQueue()->Schedule([=] {
      absl::SleepFor(absl::Milliseconds(200));
      ASSERT_OK(input_fence_0->Signal(/*success=*/true));
    });

    attach_input_events_state.emplace(std::move(input_fence_0), input_event_0,
                                      std::move(input_fence_1), input_event_1);
  }

  switch (icontext_invoke_mode) {
    case IContextInvokeMode::kSync:
      LITERT_ASSERT_OK(LiteRtDispatchInvoke(invocation_context));
      break;
    case IContextInvokeMode::kAsync:
      if ((capabilities & kLiteRtDispatchCapabilitiesAsync) == 0) {
        GTEST_SKIP() << "Async API is not supported";
      }

      LiteRtEvent output_event;
      LITERT_ASSERT_OK(LiteRtDispatchInvokeAsync(invocation_context,
                                                 /*num_output_events=*/1,
                                                 &output_event));

      LITERT_ASSERT_OK(
          LiteRtSetTensorBufferEvent(output_tensor_buffer, output_event));
      break;
  }

  {
    void* host_mem_addr;
    LITERT_ASSERT_OK(LiteRtLockTensorBuffer(
        output_tensor_buffer, &host_mem_addr, kLiteRtTensorBufferLockModeRead));

    auto output = absl::MakeConstSpan(static_cast<float*>(host_mem_addr),
                                      kTestOutputSize);
    for (size_t i = 0; i < kTestOutputSize; ++i) {
      LITERT_LOG(LITERT_INFO, "%f\t%f", output[i], kTestOutputTensor[i]);
    }
    EXPECT_THAT(output, Pointwise(testing::FloatNear(1e-3), kTestOutputTensor));

    LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(output_tensor_buffer));
  }

  if (attach_input_events_state.has_value()) {
    LiteRtDestroyEvent(attach_input_events_state->input_event_0);
    LiteRtDestroyEvent(attach_input_events_state->input_event_1);
  }

  LITERT_ASSERT_OK(LiteRtDispatchDetachInput(
      invocation_context, /*graph_input_index=*/0, input_0_handle));
  LITERT_ASSERT_OK(LiteRtDispatchDetachInput(
      invocation_context, /*graph_input_index=*/1, input_1_handle));
  LITERT_ASSERT_OK(LiteRtDispatchDetachOutput(
      invocation_context, /*graph_output_index=*/0, output_handle));

  LITERT_ASSERT_OK(
      LiteRtDispatchUnregisterTensorBuffer(device_context(), output_handle));
  LITERT_ASSERT_OK(
      LiteRtDispatchUnregisterTensorBuffer(device_context(), input_1_handle));
  LITERT_ASSERT_OK(
      LiteRtDispatchUnregisterTensorBuffer(device_context(), input_0_handle));

  LiteRtDestroyTensorBuffer(output_tensor_buffer);
  LiteRtDestroyTensorBuffer(input_1_tensor_buffer);
  LiteRtDestroyTensorBuffer(input_0_tensor_buffer);

  LITERT_ASSERT_OK(LiteRtDispatchInvocationContextDestroy(invocation_context));

  if (graph_interface_state.has_value()) {
    LITERT_ASSERT_OK(LiteRtDispatchGraphDestroy(graph_interface_state->graph));

    LITERT_ASSERT_OK(LiteRtDispatchUnloadExecutable(
        device_context(), graph_interface_state->exec_handle));
  }
}

INSTANTIATE_TEST_SUITE_P(AllInterfaces, SimpleModelEndToEndTest,
                         Combine(Values(IContextCreateMode::kInterface,
                                        IContextCreateMode::kGraphInterface),
                                 /*attach_input_events=*/testing::Bool(),
                                 Values(IContextInvokeMode::kSync,
                                        IContextInvokeMode::kAsync)),
                         SimpleModelEndToEndTestParamsToStr);
