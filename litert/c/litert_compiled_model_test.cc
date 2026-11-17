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

#include "litert/c/litert_compiled_model.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/barrier.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_resolver.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

using testing::ElementsAre;
using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

TEST(CompiledModelTest, Basic) {
  auto path = testing::GetTestFilePath(kModelFileName);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment));

  LiteRtModel model;
  LITERT_ASSERT_OK(
      LiteRtCreateModelFromFile(environment, path.c_str(), &model));

  LiteRtOptions jit_compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&jit_compilation_options));
  LITERT_ASSERT_OK(LiteRtSetOptionsHardwareAccelerators(
      jit_compilation_options, kLiteRtHwAcceleratorCpu));

  LiteRtCompiledModel compiled_model;
  LITERT_ASSERT_OK(LiteRtCreateCompiledModel(
      environment, model, jit_compilation_options, &compiled_model));

  LiteRtDestroyOptions(jit_compilation_options);

  LiteRtDelegationMetricsConst delegation_metrics = nullptr;
  EXPECT_EQ(
      LiteRtCompiledModelGetDelegationMetrics(nullptr, &delegation_metrics),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtCompiledModelGetDelegationMetrics(compiled_model, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  LITERT_ASSERT_OK(LiteRtCompiledModelGetDelegationMetrics(
      compiled_model, &delegation_metrics));
  ASSERT_NE(delegation_metrics, nullptr);

  int total_node_count = 0;
  EXPECT_EQ(
      LiteRtGetDelegationMetricsTotalNodeCount(nullptr, &total_node_count),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LiteRtGetDelegationMetricsTotalNodeCount(delegation_metrics, nullptr),
      kLiteRtStatusErrorInvalidArgument);
  LITERT_ASSERT_OK(LiteRtGetDelegationMetricsTotalNodeCount(delegation_metrics,
                                                            &total_node_count));
  EXPECT_GT(total_node_count, 0);

  LiteRtParamIndex num_accelerators = 0;
  EXPECT_EQ(
      LiteRtGetNumDelegationMetricsAccelerators(nullptr, &num_accelerators),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LiteRtGetNumDelegationMetricsAccelerators(delegation_metrics, nullptr),
      kLiteRtStatusErrorInvalidArgument);
  LITERT_ASSERT_OK(LiteRtGetNumDelegationMetricsAccelerators(
      delegation_metrics, &num_accelerators));
  ASSERT_GT(num_accelerators, 0);

  LiteRtAcceleratorDelegationMetricsConst oob_acc = nullptr;
  EXPECT_EQ(LiteRtGetDelegationMetricsAccelerator(nullptr, 0, &oob_acc),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LiteRtGetDelegationMetricsAccelerator(delegation_metrics, 0, nullptr),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtGetDelegationMetricsAccelerator(delegation_metrics,
                                                  num_accelerators, &oob_acc),
            kLiteRtStatusErrorIndexOOB);

  int cpu_delegated_nodes = 0;
  int cpu_partitions = 0;
  for (LiteRtParamIndex i = 0; i < num_accelerators; ++i) {
    LiteRtAcceleratorDelegationMetricsConst acc = nullptr;
    LITERT_ASSERT_OK(
        LiteRtGetDelegationMetricsAccelerator(delegation_metrics, i, &acc));
    ASSERT_NE(acc, nullptr);

    const char* accelerator_name = nullptr;
    EXPECT_EQ(
        LiteRtGetAcceleratorDelegationMetricsName(nullptr, &accelerator_name),
        kLiteRtStatusErrorInvalidArgument);
    EXPECT_EQ(LiteRtGetAcceleratorDelegationMetricsName(acc, nullptr),
              kLiteRtStatusErrorInvalidArgument);
    LITERT_ASSERT_OK(
        LiteRtGetAcceleratorDelegationMetricsName(acc, &accelerator_name));
    EXPECT_NE(accelerator_name, nullptr);

    LiteRtHwAcceleratorSet hardware_type = kLiteRtHwAcceleratorNone;
    EXPECT_EQ(LiteRtGetAcceleratorDelegationMetricsHardwareType(nullptr,
                                                                &hardware_type),
              kLiteRtStatusErrorInvalidArgument);
    EXPECT_EQ(LiteRtGetAcceleratorDelegationMetricsHardwareType(acc, nullptr),
              kLiteRtStatusErrorInvalidArgument);
    LITERT_ASSERT_OK(
        LiteRtGetAcceleratorDelegationMetricsHardwareType(acc, &hardware_type));

    int delegated_node_count = 0;
    EXPECT_EQ(LiteRtGetAcceleratorDelegationMetricsDelegatedNodeCount(
                  nullptr, &delegated_node_count),
              kLiteRtStatusErrorInvalidArgument);
    EXPECT_EQ(
        LiteRtGetAcceleratorDelegationMetricsDelegatedNodeCount(acc, nullptr),
        kLiteRtStatusErrorInvalidArgument);
    LITERT_ASSERT_OK(LiteRtGetAcceleratorDelegationMetricsDelegatedNodeCount(
        acc, &delegated_node_count));

    int partition_count = 0;
    EXPECT_EQ(LiteRtGetAcceleratorDelegationMetricsPartitionCount(
                  nullptr, &partition_count),
              kLiteRtStatusErrorInvalidArgument);
    EXPECT_EQ(LiteRtGetAcceleratorDelegationMetricsPartitionCount(acc, nullptr),
              kLiteRtStatusErrorInvalidArgument);
    LITERT_ASSERT_OK(LiteRtGetAcceleratorDelegationMetricsPartitionCount(
        acc, &partition_count));

    if (hardware_type & kLiteRtHwAcceleratorCpu) {
      cpu_delegated_nodes += delegated_node_count;
      cpu_partitions += partition_count;
    } else {
      EXPECT_EQ(delegated_node_count, 0);
      EXPECT_EQ(partition_count, 0);
    }
  }
  EXPECT_GT(cpu_delegated_nodes, 0);
  EXPECT_GT(cpu_partitions, 0);

  LiteRtSubgraph subgraph;
  LITERT_ASSERT_OK(LiteRtGetModelSubgraph(model, 0, &subgraph));

  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphInputs(subgraph, &num_inputs));

  std::vector<LiteRtTensorBuffer> input_tensor_buffers;
  input_tensor_buffers.reserve(num_inputs);
  for (auto i = 0; i < num_inputs; ++i) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    LITERT_ASSERT_OK(LiteRtGetCompiledModelInputBufferRequirements(
        compiled_model, /*signature_index=*/0, i, &tensor_buffer_requirements));
    LiteRtTensorBufferType tensor_buffer_type;
    LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
        tensor_buffer_requirements, /*type_index=*/0, &tensor_buffer_type));
    size_t tensor_buffer_size;
    LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsBufferSize(
        tensor_buffer_requirements, &tensor_buffer_size));
    LiteRtTensorBuffer tensor_buffer;
    LITERT_ASSERT_OK(LiteRtCreateManagedTensorBuffer(
        environment, tensor_buffer_type, &kInput0TensorType, tensor_buffer_size,
        &tensor_buffer));
    input_tensor_buffers.push_back(tensor_buffer);
  }

  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphOutputs(subgraph, &num_outputs));

  std::vector<LiteRtTensorBuffer> output_tensor_buffers;
  output_tensor_buffers.reserve(num_outputs);
  for (auto i = 0; i < num_outputs; ++i) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    LITERT_ASSERT_OK(LiteRtGetCompiledModelOutputBufferRequirements(
        compiled_model, /*signature_index=*/0, i, &tensor_buffer_requirements));
    LiteRtTensorBufferType tensor_buffer_type;
    LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
        tensor_buffer_requirements, /*type_index=*/0, &tensor_buffer_type));
    size_t tensor_buffer_size;
    LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsBufferSize(
        tensor_buffer_requirements, &tensor_buffer_size));
    LiteRtTensorBuffer tensor_buffer;
    LITERT_ASSERT_OK(LiteRtCreateManagedTensorBuffer(
        environment, tensor_buffer_type, &kInput0TensorType, tensor_buffer_size,
        &tensor_buffer));
    output_tensor_buffers.push_back(tensor_buffer);
  }

  {
    ABSL_LOG(INFO) << "Filling inputs with data";
    void* host_mem_addr;

    LITERT_ASSERT_OK(LiteRtLockTensorBuffer(input_tensor_buffers[0],
                                            &host_mem_addr,
                                            kLiteRtTensorBufferLockModeWrite));
    std::memcpy(host_mem_addr, kTestInput0Tensor, sizeof(kTestInput0Tensor));
    LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(input_tensor_buffers[0]));

    LITERT_ASSERT_OK(LiteRtLockTensorBuffer(input_tensor_buffers[1],
                                            &host_mem_addr,
                                            kLiteRtTensorBufferLockModeWrite));
    std::memcpy(host_mem_addr, kTestInput1Tensor, sizeof(kTestInput1Tensor));
    LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(input_tensor_buffers[1]));
  }

  int64_t duration = 0;
  LiteRtCompiledModelConst const_compiled_model = compiled_model;
  LITERT_ASSERT_OK(LiteRtGetCompiledModelLastInferenceDuration(
      const_compiled_model, &duration));
  EXPECT_EQ(duration, -1);

  LITERT_ASSERT_OK(LiteRtRunCompiledModel(
      compiled_model, /*signature_index=*/0, input_tensor_buffers.size(),
      input_tensor_buffers.data(), output_tensor_buffers.size(),
      output_tensor_buffers.data()));

  LITERT_ASSERT_OK(LiteRtGetCompiledModelLastInferenceDuration(
      const_compiled_model, &duration));
  EXPECT_GT(duration, 0);

  {
    ABSL_LOG(INFO) << "Checking output...";
    void* host_mem_addr;
    LITERT_ASSERT_OK(LiteRtLockTensorBuffer(output_tensor_buffers[0],
                                            &host_mem_addr,
                                            kLiteRtTensorBufferLockModeRead));
    auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                 kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-3), kTestOutputTensor));
    LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(output_tensor_buffers[0]));
  }

  // Verify that a failed run resets the duration to -1.
  EXPECT_NE(LiteRtRunCompiledModel(
                compiled_model, /*signature_index=*/0, /*num_input_buffers=*/0,
                /*input_buffers=*/nullptr, output_tensor_buffers.size(),
                output_tensor_buffers.data()),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK(LiteRtGetCompiledModelLastInferenceDuration(
      const_compiled_model, &duration));
  EXPECT_EQ(duration, -1);

  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);

  for (auto tensor_buffer : input_tensor_buffers) {
    LiteRtDestroyTensorBuffer(tensor_buffer);
  }
  for (auto tensor_buffer : output_tensor_buffers) {
    LiteRtDestroyTensorBuffer(tensor_buffer);
  }
}

TEST(CompiledModelTest, ResizeInputTensorWithDynamicModel) {
  // Use the dynamic model for testing resize functionality
  auto path = testing::GetTestFilePath(kDynamicModelFileName);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment));

  LiteRtModel model;
  LITERT_ASSERT_OK(
      LiteRtCreateModelFromFile(environment, path.c_str(), &model));

  LiteRtOptions jit_compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&jit_compilation_options));
  LITERT_ASSERT_OK(LiteRtSetOptionsHardwareAccelerators(
      jit_compilation_options, kLiteRtHwAcceleratorCpu));

  LiteRtCompiledModel compiled_model;
  LITERT_ASSERT_OK(LiteRtCreateCompiledModel(
      environment, model, jit_compilation_options, &compiled_model));

  LiteRtDestroyOptions(jit_compilation_options);

  // (?, 2, 3) => (1, 2, 3)
  const int new_dims[] = {1, 2, 3};
  LITERT_ASSERT_OK(LiteRtCompiledModelResizeInputTensor(
      compiled_model,
      /*signature_index=*/0,
      /*input_index=*/0, new_dims, /*dims_size=*/3));

  // Get new buffer requirements after resize
  LiteRtTensorBufferRequirements requirements;
  LITERT_ASSERT_OK(LiteRtGetCompiledModelInputBufferRequirements(
      compiled_model, /*signature_index=*/0, /*input_index=*/0, &requirements));

  size_t resized_input_tensor_size;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsBufferSize(
      requirements, &resized_input_tensor_size));

  // Verify that the size has doubled (batch size doubled)
  LITERT_LOG(LITERT_INFO, "New size: %zu", resized_input_tensor_size);
  EXPECT_EQ(resized_input_tensor_size, 1 * 2 * 3 * sizeof(float));

  // Test error cases
  // Invalid signature index
  EXPECT_NE(
      LiteRtCompiledModelResizeInputTensor(compiled_model,
                                           /*signature_index=*/999,
                                           /*input_index=*/0, new_dims, 3),
      kLiteRtStatusOk);

  // Invalid input index
  EXPECT_NE(
      LiteRtCompiledModelResizeInputTensor(compiled_model,
                                           /*signature_index=*/0,
                                           /*input_index=*/999, new_dims, 3),
      kLiteRtStatusOk);

  // Empty dims
  EXPECT_NE(LiteRtCompiledModelResizeInputTensor(
                compiled_model, /*signature_index=*/0, /*input_index=*/0,
                new_dims, 0),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtCompiledModelResizeInputTensor(
                compiled_model, /*signature_index=*/0, /*input_index=*/0,
                /*dims=*/nullptr, /*dims_size=*/3),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtCompiledModelResizeInputTensorNonStrict(
                compiled_model, /*signature_index=*/0, /*input_index=*/0,
                /*dims=*/nullptr, /*dims_size=*/3),
            kLiteRtStatusErrorInvalidArgument);

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

TEST(CompiledModelTest, ResizeInputTensorWithStaticModel) {
  // Use the simple model to ensure resize will error out.
  auto path = testing::GetTestFilePath(kModelFileName);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment));

  LiteRtModel model;
  LITERT_ASSERT_OK(
      LiteRtCreateModelFromFile(environment, path.c_str(), &model));

  LiteRtOptions jit_compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&jit_compilation_options));
  LITERT_ASSERT_OK(LiteRtSetOptionsHardwareAccelerators(
      jit_compilation_options, kLiteRtHwAcceleratorCpu));

  LiteRtCompiledModel compiled_model;
  LITERT_ASSERT_OK(LiteRtCreateCompiledModel(
      environment, model, jit_compilation_options, &compiled_model));

  LiteRtDestroyOptions(jit_compilation_options);

  // (?, 2, 3) => (1, 2, 3)
  const int new_dims[] = {1, 2, 3};
  ASSERT_NE(
      LiteRtCompiledModelResizeInputTensor(compiled_model,
                                           /*signature_index=*/0,
                                           /*input_index=*/0, new_dims, 3),
      kLiteRtStatusOk);
  EXPECT_EQ(LiteRtCompiledModelResizeInputTensorNonStrict(
                compiled_model, /*signature_index=*/0,
                /*input_index=*/0, new_dims, 3),
            kLiteRtStatusOk);

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

TEST(CompiledModelTest, GetOutputTensorLayoutsWithDynamicModel) {
  // Use the dynamic model for testing resize functionality
  auto path = testing::GetTestFilePath(kDynamicModelFileName);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment));

  LiteRtModel model;
  LITERT_ASSERT_OK(
      LiteRtCreateModelFromFile(environment, path.c_str(), &model));

  LiteRtOptions jit_compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&jit_compilation_options));
  LITERT_ASSERT_OK(LiteRtSetOptionsHardwareAccelerators(
      jit_compilation_options, kLiteRtHwAcceleratorCpu));

  LiteRtCompiledModel compiled_model;
  LITERT_ASSERT_OK(LiteRtCreateCompiledModel(
      environment, model, jit_compilation_options, &compiled_model));

  LiteRtDestroyOptions(jit_compilation_options);

  // Check basic subgraph information
  LiteRtSubgraph litert_subgraph;
  LITERT_ASSERT_OK(LiteRtGetModelSubgraph(model, 0, &litert_subgraph));
  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphInputs(litert_subgraph, &num_inputs));
  ASSERT_EQ(num_inputs, 2);
  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphOutputs(litert_subgraph, &num_outputs));
  ASSERT_EQ(num_outputs, 1);

  std::vector<LiteRtTensor> input_tensor = litert_subgraph->Inputs();
  for (int i = 0; i < (size_t)num_inputs; ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, input_tensor[i]->Ranked());
    // Check if the original input tensor is (?, 2, 3)
    EXPECT_THAT(absl::MakeConstSpan(tensor_type.layout.dimensions,
                                    tensor_type.layout.rank),
                ElementsAre(-1, 2, 3));
  }

  // (?, 2, 3) => (1, 2, 3)
  const int new_dims[] = {1, 2, 3};
  for (int i = 0; i < (size_t)num_inputs; ++i) {
    LITERT_ASSERT_OK(LiteRtCompiledModelResizeInputTensor(compiled_model,
                                                          /*signature_index=*/0,
                                                          /*input_index=*/i,
                                                          new_dims, 3));
  }

  // (?, 2, 3) => (1, 2, 3)
  int expected_output_tensor_shapes[] = {1, 2, 3};
  std::vector<LiteRtLayout> output_tensor_layouts(num_outputs);
  LITERT_ASSERT_OK(LiteRtGetCompiledModelOutputTensorLayouts(
      compiled_model, /*signature_index=*/0, num_outputs,
      output_tensor_layouts.data(),
      /*update_allocation=*/true));

  // Verify that the size has doubled (batch size doubled)
  // Output tensor number is 1
  LiteRtLayout layout = output_tensor_layouts[0];
  ASSERT_EQ(layout.rank, sizeof(expected_output_tensor_shapes) / sizeof(int));
  // litert check output_tensor_shapes equals expected_output_tensor_shapes
  for (int i = 0; i < layout.rank; ++i) {
    EXPECT_EQ(layout.dimensions[i], expected_output_tensor_shapes[i]);
  }

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

struct ScopedCpuModelTestInputs {
  LiteRtEnvironment environment = nullptr;
  LiteRtModel model = nullptr;
  LiteRtOptions options = nullptr;

  static void Create(const std::string& path, ScopedCpuModelTestInputs* out) {
    LITERT_ASSERT_OK(LiteRtCreateEnvironment(
        /*num_options=*/0, /*options=*/nullptr, &out->environment));
    LITERT_ASSERT_OK(
        LiteRtCreateModelFromFile(out->environment, path.c_str(), &out->model));
    LITERT_ASSERT_OK(LiteRtCreateOptions(&out->options));
    LITERT_ASSERT_OK(LiteRtSetOptionsHardwareAccelerators(
        out->options, kLiteRtHwAcceleratorCpu));
  }

  ~ScopedCpuModelTestInputs() {
    LiteRtDestroyOptions(options);
    LiteRtDestroyModel(model);
    LiteRtDestroyEnvironment(environment);
  }
};

TEST(CompiledModelTest, ConcurrentCpuModelCreation) {
  constexpr int kNumThreads = 4;
  const std::string path = testing::GetTestFilePath(kModelFileName);
  absl::Barrier start_barrier(kNumThreads);

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&]() {
      ScopedCpuModelTestInputs inputs;
      ScopedCpuModelTestInputs::Create(path, &inputs);
      start_barrier.Block();

      LiteRtCompiledModel compiled_model = nullptr;
      LITERT_ASSERT_OK(LiteRtCreateCompiledModel(
          inputs.environment, inputs.model, inputs.options, &compiled_model));
      bool fully_accelerated = false;
      LITERT_ASSERT_OK(LiteRtCompiledModelIsFullyAccelerated(
          compiled_model, &fully_accelerated));
      EXPECT_TRUE(fully_accelerated);
      LiteRtDestroyCompiledModel(compiled_model);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
}

TEST(CompiledModelTest, CpuModelCreationExecutesInParallelWithoutGlobalLock) {
  struct Rendezvous {
    absl::Mutex mutex;
    int entered = 0;
    bool timed_out = false;
  } rendezvous;

  using FindBuiltinOpResult =
      std::invoke_result_t<LiteRtFindBuiltinOpFunc, void*, int, int>;
  LiteRtFindBuiltinOpFunc find_builtin_op =
      [](void* user_data, int, int) -> FindBuiltinOpResult {
    thread_local bool arrived = false;
    if (!arrived) {
      arrived = true;
      auto* state = static_cast<Rendezvous*>(user_data);
      absl::MutexLock lock(state->mutex);
      ++state->entered;
      auto both_entered = [state]() { return state->entered >= 2; };
      if (!state->mutex.AwaitWithTimeout(absl::Condition(&both_entered),
                                         absl::Seconds(5))) {
        state->timed_out = true;
      }
    }
    return nullptr;
  };

  const std::string path = testing::GetTestFilePath(kModelFileName);
  std::vector<std::thread> threads;
  threads.reserve(2);
  for (int i = 0; i < 2; ++i) {
    threads.emplace_back([&]() {
      ScopedCpuModelTestInputs inputs;
      ScopedCpuModelTestInputs::Create(path, &inputs);
      LITERT_ASSERT_OK(LiteRtSetOpResolverOption(
          inputs.options, find_builtin_op, /*find_custom_op=*/nullptr,
          &rendezvous));

      LiteRtCompiledModel compiled_model = nullptr;
      EXPECT_NE(LiteRtCreateCompiledModel(inputs.environment, inputs.model,
                                          inputs.options, &compiled_model),
                kLiteRtStatusOk);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  absl::MutexLock lock(rendezvous.mutex);
  EXPECT_EQ(rendezvous.entered, 2);
  EXPECT_FALSE(rendezvous.timed_out);
}

}  // namespace
}  // namespace litert
