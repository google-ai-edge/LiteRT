/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/runners/ml_drift/ml_drift_cl_model_runner.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

using TensorMLD = Tensor<MlDriftMixinTag>;

struct TestInputs {
  TensorMLD input_tensor;
  TensorMLD scalar_weight;

  absl::flat_hash_map<std::string, TensorMLD*> tensors() {
    return {{"input_tensor", &input_tensor}, {"scalar_weight", &scalar_weight}};
  }
};

struct TestOutputs {
  TensorMLD output_tensor;

  absl::flat_hash_map<std::string, TensorMLD*> tensors() {
    return {{"output_tensor", &output_tensor}};
  }
};

TestOutputs TestModel(TestInputs& inputs) {
  TestOutputs outputs;
  outputs.output_tensor = Mul(inputs.input_tensor, inputs.scalar_weight);
  return outputs;
}

TEST(MlDriftModelRunnerTest, MulWithScalarBroadcastsCorrectly) {
  TestInputs inputs = {
      .input_tensor = TensorMLD({.type = Type::kFP32, .shape = {1, 1, 8}}),
      .scalar_weight = TensorMLD({.type = Type::kFP32, .shape = {}}),
  };
  TestOutputs outputs;

  absl::flat_hash_map<std::string, absl::Span<const float>> float_weights;
  absl::flat_hash_map<std::string, absl::Span<const int8_t>> int8_weights;

  MlDriftModelRunner<decltype(&TestModel), decltype(inputs), decltype(outputs)>
      runner(&TestModel, inputs, outputs, float_weights, int8_weights,
             /*build_gpu_model=*/true, ml_drift::CalculationsPrecision::kF32);

  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                         5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_OK(runner.SetInput("input_tensor", input_data));
  ASSERT_OK(runner.SetInput("scalar_weight", std::vector<float>{0.5f}));

  ASSERT_OK(runner.Run());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto result,
                                  runner.GetFloatOutput("output_tensor"));

  const std::vector<float> expected_output = {0.5f, 1.0f, 1.5f, 2.0f,
                                              2.5f, 3.0f, 3.5f, 4.0f};
  EXPECT_THAT(result, Pointwise(FloatNear(1e-6), expected_output));
}

struct FcTestInputs {
  TensorMLD input_tensor;
  TensorMLD w_fc_weight_1;

  absl::flat_hash_map<std::string, TensorMLD*> tensors() {
    return {{"input_tensor", &input_tensor}, {"w_fc_weight_1", &w_fc_weight_1}};
  }
};

struct FcTestOutputs {
  TensorMLD output_tensor;

  absl::flat_hash_map<std::string, TensorMLD*> tensors() {
    return {{"output_tensor", &output_tensor}};
  }
};

FcTestOutputs FullyConnectedModel(FcTestInputs& inputs) {
  FcTestOutputs outputs;
  outputs.output_tensor =
      FullyConnected(inputs.input_tensor, inputs.w_fc_weight_1);
  return outputs;
}

TEST(MlDriftModelRunnerTest, FullyConnectedInt4) {
  const std::vector<float> scales = {0.1f, 0.2f, 0.3f, 0.4f};
  const std::vector<int64_t> zero_points = {0, 0, 0, 0};
  FcTestInputs inputs = {
      .input_tensor = TensorMLD({.type = Type::kFP32, .shape = {1, 8}}),
      .w_fc_weight_1 = TensorMLD(
          {.type = Type::kI4,
           .shape = {4, 8},
           .quantization = std::make_shared<PerChannelAffineQuantization>(
               PerChannelAffineQuantization(scales, zero_points))}),
  };
  FcTestOutputs outputs;

  absl::flat_hash_map<std::string, absl::Span<const float>> float_weights;
  absl::flat_hash_map<std::string, absl::Span<const int8_t>> int8_weights;

  // Pack int4 weights into int8.
  std::vector<int8_t> packed_weights;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; j += 2) {
      int8_t w0 = i + 1;
      int8_t w1 = i + 1;
      packed_weights.push_back((w1 << 4) | (w0 & 0x0F));
    }
  }
  int8_weights["w_fc_weight_1"] = packed_weights;

  MlDriftModelRunner<decltype(&FullyConnectedModel), decltype(inputs),
                     decltype(outputs)>
      runner(&FullyConnectedModel, inputs, outputs, float_weights, int8_weights,
             /*build_gpu_model=*/true, ml_drift::CalculationsPrecision::kF32);

  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                         5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_OK(runner.SetInput("input_tensor", input_data));

  ASSERT_OK(runner.Run());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto result,
                                  runner.GetFloatOutput("output_tensor"));

  const float input_sum = 36.0f;
  const std::vector<float> expected_output = {
      input_sum * 1.0f * scales[0], input_sum * 2.0f * scales[1],
      input_sum * 3.0f * scales[2], input_sum * 4.0f * scales[3]};
  EXPECT_THAT(result, Pointwise(FloatNear(1e-5), expected_output));
}

struct LstmTestInputs {
  TensorMLD intermediate;
  TensorMLD prev_state;

  absl::flat_hash_map<std::string, TensorMLD*> tensors() {
    return {{"intermediate", &intermediate}, {"prev_state", &prev_state}};
  }
};

struct LstmTestOutputs {
  TensorMLD new_state;
  TensorMLD activation;

  absl::flat_hash_map<std::string, TensorMLD*> tensors() {
    return {{"new_state", &new_state}, {"activation", &activation}};
  }
};

LstmTestOutputs LstmTestModel(LstmTestInputs& inputs) {
  LstmTestOutputs outputs;
  auto lstm_outputs = Lstm(inputs.intermediate, inputs.prev_state);
  outputs.new_state = lstm_outputs[0];
  outputs.new_state.SetName("new_state");
  outputs.activation = lstm_outputs[1];
  outputs.activation.SetName("activation");
  return outputs;
}

TEST(MlDriftModelRunnerTest, LstmOp) {
  const int state_size = 4;
  LstmTestInputs inputs = {
      .intermediate =
          TensorMLD({.type = Type::kFP32, .shape = {1, 1, 1, state_size * 4}}),
      .prev_state =
          TensorMLD({.type = Type::kFP32, .shape = {1, 1, 1, state_size}}),
  };
  LstmTestOutputs outputs;

  absl::flat_hash_map<std::string, absl::Span<const float>> float_weights;
  absl::flat_hash_map<std::string, absl::Span<const int8_t>> int8_weights;

  MlDriftModelRunner<decltype(&LstmTestModel), decltype(inputs),
                     decltype(outputs)>
      runner(&LstmTestModel, inputs, outputs, float_weights, int8_weights,
             /*build_gpu_model=*/true, ml_drift::CalculationsPrecision::kF32);

  const std::vector<float> intermediate_data = {
      0.1f, 0.2f, 0.3f, 0.4f,  // input gate
      0.5f, 0.6f, 0.7f, 0.8f,  // new input
      0.9f, 1.0f, 1.1f, 1.2f,  // forget gate
      1.3f, 1.4f, 1.5f, 1.6f,  // output gate
  };
  const std::vector<float> prev_state_data = {
      0.1f,
      0.2f,
      0.3f,
      0.4f,
  };

  ASSERT_OK(runner.SetInput("intermediate", intermediate_data));
  ASSERT_OK(runner.SetInput("prev_state", prev_state_data));

  ASSERT_OK(runner.Run());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto new_state,
                                  runner.GetFloatOutput("new_state"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto activation,
                                  runner.GetFloatOutput("activation"));

  auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
  auto tanh = [](float x) { return std::tanh(x); };

  std::vector<float> exp_new_state(state_size);
  std::vector<float> exp_activation(state_size);

  for (int i = 0; i < state_size; ++i) {
    float input_gate = sigmoid(intermediate_data[i]);
    float new_input = tanh(intermediate_data[state_size + i]);
    float forget_gate = sigmoid(intermediate_data[2 * state_size + i]);
    float output_gate = sigmoid(intermediate_data[3 * state_size + i]);

    exp_new_state[i] =
        input_gate * new_input + forget_gate * prev_state_data[i];
    exp_activation[i] = output_gate * tanh(exp_new_state[i]);
  }

  EXPECT_THAT(new_state, Pointwise(FloatNear(1e-5), exp_new_state));
  EXPECT_THAT(activation, Pointwise(FloatNear(1e-5), exp_activation));
}

TEST(MlDriftModelRunnerTest, MlDriftBufferIO) {
  int32_t channel_size = 4;
  TestInputs inputs = {
      .input_tensor =
          TensorMLD({.type = Type::kFP32, .shape = {1, 1, channel_size}}),
      .scalar_weight = TensorMLD({.type = Type::kFP32, .shape = {}}),
  };
  TestOutputs outputs;

  absl::flat_hash_map<std::string, absl::Span<const float>> float_weights;
  absl::flat_hash_map<std::string, absl::Span<const int8_t>> int8_weights;

  MlDriftModelRunner<decltype(&TestModel), decltype(inputs), decltype(outputs)>
      runner(&TestModel, inputs, outputs, float_weights, int8_weights,
             /*build_gpu_model=*/true, ml_drift::CalculationsPrecision::kF32);

  // Allocate input and output buffers on device.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto input_buf, runner.AllocateBuffer(channel_size * sizeof(float)));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto output_buf, runner.AllocateBuffer(channel_size * sizeof(float)));

  // Populate input buffer.
  {
    auto locked = input_buf->LockMutable().As<float>();
    absl::c_copy(std::vector<float>({1, 2, 3, 4}), locked.begin());
  }

  ASSERT_OK(runner.SetInput("input_tensor", *input_buf.get()));
  ASSERT_OK(runner.SetInput("scalar_weight", std::vector<float>{2.0f}));

  ASSERT_OK(runner.Run());

  ASSERT_OK(runner.GetOutput("output_tensor", *output_buf.get()));

  // Verify result.
  EXPECT_THAT(output_buf->Lock().As<const float>(),
              Pointwise(FloatNear(1e-6), {2.0f, 4.0f, 6.0f, 8.0f}));
}
}  // namespace

}  // namespace litert::tensor
