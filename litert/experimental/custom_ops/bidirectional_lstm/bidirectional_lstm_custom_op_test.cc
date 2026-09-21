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

#include "litert/experimental/custom_ops/bidirectional_lstm/bidirectional_lstm_custom_op.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/experimental/custom_ops/bidirectional_lstm/bidirectional_lstm_impl.h"
#include "litert/test/matchers.h"

namespace litert {
namespace custom_ops {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

// Small enough to write golden values against a scalar reference, but large
// enough that the gate slicing, the batch stride and the padding gather are all
// exercised: sequence 0 is full length, sequence 1 is padded.
constexpr int kBatch = 2;
constexpr int kTimeSteps = 4;
constexpr int kInputSize = 3;
constexpr int kHiddenSize = 2;

// Deterministic inputs. A real RNG would make a failure impossible to
// reproduce, and fixed literals for ~100 weights would be unreadable.
std::vector<float> PseudoRandom(int count, uint32_t seed) {
  std::vector<float> values(count);
  uint32_t state = seed;
  for (int i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    values[i] = static_cast<float>((state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  }
  return values;
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Scalar reference for one direction. Deliberately written as naive triple
// loops rather than reusing the Eigen path, so that a bug in the blocked GEMM
// or in the gate slicing shows up as a mismatch.
void GoldenUnidirectionalLstm(const float* x, const float* w_ih,
                              const float* w_hh, const float* bias, float* out,
                              int t_steps, int input_size, int hidden_size) {
  const int h = hidden_size;
  std::vector<float> state(h, 0.0f);
  std::vector<float> cell(h, 0.0f);
  std::vector<float> gates(4 * h);
  for (int t = 0; t < t_steps; ++t) {
    for (int r = 0; r < 4 * h; ++r) {
      float acc = bias[r];
      for (int i = 0; i < input_size; ++i) {
        acc += w_ih[r * input_size + i] * x[t * input_size + i];
      }
      for (int i = 0; i < h; ++i) {
        acc += w_hh[r * h + i] * state[i];
      }
      gates[r] = acc;
    }
    for (int i = 0; i < h; ++i) {
      const float input_gate = Sigmoid(gates[i]);
      const float forget_gate = Sigmoid(gates[h + i]);
      const float cell_gate = std::tanh(gates[2 * h + i]);
      const float output_gate = Sigmoid(gates[3 * h + i]);
      cell[i] = forget_gate * cell[i] + input_gate * cell_gate;
      state[i] = output_gate * std::tanh(cell[i]);
      out[t * h + i] = state[i];
    }
  }
}

int GoldenReverseIndex(int length, int t, int t_steps) {
  const int idx = length - 1 - t;
  if (idx < 0) return 0;
  if (idx > t_steps - 1) return t_steps - 1;
  return idx;
}

void GoldenBidirectionalLstm(const std::vector<float>& x,
                             const std::vector<int32_t>& seq_lengths,
                             const std::vector<float>& mask,
                             const std::vector<float>& w_ih_fwd,
                             const std::vector<float>& w_hh_fwd,
                             const std::vector<float>& b_fwd,
                             const std::vector<float>& w_ih_bwd,
                             const std::vector<float>& w_hh_bwd,
                             const std::vector<float>& b_bwd,
                             std::vector<float>& out) {
  const int d = kInputSize;
  const int h = kHiddenSize;
  out.assign(kBatch * kTimeSteps * 2 * h, 0.0f);

  std::vector<float> fwd_out(kTimeSteps * h);
  std::vector<float> reversed_in(kTimeSteps * d);
  std::vector<float> reversed_out(kTimeSteps * h);

  for (int b = 0; b < kBatch; ++b) {
    const float* x_b = x.data() + b * kTimeSteps * d;
    const float* mask_b = mask.data() + b * kTimeSteps;
    const int length = seq_lengths[b];

    GoldenUnidirectionalLstm(x_b, w_ih_fwd.data(), w_hh_fwd.data(),
                             b_fwd.data(), fwd_out.data(), kTimeSteps, d, h);

    for (int t = 0; t < kTimeSteps; ++t) {
      const int idx = GoldenReverseIndex(length, t, kTimeSteps);
      for (int i = 0; i < d; ++i) {
        reversed_in[t * d + i] = x_b[idx * d + i] * (1.0f - mask_b[t]);
      }
    }
    GoldenUnidirectionalLstm(reversed_in.data(), w_ih_bwd.data(),
                             w_hh_bwd.data(), b_bwd.data(), reversed_out.data(),
                             kTimeSteps, d, h);

    for (int t = 0; t < kTimeSteps; ++t) {
      const int idx = GoldenReverseIndex(length, t, kTimeSteps);
      float* dst = out.data() + (b * kTimeSteps + t) * 2 * h;
      for (int i = 0; i < h; ++i) {
        dst[i] = fwd_out[t * h + i];
        dst[h + i] = reversed_out[idx * h + i] * (1.0f - mask_b[t]);
      }
    }
  }
}

std::vector<uint8_t> BuildAttributes(int hidden_size,
                                     absl::string_view gate_order) {
  flexbuffers::Builder fbb;
  const size_t map_start = fbb.StartMap();
  fbb.String("gate_order", std::string(gate_order));
  fbb.Int("hidden_size", hidden_size);
  fbb.EndMap(map_start);
  fbb.Finish();
  return fbb.GetBuffer();
}

template <typename T>
Expected<TensorBuffer> MakeHostBuffer(Environment& env,
                                      std::initializer_list<int32_t> shape,
                                      const std::vector<T>& data) {
  const RankedTensorType tensor_type = MakeRankedTensorType<T>(shape);
  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  tensor_type, sizeof(T) * data.size()));
  LITERT_RETURN_IF_ERROR(buffer.Write<T>(absl::MakeConstSpan(data)));
  return buffer;
}

TEST(BidirectionalLstmCustomOpTest, OpNameMatchesTheCompositeName) {
  BidirectionalLstmCustomOpKernel kernel;
  // The MLIR legalization pass copies the composite name into `custom_code`
  // verbatim, dotted prefix included, and the resolver matches on exactly this
  // string. Dropping the prefix makes the model fail to load.
  EXPECT_EQ(kernel.OpName(), "litert_custom_op.bidirectional_lstm");
  EXPECT_EQ(kernel.OpVersion(), 1);
}

TEST(BidirectionalLstmCustomOpTest, InitAcceptsExporterAttributes) {
  BidirectionalLstmCustomOpKernel kernel;
  const std::vector<uint8_t> attributes = BuildAttributes(kHiddenSize, "ifgo");
  EXPECT_TRUE(kernel.Init(attributes.data(), attributes.size()));
}

TEST(BidirectionalLstmCustomOpTest, InitToleratesMissingAttributes) {
  BidirectionalLstmCustomOpKernel kernel;
  EXPECT_TRUE(kernel.Init(nullptr, 0));
}

TEST(BidirectionalLstmCustomOpTest, InitRejectsUnsupportedGateOrder) {
  BidirectionalLstmCustomOpKernel kernel;
  // TensorFlow packs gates as `ifco`; silently running that through a kernel
  // written for PyTorch's `ifgo` would produce plausible-looking garbage.
  const std::vector<uint8_t> attributes = BuildAttributes(kHiddenSize, "ifco");
  EXPECT_FALSE(kernel.Init(attributes.data(), attributes.size()));
}

TEST(BidirectionalLstmCustomOpTest, GetOutputLayoutsDoublesTheHiddenDim) {
  BidirectionalLstmCustomOpKernel kernel;
  const std::vector<uint8_t> attributes = BuildAttributes(kHiddenSize, "ifgo");
  ASSERT_TRUE(kernel.Init(attributes.data(), attributes.size()));

  std::vector<Layout> input_layouts = {
      Layout(Dimensions({kBatch, kTimeSteps, kInputSize})),
      Layout(Dimensions({kBatch})),
      Layout(Dimensions({kBatch, kTimeSteps})),
      Layout(Dimensions({4 * kHiddenSize, kInputSize})),
      Layout(Dimensions({4 * kHiddenSize, kHiddenSize})),
      Layout(Dimensions({4 * kHiddenSize})),
      Layout(Dimensions({4 * kHiddenSize, kInputSize})),
      Layout(Dimensions({4 * kHiddenSize, kHiddenSize})),
      Layout(Dimensions({4 * kHiddenSize})),
  };
  std::vector<Layout> output_layouts(1);
  ASSERT_TRUE(kernel.GetOutputLayouts(input_layouts, output_layouts));
  EXPECT_EQ(output_layouts[0],
            Layout(Dimensions({kBatch, kTimeSteps, 2 * kHiddenSize})));
}

TEST(BidirectionalLstmCustomOpTest, GetOutputLayoutsRejectsWrongArity) {
  BidirectionalLstmCustomOpKernel kernel;
  std::vector<Layout> input_layouts = {
      Layout(Dimensions({kBatch, kTimeSteps, kInputSize}))};
  std::vector<Layout> output_layouts(1);
  EXPECT_FALSE(kernel.GetOutputLayouts(input_layouts, output_layouts));
}

TEST(BidirectionalLstmCustomOpTest, GetOutputLayoutsRejectsInconsistentWhh) {
  BidirectionalLstmCustomOpKernel kernel;
  // `hidden_size` from the composite attributes is not consulted at all, so
  // `w_hh` being [4H, H] is the only thing that pins H down.
  std::vector<Layout> input_layouts(
      9, Layout(Dimensions({4 * kHiddenSize + 1, kHiddenSize})));
  input_layouts[0] = Layout(Dimensions({kBatch, kTimeSteps, kInputSize}));
  std::vector<Layout> output_layouts(1);
  EXPECT_FALSE(kernel.GetOutputLayouts(input_layouts, output_layouts));
}

TEST(BidirectionalLstmCustomOpTest, ImplMatchesScalarReference) {
  const std::vector<float> x = PseudoRandom(kTimeSteps * kInputSize, 1);
  const std::vector<float> w_ih =
      PseudoRandom(4 * kHiddenSize * kInputSize, 2);
  const std::vector<float> w_hh =
      PseudoRandom(4 * kHiddenSize * kHiddenSize, 3);
  const std::vector<float> bias = PseudoRandom(4 * kHiddenSize, 4);

  std::vector<float> actual(kTimeSteps * kHiddenSize);
  RunUnidirectionalLstm(x.data(), w_ih.data(), w_hh.data(), bias.data(),
                        actual.data(), kTimeSteps, kInputSize, kHiddenSize);

  std::vector<float> expected(kTimeSteps * kHiddenSize);
  GoldenUnidirectionalLstm(x.data(), w_ih.data(), w_hh.data(), bias.data(),
                           expected.data(), kTimeSteps, kInputSize,
                           kHiddenSize);

  EXPECT_THAT(actual, Pointwise(FloatNear(1e-5f), expected));
}

TEST(BidirectionalLstmCustomOpTest, RunMatchesScalarReferenceWithPadding) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, Environment::Create({}));

  const std::vector<float> x =
      PseudoRandom(kBatch * kTimeSteps * kInputSize, 1);
  const std::vector<float> w_ih_fwd =
      PseudoRandom(4 * kHiddenSize * kInputSize, 2);
  const std::vector<float> w_hh_fwd =
      PseudoRandom(4 * kHiddenSize * kHiddenSize, 3);
  const std::vector<float> b_fwd = PseudoRandom(4 * kHiddenSize, 4);
  const std::vector<float> w_ih_bwd =
      PseudoRandom(4 * kHiddenSize * kInputSize, 5);
  const std::vector<float> w_hh_bwd =
      PseudoRandom(4 * kHiddenSize * kHiddenSize, 6);
  const std::vector<float> b_bwd = PseudoRandom(4 * kHiddenSize, 7);

  // Sequence 0 fills the bucket; sequence 1 has two padding frames, so the
  // backward scan must start at t=1 for it and the tail must be masked out.
  const std::vector<int32_t> seq_lengths = {kTimeSteps, 2};
  const std::vector<float> mask = {0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 1.0f, 1.0f};

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer x_buffer,
      MakeHostBuffer<float>(env, {kBatch, kTimeSteps, kInputSize}, x));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer seq_lengths_buffer,
      MakeHostBuffer<int32_t>(env, {kBatch}, seq_lengths));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer mask_buffer,
      MakeHostBuffer<float>(env, {kBatch, kTimeSteps}, mask));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer w_ih_fwd_buffer,
      MakeHostBuffer<float>(env, {4 * kHiddenSize, kInputSize}, w_ih_fwd));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer w_hh_fwd_buffer,
      MakeHostBuffer<float>(env, {4 * kHiddenSize, kHiddenSize}, w_hh_fwd));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer b_fwd_buffer,
      MakeHostBuffer<float>(env, {4 * kHiddenSize}, b_fwd));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer w_ih_bwd_buffer,
      MakeHostBuffer<float>(env, {4 * kHiddenSize, kInputSize}, w_ih_bwd));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer w_hh_bwd_buffer,
      MakeHostBuffer<float>(env, {4 * kHiddenSize, kHiddenSize}, w_hh_bwd));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer b_bwd_buffer,
      MakeHostBuffer<float>(env, {4 * kHiddenSize}, b_bwd));

  const int output_elements = kBatch * kTimeSteps * 2 * kHiddenSize;
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer output_buffer,
      TensorBuffer::CreateManaged(
          env, TensorBufferType::kHostMemory,
          MakeRankedTensorType<float>({kBatch, kTimeSteps, 2 * kHiddenSize}),
          sizeof(float) * output_elements));

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(x_buffer));
  inputs.push_back(std::move(seq_lengths_buffer));
  inputs.push_back(std::move(mask_buffer));
  inputs.push_back(std::move(w_ih_fwd_buffer));
  inputs.push_back(std::move(w_hh_fwd_buffer));
  inputs.push_back(std::move(b_fwd_buffer));
  inputs.push_back(std::move(w_ih_bwd_buffer));
  inputs.push_back(std::move(w_hh_bwd_buffer));
  inputs.push_back(std::move(b_bwd_buffer));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(output_buffer));

  BidirectionalLstmCustomOpKernel kernel;
  const std::vector<uint8_t> attributes = BuildAttributes(kHiddenSize, "ifgo");
  ASSERT_TRUE(kernel.Init(attributes.data(), attributes.size()));
  ASSERT_TRUE(kernel.Run(inputs, outputs));

  std::vector<float> expected;
  GoldenBidirectionalLstm(x, seq_lengths, mask, w_ih_fwd, w_hh_fwd, b_fwd,
                          w_ih_bwd, w_hh_bwd, b_bwd, expected);

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[0], TensorBuffer::LockMode::kRead));
  EXPECT_THAT(absl::MakeSpan(output_lock.second, output_elements),
              Pointwise(FloatNear(1e-5f), expected));
}

TEST(BidirectionalLstmCustomOpTest, RunRejectsWrongArity) {
  std::vector<TensorBuffer> inputs;
  std::vector<TensorBuffer> outputs;
  BidirectionalLstmCustomOpKernel kernel;
  EXPECT_FALSE(kernel.Run(inputs, outputs));
}

TEST(BidirectionalLstmCustomOpTest, RegisterAddsTheKernelToOptions) {
  LITERT_ASSERT_OK_AND_ASSIGN(Options options, Options::Create());
  EXPECT_TRUE(RegisterBidirectionalLstmCustomOp(options));
}

}  // namespace
}  // namespace custom_ops
}  // namespace litert
