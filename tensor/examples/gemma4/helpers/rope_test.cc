/* Copyright 2026 Google LLC.

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

#include "tensor/examples/gemma4/helpers/rope.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;
using XnnTensor = Tensor<XnnpackMixinTag>;

std::vector<float> ComputeAngles(const size_t head_dim) {
  std::vector<float> angles(head_dim / 2);
  for (size_t i = 0; i < angles.size(); ++i) {
    angles[i] = (i + 1) * M_PI / (head_dim / 2 * 3);
  }
  angles.insert(angles.end(), angles.begin(), angles.end());
  return angles;
}

template <class RetManual = void, class F, class T,
          class Ret = std::conditional_t<
              std::is_same_v<RetManual, void>,
              decltype(std::declval<F>()(std::declval<T>())), RetManual>>
std::vector<Ret> vector_from(std::vector<T> v, F&& func) {
  std::vector<Ret> res(v.size());
  std::transform(v.begin(), v.end(), res.begin(), std::forward<F>(func));
  return res;
}

TEST(Gemma4GraphTest, RoPETest) {
  XnnTensor x({.name = "x", .type = Type::kFP32, .shape = {1, 1, 1, 4}});

  const std::vector<float> angles = ComputeAngles(x.GetShape()[3]);

  XnnTensor cos({.name = "cos",
                 .type = Type::kFP32,
                 .shape = {1, 1, 1, 4},
                 .buffer = vector_from(angles, cosf)});
  XnnTensor sin({.name = "sin",
                 .type = Type::kFP32,
                 .shape = {1, 1, 1, 4},
                 .buffer = vector_from(angles, sinf)});
  XnnTensor output = RoPE(x, cos, sin);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({output}));

  const std::array<float, 4> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(x, input_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> result,
                                  runner.ReadOutput(output));
  // Expected data computed using the script in `./reference/rope.py`.
  const std::array<float, 4> expected_data = {-0.6339746f, -2.4641016f,
                                              3.0980762f, 3.7320508f};
  EXPECT_THAT(std::move(result).As<const float>(),
              Pointwise(FloatNear(1e-5f), expected_data));
}

TEST(RopeTest, RopeCosSinTest) {
  auto [cos_tensor, sin_tensor] =
      RopeCosSin<XnnpackMixinTag>(/*seq_len=*/1, /*head_dim=*/4,
                                  /*rope_base=*/10000.0f,
                                  /*rope_proportion=*/1.0f);
  EXPECT_EQ(cos_tensor.GetShape(), Shape({1, 1, 1, 4}));
  EXPECT_EQ(sin_tensor.GetShape(), Shape({1, 1, 1, 4}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & cos_buf, cos_tensor.GetBuffer());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & sin_buf, sin_tensor.GetBuffer());

  auto cos_span = cos_buf.Lock().As<const float>();
  auto sin_span = sin_buf.Lock().As<const float>();

  // For position 0: angle = 0, so cos = 1.0, sin = 0.0
  const std::array<float, 4> expected_cos = {1.0f, 1.0f, 1.0f, 1.0f};
  const std::array<float, 4> expected_sin = {0.0f, 0.0f, 0.0f, 0.0f};

  EXPECT_THAT(cos_span, Pointwise(FloatNear(1e-5f), expected_cos));
  EXPECT_THAT(sin_span, Pointwise(FloatNear(1e-5f), expected_sin));
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
