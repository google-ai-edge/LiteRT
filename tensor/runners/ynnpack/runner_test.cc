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

#include "tensor/runners/ynnpack/runner.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/runners/common_nnpack/runner_test_suite.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {

struct YnnpackTestTraits {
  using Tag = YnnpackMixinTag;
  using Runner = YnnpackRunner;

  static constexpr uint32_t kFlagExternalInput = YNN_VALUE_FLAG_EXTERNAL_INPUT;
  static constexpr uint32_t kFlagExternalOutput =
      YNN_VALUE_FLAG_EXTERNAL_OUTPUT;

  // YNNPACK has no convolution, pooling or resize primitives. These would have
  // to be composed out of stencil copies and dots, which isn't implemented.
  static constexpr bool kSupportsConv2D = false;
  static constexpr bool kSupportsDepthwiseConv2D = false;
  static constexpr bool kSupportsTransposeConv2D = false;
  static constexpr bool kSupportsResize = false;
  static constexpr bool kSupportsSoftmaxBeta = true;
};

INSTANTIATE_TYPED_TEST_SUITE_P(Ynnpack, NnpackRunnerTest, YnnpackTestTraits);

namespace {

using ::testing::ElementsAreArray;
using ::testing::FloatNear;
using ::testing::Pointwise;

using YnnTensor = Tensor<YnnpackMixinTag>;

std::vector<float> Iota(int size, float scale) {
  std::vector<float> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = std::sin(0.3f * i + 0.1f) * scale;
  }
  return values;
}

// An attention mask [1, 1, S, K] broadcasts over the heads of the scores
// [1, H, S, K]. External inputs have dynamic extents, so the backend has to
// mark the head dimension of the mask as a broadcast explicitly, and the graph
// has to keep working once S and K change.
TEST(YnnpackRunnerTest, BroadcastsDynamicExternalInput) {
  constexpr int kHeads = 2;
  YnnTensor mask({.name = "mask", .type = Type::kFP32, .shape = {1, 1, 1, 3}});
  YnnTensor scores(
      {.name = "scores", .type = Type::kFP32, .shape = {1, kHeads, 1, 3}});
  YnnTensor output = Add(scores, mask);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(YnnpackRunner runner,
                                  YnnpackRunner::Create({output}));

  for (const auto [seq, keys] : {std::pair{1, 3}, std::pair{2, 4}}) {
    SCOPED_TRACE(testing::Message() << "seq=" << seq << " keys=" << keys);
    const std::vector<float> mask_data = Iota(seq * keys, 1.f);
    const std::vector<float> scores_data = Iota(kHeads * seq * keys, 3.f);
    ASSERT_THAT(runner.ReshapeInput(mask, {1, 1, seq, keys}), IsOk());
    ASSERT_THAT(runner.ReshapeInput(scores, {1, kHeads, seq, keys}), IsOk());
    ASSERT_THAT(runner.SetInputAsCopy(mask, mask_data), IsOk());
    ASSERT_THAT(runner.SetInputAsCopy(scores, scores_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());

    std::vector<float> expected(scores_data.size());
    for (int h = 0; h < kHeads; ++h) {
      for (int i = 0; i < seq * keys; ++i) {
        expected[h * seq * keys + i] =
            scores_data[h * seq * keys + i] + mask_data[i];
      }
    }
    EXPECT_THAT(runner.ReadOutputAs<float>(output),
                IsOkAndHolds(Pointwise(FloatNear(1e-6f), expected)));
  }
}

// The key of grouped query attention is tiled over the query heads, transposed
// and consumed as the right-hand side of a batch matmul. This used to need a
// copy after the tile to work around a YNNPACK bug (fixed in cl/982569033).
TEST(YnnpackRunnerTest, TileTransposeBatchMatMul) {
  constexpr int kHeads = 4;
  constexpr int kSeq = 2;
  constexpr int kDim = 8;
  YnnTensor q(
      {.name = "q", .type = Type::kFP32, .shape = {1, kHeads, kSeq, kDim}});
  YnnTensor k({.name = "k", .type = Type::kFP32, .shape = {1, 1, 3, kDim}});
  YnnTensor tiled = Tile(k, std::vector<int>{1, kHeads, 1, 1});
  YnnTensor transposed = Transpose(tiled, std::vector<int>{0, 1, 3, 2});
  YnnTensor output = BatchMatMul(q, transposed);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(YnnpackRunner runner,
                                  YnnpackRunner::Create({output}));

  for (const int keys : {3, 5}) {
    SCOPED_TRACE(testing::Message() << "keys=" << keys);
    const std::vector<float> q_data = Iota(kHeads * kSeq * kDim, 1.f);
    const std::vector<float> k_data = Iota(keys * kDim, 2.f);
    ASSERT_THAT(runner.ReshapeInput(k, {1, 1, keys, kDim}), IsOk());
    ASSERT_THAT(runner.SetInputAsCopy(q, q_data), IsOk());
    ASSERT_THAT(runner.SetInputAsCopy(k, k_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());

    std::vector<float> expected(kHeads * kSeq * keys);
    for (int h = 0; h < kHeads; ++h) {
      for (int s = 0; s < kSeq; ++s) {
        for (int t = 0; t < keys; ++t) {
          float sum = 0.f;
          for (int d = 0; d < kDim; ++d) {
            sum += q_data[(h * kSeq + s) * kDim + d] * k_data[t * kDim + d];
          }
          expected[(h * kSeq + s) * keys + t] = sum;
        }
      }
    }
    EXPECT_THAT(runner.ReadOutputAs<float>(output),
                IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected)));
  }
}

// Squeeze drops dimensions without spelling out the output shape, so the
// dimensions it keeps stay dynamic.
TEST(YnnpackRunnerTest, SqueezeKeepsDynamicDims) {
  YnnTensor x({.name = "x", .type = Type::kFP32, .shape = {1, 2, 3}});
  YnnTensor output = Squeeze(x, {0});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(YnnpackRunner runner,
                                  YnnpackRunner::Create({output}));
  for (const int seq : {2, 4}) {
    SCOPED_TRACE(testing::Message() << "seq=" << seq);
    const std::vector<float> x_data = Iota(seq * 3, 1.f);
    ASSERT_THAT(runner.ReshapeInput(x, {1, seq, 3}), IsOk());
    ASSERT_THAT(runner.SetInputAsCopy(x, x_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    EXPECT_THAT(runner.ReadOutputAs<float>(output),
                IsOkAndHolds(ElementsAreArray(x_data)));
  }
}

TEST(YnnpackRunnerTest, SqueezeSpecificDims) {
  const std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  YnnTensor x({.name = "x",
               .type = Type::kFP32,
               .shape = {2, 1, 3, 1},
               .buffer = data});
  YnnTensor one_dim = Squeeze(x, {1});
  YnnTensor negative_dim = Squeeze(x, {-1});
  YnnTensor all_dims = Squeeze(x);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      YnnpackRunner runner,
      YnnpackRunner::Create({one_dim, negative_dim, all_dims}));
  ASSERT_THAT(runner.Run(), IsOk());
  EXPECT_THAT(runner.ReadOutputAs<float>(one_dim),
              IsOkAndHolds(ElementsAreArray(data)));
  EXPECT_THAT(runner.ReadOutputAs<float>(negative_dim),
              IsOkAndHolds(ElementsAreArray(data)));
  EXPECT_THAT(runner.ReadOutputAs<float>(all_dims),
              IsOkAndHolds(ElementsAreArray(data)));
}

TEST(YnnpackRunnerTest, SqueezeRejectsDimsWithExtentOtherThanOne) {
  YnnTensor x({.name = "x",
               .type = Type::kFP32,
               .shape = {2, 1},
               .buffer = std::vector<float>{1.f, 2.f}});
  YnnTensor output = Squeeze(x, {0});
  EXPECT_EQ(YnnpackRunner::Create({output}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(YnnpackRunnerTest, RejectsConstantWithTooSmallBuffer) {
  YnnTensor x({.name = "x",
               .type = Type::kFP32,
               .shape = {4},
               .buffer = std::vector<float>{1.f, 2.f}});
  YnnTensor output = Add(x, x);
  EXPECT_EQ(YnnpackRunner::Create({output}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace litert::tensor
