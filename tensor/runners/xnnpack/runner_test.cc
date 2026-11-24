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

#include "tensor/runners/xnnpack/runner.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/graph.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/common_nnpack/runner_test_suite.h"
#include "tensor/tensor.h"
#include "pthreadpool.h"  // from @pthreadpool

namespace litert::tensor {

struct XnnpackTestTraits {
  using Tag = XnnpackMixinTag;
  using Runner = XnnpackRunner;

  static constexpr uint32_t kFlagExternalInput = XNN_VALUE_FLAG_EXTERNAL_INPUT;
  static constexpr uint32_t kFlagExternalOutput =
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT;

  static constexpr bool kSupportsConv2D = true;
  static constexpr bool kSupportsDepthwiseConv2D = true;
  static constexpr bool kSupportsTransposeConv2D = true;
  static constexpr bool kSupportsResize = true;
  static constexpr bool kSupportsSoftmaxBeta = false;
  // TODO(b/574438217): Enable once the XNNPACK backend is fixed. Its squeeze
  // freezes the dimensions it keeps, so reshaping the input fails in
  // `xnn_reshape_runtime`.
  static constexpr bool kSupportsSqueezeOfReshapedInput = false;
  // TODO(b/574438217): Enable once the XNNPACK backend rejects squeezing a
  // dimension whose extent isn't 1.
  static constexpr bool kRejectsInvalidSqueeze = false;
};

INSTANTIATE_TYPED_TEST_SUITE_P(Xnnpack, NnpackRunnerTest, XnnpackTestTraits);

TEST(XnnpackRunnerTest, SoftmaxRejectsBetaNotEqualToOne) {
  Tensor<XnnpackMixinTag> input({.name = "input",
                                 .type = Type::kFP32,
                                 .shape = {2},
                                 .buffer = std::vector<float>{0.f, 0.f}});
  Tensor<XnnpackMixinTag> output = Softmax(input, /*beta=*/2.0f);

  auto runner = XnnpackRunner::Create({output});
  EXPECT_TRUE(absl::IsUnimplemented(runner.status()));
}

TEST(XnnpackRunnerTest, SharesWeightsCacheAndWorkspaceAcrossRunners) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_weights_cache_t raw_weights_cache = nullptr;
  ASSERT_EQ(xnn_create_weights_cache(&raw_weights_cache), xnn_status_success);
  std::unique_ptr<xnn_weights_cache_provider,
                  decltype(&xnn_delete_weights_cache)>
      weights_cache(raw_weights_cache, &xnn_delete_weights_cache);
  xnn_workspace_t raw_workspace = nullptr;
  ASSERT_EQ(xnn_create_workspace(&raw_workspace), xnn_status_success);
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> workspace(
      raw_workspace, &xnn_release_workspace);

  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<XnnpackMixinTag> w(
      {.name = "w", .type = Type::kFP32, .shape = {2, 3}, .buffer = weights});

  Tensor<XnnpackMixinTag> x_prefill(
      {.name = "x_prefill", .type = Type::kFP32, .shape = {2, 3}});
  Tensor<XnnpackMixinTag> y_prefill = FullyConnected(x_prefill, w);

  Tensor<XnnpackMixinTag> x_decode(
      {.name = "x_decode", .type = Type::kFP32, .shape = {1, 3}});
  Tensor<XnnpackMixinTag> y_decode = FullyConnected(x_decode, w);

  auto prefill_runner = XnnpackRunner::Create({y_prefill});
  ASSERT_TRUE(prefill_runner.ok());
  prefill_runner->SetWeightsCache(weights_cache.get());
  prefill_runner->SetWorkspace(workspace.get());
  ASSERT_TRUE(prefill_runner->PrepareRuntime().ok());

  auto decode_runner = XnnpackRunner::Create({y_decode});
  ASSERT_TRUE(decode_runner.ok());
  decode_runner->SetWeightsCache(weights_cache.get());
  decode_runner->SetWorkspace(workspace.get());
  ASSERT_TRUE(decode_runner->PrepareRuntime().ok());
  EXPECT_EQ(decode_runner->workspace(), workspace.get());
  ASSERT_EQ(xnn_finalize_weights_cache(
                weights_cache.get(), xnn_weights_cache_finalization_kind_soft),
            xnn_status_success);

  const std::vector<float> prefill_in = {1.0f, 0.0f, 2.0f, 0.5f, 1.0f, -1.0f};
  ASSERT_TRUE(prefill_runner->SetInput(x_prefill, prefill_in).ok());
  ASSERT_TRUE(prefill_runner->Run().ok());
  auto prefill_out = prefill_runner->ReadOutputAs<float>(y_prefill);
  ASSERT_TRUE(prefill_out.ok());
  ASSERT_EQ(prefill_out->size(), 4u);
  EXPECT_FLOAT_EQ(prefill_out->data()[0], 7.0f);
  EXPECT_FLOAT_EQ(prefill_out->data()[1], 16.0f);
  EXPECT_FLOAT_EQ(prefill_out->data()[2], -0.5f);
  EXPECT_FLOAT_EQ(prefill_out->data()[3], 1.0f);

  const std::vector<float> decode_in = {0.5f, 1.0f, -1.0f};
  ASSERT_TRUE(decode_runner->SetInput(x_decode, decode_in).ok());
  ASSERT_TRUE(decode_runner->Run().ok());
  auto decode_out = decode_runner->ReadOutputAs<float>(y_decode);
  ASSERT_TRUE(decode_out.ok());
  ASSERT_EQ(decode_out->size(), 2u);
  EXPECT_FLOAT_EQ(decode_out->data()[0], -0.5f);
  EXPECT_FLOAT_EQ(decode_out->data()[1], 1.0f);
}

TEST(XnnpackRunnerTest, SharesNonOwningThreadpoolAndSkipsRedundantSetup) {
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> pool(
      pthreadpool_create(/*threads_count=*/2), &pthreadpool_destroy);
  ASSERT_NE(pool, nullptr);

  const std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<XnnpackMixinTag> w(
      {.name = "w", .type = Type::kFP32, .shape = {2, 3}, .buffer = weights});
  Tensor<XnnpackMixinTag> x(
      {.name = "x", .type = Type::kFP32, .shape = {1, 3}});
  Tensor<XnnpackMixinTag> y = FullyConnected(x, w);

  auto runner1 = XnnpackRunner::Create({y});
  ASSERT_TRUE(runner1.ok());
  ASSERT_TRUE(runner1->SetThreadpool(pool.get()).ok());
  EXPECT_EQ(runner1->threadpool(), pool.get());

  auto runner2 = XnnpackRunner::Create({y});
  ASSERT_TRUE(runner2.ok());
  ASSERT_TRUE(runner2->SetThreadpool(pool.get()).ok());
  EXPECT_EQ(runner2->threadpool(), pool.get());

  std::vector<float> in1 = {1.0f, 0.0f, 2.0f};
  ASSERT_TRUE(runner1->SetInput(x, in1).ok());
  ASSERT_TRUE(runner1->Run().ok());
  EXPECT_EQ(runner1->setup_count(), 1u);
  auto out1 = runner1->ReadOutputAs<float>(y);
  ASSERT_TRUE(out1.ok());
  EXPECT_FLOAT_EQ(out1->data()[0], 7.0f);
  EXPECT_FLOAT_EQ(out1->data()[1], 16.0f);

  // Updating input data in place keeps pointers and shapes stable, so
  // `xnn_setup_runtime_v2` is skipped while updated values are observed.
  in1 = {0.5f, 1.0f, -1.0f};
  ASSERT_TRUE(runner1->SetInput(x, in1).ok());
  ASSERT_TRUE(runner1->Run().ok());
  EXPECT_EQ(runner1->setup_count(), 1u);
  out1 = runner1->ReadOutputAs<float>(y);
  ASSERT_TRUE(out1.ok());
  EXPECT_FLOAT_EQ(out1->data()[0], -0.5f);
  EXPECT_FLOAT_EQ(out1->data()[1], 1.0f);

  // Switching to a different input buffer pointer triggers re-setup.
  std::vector<float> in2 = {2.0f, 1.0f, 0.0f};
  ASSERT_TRUE(runner1->SetInput(x, in2).ok());
  ASSERT_TRUE(runner1->Run().ok());
  EXPECT_EQ(runner1->setup_count(), 2u);
  out1 = runner1->ReadOutputAs<float>(y);
  ASSERT_TRUE(out1.ok());
  EXPECT_FLOAT_EQ(out1->data()[0], 4.0f);
  EXPECT_FLOAT_EQ(out1->data()[1], 13.0f);

  // Second runner shares the same threadpool safely.
  ASSERT_TRUE(runner2->SetInput(x, in2).ok());
  ASSERT_TRUE(runner2->Run().ok());
}

TEST(XnnpackRunnerTest, ReleasesSubgraphAndPackedFcWeightsAfterCreateRuntime) {
  const std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> bias = {0.5f, -0.5f};
  std::shared_ptr<Buffer> weight_buf =
      OwningCpuBuffer::Copy<Type::kFP32>(weights);
  std::weak_ptr<Buffer> weak_weight_buf = weight_buf;

  Tensor<XnnpackMixinTag> x(
      {.name = "x", .type = Type::kFP32, .shape = {1, 3}});
  Tensor<XnnpackMixinTag> y = TensorHandle::Invalid();
  {
    Tensor<XnnpackMixinTag> w({.name = "w",
                               .type = Type::kFP32,
                               .shape = {2, 3},
                               .buffer = std::move(weight_buf)});
    Tensor<XnnpackMixinTag> b(
        {.name = "b", .type = Type::kFP32, .shape = {2}, .buffer = bias});
    y = Add(FullyConnected(x, w), b);
  }

  auto runner = XnnpackRunner::Create({y});
  ASSERT_TRUE(runner.ok());
  ASSERT_TRUE(runner->PrepareRuntime().ok());
  // Release Tensor IR producers as runtime backends do post-compile.
  y.GetRaw().group->producer.reset();

  const auto& xnn_graph = static_cast<const XnnpackGraph&>(runner->graph());
  EXPECT_EQ(xnn_graph.GetSubgraph(), nullptr);
  EXPECT_TRUE(weak_weight_buf.expired());

  // Repeated invocations still produce exact results using packed weights and
  // retained non-packed constant `bias`.
  const std::vector<float> in = {1.0f, 0.0f, 2.0f};
  ASSERT_TRUE(runner->SetInput(x, in).ok());
  ASSERT_TRUE(runner->Run().ok());
  auto out = runner->ReadOutputAs<float>(y);
  ASSERT_TRUE(out.ok());
  EXPECT_FLOAT_EQ(out->data()[0], 7.5f);
  EXPECT_FLOAT_EQ(out->data()[1], 15.5f);
}

TEST(XnnpackRunnerTest, BlockwiseI4BiasedToZeroPoint8MatchesExplicitU4) {
  constexpr int kOut = 2;
  constexpr int kIn = 32;
  constexpr int kBlockSize = 32;
  std::vector<int4_t> i4_packed(kOut * kIn / 2);
  std::vector<uint8_t> u4_packed(kOut * kIn / 2);
  // Exercise the full signed 4-bit range [-8, 7].
  for (int r = 0; r < kOut; ++r) {
    for (int c = 0; c < kIn; c += 2) {
      const int8_t q0 = static_cast<int8_t>(((r * 5 + c) % 16) - 8);
      const int8_t q1 = static_cast<int8_t>(((r * 5 + c + 1) % 16) - 8);
      const size_t idx = static_cast<size_t>(r * (kIn / 2) + c / 2);
      i4_packed[idx].a = q0;
      i4_packed[idx].b = q1;
      u4_packed[idx] =
          static_cast<uint8_t>((q0 + 8) | ((q1 + 8) << 4));
    }
  }
  const std::vector<float> scales = {0.25f, 0.5f};

  Tensor<XnnpackMixinTag> x(
      {.name = "x", .type = Type::kFP32, .shape = {1, kIn}});
  Tensor<XnnpackMixinTag> w_i4(
      {.name = "w_i4",
       .type = Type::kI4,
       .shape = {kOut, kIn},
       .buffer = i4_packed,
       .quantization = std::make_shared<BlockwiseQuantization>(
           scales, /*zero_points=*/std::vector<int64_t>{}, kBlockSize,
           /*quantized_dimension=*/0)});
  Tensor<XnnpackMixinTag> w_u4(
      {.name = "w_u4",
       .type = Type::kU4,
       .shape = {kOut, kIn},
       .buffer = OwningCpuBuffer::Copy(
           reinterpret_cast<const char*>(u4_packed.data()), u4_packed.size()),
       .quantization = std::make_shared<BlockwiseQuantization>(
           scales, /*zero_points=*/std::vector<int64_t>{8}, kBlockSize,
           /*quantized_dimension=*/0)});

  Tensor<XnnpackMixinTag> y_i4 = FullyConnected(x, w_i4);
  Tensor<XnnpackMixinTag> y_u4 = FullyConnected(x, w_u4);

  auto r_i4 = XnnpackRunner::Create({y_i4});
  ASSERT_TRUE(r_i4.ok());
  const auto& g_i4 = static_cast<const XnnpackGraph&>(r_i4->graph());
  EXPECT_EQ(g_i4.biased_qb4w_buffers().size(), 1u);
  EXPECT_EQ(g_i4.weight_aliases().size(), 1u);

  auto r_u4 = XnnpackRunner::Create({y_u4});
  ASSERT_TRUE(r_u4.ok());

  std::vector<float> in(kIn);
  for (int c = 0; c < kIn; ++c) {
    in[c] = static_cast<float>(c - 16) * 0.1f;
  }
  ASSERT_TRUE(r_i4->SetInput(x, in).ok());
  ASSERT_TRUE(r_i4->Run().ok());
  // Biased buffer was released after packing into the runtime.
  EXPECT_TRUE(g_i4.biased_qb4w_buffers().empty());

  ASSERT_TRUE(r_u4->SetInput(x, in).ok());
  ASSERT_TRUE(r_u4->Run().ok());

  auto out_i4 = r_i4->ReadOutputAs<float>(y_i4);
  auto out_u4 = r_u4->ReadOutputAs<float>(y_u4);
  ASSERT_TRUE(out_i4.ok());
  ASSERT_TRUE(out_u4.ok());
  ASSERT_EQ(out_i4->size(), static_cast<size_t>(kOut));
  ASSERT_EQ(out_u4->size(), static_cast<size_t>(kOut));
  for (int r = 0; r < kOut; ++r) {
    EXPECT_NEAR(out_i4->data()[r], out_u4->data()[r], 1e-4f);
  }
}

}  // namespace litert::tensor
