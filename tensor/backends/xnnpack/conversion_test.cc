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

#include "tensor/backends/xnnpack/conversion.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK  // from @XNNPACK
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/graph.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::absl_testing::StatusIs;

using XnnTensor = Tensor<XnnpackMixinTag>;

TEST(XnnpackConversionTest, MarksExternalValuesAndCopiesConstants) {
  XnnTensor runtime_input({.name = "input", .type = Type::kFP32, .shape = {3}});
  XnnTensor constant_bias({.name = "bias",
                           .type = Type::kFP32,
                           .shape = {3},
                           .buffer = std::vector<float>{1.f, 2.f, 3.f}});
  XnnTensor sum = Add(runtime_input, constant_bias);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({sum}));
  ASSERT_EQ(graph->values().size(), 3);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t input_index,
                                  graph->Lookup(runtime_input));
  const NnpackValue& input_value = graph->values()[input_index];
  EXPECT_NE(input_value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT, 0);
  EXPECT_EQ(input_value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT, 0);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t bias_index,
                                  graph->Lookup(constant_bias));
  const NnpackValue& bias_value = graph->values()[bias_index];
  EXPECT_EQ(bias_value.flags, 0);
  EXPECT_EQ(bias_value.data.size(), sizeof(float) * 3);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t output_index, graph->Lookup(sum));
  const NnpackValue& output_value = graph->values()[output_index];
  EXPECT_NE(output_value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT, 0);
}

TEST(XnnpackConversionTest, SharedTensorRegisteredOnce) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1}});
  XnnTensor other({.name = "other",
                   .type = Type::kFP32,
                   .shape = {1},
                   .buffer = std::vector<float>{2.f}});
  XnnTensor first = Add(input, other);
  XnnTensor second = Add(first, input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({second}));

  int external_inputs = 0;
  for (const NnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++external_inputs;
    }
  }
  EXPECT_EQ(external_inputs, 1);
}

TEST(XnnpackConversionTest, Relu6SetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = Relu6(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const NnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 1);
}

TEST(XnnpackConversionTest, LeakyReluSetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = LeakyRelu(input, 0.2f);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const NnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 1);
}

TEST(XnnpackConversionTest, EluSetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = Elu(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const NnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 1);
}

TEST(XnnpackConversionTest, GeluSetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = Gelu(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const NnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 1);
}

TEST(XnnpackConversionTest, HardSwishSetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = HardSwish(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const NnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 1);
}

TEST(XnnpackConversionTest, PReluSetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor alpha({.name = "alpha", .type = Type::kFP32, .shape = {2}});
  XnnTensor output = PRelu(input, alpha);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const NnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 2);
}

TEST(XnnpackConversionTest, L2NormalizationReturnsUnimplemented) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = L2Normalization(input);

  EXPECT_THAT(BuildXnnpackGraph({output}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(XnnpackConversionTest, BlockwiseScalarZeroPointWithEmptyScales) {
  XnnTensor weights(TensorInit{
      .name = "weights",
      .type = Type::kI4,
      .shape = {0, 32},
      .quantization = std::make_shared<BlockwiseQuantization>(
          std::vector<float>{}, ZeroPoints::Int({8}),
          /*block_size=*/32, /*quantized_dimension=*/0)});
  XnnpackBuildContext ctx({}, {});
  ASSERT_THAT(ctx.Init(), ::testing::tensor::IsOk());
  EXPECT_THAT(ctx.DefineValue(weights.GetRaw()).status(),
              ::testing::tensor::IsOk());
}

TEST(XnnpackConversionTest, BlockwiseEmptyScalesRejectsMultipleZeroPoints) {
  XnnTensor weights(TensorInit{
      .name = "weights",
      .type = Type::kI4,
      .shape = {0, 32},
      .quantization = std::make_shared<BlockwiseQuantization>(
          std::vector<float>{}, ZeroPoints::Int({8, 8}),
          /*block_size=*/32, /*quantized_dimension=*/0)});
  XnnpackBuildContext ctx({}, {});
  ASSERT_THAT(ctx.Init(), ::testing::tensor::IsOk());
  EXPECT_THAT(ctx.DefineValue(weights.GetRaw()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(XnnpackConversionTest, BlockwiseUniformAndNonUniformZeroPoints) {
  auto build_fc = [&](ZeroPoints zero_points) {
    XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 32}});
    XnnTensor weights(TensorInit{
        .name = "weights",
        .type = Type::kI4,
        .shape = {2, 32},
        .buffer = std::vector<int8_t>(32, 0),
        .quantization = std::make_shared<BlockwiseQuantization>(
            std::vector<float>{0.5f, 0.25f}, std::move(zero_points),
            /*block_size=*/32, /*quantized_dimension=*/0)});
    XnnTensor output = FullyConnected(input, weights);
    return BuildXnnpackGraph({output});
  };

  EXPECT_THAT(build_fc(ZeroPoints::Int({8, 8})), ::testing::tensor::IsOk());
  EXPECT_THAT(build_fc(ZeroPoints::Int({8, 5})),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(XnnpackConversionTest,
     DefinePerTensorQuantizedWeightsAsChannelwiseValidation) {
  auto define_weights = [](ZeroPoints zero_points, bool with_buffer) {
    TensorInit init{
        .name = "weights",
        .type = Type::kI8,
        .shape = {2, 4},
        .quantization = std::make_shared<PerChannelAffineQuantization>(
            std::vector<float>{0.5f}, std::move(zero_points),
            /*quantized_dimension=*/0)};
    if (with_buffer) {
      init.buffer = std::vector<int8_t>(8, 1);
    }
    XnnTensor weights(std::move(init));
    XnnpackBuildContext ctx({}, {});
    absl::Status init_status = ctx.Init();
    if (!init_status.ok()) {
      return absl::StatusOr<uint32_t>(init_status);
    }
    return ctx.DefinePerTensorQuantizedWeightsAsChannelwise(weights.GetRaw());
  };

  EXPECT_THAT(define_weights(ZeroPoints::Int({0}), /*with_buffer=*/true),
              IsOk());
  EXPECT_THAT(define_weights(ZeroPoints::Float({0.0f}), /*with_buffer=*/true),
              IsOk());
  EXPECT_THAT(define_weights(ZeroPoints::Float({-0.5f}), /*with_buffer=*/true),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(define_weights(ZeroPoints::Int({2, 2}), /*with_buffer=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument));
  const int64_t overflow_zp =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  EXPECT_THAT(
      define_weights(ZeroPoints::Int({overflow_zp}), /*with_buffer=*/true),
      StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(define_weights(ZeroPoints::Int({0}), /*with_buffer=*/false),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(XnnpackConversionTest,
     FullyConnectedPerTensorInt8WeightsWithFloatZeroPoints) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 4}});
  XnnTensor weights(
      TensorInit{.name = "weights",
                 .type = Type::kI8,
                 .shape = {2, 4},
                 .buffer = std::vector<int8_t>{1, 2, 3, 4, -1, -2, -3, -4},
                 .quantization = std::make_shared<PerChannelAffineQuantization>(
                     std::vector<float>{0.5f}, ZeroPoints::Float({-0.5f}),
                     /*quantized_dimension=*/0)});
  XnnTensor output = FullyConnected(input, weights);
  EXPECT_THAT(BuildXnnpackGraph({output}), IsOk());
}

TEST(XnnpackConversionTest, FingerprintSinkRegistrationDispatchAndEdgeCases) {
  struct SinkState {
    int calls = 0;
    std::vector<XnnConstantFingerprint> received;
  };
  SinkState state;
  xnn_weights_cache_provider provider{};
  provider.context = &state;
  xnn_weights_cache_t cache = &provider;

  const auto sink = +[](void* cache_context,
                        absl::Span<const XnnConstantFingerprint> fingerprints) {
    auto* s = static_cast<SinkState*>(cache_context);
    ++s->calls;
    s->received.assign(fingerprints.begin(), fingerprints.end());
  };

  int dummy_weight = 42;
  const std::vector<XnnConstantFingerprint> fps = {
      XnnConstantFingerprint{.data_ptr = &dummy_weight,
                             .content_hash = 0xCAFEBABEULL,
                             .min_packed_bytes = 64},
  };

  // Null cache or null sink registration is a safe no-op.
  RegisterXnnWeightsCacheFingerprintSink(/*cache=*/nullptr, sink);
  RegisterXnnWeightsCacheFingerprintSink(cache, /*sink=*/nullptr);
  NotifyXnnWeightsCacheConstantFingerprints(cache, fps);
  EXPECT_EQ(state.calls, 0);

  // Valid registration dispatches non-empty fingerprints and skips empty/null.
  RegisterXnnWeightsCacheFingerprintSink(cache, sink);
  const absl::Cleanup unregister_sink = [cache]() {
    UnregisterXnnWeightsCacheFingerprintSink(cache);
  };
  NotifyXnnWeightsCacheConstantFingerprints(/*cache=*/nullptr, fps);
  NotifyXnnWeightsCacheConstantFingerprints(cache, {});
  EXPECT_EQ(state.calls, 0);

  NotifyXnnWeightsCacheConstantFingerprints(cache, fps);
  EXPECT_EQ(state.calls, 1);
  ASSERT_EQ(state.received.size(), 1u);
  EXPECT_EQ(state.received[0].data_ptr, &dummy_weight);
  EXPECT_EQ(state.received[0].content_hash, 0xCAFEBABEULL);
  EXPECT_EQ(state.received[0].min_packed_bytes, 64u);

  // Unregistering stops future notifications (and null unregister is a no-op).
  UnregisterXnnWeightsCacheFingerprintSink(/*cache=*/nullptr);
  UnregisterXnnWeightsCacheFingerprintSink(cache);
  NotifyXnnWeightsCacheConstantFingerprints(cache, fps);
  EXPECT_EQ(state.calls, 1);
}
// This test is disabled because the lowering should not do implicit data
// conversions.
//
// TODO: b/493560478 - Remove this.
//
// TEST(XnnpackConversionTest, DISABLED_DequantizesQuantizedInt8ConstantWeights)
// {
//   XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 4}});
//   auto quantization = std::make_shared<PerChannelAffineQuantization>(
//       std::vector<float>{0.5f, 0.25f}, std::vector<int64_t>{0, 0},
//       /*quantized_dimension=*/0);
//   XnnTensor weights({.name = "weights",
//                      .type = Type::kI8,
//                      .shape = {2, 4},
//                      .buffer = std::vector<int8_t>{2, 4, -2, -4, 4, 8, -4,
//                      -8}, .quantization = quantization});
//   XnnTensor output = FullyConnected(input, weights);
//
//   LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));
//   LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t weights_index,
//   graph->Lookup(weights));
//   const NnpackValue& weights_value = graph->values()[weights_index];
//   ASSERT_EQ(weights_value.buffer.size(),
//   sizeof(float) * 8); EXPECT_EQ(weights_value.info.type, Type::kFP32);
//
//   std::array<float, 8> expected = {1.0f, 2.0f, -1.0f, -2.0f,
//                                    1.0f, 2.0f, -1.0f, -2.0f};
//   for (size_t i = 0; i < expected.size(); ++i) {
//     float value = 0.0f;
//     std::memcpy(&value, weights_value.buffer.data() + i * sizeof(float),
//                 sizeof(float));
//     EXPECT_FLOAT_EQ(value, expected[i]);
//   }
// }

}  // namespace
}  // namespace litert::tensor
