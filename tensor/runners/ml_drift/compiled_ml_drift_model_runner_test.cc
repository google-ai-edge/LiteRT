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

#include "tensor/runners/ml_drift/compiled_ml_drift_model_runner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/compile.h"
#include "tensor/internal/graph.h"
#include "tensor/runners/ml_drift/ml_drift_cl_model_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using Runner = CompiledMlDriftModelRunner<MlDriftClBackend>;
using TensorMLD = Tensor<MlDriftMixinTag>;

// Forces `UseBufferForIntWeights(..., int_bit_size=4, ...)` to select the 2D
// texture weights layout (`k2DYIsSpatialIOAndXIsOGroupI4O4`) even when the host
// test runs on an NVIDIA OpenCL GPU, matching Mali, PowerVR, and Adreno < 740.
struct MlDriftClTextureWeightsBackend : public MlDriftClBackend {
  static ml_drift::GpuInfo GetGpuInfoForBuilder(Environment& env,
                                                bool env_initialized) {
    ml_drift::GpuInfo info =
        MlDriftClBackend::GetGpuInfoForBuilder(env, env_initialized);
    info.vendor = ml_drift::GpuVendor::kMali;
    return info;
  }
};

TEST(CompiledMlDriftModelRunnerTest, BasicExecution) {
  auto lambda = [](TensorMLD x) {
    auto y = litert::tensor::Add(x, x);
    y.SetName("out");
    return y;
  };

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2}});
  auto compiled_runner = compile<Runner>(lambda, x);

  std::vector<float> data_x = {1.0f, 2.0f};
  TensorMLD x_input(
      {.name = "x", .type = Type::kFP32, .shape = {1, 2}, .buffer = data_x});

  auto y = compiled_runner(x_input);

  EXPECT_THAT(y.GetShape(), testing::ElementsAre(1, 2));
}

// Constant FP32 FullyConnected weights must be packed into the kernel's
// weights layout; with the generic linear-constant path the kernel reads a
// truncated, un-rearranged buffer and only the first output channels are
// non-zero. The sizes are deliberately not multiples of 4.
TEST(CompiledMlDriftModelRunnerTest, ConstantFp32FullyConnectedWeights) {
  constexpr int kBatch = 3;
  constexpr int kIn = 7;
  constexpr int kOut = 10;
  std::vector<float> weights(kOut * kIn);
  for (int i = 0; i < weights.size(); ++i)
    weights[i] = 0.01f * (i % 13) - 0.05f;
  std::vector<float> input(kBatch * kIn);
  for (int i = 0; i < input.size(); ++i) input[i] = 0.1f * (i % 5) - 0.2f;

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, kBatch, kIn}});
  TensorMLD w({.name = "w",
               .type = Type::kFP32,
               .shape = {kOut, kIn},
               .buffer = weights});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");

  Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, kBatch, kIn},
                 .buffer = input});
  ASSERT_THAT(runner.SetInput("x", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("y"));

  std::vector<float> expected(kBatch * kOut, 0.0f);
  for (int b = 0; b < kBatch; ++b) {
    for (int o = 0; o < kOut; ++o) {
      for (int i = 0; i < kIn; ++i) {
        expected[b * kOut + o] += input[b * kIn + i] * weights[o * kIn + i];
      }
    }
  }
  EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(1e-5f), expected));
}

TEST(CompiledMlDriftModelRunnerTest, ConstantInt8FullyConnectedWeights) {
  constexpr int kBatch = 2;
  constexpr int kIn = 8;
  constexpr int kOut = 6;
  std::vector<int8_t> qweights(kOut * kIn);
  for (int i = 0; i < qweights.size(); ++i) {
    qweights[i] = static_cast<int8_t>((i % 11) - 5);
  }
  std::vector<float> scales(kOut);
  for (int o = 0; o < kOut; ++o) {
    scales[o] = 0.05f * static_cast<float>(o + 1);
  }
  std::vector<float> input(kBatch * kIn);
  for (int i = 0; i < input.size(); ++i) {
    input[i] = 0.1f * static_cast<float>((i % 7) - 3);
  }

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, kBatch, kIn}});
  TensorMLD w({.name = "w",
               .type = Type::kI8,
               .shape = {kOut, kIn},
               .buffer = qweights,
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   scales, std::vector<int64_t>(kOut, 0),
                   /*quantized_dimension=*/0)});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");

  Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, kBatch, kIn},
                 .buffer = input});
  ASSERT_THAT(runner.SetInput("x", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("y"));

  std::vector<float> expected(kBatch * kOut, 0.0f);
  for (int b = 0; b < kBatch; ++b) {
    for (int o = 0; o < kOut; ++o) {
      for (int i = 0; i < kIn; ++i) {
        expected[b * kOut + o] +=
            input[b * kIn + i] *
            (static_cast<float>(qweights[o * kIn + i]) * scales[o]);
      }
    }
  }
  EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(1e-4f), expected));
}

TEST(CompiledMlDriftModelRunnerTest,
     ConstantInt8FullyConnectedWeightsWithFloatZeroPoints) {
  constexpr int kBatch = 2;
  constexpr int kIn = 8;
  constexpr int kOut = 6;
  std::vector<int8_t> qweights(kOut * kIn);
  for (int i = 0; i < qweights.size(); ++i) {
    qweights[i] = static_cast<int8_t>((i % 11) - 5);
  }
  std::vector<float> scales(kOut);
  std::vector<float> zero_points(kOut);
  for (int o = 0; o < kOut; ++o) {
    scales[o] = 0.05f * static_cast<float>(o + 1);
    zero_points[o] = static_cast<float>((o % 3) - 1) + 0.5f;
  }
  std::vector<float> input(kBatch * kIn);
  for (int i = 0; i < input.size(); ++i) {
    input[i] = 0.1f * static_cast<float>((i % 7) - 3);
  }
  auto quantization = std::make_shared<PerChannelAffineQuantization>(
      scales, ZeroPoints::Float(zero_points), /*quantized_dimension=*/0);

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, kBatch, kIn}});
  TensorMLD w({.name = "w",
               .type = Type::kI8,
               .shape = {kOut, kIn},
               .buffer = qweights,
               .quantization = quantization});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");

  Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, kBatch, kIn},
                 .buffer = input});
  ASSERT_THAT(runner.SetInput("x", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("y"));

  std::vector<float> expected(kBatch * kOut, 0.0f);
  for (int b = 0; b < kBatch; ++b) {
    for (int o = 0; o < kOut; ++o) {
      for (int i = 0; i < kIn; ++i) {
        expected[b * kOut + o] +=
            input[b * kIn + i] *
            ((static_cast<float>(qweights[o * kIn + i]) - zero_points[o]) *
             scales[o]);
      }
    }
  }
  EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(1e-4f), expected));
}

TEST(CompiledMlDriftModelRunnerTest, ConstantInt2FullyConnectedWeights) {
  constexpr int kBatch = 2;
  constexpr int kIn = 8;
  constexpr int kOut = 4;
  std::shared_ptr<OwningCpuBuffer> w_buf =
      OwningCpuBuffer::Allocate<Type::kI2>(kOut * kIn);
  int2_t* packed = w_buf->Span<int2_t>().data();
  std::vector<int> unpacked(kOut * kIn);
  for (int idx = 0; idx < (kOut * kIn) / 4; ++idx) {
    const int v0 = (idx % 3) - 1;
    const int v1 = ((idx + 1) % 3) - 1;
    const int v2 = ((idx + 2) % 3) - 1;
    const int v3 = -1;
    packed[idx].a = v0;
    packed[idx].b = v1;
    packed[idx].c = v2;
    packed[idx].d = v3;
    unpacked[4 * idx + 0] = v0;
    unpacked[4 * idx + 1] = v1;
    unpacked[4 * idx + 2] = v2;
    unpacked[4 * idx + 3] = v3;
  }
  std::vector<float> scales(kOut, 0.25f);
  std::vector<float> input(kBatch * kIn, 0.5f);

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, kBatch, kIn}});
  TensorMLD w({.name = "w",
               .type = Type::kI2,
               .shape = {kOut, kIn},
               .buffer = w_buf,
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   scales, std::vector<int64_t>(kOut, 0),
                   /*quantized_dimension=*/0)});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");

  Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, kBatch, kIn},
                 .buffer = input});
  ASSERT_THAT(runner.SetInput("x", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("y"));

  std::vector<float> expected(kBatch * kOut, 0.0f);
  for (int b = 0; b < kBatch; ++b) {
    for (int o = 0; o < kOut; ++o) {
      for (int i = 0; i < kIn; ++i) {
        expected[b * kOut + o] +=
            input[b * kIn + i] *
            (static_cast<float>(unpacked[o * kIn + i]) * scales[o]);
      }
    }
  }
  EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(1e-4f), expected));
}

TEST(CompiledMlDriftModelRunnerTest,
     ConstantInt4TextureFullyConnectedWeights) {
  using TextureRunner =
      CompiledMlDriftModelRunner<MlDriftClTextureWeightsBackend>;
  constexpr int kBatch = 2;
  constexpr int kIn = 32;
  constexpr int kOut = 8;
  std::shared_ptr<OwningCpuBuffer> w_buf =
      OwningCpuBuffer::Allocate<Type::kI4>(kOut * kIn);
  int4_t* packed = w_buf->Span<int4_t>().data();
  std::vector<int> unpacked(kOut * kIn);
  for (int idx = 0; idx < (kOut * kIn) / 2; ++idx) {
    const int v0 = (idx % 11) - 5;
    const int v1 = ((idx * 3 + 2) % 11) - 5;
    packed[idx].a = v0;
    packed[idx].b = v1;
    unpacked[2 * idx + 0] = v0;
    unpacked[2 * idx + 1] = v1;
  }
  std::vector<float> scales(kOut);
  for (int o = 0; o < kOut; ++o) {
    scales[o] = 0.05f * static_cast<float>(o + 1);
  }
  std::vector<float> input(kBatch * kIn);
  for (int i = 0; i < input.size(); ++i) {
    input[i] = 0.1f * static_cast<float>((i % 7) - 3);
  }

  MlDriftClTextureWeightsBackend::Environment shared_env;
  ASSERT_THAT(MlDriftClTextureWeightsBackend::InitEnvironment(&shared_env),
              IsOk());
  TextureRunner::SharedWeightsCache shared_weights;

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, kBatch, kIn}});
  TensorMLD w({.name = "w_i4",
               .type = Type::kI4,
               .shape = {kOut, kIn},
               .buffer = w_buf,
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   scales, std::vector<int64_t>(kOut, 0),
                   /*quantized_dimension=*/0)});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");

  TextureRunner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32,
                       &shared_env, &shared_weights);
  ASSERT_THAT(runner.BuildModel(), IsOk());

  auto desc_it = shared_weights.descriptors.find("w_i4");
  ASSERT_NE(desc_it, shared_weights.descriptors.end());
  EXPECT_EQ(desc_it->second.GetStorageType(),
            ml_drift::TensorStorageType::kTexture2D);
  // Each RGBA texel holds 4 uint16 channels (16 uint4 weights), so the 2D
  // texture BHWDC shape has C = 4 and W = tex_size.x / 4 = DivideRoundUp(O, 4).
  EXPECT_EQ(desc_it->second.GetBHWDCShape(),
            ml_drift::BHWDC(1, kIn / 4, kOut / 4, 1, 4));

  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, kBatch, kIn},
                 .buffer = input});
  ASSERT_THAT(runner.SetInput("x", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("y"));

  std::vector<float> expected(kBatch * kOut, 0.0f);
  for (int b = 0; b < kBatch; ++b) {
    for (int o = 0; o < kOut; ++o) {
      for (int i = 0; i < kIn; ++i) {
        expected[b * kOut + o] +=
            input[b * kIn + i] *
            (static_cast<float>(unpacked[o * kIn + i]) * scales[o]);
      }
    }
  }
  EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(1e-4f), expected));
}

// INT4 weights tied between an EmbeddingLookup and a later FullyConnected
// (e.g. an LM head) get the FullyConnected's `sum_i` when 8-bit convolutions
// are allowed.
TEST(CompiledMlDriftModelRunnerTest,
     TiedInt4EmbeddingAndFullyConnectedWeightsGetSumI) {
  constexpr int kTokens = 2;
  constexpr int kVocab = 8;
  constexpr int kDim = 32;
  std::shared_ptr<OwningCpuBuffer> w_buf =
      OwningCpuBuffer::Allocate<Type::kI4>(kVocab * kDim);
  int4_t* packed = w_buf->Span<int4_t>().data();
  std::vector<int> unpacked(kVocab * kDim);
  for (int idx = 0; idx < (kVocab * kDim) / 2; ++idx) {
    const int v0 = (idx % 11) - 5;
    const int v1 = ((idx * 3 + 2) % 11) - 5;
    packed[idx].a = v0;
    packed[idx].b = v1;
    unpacked[2 * idx + 0] = v0;
    unpacked[2 * idx + 1] = v1;
  }
  std::vector<float> scales(kVocab);
  for (int v = 0; v < kVocab; ++v) {
    scales[v] = 0.01f * static_cast<float>(v + 1);
  }
  const std::vector<int32_t> ids = {1, 6};

  MlDriftClBackend::Environment shared_env;
  ASSERT_THAT(MlDriftClBackend::InitEnvironment(&shared_env), IsOk());
  Runner::SharedWeightsCache shared_weights;

  TensorMLD ids_in({.name = "ids", .type = Type::kI32, .shape = {1, kTokens}});
  TensorMLD w({.name = "tied",
               .type = Type::kI4,
               .shape = {kVocab, kDim},
               .buffer = w_buf,
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   scales, std::vector<int64_t>(kVocab, 0),
                   /*quantized_dimension=*/0)});
  TensorMLD logits = FullyConnected(EmbeddingLookup(ids_in, w), w);
  logits.SetName("logits");

  Runner runner({ids_in}, {logits}, ml_drift::CalculationsPrecision::kF32,
                &shared_env, &shared_weights, /*allow_8bit_convs=*/true);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  EXPECT_TRUE(shared_weights.descriptors.contains("tied.sum_i"));

  TensorMLD fed({.name = "ids",
                 .type = Type::kI32,
                 .shape = {1, kTokens},
                 .buffer = ids});
  ASSERT_THAT(runner.SetInput("ids", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("logits"));

  std::vector<float> expected(kTokens * kVocab, 0.0f);
  for (int t = 0; t < kTokens; ++t) {
    for (int v = 0; v < kVocab; ++v) {
      for (int d = 0; d < kDim; ++d) {
        const float embedding =
            static_cast<float>(unpacked[ids[t] * kDim + d]) * scales[ids[t]];
        expected[t * kVocab + v] +=
            embedding * static_cast<float>(unpacked[v * kDim + d]) * scales[v];
      }
    }
  }
  EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(1e-4f), expected));
}

// Runs a long prefill (128 tokens) of per-channel INT4 weights with non-zero
// integer (or, with `float_zero_points`, float) zero points and 8-bit
// convolutions allowed: where ML Drift supports INT8 convolutions, this runs on
// dynamically INT8-quantized activations with `sum_i` and additive zero points
// (elsewhere the regular INT4 kernel).
void TestLongPrefillInt4WeightsWithZeroPoints(bool float_zero_points) {
  constexpr int kTokens = 128;
  constexpr int kIn = 64;
  constexpr int kOut = 12;
  std::shared_ptr<OwningCpuBuffer> w_buf =
      OwningCpuBuffer::Allocate<Type::kI4>(kOut * kIn);
  int4_t* packed = w_buf->Span<int4_t>().data();
  std::vector<int> unpacked(kOut * kIn);
  for (int idx = 0; idx < (kOut * kIn) / 2; ++idx) {
    const int v0 = (idx * 7 + 1) % 16 - 8;
    const int v1 = (idx * 3 + 5) % 16 - 8;
    packed[idx].a = v0;
    packed[idx].b = v1;
    unpacked[2 * idx + 0] = v0;
    unpacked[2 * idx + 1] = v1;
  }
  std::vector<float> scales(kOut);
  std::vector<float> zero_points(kOut);
  for (int o = 0; o < kOut; ++o) {
    scales[o] = 0.02f * static_cast<float>(o + 1);
    zero_points[o] =
        static_cast<float>(o % 5 - 2) + (float_zero_points ? 0.5f : 0.0f);
  }
  ZeroPoints quantization_zero_points;
  if (float_zero_points) {
    quantization_zero_points = ZeroPoints::Float(zero_points);
  } else {
    quantization_zero_points =
        std::vector<int64_t>(zero_points.begin(), zero_points.end());
  }
  std::vector<float> input(kTokens * kIn);
  for (int k = 0; k < input.size(); ++k) {
    input[k] = 0.05f * static_cast<float>((k * 13) % 41 - 20);
  }

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, kTokens, kIn}});
  TensorMLD w({.name = "w_i4_zp",
               .type = Type::kI4,
               .shape = {kOut, kIn},
               .buffer = w_buf,
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   scales, std::move(quantization_zero_points),
                   /*quantized_dimension=*/0)});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");

  Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32,
                /*shared_env=*/nullptr, /*shared_weights=*/nullptr,
                /*allow_8bit_convs=*/true);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, kTokens, kIn},
                 .buffer = input});
  ASSERT_THAT(runner.SetInput("x", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("y"));

  std::vector<float> expected(kTokens * kOut, 0.0f);
  float max_abs = 0.0f;
  for (int t = 0; t < kTokens; ++t) {
    for (int o = 0; o < kOut; ++o) {
      float sum = 0.0f;
      for (int i = 0; i < kIn; ++i) {
        sum += input[t * kIn + i] *
               ((static_cast<float>(unpacked[o * kIn + i]) - zero_points[o]) *
                scales[o]);
      }
      expected[t * kOut + o] = sum;
      max_abs = std::max(max_abs, std::abs(sum));
    }
  }
  // INT8 activations round each input to 1/127 of its row's range.
  EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(0.02f * max_abs),
                                         expected));
}

TEST(CompiledMlDriftModelRunnerTest,
     LongPrefillInt4WeightsWithZeroPointsAndInt8Activations) {
  TestLongPrefillInt4WeightsWithZeroPoints(/*float_zero_points=*/false);
}

TEST(CompiledMlDriftModelRunnerTest,
     LongPrefillInt4WeightsWithFloatZeroPointsAndInt8Activations) {
  TestLongPrefillInt4WeightsWithZeroPoints(/*float_zero_points=*/true);
}

TEST(CompiledMlDriftModelRunnerTest,
     UnsupportedFp16AndBf16ConstantFullyConnectedWeightsReturnUnimplemented) {
  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2, 4}});
  {
    TensorMLD w_fp16({.name = "w_fp16",
                      .type = Type::kFP16,
                      .shape = {4, 4},
                      .buffer = std::vector<fp16_t>(16, fp16_t(1.0f))});
    TensorMLD y = FullyConnected(x, w_fp16);
    y.SetName("y");
    Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
    EXPECT_EQ(runner.BuildModel().code(), absl::StatusCode::kUnimplemented);
  }
  {
    TensorMLD w_bf16({.name = "w_bf16",
                      .type = Type::kBF16,
                      .shape = {4, 4},
                      .buffer = std::vector<bf16_t>(16, bf16_t(1.0f))});
    TensorMLD y = FullyConnected(x, w_bf16);
    y.SetName("y");
    Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
    EXPECT_EQ(runner.BuildModel().code(), absl::StatusCode::kUnimplemented);
  }
}

TEST(CompiledMlDriftModelRunnerTest,
     SharesConstantWeightsAcrossPrefillAndDecodeRunners) {
  constexpr int kPrefillLen = 3;
  constexpr int kIn = 8;
  constexpr int kOut = 6;
  std::vector<int8_t> qweights(kOut * kIn);
  for (int i = 0; i < qweights.size(); ++i) {
    qweights[i] = static_cast<int8_t>((i % 11) - 5);
  }
  std::vector<float> scales(kOut);
  for (int o = 0; o < kOut; ++o) {
    scales[o] = 0.05f * static_cast<float>(o + 1);
  }
  std::vector<float> prefill_input(kPrefillLen * kIn);
  for (int i = 0; i < prefill_input.size(); ++i) {
    prefill_input[i] = 0.1f * static_cast<float>((i % 7) - 3);
  }
  std::vector<float> decode_input(prefill_input.end() - kIn,
                                  prefill_input.end());

  MlDriftClBackend::Environment shared_env;
  ASSERT_THAT(MlDriftClBackend::InitEnvironment(&shared_env), IsOk());
  Runner::SharedWeightsCache shared_weights;

  TensorMLD w({.name = "linear.weight",
               .type = Type::kI8,
               .shape = {kOut, kIn},
               .buffer = qweights,
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   scales, std::vector<int64_t>(kOut, 0),
                   /*quantized_dimension=*/0)});

  TensorMLD x_prefill(
      {.name = "x", .type = Type::kFP32, .shape = {1, kPrefillLen, kIn}});
  TensorMLD y_prefill = FullyConnected(x_prefill, w);
  y_prefill.SetName("y");

  Runner prefill_runner({x_prefill}, {y_prefill},
                        ml_drift::CalculationsPrecision::kF32, &shared_env,
                        &shared_weights);
  ASSERT_THAT(prefill_runner.BuildModel(), IsOk());
  const size_t initial_misses = shared_weights.cache_misses;
  EXPECT_GT(initial_misses, 0u);
  EXPECT_EQ(shared_weights.cache_hits, 0u);

  TensorMLD x_decode({.name = "x", .type = Type::kFP32, .shape = {1, 1, kIn}});
  TensorMLD y_decode = FullyConnected(x_decode, w);
  y_decode.SetName("y");

  Runner decode_runner({x_decode}, {y_decode},
                       ml_drift::CalculationsPrecision::kF32, &shared_env,
                       &shared_weights);
  ASSERT_THAT(decode_runner.BuildModel(), IsOk());
  EXPECT_EQ(shared_weights.cache_misses, initial_misses);
  EXPECT_EQ(shared_weights.cache_hits, initial_misses);

  // Verify both runners share the exact same underlying GPU weight tensor.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto* prefill_w_tensor, prefill_runner.GetBackendTensor("linear.weight"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto* decode_w_tensor, decode_runner.GetBackendTensor("linear.weight"));
  EXPECT_EQ(prefill_w_tensor, decode_w_tensor);

  TensorMLD fed_prefill({.name = "x",
                         .type = Type::kFP32,
                         .shape = {1, kPrefillLen, kIn},
                         .buffer = prefill_input});
  ASSERT_THAT(prefill_runner.SetInput("x", fed_prefill), IsOk());
  ASSERT_THAT(prefill_runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> prefill_out,
                                  prefill_runner.GetFloatOutput("y"));

  TensorMLD fed_decode({.name = "x",
                        .type = Type::kFP32,
                        .shape = {1, 1, kIn},
                        .buffer = decode_input});
  ASSERT_THAT(decode_runner.SetInput("x", fed_decode), IsOk());
  ASSERT_THAT(decode_runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> decode_out,
                                  decode_runner.GetFloatOutput("y"));

  EXPECT_THAT(decode_out,
              testing::Pointwise(testing::FloatNear(1e-4f),
                                 absl::MakeConstSpan(prefill_out).last(kOut)));
}

TEST(CompiledMlDriftModelRunnerTest,
     SharesWeightsWhenBothRunnersCallPrepareBeforeBuildModel) {
  constexpr int kPrefillLen = 3;
  constexpr int kIn = 8;
  constexpr int kOut = 4;
  std::vector<float> fc_weights(kOut * kIn, 0.25f);
  std::vector<float> scale_weights(kOut, 1.5f);

  MlDriftClBackend::Environment shared_env;
  ASSERT_THAT(MlDriftClBackend::InitEnvironment(&shared_env), IsOk());
  Runner::SharedWeightsCache shared_weights;

  TensorMLD w({.name = "fc.weight",
               .type = Type::kFP32,
               .shape = {kOut, kIn},
               .buffer = fc_weights});
  TensorMLD scale({.name = "norm.weight",
                   .type = Type::kFP32,
                   .shape = {1, 1, kOut},
                   .buffer = scale_weights});

  TensorMLD x_prefill(
      {.name = "x", .type = Type::kFP32, .shape = {1, kPrefillLen, kIn}});
  TensorMLD y_prefill = Mul(FullyConnected(x_prefill, w), scale);
  y_prefill.SetName("y");

  TensorMLD x_decode({.name = "x", .type = Type::kFP32, .shape = {1, 1, kIn}});
  TensorMLD y_decode = Mul(FullyConnected(x_decode, w), scale);
  y_decode.SetName("y");

  Runner prefill_runner({x_prefill}, {y_prefill},
                        ml_drift::CalculationsPrecision::kF32, &shared_env,
                        &shared_weights);
  Runner decode_runner({x_decode}, {y_decode},
                       ml_drift::CalculationsPrecision::kF32, &shared_env,
                       &shared_weights);

  // Call Prepare() on BOTH runners before calling BuildModel() on either.
  ASSERT_THAT(prefill_runner.Prepare(), IsOk());
  EXPECT_EQ(shared_weights.descriptors.size(), 2u);
  ASSERT_THAT(decode_runner.Prepare(), IsOk());

  ASSERT_THAT(prefill_runner.BuildModel(), IsOk());
  ASSERT_THAT(decode_runner.BuildModel(), IsOk());
  EXPECT_EQ(shared_weights.cache_misses, 2u);
  EXPECT_EQ(shared_weights.cache_hits, 2u);
}

// Regression test for large vocabulary / projection matrices (such as
// Qwen3.5-27B's [248320, 5120] lm_head, 1.27B elements = 5.08B INT4 bits):
// `elements_count * SizeInBitsOf(desc.type)` exceeds INT32_MAX (and UINT32_MAX
// for 1B+ elements) and must use 64-bit `size_t` arithmetic.
TEST(CompiledMlDriftModelRunnerTest,
     LargeConstantInt4FullyConnectedBitCountExceedingInt32Max) {
  constexpr int kOut = 16777216;
  constexpr int kIn = 32;  // 536,870,912 elements -> 2,147,483,648 bits.
  const size_t num_elements = static_cast<size_t>(kOut) * kIn;
  std::shared_ptr<OwningCpuBuffer> w_buf =
      OwningCpuBuffer::Allocate<Type::kI4>(num_elements);
  // Set row 0 to +1 across all 32 inputs and row `kOut - 1` to -2 across all
  // 32 inputs; leave the interior rows at 0.
  int4_t* packed = w_buf->Span<int4_t>().data();
  for (int j = 0; j < kIn / 2; ++j) {
    packed[j].a = 1;
    packed[j].b = 1;
    const size_t last_row_offset =
        static_cast<size_t>(kOut - 1) * (kIn / 2) + j;
    packed[last_row_offset].a = -2;
    packed[last_row_offset].b = -2;
  }
  std::vector<float> scales(kOut, 0.5f);
  std::vector<float> input(kIn, 1.0f);

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 1, kIn}});
  TensorMLD w({.name = "large_lm_head.weight",
               .type = Type::kI4,
               .shape = {kOut, kIn},
               .buffer = w_buf,
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   scales, std::vector<int64_t>(kOut, 0),
                   /*quantized_dimension=*/0)});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");

  Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, 1, kIn},
                 .buffer = input});
  ASSERT_THAT(runner.SetInput("x", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("y"));
  ASSERT_EQ(actual.size(), static_cast<size_t>(kOut));
  EXPECT_NEAR(actual[0], 16.0f, 1e-3f);
  EXPECT_NEAR(actual[kOut / 2], 0.0f, 1e-3f);
  EXPECT_NEAR(actual[kOut - 1], -32.0f, 1e-3f);
}

TEST(CompiledMlDriftModelRunnerTest, RecordsCompileTimingAndWeightPackStats) {
  constexpr int kIn = 8;
  constexpr int kOut = 4;
  std::vector<float> fc_weights(kOut * kIn, 0.25f);
  TensorMLD w({.name = "fc.weight",
               .type = Type::kFP32,
               .shape = {kOut, kIn},
               .buffer = fc_weights});

  // Without SharedWeightsCache: weight_pack_stats still records non-zero bytes.
  {
    TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2, kIn}});
    TensorMLD y = FullyConnected(x, w);
    y.SetName("y");

    Runner runner({x}, {y});
    ASSERT_THAT(runner.BuildModel(), IsOk());

    const Runner::CompileTiming& timing = runner.compile_timing();
    EXPECT_GE(timing.rearrange_weights, absl::ZeroDuration());
    EXPECT_GE(timing.lower_ops, absl::ZeroDuration());
    EXPECT_GE(timing.build_gpu_model, absl::ZeroDuration());
    EXPECT_GE(timing.upload_shared_weights, absl::ZeroDuration());
    EXPECT_EQ(timing.upload_shared_weights_bytes, 0u);
    EXPECT_GE(timing.init_inference_context, absl::ZeroDuration());
    EXPECT_GT(timing.num_nodes, 0u);

    ASSERT_TRUE(runner.weight_pack_stats().contains("fp"));
    EXPECT_EQ(runner.weight_pack_stats().at("fp").count, 1);
    EXPECT_GT(runner.weight_pack_stats().at("fp").bytes, 0u);

    const std::string timing_str = runner.CompileTimingString();
    EXPECT_THAT(timing_str, testing::HasSubstr("rearrange_weights_ms="));
    EXPECT_THAT(timing_str, testing::HasSubstr("num_nodes="));
    EXPECT_THAT(timing_str, testing::HasSubstr("pack_fp=("));
    EXPECT_THAT(timing_str, testing::Not(testing::HasSubstr("mb=0,")));
  }

  // With SharedWeightsCache: first build packs and uploads; second reuses.
  {
    MlDriftClBackend::Environment shared_env;
    ASSERT_THAT(MlDriftClBackend::InitEnvironment(&shared_env), IsOk());
    Runner::SharedWeightsCache shared_weights;

    TensorMLD x1({.name = "x", .type = Type::kFP32, .shape = {1, 2, kIn}});
    TensorMLD y1 = FullyConnected(x1, w);
    y1.SetName("y");
    Runner first({x1}, {y1}, ml_drift::CalculationsPrecision::kF32, &shared_env,
                 &shared_weights);
    ASSERT_THAT(first.BuildModel(), IsOk());
    ASSERT_TRUE(first.weight_pack_stats().contains("fp"));
    EXPECT_EQ(first.weight_pack_stats().at("fp").count, 1);
    EXPECT_GT(first.weight_pack_stats().at("fp").bytes, 0u);
    EXPECT_GT(first.compile_timing().upload_shared_weights_bytes, 0u);
    EXPECT_GT(first.compile_timing().num_nodes, 0u);
    EXPECT_THAT(first.CompileTimingString(), testing::HasSubstr("pack_fp=("));

    TensorMLD x2({.name = "x", .type = Type::kFP32, .shape = {1, 1, kIn}});
    TensorMLD y2 = FullyConnected(x2, w);
    y2.SetName("y");
    Runner second({x2}, {y2}, ml_drift::CalculationsPrecision::kF32,
                  &shared_env, &shared_weights);
    ASSERT_THAT(second.BuildModel(), IsOk());
    EXPECT_TRUE(second.weight_pack_stats().empty());
    EXPECT_EQ(second.compile_timing().upload_shared_weights_bytes, 0u);
    EXPECT_GT(second.compile_timing().num_nodes, 0u);
  }
}

TEST(CompiledMlDriftModelRunnerTest, GetTensorDescriptor) {
  constexpr int kIn = 8;
  constexpr int kOut = 4;
  constexpr int kTokens = 2;
  TensorMLD w({.name = "fc.weight",
               .type = Type::kFP32,
               .shape = {kOut, kIn},
               .buffer = std::vector<float>(kOut * kIn, 0.5f)});
  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, kTokens, kIn}});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");
  Runner runner({x}, {y});
  ASSERT_THAT(runner.Prepare(), IsOk());

  ASSERT_OK_AND_ASSIGN(const ml_drift::TensorDescriptor input,
                       runner.GetTensorDescriptor("x"));
  EXPECT_EQ(input.GetBHWCShape().DimensionsProduct(), kTokens * kIn);
  EXPECT_TRUE(input.GetData().empty());

  ASSERT_OK_AND_ASSIGN(const ml_drift::TensorDescriptor output,
                       runner.GetTensorDescriptor("y"));
  EXPECT_EQ(output.GetBHWCShape().DimensionsProduct(), kTokens * kOut);
  EXPECT_TRUE(output.GetData().empty());

  // Constant weights are found too, without their packed data.
  ASSERT_OK_AND_ASSIGN(const ml_drift::TensorDescriptor weights,
                       runner.GetTensorDescriptor("fc.weight"));
  EXPECT_TRUE(weights.GetData().empty());

  EXPECT_EQ(runner.GetTensorDescriptor("missing").status().code(),
            absl::StatusCode::kNotFound);
}

// `y = FullyConnected(x, w)` with constant FP32 weights "fc.weight".
struct FcGraph {
  static constexpr int kIn = 8;
  static constexpr int kOut = 4;
  static constexpr int kTokens = 2;
  TensorMLD x;
  TensorMLD y;
};

FcGraph MakeFcGraph() {
  std::vector<float> weights(FcGraph::kOut * FcGraph::kIn);
  for (int i = 0; i < weights.size(); ++i) {
    weights[i] = 0.125f * static_cast<float>(i % 11 - 5);
  }
  FcGraph g;
  TensorMLD w({.name = "fc.weight",
               .type = Type::kFP32,
               .shape = {FcGraph::kOut, FcGraph::kIn},
               .buffer = weights});
  g.x = TensorMLD({.name = "x",
                   .type = Type::kFP32,
                   .shape = {1, FcGraph::kTokens, FcGraph::kIn}});
  g.y = FullyConnected(g.x, w);
  g.y.SetName("y");
  return g;
}

absl::StatusOr<std::vector<float>> ComputeFcOutput(Runner& runner) {
  std::vector<float> x_data(FcGraph::kTokens * FcGraph::kIn);
  for (int i = 0; i < x_data.size(); ++i) {
    x_data[i] = 0.5f * static_cast<float>(i % 7 - 3);
  }
  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, FcGraph::kTokens, FcGraph::kIn},
                 .buffer = x_data});
  LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput("x", fed));
  LRT_TENSOR_RETURN_IF_ERROR(runner.Run());
  return runner.GetFloatOutput("y");
}

// `SharedWeightsCache::on_upload` gets the packed weights (with data) of
// every shared weight uploaded on the CPU path, once per cache.
TEST(CompiledMlDriftModelRunnerTest, OnUploadGetsThePackedSharedWeights) {
  MlDriftClBackend::Environment env;
  ASSERT_THAT(MlDriftClBackend::InitEnvironment(&env), IsOk());
  Runner::SharedWeightsCache shared_weights;
  absl::flat_hash_map<std::string, ml_drift::TensorDescriptor> uploaded;
  int num_uploads = 0;
  shared_weights.on_upload = [&](const std::string& name,
                                 const ml_drift::TensorDescriptor& desc) {
    ++num_uploads;
    uploaded.insert_or_assign(name, desc);
  };
  FcGraph g1 = MakeFcGraph();
  Runner first({g1.x}, {g1.y}, ml_drift::CalculationsPrecision::kF32, &env,
               &shared_weights);
  ASSERT_THAT(first.BuildModel(), IsOk());
  ASSERT_TRUE(uploaded.contains("fc.weight"));
  const ml_drift::TensorDescriptor& packed = uploaded.at("fc.weight");
  EXPECT_FALSE(packed.GetData().empty());
  EXPECT_GE(packed.GetData().size(), packed.GetMemorySizeInBytes());
  const int first_uploads = num_uploads;

  FcGraph g2 = MakeFcGraph();
  Runner second({g2.x}, {g2.y}, ml_drift::CalculationsPrecision::kF32, &env,
                &shared_weights);
  ASSERT_THAT(second.BuildModel(), IsOk());
  EXPECT_EQ(num_uploads, first_uploads);
}

// Shared weights given as `SharedWeightsCache::host_data` (e.g. mapped from
// a persistent weight cache) are created from that memory: copied, or wrapped
// with `zero_copy_host_data` when the data is aligned for the GPU.
class HostDataTest : public ::testing::TestWithParam<
                         std::tuple</*zero_copy=*/bool, /*misaligned=*/bool>> {
};

TEST_P(HostDataTest, CreatesSharedWeightsFromHostData) {
  const auto [zero_copy, misaligned] = GetParam();
  MlDriftClBackend::Environment env;
  ASSERT_THAT(MlDriftClBackend::InitEnvironment(&env), IsOk());
  const size_t base_alignment =
      env.device().GetInfo().opencl_info.base_addr_align_in_bits / 8;
  if (misaligned && base_alignment <= 1) {
    GTEST_SKIP() << "Every address is aligned on this GPU";
  }

  // Reference run, which also gets the packed weights.
  Runner::SharedWeightsCache reference_weights;
  std::optional<ml_drift::TensorDescriptor> packed;
  reference_weights.on_upload = [&](const std::string& name,
                                    const ml_drift::TensorDescriptor& desc) {
    if (name == "fc.weight") packed = desc;
  };
  FcGraph g1 = MakeFcGraph();
  Runner reference({g1.x}, {g1.y}, ml_drift::CalculationsPrecision::kF32, &env,
                   &reference_weights);
  ASSERT_THAT(reference.BuildModel(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const std::vector<float> expected,
                                  ComputeFcOutput(reference));
  ASSERT_TRUE(packed.has_value());

  // The packed bytes in host memory aligned to 4 KiB (plus 1 byte when
  // `misaligned`), owned by the cache through `HostData::owner`.
  constexpr size_t kAlignment = 4096;
  const absl::Span<const uint8_t> bytes = packed->GetData();
  auto memory =
      std::make_shared<std::vector<uint8_t>>(bytes.size() + 2 * kAlignment);
  const uintptr_t base = reinterpret_cast<uintptr_t>(memory->data());
  uint8_t* aligned =
      memory->data() + (kAlignment - base % kAlignment) + (misaligned ? 1 : 0);
  absl::c_copy(bytes, aligned);

  Runner::SharedWeightsCache shared_weights;
  shared_weights.zero_copy_host_data = zero_copy;
  shared_weights.descriptors.emplace("fc.weight",
                                     CopyDescriptorWithoutData(*packed));
  shared_weights.host_data.emplace(
      "fc.weight",
      Runner::SharedWeightsCache::HostData{
          .data = absl::MakeConstSpan(aligned, bytes.size()), .owner = memory});
  memory.reset();
  FcGraph g2 = MakeFcGraph();
  Runner runner({g2.x}, {g2.y}, ml_drift::CalculationsPrecision::kF32, &env,
                &shared_weights);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  EXPECT_TRUE(shared_weights.host_data.empty());
  EXPECT_TRUE(shared_weights.tensors.contains("fc.weight"));
  EXPECT_EQ(runner.compile_timing().num_zero_copy_tensors,
            zero_copy && !misaligned ? 1u : 0u);
  EXPECT_EQ(shared_weights.zero_copy_owners.size(),
            zero_copy && !misaligned ? 1u : 0u);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const std::vector<float> actual,
                                  ComputeFcOutput(runner));
  EXPECT_THAT(actual, testing::ElementsAreArray(expected));
}

INSTANTIATE_TEST_SUITE_P(
    ZeroCopy, HostDataTest,
    ::testing::Values(std::make_tuple(false, false),
                      std::make_tuple(true, false),
                      std::make_tuple(true, true)),
    [](const ::testing::TestParamInfo<std::tuple<bool, bool>>& info) {
      return std::string(std::get<0>(info.param) ? "ZeroCopy" : "Copy") +
             (std::get<1>(info.param) ? "Misaligned" : "Aligned");
    });

// After `BuildModel()` frees the model builder, the metadata queries use what
// `CaptureCompiledHandlesAndReleaseBuilder()` kept, and `BuildModel()` can't
// run again.
TEST(CompiledMlDriftModelRunnerTest, MetadataQueriesWorkAfterBuildModel) {
  FcGraph g = MakeFcGraph();
  // An input that no op uses: known to the runner but without a GPU tensor.
  TensorMLD unused({.name = "unused", .type = Type::kFP32, .shape = {1, 4}});
  Runner runner({g.x, unused}, {g.y});
  ASSERT_THAT(runner.Prepare(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const ml_drift::ValueId x_id_before,
                                  runner.GetTensorValueId("x"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const ml_drift::ValueId w_id_before,
                                  runner.GetWeightValueId("fc.weight"));
  ASSERT_THAT(runner.BuildModel(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const ml_drift::ValueId x_id,
                                  runner.GetTensorValueId("x"));
  EXPECT_EQ(x_id, x_id_before);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const ml_drift::TensorDescriptor y_desc,
                                  runner.GetTensorDescriptor("y"));
  EXPECT_EQ(y_desc.GetBHWCShape().DimensionsProduct(),
            FcGraph::kTokens * FcGraph::kOut);
  EXPECT_TRUE(y_desc.GetData().empty());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const ml_drift::ValueId w_id,
                                  runner.GetWeightValueId("fc.weight"));
  EXPECT_EQ(w_id, w_id_before);
  EXPECT_TRUE(runner.GetWeightTensors().contains(w_id));
  EXPECT_EQ(runner.GetTensorValueId("missing").status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(runner.GetWeightValueId("missing").status().code(),
            absl::StatusCode::kNotFound);

  // Setting the unused input is a no-op; unknown names are errors.
  TensorMLD unused_value({.name = "unused",
                          .type = Type::kFP32,
                          .shape = {1, 4},
                          .buffer = std::vector<float>(4, 1.0f)});
  EXPECT_THAT(runner.SetInput("unused", unused_value), IsOk());
  EXPECT_EQ(runner.SetInput("missing", unused_value).code(),
            absl::StatusCode::kInvalidArgument);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const std::vector<float> output,
                                  ComputeFcOutput(runner));
  EXPECT_EQ(output.size(), FcGraph::kTokens * FcGraph::kOut);

  EXPECT_EQ(runner.BuildModel().code(), absl::StatusCode::kFailedPrecondition);
}

// With `release_fused_host_buffers`, `BuildModel()` drops the packed data of
// the shared descriptors whose GPU tensors exist (e.g. descriptors loaded from
// a persistent weight cache for weights another runner already uploaded);
// otherwise it keeps it.
TEST(CompiledMlDriftModelRunnerTest, ReleasesUploadedSharedDescriptorData) {
  MlDriftClBackend::Environment env;
  ASSERT_THAT(MlDriftClBackend::InitEnvironment(&env), IsOk());
  for (const bool release : {false, true}) {
    Runner::SharedWeightsCache shared_weights;
    std::optional<ml_drift::TensorDescriptor> packed;
    shared_weights.on_upload = [&](const std::string& name,
                                   const ml_drift::TensorDescriptor& desc) {
      if (name == "fc.weight") packed = desc;
    };
    FcGraph g1 = MakeFcGraph();
    Runner first({g1.x}, {g1.y}, ml_drift::CalculationsPrecision::kF32, &env,
                 &shared_weights);
    ASSERT_THAT(first.BuildModel(), IsOk());
    ASSERT_TRUE(packed.has_value());
    ASSERT_TRUE(shared_weights.tensors.contains("fc.weight"));
    // The packed data is back in the cache, and the GPU tensor exists.
    shared_weights.descriptors.insert_or_assign("fc.weight", *packed);
    shared_weights.release_fused_host_buffers = release;

    FcGraph g2 = MakeFcGraph();
    Runner second({g2.x}, {g2.y}, ml_drift::CalculationsPrecision::kF32, &env,
                  &shared_weights);
    ASSERT_THAT(second.BuildModel(), IsOk());
    ASSERT_TRUE(shared_weights.descriptors.contains("fc.weight"));
    EXPECT_EQ(shared_weights.descriptors.at("fc.weight").GetData().empty(),
              release);
  }
}

TEST(CompiledMlDriftModelRunnerTest,
     TiedEmbeddingLookupAndFullyConnectedShareQuantizedWeightsForCustomNames) {
  using TextureRunner =
      CompiledMlDriftModelRunner<MlDriftClTextureWeightsBackend>;
  constexpr int kVocab = 12;
  constexpr int kDim = 64;
  constexpr int kBlockSize = 32;
  constexpr int kBlocksPerRow = kDim / kBlockSize;

  auto q4_buf = OwningCpuBuffer::Allocate<Type::kI4>(kVocab * kDim);
  int4_t* q4_packed = q4_buf->Span<int4_t>().data();
  std::vector<float> q4_scales(kVocab * kBlocksPerRow);
  std::vector<float> q4_zps(kVocab * kBlocksPerRow);
  std::vector<float> q4_ref(kVocab * kDim);

  std::vector<int8_t> q8_data(kVocab * kDim);
  std::vector<float> q8_scales(kVocab * kBlocksPerRow);
  std::vector<float> q8_ref(kVocab * kDim);

  for (int v = 0; v < kVocab; ++v) {
    for (int b = 0; b < kBlocksPerRow; ++b) {
      const int blk = v * kBlocksPerRow + b;
      q4_scales[blk] = 0.03125f * static_cast<float>((blk % 4) + 1);
      q4_zps[blk] = (blk % 2 == 0) ? -2.0f : 1.5f;
      q8_scales[blk] = 0.015625f * static_cast<float>((blk % 5) + 1);
      for (int j = 0; j < kBlockSize; j += 2) {
        const int idx = v * kDim + b * kBlockSize + j;
        const int qa = ((idx * 3 + 1) % 16) - 8;
        const int qb = ((idx * 5 + 4) % 16) - 8;
        q4_packed[idx / 2].a = static_cast<int8_t>(qa);
        q4_packed[idx / 2].b = static_cast<int8_t>(qb);
        q4_ref[idx] = q4_scales[blk] * (static_cast<float>(qa) - q4_zps[blk]);
        q4_ref[idx + 1] =
            q4_scales[blk] * (static_cast<float>(qb) - q4_zps[blk]);
      }
      for (int j = 0; j < kBlockSize; ++j) {
        const int idx = v * kDim + b * kBlockSize + j;
        q8_data[idx] = static_cast<int8_t>((idx * 7 + 3) % 51 - 25);
        q8_ref[idx] = q8_scales[blk] * static_cast<float>(q8_data[idx]);
      }
    }
  }

  for (bool is_q4_affine : {true, false}) {
    MlDriftClTextureWeightsBackend::Environment shared_env;
    ASSERT_THAT(MlDriftClTextureWeightsBackend::InitEnvironment(&shared_env),
                IsOk());
    TextureRunner::SharedWeightsCache shared_weights;

    // Deliberately use a non-default tensor name (`custom_vocab.weight`, not
    // `token_embd.weight` or `embed_tokens`) to verify graph-level shared
    // quantized weight handling across `EmbeddingLookup` and `FullyConnected`
    // on Mali's 2D texture weights layout (`k2DYIsSpatialIOAndXIsOGroupI4O4`).
    auto make_emb_w = [&]() {
      return is_q4_affine
                 ? TensorMLD(
                       {.name = "custom_vocab.weight",
                        .type = Type::kI4,
                        .shape = {kVocab, kDim},
                        .buffer = q4_buf,
                        .quantization = std::make_shared<BlockwiseQuantization>(
                            q4_scales, ZeroPoints::Float(q4_zps), kBlockSize,
                            /*quantized_dimension=*/0)})
                 : TensorMLD(
                       {.name = "custom_vocab.weight",
                        .type = Type::kI8,
                        .shape = {kVocab, kDim},
                        .buffer = q8_data,
                        .quantization = std::make_shared<BlockwiseQuantization>(
                            q8_scales, std::vector<int64_t>{}, kBlockSize,
                            /*quantized_dimension=*/0)});
    };
    TensorMLD emb_w = make_emb_w();

    TensorMLD ids_prefill({.name = "ids", .type = Type::kI32, .shape = {1, 3}});
    TensorMLD y_prefill =
        FullyConnected(EmbeddingLookup(ids_prefill, emb_w), emb_w);
    y_prefill.SetName("logits");

    TextureRunner prefill_runner({ids_prefill}, {y_prefill},
                                 ml_drift::CalculationsPrecision::kF32,
                                 &shared_env, &shared_weights);
    ASSERT_THAT(prefill_runner.BuildModel(), IsOk());

    auto desc_it = shared_weights.descriptors.find("custom_vocab.weight");
    ASSERT_NE(desc_it, shared_weights.descriptors.end());
    EXPECT_EQ(desc_it->second.GetStorageType(),
              ml_drift::TensorStorageType::kTexture2D);
    EXPECT_GT(shared_weights.cache_misses, 0u);

    TensorMLD ids_decode({.name = "ids", .type = Type::kI32, .shape = {1, 1}});
    TensorMLD y_decode =
        FullyConnected(EmbeddingLookup(ids_decode, emb_w), emb_w);
    y_decode.SetName("logits");
    TextureRunner decode_runner({ids_decode}, {y_decode},
                                ml_drift::CalculationsPrecision::kF32,
                                &shared_env, &shared_weights);
    ASSERT_THAT(decode_runner.BuildModel(), IsOk());
    EXPECT_EQ(shared_weights.cache_hits, shared_weights.cache_misses);

    const std::vector<int32_t> tokens = {1, 7, 10};
    TensorMLD fed_prefill(
        {.name = "ids", .type = Type::kI32, .shape = {1, 3}, .buffer = tokens});
    ASSERT_THAT(prefill_runner.SetInput("ids", fed_prefill), IsOk());
    ASSERT_THAT(prefill_runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> prefill_logits,
                                    prefill_runner.GetFloatOutput("logits"));

    const std::vector<int32_t> last_tok = {tokens.back()};
    TensorMLD fed_decode({.name = "ids",
                          .type = Type::kI32,
                          .shape = {1, 1},
                          .buffer = last_tok});
    ASSERT_THAT(decode_runner.SetInput("ids", fed_decode), IsOk());
    ASSERT_THAT(decode_runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> decode_logits,
                                    decode_runner.GetFloatOutput("logits"));

    const std::vector<float>& w_ref = is_q4_affine ? q4_ref : q8_ref;
    std::vector<float> expected_last(kVocab, 0.0f);
    for (int v = 0; v < kVocab; ++v) {
      for (int d = 0; d < kDim; ++d) {
        expected_last[v] +=
            w_ref[tokens.back() * kDim + d] * w_ref[v * kDim + d];
      }
    }
    EXPECT_THAT(absl::MakeConstSpan(prefill_logits).last(kVocab),
                testing::Pointwise(testing::FloatNear(1e-3f), expected_last));
    EXPECT_THAT(decode_logits,
                testing::Pointwise(testing::FloatNear(1e-3f), expected_last));

    // Also verify the default buffer-preferring `MlDriftClBackend` path.
    MlDriftClBackend::Environment buf_env;
    ASSERT_THAT(MlDriftClBackend::InitEnvironment(&buf_env), IsOk());
    Runner::SharedWeightsCache buf_weights;
    TensorMLD buf_emb_w = make_emb_w();
    TensorMLD buf_ids({.name = "ids", .type = Type::kI32, .shape = {1, 1}});
    TensorMLD buf_y =
        FullyConnected(EmbeddingLookup(buf_ids, buf_emb_w), buf_emb_w);
    buf_y.SetName("logits");
    Runner buf_runner({buf_ids}, {buf_y}, ml_drift::CalculationsPrecision::kF32,
                      &buf_env, &buf_weights);
    ASSERT_THAT(buf_runner.BuildModel(), IsOk());
    auto buf_desc_it = buf_weights.descriptors.find("custom_vocab.weight");
    ASSERT_NE(buf_desc_it, buf_weights.descriptors.end());
    EXPECT_EQ(buf_desc_it->second.GetStorageType(),
              ml_drift::TensorStorageType::kBuffer);
    ASSERT_THAT(buf_runner.SetInput("ids", fed_decode), IsOk());
    ASSERT_THAT(buf_runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> buf_decode_logits,
                                    buf_runner.GetFloatOutput("logits"));
    EXPECT_THAT(buf_decode_logits,
                testing::Pointwise(testing::FloatNear(1e-3f), expected_last));
  }
}

// Verifies that `MlDriftClBackend::ConstTensorUploadBatch` produces exact
// results across multi-tensor coalesced staging, multi-chunk uploads of a
// tensor larger than the staging buffer, and async host-data writes for small
// tensors (< 64 KiB).
TEST(CompiledMlDriftModelRunnerTest,
     BatchedStagingAndAsyncHostDataUploadMatchesReference) {
  MlDriftClBackend::Environment env;
  ASSERT_THAT(MlDriftClBackend::InitEnvironment(&env), IsOk());

  constexpr int kTokens = 2;
  constexpr int kDim = 256;
  constexpr int kBigOut = 256;  // 256*256*4 = 256 KiB (>= 64 KiB staging).
  constexpr int kSmallOut = 32;  // 32*256*4 = 32 KiB (< 64 KiB async path).

  std::vector<float> w1_data(kBigOut * kDim);
  std::vector<float> w2_data(kBigOut * kDim);
  std::vector<float> w3_data(kSmallOut * kDim);
  for (size_t i = 0; i < w1_data.size(); ++i) {
    w1_data[i] =
        0.01f * static_cast<float>(static_cast<int>((i * 7 + 3) % 19) - 9);
    w2_data[i] =
        0.01f * static_cast<float>(static_cast<int>((i * 11 + 5) % 17) - 8);
  }
  for (size_t i = 0; i < w3_data.size(); ++i) {
    w3_data[i] =
        0.02f * static_cast<float>(static_cast<int>((i * 5 + 1) % 13) - 6);
  }

  auto make_graph = [&]() {
    TensorMLD x(
        {.name = "x", .type = Type::kFP32, .shape = {1, kTokens, kDim}});
    TensorMLD w1({.name = "fc1.weight",
                  .type = Type::kFP32,
                  .shape = {kBigOut, kDim},
                  .buffer = w1_data});
    TensorMLD w2({.name = "fc2.weight",
                  .type = Type::kFP32,
                  .shape = {kBigOut, kDim},
                  .buffer = w2_data});
    TensorMLD w3({.name = "fc3.weight",
                  .type = Type::kFP32,
                  .shape = {kSmallOut, kDim},
                  .buffer = w3_data});
    TensorMLD y = FullyConnected(FullyConnected(FullyConnected(x, w1), w2), w3);
    y.SetName("y");
    return std::make_pair(x, y);
  };

  std::vector<float> x_data(kTokens * kDim);
  for (size_t i = 0; i < x_data.size(); ++i) {
    x_data[i] = 0.1f * static_cast<float>(static_cast<int>(i % 11) - 5);
  }
  auto run_once = [&](Runner& runner) -> absl::StatusOr<std::vector<float>> {
    TensorMLD fed({.name = "x",
                   .type = Type::kFP32,
                   .shape = {1, kTokens, kDim},
                   .buffer = x_data});
    LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput("x", fed));
    LRT_TENSOR_RETURN_IF_ERROR(runner.Run());
    return runner.GetFloatOutput("y");
  };

  // Reference build capturing packed weights for all three layers.
  Runner::SharedWeightsCache ref_weights;
  absl::flat_hash_map<std::string, ml_drift::TensorDescriptor> packed;
  ref_weights.on_upload = [&](const std::string& name,
                              const ml_drift::TensorDescriptor& desc) {
    packed.insert_or_assign(name, desc);
  };
  auto [ref_x, ref_y] = make_graph();
  Runner ref_runner({ref_x}, {ref_y}, ml_drift::CalculationsPrecision::kF32,
                    &env, &ref_weights);
  ASSERT_THAT(ref_runner.BuildModel(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const std::vector<float> expected,
                                  run_once(ref_runner));
  ASSERT_EQ(packed.size(), 3u);
  ASSERT_TRUE(
      absl::c_all_of(expected, [](float v) { return std::isfinite(v); }));

  // Warm-cache host_data build (exercises batched staging + async upload).
  Runner::SharedWeightsCache host_weights;
  host_weights.zero_copy_host_data = false;
  for (const auto& [name, desc] : packed) {
    auto bytes = std::make_shared<std::vector<uint8_t>>(desc.GetData().begin(),
                                                        desc.GetData().end());
    host_weights.descriptors.emplace(name, CopyDescriptorWithoutData(desc));
    host_weights.host_data.emplace(
        name, Runner::SharedWeightsCache::HostData{
                  .data = absl::MakeConstSpan(*bytes), .owner = bytes});
  }
  auto [warm_x, warm_y] = make_graph();
  Runner warm_runner({warm_x}, {warm_y}, ml_drift::CalculationsPrecision::kF32,
                     &env, &host_weights);
  ASSERT_THAT(warm_runner.BuildModel(), IsOk());
  EXPECT_TRUE(host_weights.host_data.empty());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const std::vector<float> actual,
                                  run_once(warm_runner));
  EXPECT_THAT(actual, testing::ElementsAreArray(expected));

  // Directly verify multi-chunk staging (`StageBufferCopy` + `Flush` +
  // `Finish`) when a single tensor exceeds the staging buffer capacity
  // (64 KiB capacity for a 256 KiB tensor), independent of GPU vendor defaults.
  const ml_drift::TensorDescriptor& big_desc = packed.at("fc1.weight");
  ASSERT_GT(big_desc.GetMemorySizeInBytes(), 64u * 1024u);
  MlDriftClBackend::Tensor chunked_tensor;
  auto big_bytes = std::make_shared<std::vector<uint8_t>>(
      big_desc.GetData().begin(), big_desc.GetData().end());
  MlDriftClBackend::ConstTensorUploadBatch tiny_batch(
      &env, /*staging_capacity_bytes=*/64 * 1024);
  if (big_desc.GetStorageType() == ml_drift::TensorStorageType::kBuffer ||
      big_desc.GetStorageType() == ml_drift::TensorStorageType::kImageBuffer) {
    ASSERT_THAT(chunked_tensor.CreateFromDescriptor(
                    CopyDescriptorWithoutData(big_desc), env.context()),
                IsOk());
    ASSERT_THAT(tiny_batch.StageBufferCopy(
                    chunked_tensor.GetMemoryPtrForWriting(), big_bytes->data(),
                    big_desc.GetMemorySizeInBytes()),
                IsOk());
    ASSERT_THAT(tiny_batch.Flush(), IsOk());
  } else {
    std::shared_ptr<const void> keep_alive;
    ASSERT_THAT(MlDriftClBackend::CreateConstTensorFromHostData(
                    &env, CopyDescriptorWithoutData(big_desc),
                    absl::MakeConstSpan(*big_bytes), /*zero_copy=*/false,
                    &chunked_tensor, &keep_alive, big_bytes, &tiny_batch),
                IsOk());
  }
  ASSERT_THAT(tiny_batch.Finish(), IsOk());
  ml_drift::TensorDescriptor readback;
  ASSERT_THAT(
      MlDriftClBackend::TensorToDescriptor(&chunked_tensor, &env, &readback),
      IsOk());
  EXPECT_THAT(readback.GetData(), testing::ElementsAreArray(
                                      absl::MakeConstSpan(*big_bytes).first(
                                          big_desc.GetMemorySizeInBytes())));
}

}  // namespace
}  // namespace litert::tensor
