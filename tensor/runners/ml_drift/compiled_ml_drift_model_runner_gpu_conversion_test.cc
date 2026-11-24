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

// GPU tests of `CompiledMlDriftModelRunner::SetConvertWeightsOnGpu`: weights
// rearranged by the ML Drift conversion model on the GPU must hold the same
// bytes as the CPU-packed weights, and the model outputs must be identical.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/ml_drift/compiled_ml_drift_model_runner.h"
#include "tensor/runners/ml_drift/ml_drift_cl_model_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::testing::ElementsAreArray;
using TensorMLD = Tensor<MlDriftMixinTag>;

// Selects the 2D texture weights layout (`k2DYIsSpatialIOAndXIsOGroupI4O4`)
// for INT4 / INT2 weights on the host GPU, like Mali, PowerVR and Adreno < 740.
struct MlDriftClTextureWeightsBackend : public MlDriftClBackend {
  static ml_drift::GpuInfo GetGpuInfoForBuilder(Environment& env,
                                                bool env_initialized) {
    ml_drift::GpuInfo info =
        MlDriftClBackend::GetGpuInfoForBuilder(env, env_initialized);
    info.vendor = ml_drift::GpuVendor::kMali;
    return info;
  }
};

enum class Mode { kCpu, kGpu };

struct WeightsCase {
  std::string name;
  Type type = Type::kI4;
  int out = 0;
  int in = 0;
  int tokens = 2;
  bool allow_8bit_convs = false;
  bool zero_points = false;
  // EmbeddingLookup and FullyConnected tied to the same weights.
  bool tied_embedding = false;
  bool texture_backend = false;
  // GPU conversion expected (false: the runner keeps the CPU path).
  bool converts_on_gpu = true;
};

std::shared_ptr<OwningCpuBuffer> MakeWeights(const WeightsCase& c) {
  const int n = c.out * c.in;
  if (c.type == Type::kI8) {
    auto buf = OwningCpuBuffer::Allocate<Type::kI8>(n);
    int8_t* w = buf->Span<int8_t>().data();
    for (int k = 0; k < n; ++k)
      w[k] = static_cast<int8_t>((k * 37 + 11) % 255 - 127);
    return buf;
  }
  if (c.type == Type::kI4) {
    auto buf = OwningCpuBuffer::Allocate<Type::kI4>(n);
    int4_t* w = buf->Span<int4_t>().data();
    for (int k = 0; k < n / 2; ++k) {
      w[k].a = (k * 7 + 1) % 16 - 8;
      w[k].b = (k * 3 + 5) % 16 - 8;
    }
    return buf;
  }
  auto buf = OwningCpuBuffer::Allocate<Type::kI2>(n);
  int2_t* w = buf->Span<int2_t>().data();
  for (int k = 0; k < n / 4; ++k) {
    w[k].a = k % 4 - 2;
    w[k].b = (k + 1) % 4 - 2;
    w[k].c = (k * 3 + 2) % 4 - 2;
    w[k].d = (k * 5 + 3) % 4 - 2;
  }
  return buf;
}

struct RunResult {
  std::vector<float> output;
  // Readback of the GPU weights (and `sum_i`, if any).
  ml_drift::TensorDescriptor weights;
  std::optional<ml_drift::TensorDescriptor> sum_i;
  int gpu_converted = 0;
};

template <typename Backend>
absl::StatusOr<RunResult> RunConversion(const WeightsCase& c, Mode mode,
                                        typename Backend::Environment* env) {
  using Runner = CompiledMlDriftModelRunner<Backend>;
  typename Runner::SharedWeightsCache shared_weights;
  std::vector<float> scales(c.out);
  std::vector<int64_t> zero_points(c.out, 0);
  for (int o = 0; o < c.out; ++o) {
    scales[o] = 0.01f * static_cast<float>(o % 7 + 1);
    if (c.zero_points) zero_points[o] = o % 5 - 2;
  }
  const std::string weights_name = absl::StrCat("w_", c.name);
  TensorMLD w({.name = weights_name,
               .type = c.type,
               .shape = {c.out, c.in},
               .buffer = MakeWeights(c),
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   scales, zero_points, /*quantized_dimension=*/0)});
  TensorMLD x;
  TensorMLD y;
  std::vector<float> x_data;
  std::vector<int32_t> ids;
  if (c.tied_embedding) {
    x = TensorMLD({.name = "x", .type = Type::kI32, .shape = {1, c.tokens}});
    for (int t = 0; t < c.tokens; ++t) ids.push_back((t * 5 + 1) % c.out);
    y = FullyConnected(EmbeddingLookup(x, w), w);
  } else {
    x = TensorMLD(
        {.name = "x", .type = Type::kFP32, .shape = {1, c.tokens, c.in}});
    for (int k = 0; k < c.tokens * c.in; ++k) {
      x_data.push_back(0.05f * static_cast<float>((k * 13) % 41 - 20));
    }
    y = FullyConnected(x, w);
  }
  y.SetName("y");

  Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32, env,
                &shared_weights, c.allow_8bit_convs);
  if (mode == Mode::kGpu) runner.ForceGpuWeightConversionForTesting();
  RunResult result;
  LRT_TENSOR_RETURN_IF_ERROR(runner.BuildModel());
  if (auto it = runner.weight_pack_stats().find("gpu_convert");
      it != runner.weight_pack_stats().end()) {
    result.gpu_converted = it->second.count;
  }
  if (c.tied_embedding) {
    TensorMLD fed({.name = "x",
                   .type = Type::kI32,
                   .shape = {1, c.tokens},
                   .buffer = ids});
    LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput("x", fed));
  } else {
    TensorMLD fed({.name = "x",
                   .type = Type::kFP32,
                   .shape = {1, c.tokens, c.in},
                   .buffer = x_data});
    LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput("x", fed));
  }
  LRT_TENSOR_RETURN_IF_ERROR(runner.Run());
  LRT_TENSOR_ASSIGN_OR_RETURN(result.output, runner.GetFloatOutput("y"));
  auto weights_it = shared_weights.tensors.find(weights_name);
  if (weights_it == shared_weights.tensors.end()) {
    return absl::InternalError(
        absl::StrCat("Missing shared weights tensor: ", weights_name));
  }
  LRT_TENSOR_RETURN_IF_ERROR(Backend::TensorToDescriptor(
      weights_it->second.get(), env, &result.weights));
  if (auto it = shared_weights.tensors.find(weights_name + ".sum_i");
      it != shared_weights.tensors.end()) {
    result.sum_i.emplace();
    LRT_TENSOR_RETURN_IF_ERROR(
        Backend::TensorToDescriptor(it->second.get(), env, &*result.sum_i));
  }
  return result;
}

// Values of a linear INT32 tensor read back from the GPU (buffer or texture
// storage, which hold the values in order).
std::vector<int32_t> LinearInt32Values(const ml_drift::TensorDescriptor& desc,
                                       int count) {
  const absl::Span<const uint8_t> data = desc.GetData();
  std::vector<int32_t> values(count, 0);
  std::copy_n(data.data(), std::min(data.size(), sizeof(int32_t) * count),
              reinterpret_cast<uint8_t*>(values.data()));
  return values;
}

class GpuWeightConversionTest : public ::testing::TestWithParam<WeightsCase> {};

TEST_P(GpuWeightConversionTest, MatchesCpuPackedWeights) {
  const WeightsCase& c = GetParam();
  absl::StatusOr<RunResult> cpu_result;
  absl::StatusOr<RunResult> gpu_result;
  if (c.texture_backend) {
    MlDriftClTextureWeightsBackend::Environment env;
    ASSERT_THAT(MlDriftClTextureWeightsBackend::InitEnvironment(&env), IsOk());
    cpu_result =
        RunConversion<MlDriftClTextureWeightsBackend>(c, Mode::kCpu, &env);
    gpu_result =
        RunConversion<MlDriftClTextureWeightsBackend>(c, Mode::kGpu, &env);
  } else {
    MlDriftClBackend::Environment env;
    ASSERT_THAT(MlDriftClBackend::InitEnvironment(&env), IsOk());
    cpu_result = RunConversion<MlDriftClBackend>(c, Mode::kCpu, &env);
    gpu_result = RunConversion<MlDriftClBackend>(c, Mode::kGpu, &env);
  }
  ASSERT_TRUE(cpu_result.ok()) << cpu_result.status();
  ASSERT_TRUE(gpu_result.ok()) << gpu_result.status();
  const RunResult& cpu = *cpu_result;
  const RunResult& gpu = *gpu_result;
  EXPECT_EQ(cpu.gpu_converted, 0);
  EXPECT_EQ(gpu.gpu_converted, c.converts_on_gpu ? 1 : 0);
  // Identical packed bytes. The CPU path may allocate a longer linear buffer
  // (it sizes INT4 / INT2 buffers in elements); the extra tail is padding the
  // kernels don't read.
  EXPECT_EQ(cpu.weights.GetStorageType(), gpu.weights.GetStorageType());
  const absl::Span<const uint8_t> cpu_bytes = cpu.weights.GetData();
  const absl::Span<const uint8_t> gpu_bytes = gpu.weights.GetData();
  ASSERT_FALSE(gpu_bytes.empty());
  ASSERT_LE(gpu_bytes.size(), cpu_bytes.size());
  EXPECT_THAT(gpu_bytes,
              ElementsAreArray(cpu_bytes.begin(),
                               cpu_bytes.begin() + gpu_bytes.size()));
  ASSERT_EQ(cpu.sum_i.has_value(), gpu.sum_i.has_value());
  if (cpu.sum_i.has_value()) {
    EXPECT_THAT(LinearInt32Values(*gpu.sum_i, c.out),
                ElementsAreArray(LinearInt32Values(*cpu.sum_i, c.out)));
  }
  // Bitwise identical outputs.
  ASSERT_EQ(cpu.output.size(), gpu.output.size());
  EXPECT_THAT(gpu.output, ElementsAreArray(cpu.output));
}

INSTANTIATE_TEST_SUITE_P(
    Weights, GpuWeightConversionTest,
    ::testing::Values(
        WeightsCase{.name = "int4", .type = Type::kI4, .out = 12, .in = 64},
        WeightsCase{.name = "int4_texture",
                    .type = Type::kI4,
                    .out = 8,
                    .in = 32,
                    .texture_backend = true},
        // Long prefill with INT8 activations: `sum_i` converted on the GPU.
        WeightsCase{.name = "int4_sum_i_prefill",
                    .type = Type::kI4,
                    .out = 12,
                    .in = 64,
                    .tokens = 128,
                    .allow_8bit_convs = true,
                    .zero_points = true},
        // INT4 `sum_i` requires `i % 8 == 0 && o % 4 == 0`; other shapes keep
        // the CPU path.
        WeightsCase{.name = "int4_sum_i_in_not_div8",
                    .type = Type::kI4,
                    .out = 8,
                    .in = 12,
                    .tokens = 128,
                    .allow_8bit_convs = true,
                    .converts_on_gpu = false},
        WeightsCase{.name = "int4_sum_i_out_not_div4",
                    .type = Type::kI4,
                    .out = 5,
                    .in = 32,
                    .tokens = 128,
                    .allow_8bit_convs = true,
                    .converts_on_gpu = false},
        WeightsCase{.name = "int4_tied_embedding",
                    .type = Type::kI4,
                    .out = 8,
                    .in = 32,
                    .allow_8bit_convs = true,
                    .tied_embedding = true},
        WeightsCase{.name = "int8", .type = Type::kI8, .out = 10, .in = 24},
        WeightsCase{
            .name = "int8_in_not_div4", .type = Type::kI8, .out = 8, .in = 10},
        WeightsCase{.name = "int8_prefill",
                    .type = Type::kI8,
                    .out = 20,
                    .in = 64,
                    .tokens = 128,
                    .allow_8bit_convs = true},
        WeightsCase{.name = "int8_texture",
                    .type = Type::kI8,
                    .out = 8,
                    .in = 32,
                    .texture_backend = true},
        WeightsCase{.name = "int2", .type = Type::kI2, .out = 8, .in = 32},
        WeightsCase{.name = "int2_texture",
                    .type = Type::kI2,
                    .out = 8,
                    .in = 32,
                    .texture_backend = true},
        // INT4 rows that are not a multiple of 4 values keep the CPU path.
        WeightsCase{.name = "int4_odd_rows",
                    .type = Type::kI4,
                    .out = 6,
                    .in = 6,
                    .converts_on_gpu = false}),
    [](const ::testing::TestParamInfo<WeightsCase>& info) {
      return info.param.name;
    });

// Prefill and decode runners sharing a `SharedWeightsCache`: the decode runner
// reuses the GPU-converted weights of the prefill runner, and `Prepare()` does
// not publish dataless descriptors before `BuildModel()` converts them.
TEST(GpuWeightConversionSharingTest, DecodeReusesConvertedWeights) {
  using Runner = CompiledMlDriftModelRunner<MlDriftClBackend>;
  constexpr int kIn = 32;
  constexpr int kOut = 8;
  const WeightsCase c{
      .name = "shared", .type = Type::kI4, .out = kOut, .in = kIn};
  MlDriftClBackend::Environment env;
  ASSERT_THAT(MlDriftClBackend::InitEnvironment(&env), IsOk());
  Runner::SharedWeightsCache shared_weights;
  shared_weights.release_fused_host_buffers = true;
  TensorMLD w({.name = "w1+w2",
               .type = Type::kI4,
               .shape = {kOut, kIn},
               .buffer = MakeWeights(c),
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   std::vector<float>(kOut, 0.1f),
                   std::vector<int64_t>(kOut, 0), /*quantized_dimension=*/0)});
  TensorMLD x_prefill({.name = "x", .type = Type::kFP32, .shape = {1, 3, kIn}});
  TensorMLD y_prefill = FullyConnected(x_prefill, w);
  y_prefill.SetName("y");
  TensorMLD x_decode({.name = "x", .type = Type::kFP32, .shape = {1, 1, kIn}});
  TensorMLD y_decode = FullyConnected(x_decode, w);
  y_decode.SetName("y");

  Runner prefill({x_prefill}, {y_prefill},
                 ml_drift::CalculationsPrecision::kF32, &env, &shared_weights);
  prefill.SetConvertWeightsOnGpu(true);
  ASSERT_THAT(prefill.Prepare(), IsOk());
  EXPECT_FALSE(shared_weights.descriptors.contains("w1+w2"));
  ASSERT_THAT(prefill.BuildModel(), IsOk());
  EXPECT_EQ(prefill.weight_pack_stats().at("gpu_convert").count, 1);
  // Both `"w1+w2"` (converted on GPU) and its scales tensor are newly inserted.
  EXPECT_EQ(shared_weights.cache_misses, 2u);
  EXPECT_EQ(shared_weights.cache_hits, 0u);
  EXPECT_TRUE(shared_weights.descriptors.contains("w1+w2"));

  Runner decode({x_decode}, {y_decode}, ml_drift::CalculationsPrecision::kF32,
                &env, &shared_weights);
  decode.SetConvertWeightsOnGpu(true);
  ASSERT_THAT(decode.BuildModel(), IsOk());
  EXPECT_FALSE(decode.weight_pack_stats().contains("gpu_convert"));
  EXPECT_EQ(shared_weights.cache_misses, 2u);
  EXPECT_EQ(shared_weights.cache_hits, 2u);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto* prefill_w,
                                  prefill.GetBackendTensor("w1+w2"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto* decode_w,
                                  decode.GetBackendTensor("w1+w2"));
  EXPECT_EQ(prefill_w, decode_w);
}

// Two runners sharing a `SharedWeightsCache` that both run `Prepare()` (and
// so both register the weights for GPU conversion) before `BuildModel()`: the
// second one takes the weights the first one converted from the cache instead
// of converting them again, with the same outputs.
TEST(GpuWeightConversionSharingTest, SkipsWeightsConvertedSincePrepare) {
  using Runner = CompiledMlDriftModelRunner<MlDriftClBackend>;
  constexpr int kIn = 32;
  constexpr int kOut = 8;
  const WeightsCase c{
      .name = "both_prepared", .type = Type::kI4, .out = kOut, .in = kIn};
  MlDriftClBackend::Environment env;
  ASSERT_THAT(MlDriftClBackend::InitEnvironment(&env), IsOk());
  Runner::SharedWeightsCache shared_weights;
  TensorMLD w({.name = "w",
               .type = Type::kI4,
               .shape = {kOut, kIn},
               .buffer = MakeWeights(c),
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   std::vector<float>(kOut, 0.1f),
                   std::vector<int64_t>(kOut, 0), /*quantized_dimension=*/0)});
  TensorMLD x1({.name = "x", .type = Type::kFP32, .shape = {1, 1, kIn}});
  TensorMLD y1 = FullyConnected(x1, w);
  y1.SetName("y");
  TensorMLD x2({.name = "x", .type = Type::kFP32, .shape = {1, 1, kIn}});
  TensorMLD y2 = FullyConnected(x2, w);
  y2.SetName("y");
  Runner first({x1}, {y1}, ml_drift::CalculationsPrecision::kF32, &env,
               &shared_weights);
  Runner second({x2}, {y2}, ml_drift::CalculationsPrecision::kF32, &env,
                &shared_weights);
  first.SetConvertWeightsOnGpu(true);
  second.SetConvertWeightsOnGpu(true);
  ASSERT_THAT(first.Prepare(), IsOk());
  ASSERT_THAT(second.Prepare(), IsOk());
  ASSERT_TRUE(second.weight_pack_stats().contains("gpu_register"));
  EXPECT_EQ(second.weight_pack_stats().at("gpu_register").count, 1);

  ASSERT_THAT(first.BuildModel(), IsOk());
  ASSERT_TRUE(first.weight_pack_stats().contains("gpu_convert"));
  EXPECT_EQ(first.weight_pack_stats().at("gpu_convert").count, 1);
  const size_t misses = shared_weights.cache_misses;
  const size_t hits = shared_weights.cache_hits;

  ASSERT_THAT(second.BuildModel(), IsOk());
  EXPECT_FALSE(second.weight_pack_stats().contains("gpu_convert"));
  EXPECT_EQ(shared_weights.cache_misses, misses);
  EXPECT_GT(shared_weights.cache_hits, hits);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto* first_w, first.GetBackendTensor("w"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto* second_w, second.GetBackendTensor("w"));
  EXPECT_EQ(first_w, second_w);

  std::vector<float> x_data(kIn);
  for (int k = 0; k < kIn; ++k) {
    x_data[k] = 0.03f * static_cast<float>(k % 13 - 6);
  }
  std::vector<float> outputs[2];
  Runner* runners[2] = {&first, &second};
  for (int r = 0; r < 2; ++r) {
    TensorMLD fed({.name = "x",
                   .type = Type::kFP32,
                   .shape = {1, 1, kIn},
                   .buffer = x_data});
    ASSERT_THAT(runners[r]->SetInput("x", fed), IsOk());
    ASSERT_THAT(runners[r]->Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(outputs[r],
                                    runners[r]->GetFloatOutput("y"));
  }
  EXPECT_THAT(outputs[1], ElementsAreArray(outputs[0]));
}

// `SetConvertWeightsOnGpu(true)` respects
// `WeightsManager::IsGpuWeightsPreparationSupported` (disabled on Mali).
TEST(GpuWeightConversionSharingTest, RespectsGpuSupportCheck) {
  using TextureRunner =
      CompiledMlDriftModelRunner<MlDriftClTextureWeightsBackend>;
  constexpr int kIn = 32;
  constexpr int kOut = 8;
  const WeightsCase c{
      .name = "mali", .type = Type::kI4, .out = kOut, .in = kIn};
  MlDriftClTextureWeightsBackend::Environment env;
  ASSERT_THAT(MlDriftClTextureWeightsBackend::InitEnvironment(&env), IsOk());
  TensorMLD w({.name = "w",
               .type = Type::kI4,
               .shape = {kOut, kIn},
               .buffer = MakeWeights(c),
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   std::vector<float>(kOut, 0.1f),
                   std::vector<int64_t>(kOut, 0), /*quantized_dimension=*/0)});
  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2, kIn}});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");
  TextureRunner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32, &env);
  runner.SetConvertWeightsOnGpu(true);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  EXPECT_FALSE(runner.weight_pack_stats().contains("gpu_convert"));
}

// Multiple weights converted across separate GPU conversion batches match CPU.
TEST(GpuWeightConversionSharingTest, MultiBatchConversionMatchesCpu) {
  using Runner = CompiledMlDriftModelRunner<MlDriftClBackend>;
  constexpr int kDim = 32;
  const WeightsCase c0{
      .name = "b0", .type = Type::kI4, .out = kDim, .in = kDim};
  const WeightsCase c1{
      .name = "b1", .type = Type::kI8, .out = kDim, .in = kDim};
  const WeightsCase c2{
      .name = "b2", .type = Type::kI2, .out = kDim, .in = kDim};
  std::vector<float> x_data(2 * kDim);
  for (int k = 0; k < x_data.size(); ++k) {
    x_data[k] = 0.02f * static_cast<float>((k * 11) % 29 - 14);
  }
  std::vector<float> outputs[2];
  for (int gpu = 0; gpu < 2; ++gpu) {
    auto make_weight = [&](const std::string& name, const WeightsCase& wc) {
      return TensorMLD(
          {.name = name,
           .type = wc.type,
           .shape = {wc.out, wc.in},
           .buffer = MakeWeights(wc),
           .quantization = std::make_shared<PerChannelAffineQuantization>(
               std::vector<float>(wc.out, 0.05f),
               std::vector<int64_t>(wc.out, 0), /*quantized_dimension=*/0)});
    };
    TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2, kDim}});
    TensorMLD y =
        FullyConnected(FullyConnected(FullyConnected(x, make_weight("w0", c0)),
                                      make_weight("w1", c1)),
                       make_weight("w2", c2));
    y.SetName("y");
    Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
    if (gpu) {
      runner.ForceGpuWeightConversionForTesting();
      runner.SetMaxGpuWeightConversionBatchBytesForTesting(1);
    }
    ASSERT_THAT(runner.BuildModel(), IsOk());
    if (gpu) {
      ASSERT_TRUE(runner.weight_pack_stats().contains("gpu_convert"));
      EXPECT_EQ(runner.weight_pack_stats().at("gpu_convert").count, 3);
    }
    TensorMLD fed({.name = "x",
                   .type = Type::kFP32,
                   .shape = {1, 2, kDim},
                   .buffer = x_data});
    ASSERT_THAT(runner.SetInput("x", fed), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(outputs[gpu], runner.GetFloatOutput("y"));
  }
  EXPECT_THAT(outputs[1], ElementsAreArray(outputs[0]));
}

// Without a `SharedWeightsCache`, the runner owns the converted weights.
TEST(GpuWeightConversionSharingTest, WorksWithoutSharedWeightsCache) {
  using Runner = CompiledMlDriftModelRunner<MlDriftClBackend>;
  constexpr int kIn = 32;
  constexpr int kOut = 8;
  const WeightsCase c{.name = "own", .type = Type::kI8, .out = kOut, .in = kIn};
  std::vector<float> outputs[2];
  for (int gpu = 0; gpu < 2; ++gpu) {
    TensorMLD w(
        {.name = "w",
         .type = Type::kI8,
         .shape = {kOut, kIn},
         .buffer = MakeWeights(c),
         .quantization = std::make_shared<PerChannelAffineQuantization>(
             std::vector<float>(kOut, 0.1f), std::vector<int64_t>(kOut, 0),
             /*quantized_dimension=*/0)});
    TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2, kIn}});
    TensorMLD y = FullyConnected(x, w);
    y.SetName("y");
    Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
    if (gpu) runner.ForceGpuWeightConversionForTesting();
    ASSERT_THAT(runner.BuildModel(), IsOk());
    EXPECT_EQ(runner.weight_pack_stats().contains("gpu_convert"), gpu == 1);
    std::vector<float> x_data(2 * kIn);
    for (int k = 0; k < x_data.size(); ++k) x_data[k] = 0.01f * (k % 17 - 8);
    TensorMLD fed({.name = "x",
                   .type = Type::kFP32,
                   .shape = {1, 2, kIn},
                   .buffer = x_data});
    ASSERT_THAT(runner.SetInput("x", fed), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(outputs[gpu], runner.GetFloatOutput("y"));
  }
  EXPECT_THAT(outputs[1], ElementsAreArray(outputs[0]));
}

}  // namespace
}  // namespace litert::tensor
