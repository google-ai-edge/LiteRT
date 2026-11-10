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

#include "tensor/examples/gemma4/gemma4_weights.h"

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/examples/utils/safetensor_test_util.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::SizeIs;

constexpr char kProjectionName[] = "model.per_layer_model_projection.weight";

// Number of checkpoint tensors that are mapped once for the whole model.
constexpr int kNumGlobalWeights = 6;
// Number of checkpoint tensors that are mapped for each layer.
constexpr int kNumWeightsPerLayer = 17;

// Small config used to slice the per-layer model projection.
Config SmallConfig() {
  Config config = Config::E2B();
  config.num_layers = 3;
  config.per_layer_input_dim = 2;
  config.embed_dim = 3;
  return config;
}

// Returns `count` consecutive floats starting at `start`.
std::vector<float> Iota(int count, float start = 0) {
  std::vector<float> values(count);
  std::iota(values.begin(), values.end(), start);
  return values;
}

// Builds the combined per-layer model projection of `config` holding
// consecutive values, i.e. layer `l` holds the values starting at
// `l * per_layer_input_dim * embed_dim`.
TensorHandle CombinedProjection(const Config& config) {
  return TensorHandle(
      {.name = kProjectionName,
       .type = Type::kFP32,
       .shape = {config.num_layers, config.per_layer_input_dim,
                 config.embed_dim},
       .buffer = Iota(config.num_layers * config.per_layer_input_dim *
                      config.embed_dim)});
}

std::string LayerProjectionName(int layer) {
  return absl::StrCat("model.layers.", layer,
                      ".per_layer_model_projection.weight");
}

TEST(GetGemma4WeightMappingTest, MapsGlobalWeightsWithoutLayers) {
  absl::flat_hash_map<std::string, std::string> mapping =
      GetGemma4WeightMapping(/*n_layers=*/0);

  EXPECT_THAT(mapping, SizeIs(kNumGlobalWeights));
  EXPECT_THAT(mapping, Contains(Pair("model.language_model.embed_tokens.weight",
                                     "model.embed_tokens.weight")));
  EXPECT_THAT(mapping, Contains(Pair("lm_head.weight", "lm_head.weight")));
  EXPECT_THAT(mapping, Contains(Pair("model.language_model.norm.weight",
                                     "model.norm.weight")));
  EXPECT_THAT(mapping, Contains(Pair(
                           "model.language_model.embed_tokens_per_layer.weight",
                           "model.embed_tokens_per_layer.weight")));
  EXPECT_THAT(
      mapping,
      Contains(Pair("model.language_model.per_layer_model_projection.weight",
                    kProjectionName)));
  EXPECT_THAT(
      mapping,
      Contains(Pair("model.language_model.per_layer_projection_norm.weight",
                    "model.per_layer_projection_norm.weight")));
}

TEST(GetGemma4WeightMappingTest, MapsEveryLayerWeight) {
  constexpr int kNumLayers = 2;
  absl::flat_hash_map<std::string, std::string> mapping =
      GetGemma4WeightMapping(kNumLayers);

  EXPECT_THAT(mapping,
              SizeIs(kNumGlobalWeights + kNumLayers * kNumWeightsPerLayer));
  for (int layer = 0; layer < kNumLayers; ++layer) {
    SCOPED_TRACE(absl::StrCat("layer ", layer));
    for (const char* suffix : {
             ".self_attn.q_proj.weight",
             ".self_attn.k_proj.weight",
             ".self_attn.v_proj.weight",
             ".self_attn.o_proj.weight",
             ".self_attn.q_norm.weight",
             ".self_attn.k_norm.weight",
             ".mlp.gate_proj.weight",
             ".mlp.up_proj.weight",
             ".mlp.down_proj.weight",
             ".input_layernorm.weight",
             ".post_attention_layernorm.weight",
             ".pre_feedforward_layernorm.weight",
             ".post_feedforward_layernorm.weight",
             ".per_layer_input_gate.weight",
             ".per_layer_projection.weight",
             ".post_per_layer_input_norm.weight",
             ".layer_scalar",
         }) {
      EXPECT_THAT(mapping,
                  Contains(Pair(absl::StrCat("model.language_model.layers.",
                                             layer, suffix),
                                absl::StrCat("model.layers.", layer, suffix))));
    }
  }
  // Layers past `n_layers` aren't mapped.
  EXPECT_FALSE(mapping.contains(absl::StrCat(
      "model.language_model.layers.", kNumLayers, ".self_attn.q_proj.weight")));
}

TEST(Gemma4WeightHooksTest, LmHeadFallsBackToTiedEmbeddings) {
  TensorHandle embeddings({.name = "model.embed_tokens.weight",
                           .type = Type::kFP32,
                           .shape = {2, 2},
                           .buffer = std::vector<float>({1, 2, 3, 4})});
  LazyTensorMapping mapping =
      LazyTensorMapping({{"model.embed_tokens.weight", embeddings}})
          .Register<Gemma4WeightHooks>(SmallConfig());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle lm_head,
                                  mapping.Get("lm_head.weight"));
  EXPECT_EQ(lm_head.GetBufferPtr(), embeddings.GetBufferPtr());
  EXPECT_THAT(lm_head.GetShape(), ElementsAre(2, 2));
}

TEST(Gemma4WeightHooksTest, LmHeadFailsWithoutEmbeddings) {
  LazyTensorMapping mapping;
  mapping.Register<Gemma4WeightHooks>(SmallConfig());

  EXPECT_THAT(mapping.Get("lm_head.weight"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(Gemma4WeightHooksTest, SlicesPerLayerModelProjection) {
  const Config config = SmallConfig();
  LazyTensorMapping mapping =
      LazyTensorMapping({{kProjectionName, CombinedProjection(config)}})
          .Register<Gemma4WeightHooks>(config);

  const int layer_elements = config.per_layer_input_dim * config.embed_dim;
  for (int layer = 0; layer < config.num_layers; ++layer) {
    SCOPED_TRACE(absl::StrCat("layer ", layer));
    const std::string name = LayerProjectionName(layer);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle slice, mapping.Get(name));
    EXPECT_EQ(slice.GetName(), name);
    EXPECT_EQ(slice.GetType(), Type::kFP32);
    EXPECT_THAT(slice.GetShape(),
                ElementsAre(config.per_layer_input_dim, config.embed_dim));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, slice.GetBuffer());
    EXPECT_THAT(buffer.Lock().As<const float>(),
                ElementsAreArray(Iota(layer_elements, layer * layer_elements)));
  }
}

TEST(Gemma4WeightHooksTest, SlicesViewTheCombinedProjection) {
  const Config config = SmallConfig();
  LazyTensorMapping mapping =
      LazyTensorMapping({{kProjectionName, CombinedProjection(config)}})
          .Register<Gemma4WeightHooks>(config);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle combined,
                                  mapping.Get(kProjectionName));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle slice,
                                  mapping.Get(LayerProjectionName(1)));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & combined_buffer,
                                  combined.GetBuffer());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & slice_buffer, slice.GetBuffer());

  // The slice doesn't copy the data of the combined weight.
  const int layer_elements = config.per_layer_input_dim * config.embed_dim;
  auto slice_locked = slice_buffer.Lock();
  auto combined_locked = combined_buffer.Lock();
  EXPECT_EQ(slice_locked.data(),
            combined_locked.data() + layer_elements * sizeof(float));
}

TEST(Gemma4WeightHooksTest, FailsForOutOfRangeLayers) {
  const Config config = SmallConfig();
  LazyTensorMapping mapping =
      LazyTensorMapping({{kProjectionName, CombinedProjection(config)}})
          .Register<Gemma4WeightHooks>(config);

  EXPECT_THAT(mapping.Get(LayerProjectionName(config.num_layers)),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("3 layers")));
  EXPECT_THAT(mapping.Get(LayerProjectionName(-1)),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(Gemma4WeightHooksTest, FailsForMalformedLayerNames) {
  const Config config = SmallConfig();
  LazyTensorMapping mapping =
      LazyTensorMapping({{kProjectionName, CombinedProjection(config)}})
          .Register<Gemma4WeightHooks>(config);

  EXPECT_THAT(mapping.Get("model.layers.x.per_layer_model_projection.weight"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(mapping.Get("model.layers..per_layer_model_projection.weight"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(mapping.Get("model.layers.0.per_layer_projection.weight"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(Gemma4WeightHooksTest, FailsForUnknownWeights) {
  LazyTensorMapping mapping;
  mapping.Register<Gemma4WeightHooks>(SmallConfig());

  EXPECT_THAT(mapping.Get("model.unknown.weight"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(Gemma4WeightHooksTest, FailsWithoutCombinedProjection) {
  LazyTensorMapping mapping;
  mapping.Register<Gemma4WeightHooks>(SmallConfig());

  EXPECT_THAT(mapping.Get(LayerProjectionName(0)),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(Gemma4WeightHooksTest, FailsForQuantizedCombinedProjection) {
  const Config config = SmallConfig();
  TensorHandle quantized(
      {.name = kProjectionName,
       .type = Type::kI8,
       .shape = {config.num_layers, config.per_layer_input_dim,
                 config.embed_dim},
       .buffer = std::vector<int8_t>(
           config.num_layers * config.per_layer_input_dim * config.embed_dim),
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>{1.0f}, std::vector<int64_t>{0})});
  LazyTensorMapping mapping = LazyTensorMapping({{kProjectionName, quantized}})
                                  .Register<Gemma4WeightHooks>(config);

  EXPECT_THAT(mapping.Get(LayerProjectionName(0)),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(Gemma4WeightHooksTest, FailsForTooSmallCombinedProjection) {
  const Config config = SmallConfig();
  // Only holds the first layer.
  TensorHandle too_small(
      {.name = kProjectionName,
       .type = Type::kFP32,
       .shape = {1, config.per_layer_input_dim, config.embed_dim},
       .buffer = Iota(config.per_layer_input_dim * config.embed_dim)});
  LazyTensorMapping mapping = LazyTensorMapping({{kProjectionName, too_small}})
                                  .Register<Gemma4WeightHooks>(config);

  EXPECT_THAT(mapping.Get(LayerProjectionName(0)), IsOk());
  EXPECT_THAT(mapping.Get(LayerProjectionName(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("too small to slice layer 1")));
}

TEST(Gemma4WeightHooksTest, FailsForLayersThatDontStartOnAByteBoundary) {
  Config config = SmallConfig();
  config.per_layer_input_dim = 1;
  config.embed_dim = 3;
  // 3 INT4 elements per layer, i.e. 1.5 bytes.
  TensorHandle int4_projection(
      {.name = kProjectionName,
       .type = Type::kI4,
       .shape = {config.num_layers, config.per_layer_input_dim,
                 config.embed_dim},
       // Each int4_t packs 2 values: 5 hold the 3 * 3 = 9 elements.
       .buffer = std::vector<int4_t>(5)});
  LazyTensorMapping mapping =
      LazyTensorMapping({{kProjectionName, int4_projection}})
          .Register<Gemma4WeightHooks>(config);

  EXPECT_THAT(
      mapping.Get(LayerProjectionName(0)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("byte boundary")));
}

TEST(FallbackBF16ToFp32HooksTest, ConvertsBF16Weights) {
  TensorHandle weight({.name = "weight",
                       .type = Type::kBF16,
                       .shape = {3},
                       .buffer = std::vector<bf16_t>({1.0f, -2.5f, 0.125f})});

  FallbackBF16ToFp32Hooks hooks;
  ASSERT_THAT(hooks.OnLoaded("weight", weight), IsOk());

  EXPECT_EQ(weight.GetType(), Type::kFP32);
  EXPECT_THAT(weight.GetShape(), ElementsAre(3));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, weight.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const float>(),
              ElementsAre(FloatEq(1.0f), FloatEq(-2.5f), FloatEq(0.125f)));
}

TEST(FallbackBF16ToFp32HooksTest, LeavesOtherTypesUnchanged) {
  TensorHandle weight({.name = "weight",
                       .type = Type::kFP32,
                       .shape = {2},
                       .buffer = std::vector<float>({1, 2})});
  const std::shared_ptr<Buffer> original_buffer = weight.GetBufferPtr();

  FallbackBF16ToFp32Hooks hooks;
  ASSERT_THAT(hooks.OnLoaded("weight", weight), IsOk());

  EXPECT_EQ(weight.GetType(), Type::kFP32);
  EXPECT_EQ(weight.GetBufferPtr(), original_buffer);
}

// Loads a HuggingFace-style checkpoint through the Gemma 4 mapping and hooks.
TEST(Gemma4WeightsTest, LoadsCheckpointWithMappingAndHooks) {
  const Config config = SmallConfig();
  const int projection_elements =
      config.num_layers * config.per_layer_input_dim * config.embed_dim;
  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.language_model.embed_tokens.weight",
           .type = Type::kBF16,
           .shape = {2, 2},
           .buffer = std::vector<bf16_t>({1, 2, 3, 4})},
          {.name = "model.language_model.per_layer_model_projection.weight",
           .type = Type::kFP32,
           .shape = {config.num_layers, config.per_layer_input_dim,
                     config.embed_dim},
           .buffer = Iota(projection_elements)},
      },
      /*quant_config_json=*/"");
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  LazyTensorMapping mapping =
      LazyTensorMapping(GetGemma4WeightMapping(config.num_layers),
                        std::move(loader))
          .Register<Gemma4WeightHooks>(config)
          .Register<FallbackBF16ToFp32Hooks>();

  // The checkpoint doesn't hold the LM head, which is tied to the BF16
  // embeddings and converted to FP32.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle lm_head,
                                  mapping.Get("lm_head.weight"));
  EXPECT_EQ(lm_head.GetType(), Type::kFP32);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & lm_head_buffer, lm_head.GetBuffer());
  EXPECT_THAT(lm_head_buffer.Lock().As<const float>(),
              ElementsAre(FloatEq(1), FloatEq(2), FloatEq(3), FloatEq(4)));

  // The checkpoint only holds the combined per-layer model projection.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle layer2,
                                  mapping.Get(LayerProjectionName(2)));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & layer2_buffer, layer2.GetBuffer());
  const int layer_elements = config.per_layer_input_dim * config.embed_dim;
  EXPECT_THAT(layer2_buffer.Lock().As<const float>(),
              ElementsAreArray(Iota(layer_elements, 2 * layer_elements)));
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
