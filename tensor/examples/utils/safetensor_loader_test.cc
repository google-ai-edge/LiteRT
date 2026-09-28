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

#include "tensor/examples/utils/safetensor_loader.h"

#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/safetensor_test_util.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::Not;

TEST(SafetensorLoaderTest, AbslStringifyMethodAndStrategy) {
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Method::kCompressedTensors),
            "compressed-tensors");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Method::kUnknown), "unknown");

  EXPECT_EQ(absl::StrCat(QuantizationConfig::Strategy::kTensor), "tensor");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Strategy::kChannel), "channel");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Strategy::kGroup), "group");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Strategy::kUnknown), "unknown");
}

TEST(SafetensorLoaderTest, ParseTopLevelConfig) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": {
        "num_bits": 4,
        "group_size": 128,
        "symmetric": true
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->quant_method,
            QuantizationConfig::Method::kCompressedTensors);
  EXPECT_EQ(quant_config->num_bits, 4);
  EXPECT_EQ(quant_config->group_size, 128);
  EXPECT_TRUE(quant_config->symmetric);
}

TEST(SafetensorLoaderTest, ParseNestedConfigGroups) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "quantization_status": "compressed",
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 4,
          "type": "int",
          "symmetric": true,
          "strategy": "group",
          "group_size": 128
        },
        "targets": ["Linear"]
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->quant_method,
            QuantizationConfig::Method::kCompressedTensors);
  EXPECT_EQ(quant_config->num_bits, 4);
  EXPECT_EQ(quant_config->group_size, 128);
  EXPECT_TRUE(quant_config->symmetric);
}

TEST(SafetensorLoaderTest, ParseInt8Config) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 8
        }
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->num_bits, 8);
}

TEST(SafetensorLoaderTest, ParsesMultipleConfigGroups) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "ignore": ["model.vision_tower", "relative_k_proj"],
    "config_groups": {
      "group_0": {
        "weights": { "num_bits": 2, "strategy": "channel", "group_size": null },
        "targets": ["model.embed_tokens", "re:.*lm_head$"]
      },
      "group_1": {
        "weights": { "num_bits": 4, "strategy": "group", "group_size": 128 },
        "targets": ["model.layers.0.mlp.down_proj"]
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  ASSERT_EQ(quant_config->schemes.size(), 2);

  // A group without a group size quantizes per channel.
  const QuantizationConfig::Scheme* embed =
      quant_config->FindScheme("model.embed_tokens");
  ASSERT_NE(embed, nullptr);
  EXPECT_EQ(embed->num_bits, 2);
  EXPECT_EQ(embed->strategy, QuantizationConfig::Strategy::kChannel);

  // Targets also match as regular expressions.
  EXPECT_EQ(quant_config->FindScheme("model.lm_head"), embed);

  const QuantizationConfig::Scheme* down_proj =
      quant_config->FindScheme("model.layers.0.mlp.down_proj");
  ASSERT_NE(down_proj, nullptr);
  EXPECT_EQ(down_proj->num_bits, 4);
  EXPECT_EQ(down_proj->strategy, QuantizationConfig::Strategy::kGroup);
  EXPECT_EQ(down_proj->group_size, 128);

  // Unclaimed modules have no scheme, since every group names its targets.
  EXPECT_EQ(quant_config->FindScheme("model.layers.0.mlp.up_proj"), nullptr);

  // Ignored modules are never quantized, whether named as a parent or a leaf.
  EXPECT_TRUE(quant_config->IsIgnored("model.vision_tower.layers.0.self_attn"));
  EXPECT_TRUE(
      quant_config->IsIgnored("model.layers.0.self_attn.relative_k_proj"));
  EXPECT_FALSE(quant_config->IsIgnored("model.embed_tokens"));
}

TEST(SafetensorLoaderTest, TargetsNamingAModuleClassApplyToEveryModule) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": {
        "weights": { "num_bits": 4, "group_size": 128 },
        "targets": ["Linear"]
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  ASSERT_EQ(quant_config->schemes.size(), 1);
  EXPECT_TRUE(quant_config->schemes.front().matches_any_module);
  EXPECT_NE(quant_config->FindScheme("model.layers.3.mlp.up_proj"), nullptr);
}

TEST(SafetensorLoaderTest, LoadsPackedInt4WeightPerChannel) {
  // Two rows of eight 4-bit values, spanning the whole signed range.
  auto values = OwningCpuBuffer::Copy<Type::kI4>(
      {-8, -1, 0, 7, 3, -4, 5, -6, 1, 2, 3, 4, 5, 6, 7, -7});
  auto scales = OwningCpuBuffer::Copy<Type::kFP32>({0.5f, 0.25f});

  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": {
        "weights": { "num_bits": 4, "strategy": "channel" },
        "targets": ["model.layers.0.mlp.down_proj"]
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.layers.0.mlp.down_proj.weight_packed",
           .type = Type::kI4,
           .shape = {2, 8},
           .buffer = values},
          {.name = "model.layers.0.mlp.down_proj.weight_scale",
           .type = Type::kFP32,
           .shape = {2, 1},
           .buffer = scales},
      },
      json);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  // The weight is asked for by the name it would have when uncompressed.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle tensor,
      loader.LoadTensor("model.layers.0.mlp.down_proj.weight"));

  EXPECT_EQ(tensor.GetType(), Type::kI4);
  EXPECT_THAT(tensor.GetShape(), ElementsAre(2, 8));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const int4_t>(),
              ElementsAreArray(values->Span<int4_t>()));

  ASSERT_NE(tensor.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const PerChannelAffineQuantization& quantization,
      tensor.GetQuantization()->As<const PerChannelAffineQuantization>());
  EXPECT_THAT(quantization.scales, ElementsAre(FloatEq(0.5f), FloatEq(0.25f)));
  EXPECT_THAT(quantization.zero_points, ElementsAre(0));
  EXPECT_EQ(quantization.quantized_dimension, 0);
}

TEST(SafetensorLoaderTest, LoadsPackedInt2WeightBlockwise) {
  // One row of sixteen 2-bit values, i.e. one int32, in two blocks of eight.
  auto values = OwningCpuBuffer::Copy<Type::kI2>(
      {-2, -1, 0, 1, 1, 0, -1, -2, 0, 1, 1, 1, -2, -2, 0, 1});
  auto scales = OwningCpuBuffer::Copy<Type::kFP32>({0.125f, 0.0625f});

  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": {
        "weights": { "num_bits": 2, "strategy": "group", "group_size": 8 },
        "targets": ["model.embed_tokens_per_layer"]
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.embed_tokens_per_layer.weight_packed",
           .type = Type::kI2,
           .shape = {1, 16},
           .buffer = values},
          {.name = "model.embed_tokens_per_layer.weight_scale",
           .type = Type::kFP32,
           .shape = {1, 2},
           .buffer = scales},
      },
      json);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle tensor,
      loader.LoadTensor("model.embed_tokens_per_layer.weight"));

  EXPECT_EQ(tensor.GetType(), Type::kI2);
  EXPECT_THAT(tensor.GetShape(), ElementsAre(1, 16));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const int2_t>(),
              ElementsAreArray(values->Span<int2_t>()));

  ASSERT_NE(tensor.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const BlockwiseQuantization& quantization,
      tensor.GetQuantization()->As<const BlockwiseQuantization>());
  EXPECT_EQ(quantization.block_size, 8);
  EXPECT_THAT(quantization.scales,
              ElementsAre(FloatEq(0.125f), FloatEq(0.0625f)));
}

TEST(SafetensorLoaderTest, WeightShapeTrimsPackingPadding) {
  // Six values, i.e. one int32 worth of 4-bit fields with two to spare.
  auto values = OwningCpuBuffer::Copy<Type::kI4>({1, 2, 3, 4, 5, 6});
  auto logical_shape = OwningCpuBuffer::Copy<Type::kI64>({1, 6});
  auto scales = OwningCpuBuffer::Copy<Type::kFP32>({1});

  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": { "weights": { "num_bits": 4, "strategy": "channel" } }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.proj.weight_packed",
           .type = Type::kI4,
           .shape = {1, 6},
           .buffer = values},
          {.name = "model.proj.weight_scale",
           .type = Type::kFP32,
           .shape = {1, 1},
           .buffer = scales},
          {.name = "model.proj.weight_shape",
           .type = Type::kI64,
           .shape = {2},
           .buffer = logical_shape},
      },
      json);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle tensor,
                                  loader.LoadTensor("model.proj.weight"));

  EXPECT_THAT(tensor.GetShape(), ElementsAre(1, 6));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const int4_t>(),
              ElementsAreArray(values->Span<int4_t>()));
}

TEST(SafetensorLoaderTest, WeightShapeTrimsPackingPaddingOfEveryRow) {
  // Two rows of six values, each padded to its own int32.
  auto values = OwningCpuBuffer::Copy<Type::kI4>(
      {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6});
  auto logical_shape = OwningCpuBuffer::Copy<Type::kI64>({2, 6});
  auto scales = OwningCpuBuffer::Copy<Type::kFP32>({1, 1});

  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": { "weights": { "num_bits": 4, "strategy": "channel" } }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.proj.weight_packed",
           .type = Type::kI4,
           .shape = {2, 6},
           .buffer = values},
          {.name = "model.proj.weight_scale",
           .type = Type::kFP32,
           .shape = {2, 1},
           .buffer = scales},
          {.name = "model.proj.weight_shape",
           .type = Type::kI64,
           .shape = {2},
           .buffer = logical_shape},
      },
      json);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle tensor,
                                  loader.LoadTensor("model.proj.weight"));

  EXPECT_THAT(tensor.GetShape(), ElementsAre(2, 6));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const int4_t>(),
              ElementsAreArray(values->Span<int4_t>()));
}

TEST(SafetensorLoaderTest, ReadsQuantizationConfigFromConfigJson) {
  // QAT checkpoints exported by HuggingFace carry no header metadata and
  // describe their quantization in config.json instead.
  auto values = OwningCpuBuffer::Copy<Type::kI4>({1, -2, 3, -4, 5, -6, 7, -8});
  auto scales = OwningCpuBuffer::Copy<Type::kFP32>({0.5});

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.layers.0.mlp.up_proj.weight_packed",
           .type = Type::kI4,
           .shape = {1, 8},
           .buffer = values},
          {.name = "model.layers.0.mlp.up_proj.weight_scale",
           .type = Type::kFP32,
           .shape = {1, 1},
           .buffer = scales},
      },
      /*quant_config_json=*/"");

  {
    std::ofstream config(file.GetConfigPath());
    config << R"({
      "model_type": "test",
      "quantization_config": {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
          "group_0": {
            "weights": { "num_bits": 4, "strategy": "channel" },
            "targets": ["Linear"]
          }
        }
      }
    })";
  }

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const std::optional<QuantizationConfig>& quant_config =
      loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_NE(quant_config->FindScheme("model.layers.0.mlp.up_proj"), nullptr);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle tensor,
      loader.LoadTensor("model.layers.0.mlp.up_proj.weight"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const int4_t>(),
              ElementsAreArray(values->Span<int4_t>()));
}

TEST(SafetensorLoaderTest, PackedWeightWithoutQuantizationConfigFails) {
  auto values = OwningCpuBuffer::Copy<Type::kI4>({1, -2, 3, -4, 5, -6, 7, -8});

  SafetensorFileGuard file =
      CreateTempSafetensor({{.name = "model.proj.weight_packed",
                             .type = Type::kI4,
                             .shape = {1, 8},
                             .buffer = values}},
                           /*quant_config_json=*/"");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  EXPECT_FALSE(loader.LoadTensor("model.proj.weight").ok());
}

TEST(SafetensorLoaderTest, RejectNonPositiveNumBits) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 0,
          "group_size": 128
        }
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  EXPECT_THAT(SafetensorLoader::Load(file.GetPath()), Not(IsOk()));
}

TEST(SafetensorLoaderTest, MalformedJsonFails) {
  std::string invalid_json =
      R"({ "quant_method": "compressed-tensors", format: })";

  SafetensorFileGuard file = CreateTempSafetensor(invalid_json);
  EXPECT_THAT(SafetensorLoader::Load(file.GetPath()), Not(IsOk()));
}

}  // namespace
}  // namespace litert::tensor::examples
