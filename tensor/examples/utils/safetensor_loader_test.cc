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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/safetensor_test_util.h"
#include "tensor/examples/utils/safetensors.h"
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

TEST(SafetensorLoaderTest, LoadsHeaderNotSortedByOffset) {
  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "b",
           .type = Type::kFP32,
           .shape = {2},
           .buffer = std::vector<float>{1, 2}},
          {.name = "a",
           .type = Type::kFP32,
           .shape = {1},
           .buffer = std::vector<float>{3}},
      },
      R"({})", Order::kReverse);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle b, loader.LoadTensor("b"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, b.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const float>(), ElementsAre(1.0f, 2.0f));
}

SafetensorFileGuard WriteRawSafetensorFile(
    std::string_view header_json, const std::vector<uint8_t>& payload) {
  SafetensorFileGuard file = SafetensorFileGuard::CreateTemp();
  std::string padded_header(header_json);
  while ((8 + padded_header.size() + payload.size()) < 16 ||
         (padded_header.size() % 8) != 0) {
    padded_header.push_back(' ');
  }
  const uint64_t header_size = padded_header.size();
  std::ofstream out(file.GetPath(), std::ios::binary);
  out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
  out.write(padded_header.data(),
            static_cast<std::streamsize>(padded_header.size()));
  if (!payload.empty()) {
    out.write(reinterpret_cast<const char*>(payload.data()),
              static_cast<std::streamsize>(payload.size()));
  }
  return file;
}

TEST(SafetensorLoaderTest, RejectsShapeProductOverflow) {
  // 2^32 * 2^32 wraps around to 0 in 64-bit unsigned arithmetic.
  constexpr std::string_view kHeader = R"({
    "overflow_tensor": {
      "dtype": "I32",
      "shape": [4294967296, 4294967296],
      "data_offsets": [64, 128]
    },
    "anchor": {
      "dtype": "F32",
      "shape": [16],
      "data_offsets": [0, 64]
    }
  })";
  SafetensorFileGuard file =
      WriteRawSafetensorFile(kHeader, std::vector<uint8_t>(64, 0));
  EXPECT_THAT(SafetensorLoader::Load(file.GetPath()), Not(IsOk()));

  safetensors::tensor_t t;
  t.dtype = safetensors::dtype::kINT32;
  if constexpr (sizeof(size_t) >= 8) {
    const size_t two_pow_32 = size_t{1} << 32;
    t.shape = {two_pow_32, two_pow_32};
  } else {
    t.shape = {65536, 65536};
  }
  bool ok = true;
  EXPECT_EQ(safetensors::get_shape_size(t, &ok), 0u);
  EXPECT_FALSE(ok);
}

TEST(SafetensorLoaderTest, RejectsByteSizeMultiplicationOverflow) {
  // For F32 (4 bytes): 4 * 2^62 == 2^64 (wraps to 0).
  SafetensorFileGuard zero_wrap = WriteRawSafetensorFile(
      R"({
        "t": {
          "dtype": "F32",
          "shape": [4611686018427387904],
          "data_offsets": [0, 64]
        }
      })",
      std::vector<uint8_t>(64, 0));
  EXPECT_THAT(SafetensorLoader::Load(zero_wrap.GetPath()), Not(IsOk()));

  // For F32 (4 bytes): 4 * (2^62 + 1) wraps to 4 bytes, matching data_offsets!
  // Note: 2^62 + 1 is not exact in IEEE-754 double, so test with I64 (8 bytes)
  // and shape [2305843009213693952, 1] (wraps to 0) plus direct C++ struct
  // validation for exact non-zero wraparound (4 * (2^62 + 1) == 4).
  if constexpr (sizeof(size_t) >= 8) {
    safetensors::safetensors_t st;
    st.storage.resize(4, 0);
    safetensors::tensor_t tensor;
    tensor.dtype = safetensors::dtype::kFLOAT32;
    tensor.shape = {(size_t{1} << 62) + 1};
    tensor.data_offsets = {0, 4};
    st.tensors.insert("t", tensor);
    std::string err;
    EXPECT_FALSE(safetensors::validate_data_offsets(st, err));
  }
}

TEST(SafetensorLoaderTest, Validates8DTensorAndRejects9DTensor) {
  std::vector<uint8_t> payload(sizeof(float), 0);
  const float value = 42.5f;
  std::memcpy(payload.data(), &value, sizeof(float));

  // Valid 8-D tensor (rank == kMaxDim) must load and validate its offsets.
  SafetensorFileGuard valid_8d = WriteRawSafetensorFile(
      R"({
        "t": {
          "dtype": "F32",
          "shape": [1, 1, 1, 1, 1, 1, 1, 1],
          "data_offsets": [0, 4]
        }
      })",
      payload);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(valid_8d.GetPath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle t, loader.LoadTensor("t"));
  EXPECT_THAT(t.GetShape(), ElementsAre(1, 1, 1, 1, 1, 1, 1, 1));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, t.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const float>(), ElementsAre(FloatEq(42.5f)));

  // 8-D tensor with out-of-bounds data_offsets must be rejected.
  SafetensorFileGuard oob_8d = WriteRawSafetensorFile(
      R"({
        "t": {
          "dtype": "F32",
          "shape": [1, 1, 1, 1, 1, 1, 1, 1],
          "data_offsets": [0, 4096]
        }
      })",
      payload);
  EXPECT_THAT(SafetensorLoader::Load(oob_8d.GetPath()), Not(IsOk()));

  // 9-D tensor (rank > kMaxDim) must be rejected.
  SafetensorFileGuard invalid_9d = WriteRawSafetensorFile(
      R"({
        "t": {
          "dtype": "F32",
          "shape": [1, 1, 1, 1, 1, 1, 1, 1, 1],
          "data_offsets": [0, 4]
        }
      })",
      payload);
  EXPECT_THAT(SafetensorLoader::Load(invalid_9d.GetPath()), Not(IsOk()));
}

TEST(SafetensorLoaderTest, RejectsInvalidJsonNumbersInShapeAndOffsets) {
  const std::vector<std::string_view> invalid_headers = {
      R"({"t": {"dtype": "F32", "shape": [-1, 4], "data_offsets": [0, 16]}})",
      R"({"t": {"dtype": "F32", "shape": [1.5], "data_offsets": [0, 4]}})",
      R"({"t": {"dtype": "F32", "shape": [18446744073709551616], "data_offsets": [0, 0]}})",
      R"({"t": {"dtype": "F32", "shape": [1e30], "data_offsets": [0, 0]}})",
      R"({"t": {"dtype": "F32", "shape": [1], "data_offsets": [-4, 0]}})",
      R"({"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4.5]}})",
      R"({"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 1e30]}})",
  };

  for (std::string_view header : invalid_headers) {
    SafetensorFileGuard file =
        WriteRawSafetensorFile(header, std::vector<uint8_t>(16, 0));
    EXPECT_THAT(SafetensorLoader::Load(file.GetPath()), Not(IsOk())) << header;
  }
}

TEST(SafetensorLoaderTest, ValidatesEmptyTensorsAndRejectsInvalidEmptyOffsets) {
  // Valid empty tensor alongside a non-empty tensor (HuggingFace format
  // includes `data_offsets` with begin == end for empty tensors).
  SafetensorFileGuard valid_mixed = WriteRawSafetensorFile(
      R"({
        "w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        "empty": {"dtype": "F32", "shape": [0, 4], "data_offsets": [16, 16]}
      })",
      std::vector<uint8_t>(16, 0));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      SafetensorLoader mixed_loader,
      SafetensorLoader::Load(valid_mixed.GetPath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle empty_tensor,
                                  mixed_loader.LoadTensor("empty"));
  EXPECT_THAT(empty_tensor.GetShape(), ElementsAre(0, 4));

  // Valid file containing only an empty tensor and 0-byte payload.
  SafetensorFileGuard valid_only_empty = WriteRawSafetensorFile(
      R"({"empty": {"dtype": "F32", "shape": [0, 4], "data_offsets": [0, 0]}})",
      {});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      SafetensorLoader empty_loader,
      SafetensorLoader::Load(valid_only_empty.GetPath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle only_empty,
                                  empty_loader.LoadTensor("empty"));
  EXPECT_THAT(only_empty.GetShape(), ElementsAre(0, 4));

  // Empty tensor claiming non-zero data range must fail.
  SafetensorFileGuard nonzero_empty = WriteRawSafetensorFile(
      R"({"empty": {"dtype": "F32", "shape": [0, 4], "data_offsets": [0, 16]}})",
      std::vector<uint8_t>(16, 0));
  EXPECT_THAT(SafetensorLoader::Load(nonzero_empty.GetPath()), Not(IsOk()));

  // Empty tensor with out-of-bounds offset must fail.
  SafetensorFileGuard oob_empty = WriteRawSafetensorFile(
      R"({
        "w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        "empty": {"dtype": "F32", "shape": [0, 4], "data_offsets": [64, 64]}
      })",
      std::vector<uint8_t>(16, 0));
  EXPECT_THAT(SafetensorLoader::Load(oob_empty.GetPath()), Not(IsOk()));

  // Empty-only tensor with unreferenced trailing payload bytes must fail.
  SafetensorFileGuard trailing_bytes = WriteRawSafetensorFile(
      R"({"empty": {"dtype": "F32", "shape": [0, 4], "data_offsets": [0, 0]}})",
      std::vector<uint8_t>(16, 0));
  EXPECT_THAT(SafetensorLoader::Load(trailing_bytes.GetPath()), Not(IsOk()));
}

TEST(SafetensorLoaderTest, RejectsHeaderSizeWraparoundAndNonObjectRoot) {
  // Craft a 16-byte buffer where `8 + header_size` wraps in uint64_t to 4.
  std::vector<uint8_t> wrapped_header(16, ' ');
  const uint64_t huge_header_size = ~uint64_t{0} - 3;
  std::memcpy(wrapped_header.data(), &huge_header_size,
              sizeof(huge_header_size));
  safetensors::safetensors_t st;
  std::string warn, err;
  EXPECT_FALSE(safetensors::load_from_memory(
      wrapped_header.data(), wrapped_header.size(), "mem", &st, &warn, &err));
  EXPECT_FALSE(safetensors::mmap_from_memory(
      wrapped_header.data(), wrapped_header.size(), "mem", &st, &warn, &err));

  // JSON root that is an array rather than an object must be rejected.
  SafetensorFileGuard array_root = WriteRawSafetensorFile("[]", {});
  EXPECT_THAT(SafetensorLoader::Load(array_root.GetPath()), Not(IsOk()));
}

TEST(SafetensorLoaderTest, RejectsShapeDimensionExceedingIntMax) {
  // Shape [0, 2147483648] has 0 elements (0 bytes on disk), so it passes file
  // offset validation on 64-bit platforms, but 2147483648 exceeds INT_MAX and
  // must be rejected when constructing TensorHandle's `int` shape.
  SafetensorFileGuard file = WriteRawSafetensorFile(
      R"({"t": {"dtype": "U8", "shape": [0, 2147483648], "data_offsets": [0, 0]}})",
      {});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  EXPECT_THAT(loader.LoadTensor("t"), Not(IsOk()));
}

}  // namespace
}  // namespace litert::tensor::examples
