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

#include "tensor/examples/gemma3/safetensor_loader.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/examples/utils/safetensor_test_util.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::VariantWith;

TEST(Gemma3SafetensorLoaderTest, AbslFlagParseAndUnparse) {
  Gemma3QuantizedLoadMode mode = Gemma3QuantizedLoadMode::kPreserveQuantized;
  std::string error;

  EXPECT_TRUE(AbslParseFlag("fp32", &mode, &error));
  EXPECT_EQ(mode, Gemma3QuantizedLoadMode::kDequantizeToFp32);
  EXPECT_EQ(AbslUnparseFlag(mode), "fp32");

  EXPECT_TRUE(AbslParseFlag("preserve", &mode, &error));
  EXPECT_EQ(mode, Gemma3QuantizedLoadMode::kPreserveQuantized);
  EXPECT_EQ(AbslUnparseFlag(mode), "preserve");

  EXPECT_TRUE(AbslParseFlag("float", &mode, &error));
  EXPECT_EQ(mode, Gemma3QuantizedLoadMode::kDequantizeToFp32);

  EXPECT_TRUE(AbslParseFlag("quantized", &mode, &error));
  EXPECT_EQ(mode, Gemma3QuantizedLoadMode::kPreserveQuantized);

  EXPECT_FALSE(AbslParseFlag("invalid_mode", &mode, &error));
  EXPECT_THAT(error, Not(IsEmpty()));
}

TEST(Gemma3SafetensorLoaderTest,
     UpcastsBf16AndFp16ToFp32WithoutOffsetForNonNormWeights) {
  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.embed_tokens.weight",
           .type = Type::kBF16,
           .shape = {2, 2},
           .buffer = std::vector<bf16_t>{bf16_t(1.5f), bf16_t(-2.0f),
                                         bf16_t(0.25f), bf16_t(4.0f)}},
          {.name = "model.layers.0.self_attn.q_proj.weight",
           .type = Type::kFP16,
           .shape = {2, 2},
           .buffer = std::vector<fp16_t>{fp16_t(-0.5f), fp16_t(3.0f),
                                         fp16_t(1.25f), fp16_t(0.0f)}},
      },
      /*quant_config_json=*/"");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle emb,
      LoadGemma3Tensor(loader, "model.embed_tokens.weight",
                       Gemma3QuantizedLoadMode::kPreserveQuantized));
  EXPECT_EQ(emb.GetType(), Type::kFP32);
  EXPECT_THAT(emb.GetShape(), ElementsAre(2, 2));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & emb_buf, emb.GetBuffer());
  EXPECT_THAT(emb_buf.Lock().As<const float>(),
              ElementsAre(FloatEq(1.5f), FloatEq(-2.0f), FloatEq(0.25f),
                          FloatEq(4.0f)));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle q_proj,
      LoadGemma3Tensor(loader, "model.layers.0.self_attn.q_proj.weight",
                       Gemma3QuantizedLoadMode::kPreserveQuantized));
  EXPECT_EQ(q_proj.GetType(), Type::kFP32);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & q_buf, q_proj.GetBuffer());
  EXPECT_THAT(q_buf.Lock().As<const float>(),
              ElementsAre(FloatEq(-0.5f), FloatEq(3.0f), FloatEq(1.25f),
                          FloatEq(0.0f)));
}

TEST(Gemma3SafetensorLoaderTest, AppliesUnitOffsetToZeroCenteredNormWeights) {
  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.norm.weight",
           .type = Type::kBF16,
           .shape = {2},
           .buffer = std::vector<bf16_t>{bf16_t(0.5f), bf16_t(-0.25f)}},
          {.name = "model.layers.0.self_attn.q_norm.weight",
           .type = Type::kFP16,
           .shape = {2},
           .buffer = std::vector<fp16_t>{fp16_t(0.0f), fp16_t(1.5f)}},
          {.name = "model.layers.0.input_layernorm.weight",
           .type = Type::kFP32,
           .shape = {2},
           .buffer = std::vector<float>{-0.5f, 0.25f}},
      },
      /*quant_config_json=*/"");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle final_norm,
      LoadGemma3Tensor(loader, "model.norm.weight",
                       Gemma3QuantizedLoadMode::kPreserveQuantized));
  EXPECT_EQ(final_norm.GetType(), Type::kFP32);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & final_buf, final_norm.GetBuffer());
  EXPECT_THAT(final_buf.Lock().As<const float>(),
              ElementsAre(FloatEq(1.5f), FloatEq(0.75f)));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle q_norm,
      LoadGemma3Tensor(loader, "model.layers.0.self_attn.q_norm.weight",
                       Gemma3QuantizedLoadMode::kPreserveQuantized));
  EXPECT_EQ(q_norm.GetType(), Type::kFP32);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & q_norm_buf, q_norm.GetBuffer());
  EXPECT_THAT(q_norm_buf.Lock().As<const float>(),
              ElementsAre(FloatEq(1.0f), FloatEq(2.5f)));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle input_ln,
      LoadGemma3Tensor(loader, "model.layers.0.input_layernorm.weight",
                       Gemma3QuantizedLoadMode::kPreserveQuantized));
  EXPECT_EQ(input_ln.GetType(), Type::kFP32);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & ln_buf, input_ln.GetBuffer());
  EXPECT_THAT(ln_buf.Lock().As<const float>(),
              ElementsAre(FloatEq(0.5f), FloatEq(1.25f)));

  // Verify the underlying mmap'ed FP32 buffer was not mutated in place.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle raw_ln,
      loader.LoadTensor("model.layers.0.input_layernorm.weight"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & raw_ln_buf, raw_ln.GetBuffer());
  EXPECT_THAT(raw_ln_buf.Lock().As<const float>(),
              ElementsAre(FloatEq(-0.5f), FloatEq(0.25f)));
}

TEST(Gemma3SafetensorLoaderTest,
     LegacyInt8SideTensorPreserveQuantizedAndDequantize) {
  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.layers.0.mlp.gate_proj.weight",
           .type = Type::kI8,
           .shape = {2, 3},
           .buffer = std::vector<int8_t>{-10, 0, 10, 20, -20, 4}},
          {.name = "model.layers.0.mlp.gate_proj.weight.scales",
           .type = Type::kBF16,
           .shape = {2},
           .buffer = std::vector<bf16_t>{bf16_t(0.5f), bf16_t(0.25f)}},
          {.name = "model.layers.0.mlp.gate_proj.weight.zero_point",
           .type = Type::kI32,
           .shape = {2},
           .buffer = std::vector<int32_t>{2, -4}},
      },
      /*quant_config_json=*/"");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle preserved,
      LoadGemma3Tensor(loader, "model.layers.0.mlp.gate_proj.weight",
                       Gemma3QuantizedLoadMode::kPreserveQuantized));
  EXPECT_EQ(preserved.GetType(), Type::kI8);
  ASSERT_NE(preserved.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const PerChannelAffineQuantization& pcq,
      preserved.GetQuantization()->As<PerChannelAffineQuantization>());
  EXPECT_THAT(pcq.scales, ElementsAre(FloatEq(0.5f), FloatEq(0.25f)));
  EXPECT_THAT(pcq.zero_points,
              VariantWith<std::vector<int64_t>>(ElementsAre(2, -4)));
  EXPECT_EQ(pcq.quantized_dimension, 0);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle dequantized,
      LoadGemma3Tensor(loader, "model.layers.0.mlp.gate_proj.weight",
                       Gemma3QuantizedLoadMode::kDequantizeToFp32));
  EXPECT_EQ(dequantized.GetType(), Type::kFP32);
  EXPECT_EQ(dequantized.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & deq_buf, dequantized.GetBuffer());
  EXPECT_THAT(deq_buf.Lock().As<const float>(),
              ElementsAre(FloatEq(-6.0f), FloatEq(-1.0f), FloatEq(4.0f),
                          FloatEq(6.0f), FloatEq(-4.0f), FloatEq(2.0f)));
}

TEST(Gemma3SafetensorLoaderTest,
     CompressedTensorsBlockwiseInt8DequantizesAndClearsQuantization) {
  const std::string quant_cfg = R"({
    "quant_method": "compressed-tensors",
    "config_groups": {
      "group_0": {
        "num_bits": 8,
        "group_size": 2,
        "strategy": "group",
        "symmetric": true
      }
    }
  })";
  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.layers.0.self_attn.v_proj.weight",
           .type = Type::kI8,
           .shape = {2, 4},
           .buffer = std::vector<int8_t>{2, -4, 6, -8, 10, -10, 4, -2}},
          {.name = "model.layers.0.self_attn.v_proj.weight_scale",
           .type = Type::kFP32,
           .shape = {2, 2},
           .buffer = std::vector<float>{0.5f, 0.25f, 0.1f, 2.0f}},
      },
      quant_cfg);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle preserved,
      LoadGemma3Tensor(loader, "model.layers.0.self_attn.v_proj.weight",
                       Gemma3QuantizedLoadMode::kPreserveQuantized));
  EXPECT_EQ(preserved.GetType(), Type::kI8);
  ASSERT_NE(preserved.GetQuantization(), nullptr);
  EXPECT_THAT(preserved.GetQuantization()->As<BlockwiseQuantization>(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle dequantized,
      LoadGemma3Tensor(loader, "model.layers.0.self_attn.v_proj.weight",
                       Gemma3QuantizedLoadMode::kDequantizeToFp32));
  EXPECT_EQ(dequantized.GetType(), Type::kFP32);
  EXPECT_EQ(dequantized.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & deq_buf, dequantized.GetBuffer());
  EXPECT_THAT(deq_buf.Lock().As<const float>(),
              ElementsAre(FloatEq(1.0f), FloatEq(-2.0f), FloatEq(1.5f),
                          FloatEq(-2.0f), FloatEq(1.0f), FloatEq(-1.0f),
                          FloatEq(8.0f), FloatEq(-4.0f)));
}

TEST(Gemma3SafetensorLoaderTest,
     LoadWeightsWithMappingRenamesPresentWeightsAndSkipsMissing) {
  const absl::flat_hash_map<std::string, std::string> gemma3_mapping =
      GetGemma3WeightMapping(/*n_layers=*/1);
  EXPECT_EQ(gemma3_mapping.size(), 15);
  EXPECT_TRUE(gemma3_mapping.contains("model.embed_tokens.weight"));
  EXPECT_TRUE(gemma3_mapping.contains("model.norm.weight"));
  EXPECT_TRUE(gemma3_mapping.contains(
      "model.layers.0.post_feedforward_layernorm.weight"));

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "hf.norm.weight",
           .type = Type::kBF16,
           .shape = {2},
           .buffer = std::vector<bf16_t>{bf16_t(0.0f), bf16_t(0.5f)}},
      },
      /*quant_config_json=*/"");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const absl::flat_hash_map<std::string, std::string> custom_mapping = {
      {"hf.norm.weight", "mapped.norm.weight"},
      {"hf.missing.weight", "mapped.missing.weight"},
  };
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto mapped_weights,
      LoadGemma3WeightsWithMapping(
          loader, custom_mapping, Gemma3QuantizedLoadMode::kPreserveQuantized));
  ASSERT_EQ(mapped_weights.size(), 1);
  ASSERT_TRUE(mapped_weights.contains("mapped.norm.weight"));
  EXPECT_EQ(mapped_weights.at("mapped.norm.weight").GetName(),
            "mapped.norm.weight");
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Buffer & buf, mapped_weights.at("mapped.norm.weight").GetBuffer());
  EXPECT_THAT(buf.Lock().As<const float>(),
              ElementsAre(FloatEq(1.0f), FloatEq(1.5f)));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto all_weights,
      LoadAllGemma3Tensors(loader,
                           Gemma3QuantizedLoadMode::kPreserveQuantized));
  EXPECT_EQ(all_weights.size(), 1);
  EXPECT_TRUE(all_weights.contains("hf.norm.weight"));
}

}  // namespace
}  // namespace litert::tensor::examples
