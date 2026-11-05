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

#include "tensor/examples/gemma3/model_config.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/gemma3_config.h"
#include "tensor/examples/gemma3/safetensor_model_config.h"
#include "tensor/examples/gemma3/tflite_loader.h"
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/examples/utils/safetensor_test_util.h"
#include "tensor/utils/matchers.h"
#include "tflite/schema/schema_generated.h"

namespace litert::tensor::examples {
namespace {

using ::litert::tensor::IsOkAndHolds;
using ::testing::HasSubstr;

std::string BuildEmptyTfliteModelBuffer() {
  flatbuffers::FlatBufferBuilder builder;
  auto subgraphs = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::SubGraph>>{});
  auto buffers =
      builder.CreateVector(std::vector<flatbuffers::Offset<tflite::Buffer>>{});
  auto model =
      tflite::CreateModel(builder, /*version=*/3, /*operator_codes=*/0,
                          subgraphs, /*description=*/0, buffers);
  tflite::FinishModelBuffer(builder, model);
  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}

TEST(ModelConfigTest, ParseAndFormatModelVariant) {
  EXPECT_THAT(ParseGemma3ModelVariant("auto"),
              IsOkAndHolds(Gemma3ModelVariant::kAuto));
  EXPECT_THAT(ParseGemma3ModelVariant("INFER"),
              IsOkAndHolds(Gemma3ModelVariant::kAuto));
  EXPECT_THAT(ParseGemma3ModelVariant("270m"),
              IsOkAndHolds(Gemma3ModelVariant::k270M));
  EXPECT_THAT(ParseGemma3ModelVariant("Gemma-3-270M-IT"),
              IsOkAndHolds(Gemma3ModelVariant::k270M));
  EXPECT_THAT(ParseGemma3ModelVariant("1b"),
              IsOkAndHolds(Gemma3ModelVariant::k1B));
  EXPECT_THAT(ParseGemma3ModelVariant("gemma-3-1b-it"),
              IsOkAndHolds(Gemma3ModelVariant::k1B));
  absl::StatusOr<Gemma3ModelVariant> invalid_variant =
      ParseGemma3ModelVariant("7b");
  EXPECT_EQ(invalid_variant.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(invalid_variant.status().message(),
              HasSubstr("Unsupported model variant"));

  EXPECT_EQ(Gemma3ModelVariantToString(Gemma3ModelVariant::kAuto), "auto");
  EXPECT_EQ(Gemma3ModelVariantToString(Gemma3ModelVariant::k270M), "270m");
  EXPECT_EQ(Gemma3ModelVariantToString(Gemma3ModelVariant::k1B), "1b");
}

TEST(ModelConfigTest, ResolveGemma3ConfigFromTfliteLoader) {
  std::string buffer = BuildEmptyTfliteModelBuffer();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TfliteLoader loader,
                                  TfliteLoader::LoadFromBuffer(buffer));

  absl::StatusOr<Gemma3Config> auto_config =
      ResolveGemma3Config(loader, Gemma3ModelVariant::kAuto);
  EXPECT_EQ(auto_config.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(auto_config.status().message(),
              HasSubstr("not supported for TFLite models"));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Gemma3Config config_270m,
      ResolveGemma3Config(loader, Gemma3ModelVariant::k270M));
  EXPECT_EQ(config_270m.n_layers, GetGemma3_270M_Config().n_layers);
  EXPECT_EQ(config_270m.emb_dim, GetGemma3_270M_Config().emb_dim);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Gemma3Config config_1b,
      ResolveGemma3Config(loader, Gemma3ModelVariant::k1B));
  EXPECT_EQ(config_1b.n_layers, GetGemma3_1B_Config().n_layers);
  EXPECT_EQ(config_1b.emb_dim, GetGemma3_1B_Config().emb_dim);
}

TEST(ModelConfigTest, InferAndResolveGemma3ConfigFromSafetensorLoader) {
  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.embed_tokens.weight",
           .type = Type::kFP32,
           .shape = {32, 16},
           .buffer = std::vector<float>(32 * 16, 0.0f)},
          {.name = "model.layers.0.mlp.gate_proj.weight",
           .type = Type::kFP32,
           .shape = {64, 16},
           .buffer = std::vector<float>(64 * 16, 0.0f)},
          {.name = "model.layers.0.self_attn.q_proj.weight",
           .type = Type::kFP32,
           .shape = {32, 16},
           .buffer = std::vector<float>(32 * 16, 0.0f)},
          {.name = "model.layers.0.self_attn.k_proj.weight",
           .type = Type::kFP32,
           .shape = {16, 16},
           .buffer = std::vector<float>(16 * 16, 0.0f)},
          {.name = "model.layers.0.self_attn.q_norm.weight",
           .type = Type::kFP32,
           .shape = {8},
           .buffer = std::vector<float>(8, 1.0f)},
          {.name = "model.layers.3.mlp.down_proj.weight",
           .type = Type::kFP32,
           .shape = {16, 64},
           .buffer = std::vector<float>(16 * 64, 0.0f)},
      },
      /*quant_config_json=*/"");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Gemma3Config resolved,
      ResolveGemma3Config(loader, Gemma3ModelVariant::kAuto));
  EXPECT_EQ(resolved.n_layers, 4);
  EXPECT_EQ(resolved.vocab_size, 32);
  EXPECT_EQ(resolved.emb_dim, 16);
  EXPECT_EQ(resolved.hidden_dim, 64);
  EXPECT_EQ(resolved.head_dim, 8);
  EXPECT_EQ(resolved.n_heads, 4);
  EXPECT_EQ(resolved.n_kv_groups, 2);
  EXPECT_FLOAT_EQ(resolved.query_pre_attn_scalar, 8.0f);
  EXPECT_EQ(resolved.sliding_window, GetGemma3_270M_Config().sliding_window);

  // Explicit variant mismatch falls back to the inferred shape config.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Gemma3Config resolved_1b,
      ResolveGemma3Config(loader, Gemma3ModelVariant::k1B));
  EXPECT_EQ(resolved_1b.n_layers, 4);
  EXPECT_EQ(resolved_1b.emb_dim, 16);
  EXPECT_EQ(resolved_1b.sliding_window, GetGemma3_1B_Config().sliding_window);
}

TEST(ModelConfigTest, InferGemma3ConfigRejectsInvalidShapes) {
  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.embed_tokens.weight",
           .type = Type::kFP32,
           .shape = {32, 16},
           .buffer = std::vector<float>(32 * 16, 0.0f)},
          {.name = "model.layers.0.mlp.gate_proj.weight",
           .type = Type::kFP32,
           .shape = {64, 16},
           .buffer = std::vector<float>(64 * 16, 0.0f)},
          {.name = "model.layers.0.self_attn.q_proj.weight",
           .type = Type::kFP32,
           .shape = {30, 16},
           .buffer = std::vector<float>(30 * 16, 0.0f)},
          {.name = "model.layers.0.self_attn.k_proj.weight",
           .type = Type::kFP32,
           .shape = {16, 16},
           .buffer = std::vector<float>(16 * 16, 0.0f)},
          {.name = "model.layers.0.self_attn.q_norm.weight",
           .type = Type::kFP32,
           .shape = {8},
           .buffer = std::vector<float>(8, 1.0f)},
      },
      /*quant_config_json=*/"");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  absl::StatusOr<Gemma3Config> inferred =
      InferGemma3ConfigFromLoader(loader, GetGemma3_270M_Config());
  EXPECT_EQ(inferred.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(inferred.status().message(),
              HasSubstr("not divisible by head_dim"));
}

}  // namespace
}  // namespace litert::tensor::examples
