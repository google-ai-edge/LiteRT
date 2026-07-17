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

#include "tensor/examples/gemma3/tflite_loader.h"

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/tensor.h"

namespace litert::tensor::examples {
namespace {

TEST(TfliteLoaderTest, LoadModelAndTensors) {
  // Path to a test model in the codebase.
  std::string model_path =
      "third_party/odml/litert/litert/test/testdata/ff.tflite";

  auto loader_or = TfliteLoader::Load(model_path);
  ASSERT_TRUE(loader_or.ok()) << loader_or.status().message();
  auto loader = std::move(*loader_or);

  auto names = loader.GetTensorNames();
  EXPECT_FALSE(names.empty());

  for (const auto& name : names) {
    auto info_or = loader.GetTensorInfo(name);
    ASSERT_TRUE(info_or.ok());

    // Try to load the tensor.
    auto tensor_or = loader.LoadTensor(
        name, TfliteLoader::QuantizedLoadMode::kPreserveQuantized);
    if (tensor_or.ok()) {
      auto tensor = std::move(*tensor_or);
      EXPECT_EQ(tensor.GetName(), name);
      auto buffer_or = tensor.GetBuffer();
      EXPECT_TRUE(buffer_or.ok());
    } else {
      FAIL() << "Failed to load tensor " << name << ": "
             << tensor_or.status().message();
    }
  }
}

TEST(TfliteLoaderTest, LoadModelWithMappingAndSlicing) {
  std::string model_path =
      "third_party/odml/litert/litert/test/testdata/ff.tflite";

  auto loader_or = TfliteLoader::Load(model_path);
  ASSERT_TRUE(loader_or.ok()) << loader_or.status().message();
  auto loader = std::move(*loader_or);

  TfliteWeightMapping mapping;
  mapping["weight_part_1"] = {"arith.constant", {0, 64}};
  mapping["weight_part_2"] = {"arith.constant", {64, 128}};
  mapping["whole_weight"] = {"arith.constant", {}};

  auto tensors_or = loader.LoadWeightsWithMapping(
      mapping, TfliteLoader::QuantizedLoadMode::kPreserveQuantized);
  ASSERT_TRUE(tensors_or.ok()) << tensors_or.status().message();
  auto tensors = std::move(*tensors_or);

  auto whole_it = tensors.find("whole_weight");
  ASSERT_TRUE(whole_it != tensors.end());
  EXPECT_EQ(whole_it->second.GetName(), "whole_weight");
  EXPECT_EQ(whole_it->second.GetShape(), std::vector<int>({128, 512}));

  auto part1_it = tensors.find("weight_part_1");
  ASSERT_TRUE(part1_it != tensors.end());
  EXPECT_EQ(part1_it->second.GetName(), "weight_part_1");
  EXPECT_EQ(part1_it->second.GetShape(), std::vector<int>({64, 512}));

  auto part2_it = tensors.find("weight_part_2");
  ASSERT_TRUE(part2_it != tensors.end());
  EXPECT_EQ(part2_it->second.GetName(), "weight_part_2");
  EXPECT_EQ(part2_it->second.GetShape(), std::vector<int>({64, 512}));
}

TEST(TfliteLoaderTest, Gemma3MappingTest) {
  int n_layers = 26;
  auto mapping = GetGemma3TfliteWeightMapping(n_layers);

  // 6 1D weights + 7 2D weights per layer = 13 weights per layer.
  // 26 layers * 13 = 338 weights.
  // + 1 embedding + 1 final norm = 340 weights total.
  EXPECT_EQ(mapping.size(), 340);

  // Verify some specific entries
  EXPECT_EQ(mapping["model.embed_tokens.weight"].tflite_tensor_name,
            "XlaCallModule/ReadVariableOp_287;StatefulPartitionedCall");
  EXPECT_TRUE(mapping["model.embed_tokens.weight"].slice_range.empty());

  EXPECT_EQ(mapping["model.norm.weight"].tflite_tensor_name,
            "arith.constant140");

  // Verify Layer 0 QKV split
  EXPECT_EQ(
      mapping["model.layers.0.self_attn.q_proj.weight"].tflite_tensor_name,
      "arith.constant129");
  EXPECT_EQ(mapping["model.layers.0.self_attn.q_proj.weight"].slice_range,
            std::vector<int>({0, 1024}));

  EXPECT_EQ(
      mapping["model.layers.0.self_attn.k_proj.weight"].tflite_tensor_name,
      "arith.constant129");
  EXPECT_EQ(mapping["model.layers.0.self_attn.k_proj.weight"].slice_range,
            std::vector<int>({1024, 1280}));

  EXPECT_EQ(
      mapping["model.layers.0.self_attn.v_proj.weight"].tflite_tensor_name,
      "arith.constant129");
  EXPECT_EQ(mapping["model.layers.0.self_attn.v_proj.weight"].slice_range,
            std::vector<int>({1280, 1536}));

  // Verify Layer 0 MLP
  EXPECT_EQ(mapping["model.layers.0.mlp.down_proj.weight"].tflite_tensor_name,
            "arith.constant125");
  EXPECT_TRUE(
      mapping["model.layers.0.mlp.down_proj.weight"].slice_range.empty());

  // Verify Layer 25 Out Proj
  EXPECT_EQ(
      mapping["model.layers.25.self_attn.o_proj.weight"].tflite_tensor_name,
      "arith.constant3");  // 128 - 5 * 25 = 3
}

}  // namespace
}  // namespace litert::tensor::examples
