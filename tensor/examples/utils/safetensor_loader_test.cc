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

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/examples/utils/safetensors.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples {
namespace {

std::string EscapeJsonString(const std::string& input) {
  std::string output;
  for (char c : input) {
    if (c == '"') {
      output += "\\\"";
    } else if (c == '\n') {
      output += "\\n";
    } else {
      output += c;
    }
  }
  return output;
}

std::string CreateTempSafetensor(const std::string& quant_config_json) {
  static int counter = 0;
  const std::string temp_path =
      (std::filesystem::path(testing::TempDir()) /
       absl::StrCat("test_safetensor_", counter++, ".safetensors"))
          .string();

  safetensors::safetensors_t st;
  if (!quant_config_json.empty()) {
    st.metadata.insert("quantization_config",
                       EscapeJsonString(quant_config_json));
  }

  safetensors::tensor_t tensor;
  tensor.dtype = safetensors::dtype::kFLOAT32;
  tensor.shape = {2, 2};
  tensor.data_offsets = {0, 16};
  st.tensors.insert("dummy_tensor", tensor);
  st.storage = std::vector<uint8_t>(16, 0);

  std::string warn, err;
  EXPECT_TRUE(safetensors::save_to_file(st, temp_path, &warn, &err)) << err;
  return temp_path;
}

TEST(SafetensorLoaderTest, AbslStringifyMethodAndFormat) {
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Method::kCompressedTensors),
            "compressed-tensors");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Method::kUnknown), "unknown");

  EXPECT_EQ(absl::StrCat(QuantizationConfig::Format::kPackQuantized),
            "pack-quantized");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Format::kIntQuantized),
            "int-quantized");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Format::kUnknown), "unknown");
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

  std::string file_path = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file_path));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->quant_method,
            QuantizationConfig::Method::kCompressedTensors);
  EXPECT_EQ(quant_config->format, QuantizationConfig::Format::kPackQuantized);
  EXPECT_EQ(quant_config->num_bits, 4);
  EXPECT_EQ(quant_config->group_size, 128);
  EXPECT_TRUE(quant_config->symmetric);

  std::remove(file_path.c_str());
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

  std::string file_path = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file_path));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->quant_method,
            QuantizationConfig::Method::kCompressedTensors);
  EXPECT_EQ(quant_config->format, QuantizationConfig::Format::kPackQuantized);
  EXPECT_EQ(quant_config->num_bits, 4);
  EXPECT_EQ(quant_config->group_size, 128);
  EXPECT_TRUE(quant_config->symmetric);

  std::remove(file_path.c_str());
}

TEST(SafetensorLoaderTest, ParseIntQuantizedFormat) {
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

  std::string file_path = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file_path));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->format, QuantizationConfig::Format::kIntQuantized);
  EXPECT_EQ(quant_config->num_bits, 8);

  std::remove(file_path.c_str());
}

TEST(SafetensorLoaderTest, GroupFormatOverridesTopLevelFormat) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "config_groups": {
      "group_0": {
        "format": "pack-quantized",
        "weights": {
          "num_bits": 4,
          "group_size": 128
        }
      }
    }
  })";

  std::string file_path = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file_path));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->format, QuantizationConfig::Format::kPackQuantized);
  EXPECT_EQ(quant_config->num_bits, 4);
  EXPECT_EQ(quant_config->group_size, 128);

  std::remove(file_path.c_str());
}

TEST(SafetensorLoaderTest, RejectMultipleConfigGroups) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": {
        "weights": { "num_bits": 4, "group_size": 128 }
      },
      "group_1": {
        "weights": { "num_bits": 8, "group_size": 64 }
      }
    }
  })";

  std::string file_path = CreateTempSafetensor(json);
  auto loader_or = SafetensorLoader::Load(file_path);
  EXPECT_FALSE(loader_or.ok());

  std::remove(file_path.c_str());
}

TEST(SafetensorLoaderTest, RejectNonSpecUnderscoreFormat) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack_quantized",
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 4,
          "group_size": 128
        }
      }
    }
  })";

  std::string file_path = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file_path));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  // "pack_quantized" with underscore must map to kUnknown under strict spec
  // validation
  EXPECT_EQ(quant_config->format, QuantizationConfig::Format::kUnknown);

  std::remove(file_path.c_str());
}

TEST(SafetensorLoaderTest, MalformedJsonFails) {
  std::string invalid_json =
      R"({ "quant_method": "compressed-tensors", format: })";

  std::string file_path = CreateTempSafetensor(invalid_json);
  auto loader_or = SafetensorLoader::Load(file_path);
  EXPECT_FALSE(loader_or.ok());

  std::remove(file_path.c_str());
}

const char kMixedConfig[] = R"json({
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "two": {
        "targets": ["model.two", "re:.*lm_head$"],
        "weights": {"num_bits": 2, "type": "int", "dynamic": false,
                    "symmetric": true, "strategy": "channel"}
      },
      "four": {
        "targets": ["re:model\\.four$"],
        "weights": {"num_bits": 4, "type": "int", "dynamic": false,
                    "symmetric": true, "strategy": "group", "group_size": 4}
      },
      "eight": {
        "format": "int-quantized", "targets": ["model.eight"],
        "weights": {"num_bits": 8, "type": "int", "dynamic": false,
                    "symmetric": true, "strategy": "channel"},
        "input_activations": {"num_bits": 8, "type": "int", "dynamic": false,
                              "symmetric": true, "strategy": "tensor"},
        "output_activations": {"num_bits": 8, "type": "int", "dynamic": false,
                               "symmetric": true, "strategy": "tensor"}
      }
    }
  }
})json";

class CompressedTensorFixture {
 public:
  CompressedTensorFixture() {
    static int counter = 0;
    directory_ = std::filesystem::path(testing::TempDir()) /
                 absl::StrCat("tensor_ct_", getpid(), "_", counter++);
    std::filesystem::create_directories(directory_);
  }
  ~CompressedTensorFixture() { std::filesystem::remove_all(directory_); }

  template <class T>
  void Add(const std::string& name, safetensors::dtype type,
           std::vector<size_t> shape, const std::vector<T>& values) {
    const size_t start = file_.storage.size();
    file_.storage.resize(start + values.size() * sizeof(T));
    std::memcpy(file_.storage.data() + start, values.data(),
                values.size() * sizeof(T));
    safetensors::tensor_t tensor;
    tensor.dtype = type;
    tensor.shape = std::move(shape);
    tensor.data_offsets = {start, file_.storage.size()};
    file_.tensors.insert(name, tensor);
  }

  void TwoBit(bool add_scale = true) {
    Add<uint32_t>("model.two.weight_packed", safetensors::kINT32, {1, 1},
                  {0xE4E4E4E4});
    Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2}, {1, 16});
    if (add_scale)
      Add<uint16_t>("model.two.weight_scale", safetensors::kBFLOAT16, {1, 1},
                    {0x3F00});
  }

  void EightBit(bool activation_scales = true) {
    Add<int8_t>("model.eight.weight", safetensors::kINT8, {2, 3},
                {-128, -1, 127, 0, 64, -64});
    Add<float>("model.eight.weight_scale", safetensors::kFLOAT32, {2, 1},
               {0.25f, 0.5f});
    if (activation_scales) {
      Add<float>("model.eight.input_scale", safetensors::kFLOAT32, {},
                 {0.125f});
      Add<float>("model.eight.output_scale", safetensors::kFLOAT32, {},
                 {0.25f});
    }
  }

  void Save(const std::string& config = kMixedConfig) {
    std::string warning, error;
    ASSERT_TRUE(safetensors::save_to_file(file_, FilePath(), &warning, &error))
        << error;
    std::ofstream output(directory_ / "config.json");
    output << config;
    ASSERT_TRUE(output.good());
  }

  std::string FilePath() const {
    return (directory_ / "model.safetensors").string();
  }
  std::string Directory() const { return directory_.string(); }

 private:
  std::filesystem::path directory_;
  safetensors::safetensors_t file_;
};

std::vector<uint8_t> Bytes(const TensorHandle& tensor) {
  auto lock = tensor.GetBufferPtr()->Lock();
  const auto* data = reinterpret_cast<const uint8_t*>(lock.data());
  return {data, data + lock.size()};
}

TEST(SafetensorLoaderTest, MixedCompanionConfigDecodesSignedWeightsAndScales) {
  CompressedTensorFixture fixture;
  fixture.TwoBit();
  fixture.EightBit();
  fixture.Add<uint32_t>("model.four.weight_packed", safetensors::kINT32, {2, 1},
                        {0x76543210, 0xFEDCBA98});
  fixture.Add<int64_t>("model.four.weight_shape", safetensors::kINT64, {2},
                       {2, 8});
  fixture.Add<uint16_t>("model.four.weight_scale", safetensors::kBFLOAT16,
                        {2, 2}, {0x3F80, 0x4000, 0x4080, 0x4100});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.Directory()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto two,
                                  loader.LoadTensor("model.two.weight"));
  EXPECT_EQ(two.GetType(), Type::kI4);
  EXPECT_EQ(two.GetShape(), Shape({1, 16}));
  EXPECT_EQ(Bytes(two), std::vector<uint8_t>(
                            {0xFE, 0x10, 0xFE, 0x10, 0xFE, 0x10, 0xFE, 0x10}));
  ASSERT_NE(two.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const auto& channel,
      two.GetQuantization()->As<PerChannelAffineQuantization>());
  EXPECT_EQ(channel.scales, std::vector<float>({0.5f}));
  EXPECT_EQ(channel.zero_points, std::vector<int64_t>({0}));
  EXPECT_EQ(channel.quantized_dimension, 0);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto four,
                                  loader.LoadTensor("model.four.weight"));
  EXPECT_EQ(four.GetShape(), Shape({2, 8}));
  EXPECT_EQ(Bytes(four), std::vector<uint8_t>(
                             {0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76}));
  ASSERT_NE(four.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const auto& grouped, four.GetQuantization()->As<BlockwiseQuantization>());
  EXPECT_EQ(grouped.block_size, 4);
  EXPECT_EQ(grouped.scales, std::vector<float>({1, 2, 4, 8}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto eight,
                                  loader.LoadTensor("model.eight.weight"));
  EXPECT_EQ(eight.GetType(), Type::kI8);
  EXPECT_EQ(Bytes(eight), std::vector<uint8_t>({128, 255, 127, 0, 64, 192}));
}

TEST(SafetensorLoaderTest, MappingIncludesActivationScalesAndExplicitLmHead) {
  CompressedTensorFixture fixture;
  fixture.EightBit();
  fixture.Add<uint32_t>("lm_head.weight_packed", safetensors::kINT32, {1, 1},
                        {0xE4E4E4E4});
  fixture.Add<int64_t>("lm_head.weight_shape", safetensors::kINT64, {2},
                       {1, 16});
  fixture.Add<float>("lm_head.weight_scale", safetensors::kFLOAT32, {1, 1},
                     {1.0f});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto mapped,
      loader.LoadWeightsWithMapping({{"model.eight.weight", "renamed.weight"},
                                     {"lm_head.weight", "lm_head.weight"}}));
  ASSERT_TRUE(mapped.contains("renamed.weight"));
  ASSERT_TRUE(mapped.contains("renamed.input_scale"));
  ASSERT_TRUE(mapped.contains("renamed.output_scale"));
  ASSERT_TRUE(mapped.contains("lm_head.weight"));
  EXPECT_EQ(mapped.at("lm_head.weight").GetType(), Type::kI4);
  EXPECT_EQ(mapped.at("renamed.input_scale").GetName(), "renamed.input_scale");
  const auto scale =
      mapped.at("renamed.input_scale").GetBufferPtr()->Lock().As<const float>();
  EXPECT_FLOAT_EQ(scale.data()[0], 0.125f);
}

TEST(SafetensorLoaderTest, PackedRowsTrimPaddingAtOriginalShape) {
  CompressedTensorFixture fixture;
  fixture.Add<uint32_t>("model.two.weight_packed", safetensors::kINT32, {2, 1},
                        {0x000000E4, 0x0000031B});
  fixture.Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2},
                       {2, 5});
  fixture.Add<float>("model.two.weight_scale", safetensors::kFLOAT32, {2, 1},
                     {1, 1});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto tensor,
                                  loader.LoadTensor("model.two.weight"));
  EXPECT_EQ(Bytes(tensor),
            std::vector<uint8_t>({0xFE, 0x10, 0x1E, 0xF0, 0x1E}));
  EXPECT_EQ(tensor.GetShape(), Shape({2, 5}));
}

TEST(SafetensorLoaderTest,
     ConvertsFloatWeightsButKeepsPerLayerEmbeddingMapped) {
  CompressedTensorFixture fixture;
  fixture.Add<uint16_t>("norm.weight", safetensors::kBFLOAT16, {3},
                        {0x3F80, 0xC000, 0x3F00});
  fixture.Add<uint16_t>("projection.weight", safetensors::kFLOAT16, {3},
                        {0x3C00, 0xC000, 0x3800});
  fixture.Add<uint16_t>("model.embed_tokens_per_layer.weight",
                        safetensors::kBFLOAT16, {1, 3},
                        {0x3F80, 0xC000, 0x3F00});
  fixture.Save("{}");
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  for (const char* name : {"norm.weight", "projection.weight"}) {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto tensor, loader.LoadTensor(name));
    EXPECT_EQ(tensor.GetType(), Type::kFP32);
    const auto lock = tensor.GetBufferPtr()->Lock().As<const float>();
    EXPECT_EQ(std::vector<float>(lock.begin(), lock.end()),
              std::vector<float>({1, -2, 0.5f}));
  }
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto embedding, loader.LoadTensor("model.embed_tokens_per_layer.weight"));
  EXPECT_EQ(embedding.GetType(), Type::kBF16);
}

TEST(SafetensorLoaderTest, MissingWeightScaleIsAnErrorIncludingMappedLoad) {
  CompressedTensorFixture fixture;
  fixture.TwoBit(false);
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
  EXPECT_FALSE(
      loader.LoadWeightsWithMapping({{"model.two.weight", "two.weight"}}).ok());
}

TEST(SafetensorLoaderTest, RejectsPackedShapeDisagreement) {
  CompressedTensorFixture fixture;
  fixture.Add<uint32_t>("model.two.weight_packed", safetensors::kINT32, {1, 1},
                        {0});
  fixture.Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2},
                       {1, 17});
  fixture.Add<float>("model.two.weight_scale", safetensors::kFLOAT32, {1, 1},
                     {1});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
}

TEST(SafetensorLoaderTest, RejectsWrongPackedDtype) {
  CompressedTensorFixture fixture;
  fixture.Add<float>("model.two.weight_packed", safetensors::kFLOAT32, {1, 1},
                     {0});
  fixture.Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2},
                       {1, 16});
  fixture.Add<float>("model.two.weight_scale", safetensors::kFLOAT32, {1, 1},
                     {1});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
}

TEST(SafetensorLoaderTest, RejectsNonpositiveScalesAndNonzeroZeroPoints) {
  for (bool bad_scale : {true, false}) {
    CompressedTensorFixture fixture;
    fixture.Add<uint32_t>("model.two.weight_packed", safetensors::kINT32,
                          {1, 1}, {0});
    fixture.Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2},
                         {1, 16});
    fixture.Add<float>("model.two.weight_scale", safetensors::kFLOAT32, {1, 1},
                       {bad_scale ? 0.0f : 1.0f});
    fixture.Add<int8_t>("model.two.weight_zero_point", safetensors::kINT8,
                        {1, 1}, {bad_scale ? int8_t{0} : int8_t{1}});
    fixture.Save();
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                    SafetensorLoader::Load(fixture.FilePath()));
    EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
  }
}

TEST(SafetensorLoaderTest, RejectsMissingStaticActivationScale) {
  CompressedTensorFixture fixture;
  fixture.EightBit(false);
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.eight.weight").ok());
}

TEST(SafetensorLoaderTest, RejectsAmbiguousTargetGroups) {
  CompressedTensorFixture fixture;
  fixture.TwoBit();
  std::string config = kMixedConfig;
  const auto position = config.find("re:model\\\\.four$");
  ASSERT_NE(position, std::string::npos);
  config.replace(position, std::string("re:model\\\\.four$").size(),
                 "model.two");
  fixture.Save(config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
}

TEST(SafetensorLoaderTest, MappedInt8BufferOutlivesLoaderAndFile) {
  TensorHandle tensor;
  {
    CompressedTensorFixture fixture;
    fixture.EightBit();
    fixture.Save();
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                    SafetensorLoader::Load(fixture.FilePath()));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(tensor,
                                    loader.LoadTensor("model.eight.weight"));
  }
  EXPECT_EQ(Bytes(tensor), std::vector<uint8_t>({128, 255, 127, 0, 64, 192}));
}

TEST(SafetensorLoaderTest, TwoBitGroupedWeightsPreserveEachGroupScale) {
  CompressedTensorFixture fixture;
  fixture.Add<uint32_t>("model.four.weight_packed", safetensors::kINT32, {1, 1},
                        {0xE4E4E4E4});
  fixture.Add<int64_t>("model.four.weight_shape", safetensors::kINT64, {2},
                       {1, 16});
  fixture.Add<float>("model.four.weight_scale", safetensors::kFLOAT32, {1, 4},
                     {1, 2, 4, 8});
  std::string config = kMixedConfig;
  const std::string before = "\"num_bits\": 4";
  const auto position = config.find(before);
  ASSERT_NE(position, std::string::npos);
  config.replace(position, before.size(), "\"num_bits\": 2");
  fixture.Save(config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto tensor,
                                  loader.LoadTensor("model.four.weight"));
  EXPECT_EQ(tensor.GetType(), Type::kI4);
  EXPECT_EQ(Bytes(tensor), std::vector<uint8_t>({0xFE, 0x10, 0xFE, 0x10, 0xFE,
                                                 0x10, 0xFE, 0x10}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const auto& quantization,
      tensor.GetQuantization()->As<BlockwiseQuantization>());
  EXPECT_EQ(quantization.block_size, 4);
  EXPECT_EQ(quantization.scales, std::vector<float>({1, 2, 4, 8}));
}

TEST(SafetensorLoaderTest, RejectsScaleShapeDisagreement) {
  CompressedTensorFixture fixture;
  fixture.Add<uint32_t>("model.two.weight_packed", safetensors::kINT32, {2, 1},
                        {0, 0});
  fixture.Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2},
                       {2, 16});
  fixture.Add<float>("model.two.weight_scale", safetensors::kFLOAT32, {1, 2},
                     {1, 1});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
}

TEST(SafetensorLoaderTest, CompanionConfigAcceptsScientificNotation) {
  for (const char* number : {"1e-06", "2E+3", "3e4", "-4.5E-2"}) {
    CompressedTensorFixture fixture;
    fixture.TwoBit();
    std::string config = kMixedConfig;
    config.insert(1, absl::StrCat("\"rms_norm_eps\":", number, ","));
    fixture.Save(config);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                    SafetensorLoader::Load(fixture.FilePath()));
    EXPECT_TRUE(loader.LoadTensor("model.two.weight").ok());
    const char* position = number;
    minijson::value parsed;
    ASSERT_EQ(minijson::parse(position, parsed), minijson::no_error);
    ASSERT_NE(parsed.as<minijson::number>(), nullptr);
    EXPECT_DOUBLE_EQ(*parsed.as<minijson::number>(), std::stod(number));
  }
}

TEST(SafetensorLoaderTest, CompanionConfigRejectsIncompleteExponent) {
  for (const char* number : {"1e", "1e+", "1e-", "1E", "1E+", "1E-"}) {
    CompressedTensorFixture fixture;
    fixture.TwoBit();
    std::string config = kMixedConfig;
    config.insert(1, absl::StrCat("\"rms_norm_eps\":", number, ","));
    fixture.Save(config);
    EXPECT_FALSE(SafetensorLoader::Load(fixture.FilePath()).ok());
  }
}

}  // namespace
}  // namespace litert::tensor::examples
