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

#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
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

}  // namespace
}  // namespace litert::tensor::examples
