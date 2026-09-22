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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/safetensors.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples {
namespace {

using ::testing::Not;

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

// Returns the container type the safetensors file declares for a tensor of
// `type` values.
safetensors::dtype SafetensorDtype(Type type) {
  switch (type) {
    // `compressed-tensors` packs sub-byte fields into int32 containers.
    case Type::kI2:
    case Type::kI4:
    case Type::kI32:
      return safetensors::dtype::kINT32;
    case Type::kI8:
      return safetensors::dtype::kINT8;
    case Type::kI64:
      return safetensors::dtype::kINT64;
    case Type::kFP32:
      return safetensors::dtype::kFLOAT32;
    default:
      ADD_FAILURE() << "Unsupported test tensor type: " << ToString(type);
      return safetensors::dtype::kFLOAT32;
  }
}

// Safetensor compressed-tensors shift every packed field into unsigned range by
// adding pow(2, num_bits-1) before packing it into a container.
//
// `(v + pow(2, b-1)) % pow(2, b) == v ^ pow(2, b-1)` so we can XOR the mask
// returned to apply the shift.
constexpr uint8_t OffsetMask(Type type) {
  switch (type) {
    case Type::kI2:
      return 0b10101010;
    case Type::kI4:
      return 0b10001000;
    default:
      return 0;
  }
}

// Returns whether `compressed-tensors` packs `type` values.
constexpr bool IsPacked(Type type) { return OffsetMask(type) != 0; }

// Returns the shape the safetensors file declares for a tensor of `shape`
// `type` elements.
//
// Safetensors store the shape of the buffer container type instead of the shape
// of the buffer data type.
std::vector<size_t> SafetensorShape(Type type, const Shape& shape) {
  std::vector<size_t> file_shape(shape.begin(), shape.end());
  if (IsPacked(type) && !file_shape.empty()) {
    constexpr size_t kContainerSize = sizeof(int32_t);
    file_shape.back() =
        (BufferSize(type, file_shape.back()) + kContainerSize - 1) /
        kContainerSize;
  }
  return file_shape;
}

// RAII guard that will automatically remove the files in a given folder.
class SafetensorFileGuard {
 public:
  // Creates a guard for the folder pointed to by `path`.
  //
  // If `path` points to a '.safetensors' file, the containing folder will be
  // targetted.
  explicit SafetensorFileGuard(std::filesystem::path path) : file_(path) {
    if (file_.extension() != ".safetensors") {
      file_ /= "model.safetensors";
    }
    std::filesystem::create_directories(file_.parent_path());
  }

  SafetensorFileGuard(const SafetensorFileGuard&) = delete;
  SafetensorFileGuard& operator=(const SafetensorFileGuard&) = delete;

  SafetensorFileGuard(SafetensorFileGuard&& other)
      : file_(std::exchange(other.file_, std::filesystem::path())) {}
  SafetensorFileGuard& operator=(SafetensorFileGuard&& other) {
    Clear();
    file_ = std::exchange(other.file_, std::filesystem::path());
    return *this;
  };

  ~SafetensorFileGuard() { Clear(); }

  void Clear() {
    if (!file_.empty()) {
      auto Remove = [](std::filesystem::path p) {
        std::error_code ec;
        std::filesystem::remove(p, ec);
        if (ec) {
          FAIL() << "Could not remove " << p
                 << " because of error: " << ec.message();
        }
      };
      Remove(file_);
      Remove(file_.parent_path() / "config.json");
      Remove(file_.parent_path());
    }
  }

  static SafetensorFileGuard CreateTemp() {
    static std::atomic<int> counter = 0;
    return SafetensorFileGuard(
        std::filesystem::path(testing::TempDir()) /
        absl::StrCat("safetensor_loader_test_", counter++));
  }

  const std::filesystem::path& GetPath() const { return file_; }
  std::filesystem::path GetFolder() const { return file_.parent_path(); }
  std::filesystem::path GetConfigPath() const {
    return file_.parent_path() / "config.json";
  }

 private:
  std::filesystem::path file_;
};

// Writes `tensors` into a temporary safetensors file.
//
// - `quant_config_json` is written to the header metadata if not empty.
//
// Each file gets its own folder, so that a neighbouring config.json only
// affects the test that wrote it.
SafetensorFileGuard CreateTempSafetensor(const std::vector<TensorInit>& tensors,
                                         const std::string& quant_config_json) {
  SafetensorFileGuard temp_file = SafetensorFileGuard::CreateTemp();
  safetensors::safetensors_t st;
  if (!quant_config_json.empty()) {
    st.metadata.insert("quantization_config",
                       EscapeJsonString(quant_config_json));
  }

  // We use a tensor handle to process the buffer initialization.
  TensorHandle handle;
  for (const TensorInit& init : tensors) {
    handle.Set(init);
    safetensors::tensor_t entry;
    entry.dtype = SafetensorDtype(handle.GetType());
    entry.shape = SafetensorShape(handle.GetType(), handle.GetShape());
    std::shared_ptr<Buffer> buffer = handle.GetBufferPtr();
    if (buffer) {
      const LockedBufferSpan<const uint8_t> data =
          buffer->Lock().As<const uint8_t>();
      entry.data_offsets = {st.storage.size(), st.storage.size() + data.size()};
      const uint8_t mask = OffsetMask(handle.GetType());
      for (uint8_t byte : data) {
        st.storage.push_back(byte ^ mask);
      }
    }
    st.tensors.insert(init.name, entry);
  }

  std::string warn, err;
  EXPECT_TRUE(
      safetensors::save_to_file(st, temp_file.GetPath().string(), &warn, &err))
      << err;
  return temp_file;
}

SafetensorFileGuard CreateTempSafetensor(const std::string& quant_config_json) {
  return CreateTempSafetensor({{.name = "dummy_tensor",
                                .shape = {2, 2},
                                .buffer = std::vector<float>({1, 2, 3, 4})}},
                              quant_config_json);
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

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->quant_method,
            QuantizationConfig::Method::kCompressedTensors);
  EXPECT_EQ(quant_config->format, QuantizationConfig::Format::kPackQuantized);
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
  EXPECT_EQ(quant_config->format, QuantizationConfig::Format::kPackQuantized);
  EXPECT_EQ(quant_config->num_bits, 4);
  EXPECT_EQ(quant_config->group_size, 128);
  EXPECT_TRUE(quant_config->symmetric);
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

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->format, QuantizationConfig::Format::kIntQuantized);
  EXPECT_EQ(quant_config->num_bits, 8);
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

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const std::optional<QuantizationConfig>& quant_config =
      loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->format, QuantizationConfig::Format::kPackQuantized);
  EXPECT_EQ(quant_config->num_bits, 4);
  EXPECT_EQ(quant_config->group_size, 128);
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
