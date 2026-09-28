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

#include "tensor/examples/utils/safetensor_test_util.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/safetensors.h"
#include "tensor/tensor.h"

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

// Appends `data`, holding a tensor of `shape` `type` elements, to the `storage`
// of a safetensors file.
//
// Rows are appended one by one: `compressed-tensors` pads each row of a packed
// weight up to a whole container.
void AppendTensorData(Type type, const Shape& shape,
                      const LockedBufferSpan<const uint8_t>& data,
                      std::vector<uint8_t>& storage) {
  const size_t row_size =
      shape.empty() ? data.size() : BufferSize(type, shape.back());
  const size_t rows = row_size == 0 ? 0 : data.size() / row_size;
  if (rows * row_size != data.size()) {
    // A row of sub-byte elements that does not hold a whole number of bytes
    // would start in the middle of a byte, which neither a safetensors file nor
    // the loader can express.
    ADD_FAILURE() << "Rows of " << shape.back() << " " << ToString(type)
                  << " elements do not hold a whole number of bytes.";
    return;
  }
  const size_t container_size = IsPacked(type) ? sizeof(int32_t) : 1;
  const size_t padded_row_size =
      (row_size + container_size - 1) / container_size * container_size;
  const uint8_t mask = OffsetMask(type);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t i = row * row_size; i < (row + 1) * row_size; ++i) {
      storage.push_back(data.data()[i] ^ mask);
    }
    storage.insert(storage.end(), padded_row_size - row_size, 0);
  }
}

}  // namespace

SafetensorFileGuard::SafetensorFileGuard(std::filesystem::path path)
    : file_(std::move(path)) {
  if (file_.extension() != ".safetensors") {
    file_ /= "model.safetensors";
  }
  std::filesystem::create_directories(file_.parent_path());
}

void SafetensorFileGuard::Clear() {
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

SafetensorFileGuard SafetensorFileGuard::CreateTemp() {
  static std::atomic<int> counter = 0;
  return SafetensorFileGuard(std::filesystem::path(testing::TempDir()) /
                             absl::StrCat("safetensor_test_", counter++));
}

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
      const size_t start = st.storage.size();
      AppendTensorData(handle.GetType(), handle.GetShape(),
                       buffer->Lock().As<const uint8_t>(), st.storage);
      entry.data_offsets = {start, st.storage.size()};
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

}  // namespace litert::tensor::examples
