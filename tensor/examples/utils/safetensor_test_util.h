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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_SAFETENSOR_TEST_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_SAFETENSOR_TEST_UTIL_H_

#include <filesystem>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "tensor/tensor.h"

namespace litert::tensor::examples {

// RAII guard that will automatically remove the files in a given folder.
class SafetensorFileGuard {
 public:
  // Creates a guard for the folder pointed to by `path`.
  //
  // If `path` points to a '.safetensors' file, the containing folder will be
  // targetted.
  explicit SafetensorFileGuard(std::filesystem::path path);

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

  void Clear();

  static SafetensorFileGuard CreateTemp();

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
                                         const std::string& quant_config_json);

// Writes a single dummy FP32 tensor into a temporary safetensors file.
//
// - `quant_config_json` is written to the header metadata if not empty.
SafetensorFileGuard CreateTempSafetensor(const std::string& quant_config_json);

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_SAFETENSOR_TEST_UTIL_H_
