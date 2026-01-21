// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_PYTHON_CONVERTER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_PYTHON_CONVERTER_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "tflite/converter/converter_flags.pb.h"
#include "tflite/converter/model_flags.pb.h"

namespace litert {

struct ConversionConfig {
  // The original model type that the converter is converting from
  // This corresponds to ModelType in
  // third_party/tensorflow/compiler/mlir/lite/schema/conversion_metadata.fbs
  enum class ModelType : unsigned char {
    kUnknown = 0,
    kJax = 6,
    kPyTorch = 7,
  };

  tflite::ConverterFlags converter_flags;
  tflite::ModelFlags model_flags;
  std::string model_flags_txt_raw;
  std::string converter_flags_txt_raw;
  ModelType original_model_type = ModelType::kUnknown;
};

class Converter {
 public:
  explicit Converter(const ConversionConfig& config) : config_(config) {};
  ~Converter();

  void AddSignature(const std::string& signature_name,
                    const std::vector<std::string>& input_names,
                    const std::vector<std::string>& output_names,
                    const std::vector<std::byte>& data);

  absl::StatusOr<std::string> Convert();

 private:
  struct ConverterSignature {
    std::string signature_name;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::byte> data;
  };

  ConversionConfig config_;
  std::vector<ConverterSignature> signatures_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_PYTHON_CONVERTER_H_
