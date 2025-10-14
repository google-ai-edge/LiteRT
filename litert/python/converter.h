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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tflite/converter/converter_flags.pb.h"
#include "tflite/converter/model_flags.pb.h"
#include "tflite/converter/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/core/framework/graph.pb.h"

struct ConversionConfig {
  enum class ModelType : unsigned char {
    Jax = 6,
    PyTorch = 7,
  };

  std::string input_contents_txt_raw;
  ModelType original_model_type;
};

struct ConverterSignature {
  std::string signature_name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::byte> data;
};

class LiteRtConverter {
 public:
  explicit LiteRtConverter(const ConversionConfig& config) : config_(config) {};
  ~LiteRtConverter();

  void addSignature(const ConverterSignature& signature);

  absl::StatusOr<std::string> convert();

 private:
  ConversionConfig config_;
  std::vector<ConverterSignature> signatures_;

  void set_python_error_from_status(const absl::Status& status);
  absl::StatusOr<tensorflow::GraphDef> buildGraphDef(
      const tflite::ModelFlags& model_flags);
  absl::StatusOr<tflite::ModelFlags> buildModelFlags();
  absl::StatusOr<tflite::ConverterFlags> buildConverterFlags();
  mlir::TFL::QuantizationSpecs buildQuantizationSpecs();
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_PYTHON_CONVERTER_H_
