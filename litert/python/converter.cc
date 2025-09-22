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

#include "litert/python/converter.h"

#include <Python.h>

#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tflite/converter/converter_flags.pb.h"
#include "tflite/converter/model_flags.pb.h"
#include "tflite/converter/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/core/framework/graph.pb.h"

LiteRtConverter::~LiteRtConverter() = default;

void LiteRtConverter::addSignature(const ConverterSignature& signature){
  signatures_.push_back(signature);
}

absl::StatusOr<std::string> LiteRtConverter::convert() {
  auto model_flags_status = buildModelFlags();
  if (!model_flags_status.ok()) {
    set_python_error_from_status(model_flags_status.status());
    return absl::InvalidArgumentError(model_flags_status.status().message());
  }

  auto model_flags = model_flags_status.value();

  return absl::UnimplementedError("Not implemented yet.");
}

void LiteRtConverter::set_python_error_from_status(const absl::Status& status) {
  PyErr_SetString(PyExc_ValueError, std::string(status.message()).c_str());
}

absl::StatusOr<tflite::ModelFlags> LiteRtConverter::buildModelFlags() {
  return absl::UnimplementedError("Not implemented yet.");
}

absl::StatusOr<tflite::ConverterFlags> LiteRtConverter::buildConverterFlags() {
  return absl::UnimplementedError("Not implemented yet.");
}

absl::StatusOr<tensorflow::GraphDef> LiteRtConverter::buildGraphDef(
    const tflite::ModelFlags& model_flags) {
  tensorflow::GraphDef graph_def;
  if (!model_flags.use_hlo_import() &&
      !graph_def.ParseFromString(config_.input_contents_txt_raw)) {
    return absl::InvalidArgumentError(
        "Failed to parse GraphDef from input_contents_txt_raw.");
  }

  return graph_def;
}

mlir::TFL::QuantizationSpecs LiteRtConverter::buildQuantizationSpecs() {
  mlir::TFL::QuantizationSpecs specs;
  return specs;
}
