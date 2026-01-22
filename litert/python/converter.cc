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

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tflite/converter/converter_flags.pb.h"
#include "tflite/converter/model_flags.pb.h"

namespace litert {

Converter::~Converter() = default;

void Converter::AddSignature(const std::string& signature_name,
                             const std::vector<std::string>& input_names,
                             const std::vector<std::string>& output_names,
                             const std::vector<std::byte>& data) {
  ConverterSignature signature;
  signature.signature_name = signature_name;
  signature.input_names = input_names;
  signature.output_names = output_names;
  signature.data = data;
  signatures_.push_back(signature);
}

absl::StatusOr<std::string> Converter::Convert() {
  return absl::UnimplementedError("Not implemented yet.");
}

}  // namespace litert
