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

#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl

struct ConversionConfig {};

struct ConverterSignature {};

class LiteRtConverter {
 public:
  explicit LiteRtConverter(const ConversionConfig& config) : config_(config) {};
  ~LiteRtConverter();

  void addSignature(const ConverterSignature& signature);

  absl::StatusOr<std::string> convert();

 private:
  ConversionConfig config_;
  std::vector<ConverterSignature> signatures_;
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_PYTHON_CONVERTER_H_
