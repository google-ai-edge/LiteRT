// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_CUSTOM_PARSERS_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_CUSTOM_PARSERS_H_

#include <memory>
#include <string_view>

#include "ml_drift_delegate/tflite/operation_parser.h"

namespace litert::ml_drift {

// Factory to create custom operation parsers for new LiteRT composite ops.
class CustomOperationParserFactory
    : public TFLiteStablehloCompositeParserFactory {
 public:
  ~CustomOperationParserFactory() override = default;

  // TFLiteStablehloCompositeParserFactory implementation.
  std::unique_ptr<TFLiteOperationParser> Create(
      std::string_view op_name) override;
  bool SupportsIntegerTypes(std::string_view op_name) override;
  bool SupportsBoolTypes(std::string_view op_name) override;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_CUSTOM_PARSERS_H_
