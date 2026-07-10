// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_UNIMPLEMENTED_OPERATION_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_UNIMPLEMENTED_OPERATION_PARSER_H_

#include <string>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/object_reader.h"
#include "third_party/odml/litert/ml_drift/tflite/operation_parser.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

class UnimplementedOperationParser : public TFLiteOperationParser {
 public:
  explicit UnimplementedOperationParser(absl::string_view op_name)
      : op_name_(op_name) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return absl::UnimplementedError(op_name_);
  }

  void Parse(const TfLiteNode* tflite_node,
             const TfLiteRegistration* registration,
             ::ml_drift::GraphFloat32* graph, ObjectReader* reader) final {}

 private:
  std::string op_name_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_UNIMPLEMENTED_OPERATION_PARSER_H_
