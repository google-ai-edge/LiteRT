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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_SUPPORT_TRANSPOSE_CONV_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_SUPPORT_TRANSPOSE_CONV_H_

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

// Validates a transpose_conv operation node to determine if it's supported.
// Note that we must match tensorflow/lite/delegates/utils.h (IsNodeSupported).
// In an ideal world we would pass by reference and return absl::Status.
//
// This function checks the node's version, I/O tensors, data types,
// dimensions, and operation-specific parameters. On failure, it populates the
// `error` string with a descriptive message and returns false.
bool IsTransposeConvSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    std::string* absl_nonnull error);

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SUPPORT_SUPPORT_TRANSPOSE_CONV_H_
