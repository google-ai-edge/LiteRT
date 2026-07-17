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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_UTILS_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace litert::ml_drift {

// Creates a ML Drift tensor descriptor from the given LiteRT tensor type and
// buffer type.
absl::StatusOr<::ml_drift::TensorDescriptor> CreateTensorDescriptor(
    const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type);

// Converts the data in `desc` and stores in `host_memory`.
void ConvertDataToDescriptor(void* host_memory,
                             ::ml_drift::TensorDescriptor& desc,
                             LiteRtElementType src_type);
// Converts the data in `host_memory` and stores in `desc`.
void ConvertDataFromDescriptor(const ::ml_drift::TensorDescriptor& desc,
                               void* host_memory, LiteRtElementType dst_type);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_UTILS_H_
