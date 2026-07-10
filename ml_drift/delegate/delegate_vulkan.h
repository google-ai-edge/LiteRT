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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_VULKAN_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_VULKAN_H_

#include "litert/c/litert_common.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_options.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_types.h"
#include "tflite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Deletes an existing ML Drift Vulkan delegate object incl. all its resources.
void LiteRtDeleteMlDriftVulkanDelegate(TfLiteDelegate* delegate);

#ifdef __cplusplus
}  // extern "C"

namespace litert::ml_drift {

// Additional ML Drift Vulkan Delegate C++ APIs.
//
// Typical usage:
//
//   // Initialize.
//   MlDriftDelegateOptionsPtr options =
//       MlDriftVulkanDelegateDefaultOptionsPtr();
//   tflite::TfLiteDelegatePtr delegate =
//       CreateMlDriftVulkanDelegate(std::move(options));
//
//   QCHECK(delegate != nullptr);
//   QCHECK_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
//             kTfLiteOk);
//
//   // Run inference.
//   QCHECK_EQ(interpreter->Invoke(), kTfLiteOk);

// Returns default options for ML Drift Vulkan delegate.
//
// This calls `MlDriftVulkanDelegateDefaultOptions()` add return the result in
// an RAII wrapper.
MlDriftDelegateOptionsPtr MlDriftVulkanDelegateDefaultOptionsPtr();

// Creates a new ML Drift Vulkan delegate object.
TfLiteDelegatePtr CreateMlDriftVulkanDelegate(MlDriftDelegateOptionsPtr options,
                                              LiteRtEnvironment env);

}  // namespace litert::ml_drift

#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_VULKAN_H_
