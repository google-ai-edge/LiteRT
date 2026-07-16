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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ML_DRIFT_DELEGATE_CREATE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ML_DRIFT_DELEGATE_CREATE_H_

#include <memory>

#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_gpu_options.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/delegate_types.h"

namespace litert::ml_drift {

using DelegateCreator = TfLiteDelegatePtr (*)(MlDriftDelegateOptionsPtr,
                                              LiteRtEnvironment);

// Creates a new ML Drift delegate object.
LiteRtStatus CreateDelegate(
    LiteRtRuntimeContext* runtime_context, LiteRtEnvironment env,
    LiteRtAccelerator accelerator, LrtGpuOptions* gpu_options_payload,
    std::unique_ptr<MlDriftDelegateOptions> gpu_delegate_options,
    DelegateCreator delegate_creator, TfLiteDelegatePtr& delegate);

LrtGpuOptions* GetGpuOptionsPayload(LiteRtRuntimeContext* runtime_context,
                                    LiteRtOptions options);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ML_DRIFT_DELEGATE_CREATE_H_
