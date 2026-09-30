// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_TOOLS_BATCH_UTILS_H_
#define ODML_LITERT_LITERT_TOOLS_BATCH_UTILS_H_

#include <cstdint>

#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"

namespace litert::tools {

// Validates the model for batch dimension modification.
// Rejects models with shape-altering operations that move the dynamic batch
// from index 0, or models with dynamic dimensions beyond index 0.
LiteRtStatus ValidateModelForBatchFix(const LiteRtModelT& model);

// Updates the first dimension of all ranked tensors in the model to the target
// batch size.
void FixBatchDimension(LiteRtModelT& model, int32_t batch_size);

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_TOOLS_BATCH_UTILS_H_
