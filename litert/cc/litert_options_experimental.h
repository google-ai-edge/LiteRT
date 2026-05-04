// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CC_LITERT_OPTIONS_EXPERIMENTAL_H_
#define ODML_LITERT_LITERT_CC_LITERT_OPTIONS_EXPERIMENTAL_H_

#include "litert/cc/litert_options.h"

struct TfLiteRegistration;
struct TfLiteOperator;

namespace litert {
namespace internal {

Expected<void> AddCustomOp(Options& options, const TfLiteRegistration* reg);
Expected<void> AddCustomOp(Options& options, const TfLiteOperator* op);

}  // namespace internal
}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_OPTIONS_EXPERIMENTAL_H_
