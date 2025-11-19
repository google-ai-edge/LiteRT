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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_MAGIC_NUMBER_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_MAGIC_NUMBER_UTILS_H_

#include "litert/cc/litert_expected.h"
#include "litert/core/environment.h"
#include "litert/core/model/model.h"

namespace litert::internal {

// Replaces magic numbers in the model if any.
//
// Magic numbers are prime numbers to identify the placeholders of the
// dimensions in some input tensors in the model that are supposed to be
// replaced with actual dimensions. It is to support dynamic shapes on runtime
// backends that do not support dynamic shapes including GPUs.
//
// What magic numbers are replaced with what real dimensions is defined in
// LiteRtMagicNumberConfigs in `env`. See litert/c/litert_environment_options.h
// for more details.
//
// Returns the number of nodes has been changed.
Expected<int> ReplaceMagicNumbersIfAny(const LiteRtEnvironmentT& env,
                                       LiteRtModelT& model);

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_MAGIC_NUMBER_UTILS_H_
