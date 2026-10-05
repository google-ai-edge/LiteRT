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

#ifndef ODML_LITERT_LITERT_RUNTIME_OP_RESOLVER_H_
#define ODML_LITERT_LITERT_RUNTIME_OP_RESOLVER_H_

#include <memory>

#include "litert/cc/litert_expected.h"
#include "tflite/mutable_op_resolver.h"

namespace litert::internal {

// Link-time customization point for statically linked CompiledModel runtimes.
// The build selects exactly one implementation. Each call returns a fresh
// resolver, to which CompiledModel may add model-specific custom operators.
Expected<std::unique_ptr<tflite::MutableOpResolver>> CreateOpResolver(
    bool use_reference_kernels);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_OP_RESOLVER_H_
