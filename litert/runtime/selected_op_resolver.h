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

#ifndef ODML_LITERT_LITERT_RUNTIME_SELECTED_OP_RESOLVER_H_
#define ODML_LITERT_LITERT_RUNTIME_SELECTED_OP_RESOLVER_H_

// Facade for generated registration code. Applications depend on the LiteRT
// helper rather than directly requesting access to restricted TFLite targets.
#include "litert/runtime/op_resolver.h"  // IWYU pragma: export
#include "tflite/kernels/builtin_op_kernels.h"  // IWYU pragma: export
#include "tflite/schema/schema_generated.h"  // IWYU pragma: export

#endif  // ODML_LITERT_LITERT_RUNTIME_SELECTED_OP_RESOLVER_H_
