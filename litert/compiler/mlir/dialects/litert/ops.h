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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_OPS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_OPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/IR/SymbolTable.h"  // IWYU pragma: keep
#include "mlir/Interfaces/InferTypeOpInterface.h"  // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h"  // IWYU pragma: keep
#include "litert/compiler/mlir/dialects/litert/traits.h"  // IWYU pragma: keep
#include "litert/compiler/mlir/dialects/litert/types.h"  // IWYU pragma: keep
// clang-format off
#include "litert/compiler/mlir/dialects/litert/ops_interface.h.inc"
// clang-format on

#define GET_OP_CLASSES
#include "litert/compiler/mlir/dialects/litert/ops.h.inc"  // IWYU pragma: export

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_OPS_H_
