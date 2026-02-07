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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "litert/compiler/mlir/dialects/litert/attributes.h"

#define GET_TYPEDEF_CLASSES
#include "litert/compiler/mlir/dialects/litert/types.h.inc"  // IWYU pragma: export

namespace litert {

mlir::RankedTensorType GetRankedTensorType(mlir::Type type);
mlir::RankedTensorType GetRankedTensorType(mlir::Value value);
litert::SymTensorType GetSymTensorType(mlir::Type type);
litert::SymTensorType GetSymTensorType(mlir::Value value);

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TYPES_H_
