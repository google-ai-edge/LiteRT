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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TRAITS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TRAITS_H_

#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

namespace litert {

mlir::Type GetBroadcastedType(mlir::Type type1, mlir::Type type2,
                              mlir::Type element_type = nullptr);

namespace impl {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes.  This avoids them being template
// instantiated/duplicated.
llvm::LogicalResult VerifyOperandsBroadcastable(mlir::Operation* op);

}  // namespace impl
}  // namespace litert

// MLIR requires traits to be in mlir::OpTrait namespace.
namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class BroadcastableOperands
    : public TraitBase<ConcreteType, BroadcastableOperands> {
 public:
  static llvm::LogicalResult verifyTrait(Operation* op) {
    return litert::impl::VerifyOperandsBroadcastable(op);
  }
};

}  // namespace OpTrait
}  // namespace mlir

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TRAITS_H_
