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
#include "litert/compiler/mlir/dialects/litert/dialect.h"

#include "absl/log/log.h"  // from @com_google_absl
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/Dialect/Arith/IR/Arith.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"  // IWYU pragma: keep
#include "mlir/IR/Dialect.h"  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"
#include "litert/compiler/mlir/dialects/litert/attributes.h"
#include "litert/compiler/mlir/dialects/litert/callback_resource.h"
#include "litert/compiler/mlir/dialects/litert/lazy_blob_manager.h"
// clang-format off
#include "litert/compiler/mlir/dialects/litert/dialect.cc.inc"  // IWYU pragma: keep
// clang-format on

namespace litert {

void LITERTDialect::initialize() {
  registerOps();
  addInterface<LazyBlobManagerDialectInterface>();
  addInterface<CallbackResourceManagerDialectInterface>();
  registerAttributes();
  registerTypes();
}

}  // namespace litert

namespace litert {

mlir::Operation* LITERTDialect::materializeConstant(mlir::OpBuilder& builder,
                                                    mlir::Attribute value,
                                                    mlir::Type type,
                                                    mlir::Location loc) {
  return mlir::arith::ConstantOp::materialize(builder, value, type, loc);
}

}  // namespace litert
