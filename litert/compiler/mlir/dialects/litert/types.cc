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
#include "litert/compiler/mlir/dialects/litert/types.h"

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "litert/compiler/mlir/dialects/litert/attributes.h"
#include "litert/compiler/mlir/dialects/litert/dialect.h"

#define GET_TYPEDEF_CLASSES
#include "litert/compiler/mlir/dialects/litert/types.cc.inc"  // IWYU pragma: keep

namespace litert {

void LITERTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "litert/compiler/mlir/dialects/litert/types.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SymbolicRankedTensorType
//===----------------------------------------------------------------------===//

mlir::Type SymTensorType::parse(mlir::AsmParser& parser) {
  llvm::SmallVector<SymDimAttr, 4> dimensions;
  auto parse_fn = [&]() {
    llvm::APInt size;
    if (parser.parseOptionalInteger(size).has_value()) {
      dimensions.push_back(
          SymDimAttr::get(parser.getContext(), size.getSExtValue()));
      return mlir::success();
    }
    if (mlir::succeeded(parser.parseOptionalQuestion())) {
      dimensions.push_back(
          SymDimAttr::get(parser.getContext(), mlir::ShapedType::kDynamic));
      return mlir::success();
    }

    mlir::StringAttr sym_name;
    if (mlir::succeeded(parser.parseOptionalSymbolName(sym_name))) {
      auto sym_attr = SymDimAttr::get(parser.getContext(),
                                      mlir::SymbolRefAttr::get(sym_name));
      dimensions.push_back(sym_attr);
      return mlir::success();
    }

    return mlir::failure();
  };

  if (parser.parseLess()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected opening '<' bracket");
    return nullptr;
  }

  if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::None,
                                     parse_fn)) {
    parser.emitError(parser.getCurrentLocation(),
                     "unable to parse dimensions list");
    return nullptr;
  }

  if (dimensions.empty()) {
    parser.emitError(parser.getCurrentLocation(), "empty dimensions list");
    return nullptr;
  }

  if (parser.parseColon()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected colon after dimensions list");
    return nullptr;
  }

  mlir::Type element_type;
  if (parser.parseType(element_type) || !element_type) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected element type after colon");
    return nullptr;
  }

  // Parse an optional encoding attribute.
  mlir::Attribute encoding;
  if (mlir::succeeded(parser.parseOptionalComma())) {
    // If a comma is present, an encoding attribute MUST follow.
    if (parser.parseAttribute(encoding)) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected encoding attribute after comma");
      return nullptr;
    }
    // TODO(aarfaian): consider verifying the encoding here.
  }

  if (parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected closing '>' bracket");
    return nullptr;
  }

  // TODO(aarfaian): enforce valid element types

  return SymTensorType::get(dimensions, element_type, encoding);
}

void SymTensorType::print(mlir::AsmPrinter& printer) const {
  printer << "<";

  auto shape = getShape();
  llvm::interleave(
      shape, printer, [&](const auto& sym_dim) { sym_dim.print(printer); },
      ",");

  printer << ":";
  printer << getElementType();
  printer << ">";
}

mlir::RankedTensorType GetRankedTensorType(mlir::Type type) {
  if (auto sym_tensor_type = mlir::dyn_cast<SymTensorType>(type)) {
    return mlir::RankedTensorType::get(sym_tensor_type.getSizes(),
                                       sym_tensor_type.getElementType(),
                                       sym_tensor_type.getEncoding());
  }
  return mlir::dyn_cast<mlir::RankedTensorType>(type);
}

litert::SymTensorType GetSymTensorType(mlir::Type type) {
  if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    return SymTensorType::get(tensor_type.getShape(),
                              tensor_type.getElementType(),
                              tensor_type.getEncoding());
  }
  return mlir::dyn_cast<SymTensorType>(type);
}

mlir::RankedTensorType GetRankedTensorType(mlir::Value value) {
  return GetRankedTensorType(value.getType());
}
litert::SymTensorType GetSymTensorType(mlir::Value value) {
  return GetSymTensorType(value.getType());
}

}  // namespace litert
