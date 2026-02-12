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
#ifndef LITERT_TYPES
#define LITERT_TYPES

include "litert/compiler/mlir/dialects/litert/dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/Traits.td"

class LITERT_Type<string name, string type_mnemonic, list<Trait> traits = [],
                  string base_cpp_class = "::mlir::Type">
    : TypeDef<LITERT_Dialect, name, traits, base_cpp_class> {
  let mnemonic = type_mnemonic;
}

def SymTensor : LITERT_Type<"SymTensor", "sym_tensor",
                                       [ValueSemantics]> {
  let parameters = (ins
    ArrayRefParameter<"SymDimAttr">:$shape, // SymbolRefAttr or IntegerAttr
    "mlir::Type":$element_type,
    "mlir::Attribute":$encoding
  );
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<SymDimAttr>":$shape,
      "mlir::Type":$element_type,
      CArg<"mlir::Attribute", "{}">:$encoding
    ), [{
      auto* ctx = element_type.getContext();
      return $_get(ctx, shape, element_type, encoding);
    }]>,

    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<int64_t>":$shape,
      "mlir::Type":$element_type,
      CArg<"mlir::Attribute", "{}">:$encoding
    ), [{
      auto* ctx = element_type.getContext();
      llvm::SmallVector<SymDimAttr> sym_shape;
      for (auto dim : shape) {
        sym_shape.push_back(SymDimAttr::get(ctx, dim));
      }
      return $_get(ctx, sym_shape, element_type, encoding);
    }]>,

    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<mlir::SymbolRefAttr>":$shape,
      "mlir::Type":$element_type,
      CArg<"mlir::Attribute", "{}">:$encoding
    ), [{
      auto* ctx = element_type.getContext();
      llvm::SmallVector<SymDimAttr> sym_shape;
      for (auto dim : shape) {
        sym_shape.push_back(SymDimAttr::get(ctx, dim));
      }
      return $_get(ctx, sym_shape, element_type, encoding);
    }]>
  ];

  let extraClassDeclaration = [{
    int64_t getRank() const { return getShape().size(); }
    llvm::SmallVector<int64_t> getSizes() const { return getShapeSizes(); }
    llvm::SmallVector<int64_t> getShapeSizes() const {
      llvm::SmallVector<int64_t> sizes;
      for (auto dim : getShape()) {
        sizes.push_back(dim.getSize());
      }
      return sizes;
    }

    llvm::SmallVector<mlir::SymbolRefAttr> getSymbols() const {
      return getShapeSymbols();
    }
    llvm::SmallVector<mlir::SymbolRefAttr> getShapeSymbols() const {
      llvm::SmallVector<mlir::SymbolRefAttr> symbols;
      for (auto dim : getShape()) {
        symbols.push_back(dim.getSymbol());
      }
      return symbols;
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

#endif // LITERT_TYPES
