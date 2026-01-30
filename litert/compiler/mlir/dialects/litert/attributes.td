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
#ifndef LITERT_ATTRIBUTES
#define LITERT_ATTRIBUTES

include "third_party/odml/litert/litert/compiler/mlir/dialects/litert/dialect.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinAttributeInterfaces.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/CommonAttrConstraints.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OpAsmInterface.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OpBase.td"

class LITERT_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                 string baseCppClass = "::mlir::Attribute">
  : AttrDef<LITERT_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "litert." # attrMnemonic;
}

def LITERT_PyDenseElementsAttr : LITERT_Attr<"PyDenseElements", 
  "py_dense", [ElementsAttrInterface]> {
  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
    "litert::PyBlob":$py_blob
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "mlir::ShapedType":$type, "litert::PyBlob&&":$py_blob
    )>,
  ];

  let extraClassDeclaration = [{
    const litert::PyBlob& GetPyBlob() const;
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}


def LITERT_LazyDenseElementsAttr : LITERT_Attr<"LazyDenseElements",
    "lazy_dense", [ElementsAttrInterface]> {

  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
    ResourceHandleParameter<"LazyResourceElementsHandle">:$raw_handle
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "mlir::ShapedType":$type, "LazyResourceElementsHandle":$handle
    )>,
    AttrBuilderWithInferredContext<(ins
      "mlir::ShapedType":$type, "LazyResourceBlob":$blob
    )>,
    AttrBuilderWithInferredContext<(ins
      "mlir::ShapedType":$type, "llvm::ArrayRef<uint8_t>":$data, "size_t":$alignment
    )>,
  ];

  let extraClassDeclaration = [{
    template <typename T>
    static LazyDenseElementsAttr get(mlir::ShapedType type, llvm::ArrayRef<T> data);
    ScopedDataHandle GetDataHandle() const;
    const LazyResourceBlob& GetBlob() const;

    //using NonContiguousIterableTypesT = std::tuple<llvm::APInt>;

    //auto value_begin_impl(OverloadToken<llvm::APInt>) const {
    //  auto data_handle_or = GetDataHandle();
    //  if (!data_handle_or.ok()) {
    //    return mlir::failure();
    //  }
    //  auto data_handle = *std::move(data_handle_or);
    //  auto it = llvm::map_range(data_handle, [=](uint64_t value) {
    //    return llvm::APInt(/*numBits=*/64, value);
    //  }).begin();
    //  return llvm::FailureOr<decltype(it)>(std::move(it));
    //}
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}

def LITERT_SymDimAttr : LITERT_Attr<"SymDim", "sym_dim"> {
  let parameters = (ins
    "int64_t":$size,
    OptionalParameter<"mlir::SymbolRefAttr">:$symbol
  );

  let skipDefaultBuilders = 1;

  let builders = [
    AttrBuilder<(ins "int64_t":$size), [{
      return $_get($_ctxt, size, nullptr);
    }]>,
    AttrBuilder<(ins "mlir::SymbolRefAttr":$symbol), [{
      return $_get($_ctxt, mlir::ShapedType::kDynamic, symbol);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

#endif // LITERT_ATTRIBUTES
