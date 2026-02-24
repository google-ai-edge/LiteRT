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
include "mlir/IR/CommonTypeConstraints.td"

def Shape : TypeOrValueSemanticsContainer<
    1DTensorOf<[I32]>, "shape">;

class LITERT_SymTensorContainerType<list<Type> allowedTypes,
                                         Pred containerPred, string descr,
                                         string cppType = "::mlir::Type"> :
    Type<And<[containerPred,
              Concat<"[](::mlir::Type elementType) { return ",
                SubstLeaves<"$_self", "elementType",
                AnyTypeOf<allowedTypes>.predicate>,
                "; }(::llvm::cast<litert::SymTensorType>($_self).getElementType())">]>,
         descr # " of " # AnyTypeOf<allowedTypes>.summary # " values", cppType>;

def IsSymTensorTypePred : CPred<"::llvm::isa<litert::SymTensorType>($_self)">;

class LITERT_TensorOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "tensor">
  : Type<Or<[
      LITERT_SymTensorContainerType<allowedTypes, And<!listconcat([IsSymTensorTypePred], preds)>, summary>.predicate,
      TensorOf<allowedTypes, preds, summary>.predicate]>,
      summary, "::mlir::Type">;

def LITERT_AllowedTensorElementTypes : AnyTypeOf<[F32, F64, I1, I32, I64], "f32, f64, bool, i32, i64">;

def LITERT_Tensor : TypeOrValueSemanticsContainer<LITERT_TensorOf<[LITERT_AllowedTensorElementTypes]>, "tensor">;
def LITERT_I64Tensor : TypeOrValueSemanticsContainer<LITERT_TensorOf<[I64]>, "i64 tensor">;
