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
#ifndef LITERT_DIALECT
#define LITERT_DIALECT

include "third_party/llvm/llvm-project/mlir/include/mlir/IR/DialectBase.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OpBase.td"

def LITERT_Dialect : Dialect {
  let name = "litert";
  let cppNamespace = "::litert";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
  private:
    void registerAttributes();
    void registerTypes();
    void registerOps();
  }];
}

#endif // LITERT_DIALECT
