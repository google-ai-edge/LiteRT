// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_PYTHON_MLIR__MLIR_LIBS_MODEL_UTILS_CORE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_PYTHON_MLIR__MLIR_LIBS_MODEL_UTILS_CORE_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace litert::model_utils {

void RegisterDialects(mlir::DialectRegistry& registry);
void RegisterPasses();

mlir::OwningOpRef<mlir::ModuleOp> FlatbufferToMlir(mlir::MLIRContext* context,
                                                   absl::string_view buffer);
std::string MlirToFlatbuffer(mlir::ModuleOp module_op);
std::vector<std::string> GetOperationAttributeNames(mlir::Operation* op);
std::vector<std::string> GetDictionaryAttrNames(mlir::DictionaryAttr attr);
absl::string_view GetDenseElementsAttrBytes(mlir::DenseElementsAttr attr);
bool FileCheckCheckInput(absl::string_view input, absl::string_view check);

}  // namespace litert::model_utils

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_PYTHON_MLIR__MLIR_LIBS_MODEL_UTILS_CORE_H_
