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
#include "litert/compiler/mlir/model_utils_core.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "llvm/ADT/StringRef.h"
#include "llvm/FileCheck/FileCheck.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "litert/compiler/mlir/converter_api_core.h"
#include "tflite/converter/flatbuffer_import.h"
#include "tflite/converter/transforms/passes.h"

namespace litert::model_utils {

mlir::OwningOpRef<mlir::ModuleOp> FlatbufferToMlir(mlir::MLIRContext* context,
                                                   absl::string_view buffer) {
  PrepareMlirContext(context);
  return tflite::FlatBufferToMlir(buffer, context,
                                  mlir::UnknownLoc::get(context));
}

std::vector<std::string> GetOperationAttributeNames(mlir::Operation* op) {
  if (op == nullptr) {
    return {};
  }

  std::vector<std::string> attr_names;
  for (auto attr : op->getAttrs()) {
    attr_names.push_back(attr.getName().str());
  }
  return attr_names;
}

std::vector<std::string> GetDictionaryAttrNames(mlir::Attribute attr) {
  auto dict_attr = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
  if (dict_attr == nullptr) {
    return {};
  }

  std::vector<std::string> attr_names;
  for (auto attr : dict_attr) {
    attr_names.push_back(attr.getName().str());
  }
  return attr_names;
}

absl::string_view GetDenseElementsAttrBytes(mlir::Attribute attr) {
  auto dense_attr = llvm::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (dense_attr == nullptr) {
    return "";
  }
  return absl::string_view(dense_attr.getRawData().data(),
                           dense_attr.getRawData().size());
}

bool FileCheckCheckInput(absl::string_view input, absl::string_view check) {
  llvm::FileCheckRequest fcr;
  llvm::FileCheck fc(fcr);
  llvm::SourceMgr SM = llvm::SourceMgr();
  SM.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(input), llvm::SMLoc());
  SM.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(check), llvm::SMLoc());
  fc.readCheckFile(SM, llvm::StringRef(check));
  return fc.checkInput(SM, llvm::StringRef(input));
}

}  // namespace litert::model_utils
