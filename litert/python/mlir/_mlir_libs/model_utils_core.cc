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
#include "litert/python/mlir/_mlir_libs/model_utils_core.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "llvm/ADT/StringRef.h"
#include "llvm/FileCheck/FileCheck.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "tflite/converter/flatbuffer_export.h"
#include "tflite/converter/flatbuffer_import.h"
#include "tflite/converter/ir/tfl_ops.h"
#include "tflite/converter/quantization/ir/QuantOps.h"
#include "tflite/converter/stablehlo/transforms/stablehlo_passes.h"
#include "tflite/converter/transforms/optimize_pass.h"
#include "tflite/converter/transforms/passes.h"

namespace litert::model_utils {

void RegisterDialects(mlir::DialectRegistry& registry) {
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::quant::QuantDialect,
                  mlir::quantfork::QuantizationForkDialect,
                  mlir::TFL::TensorFlowLiteDialect,
                  mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
}

void RegisterPasses() {
  mlir::registerTransformsPasses();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::func::registerFuncPasses();
  mlir::stablehlo::registerPassPipelines();
  mlir::stablehlo::registerPasses();
  mlir::stablehlo::registerOptimizationPasses();

  mlir::odml::registerLegalizeStablehloToVhloPass();
  mlir::PassRegistration<mlir::TFL::OptimizePass>(
      []() { return mlir::TFL::CreateOptimizePass(); });
}

mlir::OwningOpRef<mlir::ModuleOp> FlatbufferToMlir(mlir::MLIRContext* context,
                                                   absl::string_view buffer) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
  return tflite::FlatBufferToMlir(buffer, context,
                                  mlir::UnknownLoc::get(context));
}

std::string MlirToFlatbuffer(mlir::ModuleOp module_op) {
  tflite::FlatbufferExportOptions options;
  std::string bytes;
  tflite::MlirToFlatBufferTranslateFunction(module_op, options, &bytes, true);
  return bytes;
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

std::vector<std::string> GetDictionaryAttrNames(mlir::DictionaryAttr attr) {
  if (attr == nullptr) {
    return {};
  }

  std::vector<std::string> attr_names;
  for (auto attr : attr) {
    attr_names.push_back(attr.getName().str());
  }
  return attr_names;
}

absl::string_view GetDenseElementsAttrBytes(mlir::DenseElementsAttr attr) {
  return absl::string_view(attr.getRawData().data(), attr.getRawData().size());
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
