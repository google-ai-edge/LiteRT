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
#include "litert/compiler/mlir/converter_api_core.h"

#include <stdlib.h>

#include <memory>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Transform/IR/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "litert/compiler/mlir/status_scoped_diagnostic_handler.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "tflite/converter/common/tfl_pass_config.h"
#include "tflite/converter/converter_flags.pb.h"
#include "tflite/converter/flatbuffer_export.h"
#include "tflite/converter/metrics/converter_error_data.pb.h"
#include "tflite/converter/quantization/common/quantization_lib/quantization_config.h"
#include "tflite/converter/stablehlo/transforms/stablehlo_passes.h"
#include "tflite/converter/tf_tfl_passes.h"
#include "tflite/converter/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace litert {

namespace {

mlir::func::FuncOp GetFuncOpByName(mlir::ModuleOp module_op,
                                   absl::string_view name = "main") {
  mlir::func::FuncOp main_func = module_op.lookupSymbol<mlir::func::FuncOp>(
      mlir::StringAttr::get(module_op.getContext(), name));
  return main_func;
}

}  // namespace

void PrepareMlirContext(mlir::MLIRContext* context) {
  context->printOpOnDiagnostic(false);
  context->disableMultithreading();
  context->allowUnregisteredDialects();

  // Register all available dialects.
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::quant::QuantDialect,
                  mlir::quantfork::QuantizationForkDialect,
                  mlir::TFL::TensorFlowLiteDialect,
                  mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

absl::Status RunStableHLOToTFLPasses(mlir::ModuleOp module_op) {
  mlir::MLIRContext* context = module_op.getContext();
  // Explicitly disable dumping Op details on failures.
  PrepareMlirContext(context);

  auto status_handler =
      std::make_unique<litert::StatusScopedDiagnosticHandler>(context);
  auto pass_manager = std::make_unique<mlir::PassManager>(context);

  mlir::registerPassManagerCLOptions();
  if (mlir::failed(mlir::applyPassManagerCLOptions(*pass_manager))) {
    return absl::InternalError("Failed to apply MLIR pass manager CL options.");
  }

  auto tfl_converter_flags = tflite::ConverterFlags();
  auto tfl_pass_config = mlir::TFL::PassConfig(mlir::TFL::QuantizationSpecs());

  // TODO(cnchan): Make the following conversion path configurable.
  // ======== Pass configs for PyTorch conversion ========
  tfl_pass_config.model_origin_framework = tflite::ConverterFlags::PYTORCH;
  tfl_pass_config.enable_composite_direct_lowering = true;
  tfl_pass_config.enable_dynamic_update_slice = true;
  // Disable to avoid reading values from LazyDenseElementsAttr.
  tfl_pass_config.canonicalizing_inf_as_min_max_float = false;
  // ====================================================

  pass_manager->clear();
  tensorflow::AddPreQuantizationStableHloToTfPasses(
      /*entry_function_name=*/"main", tfl_pass_config, *pass_manager);
  if (failed(pass_manager->run(module_op))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager->clear();
  tensorflow::AddPostQuantizationStableHloToTfPasses(tfl_pass_config,
                                                     *pass_manager);
  if (failed(pass_manager->run(module_op))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager->clear();
  tensorflow::AddPreVariableFreezingTFToTFLConversionPasses(tfl_pass_config,
                                                            pass_manager.get());
  if (failed(pass_manager->run(module_op))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager->clear();
  tensorflow::AddVariableFreezingFromGlobalTensorsPasses(
      tfl_converter_flags, tfl_pass_config, pass_manager.get());
  if (failed(pass_manager->run(module_op))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager->clear();
  tensorflow::AddPostVariableFreezingTFToTFLConversionPasses(
      /*saved_model_dir=*/"", tfl_converter_flags, tfl_pass_config,
      pass_manager.get());
  if (failed(pass_manager->run(module_op))) {
    return status_handler->ConsumeStatus();
  }

  // if (failed(tfl::GraphContainsStatefulPartitionedOp(module))) {
  //   return status_handler->ConsumeStatus();
  // }

  return status_handler->ConsumeStatus();
}

absl::Status SetSignature(mlir::Operation* op, absl::string_view signature_name,
                          absl::Span<const std::string> input_names,
                          absl::Span<const std::string> output_names) {
  mlir::func::FuncOp func_op;
  if (auto module_op = mlir::dyn_cast<mlir::ModuleOp>(op)) {
    func_op = GetFuncOpByName(module_op, "main");
  } else if (auto func_op_ = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
    func_op = func_op_;
  } else {
    return absl::InvalidArgumentError(
        "The given operation is not a ModuleOp or a func.FuncOp.");
  }

  auto module_op = mlir::dyn_cast<mlir::ModuleOp>(func_op->getParentOp());
  if (!module_op) {
    return absl::InvalidArgumentError("Failed to get parent module op.");
  }

  auto context = module_op.getContext();
  for (int i = 0; i < input_names.size(); ++i) {
    func_op.setArgAttr(
        i, "tf_saved_model.index_path",
        mlir::ArrayAttr::get(context,
                             {mlir::StringAttr::get(context, input_names[i])}));
  }
  for (int i = 0; i < output_names.size(); ++i) {
    func_op.setResultAttr(
        i, "tf_saved_model.index_path",
        mlir::ArrayAttr::get(
            context, {mlir::StringAttr::get(context, output_names[i])}));
  }

  std::string joined_input_names = "";
  for (const auto& input_name : input_names) {
    if (!joined_input_names.empty()) {
      absl::StrAppend(&joined_input_names, ",");
    }
    absl::StrAppend(&joined_input_names, signature_name, "_", input_name);
  }
  std::string joined_output_names = "";
  for (const auto& output_name : output_names) {
    if (!joined_output_names.empty()) {
      absl::StrAppend(&joined_output_names, ",");
    }
    absl::StrAppend(&joined_output_names, signature_name, "_", output_name, "_",
                    "output");
  }

  func_op->setAttr(
      "tf.entry_function",
      mlir::DictionaryAttr::get(
          context,
          {{"inputs", mlir::StringAttr::get(context, joined_input_names)},
           {"outputs", mlir::StringAttr::get(context, joined_output_names)}}));
  func_op->setAttr(
      "tf_saved_model.exported_names",
      mlir::ArrayAttr::get(context,
                           {mlir::StringAttr::get(context, signature_name)}));
  func_op.setSymName(signature_name);
  module_op->setAttr("tf_saved_model.semantics", mlir::UnitAttr::get(context));
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> MergeModuleOps(
    std::vector<mlir::ModuleOp>& module_ops) {
  mlir::MLIRContext* context = nullptr;

  if (module_ops.empty()) {
    return absl::InvalidArgumentError("No module ops to merge.");
  }

  for (auto& module_op : module_ops) {
    if (context == nullptr) {
      context = module_op.getContext();
    } else if (context != module_op.getContext()) {
      return absl::InvalidArgumentError(
          "All module ops must have the same context.");
    }
  }

  // Merge all module ops into one module op.
  mlir::OpBuilder builder(context);
  mlir::OwningOpRef<mlir::ModuleOp> merged_module =
      builder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(context));

  for (mlir::ModuleOp module_op : module_ops) {
    merged_module.get()->setAttrs(module_op->getAttrs());

    // Clone module_op because mergeSymbolsInto consumes the source module
    // ('other'). This preserves the original modules in the module_ops
    // vector.
    mlir::OwningOpRef<mlir::ModuleOp> cloned_module = module_op.clone();

    // Merge the cloned module into merged_module.
    // mlir::transform::detail::mergeSymbolsInto handles symbol renaming for
    // private symbols. Public symbol name collisions will cause this to fail.
    if (mlir::failed(mlir::transform::detail::mergeSymbolsInto(
            merged_module.get(), std::move(cloned_module)))) {
      return absl::InvalidArgumentError(
          "Failed to merge modules. This is likely due to a "
          "collision between public symbols, or an unsupported "
          "scenario.");
    }
  }
  return merged_module;
}

absl::Status ExportFlatbufferToFile(mlir::ModuleOp module_op,
                                    absl::string_view output_path) {
  auto context = module_op.getContext();
  PrepareMlirContext(context);

  // Convert SHLO to VHLO for serialization.
  auto status_handler =
      std::make_unique<litert::StatusScopedDiagnosticHandler>(context);
  auto pass_manager = std::make_unique<mlir::PassManager>(context);
  pass_manager->addPass(mlir::odml::createLegalizeStablehloToVhloPass());
  pass_manager->addPass(mlir::createReconcileUnrealizedCastsPass());

  (void)pass_manager->run(module_op);
  if (auto status = status_handler->ConsumeStatus(); !status.ok()) {
    return status;
  }

  // Export module to flatbuffer.
  tflite::FlatbufferExportOptions options;
  options.metadata["keep_stablehlo_constant"] =
      "true";  // tflite::kModelUseStablehloTensorKey
  options.serialize_stablehlo_ops = true;

  std::error_code ec;
  llvm::raw_fd_ostream export_stream(output_path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    return absl::InternalError(absl::StrCat("Failed to open output file ",
                                            output_path, ": ", ec.message()));
  }

  return tflite::MlirToFlatBufferTranslateFunction(module_op, options,
                                                   export_stream);
}

}  // namespace litert
