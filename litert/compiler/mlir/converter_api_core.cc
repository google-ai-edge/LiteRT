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
#include <optional>
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
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
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
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "litert/compiler/mlir/status_scoped_diagnostic_handler.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "tflite/converter/common/tfl_pass_config.h"
#include "tflite/converter/converter_flags.pb.h"
#include "tflite/converter/flatbuffer_export.h"
#include "tflite/converter/metrics/converter_error_data.pb.h"
#include "tflite/converter/model_flags.pb.h"
#include "tflite/converter/python/tf_tfl_flatbuffer_helpers.h"
#include "tflite/converter/quantization/common/quantization_lib/quantization_config.h"
#include "tflite/converter/stablehlo/transforms/stablehlo_passes.h"
#include "tflite/converter/tf_to_tfl_flatbuffer.h"
#include "tflite/converter/transforms/optimize_pass.h"
#include "tflite/converter/transforms/passes.h"
#include "tflite/converter/types.pb.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace litert {

namespace {

mlir::func::FuncOp GetFuncOpByName(mlir::ModuleOp module_op,
                                   absl::string_view name = "main") {
  mlir::func::FuncOp main_func = module_op.lookupSymbol<mlir::func::FuncOp>(
      mlir::StringAttr::get(module_op.getContext(), name));
  return main_func;
}

template <typename OStream>
absl::Status ExportFlatbuffer(mlir::ModuleOp module_op,
                              OStream& export_stream) {
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

  return tflite::MlirToFlatBufferTranslateFunction(module_op, options,
                                                   export_stream);
}

// Returns the exported names of the given module op. The exported names are
// defined in the `tf_saved_model.exported_names` attribute of the function
// ops (by SetSignature).
std::vector<std::string> GetSavedModelExportedNames(mlir::ModuleOp module_op) {
  std::vector<std::string> exported_names;
  for (mlir::func::FuncOp func_op : module_op.getOps<mlir::func::FuncOp>()) {
    auto exported_names_attr = func_op->getAttrOfType<mlir::ArrayAttr>(
        "tf_saved_model.exported_names");
    if (!exported_names_attr || exported_names_attr.empty()) {
      continue;
    }
    auto exported_name =
        mlir::dyn_cast<mlir::StringAttr>(exported_names_attr.getValue()[0]);
    if (!exported_name) {
      continue;
    }
    exported_names.push_back(exported_name.getValue().str());
  }
  return exported_names;
}

// Transforms the given `ConvertToTFLConfig` to the configs used in the
// Tensorflow-TFL converter. The setup is based on the values recorded from
// standard saved model conversion flow with Python API.
absl::StatusOr<std::pair<tflite::ConverterFlags, mlir::TFL::PassConfig>>
GetTFLConverterFlagsAndPassConfig(mlir::ModuleOp module_op,
                                  const ConvertToTFLConfig& config) {
  // Prepare model flags with fake saved model info.
  tflite::ModelFlags model_flags;
  {
    model_flags.set_saved_model_dir("");
    model_flags.set_saved_model_version(1);
    model_flags.add_saved_model_tags("serve");
    for (const auto& exported_name : GetSavedModelExportedNames(module_op)) {
      model_flags.add_saved_model_exported_names(exported_name);
    }
  }

  // Prepare convert flags.
  tflite::ConverterFlags converter_flags;
  {
    converter_flags.set_input_format(tflite::TENSORFLOW_GRAPHDEF);
    converter_flags.set_output_format(tflite::TFLITE);
    converter_flags.set_inference_type(tflite::IODataType::FLOAT);
    converter_flags.set_inference_input_type(tflite::IODataType::FLOAT);
    converter_flags.set_drop_control_dependency(true);
    converter_flags.set_enable_dynamic_update_slice(true);
    converter_flags.set_enable_composite_direct_lowering(true);
    converter_flags.set_convert_to_stablehlo(false);
    converter_flags.set_canonicalizing_inf_as_min_max_float(
        config.canonicalizing_inf_as_min_max_float);
    converter_flags.set_qdq_conversion_mode(config.qdq_conversion_mode);
    converter_flags.set_unsafe_fuse_dynamic_shaped_broadcast(
        config.unsafe_fuse_dynamic_shaped_broadcast);

    if (config.model_origin_framework == "UNSET") {
      converter_flags.set_model_origin_framework(tflite::ConverterFlags::UNSET);
    } else if (config.model_origin_framework == "PYTORCH") {
      converter_flags.set_model_origin_framework(
          tflite::ConverterFlags::PYTORCH);
    } else if (config.model_origin_framework == "JAX") {
      converter_flags.set_model_origin_framework(tflite::ConverterFlags::JAX);
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unknown model origin framework: ", config.model_origin_framework));
    }
  }

  mlir::TFL::QuantizationSpecs quant_specs;
  {
    std::vector<std::string> node_names;
    std::vector<std::string> node_dtypes;
    std::vector<std::optional<std::vector<int>>> node_shapes;
    std::vector<std::optional<double>> node_mins;
    std::vector<std::optional<double>> node_maxs;
    if (auto status = tensorflow::internal::PopulateQuantizationSpecs(
            model_flags, converter_flags, &quant_specs, &node_names,
            &node_dtypes, &node_shapes, &node_mins, &node_maxs);
        !status.ok()) {
      return status;
    }
  }

  // Prepare pass config.
  mlir::TFL::PassConfig pass_config(quant_specs);
  {
    // Reference:
    // tensorflow/compiler/mlir/lite/python/graphdef_to_tfl_flatbuffer.cc
    bool emit_builtin_tflite_ops = !converter_flags.force_select_tf_ops();
    pass_config.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
    pass_config.enable_tflite_variables =
        converter_flags.enable_tflite_resource_variables();
    pass_config.unfold_batch_matmul = converter_flags.unfold_batchmatmul();
    pass_config.lower_tensor_list_ops = converter_flags.lower_tensor_list_ops();
    // Disable the unfolding of the 16x16 TF::BatchMatMulOp to avoid the
    // conversion to an unsupported 16x16 TFL::FullyConnectedOp.
    if (converter_flags.inference_type() ==
        tflite::IODataType::QUANTIZED_INT16) {
      pass_config.unfold_batch_matmul = false;
    }
    pass_config.unfold_large_splat_constant =
        converter_flags.unfold_large_splat_constant();
    pass_config.enable_dynamic_update_slice =
        converter_flags.enable_dynamic_update_slice();
    pass_config.preserve_assert_op = converter_flags.preserve_assert_op();
    pass_config.guarantee_all_funcs_one_use =
        converter_flags.guarantee_all_funcs_one_use();
    pass_config.enable_stablehlo_conversion =
        converter_flags.convert_to_stablehlo();
    pass_config.legalize_custom_tensor_list_ops =
        converter_flags.legalize_custom_tensor_list_ops();
    pass_config.enable_composite_direct_lowering =
        converter_flags.enable_composite_direct_lowering();
    pass_config.model_origin_framework =
        converter_flags.model_origin_framework();
    pass_config.canonicalizing_inf_as_min_max_float =
        converter_flags.canonicalizing_inf_as_min_max_float();
    pass_config.unsafe_fuse_dynamic_shaped_broadcast =
        converter_flags.unsafe_fuse_dynamic_shaped_broadcast();

    pass_config.enable_hlo_to_tf_conversion = true;

    if (converter_flags.strict_qdq_mode()) {
      pass_config.quant_specs.qdq_conversion_mode =
          mlir::TFL::QDQConversionMode::kQDQStrict;
    } else if (converter_flags.qdq_conversion_mode() == "STATIC") {
      pass_config.quant_specs.qdq_conversion_mode =
          mlir::TFL::QDQConversionMode::kQDQStatic;
    } else if (converter_flags.qdq_conversion_mode() == "DYNAMIC") {
      pass_config.quant_specs.qdq_conversion_mode =
          mlir::TFL::QDQConversionMode::kQDQDynamic;
      // Need to set this or else the ops will still use floating point kernels
      pass_config.quant_specs.inference_type = tensorflow::DT_QINT8;
    } else if (converter_flags.qdq_conversion_mode() == "NONE") {
      pass_config.quant_specs.qdq_conversion_mode =
          mlir::TFL::QDQConversionMode::kQDQNone;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown QDQ conversion mode: ",
                       converter_flags.qdq_conversion_mode()));
    }

    if (converter_flags.has_qdq_conversion_mode() &&
        converter_flags.qdq_conversion_mode() != "NONE") {
      // Setting this flag causes
      // PrepareQuantize::SetInputNodesQuantizationParams() to be false and
      // allows PrepareQuantizePass to complete. For the most part this step is
      // unnecessary for non-TF QDQ models.
      pass_config.quant_specs.disable_set_input_nodes_quantization_params =
          true;
    }
  }

  return std::make_pair(converter_flags, pass_config);
}

}  // namespace

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
  mlir::PassRegistration<mlir::OperationPass<mlir::func::FuncOp>>(
      []() { return mlir::TFL::CreatePrepareQuantizePass(); });
  mlir::PassRegistration<mlir::OperationPass<mlir::ModuleOp>>(
      []() { return mlir::TFL::CreatePropagateQsvPass(); });
}

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

absl::Status RunConvertToTFLPasses(mlir::ModuleOp module_op,
                                   mlir::PassManager& pass_manager,
                                   const ConvertToTFLConfig& config) {
  mlir::MLIRContext* context = module_op.getContext();
  PrepareMlirContext(context);

  auto tfl_configs_or = GetTFLConverterFlagsAndPassConfig(module_op, config);
  if (!tfl_configs_or.ok()) {
    return tfl_configs_or.status();
  }

  auto [tfl_converter_flags, tfl_pass_config] = tfl_configs_or.value();
  return tensorflow::RunConvertTFExecutorToTFLPasses(
      module_op, tfl_converter_flags, tfl_pass_config, pass_manager,
      /*saved_model_tags=*/{"serve"}, /*saved_model_dir=*/"");
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

absl::StatusOr<llvm::SmallVector<char>> ExportFlatbufferToBytes(
    mlir::ModuleOp module_op) {
  llvm::SmallVector<char> buffer;
  buffer.reserve(32 * 1024);

  llvm::raw_svector_ostream export_stream(buffer);
  if (auto status = ExportFlatbuffer(module_op, export_stream); !status.ok()) {
    return status;
  }
  return std::move(buffer);
}

absl::Status ExportFlatbufferToFile(mlir::ModuleOp module_op,
                                    absl::string_view output_path) {
  std::error_code ec;
  llvm::raw_fd_ostream export_stream(output_path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    return absl::InternalError(absl::StrCat("Failed to open output file ",
                                            output_path, ": ", ec.message()));
  }
  return ExportFlatbuffer(module_op, export_stream);
}

}  // namespace litert
