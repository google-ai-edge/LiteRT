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
#include "litert/compiler/mlir/tf_to_tfl_flatbuffer.h"

#include <memory>
#include <string>
#include <unordered_set>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "litert/compiler/mlir/status_scoped_diagnostic_handler.h"
#include "tflite/converter/common/tfl_pass_config.h"
#include "tflite/converter/converter_flags.pb.h"
#include "tflite/converter/metrics/converter_error_data.pb.h"
#include "tflite/converter/tf_tfl_passes.h"

namespace litert {

absl::Status RunConvertTFExecutorToTFLPasses(
    mlir::ModuleOp module, tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config, mlir::PassManager& pass_manager,
    const std::unordered_set<std::string>& saved_model_tags,
    absl::string_view saved_model_dir) {
  auto context = module.getContext();
  auto status_handler =
      std::make_unique<litert::StatusScopedDiagnosticHandler>(context);

  if (pass_config.enable_hlo_to_tf_conversion) {
    // TODO: b/194747383 - We need to valid that indeed the "main" func is
    // presented.
    tensorflow::AddPreQuantizationStableHloToTfPasses(
        /*entry_function_name=*/"main", pass_config, pass_manager);
    if (failed(pass_manager.run(module))) {
      return status_handler->ConsumeStatus();
    }
    pass_manager.clear();

    tensorflow::AddPostQuantizationStableHloToTfPasses(pass_config,
                                                       pass_manager);
    if (failed(pass_manager.run(module))) {
      return status_handler->ConsumeStatus();
    }
    pass_manager.clear();
  }

  tensorflow::AddPreVariableFreezingTFToTFLConversionPasses(pass_config,
                                                            &pass_manager);
  if (failed(pass_manager.run(module))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager.clear();

  tensorflow::AddVariableFreezingFromGlobalTensorsPasses(
      converter_flags, pass_config, &pass_manager);
  if (failed(pass_manager.run(module))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager.clear();

  tensorflow::AddPostVariableFreezingTFToTFLConversionPasses(
      saved_model_dir, converter_flags, pass_config, &pass_manager);
  if (failed(pass_manager.run(module))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager.clear();
  return status_handler->ConsumeStatus();
}

}  // namespace litert
