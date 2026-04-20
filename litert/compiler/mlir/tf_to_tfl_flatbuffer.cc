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
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "tflite/converter/common/tfl_pass_config.h"
#include "tflite/converter/converter_flags.pb.h"
#include "tflite/converter/metrics/converter_error_data.pb.h"
#include "tflite/converter/metrics/error_collector_inst.h"
#include "tflite/converter/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace litert {

namespace {

mlir::LogicalResult GraphContainsStatefulPartitionedOp(mlir::ModuleOp module) {
  auto result = module.walk([&](mlir::Operation* op) {
    return llvm::isa_and_nonnull<mlir::TF::StatefulPartitionedCallOp>(op)
               ? mlir::WalkResult::interrupt()
               : mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    // StatefulPartitionedCall ops are not supported by the tflite runtime.
    mlir::TFL::AttachErrorCode(
        module.emitError(
            "The Graph contains unsupported `StatefulPartionedCallOp`(s), will "
            "retry with `guarantee_all_funcs_used_once`"),
        tflite::metrics::ConverterErrorData::
            ERROR_STATEFUL_PARTITIONED_CALL_IN_FINAL_IR);
    return mlir::failure();
  }
  return mlir::success();
}

}  // namespace

absl::Status RunConvertTFExecutorToTFLPasses(
    mlir::ModuleOp module, tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config, mlir::PassManager& pass_manager,
    const std::unordered_set<std::string>& saved_model_tags,
    absl::string_view saved_model_dir) {
  auto context = module.getContext();
  auto status_handler =
      std::make_unique<mlir::StatusScopedDiagnosticHandler>(context,
                                                            /*propagate=*/true);

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

  if (failed(GraphContainsStatefulPartitionedOp(module))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager.clear();
  return absl::OkStatus();
}

}  // namespace litert
