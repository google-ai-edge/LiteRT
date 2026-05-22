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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_TF_TO_TFL_FLATBUFFER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_TF_TO_TFL_FLATBUFFER_H_

#include <string>
#include <unordered_set>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "tflite/converter/common/tfl_pass_config.h"
#include "tflite/converter/converter_flags.pb.h"

namespace litert {

// Runs the TF Executor to TFL passes on the given module. This is the main
// conversion pass pipeline to convert a module of TF or StableHLO dialect to
// TFL dialect.
absl::Status RunConvertTFExecutorToTFLPasses(
    mlir::ModuleOp module, tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config, mlir::PassManager& pass_manager,
    const std::unordered_set<std::string>& saved_model_tags,
    absl::string_view saved_model_dir);

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_TF_TO_TFL_FLATBUFFER_H_
