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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_CONVERTER_API_CORE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_CONVERTER_API_CORE_H_

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace litert {

// Config and register the dialects and passes for MLIR context.
void PrepareMlirContext(mlir::MLIRContext* context);

// Runs all passes from StableHLO to TFL (the default conversion path).
absl::Status RunStableHLOToTFLPasses(mlir::ModuleOp module);

// Sets the flatbuffer signature for the given MLIR operation. The operation
// can be a func.FuncOp or a ModuleOp (when there is a func.FuncOp named
// `main`).
absl::Status SetSignature(mlir::Operation* op, absl::string_view signature_name,
                          absl::Span<const std::string> input_names,
                          absl::Span<const std::string> output_names);

// Merges multiple MLIR module ops into one. Public symbol name collisions will
// cause this to fail.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> MergeModuleOps(
    std::vector<mlir::ModuleOp>& module_ops);

// Exports the MLIR module to flatbuffer and exports it to the given file
// path.
absl::Status ExportFlatbufferToFile(mlir::ModuleOp module_op,
                                    absl::string_view output_path);

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_CONVERTER_API_CORE_H_
