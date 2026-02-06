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
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"

namespace litert {

// Config for the conversion from StableHLO to TFL. This config would be
// translated to the `tflite::ConverterFlags` and `mlir::TFL::PassConfig` in the
// TFL converter.
struct ConvertToTFLConfig {
  // The source model type.
  std::string model_origin_framework = "UNSET";

  // When set to true, convert +Inf/-Inf to MIN/MAX float value and output of
  // convert only contains finite values.
  bool canonicalizing_inf_as_min_max_float = true;

  // Whether to consider this model a quantized model with quantize/dequantize
  // ops and to convert kernels to quantized kernels wherever appropriate.
  // WARNING: Experimental interface, subject to change.
  std::string qdq_conversion_mode = "NONE";

  // When set to true, allows fusion of dynamic shaped broadcast ops. It helps
  // fusing implicit broadcasting ops when output shape has dynamic dimensions,
  // but it may cause incorrect results when broadcasting ops are introduced by
  // explicit broadcasting in the source model.
  bool unsafe_fuse_dynamic_shaped_broadcast = false;
};

// Register passes to be used via MLIR pybindings pass manager.
void RegisterPasses();

// Config and register the dialects and passes for MLIR context.
void PrepareMlirContext(mlir::MLIRContext* context);

// Runs all passes from TF/StableHLO to TFL (the default conversion path).
absl::Status RunConvertToTFLPasses(mlir::ModuleOp module,
                                   mlir::PassManager& pass_manager,
                                   const ConvertToTFLConfig& config);

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

// Exports the MLIR module to flatbuffer and returns the bytes.
absl::StatusOr<llvm::SmallVector<char>> ExportFlatbufferToBytes(
    mlir::ModuleOp module_op);

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_CONVERTER_API_CORE_H_
