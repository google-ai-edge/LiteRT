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

#include <Python.h>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Operation.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/vector.h"
#include "litert/compiler/mlir/converter_api_core.h"

namespace {

namespace nb = nanobind;

using namespace nb::literals;  // NOLINT

void ThrowIfFailed(absl::string_view prefix, const absl::Status& status) {
  if (!status.ok()) {
    throw nb::value_error(absl::StrCat(prefix, ": ", status.message()).c_str());
  }
}

NB_MODULE(converter_api_ext, m) {
  m.doc() = "LiteRT Converter API Extensions";

  m.def(
      "run_converter_passes",
      [](MlirModule c_module_op) {
        mlir::ModuleOp module_op = unwrap(c_module_op);
        absl::Status status = litert::RunStableHLOToTFLPasses(module_op);
        ThrowIfFailed("Failed to run converter passes", status);
      },
      nb::arg("module"),
      "Runs all passes from StableHLO to TFL (the default conversion path).");

  m.def(
      "set_signature",
      [](MlirOperation c_op, std::string signature_name,
         std::vector<std::string> input_names,
         std::vector<std::string> output_names) {
        mlir::Operation* op = unwrap(c_op);
        absl::Status status =
            litert::SetSignature(op, signature_name, input_names, output_names);
        ThrowIfFailed("Failed to set signature", status);
      },
      nb::arg("op"), nb::arg("signature_name"), nb::arg("input_names"),
      nb::arg("output_names"),
      "Sets the flatbuffer signature for the given MLIR operation. The "
      "operation can be a func.FuncOp or a ModuleOp (when there is a "
      "func.FuncOp named `main`).");

  m.def(
      "merge_modules",
      [](std::vector<MlirOperation> c_module_ops) {
        if (c_module_ops.empty()) {
          throw nb::value_error(
              "Input must be a non-empty list of module ops.");
        }

        std::vector<mlir::ModuleOp> module_ops;
        for (const auto& c_module_op : c_module_ops) {
          auto module_op = llvm::dyn_cast<mlir::ModuleOp>(unwrap(c_module_op));
          if (!module_op) {
            throw nb::value_error("Element in list is not a valid module op.");
          }
          module_ops.push_back(module_op);
        }

        auto merged_module_or = litert::MergeModuleOps(module_ops);
        ThrowIfFailed("Failed to merge modules", merged_module_or.status());
        return wrap(merged_module_or.value().release());
      },
      nb::arg("module_ops"),
      "Merges multiple MLIR module ops into one. Public symbol name collisions "
      "will cause this to fail.");

  m.def(
      "export_flatbuffer_to_file",
      [](MlirModule c_module_op, std::string export_path) {
        absl::Status status =
            litert::ExportFlatbufferToFile(unwrap(c_module_op), export_path);
        ThrowIfFailed("Failed to export flatbuffer", status);
      },
      nb::arg("module"), nb::arg("export_path"),
      "Exports the MLIR module to flatbuffer and exports it to the given file "
      "path.");
}
}  // namespace
