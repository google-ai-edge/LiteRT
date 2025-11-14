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

#include <Python.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "llvm/Support/Casting.h"
#include "mlir-c/IR.h"
#include "mlir-c/Transforms.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "mlir/CAPI/IR.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/vector.h"
#include "litert/python/mlir/_mlir_libs/model_utils_core.h"

namespace {

namespace nb = nanobind;
namespace model_utils = litert::model_utils;

NB_MODULE(model_utils_ext, m) {
  Py_Initialize();

  m.doc() = "LiteRT ModelUtils Extensions";

  model_utils::RegisterPasses();

  m.def("register_dialects", [](MlirContext context) {
    mlir::DialectRegistry registry;
    model_utils::RegisterDialects(registry);
    unwrap(context)->appendDialectRegistry(registry);
    unwrap(context)->loadAllAvailableDialects();
  });

  m.def("flatbuffer_to_mlir",
        [](nb::bytes buffer, MlirContext context) -> MlirModule {
          auto module_op = model_utils::FlatbufferToMlir(
              unwrap(context),
              absl::string_view(buffer.c_str(), buffer.size()));
          return wrap(module_op.release());
        });

  m.def("mlir_to_flatbuffer", [](MlirOperation c_op) {
    auto op = unwrap(c_op);
    auto module_op = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (module_op == nullptr) {
      throw nb::value_error("Failed to cast the input to mlir::ModuleOp.");
    }
    std::string data = model_utils::MlirToFlatbuffer(module_op);
    return nb::bytes(data.c_str(), data.size());
  });

  m.def("get_operation_attribute_names", [](MlirOperation c_op) {
    mlir::Operation* op = unwrap(c_op);
    return model_utils::GetOperationAttributeNames(op);
  });

  m.def("get_dictionary_attr_names", [](MlirAttribute c_attr) {
    auto attr = llvm::dyn_cast<mlir::DictionaryAttr>(unwrap(c_attr));
    if (attr == nullptr) {
      throw nb::value_error(
          "Failed to cast the input to mlir::DictionaryAttr.");
    }
    return model_utils::GetDictionaryAttrNames(attr);
  });

  m.def("get_dense_elements_attr_bytes", [](MlirAttribute c_attr) {
    auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(unwrap(c_attr));
    if (attr == nullptr) {
      throw nb::value_error(
          "Failed to cast the input to mlir::DenseElementsAttr.");
    }
    auto bytes = model_utils::GetDenseElementsAttrBytes(attr);
    return nb::bytes(bytes.data(), bytes.size());
  });
}

}  // namespace
