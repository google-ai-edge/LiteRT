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

#include <string>

#include "pybind11/pybind11.h"  // from @pybind11
#include "litert/python/litert_wrapper/environment_wrapper/environment_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_litert_environment_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_litert_environment_wrapper
    Python bindings for LiteRT Environment.
  )pbdoc";

  m.def(
      "CreateEnvironment",
      [](const std::string& runtime_path, const std::string& compiler_plugin,
         const std::string& dispatch_library) {
        PyObject* res = litert::environment_wrapper::CreateEnvironment(
            runtime_path.empty() ? nullptr : runtime_path.c_str(),
            compiler_plugin.empty() ? nullptr : compiler_plugin.c_str(),
            dispatch_library.empty() ? nullptr : dispatch_library.c_str());
        if (!res) {
          throw py::error_already_set();
        }
        return py::reinterpret_steal<py::object>(res);
      },
      py::arg("runtime_path") = "", py::arg("compiler_plugin_path") = "",
      py::arg("dispatch_library_path") = "");
}
