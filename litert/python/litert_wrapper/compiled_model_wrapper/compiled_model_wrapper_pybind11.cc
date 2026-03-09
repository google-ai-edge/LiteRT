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

#include <stdexcept>
#include <string>

#include "litert/python/litert_wrapper/compiled_model_wrapper/compiled_model_wrapper.h"
#include "pybind11/functional.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11

namespace py = pybind11;

using litert::compiled_model_wrapper::CompiledModelWrapper;

PYBIND11_MODULE(_pywrap_litert_compiled_model_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_litert_compiled_model_wrapper
    Python bindings for LiteRT CompiledModel.
  )pbdoc";

  // Factory method to create a CompiledModelWrapper from a model file.
  m.def(
      "CreateCompiledModelFromFile",
      [](const std::string& model_path, const std::string& runtime_path,
         const std::string& compiler_plugin_path,
         const std::string& dispatch_library_path, int hardware_accel) {
        std::string error;
        CompiledModelWrapper* wrapper =
            CompiledModelWrapper::CreateWrapperFromFile(
                model_path.c_str(),
                runtime_path.empty() ? nullptr : runtime_path.c_str(),
                compiler_plugin_path.empty() ? nullptr
                                             : compiler_plugin_path.c_str(),
                dispatch_library_path.empty() ? nullptr
                                              : dispatch_library_path.c_str(),
                hardware_accel, &error);
        if (!wrapper) {
          throw std::runtime_error(error);
        }
        return wrapper;  // Ownership transferred to pybind11
      },
      py::arg("model_path"), py::arg("runtime_path") = "",
      py::arg("compiler_plugin_path") = "",
      py::arg("dispatch_library_path") = "", py::arg("hardware_accel") = 0);

  // Factory method to create a CompiledModelWrapper from a model buffer.
  m.def(
      "CreateCompiledModelFromBuffer",
      [](py::bytes model_data, const std::string& runtime_path,
         const std::string& compiler_plugin_path,
         const std::string& dispatch_library_path, int hardware_accel) {
        std::string error;
        PyObject* data_obj = model_data.ptr();
        CompiledModelWrapper* wrapper =
            CompiledModelWrapper::CreateWrapperFromBuffer(
                data_obj, runtime_path.empty() ? nullptr : runtime_path.c_str(),
                compiler_plugin_path.empty() ? nullptr
                                             : compiler_plugin_path.c_str(),
                dispatch_library_path.empty() ? nullptr
                                              : dispatch_library_path.c_str(),
                hardware_accel, &error);
        if (!wrapper) {
          throw std::runtime_error(error);
        }
        return wrapper;
      },
      py::arg("model_data"), py::arg("runtime_path") = "",
      py::arg("compiler_plugin_path") = "",
      py::arg("dispatch_library_path") = "", py::arg("hardware_accel") = 0);

  // Bindings for the CompiledModelWrapper class.
  py::class_<CompiledModelWrapper>(m, "CompiledModelWrapper")
      .def("GetSignatureList",
           [](CompiledModelWrapper& self) {
             PyObject* r = self.GetSignatureList();
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetSignatureByIndex",
           [](CompiledModelWrapper& self, int index) {
             PyObject* r = self.GetSignatureByIndex(index);
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetNumSignatures",
           [](CompiledModelWrapper& self) {
             PyObject* r = self.GetNumSignatures();
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetSignatureIndex",
           [](CompiledModelWrapper& self, const std::string& key) {
             PyObject* r = self.GetSignatureIndex(key.c_str());
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetInputBufferRequirements",
           [](CompiledModelWrapper& self, int sig_idx, int in_idx) {
             PyObject* r = self.GetInputBufferRequirements(sig_idx, in_idx);
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetOutputBufferRequirements",
           [](CompiledModelWrapper& self, int sig_idx, int out_idx) {
             PyObject* r = self.GetOutputBufferRequirements(sig_idx, out_idx);
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateInputBufferByName",
           [](CompiledModelWrapper& self, const std::string& sig_key,
              const std::string& input_name) {
             // Pass Python wrapper reference so buffer keeps model alive
             py::object self_obj = py::cast(&self);
             PyObject* r = self.CreateInputBufferByName(
                 self_obj.ptr(), sig_key.c_str(), input_name.c_str());
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateOutputBufferByName",
           [](CompiledModelWrapper& self, const std::string& sig_key,
              const std::string& out_name) {
             // Pass Python wrapper reference so buffer keeps model alive
             py::object self_obj = py::cast(&self);
             PyObject* r = self.CreateOutputBufferByName(
                 self_obj.ptr(), sig_key.c_str(), out_name.c_str());
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateInputBuffers",
           [](CompiledModelWrapper& self, int sig_index) {
             // Pass Python wrapper reference so buffers keep model alive
             py::object self_obj = py::cast(&self);
             PyObject* r = self.CreateInputBuffers(self_obj.ptr(), sig_index);
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateOutputBuffers",
           [](CompiledModelWrapper& self, int sig_index) {
             // Pass Python wrapper reference so buffers keep model alive
             py::object self_obj = py::cast(&self);
             PyObject* r = self.CreateOutputBuffers(self_obj.ptr(), sig_index);
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetInputTensorDetails",
           [](CompiledModelWrapper& self, const std::string& sig_key) {
             PyObject* r = self.GetInputTensorDetails(sig_key.c_str());
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("RunByName",
           [](CompiledModelWrapper& self, const std::string& sig_key,
              py::object input_map, py::object output_map) {
             PyObject* r = self.RunByName(sig_key.c_str(), input_map.ptr(),
                                          output_map.ptr());
             if (!r) {
               throw py::error_already_set();
             }
             return py::none();
           })
      .def("RunByIndex",
           [](CompiledModelWrapper& self, int sig_index, py::object in_list,
              py::object out_list) {
             PyObject* r =
                 self.RunByIndex(sig_index, in_list.ptr(), out_list.ptr());
             if (!r) {
               throw py::error_already_set();
             }
             return py::none();
           })
      .def("Run", [](CompiledModelWrapper& self, py::object in_list,
                     py::object out_list) {
        PyObject* r = self.RunByIndex(0, in_list.ptr(), out_list.ptr());
        if (!r) {
          throw py::error_already_set();
        }
        return py::none();
      });
}
