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

#include <string>

#include "litert/python/litert_wrapper/tensor_buffer_wrapper/tensor_buffer_wrapper.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11

namespace py = pybind11;

using litert::tensor_buffer_wrapper::TensorBufferWrapper;

PYBIND11_MODULE(_pywrap_litert_tensor_buffer_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_litert_tensor_buffer_wrapper
    Python bindings for LiteRT TensorBuffers.
  )pbdoc";

  // Creates a TensorBuffer from existing host memory.
  // The memory is not copied but referenced, so the original data must outlive
  // the TensorBuffer unless it's explicitly copied.
  m.def(
      "CreateTensorBufferFromHostMemory",
      [](py::object py_data, const std::string& dtype,
         py::ssize_t num_elements) {
        PyObject* res = TensorBufferWrapper::CreateFromHostMemory(
            py_data.ptr(), dtype, num_elements);
        if (!res) {
          throw py::error_already_set();
        }
        return py::reinterpret_steal<py::object>(res);
      },
      py::arg("py_data"), py::arg("dtype"), py::arg("num_elements"));

  // Writes data to an existing TensorBuffer.
  // The data is copied from the provided Python list into the TensorBuffer.
  m.def("WriteTensor",
        [](py::object capsule, py::object data_list, const std::string& dtype) {
          PyObject* res = TensorBufferWrapper::WriteTensor(
              capsule.ptr(), data_list.ptr(), dtype);
          if (!res) {
            throw py::error_already_set();
          }
          // No return value needed
        });

  // Reads data from a TensorBuffer into a Python list.
  // The data is copied from the TensorBuffer into a new Python list.
  m.def("ReadTensor",
        [](py::object capsule, int num_elements, const std::string& dtype) {
          PyObject* res = TensorBufferWrapper::ReadTensor(capsule.ptr(),
                                                          num_elements, dtype);
          if (!res) {
            throw py::error_already_set();
          }
          return py::reinterpret_steal<py::object>(res);
        });

  // Destroys a TensorBuffer and releases associated resources.
  // This should be called when the TensorBuffer is no longer needed.
  m.def("DestroyTensorBuffer", [](py::object capsule) {
    if (PyObject* res = TensorBufferWrapper::DestroyTensorBuffer(capsule.ptr());
        !res) {
      throw py::error_already_set();
    }
    // No return value needed
  });
}
