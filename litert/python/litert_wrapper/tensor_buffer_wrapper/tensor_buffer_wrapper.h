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

#ifndef LITERT_PYTHON_LITERT_WRAPPER_TENSOR_BUFFER_WRAPPER_TENSOR_BUFFER_WRAPPER_H_
#define LITERT_PYTHON_LITERT_WRAPPER_TENSOR_BUFFER_WRAPPER_TENSOR_BUFFER_WRAPPER_H_

#include <Python.h>

#include <string>

#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert::tensor_buffer_wrapper {

/**
 * Wrapper class for LiteRtTensorBuffer operations exposed to Python.
 *
 * This class provides a C++ interface for creating, reading from, and writing
 * to LiteRtTensorBuffer objects from Python. It encapsulates tensor buffer
 * management functionality in a centralized location, allowing direct
 * manipulation of tensor buffers without requiring a model instance.
 */
class TensorBufferWrapper {
 public:
  /**
   * Creates a TensorBuffer from host memory and returns it as a Python capsule.
   *
   * @param py_data Python object containing the source data.
   * @param dtype String representation of the data type (e.g., "float32").
   * @param num_elements Number of elements in the tensor.
   * @return A new Python capsule containing the LiteRtTensorBuffer, or nullptr
   * on error.
   */
  static PyObject* CreateFromHostMemory(PyObject* py_data,
                                        const std::string& dtype,
                                        Py_ssize_t num_elements);

  /**
   * Writes data from a Python list to a TensorBuffer.
   *
   * @param buffer_capsule Python capsule containing the LiteRtTensorBuffer.
   * @param data_list Python list containing the data to write.
   * @param dtype String representation of the data type.
   * @return PyObject* indicating success (Py_None) or nullptr on error.
   */
  static PyObject* WriteTensor(PyObject* buffer_capsule, PyObject* data_list,
                               const std::string& dtype);

  /**
   * Reads data from a TensorBuffer into a new Python list.
   *
   * @param buffer_capsule Python capsule containing the LiteRtTensorBuffer.
   * @param num_elements Number of elements to read.
   * @param dtype String representation of the data type.
   * @return A new Python list containing the tensor data, or nullptr on error.
   */
  static PyObject* ReadTensor(PyObject* buffer_capsule, int num_elements,
                              const std::string& dtype);

  /**
   * Explicitly destroys a TensorBuffer and releases associated resources.
   *
   * @param buffer_capsule Python capsule containing the LiteRtTensorBuffer.
   * @return PyObject* indicating success (Py_None) or nullptr on error.
   */
  static PyObject* DestroyTensorBuffer(PyObject* buffer_capsule);

  /**
   * Creates a Python exception with the given error message.
   *
   * @param msg The error message.
   * @return nullptr after setting the Python exception.
   */
  static PyObject* ReportError(const std::string& msg);

  /**
   * Converts a LiteRT error to a Python exception.
   *
   * @param error The LiteRT error to convert.
   * @return nullptr after setting the Python exception.
   */
  static PyObject* ConvertErrorToPyExc(const Error& error);

 private:
  /**
   * Returns the byte width of a data type.
   *
   * @param dtype String representation of the data type.
   * @return Size in bytes of the data type, or 0 if unknown.
   */
  static size_t ByteWidthOfDType(const std::string& dtype);
};

}  // namespace litert::tensor_buffer_wrapper

#endif  // LITERT_PYTHON_LITERT_WRAPPER_TENSOR_BUFFER_WRAPPER_TENSOR_BUFFER_WRAPPER_H_
