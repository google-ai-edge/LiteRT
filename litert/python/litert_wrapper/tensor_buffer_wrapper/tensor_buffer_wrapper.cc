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

#include "litert/python/litert_wrapper/tensor_buffer_wrapper/tensor_buffer_wrapper.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/python/litert_wrapper/common/litert_wrapper_utils.h"

namespace litert::tensor_buffer_wrapper {

namespace {
// A deallocator that performs no operation, used when the memory is managed
// externally.
void NoopDeallocator(void*) {}

// Converts a Python list of numeric values to a std::vector<float>.
// Returns true on success, false on failure with error message populated.
bool ConvertPyListToFloatVector(PyObject* py_list, std::vector<float>* out,
                                std::string* error) {
  if (!PyList_Check(py_list)) {
    *error = "Expected a Python list for float32 data";
    return false;
  }
  Py_ssize_t length = PyList_Size(py_list);
  out->reserve(length);
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PyList_GetItem(py_list, i);
    double val = PyFloat_AsDouble(item);
    if ((val == -1.0) && PyErr_Occurred()) {
      *error = "Non-numeric value in float list.";
      return false;
    }
    out->push_back(static_cast<float>(val));
  }
  return true;
}

// Converts a Python list of integers to a std::vector<int32_t>.
// Returns true on success, false on failure with error message populated.
bool ConvertPyListToInt32Vector(PyObject* py_list, std::vector<int32_t>* out,
                                std::string* error) {
  if (!PyList_Check(py_list)) {
    *error = "Expected a Python list for int32 data";
    return false;
  }
  Py_ssize_t length = PyList_Size(py_list);
  out->reserve(length);
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PyList_GetItem(py_list, i);
    if (!PyLong_Check(item)) {
      *error = "Non-integer value in int32 list.";
      return false;
    }
    int64_t val = PyLong_AsLong(item);
    if ((val == -1) && PyErr_Occurred()) {
      *error = "Error converting python int to int32.";
      return false;
    }
    out->push_back(static_cast<int32_t>(val));
  }
  return true;
}

// Converts a Python list of integers to a std::vector<int8_t>.
// Returns true on success, false on failure with error message populated.
// Validates that values are within the int8_t range [-128, 127].
bool ConvertPyListToInt8Vector(PyObject* py_list, std::vector<int8_t>* out,
                               std::string* error) {
  if (!PyList_Check(py_list)) {
    *error = "Expected a Python list for int8 data";
    return false;
  }
  Py_ssize_t length = PyList_Size(py_list);
  out->reserve(length);
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PyList_GetItem(py_list, i);
    if (!PyLong_Check(item)) {
      *error = "Non-integer value in int8 list.";
      return false;
    }
    int64_t val = PyLong_AsLong(item);
    if ((val == -1) && PyErr_Occurred()) {
      *error = "Error converting python int to int8.";
      return false;
    }
    if (val < -128 || val > 127) {
      *error = "Value out of range for int8 [-128..127].";
      return false;
    }
    out->push_back(static_cast<int8_t>(val));
  }
  return true;
}

// Creates a Python list from a span of float values.
PyObject* BuildPyListFromFloat(absl::Span<const float> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyFloat_FromDouble(data[i]));
  }
  return py_list;
}

// Creates a Python list from a span of int32_t values.
PyObject* BuildPyListFromInt32(absl::Span<const int32_t> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(data[i]));
  }
  return py_list;
}

// Creates a Python list from a span of int8_t values.
PyObject* BuildPyListFromInt8(absl::Span<const int8_t> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(data[i]));
  }
  return py_list;
}

}  // namespace

// Returns the size in bytes for the given data type.
size_t TensorBufferWrapper::ByteWidthOfDType(const std::string& dtype) {
  if (dtype == "float32") return 4;
  if (dtype == "int32") return 4;
  if (dtype == "int8") return 1;
  // Return 0 for unsupported data types
  return 0;
}

// Creates a TensorBuffer from existing host memory.
// The memory is referenced, not copied, so the original data must outlive
// the TensorBuffer unless it's explicitly copied.
PyObject* TensorBufferWrapper::CreateFromHostMemory(PyObject* py_data,
                                                    const std::string& dtype,
                                                    Py_ssize_t num_elements) {
  // Acquire a read buffer from py_data
  Py_buffer py_buf;
  if (PyObject_GetBuffer(py_data, &py_buf, PyBUF_CONTIG_RO) < 0) {
    return nullptr;  // PyErr already set
  }

  size_t dtype_size = ByteWidthOfDType(dtype);
  if (dtype_size == 0) {
    PyBuffer_Release(&py_buf);
    return ReportError("Unsupported dtype in CreateFromHostMemory: " + dtype);
  }
  size_t required_size = static_cast<size_t>(num_elements) * dtype_size;
  if (static_cast<size_t>(py_buf.len) < required_size) {
    PyBuffer_Release(&py_buf);
    return ReportError("Python buffer is too small for required size");
  }

  // Create a LiteRtRankedTensorType for 1-D shape
  LiteRtRankedTensorType dummy_type;
  dummy_type.layout.rank = 1;
  dummy_type.layout.dimensions[0] = static_cast<int32_t>(num_elements);
  dummy_type.layout.has_strides = false;

  // Set the element type based on the dtype string
  if (dtype == "float32") {
    dummy_type.element_type = kLiteRtElementTypeFloat32;
  } else if (dtype == "int8") {
    dummy_type.element_type = kLiteRtElementTypeInt8;
  } else if (dtype == "int32") {
    dummy_type.element_type = kLiteRtElementTypeInt32;
  } else {
    dummy_type.element_type = kLiteRtElementTypeNone;
  }

  // Create the tensor buffer
  LiteRtTensorBuffer tensor_buffer = nullptr;
  LiteRtStatus status = LiteRtCreateTensorBufferFromHostMemory(
      &dummy_type, py_buf.buf, required_size, &NoopDeallocator, &tensor_buffer);
  if (status != kLiteRtStatusOk) {
    PyBuffer_Release(&py_buf);
    return ReportError("Failed LiteRtCreateTensorBufferFromHostMemory");
  }

  // Create a context structure to manage the lifetime of resources
  struct CapsuleContext {
    Py_buffer py_buf;
    PyObject* py_obj;
    LiteRtTensorBuffer c_tensor_buffer;
  };
  auto* ctx = new CapsuleContext();
  ctx->py_buf = py_buf;
  ctx->py_obj = py_data;
  ctx->c_tensor_buffer = tensor_buffer;

  // Keep a reference to the original py_data to prevent garbage collection
  Py_INCREF(py_data);

  // Define the capsule destructor to clean up resources
  auto capsule_destructor = [](PyObject* capsule) {
    // Safely destroy the tensor buffer if needed
    litert_wrapper_utils::DestroyTensorBufferFromCapsule(capsule);

    // Clean up the capsule context
    auto* context = static_cast<CapsuleContext*>(PyCapsule_GetContext(capsule));
    if (context) {
      PyBuffer_Release(&context->py_buf);
      Py_DECREF(context->py_obj);
      delete context;
    }
  };

  // Create a PyCapsule to own the tensor buffer and associated resources
  PyObject* capsule = PyCapsule_New(
      tensor_buffer, litert_wrapper_utils::kLiteRtTensorBufferName.data(),
      capsule_destructor);
  if (!capsule) {
    LiteRtDestroyTensorBuffer(tensor_buffer);
    PyBuffer_Release(&py_buf);
    Py_DECREF(py_data);
    delete ctx;
    return ReportError("Failed to create capsule in CreateFromHostMemory");
  }
  PyCapsule_SetContext(capsule, ctx);
  return capsule;
}

// Writes data from a Python list to a TensorBuffer.
// Supports float32, int32, and int8 data types.
PyObject* TensorBufferWrapper::WriteTensor(PyObject* buffer_capsule,
                                           PyObject* data_list,
                                           const std::string& dtype) {
  if (!PyCapsule_CheckExact(buffer_capsule)) {
    return ReportError("WriteTensor: invalid capsule");
  }
  void* ptr = PyCapsule_GetPointer(
      buffer_capsule, litert_wrapper_utils::kLiteRtTensorBufferName.data());
  if (!ptr) {
    return ReportError("WriteTensor: null pointer in capsule");
  }
  TensorBuffer tb = TensorBuffer::WrapCObject(
      static_cast<LiteRtTensorBuffer>(ptr), OwnHandle::kNo);

  // Convert the Python list to a C++ vector based on the data type
  std::string error;
  if (dtype == "float32") {
    std::vector<float> host_data;
    if (!ConvertPyListToFloatVector(data_list, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    if (auto status = tb.Write<float>(absl::MakeConstSpan(host_data)); !status)
      return ConvertErrorToPyExc(status.Error());
    Py_RETURN_NONE;
  }
  if (dtype == "int32") {
    std::vector<int32_t> host_data;
    if (!ConvertPyListToInt32Vector(data_list, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    if (auto status = tb.Write<int32_t>(absl::MakeConstSpan(host_data));
        !status)
      return ConvertErrorToPyExc(status.Error());
    Py_RETURN_NONE;
  }
  if (dtype == "int8") {
    std::vector<int8_t> host_data;
    if (!ConvertPyListToInt8Vector(data_list, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    if (auto status = tb.Write<int8_t>(absl::MakeConstSpan(host_data)); !status)
      return ConvertErrorToPyExc(status.Error());
    Py_RETURN_NONE;
  }
  return ReportError("WriteTensor: unsupported dtype '" + dtype + "'");
}

// Reads data from a TensorBuffer into a Python list.
// Supports float32, int32, and int8 data types.
PyObject* TensorBufferWrapper::ReadTensor(PyObject* buffer_capsule,
                                          int num_elements,
                                          const std::string& dtype) {
  if (!PyCapsule_CheckExact(buffer_capsule)) {
    return ReportError("ReadTensor: invalid capsule");
  }
  void* ptr = PyCapsule_GetPointer(
      buffer_capsule, litert_wrapper_utils::kLiteRtTensorBufferName.data());
  if (!ptr) {
    return ReportError("ReadTensor: null pointer in capsule");
  }
  TensorBuffer tb = TensorBuffer::WrapCObject(
      static_cast<LiteRtTensorBuffer>(ptr), OwnHandle::kNo);

  if (dtype == "float32") {
    std::vector data(num_elements, 0.f);
    if (auto status = tb.Read<float>(absl::MakeSpan(data)); !status)
      return ConvertErrorToPyExc(status.Error());
    return BuildPyListFromFloat(data);
  }
  if (dtype == "int32") {
    std::vector data(num_elements, 0);
    if (auto status = tb.Read<int32_t>(absl::MakeSpan(data)); !status)
      return ConvertErrorToPyExc(status.Error());
    return BuildPyListFromInt32(data);
  }
  if (dtype == "int8") {
    std::vector<int8_t> data(num_elements, 0);
    if (auto status = tb.Read<int8_t>(absl::MakeSpan(data)); !status)
      return ConvertErrorToPyExc(status.Error());
    return BuildPyListFromInt8(data);
  }
  return ReportError("ReadTensor: unsupported dtype '" + dtype + "'");
}

// Explicitly destroys a TensorBuffer and releases associated resources.
PyObject* TensorBufferWrapper::DestroyTensorBuffer(PyObject* buffer_capsule) {
  if (PyCapsule_CheckExact(buffer_capsule)) {
    litert_wrapper_utils::DestroyTensorBufferFromCapsule(buffer_capsule);
  }
  Py_RETURN_NONE;
}

// Reports an error by setting a Python exception.
PyObject* TensorBufferWrapper::ReportError(const std::string& msg) {
  PyErr_SetString(PyExc_RuntimeError, msg.c_str());
  return nullptr;
}

// Converts a LiteRT Error to a Python exception.
PyObject* TensorBufferWrapper::ConvertErrorToPyExc(const Error& error) {
  PyErr_Format(PyExc_RuntimeError,
               "TensorBufferWrapper error: code=%d, message=%s",
               static_cast<int>(error.Status()), error.Message().c_str());
  return nullptr;
}

}  // namespace litert::tensor_buffer_wrapper
