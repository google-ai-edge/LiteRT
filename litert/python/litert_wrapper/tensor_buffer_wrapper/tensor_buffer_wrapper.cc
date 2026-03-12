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
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/python/litert_wrapper/common/litert_wrapper_utils.h"

namespace litert::tensor_buffer_wrapper {

namespace {
const char* ElementTypeToString(ElementType dtype) {
  switch (dtype) {
    case ElementType::Float32:
      return "float32";
    case ElementType::Float16:
      return "float16";
    case ElementType::Int32:
      return "int32";
    case ElementType::UInt8:
      return "uint8";
    case ElementType::Int64:
      return "int64";
    case ElementType::Bool:
      return "bool";
    case ElementType::Int16:
      return "int16";
    case ElementType::Int8:
      return "int8";
    case ElementType::Float64:
      return "float64";
    case ElementType::UInt32:
      return "uint32";
    case ElementType::UInt16:
      return "uint16";
    default:
      return "unknown";
  }
}

// A deallocator that performs no operation, used when the memory is managed
// externally.
void NoopDeallocator(void*) {}

template <typename T>
struct ConvertTraits;

template <>
struct ConvertTraits<float> {
  using StorageType = float;
  using IntermediateType = double;
  static constexpr const char* kName = "float32";
  static constexpr bool kPyCheck = false;
  static bool PythonCheck(PyObject* item) { return true; }
  static IntermediateType Convert(PyObject* item) {
    return PyFloat_AsDouble(item);
  }
  static bool IsError(IntermediateType val) {
    return val == -1.0 && PyErr_Occurred();
  }
  static constexpr const char* kItemError = "Non-numeric";
};

template <>
struct ConvertTraits<int32_t> {
  using StorageType = int32_t;
  using IntermediateType = int64_t;
  static constexpr const char* kName = "int32";
  static constexpr bool kPyCheck = true;
  static bool PythonCheck(PyObject* item) { return PyLong_Check(item); }
  static IntermediateType Convert(PyObject* item) { return PyLong_AsLong(item); }
  static bool IsError(IntermediateType val) {
    return val == -1 && PyErr_Occurred();
  }
  static constexpr const char* kItemError = "Non-integer";
};

template <>
struct ConvertTraits<uint16_t> {
  using StorageType = uint16_t;
  using IntermediateType = unsigned long;  // NOLINT(google-runtime-int)
  static constexpr const char* kName = "float16 bit-pattern";
  static constexpr bool kPyCheck = true;
  static bool PythonCheck(PyObject* item) { return PyLong_Check(item); }
  static IntermediateType Convert(PyObject* item) {
    return PyLong_AsUnsignedLong(item);
  }
  static bool IsError(IntermediateType val) {
    // NOLINTNEXTLINE(google-runtime-int)
    return val == static_cast<unsigned long>(-1) && PyErr_Occurred();
  }
  static constexpr const char* kItemError = "Non-integer";
};

template <>
struct ConvertTraits<int8_t> {
  using StorageType = int8_t;
  using IntermediateType = int64_t;
  static constexpr const char* kName = "int8";
  static constexpr bool kPyCheck = true;
  static bool PythonCheck(PyObject* item) { return PyLong_Check(item); }
  static IntermediateType Convert(PyObject* item) { return PyLong_AsLong(item); }
  static bool IsError(IntermediateType val) {
    return val == -1 && PyErr_Occurred();
  }
  static constexpr const char* kItemError = "Non-integer";
};

template <class T>
bool ConvertPyListToVector(
    PyObject* py_list,
    std::vector<typename ConvertTraits<T>::StorageType>* out,
    std::string* error) {
  using Traits = ConvertTraits<T>;
  using StorageType = typename Traits::StorageType;
  if (!PyList_Check(py_list)) {
    *error =
        "Expected a Python list for " + std::string(Traits::kName) + " data";
    return false;
  }
  Py_ssize_t length = PyList_Size(py_list);
  out->reserve(length);
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PyList_GetItem(py_list, i);
    if (Traits::kPyCheck && !Traits::PythonCheck(item)) {
      *error = std::string(Traits::kItemError) + " value in " +
               std::string(Traits::kName) + " list.";
      return false;
    }
    auto val = Traits::Convert(item);
    if (Traits::IsError(val)) {
      *error =
          "Error converting python value to " + std::string(Traits::kName);
      return false;
    }
    if (val < std::numeric_limits<StorageType>::lowest() ||
        val > std::numeric_limits<StorageType>::max()) {
      *error = "Value out of range for " + std::string(Traits::kName);
      return false;
    }
    out->push_back(static_cast<StorageType>(val));
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

// Creates a Python list from a span of uint16_t values.
PyObject* BuildPyListFromUInt16(absl::Span<const uint16_t> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromUnsignedLong(data[i]));
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
  if (dtype == "float16") return 2;
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
  } else if (dtype == "float16") {
    dummy_type.element_type = kLiteRtElementTypeFloat16;
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
    if (!ConvertPyListToVector<float>(data_list, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    if (auto status = tb.Write<float>(absl::MakeConstSpan(host_data)); !status)
      return ConvertErrorToPyExc(status.Error());
    Py_RETURN_NONE;
  }
  if (dtype == "float16") {
    std::vector<uint16_t> host_data;
    if (!ConvertPyListToVector<uint16_t>(data_list, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    if (auto status = tb.Write<uint16_t>(absl::MakeConstSpan(host_data));
        !status) {
      return ConvertErrorToPyExc(status.Error());
    }
    Py_RETURN_NONE;
  }
  if (dtype == "int32") {
    std::vector<int32_t> host_data;
    if (!ConvertPyListToVector<int32_t>(data_list, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    if (auto status = tb.Write<int32_t>(absl::MakeConstSpan(host_data));
        !status)
      return ConvertErrorToPyExc(status.Error());
    Py_RETURN_NONE;
  }
  if (dtype == "int8") {
    std::vector<int8_t> host_data;
    if (!ConvertPyListToVector<int8_t>(data_list, &host_data, &error)) {
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
  if (dtype == "float16") {
    std::vector<uint16_t> data(num_elements, 0);
    if (auto status = tb.Read<uint16_t>(absl::MakeSpan(data)); !status)
      return ConvertErrorToPyExc(status.Error());
    return BuildPyListFromUInt16(data);
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

PyObject* TensorBufferWrapper::GetTensorDetails(PyObject* buffer_capsule) {
  if (!PyCapsule_CheckExact(buffer_capsule)) {
    return ReportError("GetTensorDetails: invalid capsule");
  }
  void* ptr = PyCapsule_GetPointer(
      buffer_capsule, litert_wrapper_utils::kLiteRtTensorBufferName.data());
  if (!ptr) {
    return ReportError("GetTensorDetails: null pointer in capsule");
  }
  TensorBuffer tb = TensorBuffer::WrapCObject(
      static_cast<LiteRtTensorBuffer>(ptr), OwnHandle::kNo);
  auto tensor_type_or = tb.TensorType();
  if (!tensor_type_or) {
    return ConvertErrorToPyExc(tensor_type_or.Error());
  }
  const auto& tensor_type = *tensor_type_or;

  PyObject* tensor_dict = PyDict_New();
  PyDict_SetItemString(
      tensor_dict, "dtype",
      PyUnicode_FromString(ElementTypeToString(tensor_type.ElementType())));
  const auto shape = tensor_type.Layout().Dimensions();
  PyObject* shape_list = PyList_New(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    PyList_SetItem(shape_list, i, PyLong_FromLong(shape[i]));
  }
  PyDict_SetItemString(tensor_dict, "shape", shape_list);
  Py_DECREF(shape_list);
  return tensor_dict;
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
