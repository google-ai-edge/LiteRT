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

#include "litert/python/litert_wrapper/compiled_model_wrapper/compiled_model_wrapper.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/python/litert_wrapper/common/litert_wrapper_utils.h"

namespace litert::compiled_model_wrapper {

namespace {
// Converts ElementType to a string representation for Python.
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
}  // namespace

// Returns the byte width of a data type.
size_t CompiledModelWrapper::ByteWidthOfDType(const std::string& dtype) {
  if (dtype == "float32") return 4;
  if (dtype == "int8") return 1;
  if (dtype == "int32") return 4;
  return 0;  // Unknown type
}

// Constructor for CompiledModelWrapper.
CompiledModelWrapper::CompiledModelWrapper(Environment env, ExtendedModel model,
                                           CompiledModel compiled)
    : environment_(std::move(env)),
      model_(std::move(model)),
      compiled_model_(std::move(compiled)) {}

// Destructor for CompiledModelWrapper.
CompiledModelWrapper::~CompiledModelWrapper() {
  // Release Python buffer reference if we're holding one
  // Check Py_IsInitialized to handle destruction during Python shutdown
  if (model_buffer_ && Py_IsInitialized()) {
    Py_DECREF(model_buffer_);
    model_buffer_ = nullptr;
  }
}

// Reports an error by setting a Python exception.
PyObject* CompiledModelWrapper::ReportError(const std::string& msg) {
  PyErr_SetString(PyExc_RuntimeError, msg.c_str());
  return nullptr;
}

// Converts a LiteRT error to a Python exception.
PyObject* CompiledModelWrapper::ConvertErrorToPyExc(const Error& error) {
  PyErr_Format(PyExc_RuntimeError, "CompiledModel error: code=%d, message=%s",
               static_cast<int>(error.Status()), error.Message().c_str());
  return nullptr;
}

// Creates a CompiledModelWrapper from a model file.
CompiledModelWrapper* CompiledModelWrapper::CreateWrapperFromFile(
    const char* model_path, const char* runtime_path,
    const char* compiler_plugin_path, const char* dispatch_library_path,
    int hardware_accel, std::string* out_error) {
  // Create an environment with options
  std::vector<litert::EnvironmentOptions::Option> options;
  std::string runtime_path_str = std::string(runtime_path);
  if (!runtime_path_str.empty()) {
    options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, runtime_path_str});
  }
  if (compiler_plugin_path && *compiler_plugin_path) {
    options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
        std::string(compiler_plugin_path)});
  }
  if (dispatch_library_path && *dispatch_library_path) {
    options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
        std::string(dispatch_library_path)});
  }
  auto env_or = litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(options)));
  if (!env_or) {
    if (out_error) *out_error = env_or.Error().Message();
    return nullptr;
  }
  Environment env = std::move(*env_or);

  // Load model from a file
  auto model_or = ExtendedModel::CreateFromFile(model_path);
  if (!model_or) {
    if (out_error) *out_error = model_or.Error().Message();
    return nullptr;
  }
  ExtendedModel model = std::move(*model_or);

  // Create a compiled model
  auto compiled_or = CompiledModel::Create(
      env, model.Get(), static_cast<HwAccelerators>(hardware_accel));
  if (!compiled_or) {
    if (out_error) *out_error = compiled_or.Error().Message();
    return nullptr;
  }

  return new CompiledModelWrapper(std::move(env), std::move(model),
                                  std::move(*compiled_or));
}

// Converts a Python string or bytes object to a C string.
int ConvertFromPyString(PyObject* obj, char** data, Py_ssize_t* length) {
  if (PyUnicode_Check(obj)) {
    *data = const_cast<char*>(PyUnicode_AsUTF8AndSize(obj, length));
    return *data == nullptr ? -1 : 0;
  }
  return PyBytes_AsStringAndSize(obj, data, length);
}

// Creates a CompiledModelWrapper from a model buffer.
CompiledModelWrapper* CompiledModelWrapper::CreateWrapperFromBuffer(
    PyObject* model_data, const char* runtime_path,
    const char* compiler_plugin_path, const char* dispatch_library_path,
    int hardware_accel, std::string* out_error) {
  // Extract buffer from Python object
  char* buf = nullptr;
  Py_ssize_t length = 0;
  if (ConvertFromPyString(model_data, &buf, &length) == -1) {
    if (out_error) *out_error = "Failed converting PyObject to buffer";
    return nullptr;
  }

  // Create environment with options
  std::vector<litert::EnvironmentOptions::Option> options;
  std::string runtime_path_str = std::string(runtime_path);
  if (!runtime_path_str.empty()) {
    options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, runtime_path_str});
  }
  if (compiler_plugin_path && *compiler_plugin_path) {
    options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
        std::string(compiler_plugin_path)});
  }
  if (dispatch_library_path && *dispatch_library_path) {
    options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
        std::string(dispatch_library_path)});
  }

  auto env_or = litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(options)));
  if (!env_or) {
    if (out_error) *out_error = env_or.Error().Message();
    return nullptr;
  }
  Environment env = std::move(*env_or);

  // Create model from buffer
  BufferRef<uint8_t> ref(reinterpret_cast<uint8_t*>(buf),
                         static_cast<size_t>(length));
  auto model_or = ExtendedModel::CreateFromBuffer(ref);
  if (!model_or) {
    if (out_error) *out_error = model_or.Error().Message();
    return nullptr;
  }
  ExtendedModel model = std::move(*model_or);

  // Create a compiled model
  auto compiled_or = CompiledModel::Create(
      env, model.Get(), static_cast<HwAccelerators>(hardware_accel));
  if (!compiled_or) {
    if (out_error) *out_error = compiled_or.Error().Message();
    return nullptr;
  }

  auto* wrapper = new CompiledModelWrapper(std::move(env), std::move(model),
                                           std::move(*compiled_or));
  // Keep the Python buffer alive for the lifetime of the wrapper
  Py_INCREF(model_data);
  wrapper->model_buffer_ = model_data;
  return wrapper;
}

// Returns a dictionary of all signatures in the model.
PyObject* CompiledModelWrapper::GetSignatureList() {
  auto sigs_or = model_.GetSignatures();
  if (!sigs_or) {
    return ConvertErrorToPyExc(sigs_or.Error());
  }
  auto sigs = std::move(*sigs_or);
  PyObject* py_dict = PyDict_New();

  for (size_t i = 0; i < sigs.size(); ++i) {
    const auto& sig = sigs[i];
    PyObject* sig_info = PyDict_New();

    // Add input names
    PyObject* py_in = PyList_New(0);
    for (auto& n : sig.InputNames()) {
      PyList_Append(py_in, PyUnicode_FromString(n.data()));
    }

    // Add output names
    PyObject* py_out = PyList_New(0);
    for (auto& n : sig.OutputNames()) {
      PyList_Append(py_out, PyUnicode_FromString(n.data()));
    }

    PyDict_SetItemString(sig_info, "inputs", py_in);
    PyDict_SetItemString(sig_info, "outputs", py_out);

    Py_DECREF(py_in);
    Py_DECREF(py_out);

    // Add signature to root dictionary
    PyDict_SetItemString(py_dict, sig.Key().data(), sig_info);
    Py_DECREF(sig_info);
  }
  return py_dict;
}

// Returns details about a signature by index.
PyObject* CompiledModelWrapper::GetSignatureByIndex(int signature_index) {
  auto sig_or = model_.GetSignature(signature_index);
  if (!sig_or) {
    return ConvertErrorToPyExc(sig_or.Error());
  }
  auto sig = std::move(*sig_or);

  PyObject* result = PyDict_New();
  // Add signature key
  PyDict_SetItemString(result, "key", PyUnicode_FromString(sig.Key().data()));

  // Add input names
  {
    PyObject* py_in = PyList_New(0);
    for (auto& nm : sig.InputNames()) {
      PyList_Append(py_in, PyUnicode_FromString(nm.data()));
    }
    PyDict_SetItemString(result, "inputs", py_in);
    Py_DECREF(py_in);
  }

  // Add output names
  {
    PyObject* py_out = PyList_New(0);
    for (auto& nm : sig.OutputNames()) {
      PyList_Append(py_out, PyUnicode_FromString(nm.data()));
    }
    PyDict_SetItemString(result, "outputs", py_out);
    Py_DECREF(py_out);
  }

  return result;
}

// Returns the number of signatures in the model.
PyObject* CompiledModelWrapper::GetNumSignatures() {
  auto num = model_.GetNumSignatures();
  return PyLong_FromLong(static_cast<int64_t>(num));
}

// Returns the index of a signature by key.
PyObject* CompiledModelWrapper::GetSignatureIndex(const char* signature_key) {
  auto idx_or = model_.GetSignatureIndex(signature_key);
  if (!idx_or) {
    // Return -1 if not found
    return PyLong_FromLong(-1);
  }
  return PyLong_FromLong(*idx_or);
}

// Returns requirements for an input buffer.
PyObject* CompiledModelWrapper::GetInputBufferRequirements(int signature_index,
                                                           int input_index) {
  auto req_or = compiled_model_.GetInputBufferRequirements(
      (size_t)signature_index, (size_t)input_index);
  if (!req_or) {
    return ConvertErrorToPyExc(req_or.Error());
  }
  auto req = std::move(*req_or);
  PyObject* dict = PyDict_New();

  // Add buffer size
  auto size_or = req.BufferSize();
  if (!size_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(size_or.Error());
  }
  PyDict_SetItemString(dict, "buffer_size",
                       PyLong_FromLong(static_cast<int64_t>(*size_or)));

  // Add supported types
  auto types_or = req.SupportedTypes();
  if (!types_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(types_or.Error());
  }
  auto types = std::move(*types_or);
  PyObject* py_list = PyList_New(static_cast<Py_ssize_t>(types.size()));
  for (size_t i = 0; i < types.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(static_cast<int64_t>(types[i])));
  }
  PyDict_SetItemString(dict, "supported_types", py_list);
  Py_DECREF(py_list);

  return dict;
}

// Returns requirements for an output buffer.
PyObject* CompiledModelWrapper::GetOutputBufferRequirements(int signature_index,
                                                            int output_index) {
  auto req_or = compiled_model_.GetOutputBufferRequirements(
      static_cast<size_t>(signature_index), static_cast<size_t>(output_index));
  if (!req_or) {
    return ConvertErrorToPyExc(req_or.Error());
  }
  auto req = std::move(*req_or);

  PyObject* dict = PyDict_New();

  // Add buffer size
  auto size_or = req.BufferSize();
  if (!size_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(size_or.Error());
  }
  PyDict_SetItemString(dict, "buffer_size",
                       PyLong_FromLong(static_cast<int64_t>(*size_or)));

  auto types_or = req.SupportedTypes();
  if (!types_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(types_or.Error());
  }
  auto types = std::move(*types_or);
  PyObject* py_list = PyList_New(static_cast<Py_ssize_t>(types.size()));
  for (size_t i = 0; i < types.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(static_cast<int64_t>(types[i])));
  }
  PyDict_SetItemString(dict, "supported_types", py_list);
  Py_DECREF(py_list);

  return dict;
}

PyObject* CompiledModelWrapper::CreateInputBufferByName(
    PyObject* self_wrapper, const char* signature_key, const char* input_name) {
  auto buffer_or = compiled_model_.CreateInputBuffer(signature_key, input_name);
  if (!buffer_or) {
    return ConvertErrorToPyExc(buffer_or.Error());
  }
  auto buffer = std::move(*buffer_or);

  PyObject* capsule =
      litert_wrapper_utils::MakeTensorBufferCapsule(buffer, self_wrapper);
  return capsule;
}

PyObject* CompiledModelWrapper::CreateOutputBufferByName(
    PyObject* self_wrapper, const char* signature_key,
    const char* output_name) {
  auto buffer_or =
      compiled_model_.CreateOutputBuffer(signature_key, output_name);
  if (!buffer_or) {
    return ConvertErrorToPyExc(buffer_or.Error());
  }
  auto buffer = std::move(*buffer_or);

  PyObject* capsule =
      litert_wrapper_utils::MakeTensorBufferCapsule(buffer, self_wrapper);
  return capsule;
}

PyObject* CompiledModelWrapper::CreateInputBuffers(PyObject* self_wrapper,
                                                   int signature_index) {
  auto buffers_or =
      compiled_model_.CreateInputBuffers(static_cast<size_t>(signature_index));
  if (!buffers_or) {
    return ConvertErrorToPyExc(buffers_or.Error());
  }
  auto buffers = std::move(*buffers_or);
  PyObject* py_list = PyList_New(buffers.size());
  for (size_t i = 0; i < buffers.size(); i++) {
    // Python owns them. Destroy on capsule destructor.
    PyObject* capsule =
        litert_wrapper_utils::MakeTensorBufferCapsule(buffers[i], self_wrapper);
    PyList_SetItem(py_list, i, capsule);  // steal ref
  }
  return py_list;
}

PyObject* CompiledModelWrapper::CreateOutputBuffers(PyObject* self_wrapper,
                                                    int signature_index) {
  auto buffers_or =
      compiled_model_.CreateOutputBuffers(static_cast<size_t>(signature_index));
  if (!buffers_or) {
    return ConvertErrorToPyExc(buffers_or.Error());
  }
  auto buffers = std::move(*buffers_or);
  PyObject* py_list = PyList_New(buffers.size());
  for (size_t i = 0; i < buffers.size(); i++) {
    PyObject* capsule =
        litert_wrapper_utils::MakeTensorBufferCapsule(buffers[i], self_wrapper);
    PyList_SetItem(py_list, i, capsule);
  }
  return py_list;
}

PyObject* CompiledModelWrapper::GetInputTensorDetails(
    const char* signature_key) {
  auto sig_or = model_.FindSignature(signature_key);
  if (!sig_or) {
    return ConvertErrorToPyExc(sig_or.Error());
  }
  auto sig = std::move(*sig_or);
  auto input_names = sig.InputNames();
  PyObject* result_dict = PyDict_New();
  for (const auto& name : input_names) {
    auto tensor_or = sig.InputTensor(name);
    if (!tensor_or) {
      Py_DECREF(result_dict);
      return ConvertErrorToPyExc(tensor_or.Error());
    }
    auto tensor = std::move(*tensor_or);
    PyObject* tensor_dict = PyDict_New();
    PyDict_SetItemString(tensor_dict, "name",
                         PyUnicode_FromString(tensor.Name().data()));
    PyDict_SetItemString(tensor_dict, "index",
                         PyLong_FromLong(tensor.TensorIndex()));
    PyDict_SetItemString(
        tensor_dict, "dtype",
        PyUnicode_FromString(ElementTypeToString(tensor.ElementType())));
    if (tensor.TypeId() == kLiteRtRankedTensorType) {
      auto ranked_type_or = tensor.RankedTensorType();
      if (!ranked_type_or) {
        Py_DECREF(tensor_dict);
        Py_DECREF(result_dict);
        return ConvertErrorToPyExc(ranked_type_or.Error());
      }
      auto shape = ranked_type_or->Layout().Dimensions();
      PyObject* shape_list = PyList_New(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) {
        PyList_SetItem(shape_list, i, PyLong_FromLong(shape[i]));
      }
      PyDict_SetItemString(tensor_dict, "shape", shape_list);
      Py_DECREF(shape_list);
    }
    PyDict_SetItemString(result_dict, name.data(), tensor_dict);
    Py_DECREF(tensor_dict);
  }
  return result_dict;
}

PyObject* CompiledModelWrapper::RunByName(const char* signature_key,
                                          PyObject* input_map,
                                          PyObject* output_map) {
  if (!PyDict_Check(input_map) || !PyDict_Check(output_map)) {
    return ReportError("RunByName expects input_map & output_map as dict");
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_map;
  absl::flat_hash_map<absl::string_view, TensorBuffer> out_map;

  PyObject* key;
  PyObject* val;
  Py_ssize_t pos = 0;
  while (PyDict_Next(input_map, &pos, &key, &val)) {
    if (!PyUnicode_Check(key)) {
      return ReportError("input_map key not a string.");
    }
    const char* nm = PyUnicode_AsUTF8(key);

    if (!PyCapsule_CheckExact(val)) {
      return ReportError("input_map value not a capsule.");
    }
    void* ptr = PyCapsule_GetPointer(
        val, litert_wrapper_utils::kLiteRtTensorBufferName.data());
    if (!ptr) {
      return ReportError("capsule missing pointer in input_map");
    }
    in_map[nm] = TensorBuffer::WrapCObject(static_cast<LiteRtTensorBuffer>(ptr),
                                           OwnHandle::kNo);
  }

  pos = 0;
  while (PyDict_Next(output_map, &pos, &key, &val)) {
    if (!PyUnicode_Check(key)) {
      return ReportError("output_map key not a string.");
    }
    const char* nm = PyUnicode_AsUTF8(key);

    if (!PyCapsule_CheckExact(val)) {
      return ReportError("output_map value not a capsule.");
    }
    void* ptr = PyCapsule_GetPointer(
        val, litert_wrapper_utils::kLiteRtTensorBufferName.data());
    if (!ptr) {
      return ReportError("capsule missing pointer in output_map");
    }
    out_map[nm] = TensorBuffer::WrapCObject(
        static_cast<LiteRtTensorBuffer>(ptr), OwnHandle::kNo);
  }

  if (auto run_or = compiled_model_.Run(signature_key, in_map, out_map);
      !run_or) {
    return ConvertErrorToPyExc(run_or.Error());
  }
  Py_RETURN_NONE;
}

PyObject* CompiledModelWrapper::RunByIndex(int signature_index,
                                           PyObject* input_caps_list,
                                           PyObject* output_caps_list) {
  if (!PyList_Check(input_caps_list)) {
    return ReportError("RunByIndex input_caps_list not list");
  }
  if (!PyList_Check(output_caps_list)) {
    return ReportError("RunByIndex output_caps_list not list");
  }
  std::vector<TensorBuffer> inputs;
  std::vector<TensorBuffer> outputs;

  Py_ssize_t n_in = PyList_Size(input_caps_list);
  inputs.reserve(n_in);
  for (Py_ssize_t i = 0; i < n_in; i++) {
    PyObject* elem = PyList_GetItem(input_caps_list, i);  // borrowed
    if (!PyCapsule_CheckExact(elem)) {
      return ReportError("input_caps_list element not a capsule");
    }
    void* ptr = PyCapsule_GetPointer(
        elem, litert_wrapper_utils::kLiteRtTensorBufferName.data());
    if (!ptr) {
      return ReportError("Missing pointer in input capsule");
    }
    inputs.emplace_back(TensorBuffer::WrapCObject(
        static_cast<LiteRtTensorBuffer>(ptr), OwnHandle::kNo));
  }

  Py_ssize_t n_out = PyList_Size(output_caps_list);
  outputs.reserve(n_out);
  for (Py_ssize_t i = 0; i < n_out; i++) {
    PyObject* elem = PyList_GetItem(output_caps_list, i);
    if (!PyCapsule_CheckExact(elem)) {
      return ReportError("output_caps_list element not a capsule");
    }
    void* ptr = PyCapsule_GetPointer(
        elem, litert_wrapper_utils::kLiteRtTensorBufferName.data());
    if (!ptr) {
      return ReportError("Missing pointer in output capsule");
    }
    outputs.emplace_back(TensorBuffer::WrapCObject(
        static_cast<LiteRtTensorBuffer>(ptr), OwnHandle::kNo));
  }

  if (auto run_or = compiled_model_.Run(static_cast<size_t>(signature_index),
                                        inputs, outputs);
      !run_or) {
    return ConvertErrorToPyExc(run_or.Error());
  }
  Py_RETURN_NONE;
}

}  // namespace litert::compiled_model_wrapper
