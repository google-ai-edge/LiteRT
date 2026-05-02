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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
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
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/python/litert_wrapper/common/litert_wrapper_utils.h"

namespace litert::compiled_model_wrapper {

namespace {

bool SetDictItemStringSteal(PyObject* dict, const char* key, PyObject* value) {
  if (value == nullptr) {
    return false;
  }
  const int status = PyDict_SetItemString(dict, key, value);
  Py_DECREF(value);
  return status == 0;
}

bool AppendUnicodeToList(PyObject* list, absl::string_view value) {
  PyObject* unicode = PyUnicode_FromStringAndSize(value.data(), value.size());
  if (unicode == nullptr) {
    return false;
  }
  const int status = PyList_Append(list, unicode);
  Py_DECREF(unicode);
  return status == 0;
}

PyObject* BuildShapeList(const Layout& layout) {
  const auto dims = layout.Dimensions();
  PyObject* shape_list = PyList_New(dims.size());
  if (shape_list == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    PyObject* dim = PyLong_FromLong(dims[i]);
    if (dim == nullptr) {
      Py_DECREF(shape_list);
      return nullptr;
    }
    PyList_SetItem(shape_list, i, dim);  // steal ref
  }
  return shape_list;
}

PyObject* BuildTensorDetailsDict(const SimpleTensor& tensor,
                                 const Layout* layout) {
  PyObject* tensor_dict = PyDict_New();
  if (tensor_dict == nullptr) {
    return nullptr;
  }

  std::string tensor_name(tensor.Name());
  if (!SetDictItemStringSteal(tensor_dict, "name",
                              PyUnicode_FromString(tensor_name.c_str())) ||
      !SetDictItemStringSteal(tensor_dict, "index",
                              PyLong_FromUnsignedLong(tensor.TensorIndex())) ||
      !SetDictItemStringSteal(
          tensor_dict, "dtype",
          PyUnicode_FromString(litert_wrapper_utils::ElementTypeToString(
              tensor.ElementType())))) {
    Py_DECREF(tensor_dict);
    return nullptr;
  }

  if (layout != nullptr) {
    PyObject* shape_list = BuildShapeList(*layout);
    if (shape_list == nullptr) {
      Py_DECREF(tensor_dict);
      return nullptr;
    }
    if (PyDict_SetItemString(tensor_dict, "shape", shape_list) != 0) {
      Py_DECREF(shape_list);
      Py_DECREF(tensor_dict);
      return nullptr;
    }
    Py_DECREF(shape_list);
  }

  return tensor_dict;
}

bool SetTensorDetailsDictItem(PyObject* result_dict, absl::string_view key,
                              PyObject* value) {
  if (value == nullptr) {
    return false;
  }
  std::string key_string(key);
  const int status =
      PyDict_SetItemString(result_dict, key_string.c_str(), value);
  Py_DECREF(value);
  return status == 0;
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
CompiledModelWrapper::CompiledModelWrapper(ExtendedModel model,
                                           CompiledModel compiled)
    : model_(std::move(model)), compiled_model_(std::move(compiled)) {}

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
    PyObject* environment_capsule, const char* model_path, int hardware_accel,
    int cpu_num_threads, bool gpu_enforce_f32, bool gpu_share_constant_tensors,
    int cpu_kernel_mode, int xnnpack_flags, const char* xnnpack_weight_cache_path,
    bool enable_constant_tensor_sharing, bool enable_infinite_float_capping,
    bool enable_benchmark_mode, bool enable_allow_src_quantized_fc_conv_ops,
    bool enable_hint_waiting_for_completion, std::string* out_error) {
  auto* env =
      litert_wrapper_utils::GetEnvironmentFromCapsule(environment_capsule);
  if (env == nullptr) {
    if (out_error) *out_error = "Invalid LiteRT Environment capsule";
    return nullptr;
  }

  // Load model from a file
  auto model_or = ExtendedModel::CreateFromFile(model_path);
  if (!model_or) {
    if (out_error) *out_error = model_or.Error().Message();
    return nullptr;
  }
  ExtendedModel model = std::move(*model_or);

  // Create a compiled model options
  auto options_or = litert::Options::Create();
  if (!options_or) {
    if (out_error) *out_error = options_or.Error().Message();
    return nullptr;
  }
  auto& options = *options_or;
  options.SetHardwareAccelerators(static_cast<HwAccelerators>(hardware_accel));

  if (gpu_enforce_f32 || gpu_share_constant_tensors || enable_constant_tensor_sharing || enable_infinite_float_capping || enable_benchmark_mode || enable_allow_src_quantized_fc_conv_ops || enable_hint_waiting_for_completion) {
    auto gpu_options_or = options.GetGpuOptions();
    if (!gpu_options_or) {
      if (out_error) *out_error = gpu_options_or.Error().Message();
      return nullptr;
    }
    if (gpu_enforce_f32) {
      gpu_options_or->SetPrecision(GpuOptions::Precision::kFp32);
    }
    if (gpu_share_constant_tensors || enable_constant_tensor_sharing) {
      gpu_options_or->EnableConstantTensorSharing(true);
    }
    if (enable_infinite_float_capping) {
      gpu_options_or->EnableInfiniteFloatCapping(true);
    }
    if (enable_benchmark_mode) {
      gpu_options_or->EnableBenchmarkMode(true);
    }
    if (enable_allow_src_quantized_fc_conv_ops) {
      gpu_options_or->EnableAllowSrcQuantizedFcConvOps(true);
    }
    if (enable_hint_waiting_for_completion) {
      gpu_options_or->HintWaitingForCompletion(true);
    }
  }

  if (cpu_num_threads > 0 || cpu_kernel_mode >= 0 || xnnpack_flags >= 0 ||
      (xnnpack_weight_cache_path != nullptr && *xnnpack_weight_cache_path)) {
    auto cpu_options_or = options.GetCpuOptions();
    if (!cpu_options_or) {
      if (out_error) *out_error = cpu_options_or.Error().Message();
      return nullptr;
    }
    if (cpu_num_threads > 0) {
      cpu_options_or->SetNumThreads(cpu_num_threads);
    }
    if (cpu_kernel_mode >= 0) {
      cpu_options_or->SetKernelMode(
          static_cast<LiteRtCpuKernelMode>(cpu_kernel_mode));
    }
    if (xnnpack_flags >= 0) {
      cpu_options_or->SetXNNPackFlags(static_cast<uint32_t>(xnnpack_flags));
    }
    if (xnnpack_weight_cache_path != nullptr && *xnnpack_weight_cache_path) {
      cpu_options_or->SetXNNPackWeightCachePath(xnnpack_weight_cache_path);
    }
  }

  // Create a compiled model
  auto compiled_or = CompiledModel::Create(*env, model.Get(), options);
  if (!compiled_or) {
    if (out_error) *out_error = compiled_or.Error().Message();
    return nullptr;
  }

  return new CompiledModelWrapper(std::move(model), std::move(*compiled_or));
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
    PyObject* environment_capsule, PyObject* model_data, int hardware_accel,
    int cpu_num_threads, bool gpu_enforce_f32, bool gpu_share_constant_tensors,
    int cpu_kernel_mode, int xnnpack_flags, const char* xnnpack_weight_cache_path,
    bool enable_constant_tensor_sharing, bool enable_infinite_float_capping,
    bool enable_benchmark_mode, bool enable_allow_src_quantized_fc_conv_ops,
    bool enable_hint_waiting_for_completion, std::string* out_error) {
  auto* env =
      litert_wrapper_utils::GetEnvironmentFromCapsule(environment_capsule);
  if (env == nullptr) {
    if (out_error) *out_error = "Invalid LiteRT Environment capsule";
    return nullptr;
  }

  // Extract buffer from Python object
  char* buf = nullptr;
  Py_ssize_t length = 0;
  if (ConvertFromPyString(model_data, &buf, &length) == -1) {
    if (out_error) *out_error = "Failed converting PyObject to buffer";
    return nullptr;
  }

  // Create model from buffer
  BufferRef<uint8_t> ref(reinterpret_cast<uint8_t*>(buf),
                         static_cast<size_t>(length));
  auto model_or = ExtendedModel::CreateFromBuffer(ref);
  if (!model_or) {
    if (out_error) *out_error = model_or.Error().Message();
    return nullptr;
  }
  ExtendedModel model = std::move(*model_or);

  // Create a compiled model options
  auto options_or = litert::Options::Create();
  if (!options_or) {
    if (out_error) *out_error = options_or.Error().Message();
    return nullptr;
  }
  auto& options = *options_or;
  options.SetHardwareAccelerators(static_cast<HwAccelerators>(hardware_accel));

  if (gpu_enforce_f32 || gpu_share_constant_tensors || enable_constant_tensor_sharing || enable_infinite_float_capping || enable_benchmark_mode || enable_allow_src_quantized_fc_conv_ops || enable_hint_waiting_for_completion) {
    auto gpu_options_or = options.GetGpuOptions();
    if (!gpu_options_or) {
      if (out_error) *out_error = gpu_options_or.Error().Message();
      return nullptr;
    }
    if (gpu_enforce_f32) {
      gpu_options_or->SetPrecision(GpuOptions::Precision::kFp32);
    }
    if (gpu_share_constant_tensors || enable_constant_tensor_sharing) {
      gpu_options_or->EnableConstantTensorSharing(true);
    }
    if (enable_infinite_float_capping) {
      gpu_options_or->EnableInfiniteFloatCapping(true);
    }
    if (enable_benchmark_mode) {
      gpu_options_or->EnableBenchmarkMode(true);
    }
    if (enable_allow_src_quantized_fc_conv_ops) {
      gpu_options_or->EnableAllowSrcQuantizedFcConvOps(true);
    }
    if (enable_hint_waiting_for_completion) {
      gpu_options_or->HintWaitingForCompletion(true);
    }
  }

  if (cpu_num_threads > 0 || cpu_kernel_mode >= 0 || xnnpack_flags >= 0 ||
      (xnnpack_weight_cache_path != nullptr && *xnnpack_weight_cache_path)) {
    auto cpu_options_or = options.GetCpuOptions();
    if (!cpu_options_or) {
      if (out_error) *out_error = cpu_options_or.Error().Message();
      return nullptr;
    }
    if (cpu_num_threads > 0) {
      cpu_options_or->SetNumThreads(cpu_num_threads);
    }
    if (cpu_kernel_mode >= 0) {
      cpu_options_or->SetKernelMode(
          static_cast<LiteRtCpuKernelMode>(cpu_kernel_mode));
    }
    if (xnnpack_flags >= 0) {
      cpu_options_or->SetXNNPackFlags(static_cast<uint32_t>(xnnpack_flags));
    }
    if (xnnpack_weight_cache_path != nullptr && *xnnpack_weight_cache_path) {
      cpu_options_or->SetXNNPackWeightCachePath(xnnpack_weight_cache_path);
    }
  }

  // Create a compiled model
  auto compiled_or = CompiledModel::Create(*env, model.Get(), options);
  if (!compiled_or) {
    if (out_error) *out_error = compiled_or.Error().Message();
    return nullptr;
  }

  auto* wrapper =
      new CompiledModelWrapper(std::move(model), std::move(*compiled_or));
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
  if (py_dict == nullptr) {
    return nullptr;
  }

  for (size_t i = 0; i < sigs.size(); ++i) {
    const auto& sig = sigs[i];
    PyObject* sig_info = PyDict_New();
    if (sig_info == nullptr) {
      Py_DECREF(py_dict);
      return nullptr;
    }

    // Add input names
    PyObject* py_in = PyList_New(0);
    if (py_in == nullptr) {
      Py_DECREF(sig_info);
      Py_DECREF(py_dict);
      return nullptr;
    }
    for (auto& n : sig.InputNames()) {
      if (!AppendUnicodeToList(py_in, n)) {
        Py_DECREF(py_in);
        Py_DECREF(sig_info);
        Py_DECREF(py_dict);
        return nullptr;
      }
    }

    // Add output names
    PyObject* py_out = PyList_New(0);
    if (py_out == nullptr) {
      Py_DECREF(py_in);
      Py_DECREF(sig_info);
      Py_DECREF(py_dict);
      return nullptr;
    }
    for (auto& n : sig.OutputNames()) {
      if (!AppendUnicodeToList(py_out, n)) {
        Py_DECREF(py_out);
        Py_DECREF(py_in);
        Py_DECREF(sig_info);
        Py_DECREF(py_dict);
        return nullptr;
      }
    }

    if (!SetDictItemStringSteal(sig_info, "inputs", py_in) ||
        !SetDictItemStringSteal(sig_info, "outputs", py_out)) {
      Py_DECREF(sig_info);
      Py_DECREF(py_dict);
      return nullptr;
    }

    // Add signature to root dictionary
    std::string key(sig.Key());
    if (PyDict_SetItemString(py_dict, key.c_str(), sig_info) != 0) {
      Py_DECREF(sig_info);
      Py_DECREF(py_dict);
      return nullptr;
    }
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
  if (result == nullptr) {
    return nullptr;
  }
  // Add signature key
  if (!SetDictItemStringSteal(
          result, "key",
          PyUnicode_FromStringAndSize(sig.Key().data(), sig.Key().size()))) {
    Py_DECREF(result);
    return nullptr;
  }

  // Add input names
  {
    PyObject* py_in = PyList_New(0);
    if (py_in == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    for (auto& nm : sig.InputNames()) {
      if (!AppendUnicodeToList(py_in, nm)) {
        Py_DECREF(py_in);
        Py_DECREF(result);
        return nullptr;
      }
    }
    if (!SetDictItemStringSteal(result, "inputs", py_in)) {
      Py_DECREF(result);
      return nullptr;
    }
  }

  // Add output names
  {
    PyObject* py_out = PyList_New(0);
    if (py_out == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    for (auto& nm : sig.OutputNames()) {
      if (!AppendUnicodeToList(py_out, nm)) {
        Py_DECREF(py_out);
        Py_DECREF(result);
        return nullptr;
      }
    }
    if (!SetDictItemStringSteal(result, "outputs", py_out)) {
      Py_DECREF(result);
      return nullptr;
    }
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
  if (dict == nullptr) {
    return nullptr;
  }

  // Add buffer size
  auto size_or = req.BufferSize();
  if (!size_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(size_or.Error());
  }
  if (!SetDictItemStringSteal(
          dict, "buffer_size",
          PyLong_FromLong(static_cast<int64_t>(*size_or)))) {
    Py_DECREF(dict);
    return nullptr;
  }

  // Add supported types
  auto types_or = req.SupportedTypes();
  if (!types_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(types_or.Error());
  }
  auto types = std::move(*types_or);
  PyObject* py_list = PyList_New(static_cast<Py_ssize_t>(types.size()));
  if (py_list == nullptr) {
    Py_DECREF(dict);
    return nullptr;
  }
  for (size_t i = 0; i < types.size(); i++) {
    PyObject* type = PyLong_FromLong(static_cast<int64_t>(types[i]));
    if (type == nullptr) {
      Py_DECREF(py_list);
      Py_DECREF(dict);
      return nullptr;
    }
    PyList_SetItem(py_list, i, type);  // steal ref
  }
  if (!SetDictItemStringSteal(dict, "supported_types", py_list)) {
    Py_DECREF(dict);
    return nullptr;
  }

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
  if (dict == nullptr) {
    return nullptr;
  }

  // Add buffer size
  auto size_or = req.BufferSize();
  if (!size_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(size_or.Error());
  }
  if (!SetDictItemStringSteal(
          dict, "buffer_size",
          PyLong_FromLong(static_cast<int64_t>(*size_or)))) {
    Py_DECREF(dict);
    return nullptr;
  }

  auto types_or = req.SupportedTypes();
  if (!types_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(types_or.Error());
  }
  auto types = std::move(*types_or);
  PyObject* py_list = PyList_New(static_cast<Py_ssize_t>(types.size()));
  if (py_list == nullptr) {
    Py_DECREF(dict);
    return nullptr;
  }
  for (size_t i = 0; i < types.size(); i++) {
    PyObject* type = PyLong_FromLong(static_cast<int64_t>(types[i]));
    if (type == nullptr) {
      Py_DECREF(py_list);
      Py_DECREF(dict);
      return nullptr;
    }
    PyList_SetItem(py_list, i, type);  // steal ref
  }
  if (!SetDictItemStringSteal(dict, "supported_types", py_list)) {
    Py_DECREF(dict);
    return nullptr;
  }

  return dict;
}

PyObject* CompiledModelWrapper::CreateInputBufferByName(
    const char* signature_key, const char* input_name) {
  auto buffer_or = compiled_model_.CreateInputBuffer(signature_key, input_name);
  if (!buffer_or) {
    return ConvertErrorToPyExc(buffer_or.Error());
  }
  auto buffer = std::move(*buffer_or);

  PyObject* capsule = litert_wrapper_utils::MakeTensorBufferCapsule(buffer);
  return capsule;
}

PyObject* CompiledModelWrapper::CreateOutputBufferByName(
    const char* signature_key, const char* output_name) {
  auto buffer_or =
      compiled_model_.CreateOutputBuffer(signature_key, output_name);
  if (!buffer_or) {
    return ConvertErrorToPyExc(buffer_or.Error());
  }
  auto buffer = std::move(*buffer_or);

  PyObject* capsule = litert_wrapper_utils::MakeTensorBufferCapsule(buffer);
  return capsule;
}

PyObject* CompiledModelWrapper::CreateInputBuffers(int signature_index) {
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
        litert_wrapper_utils::MakeTensorBufferCapsule(buffers[i]);
    PyList_SetItem(py_list, i, capsule);  // steal ref
  }
  return py_list;
}

PyObject* CompiledModelWrapper::CreateOutputBuffers(int signature_index) {
  auto buffers_or =
      compiled_model_.CreateOutputBuffers(static_cast<size_t>(signature_index));
  if (!buffers_or) {
    return ConvertErrorToPyExc(buffers_or.Error());
  }
  auto buffers = std::move(*buffers_or);
  PyObject* py_list = PyList_New(buffers.size());
  for (size_t i = 0; i < buffers.size(); i++) {
    PyObject* capsule =
        litert_wrapper_utils::MakeTensorBufferCapsule(buffers[i]);
    PyList_SetItem(py_list, i, capsule);
  }
  return py_list;
}

PyObject* CompiledModelWrapper::GetInputTensorDetails(
    const char* signature_key) {
  auto signature_index_or = compiled_model_.GetSignatureIndex(signature_key);
  if (!signature_index_or) {
    return ConvertErrorToPyExc(signature_index_or.Error());
  }
  const size_t signature_index = *signature_index_or;
  auto sig_or = compiled_model_.GetSignature(signature_index);
  if (!sig_or) {
    return ConvertErrorToPyExc(sig_or.Error());
  }
  auto sig = std::move(*sig_or);
  auto input_names = sig.InputNames();
  PyObject* result_dict = PyDict_New();
  if (result_dict == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto tensor_or = sig.InputTensor(i);
    if (!tensor_or) {
      Py_DECREF(result_dict);
      return ConvertErrorToPyExc(tensor_or.Error());
    }
    const SimpleTensor& tensor = *tensor_or;
    std::optional<Layout> input_layout;
    if (tensor.TypeId() == kLiteRtRankedTensorType) {
      auto input_layout_or =
          compiled_model_.GetInputTensorLayout(signature_index, i);
      if (!input_layout_or) {
        Py_DECREF(result_dict);
        return ConvertErrorToPyExc(input_layout_or.Error());
      }
      input_layout = std::move(*input_layout_or);
    }
    PyObject* tensor_dict =
        BuildTensorDetailsDict(tensor, input_layout ? &*input_layout : nullptr);
    if (!SetTensorDetailsDictItem(result_dict, input_names[i], tensor_dict)) {
      Py_DECREF(result_dict);
      return nullptr;
    }
  }
  return result_dict;
}

PyObject* CompiledModelWrapper::GetOutputTensorDetails(
    const char* signature_key) {
  auto signature_index_or = compiled_model_.GetSignatureIndex(signature_key);
  if (!signature_index_or) {
    return ConvertErrorToPyExc(signature_index_or.Error());
  }
  const size_t signature_index = *signature_index_or;
  auto sig_or = compiled_model_.GetSignature(signature_index);
  if (!sig_or) {
    return ConvertErrorToPyExc(sig_or.Error());
  }
  auto sig = std::move(*sig_or);
  auto output_names = sig.OutputNames();
  auto output_layouts_or =
      compiled_model_.GetOutputTensorLayouts(signature_index, true);
  if (!output_layouts_or) {
    return ConvertErrorToPyExc(output_layouts_or.Error());
  }
  auto output_layouts = std::move(*output_layouts_or);
  if (output_layouts.size() != output_names.size()) {
    return ReportError(
        "Output tensor metadata does not match runtime output layout count");
  }
  PyObject* result_dict = PyDict_New();
  if (result_dict == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    auto tensor_or = sig.OutputTensor(i);
    if (!tensor_or) {
      Py_DECREF(result_dict);
      return ConvertErrorToPyExc(tensor_or.Error());
    }
    const SimpleTensor& tensor = *tensor_or;
    const Layout* output_layout = tensor.TypeId() == kLiteRtRankedTensorType
                                      ? &output_layouts[i]
                                      : nullptr;
    PyObject* tensor_dict = BuildTensorDetailsDict(tensor, output_layout);
    if (!SetTensorDetailsDictItem(result_dict, output_names[i], tensor_dict)) {
      Py_DECREF(result_dict);
      return nullptr;
    }
  }
  return result_dict;
}

PyObject* CompiledModelWrapper::IsFullyAccelerated() {
  auto is_fully_accelerated_or = compiled_model_.IsFullyAccelerated();
  if (!is_fully_accelerated_or) {
    return ConvertErrorToPyExc(is_fully_accelerated_or.Error());
  }
  return PyBool_FromLong(*is_fully_accelerated_or ? 1 : 0);
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

PyObject* CompiledModelWrapper::ResizeInputTensor(int signature_index,
                                                  int input_index,
                                                  const std::vector<int>& dims,
                                                  bool strict) {
  auto resize_or =
      strict ? compiled_model_.ResizeInputTensor(signature_index, input_index,
                                                 absl::MakeConstSpan(dims))
             : compiled_model_.ResizeInputTensorNonStrict(
                   signature_index, input_index, absl::MakeConstSpan(dims));
  if (!resize_or) {
    return ConvertErrorToPyExc(resize_or.Error());
  }
  Py_RETURN_NONE;
}

}  // namespace litert::compiled_model_wrapper
