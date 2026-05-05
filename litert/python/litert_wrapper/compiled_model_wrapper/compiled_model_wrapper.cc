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
#include "litert/c/options/litert_cpu_options.h"
#include "litert/c/options/litert_intel_openvino_options.h"
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
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
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

bool TriStateBoolIsSet(int value) { return value >= 0; }

bool TriStateBoolValue(int value) { return value != 0; }

bool HasQualcommOptions(const CompilationOptions& compilation_options) {
  return compilation_options.qualcomm_log_level >= 0 ||
         compilation_options.qualcomm_htp_performance_mode >= 0 ||
         compilation_options.qualcomm_dsp_performance_mode >= 0 ||
         TriStateBoolIsSet(
             compilation_options.qualcomm_use_int64_bias_as_int32) ||
         TriStateBoolIsSet(
             compilation_options.qualcomm_enable_weight_sharing) ||
         TriStateBoolIsSet(compilation_options.qualcomm_use_conv_hmx) ||
         TriStateBoolIsSet(compilation_options.qualcomm_use_fold_relu) ||
         compilation_options.qualcomm_profiling >= 0 ||
         compilation_options.qualcomm_has_dump_tensor_ids ||
         !compilation_options.qualcomm_ir_json_dir.empty() ||
         !compilation_options.qualcomm_dlc_dir.empty() ||
         compilation_options.qualcomm_vtcm_size >= 0 ||
         compilation_options.qualcomm_num_hvx_threads >= 0 ||
         compilation_options.qualcomm_optimization_level >= 0 ||
         compilation_options.qualcomm_graph_priority >= 0 ||
         compilation_options.qualcomm_backend >= 0 ||
         !compilation_options.qualcomm_saver_output_dir.empty() ||
         compilation_options.qualcomm_graph_io_tensor_mem_type >= 0;
}

bool HasIntelOpenVinoOptions(const CompilationOptions& compilation_options) {
  return compilation_options.intel_openvino_device_type >= 0 ||
         compilation_options.intel_openvino_performance_mode >= 0 ||
         !compilation_options.intel_openvino_configs_map.empty();
}

bool PopulateCompilationOptions(litert::Options& options,
                                const CompilationOptions& compilation_options,
                                std::string* out_error) {
  auto accelerator_status = options.SetHardwareAccelerators(
      static_cast<HwAccelerators>(compilation_options.hardware_accel));
  if (!accelerator_status) {
    if (out_error) *out_error = accelerator_status.Error().Message();
    return false;
  }

  if (compilation_options.gpu_enforce_f32 ||
      compilation_options.gpu_share_constant_tensors ||
      compilation_options.enable_constant_tensor_sharing ||
      compilation_options.enable_infinite_float_capping ||
      compilation_options.enable_benchmark_mode ||
      compilation_options.enable_allow_src_quantized_fc_conv_ops ||
      compilation_options.enable_hint_waiting_for_completion) {
    auto gpu_options_or = options.GetGpuOptions();
    if (!gpu_options_or) {
      if (out_error) *out_error = gpu_options_or.Error().Message();
      return false;
    }
    if (compilation_options.gpu_enforce_f32) {
      gpu_options_or->SetPrecision(GpuOptions::Precision::kFp32);
    }
    if (compilation_options.gpu_share_constant_tensors ||
        compilation_options.enable_constant_tensor_sharing) {
      gpu_options_or->EnableConstantTensorSharing(true);
    }
    if (compilation_options.enable_infinite_float_capping) {
      gpu_options_or->EnableInfiniteFloatCapping(true);
    }
    if (compilation_options.enable_benchmark_mode) {
      gpu_options_or->EnableBenchmarkMode(true);
    }
    if (compilation_options.enable_allow_src_quantized_fc_conv_ops) {
      gpu_options_or->EnableAllowSrcQuantizedFcConvOps(true);
    }
    if (compilation_options.enable_hint_waiting_for_completion) {
      gpu_options_or->HintWaitingForCompletion(true);
    }
  }

  if (compilation_options.cpu_num_threads > 0 ||
      compilation_options.cpu_kernel_mode >= 0 ||
      compilation_options.xnnpack_flags >= 0 ||
      !compilation_options.xnnpack_weight_cache_path.empty()) {
    auto cpu_options_or = options.GetCpuOptions();
    if (!cpu_options_or) {
      if (out_error) *out_error = cpu_options_or.Error().Message();
      return false;
    }
    if (compilation_options.cpu_num_threads > 0) {
      cpu_options_or->SetNumThreads(compilation_options.cpu_num_threads);
    }
    if (compilation_options.cpu_kernel_mode >= 0) {
      cpu_options_or->SetKernelMode(static_cast<LiteRtCpuKernelMode>(
          compilation_options.cpu_kernel_mode));
    }
    if (compilation_options.xnnpack_flags >= 0) {
      cpu_options_or->SetXNNPackFlags(
          static_cast<uint32_t>(compilation_options.xnnpack_flags));
    }
    if (!compilation_options.xnnpack_weight_cache_path.empty()) {
      cpu_options_or->SetXNNPackWeightCachePath(
          compilation_options.xnnpack_weight_cache_path.c_str());
    }
  }

  if (HasQualcommOptions(compilation_options)) {
    auto qualcomm_options_or = options.GetQualcommOptions();
    if (!qualcomm_options_or) {
      if (out_error) *out_error = qualcomm_options_or.Error().Message();
      return false;
    }
    auto& qualcomm_options = *qualcomm_options_or;
    if (compilation_options.qualcomm_log_level >= 0) {
      qualcomm_options.SetLogLevel(
          static_cast<qualcomm::QualcommOptions::LogLevel>(
              compilation_options.qualcomm_log_level));
    }
    if (compilation_options.qualcomm_htp_performance_mode >= 0) {
      qualcomm_options.SetHtpPerformanceMode(
          static_cast<qualcomm::QualcommOptions::HtpPerformanceMode>(
              compilation_options.qualcomm_htp_performance_mode));
    }
    if (compilation_options.qualcomm_dsp_performance_mode >= 0) {
      qualcomm_options.SetDspPerformanceMode(
          static_cast<qualcomm::QualcommOptions::DspPerformanceMode>(
              compilation_options.qualcomm_dsp_performance_mode));
    }
    if (TriStateBoolIsSet(
            compilation_options.qualcomm_use_int64_bias_as_int32)) {
      qualcomm_options.SetUseInt64BiasAsInt32(TriStateBoolValue(
          compilation_options.qualcomm_use_int64_bias_as_int32));
    }
    if (TriStateBoolIsSet(compilation_options.qualcomm_enable_weight_sharing)) {
      qualcomm_options.SetEnableWeightSharing(TriStateBoolValue(
          compilation_options.qualcomm_enable_weight_sharing));
    }
    if (TriStateBoolIsSet(compilation_options.qualcomm_use_conv_hmx)) {
      qualcomm_options.SetUseConvHMX(
          TriStateBoolValue(compilation_options.qualcomm_use_conv_hmx));
    }
    if (TriStateBoolIsSet(compilation_options.qualcomm_use_fold_relu)) {
      qualcomm_options.SetUseFoldReLU(
          TriStateBoolValue(compilation_options.qualcomm_use_fold_relu));
    }
    if (compilation_options.qualcomm_profiling >= 0) {
      qualcomm_options.SetProfiling(
          static_cast<qualcomm::QualcommOptions::Profiling>(
              compilation_options.qualcomm_profiling));
    }
    if (compilation_options.qualcomm_has_dump_tensor_ids) {
      qualcomm_options.SetDumpTensorIds(
          compilation_options.qualcomm_dump_tensor_ids);
    }
    if (!compilation_options.qualcomm_ir_json_dir.empty()) {
      qualcomm_options.SetIrJsonDir(compilation_options.qualcomm_ir_json_dir);
    }
    if (!compilation_options.qualcomm_dlc_dir.empty()) {
      qualcomm_options.SetDlcDir(compilation_options.qualcomm_dlc_dir);
    }
    if (compilation_options.qualcomm_vtcm_size >= 0) {
      qualcomm_options.SetVtcmSize(
          static_cast<uint32_t>(compilation_options.qualcomm_vtcm_size));
    }
    if (compilation_options.qualcomm_num_hvx_threads >= 0) {
      qualcomm_options.SetNumHvxThreads(
          static_cast<uint32_t>(compilation_options.qualcomm_num_hvx_threads));
    }
    if (compilation_options.qualcomm_optimization_level >= 0) {
      qualcomm_options.SetOptimizationLevel(
          static_cast<qualcomm::QualcommOptions::OptimizationLevel>(
              compilation_options.qualcomm_optimization_level));
    }
    if (compilation_options.qualcomm_graph_priority >= 0) {
      qualcomm_options.SetGraphPriority(
          static_cast<qualcomm::QualcommOptions::GraphPriority>(
              compilation_options.qualcomm_graph_priority));
    }
    if (compilation_options.qualcomm_backend >= 0) {
      qualcomm_options.SetBackend(
          static_cast<qualcomm::QualcommOptions::Backend>(
              compilation_options.qualcomm_backend));
    }
    if (!compilation_options.qualcomm_saver_output_dir.empty()) {
      qualcomm_options.SetSaverOutputDir(
          compilation_options.qualcomm_saver_output_dir);
    }
    if (compilation_options.qualcomm_graph_io_tensor_mem_type >= 0) {
      qualcomm_options.SetGraphIOTensorMemType(
          static_cast<qualcomm::QualcommOptions::GraphIOTensorMemType>(
              compilation_options.qualcomm_graph_io_tensor_mem_type));
    }
  }

  if (HasIntelOpenVinoOptions(compilation_options)) {
    auto intel_openvino_options_or = options.GetIntelOpenVinoOptions();
    if (!intel_openvino_options_or) {
      if (out_error) *out_error = intel_openvino_options_or.Error().Message();
      return false;
    }
    auto& intel_openvino_options = *intel_openvino_options_or;
    if (compilation_options.intel_openvino_device_type >= 0) {
      intel_openvino_options.SetDeviceType(
          static_cast<LiteRtIntelOpenVinoDeviceType>(
              compilation_options.intel_openvino_device_type));
    }
    if (compilation_options.intel_openvino_performance_mode >= 0) {
      intel_openvino_options.SetPerformanceMode(
          static_cast<LiteRtIntelOpenVinoPerformanceMode>(
              compilation_options.intel_openvino_performance_mode));
    }
    for (const auto& [key, value] :
         compilation_options.intel_openvino_configs_map) {
      intel_openvino_options.SetConfigsMapOption(key.c_str(), value.c_str());
    }
  }

  return true;
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
    PyObject* environment_capsule, const char* model_path,
    const CompilationOptions& compilation_options, std::string* out_error) {
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
  if (!PopulateCompilationOptions(options, compilation_options, out_error)) {
    return nullptr;
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
    PyObject* environment_capsule, PyObject* model_data,
    const CompilationOptions& compilation_options, std::string* out_error) {
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
  if (!PopulateCompilationOptions(options, compilation_options, out_error)) {
    return nullptr;
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
