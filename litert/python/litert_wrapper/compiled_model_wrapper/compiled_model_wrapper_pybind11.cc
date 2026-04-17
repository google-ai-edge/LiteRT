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

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "litert/python/litert_wrapper/compiled_model_wrapper/compiled_model_wrapper.h"
#include "pybind11/functional.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11

namespace py = pybind11;

using litert::compiled_model_wrapper::CompilationOptions;
using litert::compiled_model_wrapper::CompiledModelWrapper;

namespace {

CompilationOptions BuildCompilationOptions(
    int hardware_accel, int cpu_num_threads, bool gpu_enforce_f32,
    bool gpu_share_constant_tensors, int cpu_kernel_mode, int xnnpack_flags,
    const std::string& xnnpack_weight_cache_path,
    bool enable_constant_tensor_sharing, bool enable_infinite_float_capping,
    bool enable_benchmark_mode, bool enable_allow_src_quantized_fc_conv_ops,
    bool enable_hint_waiting_for_completion, int qualcomm_log_level,
    int qualcomm_htp_performance_mode, int qualcomm_dsp_performance_mode,
    int qualcomm_use_int64_bias_as_int32, int qualcomm_enable_weight_sharing,
    int qualcomm_use_conv_hmx, int qualcomm_use_fold_relu,
    int qualcomm_profiling, bool qualcomm_has_dump_tensor_ids,
    const std::vector<std::int32_t>& qualcomm_dump_tensor_ids,
    const std::string& qualcomm_ir_json_dir,
    const std::string& qualcomm_dlc_dir, int qualcomm_vtcm_size,
    int qualcomm_num_hvx_threads, int qualcomm_optimization_level,
    int qualcomm_graph_priority, int qualcomm_backend,
    const std::string& qualcomm_saver_output_dir,
    int qualcomm_graph_io_tensor_mem_type, int intel_openvino_device_type,
    int intel_openvino_performance_mode,
    const std::map<std::string, std::string>& intel_openvino_configs_map) {
  CompilationOptions options;
  options.hardware_accel = hardware_accel;
  options.cpu_num_threads = cpu_num_threads;
  options.gpu_enforce_f32 = gpu_enforce_f32;
  options.gpu_share_constant_tensors = gpu_share_constant_tensors;
  options.cpu_kernel_mode = cpu_kernel_mode;
  options.xnnpack_flags = xnnpack_flags;
  options.xnnpack_weight_cache_path = xnnpack_weight_cache_path;
  options.enable_constant_tensor_sharing = enable_constant_tensor_sharing;
  options.enable_infinite_float_capping = enable_infinite_float_capping;
  options.enable_benchmark_mode = enable_benchmark_mode;
  options.enable_allow_src_quantized_fc_conv_ops =
      enable_allow_src_quantized_fc_conv_ops;
  options.enable_hint_waiting_for_completion =
      enable_hint_waiting_for_completion;
  options.qualcomm_log_level = qualcomm_log_level;
  options.qualcomm_htp_performance_mode = qualcomm_htp_performance_mode;
  options.qualcomm_dsp_performance_mode = qualcomm_dsp_performance_mode;
  options.qualcomm_use_int64_bias_as_int32 = qualcomm_use_int64_bias_as_int32;
  options.qualcomm_enable_weight_sharing = qualcomm_enable_weight_sharing;
  options.qualcomm_use_conv_hmx = qualcomm_use_conv_hmx;
  options.qualcomm_use_fold_relu = qualcomm_use_fold_relu;
  options.qualcomm_profiling = qualcomm_profiling;
  options.qualcomm_has_dump_tensor_ids = qualcomm_has_dump_tensor_ids;
  options.qualcomm_dump_tensor_ids = qualcomm_dump_tensor_ids;
  options.qualcomm_ir_json_dir = qualcomm_ir_json_dir;
  options.qualcomm_dlc_dir = qualcomm_dlc_dir;
  options.qualcomm_vtcm_size = qualcomm_vtcm_size;
  options.qualcomm_num_hvx_threads = qualcomm_num_hvx_threads;
  options.qualcomm_optimization_level = qualcomm_optimization_level;
  options.qualcomm_graph_priority = qualcomm_graph_priority;
  options.qualcomm_backend = qualcomm_backend;
  options.qualcomm_saver_output_dir = qualcomm_saver_output_dir;
  options.qualcomm_graph_io_tensor_mem_type = qualcomm_graph_io_tensor_mem_type;
  options.intel_openvino_device_type = intel_openvino_device_type;
  options.intel_openvino_performance_mode = intel_openvino_performance_mode;
  options.intel_openvino_configs_map = intel_openvino_configs_map;
  return options;
}

}  // namespace

PYBIND11_MODULE(_pywrap_litert_compiled_model_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_litert_compiled_model_wrapper
    Python bindings for LiteRT CompiledModel.
  )pbdoc";

  // Factory method to create a CompiledModelWrapper from a model file.
  m.def(
      "CreateCompiledModelFromFile",
      [](py::object environment_capsule, const std::string& model_path,
         int hardware_accel, int cpu_num_threads, bool gpu_enforce_f32,
         bool gpu_share_constant_tensors, int cpu_kernel_mode,
         int xnnpack_flags, const std::string& xnnpack_weight_cache_path,
         bool enable_constant_tensor_sharing,
         bool enable_infinite_float_capping, bool enable_benchmark_mode,
         bool enable_allow_src_quantized_fc_conv_ops,
         bool enable_hint_waiting_for_completion, int qualcomm_log_level,
         int qualcomm_htp_performance_mode, int qualcomm_dsp_performance_mode,
         int qualcomm_use_int64_bias_as_int32,
         int qualcomm_enable_weight_sharing, int qualcomm_use_conv_hmx,
         int qualcomm_use_fold_relu, int qualcomm_profiling,
         bool qualcomm_has_dump_tensor_ids,
         const std::vector<std::int32_t>& qualcomm_dump_tensor_ids,
         const std::string& qualcomm_ir_json_dir,
         const std::string& qualcomm_dlc_dir, int qualcomm_vtcm_size,
         int qualcomm_num_hvx_threads, int qualcomm_optimization_level,
         int qualcomm_graph_priority, int qualcomm_backend,
         const std::string& qualcomm_saver_output_dir,
         int qualcomm_graph_io_tensor_mem_type, int intel_openvino_device_type,
         int intel_openvino_performance_mode,
         const std::map<std::string, std::string>& intel_openvino_configs_map) {
        std::string error;
        CompilationOptions compilation_options = BuildCompilationOptions(
            hardware_accel, cpu_num_threads, gpu_enforce_f32,
            gpu_share_constant_tensors, cpu_kernel_mode, xnnpack_flags,
            xnnpack_weight_cache_path, enable_constant_tensor_sharing,
            enable_infinite_float_capping, enable_benchmark_mode,
            enable_allow_src_quantized_fc_conv_ops,
            enable_hint_waiting_for_completion, qualcomm_log_level,
            qualcomm_htp_performance_mode, qualcomm_dsp_performance_mode,
            qualcomm_use_int64_bias_as_int32, qualcomm_enable_weight_sharing,
            qualcomm_use_conv_hmx, qualcomm_use_fold_relu, qualcomm_profiling,
            qualcomm_has_dump_tensor_ids, qualcomm_dump_tensor_ids,
            qualcomm_ir_json_dir, qualcomm_dlc_dir, qualcomm_vtcm_size,
            qualcomm_num_hvx_threads, qualcomm_optimization_level,
            qualcomm_graph_priority, qualcomm_backend,
            qualcomm_saver_output_dir, qualcomm_graph_io_tensor_mem_type,
            intel_openvino_device_type, intel_openvino_performance_mode,
            intel_openvino_configs_map);
        CompiledModelWrapper* wrapper =
            CompiledModelWrapper::CreateWrapperFromFile(
                environment_capsule.ptr(), model_path.c_str(),
                compilation_options, &error);
        if (!wrapper) {
          throw std::runtime_error(error);
        }
        return wrapper;  // Ownership transferred to pybind11
      },
      py::arg("environment_capsule"), py::arg("model_path"),
      py::arg("hardware_accel") = 0, py::arg("cpu_num_threads") = 0,
      py::arg("gpu_enforce_f32") = false,
      py::arg("gpu_share_constant_tensors") = false,
      py::arg("cpu_kernel_mode") = -1, py::arg("xnnpack_flags") = -1,
      py::arg("xnnpack_weight_cache_path") = "",
      py::arg("enable_constant_tensor_sharing") = false,
      py::arg("enable_infinite_float_capping") = false,
      py::arg("enable_benchmark_mode") = false,
      py::arg("enable_allow_src_quantized_fc_conv_ops") = false,
      py::arg("enable_hint_waiting_for_completion") = false,
      py::arg("qualcomm_log_level") = -1,
      py::arg("qualcomm_htp_performance_mode") = -1,
      py::arg("qualcomm_dsp_performance_mode") = -1,
      py::arg("qualcomm_use_int64_bias_as_int32") = -1,
      py::arg("qualcomm_enable_weight_sharing") = -1,
      py::arg("qualcomm_use_conv_hmx") = -1,
      py::arg("qualcomm_use_fold_relu") = -1,
      py::arg("qualcomm_profiling") = -1,
      py::arg("qualcomm_has_dump_tensor_ids") = false,
      py::arg("qualcomm_dump_tensor_ids") = std::vector<std::int32_t>(),
      py::arg("qualcomm_ir_json_dir") = "", py::arg("qualcomm_dlc_dir") = "",
      py::arg("qualcomm_vtcm_size") = -1,
      py::arg("qualcomm_num_hvx_threads") = -1,
      py::arg("qualcomm_optimization_level") = -1,
      py::arg("qualcomm_graph_priority") = -1, py::arg("qualcomm_backend") = -1,
      py::arg("qualcomm_saver_output_dir") = "",
      py::arg("qualcomm_graph_io_tensor_mem_type") = -1,
      py::arg("intel_openvino_device_type") = -1,
      py::arg("intel_openvino_performance_mode") = -1,
      py::arg("intel_openvino_configs_map") =
          std::map<std::string, std::string>());

  // Factory method to create a CompiledModelWrapper from a model buffer.
  m.def(
      "CreateCompiledModelFromBuffer",
      [](py::object environment_capsule, py::bytes model_data,
         int hardware_accel, int cpu_num_threads, bool gpu_enforce_f32,
         bool gpu_share_constant_tensors, int cpu_kernel_mode,
         int xnnpack_flags, const std::string& xnnpack_weight_cache_path,
         bool enable_constant_tensor_sharing,
         bool enable_infinite_float_capping, bool enable_benchmark_mode,
         bool enable_allow_src_quantized_fc_conv_ops,
         bool enable_hint_waiting_for_completion, int qualcomm_log_level,
         int qualcomm_htp_performance_mode, int qualcomm_dsp_performance_mode,
         int qualcomm_use_int64_bias_as_int32,
         int qualcomm_enable_weight_sharing, int qualcomm_use_conv_hmx,
         int qualcomm_use_fold_relu, int qualcomm_profiling,
         bool qualcomm_has_dump_tensor_ids,
         const std::vector<std::int32_t>& qualcomm_dump_tensor_ids,
         const std::string& qualcomm_ir_json_dir,
         const std::string& qualcomm_dlc_dir, int qualcomm_vtcm_size,
         int qualcomm_num_hvx_threads, int qualcomm_optimization_level,
         int qualcomm_graph_priority, int qualcomm_backend,
         const std::string& qualcomm_saver_output_dir,
         int qualcomm_graph_io_tensor_mem_type, int intel_openvino_device_type,
         int intel_openvino_performance_mode,
         const std::map<std::string, std::string>& intel_openvino_configs_map) {
        std::string error;
        PyObject* data_obj = model_data.ptr();
        CompilationOptions compilation_options = BuildCompilationOptions(
            hardware_accel, cpu_num_threads, gpu_enforce_f32,
            gpu_share_constant_tensors, cpu_kernel_mode, xnnpack_flags,
            xnnpack_weight_cache_path, enable_constant_tensor_sharing,
            enable_infinite_float_capping, enable_benchmark_mode,
            enable_allow_src_quantized_fc_conv_ops,
            enable_hint_waiting_for_completion, qualcomm_log_level,
            qualcomm_htp_performance_mode, qualcomm_dsp_performance_mode,
            qualcomm_use_int64_bias_as_int32, qualcomm_enable_weight_sharing,
            qualcomm_use_conv_hmx, qualcomm_use_fold_relu, qualcomm_profiling,
            qualcomm_has_dump_tensor_ids, qualcomm_dump_tensor_ids,
            qualcomm_ir_json_dir, qualcomm_dlc_dir, qualcomm_vtcm_size,
            qualcomm_num_hvx_threads, qualcomm_optimization_level,
            qualcomm_graph_priority, qualcomm_backend,
            qualcomm_saver_output_dir, qualcomm_graph_io_tensor_mem_type,
            intel_openvino_device_type, intel_openvino_performance_mode,
            intel_openvino_configs_map);
        CompiledModelWrapper* wrapper =
            CompiledModelWrapper::CreateWrapperFromBuffer(
                environment_capsule.ptr(), data_obj, compilation_options,
                &error);
        if (!wrapper) {
          throw std::runtime_error(error);
        }
        return wrapper;
      },
      py::arg("environment_capsule"), py::arg("model_data"),
      py::arg("hardware_accel") = 0, py::arg("cpu_num_threads") = 0,
      py::arg("gpu_enforce_f32") = false,
      py::arg("gpu_share_constant_tensors") = false,
      py::arg("cpu_kernel_mode") = -1, py::arg("xnnpack_flags") = -1,
      py::arg("xnnpack_weight_cache_path") = "",
      py::arg("enable_constant_tensor_sharing") = false,
      py::arg("enable_infinite_float_capping") = false,
      py::arg("enable_benchmark_mode") = false,
      py::arg("enable_allow_src_quantized_fc_conv_ops") = false,
      py::arg("enable_hint_waiting_for_completion") = false,
      py::arg("qualcomm_log_level") = -1,
      py::arg("qualcomm_htp_performance_mode") = -1,
      py::arg("qualcomm_dsp_performance_mode") = -1,
      py::arg("qualcomm_use_int64_bias_as_int32") = -1,
      py::arg("qualcomm_enable_weight_sharing") = -1,
      py::arg("qualcomm_use_conv_hmx") = -1,
      py::arg("qualcomm_use_fold_relu") = -1,
      py::arg("qualcomm_profiling") = -1,
      py::arg("qualcomm_has_dump_tensor_ids") = false,
      py::arg("qualcomm_dump_tensor_ids") = std::vector<std::int32_t>(),
      py::arg("qualcomm_ir_json_dir") = "", py::arg("qualcomm_dlc_dir") = "",
      py::arg("qualcomm_vtcm_size") = -1,
      py::arg("qualcomm_num_hvx_threads") = -1,
      py::arg("qualcomm_optimization_level") = -1,
      py::arg("qualcomm_graph_priority") = -1, py::arg("qualcomm_backend") = -1,
      py::arg("qualcomm_saver_output_dir") = "",
      py::arg("qualcomm_graph_io_tensor_mem_type") = -1,
      py::arg("intel_openvino_device_type") = -1,
      py::arg("intel_openvino_performance_mode") = -1,
      py::arg("intel_openvino_configs_map") =
          std::map<std::string, std::string>());

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
      .def("IsFullyAccelerated",
           [](CompiledModelWrapper& self) {
             PyObject* r = self.IsFullyAccelerated();
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateInputBufferByName",
           [](CompiledModelWrapper& self, const std::string& sig_key,
              const std::string& input_name) {
             PyObject* r = self.CreateInputBufferByName(sig_key.c_str(),
                                                        input_name.c_str());
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateOutputBufferByName",
           [](CompiledModelWrapper& self, const std::string& sig_key,
              const std::string& out_name) {
             PyObject* r = self.CreateOutputBufferByName(sig_key.c_str(),
                                                         out_name.c_str());
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateInputBuffers",
           [](CompiledModelWrapper& self, int sig_index) {
             PyObject* r = self.CreateInputBuffers(sig_index);
             if (!r) {
               throw py::error_already_set();
             }
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateOutputBuffers",
           [](CompiledModelWrapper& self, int sig_index) {
             PyObject* r = self.CreateOutputBuffers(sig_index);
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
      .def("GetOutputTensorDetails",
           [](CompiledModelWrapper& self, const std::string& sig_key) {
             PyObject* r = self.GetOutputTensorDetails(sig_key.c_str());
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
      .def("Run",
           [](CompiledModelWrapper& self, py::object in_list,
              py::object out_list) {
             PyObject* r = self.RunByIndex(0, in_list.ptr(), out_list.ptr());
             if (!r) {
               throw py::error_already_set();
             }
             return py::none();
           })
      .def("ResizeInputTensor",
           [](CompiledModelWrapper& self, int sig_idx, int input_idx,
              const std::vector<int>& dims) {
             PyObject* r =
                 self.ResizeInputTensor(sig_idx, input_idx, dims, true);
             if (!r) {
               throw py::error_already_set();
             }
             return py::none();
           })
      .def("ResizeInputTensorNonStrict", [](CompiledModelWrapper& self,
                                            int sig_idx, int input_idx,
                                            const std::vector<int>& dims) {
        PyObject* r = self.ResizeInputTensor(sig_idx, input_idx, dims, false);
        if (!r) {
          throw py::error_already_set();
        }
        return py::none();
      });
}
