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

#ifndef LITERT_PYTHON_LITERT_WRAPPER_COMPILED_MODEL_WRAPPER_COMPILED_MODEL_WRAPPER_H_
#define LITERT_PYTHON_LITERT_WRAPPER_COMPILED_MODEL_WRAPPER_COMPILED_MODEL_WRAPPER_H_

#include <Python.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_compiled_model.h"

namespace litert {
namespace compiled_model_wrapper {

struct CompilationOptions {
  int hardware_accel = 0;
  int cpu_num_threads = 0;
  bool gpu_enforce_f32 = false;
  bool gpu_share_constant_tensors = false;
  int cpu_kernel_mode = -1;
  int xnnpack_flags = -1;
  std::string xnnpack_weight_cache_path;
  bool enable_constant_tensor_sharing = false;
  bool enable_infinite_float_capping = false;
  bool enable_benchmark_mode = false;
  bool enable_allow_src_quantized_fc_conv_ops = false;
  bool enable_hint_waiting_for_completion = false;

  int qualcomm_log_level = -1;
  int qualcomm_htp_performance_mode = -1;
  int qualcomm_dsp_performance_mode = -1;
  int qualcomm_use_int64_bias_as_int32 = -1;
  int qualcomm_enable_weight_sharing = -1;
  int qualcomm_use_conv_hmx = -1;
  int qualcomm_use_fold_relu = -1;
  int qualcomm_profiling = -1;
  bool qualcomm_has_dump_tensor_ids = false;
  std::vector<std::int32_t> qualcomm_dump_tensor_ids;
  std::string qualcomm_ir_json_dir;
  std::string qualcomm_dlc_dir;
  int qualcomm_vtcm_size = -1;
  int qualcomm_num_hvx_threads = -1;
  int qualcomm_optimization_level = -1;
  int qualcomm_graph_priority = -1;
  int qualcomm_backend = -1;
  std::string qualcomm_saver_output_dir;
  int qualcomm_graph_io_tensor_mem_type = -1;

  int intel_openvino_device_type = -1;
  int intel_openvino_performance_mode = -1;
  std::map<std::string, std::string> intel_openvino_configs_map;
};

/**
 * Wrapper class for LiteRT models that provides Python bindings.
 *
 * This class manages the lifecycle of LiteRT Environment, Model, and
 * CompiledModel objects while exposing their functionality to Python.
 * It handles Python object conversions and error reporting.
 */
class CompiledModelWrapper {
 public:
  /**
   * Creates a wrapper from a model file path.
   *
   * @param model_path Path to the model file
   * @param environment_capsule PyCapsule containing a LiteRT Environment
   * @param hardware_accel Hardware acceleration option (LiteRtHwAccelerators).
   *        These are bit flags that can be combined with bitwise OR:
   *        1 (kCpu)  - CPU acceleration (always works)
   *        2 (kGpu)  - GPU acceleration (WebGPU/OpenCL/Metal)
   *        Use kCpu | kGpu (3) for GPU with CPU fallback.
   *        Note: 0 (kNone) will fail; at least one accelerator must be set.
   * @param cpu_num_threads Number of threads for CPU execution.
   * @param gpu_enforce_f32 Enforce F32 precision on GPU.
   * @param gpu_share_constant_tensors Share constant tensors among subgraphs on
   *        GPU.
   * @param out_error String to store error message if creation fails
   * @return A new CompiledModelWrapper instance, or nullptr on failure
   */
  static CompiledModelWrapper* CreateWrapperFromFile(
      PyObject* environment_capsule, const char* model_path,
      const CompilationOptions& compilation_options, std::string* out_error);

  /**
   * Creates a wrapper from a model buffer in memory.
   *
   * @param environment_capsule PyCapsule containing a LiteRT Environment
   * @param model_data Python bytes object containing the model data
   *        (created from reading a model file or receiving serialized model
   * data)
   * @param hardware_accel Hardware acceleration option (LiteRtHwAccelerators).
   *        These are bit flags that can be combined with bitwise OR:
   *        1 (kCpu)  - CPU acceleration (always works)
   *        2 (kGpu)  - GPU acceleration (WebGPU/OpenCL/Metal)
   *        Use kCpu | kGpu (3) for GPU with CPU fallback.
   *        Note: 0 (kNone) will fail; at least one accelerator must be set.
   * @param cpu_num_threads Number of threads for CPU execution.
   * @param gpu_enforce_f32 Enforce F32 precision on GPU.
   * @param gpu_share_constant_tensors Share constant tensors among subgraphs on
   *        GPU.
   * @param out_error String to store error message if creation fails
   * @return A new CompiledModelWrapper instance, or nullptr on failure
   */
  static CompiledModelWrapper* CreateWrapperFromBuffer(
      PyObject* environment_capsule, PyObject* model_data,
      const CompilationOptions& compilation_options, std::string* out_error);

  CompiledModelWrapper(litert::ExtendedModel model,
                       litert::CompiledModel compiled);

  ~CompiledModelWrapper();

  // Disable copy semantics to prevent double-free of Python buffer reference
  CompiledModelWrapper(const CompiledModelWrapper&) = delete;
  CompiledModelWrapper& operator=(const CompiledModelWrapper&) = delete;
  // Disable move semantics
  CompiledModelWrapper(CompiledModelWrapper&&) = delete;
  CompiledModelWrapper& operator=(CompiledModelWrapper& s) = delete;

  // Returns a Python object containing the model's signatures.
  PyObject* GetSignatureList();

  // Returns a Python object with details about the signature at the given
  // index.
  PyObject* GetSignatureByIndex(int signature_index);

  // Returns the number of signatures in the model.
  PyObject* GetNumSignatures();

  // Returns the index of a signature by key, or -1 if not found.
  PyObject* GetSignatureIndex(const char* signature_key);

  // Returns buffer requirements for an input tensor.
  PyObject* GetInputBufferRequirements(int signature_index, int input_index);

  // Returns buffer requirements for an output tensor.
  PyObject* GetOutputBufferRequirements(int signature_index, int output_index);

  // Creates an input buffer for a tensor identified by signature key and input
  // name.
  PyObject* CreateInputBufferByName(const char* signature_key,
                                    const char* input_name);

  // Creates an output buffer for a tensor identified by signature key and
  // output name.
  PyObject* CreateOutputBufferByName(const char* signature_key,
                                     const char* output_name);

  // Creates all input buffers for a signature and returns them as a list of
  // capsules.
  PyObject* CreateInputBuffers(int signature_index);

  // Creates all output buffers for a signature and returns them as a list of
  // capsules.
  PyObject* CreateOutputBuffers(int signature_index);

  // Executes the model using a signature key and name-to-buffer mappings.
  PyObject* RunByName(const char* signature_key, PyObject* input_map,
                      PyObject* output_map);

  // Executes the model using a signature index and lists of buffer capsules.
  PyObject* RunByIndex(int signature_index, PyObject* input_caps_list,
                       PyObject* output_caps_list);

  // Returns input tensor details for a given signature.
  PyObject* GetInputTensorDetails(const char* signature_key);

  // Returns output tensor details for a given signature.
  PyObject* GetOutputTensorDetails(const char* signature_key);

  // Returns whether the model is fully accelerated with selected accelerators.
  PyObject* IsFullyAccelerated();

  // Resizes an input tensor by signature and input index.
  PyObject* ResizeInputTensor(int signature_index, int input_index,
                              const std::vector<int>& dims, bool strict);

 private:
  // Returns the size in bytes of a single element of the given data type.
  static size_t ByteWidthOfDType(const std::string& dtype);

  // Reports an error to Python and returns nullptr.
  static PyObject* ReportError(const std::string& msg);

  // Converts a LiteRT error to a Python exception and returns nullptr.
  static PyObject* ConvertErrorToPyExc(const litert::Error& error);

  // Member variables holding the LiteRT C++ objects.
  ExtendedModel model_;
  litert::CompiledModel compiled_model_;

  // Python buffer object to keep it alive for models created from buffer
  PyObject* model_buffer_ = nullptr;
};

}  // namespace compiled_model_wrapper
}  // namespace litert

#endif  // LITERT_PYTHON_LITERT_WRAPPER_COMPILED_MODEL_WRAPPER_COMPILED_MODEL_WRAPPER_H_
