// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_RUNTIME_PROXY_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_RUNTIME_PROXY_H_

#include <cstddef>
#include <cstdint>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/die_if_null.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_c_api.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_opencl_types.h"
#include "litert/c/litert_profiler_event.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/c/litert_webgpu_types.h"

namespace litert {
namespace internal {

#define LITERT_PROXY_METHOD_STATUS(method, ...) \
  ABSL_CHECK(runtime_c_api_->method);           \
  return runtime_c_api_->method(__VA_ARGS__);

#define LITERT_PROXY_METHOD_VOID(method, ...) \
  ABSL_CHECK(runtime_c_api_->method);         \
  runtime_c_api_->method(__VA_ARGS__);

// A proxy class that provides a C++ interface to the LiteRT Runtime C Api.
//
// This class is used to implement the LiteRT C++ API, which could talk to
// different runtime implementations (e.g. real runtime, mock runtime).
class RuntimeProxy {
 public:
  explicit RuntimeProxy(const LiteRtRuntimeCApiStruct* runtime_c_api)
      : runtime_c_api_(ABSL_DIE_IF_NULL(runtime_c_api)) {};

  ~RuntimeProxy() = default;

  //
  // LiteRtEnvironment
  //

  LiteRtStatus CreateEnvironment(int num_options,
                                 const LiteRtEnvOption* options,
                                 LiteRtEnvironment* environment) {
    LITERT_PROXY_METHOD_STATUS(litert_create_environment, num_options, options,
                               environment);
  }

  void DestroyEnvironment(LiteRtEnvironment environment) {
    LITERT_PROXY_METHOD_VOID(litert_destroy_environment, environment);
  }

  LiteRtStatus GetEnvironmentOptions(LiteRtEnvironment environment,
                                     LiteRtEnvironmentOptions* options) {
    LITERT_PROXY_METHOD_STATUS(litert_get_environment_options, environment,
                               options);
  }

  LiteRtStatus AddEnvironmentOptions(LiteRtEnvironment environment,
                                     int num_options,
                                     const LiteRtEnvOption* options,
                                     bool overwrite) {
    LITERT_PROXY_METHOD_STATUS(litert_add_environment_options, environment,
                               num_options, options, overwrite);
  }

  LiteRtStatus GpuEnvironmentCreate(LiteRtEnvironment environment,
                                    int num_options,
                                    const LiteRtEnvOption* options) {
    LITERT_PROXY_METHOD_STATUS(litert_gpu_environment_create, environment,
                               num_options, options);
  }

  LiteRtStatus EnvironmentSupportsClGlInterop(LiteRtEnvironment environment,
                                              bool* is_supported) {
    LITERT_PROXY_METHOD_STATUS(litert_environment_supports_cl_gl_interop,
                               environment, is_supported);
  }

  LiteRtStatus EnvironmentSupportsAhwbClInterop(LiteRtEnvironment environment,
                                                bool* is_supported) {
    LITERT_PROXY_METHOD_STATUS(litert_environment_supports_ahwb_cl_interop,
                               environment, is_supported);
  }

  LiteRtStatus EnvironmentSupportsAhwbGlInterop(LiteRtEnvironment environment,
                                                bool* is_supported) {
    LITERT_PROXY_METHOD_STATUS(litert_environment_supports_ahwb_gl_interop,
                               environment, is_supported);
  }

  void EnvironmentHasGpuEnvironment(LiteRtEnvironment environment,
                                    bool* has_gpu_environment) {
    LITERT_PROXY_METHOD_VOID(litert_environment_has_gpu_environment,
                             environment, has_gpu_environment);
  }

  //
  // LiteRtEnvironmentOptions
  //

  LiteRtStatus GetEnvironmentOptionsValue(LiteRtEnvironmentOptions options,
                                          LiteRtEnvOptionTag tag,
                                          LiteRtAny* value) {
    LITERT_PROXY_METHOD_STATUS(litert_get_environment_options_value, options,
                               tag, value);
  }

  LiteRtStatus SetEnvironmentOptionsValue(LiteRtEnvironmentOptions options,
                                          LiteRtEnvOption env_option) {
    LITERT_PROXY_METHOD_STATUS(litert_set_environment_options_value, options,
                               env_option);
  }

  //
  // LiteRtTensor
  //

  LiteRtStatus GetTensorName(LiteRtTensor tensor, const char** name) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_name, tensor, name);
  }

  LiteRtStatus GetTensorIndex(LiteRtTensor tensor, uint32_t* tensor_index) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_index, tensor, tensor_index);
  }

  LiteRtStatus GetTensorTypeId(LiteRtTensor tensor,
                               LiteRtTensorTypeId* type_id) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_type_id, tensor, type_id);
  }

  LiteRtStatus GetUnrankedTensorType(
      LiteRtTensor tensor, LiteRtUnrankedTensorType* unranked_tensor_type) {
    LITERT_PROXY_METHOD_STATUS(litert_get_unranked_tensor_type, tensor,
                               unranked_tensor_type);
  }

  LiteRtStatus GetRankedTensorType(LiteRtTensor tensor,
                                   LiteRtRankedTensorType* ranked_tensor_type) {
    LITERT_PROXY_METHOD_STATUS(litert_get_ranked_tensor_type, tensor,
                               ranked_tensor_type);
  }

  LiteRtStatus GetQuantizationTypeId(LiteRtTensor tensor,
                                     LiteRtQuantizationTypeId* q_type_id) {
    LITERT_PROXY_METHOD_STATUS(litert_get_quantization_type_id, tensor,
                               q_type_id);
  }

  LiteRtStatus GetPerTensorQuantization(
      LiteRtTensor tensor,
      LiteRtQuantizationPerTensor* per_tensor_quantization) {
    LITERT_PROXY_METHOD_STATUS(litert_get_per_tensor_quantization, tensor,
                               per_tensor_quantization);
  }

  LiteRtStatus GetPerChannelQuantization(
      LiteRtTensor tensor,
      LiteRtQuantizationPerChannel* per_channel_quantization) {
    LITERT_PROXY_METHOD_STATUS(litert_get_per_channel_quantization, tensor,
                               per_channel_quantization);
  }

  LiteRtStatus GetNumTensorUses(LiteRtTensor tensor,
                                LiteRtParamIndex* num_uses) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_tensor_uses, tensor, num_uses);
  }

  LiteRtStatus GetTensorUse(LiteRtTensor tensor, LiteRtParamIndex use_index,
                            LiteRtOp* user, LiteRtParamIndex* user_arg_index) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_use, tensor, use_index, user,
                               user_arg_index);
  }

  LiteRtStatus GetTensorDefiningOp(LiteRtTensor tensor, bool* has_defining_op,
                                   LiteRtTensorDefiningOp* defining_op) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_defining_op, tensor,
                               has_defining_op, defining_op);
  }

  LiteRtStatus GetTensorWeights(LiteRtTensor tensor, LiteRtWeights* weights) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_weights, tensor, weights);
  }

  //
  // LiteRtSubgraph
  //

  LiteRtStatus GetNumSubgraphInputs(LiteRtSubgraph subgraph,
                                    LiteRtParamIndex* num_inputs) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_subgraph_inputs, subgraph,
                               num_inputs);
  }

  LiteRtStatus GetSubgraphInput(LiteRtSubgraph subgraph,
                                LiteRtParamIndex input_index,
                                LiteRtTensor* input) {
    LITERT_PROXY_METHOD_STATUS(litert_get_subgraph_input, subgraph, input_index,
                               input);
  }

  LiteRtStatus GetNumSubgraphOutputs(LiteRtSubgraph subgraph,
                                     LiteRtParamIndex* num_outputs) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_subgraph_outputs, subgraph,
                               num_outputs);
  }

  LiteRtStatus GetSubgraphOutput(LiteRtSubgraph subgraph,
                                 LiteRtParamIndex output_index,
                                 LiteRtTensor* output) {
    LITERT_PROXY_METHOD_STATUS(litert_get_subgraph_output, subgraph,
                               output_index, output);
  }

  LiteRtStatus GetNumSubgraphOps(LiteRtSubgraph subgraph,
                                 LiteRtParamIndex* num_ops) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_subgraph_ops, subgraph, num_ops);
  }

  LiteRtStatus GetSubgraphOp(LiteRtSubgraph subgraph, LiteRtParamIndex op_index,
                             LiteRtOp* op) {
    LITERT_PROXY_METHOD_STATUS(litert_get_subgraph_op, subgraph, op_index, op);
  }

  //
  // LiteRtSignature
  //
  LiteRtStatus GetDefaultSignatureKey(const char** signature_key) {
    LITERT_PROXY_METHOD_STATUS(litert_get_default_signature_key, signature_key);
  }

  LiteRtStatus GetSignatureKey(LiteRtSignature signature,
                               const char** signature_key) {
    LITERT_PROXY_METHOD_STATUS(litert_get_signature_key, signature,
                               signature_key);
  }

  LiteRtStatus GetSignatureSubgraph(LiteRtSignature signature,
                                    LiteRtSubgraph* subgraph) {
    LITERT_PROXY_METHOD_STATUS(litert_get_signature_subgraph, signature,
                               subgraph);
  }

  LiteRtStatus GetNumSignatureInputs(LiteRtSignature signature,
                                     LiteRtParamIndex* num_inputs) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_signature_inputs, signature,
                               num_inputs);
  }

  LiteRtStatus GetSignatureInputName(LiteRtSignature signature,
                                     LiteRtParamIndex input_idx,
                                     const char** input_name) {
    LITERT_PROXY_METHOD_STATUS(litert_get_signature_input_name, signature,
                               input_idx, input_name);
  }

  LiteRtStatus GetSignatureInputTensor(LiteRtSignature signature,
                                       const char* input_name,
                                       LiteRtTensor* tensor) {
    LITERT_PROXY_METHOD_STATUS(litert_get_signature_input_tensor, signature,
                               input_name, tensor);
  }

  LiteRtStatus GetSignatureInputTensorByIndex(LiteRtSignature signature,
                                              LiteRtParamIndex input_idx,
                                              LiteRtTensor* tensor) {
    LITERT_PROXY_METHOD_STATUS(litert_get_signature_input_tensor_by_index,
                               signature, input_idx, tensor);
  }

  LiteRtStatus GetNumSignatureOutputs(LiteRtSignature signature,
                                      LiteRtParamIndex* num_outputs) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_signature_outputs, signature,
                               num_outputs);
  }

  LiteRtStatus GetSignatureOutputName(LiteRtSignature signature,
                                      LiteRtParamIndex output_idx,
                                      const char** output_name) {
    LITERT_PROXY_METHOD_STATUS(litert_get_signature_output_name, signature,
                               output_idx, output_name);
  }

  LiteRtStatus GetSignatureOutputTensor(LiteRtSignature signature,
                                        const char* output_name,
                                        LiteRtTensor* tensor) {
    LITERT_PROXY_METHOD_STATUS(litert_get_signature_output_tensor, signature,
                               output_name, tensor);
  }

  LiteRtStatus GetSignatureOutputTensorByIndex(LiteRtSignature signature,
                                               LiteRtParamIndex output_idx,
                                               LiteRtTensor* tensor) {
    LITERT_PROXY_METHOD_STATUS(litert_get_signature_output_tensor_by_index,
                               signature, output_idx, tensor);
  }

  //
  // LiteRtModel
  //

  LiteRtStatus CreateModelFromFile(const char* filename, LiteRtModel* model) {
    LITERT_PROXY_METHOD_STATUS(litert_create_model_from_file, filename, model);
  }

  LiteRtStatus CreateModelFromBuffer(const void* buffer_addr,
                                     size_t buffer_size, LiteRtModel* model) {
    LITERT_PROXY_METHOD_STATUS(litert_create_model_from_buffer, buffer_addr,
                               buffer_size, model);
  }

  LiteRtStatus GetModelMetadata(LiteRtModel model, const char* metadata_key,
                                const void** metadata_buffer,
                                size_t* metadata_buffer_size) {
    LITERT_PROXY_METHOD_STATUS(litert_get_model_metadata, model, metadata_key,
                               metadata_buffer, metadata_buffer_size);
  }

  LiteRtStatus AddModelMetadata(LiteRtModel model, const char* metadata_key,
                                const void* metadata_buffer,
                                size_t metadata_buffer_size) {
    LITERT_PROXY_METHOD_STATUS(litert_add_model_metadata, model, metadata_key,
                               metadata_buffer, metadata_buffer_size);
  }

  LiteRtStatus GetMainModelSubgraphIndex(
      LiteRtModel model, LiteRtParamIndex* main_subgraph_index) {
    LITERT_PROXY_METHOD_STATUS(litert_get_main_model_subgraph_index, model,
                               main_subgraph_index);
  }

  LiteRtStatus GetNumModelSubgraphs(LiteRtModel model,
                                    LiteRtParamIndex* num_subgraphs) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_model_subgraphs, model,
                               num_subgraphs);
  }

  LiteRtStatus GetModelSubgraph(LiteRtModel model,
                                LiteRtParamIndex subgraph_index,
                                LiteRtSubgraph* subgraph) {
    LITERT_PROXY_METHOD_STATUS(litert_get_model_subgraph, model, subgraph_index,
                               subgraph);
  }

  LiteRtStatus GetNumModelSignatures(LiteRtModel model,
                                     LiteRtParamIndex* num_signatures) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_model_signatures, model,
                               num_signatures);
  }

  LiteRtStatus GetModelSignature(LiteRtModel model,
                                 LiteRtParamIndex signature_index,
                                 LiteRtSignature* signature) {
    LITERT_PROXY_METHOD_STATUS(litert_get_model_signature, model,
                               signature_index, signature);
  }

  void DestroyModel(LiteRtModel model) {
    LITERT_PROXY_METHOD_VOID(litert_destroy_model, model);
  }

  LiteRtStatus PushOp(LiteRtOpList op_list, LiteRtOp op,
                      LiteRtParamIndex partition_index) {
    LITERT_PROXY_METHOD_STATUS(litert_push_op, op_list, op, partition_index);
  }

  LiteRtStatus SerializeModelWithSignatures(
      LiteRtModel model, uint8_t** buf, size_t* size, size_t* offset,
      bool destroy_model, char** signatures, LiteRtParamIndex num_signatures,
      LiteRtModelSerializationOptions options) {
    LITERT_PROXY_METHOD_STATUS(litert_serialize_model_with_signatures, model,
                               buf, size, offset, destroy_model, signatures,
                               num_signatures, options);
  }

  LiteRtStatus SerializeModel(LiteRtModel model, uint8_t** buf, size_t* size,
                              size_t* offset, bool destroy_model,
                              LiteRtModelSerializationOptions options) {
    LITERT_PROXY_METHOD_STATUS(litert_serialize_model, model, buf, size, offset,
                               destroy_model, options);
  }

  //
  // LiteRtCompiledModel
  //

  LiteRtStatus CreateCompiledModel(LiteRtEnvironment environment,
                                   LiteRtModel model,
                                   LiteRtOptions compilation_options,
                                   LiteRtCompiledModel* compiled_model) {
    LITERT_PROXY_METHOD_STATUS(litert_create_compiled_model, environment, model,
                               compilation_options, compiled_model);
  }

  LiteRtStatus GetCompiledModelInputBufferRequirements(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      LiteRtParamIndex input_index,
      LiteRtTensorBufferRequirements* buffer_requirements) {
    LITERT_PROXY_METHOD_STATUS(
        litert_get_compiled_model_input_buffer_requirements, compiled_model,
        signature_index, input_index, buffer_requirements);
  }

  LiteRtStatus GetCompiledModelOutputBufferRequirements(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      LiteRtParamIndex output_index,
      LiteRtTensorBufferRequirements* buffer_requirements) {
    LITERT_PROXY_METHOD_STATUS(
        litert_get_compiled_model_output_buffer_requirements, compiled_model,
        signature_index, output_index, buffer_requirements);
  }

  LiteRtStatus GetCompiledModelInputTensorLayout(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      LiteRtParamIndex input_index, LiteRtLayout* layout) {
    LITERT_PROXY_METHOD_STATUS(litert_get_compiled_model_input_tensor_layout,
                               compiled_model, signature_index, input_index,
                               layout);
  }

  LiteRtStatus GetCompiledModelOutputTensorLayouts(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      size_t num_layouts, LiteRtLayout* layouts, bool update_allocation) {
    LITERT_PROXY_METHOD_STATUS(litert_get_compiled_model_output_tensor_layouts,
                               compiled_model, signature_index, num_layouts,
                               layouts, update_allocation);
  }

  LiteRtStatus GetCompiledModelEnvironment(LiteRtCompiledModel compiled_model,
                                           LiteRtEnvironment* environment) {
    LITERT_PROXY_METHOD_STATUS(litert_get_compiled_model_environment,
                               compiled_model, environment);
  }

  LiteRtStatus RunCompiledModel(LiteRtCompiledModel compiled_model,
                                LiteRtParamIndex signature_index,
                                size_t num_input_buffers,
                                LiteRtTensorBuffer* input_buffers,
                                size_t num_output_buffers,
                                LiteRtTensorBuffer* output_buffers) {
    LITERT_PROXY_METHOD_STATUS(
        litert_run_compiled_model, compiled_model, signature_index,
        num_input_buffers, input_buffers, num_output_buffers, output_buffers);
  }

  LiteRtStatus RunCompiledModelAsync(LiteRtCompiledModel compiled_model,
                                     LiteRtParamIndex signature_index,
                                     size_t num_input_buffers,
                                     LiteRtTensorBuffer* input_buffers,
                                     size_t num_output_buffers,
                                     LiteRtTensorBuffer* output_buffers,
                                     bool* async) {
    LITERT_PROXY_METHOD_STATUS(litert_run_compiled_model_async, compiled_model,
                               signature_index, num_input_buffers,
                               input_buffers, num_output_buffers,
                               output_buffers, async);
  }

  LiteRtStatus SetCompiledModelCancellationFunction(
      LiteRtCompiledModel compiled_model, void* data,
      bool (*check_cancelled_func)(void*)) {
    LITERT_PROXY_METHOD_STATUS(litert_set_compiled_model_cancellation_function,
                               compiled_model, data, check_cancelled_func);
  }

  void DestroyCompiledModel(LiteRtCompiledModel compiled_model) {
    LITERT_PROXY_METHOD_VOID(litert_destroy_compiled_model, compiled_model);
  }

  LiteRtStatus CompiledModelStartMetricsCollection(
      LiteRtCompiledModel compiled_model, int detail_level) {
    LITERT_PROXY_METHOD_STATUS(litert_compiled_model_start_metrics_collection,
                               compiled_model, detail_level);
  }

  LiteRtStatus CompiledModelStopMetricsCollection(
      LiteRtCompiledModel compiled_model, LiteRtMetrics metrics) {
    LITERT_PROXY_METHOD_STATUS(litert_compiled_model_stop_metrics_collection,
                               compiled_model, metrics);
  }

  LiteRtStatus CompiledModelIsFullyAccelerated(
      LiteRtCompiledModel compiled_model, bool* fully_accelerated) {
    LITERT_PROXY_METHOD_STATUS(litert_compiled_model_is_fully_accelerated,
                               compiled_model, fully_accelerated);
  }

  LiteRtStatus CompiledModelGetProfiler(LiteRtCompiledModel compiled_model,
                                        LiteRtProfiler* profiler) {
    LITERT_PROXY_METHOD_STATUS(litert_compiled_model_get_profiler,
                               compiled_model, profiler);
  }

  LiteRtStatus CompiledModelResizeInputTensor(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      LiteRtParamIndex input_index, const int* dims, size_t dims_size) {
    LITERT_PROXY_METHOD_STATUS(litert_compiled_model_resize_input_tensor,
                               compiled_model, signature_index, input_index,
                               dims, dims_size);
  }

  LiteRtStatus CompiledModelResizeInputTensorNonStrict(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      LiteRtParamIndex input_index, const int* dims, size_t dims_size) {
    LITERT_PROXY_METHOD_STATUS(
        litert_compiled_model_resize_input_tensor_non_strict, compiled_model,
        signature_index, input_index, dims, dims_size);
  }

  LiteRtStatus CompiledModelSetDispatchAnnotation(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      const char* key, const char* value) {
    LITERT_PROXY_METHOD_STATUS(litert_compiled_model_set_dispatch_annotation,
                               compiled_model, signature_index, key, value);
  }

  LiteRtStatus CompiledModelGetDispatchAnnotation(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      const char* key, const char** value) {
    LITERT_PROXY_METHOD_STATUS(litert_compiled_model_get_dispatch_annotation,
                               compiled_model, signature_index, key, value);
  }

  LiteRtStatus CompiledModelRemoveDispatchAnnotation(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      const char* key) {
    LITERT_PROXY_METHOD_STATUS(litert_compiled_model_remove_dispatch_annotation,
                               compiled_model, signature_index, key);
  }

  template <typename... Args>
  LiteRtStatus CompiledModelReportError(LiteRtCompiledModel compiled_model,
                                        const char* format, Args&&... args) {
    // We cannot forward variadic arguments, so this function cannot use the
    // macro.
    ABSL_CHECK(runtime_c_api_->litert_compiled_model_report_error);
    return runtime_c_api_->litert_compiled_model_report_error(
        compiled_model, format, std::forward<Args>(args)...);
  }

  LiteRtStatus CompiledModelClearErrors(LiteRtCompiledModel compiled_model) {
    LITERT_PROXY_METHOD_STATUS(litert_compiled_model_clear_errors,
                               compiled_model);
  }

  LiteRtStatus CompiledModelGetErrorMessages(LiteRtCompiledModel compiled_model,
                                             char** error_messages) {
    LITERT_PROXY_METHOD_STATUS(litert_compiled_model_get_error_messages,
                               compiled_model, error_messages);
  }

  //
  // LiteRtTensorBufferRequirements
  //

  LiteRtStatus CreateTensorBufferRequirements(
      int num_supported_tensor_buffer_types,
      const LiteRtTensorBufferType* supported_tensor_buffer_types,
      size_t buffer_size, int num_strides, const uint32_t* strides,
      LiteRtTensorBufferRequirements* requirements) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_requirements,
                               num_supported_tensor_buffer_types,
                               supported_tensor_buffer_types, buffer_size,
                               num_strides, strides, requirements);
  }

  LiteRtStatus CreateTensorBufferRequirementsWithAlignment(
      int num_supported_tensor_buffer_types,
      const LiteRtTensorBufferType* supported_tensor_buffer_types,
      size_t buffer_size, int num_strides, const uint32_t* strides,
      size_t alignment, LiteRtTensorBufferRequirements* requirements) {
    LITERT_PROXY_METHOD_STATUS(
        litert_create_tensor_buffer_requirements_with_alignment,
        num_supported_tensor_buffer_types, supported_tensor_buffer_types,
        buffer_size, num_strides, strides, alignment, requirements);
  }

  LiteRtStatus GetNumTensorBufferRequirementsSupportedBufferTypes(
      LiteRtTensorBufferRequirements requirements, int* num_types) {
    LITERT_PROXY_METHOD_STATUS(
        litert_get_num_tensor_buffer_requirements_supported_buffer_types,
        requirements, num_types);
  }

  LiteRtStatus GetTensorBufferRequirementsSupportedTensorBufferType(
      LiteRtTensorBufferRequirements requirements, int type_index,
      LiteRtTensorBufferType* type) {
    LITERT_PROXY_METHOD_STATUS(
        litert_get_tensor_buffer_requirements_supported_tensor_buffer_type,
        requirements, type_index, type);
  }

  LiteRtStatus GetTensorBufferRequirementsBufferSize(
      LiteRtTensorBufferRequirements requirements, size_t* buffer_size) {
    LITERT_PROXY_METHOD_STATUS(
        litert_get_tensor_buffer_requirements_buffer_size, requirements,
        buffer_size);
  }

  LiteRtStatus GetTensorBufferRequirementsStrides(
      LiteRtTensorBufferRequirements requirements, int* num_strides,
      const uint32_t** strides) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_requirements_strides,
                               requirements, num_strides, strides);
  }

  LiteRtStatus GetTensorBufferRequirementsAlignment(
      LiteRtTensorBufferRequirements requirements, size_t* alignment) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_requirements_alignment,
                               requirements, alignment);
  }

  LiteRtStatus JoinTensorBufferRequirements(
      LiteRtTensorBufferRequirements src_requirements_1,
      LiteRtTensorBufferRequirements src_requirements_2,
      LiteRtTensorBufferRequirements* joined_requirements) {
    LITERT_PROXY_METHOD_STATUS(litert_join_tensor_buffer_requirements,
                               src_requirements_1, src_requirements_2,
                               joined_requirements);
  }

  void DestroyTensorBufferRequirements(
      LiteRtTensorBufferRequirements requirements) {
    LITERT_PROXY_METHOD_VOID(litert_destroy_tensor_buffer_requirements,
                             requirements);
  }

  //
  // LiteRtTensorBuffer
  //

  LiteRtStatus CreateManagedTensorBuffer(
      LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
      const LiteRtRankedTensorType* tensor_type, size_t buffer_size,
      LiteRtTensorBuffer* buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_managed_tensor_buffer, env,
                               buffer_type, tensor_type, buffer_size, buffer);
  }

  LiteRtStatus CreateManagedTensorBufferFromRequirements(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferRequirements requirements, LiteRtTensorBuffer* buffer) {
    LITERT_PROXY_METHOD_STATUS(
        litert_create_managed_tensor_buffer_from_requirements, env, tensor_type,
        requirements, buffer);
  }

  LiteRtStatus CreateTensorBufferFromHostMemory(
      const LiteRtRankedTensorType* tensor_type, void* host_buffer_addr,
      size_t host_buffer_size, LiteRtHostMemoryDeallocator deallocator,
      LiteRtTensorBuffer* buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_host_memory,
                               tensor_type, host_buffer_addr, host_buffer_size,
                               deallocator, buffer);
  }

  LiteRtStatus GetTensorBufferHostMemory(LiteRtTensorBuffer tensor_buffer,
                                         void** host_memory_addr) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_host_memory,
                               tensor_buffer, host_memory_addr);
  }

#if LITERT_HAS_AHWB_SUPPORT
  LiteRtStatus CreateTensorBufferFromAhwb(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      AHardwareBuffer* ahwb, size_t ahwb_offset,
      LiteRtAhwbDeallocator deallocator, LiteRtTensorBuffer* buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_ahwb, env,
                               tensor_type, ahwb, ahwb_offset, deallocator,
                               buffer);
  }

  LiteRtStatus GetTensorBufferAhwb(LiteRtTensorBuffer tensor_buffer,
                                   AHardwareBuffer** ahwb) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_ahwb, tensor_buffer,
                               ahwb);
  }

#endif  // LITERT_HAS_AHWB_SUPPORT
#if LITERT_HAS_ION_SUPPORT
  LiteRtStatus CreateTensorBufferFromIonBuffer(
      const LiteRtRankedTensorType* tensor_type, void* ion_buffer_addr,
      int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
      LiteRtIonDeallocator deallocator, LiteRtTensorBuffer* buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_ion_buffer,
                               tensor_type, ion_buffer_addr, ion_buffer_fd,
                               ion_buffer_size, ion_buffer_offset, deallocator,
                               buffer);
  }

  LiteRtStatus GetTensorBufferIonBuffer(LiteRtTensorBuffer buffer,
                                        void** ion_buffer_addr,
                                        int* ion_buffer_fd) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_ion_buffer, buffer,
                               ion_buffer_addr, ion_buffer_fd);
  }

#endif  // LITERT_HAS_ION_SUPPORT
#if LITERT_HAS_DMABUF_SUPPORT
  LiteRtStatus CreateTensorBufferFromDmaBufBuffer(
      const LiteRtRankedTensorType* tensor_type, void* dmabuf_buffer_addr,
      int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
      size_t dmabuf_buffer_offset, LiteRtDmaBufDeallocator deallocator,
      LiteRtTensorBuffer* buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_dma_buf_buffer,
                               tensor_type, dmabuf_buffer_addr,
                               dmabuf_buffer_fd, dmabuf_buffer_size,
                               dmabuf_buffer_offset, deallocator, buffer);
  }

  LiteRtStatus GetTensorBufferDmaBufBuffer(LiteRtTensorBuffer tensor_buffer,
                                           void** dmabuf_buffer_addr,
                                           int* dmabuf_buffer_fd) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_dma_buf_buffer,
                               tensor_buffer, dmabuf_buffer_addr,
                               dmabuf_buffer_fd);
  }

#endif  // LITERT_HAS_DMABUF_SUPPORT
#if LITERT_HAS_FASTRPC_SUPPORT
  LiteRtStatus CreateTensorBufferFromFastRpcBuffer(
      const LiteRtRankedTensorType* tensor_type, void* fastrpc_buffer_addr,
      int fastrpc_fd, size_t fastrpc_buffer_size, size_t fastrpc_buffer_offset,
      LiteRtFastRpcDeallocator deallocator, LiteRtTensorBuffer* buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_fast_rpc_buffer,
                               tensor_type, fastrpc_buffer_addr, fastrpc_fd,
                               fastrpc_buffer_size, fastrpc_buffer_offset,
                               deallocator, buffer);
  }

  LiteRtStatus GetTensorBufferFastRpcBuffer(LiteRtTensorBuffer tensor_buffer,
                                            void** fastrpc_buffer_addr,
                                            int* fastrpc_buffer_fd) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_fast_rpc_buffer,
                               tensor_buffer, fastrpc_buffer_addr,
                               fastrpc_buffer_fd);
  }

#endif  // LITERT_HAS_FASTRPC_SUPPORT
#if LITERT_HAS_OPENCL_SUPPORT
  LiteRtStatus CreateTensorBufferFromOpenClMemory(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferType buffer_type, LiteRtClMem cl_mem_addr,
      size_t opencl_buffer_size, LiteRtOpenClDeallocator deallocator,
      LiteRtTensorBuffer* buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_opencl_memory,
                               env, tensor_type, buffer_type, cl_mem_addr,
                               opencl_buffer_size, deallocator, buffer);
  }

  LiteRtStatus GetTensorBufferOpenClMemory(LiteRtTensorBuffer tensor_buffer,
                                           LiteRtClMem* cl_mem_addr) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_opencl_memory,
                               tensor_buffer, cl_mem_addr);
  }

#endif  // LITERT_HAS_OPENCL_SUPPORT
  LiteRtStatus GetTensorBufferCustomTensorBufferHandle(
      LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle) {
    LITERT_PROXY_METHOD_STATUS(
        litert_get_tensor_buffer_custom_tensor_buffer_handle, tensor_buffer,
        hw_memory_handle);
  }

  LiteRtStatus CreateTensorBufferFromGlBuffer(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtGLenum target, LiteRtGLuint id, size_t size_bytes, size_t offset,
      LiteRtGlBufferDeallocator deallocator, LiteRtTensorBuffer* buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_gl_buffer, env,
                               tensor_type, target, id, size_bytes, offset,
                               deallocator, buffer);
  }

  LiteRtStatus GetTensorBufferGlBuffer(LiteRtTensorBuffer tensor_buffer,
                                       LiteRtGLenum* target, LiteRtGLuint* id,
                                       size_t* size_bytes, size_t* offset) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_gl_buffer,
                               tensor_buffer, target, id, size_bytes, offset);
  }

  LiteRtStatus CreateTensorBufferFromGlTexture(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
      size_t size_bytes, LiteRtGLint layer,
      LiteRtGlTextureDeallocator deallocator, LiteRtTensorBuffer* buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_gl_texture, env,
                               tensor_type, target, id, format, size_bytes,
                               layer, deallocator, buffer);
  }

  LiteRtStatus GetTensorBufferGlTexture(LiteRtTensorBuffer tensor_buffer,
                                        LiteRtGLenum* target, LiteRtGLuint* id,
                                        LiteRtGLenum* format,
                                        size_t* size_bytes,
                                        LiteRtGLint* layer) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_gl_texture,
                               tensor_buffer, target, id, format, size_bytes,
                               layer);
  }

#if LITERT_HAS_WEBGPU_SUPPORT
  LiteRtStatus CreateTensorBufferFromWebGpuBuffer(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferType buffer_type, LiteRtWGPUBuffer wgpu_buffer,
      size_t wgpu_buffer_size, LiteRtWebGpuBufferDeallocator deallocator,
      LiteRtTensorBuffer* tensor_buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_web_gpu_buffer,
                               env, tensor_type, buffer_type, wgpu_buffer,
                               wgpu_buffer_size, deallocator, tensor_buffer);
  }

  LiteRtStatus GetTensorBufferWebGpuBuffer(LiteRtTensorBuffer tensor_buffer,
                                           HwMemoryHandle* hw_memory_handle) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_web_gpu_buffer,
                               tensor_buffer, hw_memory_handle);
  }

  LiteRtStatus CreateTensorBufferFromWebGpuTexture(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      void* webgpu_texture, size_t webgpu_texture_size,
      LiteRtWebGpuTextureDeallocator deallocator,
      LiteRtTensorBuffer* tensor_buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_web_gpu_texture,
                               env, tensor_type, webgpu_texture,
                               webgpu_texture_size, deallocator, tensor_buffer);
  }

#endif  // LITERT_HAS_WEBGPU_SUPPORT
#if LITERT_HAS_METAL_SUPPORT
  LiteRtStatus CreateTensorBufferFromMetalMemory(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferType buffer_type, void* metal_buffer,
      size_t metal_buffer_size, LiteRtMetalDeallocator deallocator,
      LiteRtTensorBuffer* tensor_buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_create_tensor_buffer_from_metal_memory,
                               env, tensor_type, buffer_type, metal_buffer,
                               metal_buffer_size, deallocator, tensor_buffer);
  }

  LiteRtStatus GetTensorBufferMetalMemory(LiteRtTensorBuffer tensor_buffer,
                                          HwMemoryHandle* hw_memory_handle) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_metal_memory,
                               tensor_buffer, hw_memory_handle);
  }

#endif  // LITERT_HAS_METAL_SUPPORT
#if LITERT_HAS_VULKAN_SUPPORT
  LiteRtStatus GetTensorBufferVulkanMemory(LiteRtTensorBuffer tensor_buffer,
                                           HwMemoryHandle* hw_memory_handle) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_vulkan_memory,
                               tensor_buffer, hw_memory_handle);
  }

#endif  // LITERT_HAS_VULKAN_SUPPORT
  LiteRtStatus DuplicateTensorBuffer(LiteRtTensorBuffer tensor_buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_duplicate_tensor_buffer, tensor_buffer);
  }

  LiteRtStatus GetTensorBufferType(LiteRtTensorBuffer tensor_buffer,
                                   LiteRtTensorBufferType* buffer_type) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_type, tensor_buffer,
                               buffer_type);
  }

  LiteRtStatus GetTensorBufferTensorType(LiteRtTensorBuffer tensor_buffer,
                                         LiteRtRankedTensorType* tensor_type) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_tensor_type,
                               tensor_buffer, tensor_type);
  }

  LiteRtStatus GetTensorBufferSize(LiteRtTensorBuffer tensor_buffer,
                                   size_t* size) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_size, tensor_buffer,
                               size);
  }

  LiteRtStatus GetTensorBufferPackedSize(LiteRtTensorBuffer tensor_buffer,
                                         size_t* size) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_packed_size,
                               tensor_buffer, size);
  }

  LiteRtStatus GetTensorBufferOffset(LiteRtTensorBuffer tensor_buffer,
                                     size_t* offset) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_offset, tensor_buffer,
                               offset);
  }

  LiteRtStatus HasTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                    bool* has_event) {
    LITERT_PROXY_METHOD_STATUS(litert_has_tensor_buffer_event, tensor_buffer,
                               has_event);
  }

  LiteRtStatus GetTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                    LiteRtEvent* event) {
    LITERT_PROXY_METHOD_STATUS(litert_get_tensor_buffer_event, tensor_buffer,
                               event);
  }

  LiteRtStatus SetTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                    LiteRtEvent event) {
    LITERT_PROXY_METHOD_STATUS(litert_set_tensor_buffer_event, tensor_buffer,
                               event);
  }

  LiteRtStatus ClearTensorBufferEvent(LiteRtTensorBuffer tensor_buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_clear_tensor_buffer_event, tensor_buffer);
  }

  LiteRtStatus LockTensorBuffer(LiteRtTensorBuffer tensor_buffer,
                                void** host_mem_addr,
                                LiteRtTensorBufferLockMode lock_mode) {
    LITERT_PROXY_METHOD_STATUS(litert_lock_tensor_buffer, tensor_buffer,
                               host_mem_addr, lock_mode);
  }

  LiteRtStatus UnlockTensorBuffer(LiteRtTensorBuffer buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_unlock_tensor_buffer, buffer);
  }

  LiteRtStatus ClearTensorBuffer(LiteRtTensorBuffer tensor_buffer) {
    LITERT_PROXY_METHOD_STATUS(litert_clear_tensor_buffer, tensor_buffer);
  }

  void DestroyTensorBuffer(LiteRtTensorBuffer buffer) {
    LITERT_PROXY_METHOD_VOID(litert_destroy_tensor_buffer, buffer);
  }

  //
  // LiteRtEvent
  //
  LiteRtStatus CreateEventFromSyncFenceFd(LiteRtEnvironment env,
                                          int sync_fence_fd, bool owns_fd,
                                          LiteRtEvent* event) {
    LITERT_PROXY_METHOD_STATUS(litert_create_event_from_sync_fence_fd, env,
                               sync_fence_fd, owns_fd, event);
  }

  LiteRtStatus CreateEventFromOpenClEvent(LiteRtEnvironment env,
                                          LiteRtClEvent cl_event,
                                          LiteRtEvent* event) {
    LITERT_PROXY_METHOD_STATUS(litert_create_event_from_opencl_event, env,
                               cl_event, event);
  }

  LiteRtStatus CreateEventFromEglSyncFence(LiteRtEnvironment env,
                                           LiteRtEglSyncKhr egl_sync,
                                           LiteRtEvent* event) {
    LITERT_PROXY_METHOD_STATUS(litert_create_event_from_egl_sync_fence, env,
                               egl_sync, event);
  }

  LiteRtStatus CreateManagedEvent(LiteRtEnvironment env, LiteRtEventType type,
                                  LiteRtEvent* event) {
    LITERT_PROXY_METHOD_STATUS(litert_create_managed_event, env, type, event);
  }

  LiteRtStatus SetCustomEvent(LiteRtEvent event,
                              LiteRtCustomEvent custom_event) {
    LITERT_PROXY_METHOD_STATUS(litert_set_custom_event, event, custom_event);
  }

  LiteRtStatus GetEventEventType(LiteRtEvent event, LiteRtEventType* type) {
    LITERT_PROXY_METHOD_STATUS(litert_get_event_event_type, event, type);
  }

  LiteRtStatus GetEventSyncFenceFd(LiteRtEvent event, int* sync_fence_fd) {
    LITERT_PROXY_METHOD_STATUS(litert_get_event_sync_fence_fd, event,
                               sync_fence_fd);
  }

  LiteRtStatus GetEventOpenClEvent(LiteRtEvent event, LiteRtClEvent* cl_event) {
    LITERT_PROXY_METHOD_STATUS(litert_get_event_opencl_event, event, cl_event);
  }

  LiteRtStatus GetEventEglSync(LiteRtEvent event, LiteRtEglSyncKhr* egl_sync) {
    LITERT_PROXY_METHOD_STATUS(litert_get_event_egl_sync, event, egl_sync);
  }

  LiteRtStatus GetEventCustomNativeEvent(LiteRtEvent event, void** native) {
    LITERT_PROXY_METHOD_STATUS(litert_get_event_custom_native_event, event,
                               native);
  }

  LiteRtStatus WaitEvent(LiteRtEvent event, int64_t timeout_in_ms) {
    LITERT_PROXY_METHOD_STATUS(litert_wait_event, event, timeout_in_ms);
  }

  LiteRtStatus SignalEvent(LiteRtEvent event) {
    LITERT_PROXY_METHOD_STATUS(litert_signal_event, event);
  }

  LiteRtStatus IsEventSignaled(LiteRtEvent event, bool* is_signaled) {
    LITERT_PROXY_METHOD_STATUS(litert_is_event_signaled, event, is_signaled);
  }

  LiteRtStatus DupFdEvent(LiteRtEvent event, int* dup_fd) {
    LITERT_PROXY_METHOD_STATUS(litert_dup_fd_event, event, dup_fd);
  }

  void DestroyEvent(LiteRtEvent event) {
    LITERT_PROXY_METHOD_VOID(litert_destroy_event, event);
  }

  //
  // LiteRtLayout
  //
  LiteRtStatus GetNumLayoutElements(const LiteRtLayout* layout,
                                    size_t* num_elements) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_layout_elements, layout,
                               num_elements);
  }

  LiteRtStatus IsSameLayout(const LiteRtLayout* layout1,
                            const LiteRtLayout* layout2, bool* result) {
    LITERT_PROXY_METHOD_STATUS(litert_is_same_layout, layout1, layout2, result);
  }

  //
  // LiteRtMetrics
  //
  LiteRtStatus CreateMetrics(LiteRtMetrics* metrics) {
    LITERT_PROXY_METHOD_STATUS(litert_create_metrics, metrics);
  }

  LiteRtStatus GetNumMetrics(LiteRtMetrics metrics, int* num_metrics) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_metrics, metrics, num_metrics);
  }

  LiteRtStatus GetMetric(LiteRtMetrics metrics, int metric_index,
                         LiteRtMetric* metric) {
    LITERT_PROXY_METHOD_STATUS(litert_get_metric, metrics, metric_index,
                               metric);
  }

  void DestroyMetrics(LiteRtMetrics metrics) {
    LITERT_PROXY_METHOD_VOID(litert_destroy_metrics, metrics);
  }

  //
  // LiteRtOpaqueOptions
  //
  LiteRtStatus CreateOpaqueOptions(
      const char* payload_identifier, void* payload_data,
      void (*payload_destructor)(void* payload_data),
      LiteRtOpaqueOptions* options) {
    LITERT_PROXY_METHOD_STATUS(litert_create_opaque_options, payload_identifier,
                               payload_data, payload_destructor, options);
  }

  void DestroyOpaqueOptions(LiteRtOpaqueOptions options) {
    LITERT_PROXY_METHOD_VOID(litert_destroy_opaque_options, options);
  }

  LiteRtStatus GetOpaqueOptionsIdentifier(LiteRtOpaqueOptions options,
                                          const char** payload_identifier) {
    LITERT_PROXY_METHOD_STATUS(litert_get_opaque_options_identifier, options,
                               payload_identifier);
  }

  LiteRtStatus GetOpaqueOptionsData(LiteRtOpaqueOptions options,
                                    void** payload_data) {
    LITERT_PROXY_METHOD_STATUS(litert_get_opaque_options_data, options,
                               payload_data);
  }

  LiteRtStatus FindOpaqueOptionsData(LiteRtOpaqueOptions options,
                                     const char* payload_identifier,
                                     void** payload_data) {
    LITERT_PROXY_METHOD_STATUS(litert_find_opaque_options_data, options,
                               payload_identifier, payload_data);
  }

  LiteRtStatus GetNextOpaqueOptions(LiteRtOpaqueOptions* options) {
    LITERT_PROXY_METHOD_STATUS(litert_get_next_opaque_options, options);
  }

  LiteRtStatus AppendOpaqueOptions(LiteRtOpaqueOptions* options,
                                   LiteRtOpaqueOptions appended_options) {
    LITERT_PROXY_METHOD_STATUS(litert_append_opaque_options, options,
                               appended_options);
  }

  LiteRtStatus PopOpaqueOptions(LiteRtOpaqueOptions* options) {
    LITERT_PROXY_METHOD_STATUS(litert_pop_opaque_options, options);
  }

  LiteRtStatus SetOpaqueOptionsHash(
      LiteRtOpaqueOptions options,
      LiteRtOpaqueOptionsHashFunc payload_hash_func) {
    LITERT_PROXY_METHOD_STATUS(litert_set_opaque_options_hash, options,
                               payload_hash_func);
  }

  LiteRtStatus GetOpaqueOptionsHash(LiteRtOpaqueOptions options,
                                    uint64_t* hash) {
    LITERT_PROXY_METHOD_STATUS(litert_get_opaque_options_hash, options, hash);
  }

  //
  // LiteRtOptions
  //
  LiteRtStatus CreateOptions(LiteRtOptions* options) {
    LITERT_PROXY_METHOD_STATUS(litert_create_options, options);
  }

  void DestroyOptions(LiteRtOptions options) {
    LITERT_PROXY_METHOD_VOID(litert_destroy_options, options);
  }

  LiteRtStatus SetOptionsHardwareAccelerators(
      LiteRtOptions options, LiteRtHwAcceleratorSet hardware_accelerators) {
    LITERT_PROXY_METHOD_STATUS(litert_set_options_hardware_accelerators,
                               options, hardware_accelerators);
  }

  LiteRtStatus GetOptionsHardwareAccelerators(
      LiteRtOptions options, LiteRtHwAcceleratorSet* hardware_accelerators) {
    LITERT_PROXY_METHOD_STATUS(litert_get_options_hardware_accelerators,
                               options, hardware_accelerators);
  }

  LiteRtStatus AddOpaqueOptions(LiteRtOptions options,
                                LiteRtOpaqueOptions opaque_options) {
    LITERT_PROXY_METHOD_STATUS(litert_add_opaque_options, options,
                               opaque_options);
  }

  LiteRtStatus GetOpaqueOptions(LiteRtOptions options,
                                LiteRtOpaqueOptions* opaque_options) {
    LITERT_PROXY_METHOD_STATUS(litert_get_opaque_options, options,
                               opaque_options);
  }

  LiteRtStatus AddCustomOpKernelOption(
      LiteRtOptions options, const char* custom_op_name, int custom_op_version,
      const LiteRtCustomOpKernel* custom_op_kernel,
      void* custom_op_kernel_user_data) {
    LITERT_PROXY_METHOD_STATUS(litert_add_custom_op_kernel_option, options,
                               custom_op_name, custom_op_version,
                               custom_op_kernel, custom_op_kernel_user_data);
  }

  LiteRtStatus AddExternalTensorBinding(LiteRtOptions options,
                                        const char* signature_name,
                                        const char* tensor_name, void* data,
                                        int size_bytes) {
    LITERT_PROXY_METHOD_STATUS(litert_add_external_tensor_binding, options,
                               signature_name, tensor_name, data, size_bytes);
  }

  //
  // LiteRtProfiler
  //
  LiteRtStatus CreateProfiler(int size, LiteRtProfiler* profiler) {
    LITERT_PROXY_METHOD_STATUS(litert_create_profiler, size, profiler);
  }

  void DestroyProfiler(LiteRtProfiler profiler) {
    LITERT_PROXY_METHOD_VOID(litert_destroy_profiler, profiler);
  }

  LiteRtStatus StartProfiler(LiteRtProfiler profiler) {
    LITERT_PROXY_METHOD_STATUS(litert_start_profiler, profiler);
  }

  LiteRtStatus StopProfiler(LiteRtProfiler profiler) {
    LITERT_PROXY_METHOD_STATUS(litert_stop_profiler, profiler);
  }

  LiteRtStatus ResetProfiler(LiteRtProfiler profiler) {
    LITERT_PROXY_METHOD_STATUS(litert_reset_profiler, profiler);
  }

  LiteRtStatus SetProfilerCurrentEventSource(LiteRtProfiler profiler,
                                             ProfiledEventSource event_source) {
    LITERT_PROXY_METHOD_STATUS(litert_set_profiler_current_event_source,
                               profiler, event_source);
  }

  LiteRtStatus GetNumProfilerEvents(LiteRtProfiler profiler, int* num_events) {
    LITERT_PROXY_METHOD_STATUS(litert_get_num_profiler_events, profiler,
                               num_events);
  }

  LiteRtStatus GetProfilerEvents(LiteRtProfiler profiler, int num_events,
                                 ProfiledEventData* events) {
    LITERT_PROXY_METHOD_STATUS(litert_get_profiler_events, profiler, num_events,
                               events);
  }

  LiteRtStatus GetProfileSummary(LiteRtProfiler profiler,
                                 LiteRtCompiledModel compiled_model,
                                 const char** summary) {
    LITERT_PROXY_METHOD_STATUS(litert_get_profile_summary, profiler,
                               compiled_model, summary);
  }

 protected:
  const LiteRtRuntimeCApiStruct* runtime_c_api_;
};

#undef LITERT_PROXY_METHOD_STATUS
#undef LITERT_PROXY_METHOD_VOID

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_RUNTIME_PROXY_H_
