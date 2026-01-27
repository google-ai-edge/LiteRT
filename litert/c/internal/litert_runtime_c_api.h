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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_RUNTIME_C_API_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_RUNTIME_C_API_H_

#include <cstddef>
#include <cstdint>

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

extern "C" {

// TODO(b/475253786): Clean up build time macros in the function table.

// A struct that contains all the LiteRT runtime C API functions.
//
// This struct is used to provide a unified interface for the LiteRT runtime
// regardless of the underlying runtime implementation.
//
// NOTE: All new fields should be added to the end of the struct.
typedef struct LiteRtRuntimeCApiStruct {
  //
  // LiteRtEnvironment
  //
  // litert_environment.h: LiteRtCreateEnvironment
  LiteRtStatus (*litert_create_environment)(int num_options,
                                            const LiteRtEnvOption* options,
                                            LiteRtEnvironment* environment);
  // litert_environment.h: LiteRtDestroyEnvironment
  void (*litert_destroy_environment)(LiteRtEnvironment environment);
  // litert_environment.h: LiteRtGetEnvironmentOptions
  LiteRtStatus (*litert_get_environment_options)(
      LiteRtEnvironment environment, LiteRtEnvironmentOptions* options);
  // litert_environment.h: LiteRtAddEnvironmentOptions
  LiteRtStatus (*litert_add_environment_options)(LiteRtEnvironment environment,
                                                 int num_options,
                                                 const LiteRtEnvOption* options,
                                                 bool overwrite);
  // litert_environment.h: LiteRtGpuEnvironmentCreate
  LiteRtStatus (*litert_gpu_environment_create)(LiteRtEnvironment environment,
                                                int num_options,
                                                const LiteRtEnvOption* options);
  // litert_environment.h: LiteRtEnvironmentSupportsClGlInterop
  LiteRtStatus (*litert_environment_supports_cl_gl_interop)(
      LiteRtEnvironment environment, bool* is_supported);
  // litert_environment.h: LiteRtEnvironmentSupportsAhwbClInterop
  LiteRtStatus (*litert_environment_supports_ahwb_cl_interop)(
      LiteRtEnvironment environment, bool* is_supported);
  // litert_environment.h: LiteRtEnvironmentSupportsAhwbGlInterop
  LiteRtStatus (*litert_environment_supports_ahwb_gl_interop)(
      LiteRtEnvironment environment, bool* is_supported);
  // litert_environment.h: LiteRtEnvironmentHasGpuEnvironment
  void (*litert_environment_has_gpu_environment)(LiteRtEnvironment environment,
                                                 bool* has_gpu_environment);

  //
  // LiteRtEnvironmentOptions
  //
  // litert_environment_options.h: LiteRtGetEnvironmentOptionsValue
  LiteRtStatus (*litert_get_environment_options_value)(
      LiteRtEnvironmentOptions options, LiteRtEnvOptionTag tag,
      LiteRtAny* value);
  // litert_environment_options.h: LiteRtSetEnvironmentOptionsValue
  LiteRtStatus (*litert_set_environment_options_value)(
      LiteRtEnvironmentOptions options, LiteRtEnvOption env_option);

  //
  // LiteRtTensor
  //
  // litert_model.h: LiteRtGetTensorName
  LiteRtStatus (*litert_get_tensor_name)(LiteRtTensor tensor,
                                         const char** name);
  // litert_model.h: LiteRtGetTensorIndex
  LiteRtStatus (*litert_get_tensor_index)(LiteRtTensor tensor,
                                          uint32_t* tensor_index);
  // litert_model.h: LiteRtGetTensorTypeId
  LiteRtStatus (*litert_get_tensor_type_id)(LiteRtTensor tensor,
                                            LiteRtTensorTypeId* type_id);
  // litert_model.h: LiteRtGetUnrankedTensorType
  LiteRtStatus (*litert_get_unranked_tensor_type)(
      LiteRtTensor tensor, LiteRtUnrankedTensorType* unranked_tensor_type);
  // litert_model.h: LiteRtGetRankedTensorType
  LiteRtStatus (*litert_get_ranked_tensor_type)(
      LiteRtTensor tensor, LiteRtRankedTensorType* ranked_tensor_type);
  // litert_model.h: LiteRtGetQuantizationTypeId
  LiteRtStatus (*litert_get_quantization_type_id)(
      LiteRtTensor tensor, LiteRtQuantizationTypeId* q_type_id);
  // litert_model.h: LiteRtGetPerTensorQuantization
  LiteRtStatus (*litert_get_per_tensor_quantization)(
      LiteRtTensor tensor,
      LiteRtQuantizationPerTensor* per_tensor_quantization);
  // litert_model.h: LiteRtGetPerChannelQuantization
  LiteRtStatus (*litert_get_per_channel_quantization)(
      LiteRtTensor tensor,
      LiteRtQuantizationPerChannel* per_channel_quantization);
  // litert_model.h: LiteRtGetNumTensorUses
  LiteRtStatus (*litert_get_num_tensor_uses)(LiteRtTensor tensor,
                                             LiteRtParamIndex* num_uses);
  // litert_model.h: LiteRtGetTensorUse
  LiteRtStatus (*litert_get_tensor_use)(LiteRtTensor tensor,
                                        LiteRtParamIndex use_index,
                                        LiteRtOp* user,
                                        LiteRtParamIndex* user_arg_index);
  // litert_model.h: LiteRtGetTensorDefiningOp
  LiteRtStatus (*litert_get_tensor_defining_op)(
      LiteRtTensor tensor, bool* has_defining_op,
      LiteRtTensorDefiningOp* defining_op);
  // litert_model.h: LiteRtGetTensorWeights
  LiteRtStatus (*litert_get_tensor_weights)(LiteRtTensor tensor,
                                            LiteRtWeights* weights);

  //
  // LiteRtSubgraph
  //
  // litert_model.h: LiteRtGetNumSubgraphInputs
  LiteRtStatus (*litert_get_num_subgraph_inputs)(LiteRtSubgraph subgraph,
                                                 LiteRtParamIndex* num_inputs);
  // litert_model.h: LiteRtGetSubgraphInput
  LiteRtStatus (*litert_get_subgraph_input)(LiteRtSubgraph subgraph,
                                            LiteRtParamIndex input_index,
                                            LiteRtTensor* input);
  // litert_model.h: LiteRtGetNumSubgraphOutputs
  LiteRtStatus (*litert_get_num_subgraph_outputs)(
      LiteRtSubgraph subgraph, LiteRtParamIndex* num_outputs);
  // litert_model.h: LiteRtGetSubgraphOutput
  LiteRtStatus (*litert_get_subgraph_output)(LiteRtSubgraph subgraph,
                                             LiteRtParamIndex output_index,
                                             LiteRtTensor* output);
  // litert_model.h: LiteRtGetNumSubgraphOps
  LiteRtStatus (*litert_get_num_subgraph_ops)(LiteRtSubgraph subgraph,
                                              LiteRtParamIndex* num_ops);
  // litert_model.h: LiteRtGetSubgraphOp
  LiteRtStatus (*litert_get_subgraph_op)(LiteRtSubgraph subgraph,
                                         LiteRtParamIndex op_index,
                                         LiteRtOp* op);

  //
  // LiteRtSignature
  //
  // litert_model.h: LiteRtGetDefaultSignatureKey
  LiteRtStatus (*litert_get_default_signature_key)(const char** signature_key);
  // litert_model.h: LiteRtGetSignatureKey
  LiteRtStatus (*litert_get_signature_key)(LiteRtSignature signature,
                                           const char** signature_key);
  // litert_model.h: LiteRtGetSignatureSubgraph
  LiteRtStatus (*litert_get_signature_subgraph)(LiteRtSignature signature,
                                                LiteRtSubgraph* subgraph);
  // litert_model.h: LiteRtGetNumSignatureInputs
  LiteRtStatus (*litert_get_num_signature_inputs)(LiteRtSignature signature,
                                                  LiteRtParamIndex* num_inputs);
  // litert_model.h: LiteRtGetSignatureInputName
  LiteRtStatus (*litert_get_signature_input_name)(LiteRtSignature signature,
                                                  LiteRtParamIndex input_idx,
                                                  const char** input_name);
  // litert_model.h: LiteRtGetSignatureInputTensor
  LiteRtStatus (*litert_get_signature_input_tensor)(LiteRtSignature signature,
                                                    const char* input_name,
                                                    LiteRtTensor* tensor);
  // litert_model.h: LiteRtGetSignatureInputTensorByIndex
  LiteRtStatus (*litert_get_signature_input_tensor_by_index)(
      LiteRtSignature signature, LiteRtParamIndex input_idx,
      LiteRtTensor* tensor);
  // litert_model.h: LiteRtGetNumSignatureOutputs
  LiteRtStatus (*litert_get_num_signature_outputs)(
      LiteRtSignature signature, LiteRtParamIndex* num_outputs);
  // litert_model.h: LiteRtGetSignatureOutputName
  LiteRtStatus (*litert_get_signature_output_name)(LiteRtSignature signature,
                                                   LiteRtParamIndex output_idx,
                                                   const char** output_name);
  // litert_model.h: LiteRtGetSignatureOutputTensor
  LiteRtStatus (*litert_get_signature_output_tensor)(LiteRtSignature signature,
                                                     const char* output_name,
                                                     LiteRtTensor* tensor);
  // litert_model.h: LiteRtGetSignatureOutputTensorByIndex
  LiteRtStatus (*litert_get_signature_output_tensor_by_index)(
      LiteRtSignature signature, LiteRtParamIndex output_idx,
      LiteRtTensor* tensor);

  //
  // LiteRtModel
  //
  // litert_model.h: LiteRtCreateModelFromFile
  LiteRtStatus (*litert_create_model_from_file)(const char* filename,
                                                LiteRtModel* model);
  // litert_model.h: LiteRtCreateModelFromBuffer
  LiteRtStatus (*litert_create_model_from_buffer)(const void* buffer_addr,
                                                  size_t buffer_size,
                                                  LiteRtModel* model);
  // litert_model.h: LiteRtGetModelMetadata
  LiteRtStatus (*litert_get_model_metadata)(LiteRtModel model,
                                            const char* metadata_key,
                                            const void** metadata_buffer,
                                            size_t* metadata_buffer_size);
  // litert_model.h: LiteRtAddModelMetadata
  LiteRtStatus (*litert_add_model_metadata)(LiteRtModel model,
                                            const char* metadata_key,
                                            const void* metadata_buffer,
                                            size_t metadata_buffer_size);
  // litert_model.h: LiteRtGetMainModelSubgraphIndex
  LiteRtStatus (*litert_get_main_model_subgraph_index)(
      LiteRtModel model, LiteRtParamIndex* main_subgraph_index);
  // litert_model.h: LiteRtGetNumModelSubgraphs
  LiteRtStatus (*litert_get_num_model_subgraphs)(
      LiteRtModel model, LiteRtParamIndex* num_subgraphs);
  // litert_model.h: LiteRtGetModelSubgraph
  LiteRtStatus (*litert_get_model_subgraph)(LiteRtModel model,
                                            LiteRtParamIndex subgraph_index,
                                            LiteRtSubgraph* subgraph);
  // litert_model.h: LiteRtGetNumModelSignatures
  LiteRtStatus (*litert_get_num_model_signatures)(
      LiteRtModel model, LiteRtParamIndex* num_signatures);
  // litert_model.h: LiteRtGetModelSignature
  LiteRtStatus (*litert_get_model_signature)(LiteRtModel model,
                                             LiteRtParamIndex signature_index,
                                             LiteRtSignature* signature);
  // litert_model.h: LiteRtDestroyModel
  void (*litert_destroy_model)(LiteRtModel model);
  // litert_model.h: LiteRtPushOp
  LiteRtStatus (*litert_push_op)(LiteRtOpList op_list, LiteRtOp op,
                                 LiteRtParamIndex partition_index);
  // litert_model.h: LiteRtSerializeModelWithSignatures
  LiteRtStatus (*litert_serialize_model_with_signatures)(
      LiteRtModel model, uint8_t** buf, size_t* size, size_t* offset,
      bool destroy_model, char** signatures, LiteRtParamIndex num_signatures,
      LiteRtModelSerializationOptions options);
  // litert_model.h: LiteRtSerializeModel
  LiteRtStatus (*litert_serialize_model)(
      LiteRtModel model, uint8_t** buf, size_t* size, size_t* offset,
      bool destroy_model, LiteRtModelSerializationOptions options);

  //
  // LiteRtCompiledModel
  //
  // litert_compiled_model.h: LiteRtCreateCompiledModel
  LiteRtStatus (*litert_create_compiled_model)(
      LiteRtEnvironment environment, LiteRtModel model,
      LiteRtOptions compilation_options, LiteRtCompiledModel* compiled_model);
  // litert_compiled_model.h: LiteRtGetCompiledModelInputBufferRequirements
  LiteRtStatus (*litert_get_compiled_model_input_buffer_requirements)(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      LiteRtParamIndex input_index,
      LiteRtTensorBufferRequirements* buffer_requirements);
  // litert_compiled_model.h: LiteRtGetCompiledModelOutputBufferRequirements
  LiteRtStatus (*litert_get_compiled_model_output_buffer_requirements)(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      LiteRtParamIndex output_index,
      LiteRtTensorBufferRequirements* buffer_requirements);
  // litert_compiled_model.h: LiteRtGetCompiledModelInputTensorLayout
  LiteRtStatus (*litert_get_compiled_model_input_tensor_layout)(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      LiteRtParamIndex input_index, LiteRtLayout* layout);
  // litert_compiled_model.h: LiteRtGetCompiledModelOutputTensorLayouts
  LiteRtStatus (*litert_get_compiled_model_output_tensor_layouts)(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      size_t num_layouts, LiteRtLayout* layouts, bool update_allocation);
  // litert_compiled_model.h: LiteRtGetCompiledModelEnvironment
  LiteRtStatus (*litert_get_compiled_model_environment)(
      LiteRtCompiledModel compiled_model, LiteRtEnvironment* environment);
  // litert_compiled_model.h: LiteRtRunCompiledModel
  LiteRtStatus (*litert_run_compiled_model)(LiteRtCompiledModel compiled_model,
                                            LiteRtParamIndex signature_index,
                                            size_t num_input_buffers,
                                            LiteRtTensorBuffer* input_buffers,
                                            size_t num_output_buffers,
                                            LiteRtTensorBuffer* output_buffers);
  // litert_compiled_model.h: LiteRtRunCompiledModelAsync
  LiteRtStatus (*litert_run_compiled_model_async)(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      size_t num_input_buffers, LiteRtTensorBuffer* input_buffers,
      size_t num_output_buffers, LiteRtTensorBuffer* output_buffers,
      bool* async);
  // litert_compiled_model.h: LiteRtSetCompiledModelCancellationFunction
  LiteRtStatus (*litert_set_compiled_model_cancellation_function)(
      LiteRtCompiledModel compiled_model, void* data,
      bool (*check_cancelled_func)(void*));
  // litert_compiled_model.h: LiteRtDestroyCompiledModel
  void (*litert_destroy_compiled_model)(LiteRtCompiledModel compiled_model);
  // litert_compiled_model.h: LiteRtCompiledModelStartMetricsCollection
  LiteRtStatus (*litert_compiled_model_start_metrics_collection)(
      LiteRtCompiledModel compiled_model, int detail_level);
  // litert_compiled_model.h: LiteRtCompiledModelStopMetricsCollection
  LiteRtStatus (*litert_compiled_model_stop_metrics_collection)(
      LiteRtCompiledModel compiled_model, LiteRtMetrics metrics);
  // litert_compiled_model.h: LiteRtCompiledModelIsFullyAccelerated
  LiteRtStatus (*litert_compiled_model_is_fully_accelerated)(
      LiteRtCompiledModel compiled_model, bool* fully_accelerated);
  // litert_compiled_model.h: LiteRtCompiledModelGetProfiler
  LiteRtStatus (*litert_compiled_model_get_profiler)(
      LiteRtCompiledModel compiled_model, LiteRtProfiler* profiler);
  // litert_compiled_model.h: LiteRtCompiledModelResizeInputTensor
  LiteRtStatus (*litert_compiled_model_resize_input_tensor)(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      LiteRtParamIndex input_index, const int* dims, size_t dims_size);
  // litert_compiled_model.h: LiteRtCompiledModelResizeInputTensorNonStrict
  LiteRtStatus (*litert_compiled_model_resize_input_tensor_non_strict)(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      LiteRtParamIndex input_index, const int* dims, size_t dims_size);
  // litert_compiled_model.h: LiteRtCompiledModelSetDispatchAnnotation
  LiteRtStatus (*litert_compiled_model_set_dispatch_annotation)(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      const char* key, const char* value);
  // litert_compiled_model.h: LiteRtCompiledModelGetDispatchAnnotation
  LiteRtStatus (*litert_compiled_model_get_dispatch_annotation)(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      const char* key, const char** value);
  // litert_compiled_model.h: LiteRtCompiledModelRemoveDispatchAnnotation
  LiteRtStatus (*litert_compiled_model_remove_dispatch_annotation)(
      LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
      const char* key);
  // litert_compiled_model.h: LiteRtCompiledModelReportError
  LiteRtStatus (*litert_compiled_model_report_error)(
      LiteRtCompiledModel compiled_model, const char* format, ...);
  // litert_compiled_model.h: LiteRtCompiledModelClearErrors
  LiteRtStatus (*litert_compiled_model_clear_errors)(
      LiteRtCompiledModel compiled_model);
  // litert_compiled_model.h: LiteRtCompiledModelGetErrorMessages
  LiteRtStatus (*litert_compiled_model_get_error_messages)(
      LiteRtCompiledModel compiled_model, char** error_messages);

  //
  // LiteRtTensorBufferRequirements
  //
  // litert_tensor_buffer_requirements.h: LiteRtCreateTensorBufferRequirements
  LiteRtStatus (*litert_create_tensor_buffer_requirements)(
      int num_supported_tensor_buffer_types,
      const LiteRtTensorBufferType* supported_tensor_buffer_types,
      size_t buffer_size, int num_strides, const uint32_t* strides,
      LiteRtTensorBufferRequirements* requirements);
  // litert_tensor_buffer_requirements.h:
  // LiteRtCreateTensorBufferRequirementsWithAlignment
  LiteRtStatus (*litert_create_tensor_buffer_requirements_with_alignment)(
      int num_supported_tensor_buffer_types,
      const LiteRtTensorBufferType* supported_tensor_buffer_types,
      size_t buffer_size, int num_strides, const uint32_t* strides,
      size_t alignment, LiteRtTensorBufferRequirements* requirements);
  // litert_tensor_buffer_requirements.h:
  // LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes
  LiteRtStatus (
      *litert_get_num_tensor_buffer_requirements_supported_buffer_types)(
      LiteRtTensorBufferRequirements requirements, int* num_types);
  // litert_tensor_buffer_requirements.h:
  // LiteRtGetTensorBufferRequirementsSupportedTensorBufferType
  LiteRtStatus (
      *litert_get_tensor_buffer_requirements_supported_tensor_buffer_type)(
      LiteRtTensorBufferRequirements requirements, int type_index,
      LiteRtTensorBufferType* type);
  // litert_tensor_buffer_requirements.h:
  // LiteRtGetTensorBufferRequirementsBufferSize
  LiteRtStatus (*litert_get_tensor_buffer_requirements_buffer_size)(
      LiteRtTensorBufferRequirements requirements, size_t* buffer_size);
  // litert_tensor_buffer_requirements.h:
  // LiteRtGetTensorBufferRequirementsStrides
  LiteRtStatus (*litert_get_tensor_buffer_requirements_strides)(
      LiteRtTensorBufferRequirements requirements, int* num_strides,
      const uint32_t** strides);
  // litert_tensor_buffer_requirements.h:
  // LiteRtGetTensorBufferRequirementsAlignment
  LiteRtStatus (*litert_get_tensor_buffer_requirements_alignment)(
      LiteRtTensorBufferRequirements requirements, size_t* alignment);
  // litert_tensor_buffer_requirements.h: LiteRtJoinTensorBufferRequirements
  LiteRtStatus (*litert_join_tensor_buffer_requirements)(
      LiteRtTensorBufferRequirements src_requirements_1,
      LiteRtTensorBufferRequirements src_requirements_2,
      LiteRtTensorBufferRequirements* joined_requirements);
  // litert_tensor_buffer_requirements.h:
  // LiteRtDestroyTensorBufferRequirements
  void (*litert_destroy_tensor_buffer_requirements)(
      LiteRtTensorBufferRequirements requirements);

  //
  // LiteRtTensorBuffer
  //
  // litert_tensor_buffer.h: LiteRtCreateTensorBufferFromHostMemory
  LiteRtStatus (*litert_create_tensor_buffer_from_host_memory)(
      const LiteRtRankedTensorType* tensor_type, void* host_buffer_addr,
      size_t host_buffer_size, LiteRtHostMemoryDeallocator deallocator,
      LiteRtTensorBuffer* buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferHostMemory
  LiteRtStatus (*litert_get_tensor_buffer_host_memory)(
      LiteRtTensorBuffer tensor_buffer, void** host_memory_addr);
#if LITERT_HAS_AHWB_SUPPORT
  // litert_tensor_buffer.h: LiteRtCreateTensorBufferFromAhwb
  LiteRtStatus (*litert_create_tensor_buffer_from_ahwb)(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      AHardwareBuffer* ahwb, size_t ahwb_offset,
      LiteRtAhwbDeallocator deallocator, LiteRtTensorBuffer* buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferAhwb
  LiteRtStatus (*litert_get_tensor_buffer_ahwb)(
      LiteRtTensorBuffer tensor_buffer, AHardwareBuffer** ahwb);
#endif  // LITERT_HAS_AHWB_SUPPORT
#if LITERT_HAS_ION_SUPPORT
  // litert_tensor_buffer.h: LiteRtCreateTensorBufferFromIonBuffer
  LiteRtStatus (*litert_create_tensor_buffer_from_ion_buffer)(
      const LiteRtRankedTensorType* tensor_type, void* ion_buffer_addr,
      int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
      LiteRtIonDeallocator deallocator, LiteRtTensorBuffer* buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferIonBuffer
  LiteRtStatus (*litert_get_tensor_buffer_ion_buffer)(LiteRtTensorBuffer buffer,
                                                      void** ion_buffer_addr,
                                                      int* ion_buffer_fd);
#endif  // LITERT_HAS_ION_SUPPORT
#if LITERT_HAS_DMABUF_SUPPORT
  // litert_tensor_buffer.h: LiteRtCreateTensorBufferFromDmaBufBuffer
  LiteRtStatus (*litert_create_tensor_buffer_from_dma_buf_buffer)(
      const LiteRtRankedTensorType* tensor_type, void* dmabuf_buffer_addr,
      int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
      size_t dmabuf_buffer_offset, LiteRtDmaBufDeallocator deallocator,
      LiteRtTensorBuffer* buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferDmaBufBuffer
  LiteRtStatus (*litert_get_tensor_buffer_dma_buf_buffer)(
      LiteRtTensorBuffer tensor_buffer, void** dmabuf_buffer_addr,
      int* dmabuf_buffer_fd);
#endif  // LITERT_HAS_DMABUF_SUPPORT
#if LITERT_HAS_FASTRPC_SUPPORT
  // litert_tensor_buffer.h: LiteRtCreateTensorBufferFromFastRpcBuffer
  LiteRtStatus (*litert_create_tensor_buffer_from_fast_rpc_buffer)(
      const LiteRtRankedTensorType* tensor_type, void* fastrpc_buffer_addr,
      int fastrpc_fd, size_t fastrpc_buffer_size, size_t fastrpc_buffer_offset,
      LiteRtFastRpcDeallocator deallocator, LiteRtTensorBuffer* buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferFastRpcBuffer
  LiteRtStatus (*litert_get_tensor_buffer_fast_rpc_buffer)(
      LiteRtTensorBuffer tensor_buffer, void** fastrpc_buffer_addr,
      int* fastrpc_buffer_fd);
#endif  // LITERT_HAS_FASTRPC_SUPPORT
#if LITERT_HAS_OPENCL_SUPPORT
  // litert_tensor_buffer.h: LiteRtCreateTensorBufferFromOpenClMemory
  LiteRtStatus (*litert_create_tensor_buffer_from_opencl_memory)(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferType buffer_type, LiteRtClMem cl_mem_addr,
      size_t opencl_buffer_size, LiteRtOpenClDeallocator deallocator,
      LiteRtTensorBuffer* buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferOpenClMemory
  LiteRtStatus (*litert_get_tensor_buffer_opencl_memory)(
      LiteRtTensorBuffer tensor_buffer, LiteRtClMem* cl_mem_addr);
#endif  // LITERT_HAS_OPENCL_SUPPORT
  // litert_tensor_buffer.h: LiteRtGetTensorBufferCustomTensorBufferHandle
  LiteRtStatus (*litert_get_tensor_buffer_custom_tensor_buffer_handle)(
      LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle);
  // litert_tensor_buffer.h: LiteRtCreateTensorBufferFromGlBuffer
  LiteRtStatus (*litert_create_tensor_buffer_from_gl_buffer)(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtGLenum target, LiteRtGLuint id, size_t size_bytes, size_t offset,
      LiteRtGlBufferDeallocator deallocator, LiteRtTensorBuffer* buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferGlBuffer
  LiteRtStatus (*litert_get_tensor_buffer_gl_buffer)(
      LiteRtTensorBuffer tensor_buffer, LiteRtGLenum* target, LiteRtGLuint* id,
      size_t* size_bytes, size_t* offset);
  // litert_tensor_buffer.h: LiteRtCreateTensorBufferFromGlTexture
  LiteRtStatus (*litert_create_tensor_buffer_from_gl_texture)(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
      size_t size_bytes, LiteRtGLint layer,
      LiteRtGlTextureDeallocator deallocator, LiteRtTensorBuffer* buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferGlTexture
  LiteRtStatus (*litert_get_tensor_buffer_gl_texture)(
      LiteRtTensorBuffer tensor_buffer, LiteRtGLenum* target, LiteRtGLuint* id,
      LiteRtGLenum* format, size_t* size_bytes, LiteRtGLint* layer);
#if LITERT_HAS_WEBGPU_SUPPORT
  // litert_tensor_buffer.h: LiteRtCreateTensorBufferFromWebGpuBuffer
  LiteRtStatus (*litert_create_tensor_buffer_from_web_gpu_buffer)(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferType buffer_type, LiteRtWGPUBuffer wgpu_buffer,
      size_t wgpu_buffer_size, LiteRtWebGpuBufferDeallocator deallocator,
      LiteRtTensorBuffer* tensor_buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferWebGpuBuffer
  LiteRtStatus (*litert_get_tensor_buffer_web_gpu_buffer)(
      LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle);
  LiteRtStatus (*litert_create_tensor_buffer_from_web_gpu_texture)(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      void* webgpu_texture, size_t webgpu_texture_size,
      LiteRtWebGpuTextureDeallocator deallocator,
      LiteRtTensorBuffer* tensor_buffer);
#endif  // LITERT_HAS_WEBGPU_SUPPORT
#if LITERT_HAS_METAL_SUPPORT
  // litert_tensor_buffer.h: LiteRtCreateTensorBufferFromMetalMemory
  LiteRtStatus (*litert_create_tensor_buffer_from_metal_memory)(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferType buffer_type, void* metal_buffer,
      size_t metal_buffer_size, LiteRtMetalDeallocator deallocator,
      LiteRtTensorBuffer* tensor_buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferMetalMemory
  LiteRtStatus (*litert_get_tensor_buffer_metal_memory)(
      LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle);
#endif  // LITERT_HAS_METAL_SUPPORT
#if LITERT_HAS_VULKAN_SUPPORT
  // litert_tensor_buffer.h: LiteRtGetTensorBufferVulkanMemory
  LiteRtStatus (*litert_get_tensor_buffer_vulkan_memory)(
      LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle);
#endif  // LITERT_HAS_VULKAN_SUPPORT
  // litert_tensor_buffer.h: LiteRtCreateManagedTensorBuffer
  LiteRtStatus (*litert_create_managed_tensor_buffer)(
      LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
      const LiteRtRankedTensorType* tensor_type, size_t buffer_size,
      LiteRtTensorBuffer* buffer);
  // litert_tensor_buffer.h: LiteRtCreateManagedTensorBufferFromRequirements
  LiteRtStatus (*litert_create_managed_tensor_buffer_from_requirements)(
      LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferRequirements requirements, LiteRtTensorBuffer* buffer);
  // litert_tensor_buffer.h: LiteRtDuplicateTensorBuffer
  LiteRtStatus (*litert_duplicate_tensor_buffer)(
      LiteRtTensorBuffer tensor_buffer);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferType
  LiteRtStatus (*litert_get_tensor_buffer_type)(
      LiteRtTensorBuffer tensor_buffer, LiteRtTensorBufferType* buffer_type);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferTensorType
  LiteRtStatus (*litert_get_tensor_buffer_tensor_type)(
      LiteRtTensorBuffer tensor_buffer, LiteRtRankedTensorType* tensor_type);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferSize
  LiteRtStatus (*litert_get_tensor_buffer_size)(
      LiteRtTensorBuffer tensor_buffer, size_t* size);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferPackedSize
  LiteRtStatus (*litert_get_tensor_buffer_packed_size)(
      LiteRtTensorBuffer tensor_buffer, size_t* size);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferOffset
  LiteRtStatus (*litert_get_tensor_buffer_offset)(
      LiteRtTensorBuffer tensor_buffer, size_t* offset);
  // litert_tensor_buffer.h: LiteRtHasTensorBufferEvent
  LiteRtStatus (*litert_has_tensor_buffer_event)(
      LiteRtTensorBuffer tensor_buffer, bool* has_event);
  // litert_tensor_buffer.h: LiteRtGetTensorBufferEvent
  LiteRtStatus (*litert_get_tensor_buffer_event)(
      LiteRtTensorBuffer tensor_buffer, LiteRtEvent* event);
  // litert_tensor_buffer.h: LiteRtSetTensorBufferEvent
  LiteRtStatus (*litert_set_tensor_buffer_event)(
      LiteRtTensorBuffer tensor_buffer, LiteRtEvent event);
  // litert_tensor_buffer.h: LiteRtClearTensorBufferEvent
  LiteRtStatus (*litert_clear_tensor_buffer_event)(
      LiteRtTensorBuffer tensor_buffer);
  // litert_tensor_buffer.h: LiteRtLockTensorBuffer
  LiteRtStatus (*litert_lock_tensor_buffer)(
      LiteRtTensorBuffer tensor_buffer, void** host_mem_addr,
      LiteRtTensorBufferLockMode lock_mode);
  // litert_tensor_buffer.h: LiteRtUnlockTensorBuffer
  LiteRtStatus (*litert_unlock_tensor_buffer)(LiteRtTensorBuffer buffer);
  // litert_tensor_buffer.h: LiteRtClearTensorBuffer
  LiteRtStatus (*litert_clear_tensor_buffer)(LiteRtTensorBuffer buffer);
  // litert_tensor_buffer.h: LiteRtDestroyTensorBuffer
  void (*litert_destroy_tensor_buffer)(LiteRtTensorBuffer buffer);

  //
  // LiteRtEvent
  //
  // litert_event.h: LiteRtCreateEventFromSyncFenceFd
  LiteRtStatus (*litert_create_event_from_sync_fence_fd)(LiteRtEnvironment env,
                                                         int sync_fence_fd,
                                                         bool owns_fd,
                                                         LiteRtEvent* event);
  // litert_event.h: LiteRtCreateEventFromOpenClEvent
  LiteRtStatus (*litert_create_event_from_opencl_event)(LiteRtEnvironment env,
                                                        LiteRtClEvent cl_event,
                                                        LiteRtEvent* event);
  // litert_event.h: LiteRtCreateEventFromEglSyncFence
  LiteRtStatus (*litert_create_event_from_egl_sync_fence)(
      LiteRtEnvironment env, LiteRtEglSyncKhr egl_sync, LiteRtEvent* event);
  // litert_event.h: LiteRtCreateManagedEvent
  LiteRtStatus (*litert_create_managed_event)(LiteRtEnvironment env,
                                              LiteRtEventType type,
                                              LiteRtEvent* event);
  // litert_event.h: LiteRtSetCustomEvent
  LiteRtStatus (*litert_set_custom_event)(LiteRtEvent event,
                                          LiteRtCustomEvent custom_event);
  // litert_event.h: LiteRtGetEventEventType
  LiteRtStatus (*litert_get_event_event_type)(LiteRtEvent event,
                                              LiteRtEventType* type);
  // litert_event.h: LiteRtGetEventSyncFenceFd
  LiteRtStatus (*litert_get_event_sync_fence_fd)(LiteRtEvent event,
                                                 int* sync_fence_fd);
  // litert_event.h: LiteRtGetEventOpenClEvent
  LiteRtStatus (*litert_get_event_opencl_event)(LiteRtEvent event,
                                                LiteRtClEvent* cl_event);
  // litert_event.h: LiteRtGetEventEglSync
  LiteRtStatus (*litert_get_event_egl_sync)(LiteRtEvent event,
                                            LiteRtEglSyncKhr* egl_sync);
  // litert_event.h: LiteRtGetEventCustomNativeEvent
  LiteRtStatus (*litert_get_event_custom_native_event)(LiteRtEvent event,
                                                       void** native);
  // litert_event.h: LiteRtWaitEvent
  LiteRtStatus (*litert_wait_event)(LiteRtEvent event, int64_t timeout_in_ms);
  // litert_event.h: LiteRtSignalEvent
  LiteRtStatus (*litert_signal_event)(LiteRtEvent event);
  // litert_event.h: LiteRtIsEventSignaled
  LiteRtStatus (*litert_is_event_signaled)(LiteRtEvent event,
                                           bool* is_signaled);
  // litert_event.h: LiteRtDupFdEvent
  LiteRtStatus (*litert_dup_fd_event)(LiteRtEvent event, int* dup_fd);
  // litert_event.h: LiteRtDestroyEvent
  void (*litert_destroy_event)(LiteRtEvent event);

  //
  // LiteRtLayout
  //
  // litert_layout.h: LiteRtGetNumLayoutElements
  LiteRtStatus (*litert_get_num_layout_elements)(const LiteRtLayout* layout,
                                                 size_t* num_elements);
  // litert_layout.h: LiteRtIsSameLayout
  LiteRtStatus (*litert_is_same_layout)(const LiteRtLayout* layout1,
                                        const LiteRtLayout* layout2,
                                        bool* result);

  //
  // LiteRtMetrics
  //
  // litert_metrics.h: LiteRtCreateMetrics
  LiteRtStatus (*litert_create_metrics)(LiteRtMetrics* metrics);
  // litert_metrics.h: LiteRtGetNumMetrics
  LiteRtStatus (*litert_get_num_metrics)(LiteRtMetrics metrics,
                                         int* num_metrics);
  // litert_metrics.h: LiteRtGetMetric
  LiteRtStatus (*litert_get_metric)(LiteRtMetrics metrics, int metric_index,
                                    LiteRtMetric* metric);
  // litert_metrics.h: LiteRtDestroyMetrics
  void (*litert_destroy_metrics)(LiteRtMetrics metrics);

  //
  // LiteRtOpaqueOptions
  //
  // litert_opaque_options.h: LiteRtCreateOpaqueOptions
  LiteRtStatus (*litert_create_opaque_options)(
      const char* payload_identifier, void* payload_data,
      void (*payload_destructor)(void* payload_data),
      LiteRtOpaqueOptions* options);
  // litert_opaque_options.h: LiteRtDestroyOpaqueOptions
  void (*litert_destroy_opaque_options)(LiteRtOpaqueOptions options);
  // litert_opaque_options.h: LiteRtGetOpaqueOptionsIdentifier
  LiteRtStatus (*litert_get_opaque_options_identifier)(
      LiteRtOpaqueOptions options, const char** payload_identifier);
  // litert_opaque_options.h: LiteRtGetOpaqueOptionsData
  LiteRtStatus (*litert_get_opaque_options_data)(LiteRtOpaqueOptions options,
                                                 void** payload_data);
  // litert_opaque_options.h: LiteRtFindOpaqueOptionsData
  LiteRtStatus (*litert_find_opaque_options_data)(
      LiteRtOpaqueOptions options, const char* payload_identifier,
      void** payload_data);
  // litert_opaque_options.h: LiteRtGetNextOpaqueOptions
  LiteRtStatus (*litert_get_next_opaque_options)(LiteRtOpaqueOptions* options);
  // litert_opaque_options.h: LiteRtAppendOpaqueOptions
  LiteRtStatus (*litert_append_opaque_options)(
      LiteRtOpaqueOptions* options, LiteRtOpaqueOptions appended_options);
  // litert_opaque_options.h: LiteRtPopOpaqueOptions
  LiteRtStatus (*litert_pop_opaque_options)(LiteRtOpaqueOptions* options);
  // litert_opaque_options.h: LiteRtSetOpaqueOptionsHash
  LiteRtStatus (*litert_set_opaque_options_hash)(
      LiteRtOpaqueOptions options,
      LiteRtOpaqueOptionsHashFunc payload_hash_func);
  // litert_opaque_options.h: LiteRtGetOpaqueOptionsHash
  LiteRtStatus (*litert_get_opaque_options_hash)(LiteRtOpaqueOptions options,
                                                 uint64_t* hash);

  //
  // LiteRtOptions
  //
  // litert_options.h: LiteRtCreateOptions
  LiteRtStatus (*litert_create_options)(LiteRtOptions* options);
  // litert_options.h: LiteRtDestroyOptions
  void (*litert_destroy_options)(LiteRtOptions options);
  // litert_options.h: LiteRtSetOptionsHardwareAccelerators
  LiteRtStatus (*litert_set_options_hardware_accelerators)(
      LiteRtOptions options, LiteRtHwAcceleratorSet hardware_accelerators);
  // litert_options.h: LiteRtGetOptionsHardwareAccelerators
  LiteRtStatus (*litert_get_options_hardware_accelerators)(
      LiteRtOptions options, LiteRtHwAcceleratorSet* hardware_accelerators);
  // litert_options.h: LiteRtAddOpaqueOptions
  LiteRtStatus (*litert_add_opaque_options)(LiteRtOptions options,
                                            LiteRtOpaqueOptions opaque_options);
  // litert_options.h: LiteRtGetOpaqueOptions
  LiteRtStatus (*litert_get_opaque_options)(
      LiteRtOptions options, LiteRtOpaqueOptions* opaque_options);
  // litert_options.h: LiteRtAddCustomOpKernelOption
  LiteRtStatus (*litert_add_custom_op_kernel_option)(
      LiteRtOptions options, const char* custom_op_name, int custom_op_version,
      const LiteRtCustomOpKernel* custom_op_kernel,
      void* custom_op_kernel_user_data);
  // litert_options.h: LiteRtAddExternalTensorBinding
  LiteRtStatus (*litert_add_external_tensor_binding)(LiteRtOptions options,
                                                     const char* signature_name,
                                                     const char* tensor_name,
                                                     void* data,
                                                     int size_bytes);

  //
  // LiteRtProfiler
  //
  // litert_profiler.h: LiteRtCreateProfiler
  LiteRtStatus (*litert_create_profiler)(int size, LiteRtProfiler* profiler);
  // litert_profiler.h: LiteRtDestroyProfiler
  void (*litert_destroy_profiler)(LiteRtProfiler profiler);
  // litert_profiler.h: LiteRtStartProfiler
  LiteRtStatus (*litert_start_profiler)(LiteRtProfiler profiler);
  // litert_profiler.h: LiteRtStopProfiler
  LiteRtStatus (*litert_stop_profiler)(LiteRtProfiler profiler);
  // litert_profiler.h: LiteRtResetProfiler
  LiteRtStatus (*litert_reset_profiler)(LiteRtProfiler profiler);
  // litert_profiler.h: LiteRtSetProfilerCurrentEventSource
  LiteRtStatus (*litert_set_profiler_current_event_source)(
      LiteRtProfiler profiler, ProfiledEventSource event_source);
  // litert_profiler.h: LiteRtGetNumProfilerEvents
  LiteRtStatus (*litert_get_num_profiler_events)(LiteRtProfiler profiler,
                                                 int* num_events);
  // litert_profiler.h: LiteRtGetProfilerEvents
  LiteRtStatus (*litert_get_profiler_events)(LiteRtProfiler profiler,
                                             int num_events,
                                             ProfiledEventData* events);
  // litert_profiler.h: LiteRtGetProfileSummary
  LiteRtStatus (*litert_get_profile_summary)(LiteRtProfiler profiler,
                                             LiteRtCompiledModel compiled_model,
                                             const char** summary);
} LiteRtRuntimeCApiStruct;

}  // extern "C"

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_RUNTIME_C_API_H_
