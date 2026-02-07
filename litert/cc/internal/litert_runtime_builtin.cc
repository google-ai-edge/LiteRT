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

#include "litert/cc/internal/litert_runtime_builtin.h"

#include "litert/c/internal/litert_runtime_c_api.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_profiler.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

// The implementation of the LiteRtRuntimeCApiStruct with the builtin runtime.
const LiteRtRuntimeCApiStruct kLiteRtRuntimeBuiltin = {
    // LiteRtEnvironment
    .litert_create_environment = LiteRtCreateEnvironment,
    .litert_destroy_environment = LiteRtDestroyEnvironment,
    .litert_get_environment_options = LiteRtGetEnvironmentOptions,
    .litert_add_environment_options = LiteRtAddEnvironmentOptions,
    .litert_gpu_environment_create = LiteRtGpuEnvironmentCreate,
    .litert_environment_supports_cl_gl_interop =
        LiteRtEnvironmentSupportsClGlInterop,
    .litert_environment_supports_ahwb_cl_interop =
        LiteRtEnvironmentSupportsAhwbClInterop,
    .litert_environment_supports_ahwb_gl_interop =
        LiteRtEnvironmentSupportsAhwbGlInterop,
    .litert_environment_has_gpu_environment =
        LiteRtEnvironmentHasGpuEnvironment,
    // LiteRtEnvironmentOptions
    .litert_get_environment_options_value = LiteRtGetEnvironmentOptionsValue,
    .litert_set_environment_options_value = LiteRtSetEnvironmentOptionsValue,
    // LiteRtTensor
    .litert_get_tensor_name = LiteRtGetTensorName,
    .litert_get_tensor_index = LiteRtGetTensorIndex,
    .litert_get_tensor_type_id = LiteRtGetTensorTypeId,
    .litert_get_unranked_tensor_type = LiteRtGetUnrankedTensorType,
    .litert_get_ranked_tensor_type = LiteRtGetRankedTensorType,
    .litert_get_quantization_type_id = LiteRtGetQuantizationTypeId,
    .litert_get_per_tensor_quantization = LiteRtGetPerTensorQuantization,
    .litert_get_per_channel_quantization = LiteRtGetPerChannelQuantization,
    .litert_get_num_tensor_uses = LiteRtGetNumTensorUses,
    .litert_get_tensor_use = LiteRtGetTensorUse,
    .litert_get_tensor_defining_op = LiteRtGetTensorDefiningOp,
    .litert_get_tensor_weights = LiteRtGetTensorWeights,
    // LiteRtSubgraph
    .litert_get_num_subgraph_inputs = LiteRtGetNumSubgraphInputs,
    .litert_get_subgraph_input = LiteRtGetSubgraphInput,
    .litert_get_num_subgraph_outputs = LiteRtGetNumSubgraphOutputs,
    .litert_get_subgraph_output = LiteRtGetSubgraphOutput,
    .litert_get_num_subgraph_ops = LiteRtGetNumSubgraphOps,
    .litert_get_subgraph_op = LiteRtGetSubgraphOp,
    // LiteRtSignature
    .litert_get_default_signature_key = LiteRtGetDefaultSignatureKey,
    .litert_get_signature_key = LiteRtGetSignatureKey,
    .litert_get_signature_subgraph = LiteRtGetSignatureSubgraph,
    .litert_get_num_signature_inputs = LiteRtGetNumSignatureInputs,
    .litert_get_signature_input_name = LiteRtGetSignatureInputName,
    .litert_get_signature_input_tensor = LiteRtGetSignatureInputTensor,
    .litert_get_signature_input_tensor_by_index =
        LiteRtGetSignatureInputTensorByIndex,
    .litert_get_num_signature_outputs = LiteRtGetNumSignatureOutputs,
    .litert_get_signature_output_name = LiteRtGetSignatureOutputName,
    .litert_get_signature_output_tensor = LiteRtGetSignatureOutputTensor,
    .litert_get_signature_output_tensor_by_index =
        LiteRtGetSignatureOutputTensorByIndex,
    // LiteRtModel
    .litert_create_model_from_file = LiteRtCreateModelFromFile,
    .litert_create_model_from_buffer = LiteRtCreateModelFromBuffer,
    .litert_get_model_metadata = LiteRtGetModelMetadata,
    .litert_add_model_metadata = LiteRtAddModelMetadata,
    .litert_get_main_model_subgraph_index = LiteRtGetMainModelSubgraphIndex,
    .litert_get_num_model_subgraphs = LiteRtGetNumModelSubgraphs,
    .litert_get_model_subgraph = LiteRtGetModelSubgraph,
    .litert_get_num_model_signatures = LiteRtGetNumModelSignatures,
    .litert_get_model_signature = LiteRtGetModelSignature,
    .litert_destroy_model = LiteRtDestroyModel,
    .litert_push_op = LiteRtPushOp,
    .litert_serialize_model_with_signatures =
        LiteRtSerializeModelWithSignatures,
    .litert_serialize_model = LiteRtSerializeModel,
    // LiteRtCompiledModel
    .litert_create_compiled_model = LiteRtCreateCompiledModel,
    .litert_get_compiled_model_input_buffer_requirements =
        LiteRtGetCompiledModelInputBufferRequirements,
    .litert_get_compiled_model_output_buffer_requirements =
        LiteRtGetCompiledModelOutputBufferRequirements,
    .litert_get_compiled_model_input_tensor_layout =
        LiteRtGetCompiledModelInputTensorLayout,
    .litert_get_compiled_model_output_tensor_layouts =
        LiteRtGetCompiledModelOutputTensorLayouts,
    .litert_get_compiled_model_environment = LiteRtGetCompiledModelEnvironment,
    .litert_run_compiled_model = LiteRtRunCompiledModel,
    .litert_run_compiled_model_async = LiteRtRunCompiledModelAsync,
    .litert_set_compiled_model_cancellation_function =
        LiteRtSetCompiledModelCancellationFunction,
    .litert_destroy_compiled_model = LiteRtDestroyCompiledModel,
    .litert_compiled_model_start_metrics_collection =
        LiteRtCompiledModelStartMetricsCollection,
    .litert_compiled_model_stop_metrics_collection =
        LiteRtCompiledModelStopMetricsCollection,
    .litert_compiled_model_is_fully_accelerated =
        LiteRtCompiledModelIsFullyAccelerated,
    .litert_compiled_model_get_profiler = LiteRtCompiledModelGetProfiler,
    .litert_compiled_model_resize_input_tensor =
        LiteRtCompiledModelResizeInputTensor,
    .litert_compiled_model_resize_input_tensor_non_strict =
        LiteRtCompiledModelResizeInputTensorNonStrict,
    .litert_compiled_model_set_dispatch_annotation =
        LiteRtCompiledModelSetDispatchAnnotation,
    .litert_compiled_model_get_dispatch_annotation =
        LiteRtCompiledModelGetDispatchAnnotation,
    .litert_compiled_model_remove_dispatch_annotation =
        LiteRtCompiledModelRemoveDispatchAnnotation,
    .litert_compiled_model_report_error = LiteRtCompiledModelReportError,
    .litert_compiled_model_clear_errors = LiteRtCompiledModelClearErrors,
    .litert_compiled_model_get_error_messages =
        LiteRtCompiledModelGetErrorMessages,
    // LiteRtTensorBufferRequirements
    .litert_create_tensor_buffer_requirements =
        LiteRtCreateTensorBufferRequirements,
    .litert_create_tensor_buffer_requirements_with_alignment =
        LiteRtCreateTensorBufferRequirementsWithAlignment,
    .litert_get_num_tensor_buffer_requirements_supported_buffer_types =
        LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes,
    .litert_get_tensor_buffer_requirements_supported_tensor_buffer_type =
        LiteRtGetTensorBufferRequirementsSupportedTensorBufferType,
    .litert_get_tensor_buffer_requirements_buffer_size =
        LiteRtGetTensorBufferRequirementsBufferSize,
    .litert_get_tensor_buffer_requirements_strides =
        LiteRtGetTensorBufferRequirementsStrides,
    .litert_get_tensor_buffer_requirements_alignment =
        LiteRtGetTensorBufferRequirementsAlignment,
    .litert_join_tensor_buffer_requirements =
        LiteRtJoinTensorBufferRequirements,
    .litert_destroy_tensor_buffer_requirements =
        LiteRtDestroyTensorBufferRequirements,
    // LiteRtTensorBuffer
    .litert_create_tensor_buffer_from_host_memory =
        LiteRtCreateTensorBufferFromHostMemory,
    .litert_get_tensor_buffer_host_memory = LiteRtGetTensorBufferHostMemory,
#if LITERT_HAS_AHWB_SUPPORT
    .litert_create_tensor_buffer_from_ahwb = LiteRtCreateTensorBufferFromAhwb,
    .litert_get_tensor_buffer_ahwb = LiteRtGetTensorBufferAhwb,
#endif  // LITERT_HAS_AHWB_SUPPORT
#if LITERT_HAS_ION_SUPPORT
    .litert_create_tensor_buffer_from_ion_buffer =
        LiteRtCreateTensorBufferFromIonBuffer,
    .litert_get_tensor_buffer_ion_buffer = LiteRtGetTensorBufferIonBuffer,
#endif  // LITERT_HAS_ION_SUPPORT
#if LITERT_HAS_DMABUF_SUPPORT
    .litert_create_tensor_buffer_from_dma_buf_buffer =
        LiteRtCreateTensorBufferFromDmaBufBuffer,
    .litert_get_tensor_buffer_dma_buf_buffer =
        LiteRtGetTensorBufferDmaBufBuffer,
#endif  // LITERT_HAS_DMABUF_SUPPORT
#if LITERT_HAS_FASTRPC_SUPPORT
    .litert_create_tensor_buffer_from_fast_rpc_buffer =
        LiteRtCreateTensorBufferFromFastRpcBuffer,
    .litert_get_tensor_buffer_fast_rpc_buffer =
        LiteRtGetTensorBufferFastRpcBuffer,
#endif  // LITERT_HAS_FASTRPC_SUPPORT
#if LITERT_HAS_OPENCL_SUPPORT
    .litert_create_tensor_buffer_from_opencl_memory =
        LiteRtCreateTensorBufferFromOpenClMemory,
    .litert_get_tensor_buffer_opencl_memory = LiteRtGetTensorBufferOpenClMemory,
#endif  // LITERT_HAS_OPENCL_SUPPORT
    .litert_get_tensor_buffer_custom_tensor_buffer_handle =
        LiteRtGetTensorBufferCustomTensorBufferHandle,
    .litert_create_tensor_buffer_from_gl_buffer =
        LiteRtCreateTensorBufferFromGlBuffer,
    .litert_get_tensor_buffer_gl_buffer = LiteRtGetTensorBufferGlBuffer,
    .litert_create_tensor_buffer_from_gl_texture =
        LiteRtCreateTensorBufferFromGlTexture,
    .litert_get_tensor_buffer_gl_texture = LiteRtGetTensorBufferGlTexture,
#if LITERT_HAS_WEBGPU_SUPPORT
    .litert_create_tensor_buffer_from_web_gpu_buffer =
        LiteRtCreateTensorBufferFromWebGpuBuffer,
    .litert_get_tensor_buffer_web_gpu_buffer =
        LiteRtGetTensorBufferWebGpuBuffer,
    .litert_create_tensor_buffer_from_web_gpu_texture =
        LiteRtCreateTensorBufferFromWebGpuTexture,
#endif  // LITERT_HAS_WEBGPU_SUPPORT
#if LITERT_HAS_METAL_SUPPORT
    .litert_create_tensor_buffer_from_metal_memory =
        LiteRtCreateTensorBufferFromMetalMemory,
    .litert_get_tensor_buffer_metal_memory = LiteRtGetTensorBufferMetalMemory,
#endif  // LITERT_HAS_METAL_SUPPORT
#if LITERT_HAS_VULKAN_SUPPORT
    .litert_get_tensor_buffer_vulkan_memory = LiteRtGetTensorBufferVulkanMemory,
#endif  // LITERT_HAS_VULKAN_SUPPORT
    .litert_create_managed_tensor_buffer = LiteRtCreateManagedTensorBuffer,
    .litert_create_managed_tensor_buffer_from_requirements =
        LiteRtCreateManagedTensorBufferFromRequirements,
    .litert_duplicate_tensor_buffer = LiteRtDuplicateTensorBuffer,
    .litert_get_tensor_buffer_type = LiteRtGetTensorBufferType,
    .litert_get_tensor_buffer_tensor_type = LiteRtGetTensorBufferTensorType,
    .litert_get_tensor_buffer_size = LiteRtGetTensorBufferSize,
    .litert_get_tensor_buffer_packed_size = LiteRtGetTensorBufferPackedSize,
    .litert_get_tensor_buffer_offset = LiteRtGetTensorBufferOffset,
    .litert_has_tensor_buffer_event = LiteRtHasTensorBufferEvent,
    .litert_get_tensor_buffer_event = LiteRtGetTensorBufferEvent,
    .litert_set_tensor_buffer_event = LiteRtSetTensorBufferEvent,
    .litert_clear_tensor_buffer_event = LiteRtClearTensorBufferEvent,
    .litert_lock_tensor_buffer = LiteRtLockTensorBuffer,
    .litert_unlock_tensor_buffer = LiteRtUnlockTensorBuffer,
    .litert_clear_tensor_buffer = LiteRtClearTensorBuffer,
    .litert_destroy_tensor_buffer = LiteRtDestroyTensorBuffer,
    // LiteRtEvent
    .litert_create_event_from_sync_fence_fd = LiteRtCreateEventFromSyncFenceFd,
    .litert_create_event_from_opencl_event = LiteRtCreateEventFromOpenClEvent,
    .litert_create_event_from_egl_sync_fence =
        LiteRtCreateEventFromEglSyncFence,
    .litert_create_managed_event = LiteRtCreateManagedEvent,
    .litert_set_custom_event = LiteRtSetCustomEvent,
    .litert_get_event_event_type = LiteRtGetEventEventType,
    .litert_get_event_sync_fence_fd = LiteRtGetEventSyncFenceFd,
    .litert_get_event_opencl_event = LiteRtGetEventOpenClEvent,
    .litert_get_event_egl_sync = LiteRtGetEventEglSync,
    .litert_get_event_custom_native_event = LiteRtGetEventCustomNativeEvent,
    .litert_wait_event = LiteRtWaitEvent,
    .litert_signal_event = LiteRtSignalEvent,
    .litert_is_event_signaled = LiteRtIsEventSignaled,
    .litert_dup_fd_event = LiteRtDupFdEvent,
    .litert_destroy_event = LiteRtDestroyEvent,
    // LiteRtLayout
    .litert_get_num_layout_elements = LiteRtGetNumLayoutElements,
    .litert_is_same_layout = LiteRtIsSameLayout,
    // LiteRtMetrics
    .litert_create_metrics = LiteRtCreateMetrics,
    .litert_get_num_metrics = LiteRtGetNumMetrics,
    .litert_get_metric = LiteRtGetMetric,
    .litert_destroy_metrics = LiteRtDestroyMetrics,
    // LiteRtOpaqueOptions
    .litert_create_opaque_options = LiteRtCreateOpaqueOptions,
    .litert_destroy_opaque_options = LiteRtDestroyOpaqueOptions,
    .litert_get_opaque_options_identifier = LiteRtGetOpaqueOptionsIdentifier,
    .litert_get_opaque_options_data = LiteRtGetOpaqueOptionsData,
    .litert_find_opaque_options_data = LiteRtFindOpaqueOptionsData,
    .litert_get_next_opaque_options = LiteRtGetNextOpaqueOptions,
    .litert_append_opaque_options = LiteRtAppendOpaqueOptions,
    .litert_pop_opaque_options = LiteRtPopOpaqueOptions,
    .litert_set_opaque_options_hash = LiteRtSetOpaqueOptionsHash,
    .litert_get_opaque_options_hash = LiteRtGetOpaqueOptionsHash,
    // LiteRtOptions
    .litert_create_options = LiteRtCreateOptions,
    .litert_destroy_options = LiteRtDestroyOptions,
    .litert_set_options_hardware_accelerators =
        LiteRtSetOptionsHardwareAccelerators,
    .litert_get_options_hardware_accelerators =
        LiteRtGetOptionsHardwareAccelerators,
    .litert_add_opaque_options = LiteRtAddOpaqueOptions,
    .litert_get_opaque_options = LiteRtGetOpaqueOptions,
    .litert_add_custom_op_kernel_option = LiteRtAddCustomOpKernelOption,
    .litert_add_external_tensor_binding = LiteRtAddExternalTensorBinding,
    // LiteRtProfiler
    .litert_create_profiler = LiteRtCreateProfiler,
    .litert_destroy_profiler = LiteRtDestroyProfiler,
    .litert_start_profiler = LiteRtStartProfiler,
    .litert_stop_profiler = LiteRtStopProfiler,
    .litert_reset_profiler = LiteRtResetProfiler,
    .litert_set_profiler_current_event_source =
        LiteRtSetProfilerCurrentEventSource,
    .litert_get_num_profiler_events = LiteRtGetNumProfilerEvents,
    .litert_get_profiler_events = LiteRtGetProfilerEvents,
    .litert_get_profile_summary = LiteRtGetProfileSummary,
};
