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

// This file is stub of bindgen generated file that is used only to generate
// documentation. For example, on cargo.io when a new version of the crate is published.

pub type LiteRtCompiledModel = *mut ::std::ffi::c_void;

pub type LiteRtEnvironment = *mut ::std::ffi::c_void;

pub type LiteRtTensor = *mut ::std::ffi::c_void;

pub type LiteRtSubgraph = *mut ::std::ffi::c_void;

pub type LiteRtSignature = *mut ::std::ffi::c_void;

pub type LiteRtModel = *mut ::std::ffi::c_void;

pub type LiteRtOptions = *mut ::std::ffi::c_void;

pub type LiteRtOption = *mut ::std::ffi::c_void;

pub type LiteRtProfiler = *mut ::std::ffi::c_void;

pub type LiteRtMetrics = *mut ::std::ffi::c_void;

pub type LiteRtTensorBuffer = *mut ::std::ffi::c_void;

pub type LiteRtTensorBufferRequirements = *mut ::std::ffi::c_void;

pub type LiteRtStatus = ::std::os::raw::c_uint;

pub type LiteRtHwAccelerators = ::std::os::raw::c_int;

pub type LiteRtTensorBufferLockMode = ::std::os::raw::c_uint;

pub type LiteRtHwAcceleratorSet = ::std::os::raw::c_int;

pub type LiteRtParamIndex = usize;

pub type LiteRtAnyType = ::std::os::raw::c_uint;

pub type LiteRtElementType = ::std::os::raw::c_uint;

pub type LiteRtRankedTensorType = ::std::os::raw::c_uint;

pub type LiteRtUnrankedTensorType = ::std::os::raw::c_uint;

pub type LiteRtTensorTypeId = ::std::os::raw::c_uint;

pub type LiteRtTensorBufferType = ::std::os::raw::c_uint;

pub type LiteRtEnvOption = *mut ::std::ffi::c_void;

pub type LiteRtEnvOptionTag = ::std::os::raw::c_uint;

pub const LiteRtStatus_kLiteRtStatusOk: LiteRtStatus = 0;

pub const LiteRtStatus_kLiteRtStatusErrorInvalidArgument: LiteRtStatus = 1;

pub const LiteRtStatus_kLiteRtStatusErrorMemoryAllocationFailure: LiteRtStatus = 2;

pub const LiteRtStatus_kLiteRtStatusErrorRuntimeFailure: LiteRtStatus = 3;

pub const LiteRtStatus_kLiteRtStatusErrorMissingInputTensor: LiteRtStatus = 4;

pub const LiteRtStatus_kLiteRtStatusErrorUnsupported: LiteRtStatus = 5;

pub const LiteRtStatus_kLiteRtStatusErrorNotFound: LiteRtStatus = 6;

pub const LiteRtStatus_kLiteRtStatusErrorTimeoutExpired: LiteRtStatus = 7;

pub const LiteRtStatus_kLiteRtStatusErrorWrongVersion: LiteRtStatus = 8;

pub const LiteRtStatus_kLiteRtStatusErrorUnknown: LiteRtStatus = 9;

pub const LiteRtStatus_kLiteRtStatusErrorAlreadyExists: LiteRtStatus = 10;

pub const LiteRtStatus_kLiteRtStatusCancelled: LiteRtStatus = 100;

pub const LiteRtStatus_kLiteRtStatusErrorFileIO: LiteRtStatus = 500;

pub const LiteRtStatus_kLiteRtStatusErrorInvalidFlatbuffer: LiteRtStatus = 501;

pub const LiteRtStatus_kLiteRtStatusErrorDynamicLoading: LiteRtStatus = 502;

pub const LiteRtStatus_kLiteRtStatusErrorSerialization: LiteRtStatus = 503;

pub const LiteRtStatus_kLiteRtStatusErrorCompilation: LiteRtStatus = 504;

pub const LiteRtStatus_kLiteRtStatusErrorIndexOOB: LiteRtStatus = 1000;

pub const LiteRtStatus_kLiteRtStatusErrorInvalidIrType: LiteRtStatus = 1001;

pub const LiteRtStatus_kLiteRtStatusErrorInvalidGraphInvariant: LiteRtStatus = 1002;

pub const LiteRtStatus_kLiteRtStatusErrorGraphModification: LiteRtStatus = 1003;

pub const LiteRtStatus_kLiteRtStatusErrorInvalidToolConfig: LiteRtStatus = 1500;

pub const LiteRtStatus_kLiteRtStatusLegalizeNoMatch: LiteRtStatus = 2000;

pub const LiteRtStatus_kLiteRtStatusErrorInvalidLegalization: LiteRtStatus = 2001;

pub const LiteRtStatus_kLiteRtStatusPatternNoMatch: LiteRtStatus = 3000;

pub const LiteRtStatus_kLiteRtStatusInvalidTransformation: LiteRtStatus = 3001;

pub const LiteRtStatus_kLiteRtStatusErrorUnsupportedRuntimeVersion: LiteRtStatus = 4000;

pub const LiteRtStatus_kLiteRtStatusErrorUnsupportedCompilerVersion: LiteRtStatus = 4001;

pub const LiteRtStatus_kLiteRtStatusErrorIncompatibleByteCodeVersion: LiteRtStatus = 4002;

pub const LiteRtHwAccelerators_kLiteRtHwAcceleratorNone: LiteRtHwAccelerators = 0;

pub const LiteRtHwAccelerators_kLiteRtHwAcceleratorCpu: LiteRtHwAccelerators = 1;

pub const LiteRtHwAccelerators_kLiteRtHwAcceleratorGpu: LiteRtHwAccelerators = 2;

pub const LiteRtHwAccelerators_kLiteRtHwAcceleratorNpu: LiteRtHwAccelerators = 4;

pub const LiteRtElementType_kLiteRtElementTypeNone: LiteRtElementType = 0;

pub const LiteRtElementType_kLiteRtElementTypeBool: LiteRtElementType = 6;

pub const LiteRtElementType_kLiteRtElementTypeInt2: LiteRtElementType = 20;

pub const LiteRtElementType_kLiteRtElementTypeInt4: LiteRtElementType = 18;

pub const LiteRtElementType_kLiteRtElementTypeInt8: LiteRtElementType = 9;

pub const LiteRtElementType_kLiteRtElementTypeInt16: LiteRtElementType = 7;

pub const LiteRtElementType_kLiteRtElementTypeInt32: LiteRtElementType = 2;

pub const LiteRtElementType_kLiteRtElementTypeInt64: LiteRtElementType = 4;

pub const LiteRtElementType_kLiteRtElementTypeUInt8: LiteRtElementType = 3;

pub const LiteRtElementType_kLiteRtElementTypeUInt16: LiteRtElementType = 17;

pub const LiteRtElementType_kLiteRtElementTypeUInt32: LiteRtElementType = 16;

pub const LiteRtElementType_kLiteRtElementTypeUInt64: LiteRtElementType = 13;

pub const LiteRtElementType_kLiteRtElementTypeFloat16: LiteRtElementType = 10;

pub const LiteRtElementType_kLiteRtElementTypeBFloat16: LiteRtElementType = 19;

pub const LiteRtElementType_kLiteRtElementTypeFloat32: LiteRtElementType = 1;

pub const LiteRtElementType_kLiteRtElementTypeFloat64: LiteRtElementType = 11;

pub const LiteRtElementType_kLiteRtElementTypeComplex64: LiteRtElementType = 8;

pub const LiteRtElementType_kLiteRtElementTypeComplex128: LiteRtElementType = 12;

pub const LiteRtElementType_kLiteRtElementTypeTfResource: LiteRtElementType = 14;

pub const LiteRtElementType_kLiteRtElementTypeTfString: LiteRtElementType = 5;

pub const LiteRtElementType_kLiteRtElementTypeTfVariant: LiteRtElementType = 15;

pub const LiteRtTensorTypeId_kLiteRtRankedTensorType: LiteRtTensorTypeId = 0;

pub const LiteRtTensorTypeId_kLiteRtUnrankedTensorType: LiteRtTensorTypeId = 1;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeUnknown: LiteRtTensorBufferType = 0;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeHostMemory: LiteRtTensorBufferType = 1;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeAhwb: LiteRtTensorBufferType = 2;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeIon: LiteRtTensorBufferType = 3;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeDmaBuf: LiteRtTensorBufferType = 4;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeFastRpc: LiteRtTensorBufferType = 5;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeGlBuffer: LiteRtTensorBufferType = 6;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeGlTexture: LiteRtTensorBufferType = 7;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClBuffer: LiteRtTensorBufferType = 10;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeOpenClTexture: LiteRtTensorBufferType = 12;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeWebGpuBuffer: LiteRtTensorBufferType = 20;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeWebGpuTexture: LiteRtTensorBufferType = 22;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeMetalBuffer: LiteRtTensorBufferType = 30;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeMetalTexture: LiteRtTensorBufferType = 32;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeVulkanBuffer: LiteRtTensorBufferType = 40;

pub const LiteRtTensorBufferType_kLiteRtTensorBufferTypeVulkanTexture: LiteRtTensorBufferType = 42;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagCompilerPluginLibraryDir: LiteRtEnvOptionTag = 0;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagDispatchLibraryDir: LiteRtEnvOptionTag = 1;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagOpenClDeviceId: LiteRtEnvOptionTag = 2;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagOpenClPlatformId: LiteRtEnvOptionTag = 3;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagOpenClContext: LiteRtEnvOptionTag = 4;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagOpenClCommandQueue: LiteRtEnvOptionTag = 5;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagEglDisplay: LiteRtEnvOptionTag = 6;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagEglContext: LiteRtEnvOptionTag = 7;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagWebGpuDevice: LiteRtEnvOptionTag = 8;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagWebGpuQueue: LiteRtEnvOptionTag = 9;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagMetalDevice: LiteRtEnvOptionTag = 10;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagMetalCommandQueue: LiteRtEnvOptionTag = 11;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagVulkanEnvironment: LiteRtEnvOptionTag = 12;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy: LiteRtEnvOptionTag = 14;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagMagicNumberConfigs: LiteRtEnvOptionTag = 16;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagMagicNumberVerifications: LiteRtEnvOptionTag = 17;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagCompilerCacheDir: LiteRtEnvOptionTag = 18;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagWebGpuInstance: LiteRtEnvOptionTag = 19;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagWebGpuProcs: LiteRtEnvOptionTag = 20;

pub const LiteRtEnvOptionTag_kLiteRtEnvOptionTagRuntimeLibraryDir: LiteRtEnvOptionTag = 22;

pub struct LiteRtLayout {
}

pub fn LiteRtGetNumLayoutElements(
        layout: *const LiteRtLayout,
        num_elements: *mut usize,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtIsSameLayout(
        layout1: *const LiteRtLayout,
        layout2: *const LiteRtLayout,
        result: *mut bool,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCreateCompiledModel(
        environment: LiteRtEnvironment,
        model: LiteRtModel,
        compilation_options: LiteRtOptions,
        compiled_model: *mut LiteRtCompiledModel,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtGetCompiledModelInputBufferRequirements(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        input_index: LiteRtParamIndex,
        buffer_requirements: *mut LiteRtTensorBufferRequirements,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtGetCompiledModelOutputBufferRequirements(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        output_index: LiteRtParamIndex,
        buffer_requirements: *mut LiteRtTensorBufferRequirements,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtGetCompiledModelInputTensorLayout(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        input_index: LiteRtParamIndex,
        layout: *mut LiteRtLayout,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtGetCompiledModelOutputTensorLayouts(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        num_layouts: usize,
        layouts: *mut LiteRtLayout,
        update_allocation: bool,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtGetCompiledModelEnvironment(
        compiled_model: LiteRtCompiledModel,
        environment: *mut LiteRtEnvironment,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtRunCompiledModel(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        num_input_buffers: usize,
        input_buffers: *mut LiteRtTensorBuffer,
        num_output_buffers: usize,
        output_buffers: *mut LiteRtTensorBuffer,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtRunCompiledModelAsync(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        num_input_buffers: usize,
        input_buffers: *mut LiteRtTensorBuffer,
        num_output_buffers: usize,
        output_buffers: *mut LiteRtTensorBuffer,
        async_: *mut bool,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtSetCompiledModelCancellationFunction(
        compiled_model: LiteRtCompiledModel,
        data: *mut ::std::os::raw::c_void,
        check_cancelled_func: ::std::option::Option<
            fn(arg1: *mut ::std::os::raw::c_void) -> bool,
        >,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtDestroyCompiledModel(compiled_model: LiteRtCompiledModel) { unimplemented!() }

pub fn LiteRtCompiledModelStartMetricsCollection(
        compiled_model: LiteRtCompiledModel,
        detail_level: ::std::os::raw::c_int,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCompiledModelStopMetricsCollection(
        compiled_model: LiteRtCompiledModel,
        metrics: LiteRtMetrics,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCompiledModelIsFullyAccelerated(
        compiled_model: LiteRtCompiledModel,
        fully_accelerated: *mut bool,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCompiledModelGetProfiler(
        compiled_model: LiteRtCompiledModel,
        profiler: *mut LiteRtProfiler,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCompiledModelResizeInputTensor(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        input_index: LiteRtParamIndex,
        dims: *const ::std::os::raw::c_int,
        dims_size: usize,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCompiledModelResizeInputTensorNonStrict(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        input_index: LiteRtParamIndex,
        dims: *const ::std::os::raw::c_int,
        dims_size: usize,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCompiledModelSetDispatchAnnotation(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        key: *const ::std::os::raw::c_char,
        value: *const ::std::os::raw::c_char,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCompiledModelGetDispatchAnnotation(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        key: *const ::std::os::raw::c_char,
        value: *mut *const ::std::os::raw::c_char,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCompiledModelRemoveDispatchAnnotation(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        key: *const ::std::os::raw::c_char,
    ) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCompiledModelClearErrors(compiled_model: LiteRtCompiledModel) -> LiteRtStatus { unimplemented!() }

pub fn LiteRtCompiledModelGetErrorMessages(
        compiled_model: LiteRtCompiledModel,
        error_messages: *mut *mut ::std::os::raw::c_char,
    ) -> LiteRtStatus { unimplemented!() }

pub const LiteRtAnyType_kLiteRtAnyTypeNone: LiteRtAnyType = 0;

pub const LiteRtAnyType_kLiteRtAnyTypeBool: LiteRtAnyType = 1;

pub const LiteRtAnyType_kLiteRtAnyTypeInt: LiteRtAnyType = 2;

pub const LiteRtAnyType_kLiteRtAnyTypeReal: LiteRtAnyType = 3;

pub const LiteRtAnyType_kLiteRtAnyTypeString: LiteRtAnyType = 8;

pub const LiteRtAnyType_kLiteRtAnyTypeVoidPtr: LiteRtAnyType = 9;

pub struct LiteRtAny {
}

pub fn LiteRtGetStatusString(status: LiteRtStatus)-> *const ::std::os::raw::c_char{ unimplemented!() }

pub fn LiteRtGetTensorName(tensor: LiteRtTensor, name: *mut *const ::std::os::raw::c_char,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtCreateEnvironment(num_options: ::std::os::raw::c_int, options: *const LiteRtEnvOption, environment: *mut LiteRtEnvironment,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtDestroyEnvironment(environment: LiteRtEnvironment){ unimplemented!() }

pub fn LiteRtGetTensorTypeId(tensor: LiteRtTensor, type_id: *mut LiteRtTensorTypeId,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetUnrankedTensorType(tensor: LiteRtTensor, unranked_tensor_type: *mut LiteRtUnrankedTensorType,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetRankedTensorType(tensor: LiteRtTensor, ranked_tensor_type: *mut LiteRtRankedTensorType,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetNumSubgraphInputs(subgraph: LiteRtSubgraph, num_inputs: *mut LiteRtParamIndex,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetSubgraphInput(subgraph: LiteRtSubgraph, input_index: LiteRtParamIndex, input: *mut LiteRtTensor,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetNumSubgraphOutputs(subgraph: LiteRtSubgraph, num_outputs: *mut LiteRtParamIndex,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetSubgraphOutput(subgraph: LiteRtSubgraph, output_index: LiteRtParamIndex, output: *mut LiteRtTensor,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetSignatureKey(signature: LiteRtSignature, signature_key: *mut *const ::std::os::raw::c_char,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetSignatureSubgraph(signature: LiteRtSignature, subgraph: *mut LiteRtSubgraph,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetNumSignatureInputs(signature: LiteRtSignature, num_inputs: *mut LiteRtParamIndex,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetSignatureInputName(signature: LiteRtSignature, input_idx: LiteRtParamIndex, input_name: *mut *const ::std::os::raw::c_char,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetNumSignatureOutputs(signature: LiteRtSignature, num_outputs: *mut LiteRtParamIndex,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetSignatureOutputName(signature: LiteRtSignature, output_idx: LiteRtParamIndex, output_name: *mut *const ::std::os::raw::c_char,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtCreateModelFromFile(filename: *const ::std::os::raw::c_char, model: *mut LiteRtModel,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtCreateModelFromBuffer(buffer_addr: *const ::std::os::raw::c_void, buffer_size: usize, model: *mut LiteRtModel,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetNumModelSubgraphs(model: LiteRtModel, num_subgraphs: *mut LiteRtParamIndex,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetNumModelSignatures(model: LiteRtModel, num_signatures: *mut LiteRtParamIndex,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetModelSignature(model: LiteRtModel, signature_index: LiteRtParamIndex, signature: *mut LiteRtSignature,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtDestroyModel(model: LiteRtModel){ unimplemented!() }

pub fn LiteRtCreateOptions(options: *mut LiteRtOptions)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtDestroyOptions(options: LiteRtOptions){ unimplemented!() }

pub fn LiteRtSetOptionsHardwareAccelerators(options: LiteRtOptions, hardware_accelerators: LiteRtHwAcceleratorSet,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtCreateManagedTensorBuffer(env: LiteRtEnvironment, buffer_type: LiteRtTensorBufferType, tensor_type: *const LiteRtRankedTensorType, buffer_size: usize, buffer: *mut LiteRtTensorBuffer,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetTensorBufferPackedSize(tensor_buffer: LiteRtTensorBuffer, size: *mut usize,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtLockTensorBuffer(tensor_buffer: LiteRtTensorBuffer, host_mem_addr: *mut *mut ::std::os::raw::c_void, lock_mode: LiteRtTensorBufferLockMode,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtUnlockTensorBuffer(buffer: LiteRtTensorBuffer)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtDestroyTensorBuffer(buffer: LiteRtTensorBuffer){ unimplemented!() }

pub fn LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(requirements: LiteRtTensorBufferRequirements, num_types: *mut ::std::os::raw::c_int,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(requirements: LiteRtTensorBufferRequirements, type_index: ::std::os::raw::c_int, type_: *mut LiteRtTensorBufferType,)-> LiteRtStatus{ unimplemented!() }

pub fn LiteRtGetTensorBufferRequirementsBufferSize(requirements: LiteRtTensorBufferRequirements, buffer_size: *mut usize,)-> LiteRtStatus{ unimplemented!() }
