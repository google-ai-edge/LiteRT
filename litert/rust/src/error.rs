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

//! Error types ([`Error`] and [`ErrorCause`]) returned by `LiteRT` Rust bindings.

#![allow(non_upper_case_globals)]

use crate::bindings::*;

use std::fmt;

/// Reason of the error. It provides information where in the binding code the error occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCause {
    /// Unknown or unspecified error cause.
    Unknown,
    // compiled_model
    /// Failed to create compilation `LiteRtOptions`.
    CreateOptions,
    /// Failed to set hardware accelerators on `LiteRtOptions`.
    SetOptionsHardwareAccelerators,
    /// Failed to compile and create `LiteRtCompiledModel`.
    CreateCompiledModel,
    /// Failed to query the `LiteRtEnvironment` of a compiled model.
    GetCompiledModelEnvironment,
    /// Failed to query input buffer requirements from a compiled model.
    GetCompiledModelInputBufferRequirements,
    /// Failed to query output buffer requirements from a compiled model.
    GetCompiledModelOutputBufferRequirements,
    /// The input or output tensor does not support any `TensorBufferType`.
    InputDoesntSupportAnyTensorBufferTypes,
    /// Failed to resize an input tensor on a compiled model.
    ResizeCompiledModelInputBuffer,
    /// Failed to execute inference on a compiled model.
    RunCompiledModel,
    /// Failed to query whether a compiled model is fully accelerated.
    IsCompiledModelFullyAccelerated,
    /// Failed to query whether a compiled model is 100% delegated to non-CPU accelerators.
    IsCompiledModelNonCpuFullyAccelerated,
    // environment
    /// Unsupported Rust type passed as an environment option value.
    NotSupportedLiteRtAnyType,
    /// Failed to create `LiteRtEnvironment`.
    CreateEnvironment,
    /// Failed to query the number of hardware accelerators in the environment.
    GetNumAccelerators,
    /// Failed to retrieve a hardware accelerator handle from the environment.
    GetAccelerator,
    /// Failed to query hardware support bitfield for an accelerator.
    GetAcceleratorHardwareSupport,
    // model
    /// Failed to query the key string of a model signature.
    GetSignatureKey,
    /// Failed to retrieve the subgraph associated with a model signature.
    GetSignatureSubgraph,
    /// Failed to query the number of inputs in a model signature.
    GetNumSignatureInputs,
    /// Failed to query the name of a signature input.
    GetSignatureInputName,
    /// Failed to retrieve an input tensor from a model signature.
    GetSignatureInputTensor,
    /// No signature input with the requested name was found.
    SignatureInputNotFound,
    /// Failed to query the number of outputs in a model signature.
    GetNumSignatureOutputs,
    /// Failed to query the name of a signature output.
    GetSignatureOutputName,
    /// Failed to retrieve an output tensor from a model signature.
    GetSignatureOutputTensor,
    /// No signature output with the requested name was found.
    SignatureOutputNotFound,
    /// Failed to look up a signature in the model.
    GetSignature,
    /// Failed to query the type ID of a tensor.
    GetTensorTypeId,
    /// Failed to query the unranked tensor type descriptor.
    GetUnrankedTensorType,
    /// Failed to query the ranked tensor type descriptor.
    GetRankedTensorType,
    /// Unrecognized `LiteRtTensorTypeId` value returned by the C API.
    InvalidTensorTypeId,
    /// Failed to query the name of a tensor.
    GetTensorName,
    /// Failed to query the number of inputs of a subgraph.
    GetNumSubgraphInputs,
    /// Failed to query the number of outputs of a subgraph.
    GetNumSubgraphOutputs,
    /// Failed to retrieve an input tensor of a subgraph.
    GetSubgraphInput,
    /// No subgraph input tensor with the requested name was found.
    SubgraphInputTensorByNameNotFound,
    /// Failed to retrieve an output tensor of a subgraph.
    GetSubgraphOutput,
    /// No subgraph output tensor with the requested name was found.
    SubgraphOutputTensorByNameNotFound,
    /// Failed to load and create a model from a file path.
    CreateModelFromFile,
    /// Failed to parse and create a model from an in-memory buffer.
    CreateModelFromBuffer,
    /// Failed to query the number of subgraphs in a model.
    GetNumModelSubgraphs,
    /// Failed to query the number of signatures in a model.
    GetNumModelSignatures,
    /// Failed to retrieve a signature from a model by index.
    GetModelSignature,
    // tensor_buffer
    /// Failed to query the byte size required by `TensorBufferRequirements`.
    GetTensorBufferRequirementsBufferSize,
    /// Failed to query the count of supported buffer types in `TensorBufferRequirements`.
    GetNumTensorBufferRequirementsSupportedBufferTypes,
    /// Failed to query a supported `TensorBufferType` from `TensorBufferRequirements`.
    GetTensorBufferRequirementsSupportedTensorBufferType,
    /// Unrecognized `LiteRtElementType` enum value returned by the C API.
    InvalidElementTypeEnumValue,
    /// Unrecognized `LiteRtTensorBufferType` enum value returned by the C API.
    InvalidTensorBufferTypeEnumValue,
    /// Failed to allocate or clear a managed `LiteRtTensorBuffer`.
    CreateManagedTensorBuffer,
    /// Failed to lock a `TensorBuffer` in read mode.
    LockTensorBufferRead,
    /// Failed to lock a `TensorBuffer` in write mode.
    LockTensorBufferWrite,
    /// Failed to query the packed byte size of a `TensorBuffer`.
    GetTensorBufferPackedSize,
    /// Attempted to lock or write to a `TensorBuffer` with an incompatible element type `T`.
    IncompatibleWriteType,
    /// The input slice passed to `TensorBuffer::write` exceeds the buffer's capacity.
    TensorBufferTooSmall,
    /// Attempted to lock or read from a `TensorBuffer` with an incompatible element type `T`.
    IncompatibleReadType,
    /// The destination slice passed to `TensorBuffer::read` is smaller than the buffer's length.
    ReadBufferTooSmall,
    /// Failed to check whether a synchronization event is attached to a `TensorBuffer`.
    HasTensorBufferEvent,
    /// Failed to retrieve the synchronization event attached to a `TensorBuffer`.
    GetTensorBufferEvent,
    /// Failed to query whether a synchronization event has signaled.
    IsEventSignaled,
    /// Failed while waiting for a synchronization event to signal.
    WaitEvent,
    /// Failed to create a managed synchronization event.
    CreateManagedEvent,
    /// Failed to attach or clear a synchronization event on a `TensorBuffer`.
    SetTensorBufferEvent,
    /// Failed to signal a synchronization event.
    SignalEvent,
    // util
    /// A C or Rust string contained invalid UTF-8 or an interior NUL byte.
    InvalidStringEncoding,
}

/// Error returned by the bindings.
///
/// It contains the reason of the error and the `LiteRtStatus` returned by the C code.
///
/// `Error` implements [`fmt::Display`], [`fmt::Debug`], and [`std::error::Error`].
///
/// ```ignore
/// match litert_result {
///     Ok(env) => { /* ... */ }
///     Err(error) => {
///         println!("Error: {:?}", error);
///     }
/// }
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Error {
    /// High-level operation in the Rust bindings where the error occurred.
    cause: ErrorCause,
    /// Raw `LiteRtStatus` code returned by the underlying `LiteRT` C API.
    litert_status: LiteRtStatus,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error: {:?}", self.cause)?;
        let status_description = self.litert_status_description();
        write!(f, " LiteRtStatus: {:?} [{}] ", self.litert_status, status_description)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for Error {}

impl Error {
    /// Creates a new [`Error`] from a binding [`ErrorCause`] and C [`LiteRtStatus`] code.
    #[must_use]
    pub(crate) fn new(cause: ErrorCause, status: LiteRtStatus) -> Self {
        Error { cause, litert_status: status }
    }

    /// Returns the reason of the error from the binding code.
    ///
    /// # Returns
    ///
    /// The [`ErrorCause`] identifying the operation that failed.
    #[must_use]
    pub fn cause(&self) -> ErrorCause {
        self.cause
    }

    /// Returns the `LiteRtStatus` returned by the C code.
    ///
    /// # Returns
    ///
    /// The raw `LiteRtStatus` code.
    #[must_use]
    pub fn litert_status(&self) -> LiteRtStatus {
        self.litert_status
    }

    /// Returns a string description of the `LiteRtStatus` returned by the C code.
    ///
    /// # Returns
    ///
    /// A human-readable `String` naming the `LiteRtStatus` constant (or `"???"` if unknown).
    #[must_use]
    pub fn litert_status_description(&self) -> String {
        // LINT.IfChange(status_codes)
        match self.litert_status {
            LiteRtStatus_kLiteRtStatusOk => "kLiteRtStatusOk",

            // Generic errors.
            LiteRtStatus_kLiteRtStatusErrorInvalidArgument => "kLiteRtStatusErrorInvalidArgument",
            LiteRtStatus_kLiteRtStatusErrorMemoryAllocationFailure => {
                "kLiteRtStatusErrorMemoryAllocationFailure"
            }
            LiteRtStatus_kLiteRtStatusErrorRuntimeFailure => "kLiteRtStatusErrorRuntimeFailure",
            LiteRtStatus_kLiteRtStatusErrorMissingInputTensor => {
                "kLiteRtStatusErrorMissingInputTensor"
            }
            LiteRtStatus_kLiteRtStatusErrorUnsupported => "kLiteRtStatusErrorUnsupported",
            LiteRtStatus_kLiteRtStatusErrorNotFound => "kLiteRtStatusErrorNotFound",
            LiteRtStatus_kLiteRtStatusErrorTimeoutExpired => "kLiteRtStatusErrorTimeoutExpired",
            LiteRtStatus_kLiteRtStatusErrorWrongVersion => "kLiteRtStatusErrorWrongVersion",
            LiteRtStatus_kLiteRtStatusErrorUnknown => "kLiteRtStatusErrorUnknown",
            LiteRtStatus_kLiteRtStatusErrorAlreadyExists => "kLiteRtStatusErrorAlreadyExists",
            LiteRtStatus_kLiteRtStatusErrorFailedPrecondition => {
                "kLiteRtStatusErrorFailedPrecondition"
            }

            // Inference progression errors.
            LiteRtStatus_kLiteRtStatusCancelled => "kLiteRtStatusCancelled",

            // File and loading related errors.
            LiteRtStatus_kLiteRtStatusErrorFileIO => "kLiteRtStatusErrorFileIO",
            LiteRtStatus_kLiteRtStatusErrorInvalidFlatbuffer => {
                "kLiteRtStatusErrorInvalidFlatbuffer"
            }
            LiteRtStatus_kLiteRtStatusErrorDynamicLoading => "kLiteRtStatusErrorDynamicLoading",
            LiteRtStatus_kLiteRtStatusErrorSerialization => "kLiteRtStatusErrorSerialization",
            LiteRtStatus_kLiteRtStatusErrorCompilation => "kLiteRtStatusErrorCompilation",

            // IR related errors.
            LiteRtStatus_kLiteRtStatusErrorIndexOOB => "kLiteRtStatusErrorIndexOOB",
            LiteRtStatus_kLiteRtStatusErrorInvalidIrType => "kLiteRtStatusErrorInvalidIrType",
            LiteRtStatus_kLiteRtStatusErrorInvalidGraphInvariant => {
                "kLiteRtStatusErrorInvalidGraphInvariant"
            }
            LiteRtStatus_kLiteRtStatusErrorGraphModification => {
                "kLiteRtStatusErrorGraphModification"
            }

            // Tool related errors.
            LiteRtStatus_kLiteRtStatusErrorInvalidToolConfig => {
                "kLiteRtStatusErrorInvalidToolConfig"
            }

            // Legalization related errors.
            LiteRtStatus_kLiteRtStatusLegalizeNoMatch => "kLiteRtStatusLegalizeNoMatch",
            LiteRtStatus_kLiteRtStatusErrorInvalidLegalization => {
                "kLiteRtStatusErrorInvalidLegalization"
            }

            // Transformation related errors.
            LiteRtStatus_kLiteRtStatusPatternNoMatch => "kLiteRtStatusPatternNoMatch",
            LiteRtStatus_kLiteRtStatusInvalidTransformation => "kLiteRtStatusInvalidTransformation",

            // Version related errors.
            LiteRtStatus_kLiteRtStatusErrorUnsupportedRuntimeVersion => {
                "kLiteRtStatusErrorUnsupportedRuntimeVersion"
            }
            LiteRtStatus_kLiteRtStatusErrorUnsupportedCompilerVersion => {
                "kLiteRtStatusErrorUnsupportedCompilerVersion"
            }
            LiteRtStatus_kLiteRtStatusErrorIncompatibleByteCodeVersion => {
                "kLiteRtStatusErrorIncompatibleByteCodeVersion"
            }

            // Shape related errors.
            LiteRtStatus_kLiteRtStatusErrorUnsupportedOpShapeInferer => {
                "kLiteRtStatusErrorUnsupportedOpShapeInferer"
            }
            LiteRtStatus_kLiteRtStatusErrorShapeInferenceFailed => {
                "kLiteRtStatusErrorShapeInferenceFailed"
            }

            _ => "???",
        }
        // LINT.ThenChange(../../c/litert_common.h:status_codes)
        .to_string()
    }
}
