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

#![allow(non_upper_case_globals)]

use crate::bindings::*;

use std::fmt;

/// Reason of the error. It provides information where in the binding code the error occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCause {
    Unknown,
    // compiled_model
    CreateOptions,
    SetOptionsHardwareAccelerators,
    CreateCompiledModel,
    GetCompiledModelInputBufferRequirements,
    GetCompiledModelOutputBufferRequirements,
    InputDoesntSupportAnyTensorBufferTypes,
    RunCompiledModel,
    // environment
    NotSupportedLiteRtAnyType,
    CreateEnvironment,
    // model
    GetSignatureKey,
    GetSignatureSubgraph,
    GetNumSignatureInputs,
    GetSignatureInputName,
    GetNumSignatureOutputs,
    GetSignatureOutputName,
    GetSignature,
    GetTensorTypeId,
    GetUnrankedTensorType,
    GetRankedTensorType,
    InvalidTensorTypeId,
    GetTensorName,
    GetNumSubgraphInputs,
    GetNumSubgraphOutputs,
    GetSubgraphInput,
    SubgraphInputTensorByNameNotFound,
    GetSubgraphOutput,
    SubgraphOutputTensorByNameNotFound,
    CreateModelFromFile,
    CreateModelFromBuffer,
    GetNumModelSubgraphs,
    GetNumModelSignatures,
    GetModelSignature,
    //tensor_buffer
    GetTensorBufferRequirementsBufferSize,
    GetNumTensorBufferRequirementsSupportedBufferTypes,
    GetTensorBufferRequirementsSupportedTensorBufferType,
    InvalidElementTypeEnumValue,
    InvalidTensorBufferTypeEnumValue,
    CreateManagedTensorBuffer,
    LockTensorBufferRead,
    LockTensorBufferWrite,
    GetTensorBufferPackedSize,
    IncompatibleWriteType,
    TensorBufferTooSmall,
    IncompatibleReadType,
    ReadBufferTooSmall,
    // util
    InvalidStringEncoding,
}

/// Error returned by the bindings.
///
/// It contains the reason of the error and the LiteRtStatus returned by the C code.
///
/// Error supports `fmt::Display` and `fmt::Debug` traits.
///
/// ```
/// match(litert_status) {
///     Ok(env) => { ... },
///     Err(error) => {
///         println!("Error: {:?}", error);
///     }
/// }
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Error {
    cause: ErrorCause,
    litert_status: LiteRtStatus,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error: {:?}", self.cause)?;
        let status_description = self.litert_status_description();
        write!(
            f,
            " LiteRtStatus: {:?} [{}] ",
            self.litert_status, status_description
        )
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for Error {}

impl Error {
    pub(crate) fn new(cause: ErrorCause, status: LiteRtStatus) -> Self {
        Error {
            cause,
            litert_status: status,
        }
    }

    /// Returns the reason of the error from the binding code.
    pub fn cause(&self) -> ErrorCause {
        self.cause
    }

    /// Returns the LiteRtStatus returned by the C code.
    pub fn litert_status(&self) -> LiteRtStatus {
        self.litert_status
    }

    /// Returns a string description of the LiteRtStatus returned by the C code.
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
        // LINT.ThenChange(../c/litert_common.h:status_codes)
        .to_string()
    }
}
