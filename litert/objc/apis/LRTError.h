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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/** Domain for errors returned by LiteRT Objective-C APIs. */
FOUNDATION_EXPORT NSString* const LRTErrorDomain;

// LINT.IfChange(status_codes)
/** Error codes matching LiteRT status codes. */
typedef NS_ERROR_ENUM(LRTErrorDomain, LRTErrorCode){
    LRTErrorCodeOk = 0,                             /* Success status. */
    LRTErrorCodeInvalidArgument = 1,                /* Invalid argument supplied. */
    LRTErrorCodeMemoryAllocationFailure = 2,        /* Memory allocation failure. */
    LRTErrorCodeRuntimeFailure = 3,                 /* Runtime execution failure. */
    LRTErrorCodeMissingInputTensor = 4,             /* Missing input tensor. */
    LRTErrorCodeUnsupported = 5,                    /* Operation or backend unsupported. */
    LRTErrorCodeNotFound = 6,                       /* Target resource not found. */
    LRTErrorCodeTimeoutExpired = 7,                 /* Operation timeout expired. */
    LRTErrorCodeWrongVersion = 8,                   /* Incompatible version. */
    LRTErrorCodeUnknown = 9,                        /* Unknown error occurred. */
    LRTErrorCodeAlreadyExists = 10,                 /* Resource already exists. */
    LRTErrorCodeCancelled = 100,                    /* Inference progression cancelled. */
    LRTErrorCodeFileIO = 500,                       /* File I/O error. */
    LRTErrorCodeInvalidFlatbuffer = 501,            /* Invalid FlatBuffer format. */
    LRTErrorCodeDynamicLoading = 502,               /* Dynamic library loading failure. */
    LRTErrorCodeSerialization = 503,                /* Serialization failure. */
    LRTErrorCodeCompilation = 504,                  /* Compilation failure. */
    LRTErrorCodeIndexOOB = 1000,                    /* Index out of bounds. */
    LRTErrorCodeInvalidIRType = 1001,               /* Invalid IR type. */
    LRTErrorCodeInvalidGraphInvariant = 1002,       /* Invalid graph invariant. */
    LRTErrorCodeGraphModification = 1003,           /* Graph modification failure. */
    LRTErrorCodeInvalidToolConfig = 1500,           /* Invalid tool configuration. */
    LRTErrorCodeLegalizeNoMatch = 2000,             /* Legalization pattern no match. */
    LRTErrorCodeInvalidLegalization = 2001,         /* Invalid legalization. */
    LRTErrorCodePatternNoMatch = 3000,              /* Pattern match failure. */
    LRTErrorCodeInvalidTransformation = 3001,       /* Invalid transformation. */
    LRTErrorCodeUnsupportedRuntimeVersion = 4000,   /* Unsupported runtime version. */
    LRTErrorCodeUnsupportedCompilerVersion = 4001,  /* Unsupported compiler version. */
    LRTErrorCodeIncompatibleByteCodeVersion = 4002, /* Incompatible bytecode version. */
    LRTErrorCodeUnsupportedOpShapeInferer = 5000,   /* Unsupported op shape inferer. */
    LRTErrorCodeShapeInferenceFailed = 5001,        /* Shape inference failed. */
};
// LINT.ThenChange(
//   ../../c/litert_common.h:status_codes,
//   ../../cc/litert_common.h:status_codes
// )

NS_ASSUME_NONNULL_END
