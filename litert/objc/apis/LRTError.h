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

/** Error codes matching LiteRT status codes. */
typedef NS_ERROR_ENUM(LRTErrorDomain, LRTErrorCode){
    LRTErrorCodeOk = 0,              /* Success status. */
    LRTErrorCodeErrorUnknown = 1,    /* Unknown error occurred. */
    LRTErrorCodeInvalidArgument = 2, /* Invalid argument supplied. */
    LRTErrorCodeNotFound = 3,        /* Target resource not found. */
    LRTErrorCodeRuntimeFailure = 4,  /* Runtime execution failure. */
    LRTErrorCodeUnsupported = 5,     /* Operation or backend unsupported. */
};

NS_ASSUME_NONNULL_END
