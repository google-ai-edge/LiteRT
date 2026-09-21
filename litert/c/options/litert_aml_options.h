// Copyright (C) 2023 Amlogic, Inc. All rights reserved.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_AML_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_AML_OPTIONS_H_

#include <stdint.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    // ========== HANDLE 定义 ==========
    LITERT_DEFINE_HANDLE(LiteRtAmlOptions);

    // Create an Aml options object that is type erased.
    LiteRtStatus LiteRtAmlOptionsCreate(LiteRtOpaqueOptions *options);

    // Identifier string for AML options in type-erased options.
    const char *LiteRtAmlOptionsGetIdentifier();

    // Attempt to retrieve AML options from the opaque options.
    LiteRtStatus LiteRtAmlOptionsGet(LiteRtOpaqueOptions options,
                                     LiteRtAmlOptions *options_data);

    // ========== GENERAL SDK SETTINGS ==========

    // log_level
    typedef enum LiteRtAmlOptionsLogLevel
    {
        kLiteRtAmlLogOff = 0,
        kLiteRtAmlLogLevelError = 1,
        kLiteRtAmlLogLevelWarn = 2,
        kLiteRtAmlLogLevelInfo = 3,
        kLiteRtAmlLogLevelVerbose = 4,
        kLiteRtAmlLogLevelDebug = 5,
    } LiteRtAmlOptionsLogLevel;

    LiteRtStatus LiteRtAmlOptionsSetLogLevel(
        LiteRtAmlOptions options, LiteRtAmlOptionsLogLevel log_level);

    LiteRtStatus LiteRtAmlOptionsGetLogLevel(
        LiteRtAmlOptions options, LiteRtAmlOptionsLogLevel *log_level);

    // ========== EXECUTION / DISPATCH OPTIONS ==========

    // performance mode (针对 Amlogic NPU 运行模式)
    typedef enum LiteRtAmlOptionsPerfMode
    {
        kLiteRtAmlPerfDefault = 0,
        kLiteRtAmlPerfHighPerformance,
        kLiteRtAmlPerfBalanced,
        kLiteRtAmlPerfPowerSaver,
    } LiteRtAmlOptionsPerfMode;

    LiteRtStatus LiteRtAmlOptionsSetPerfMode(
        LiteRtAmlOptions options, LiteRtAmlOptionsPerfMode mode);

    LiteRtStatus LiteRtAmlOptionsGetPerfMode(
        LiteRtAmlOptions options, LiteRtAmlOptionsPerfMode *mode);

    // profiling
    typedef enum LiteRtAmlOptionsProfiling
    {
        kLiteRtAmlProfilingOff = 0,
        kLiteRtAmlProfilingBasic,
        kLiteRtAmlProfilingDetailed,
    } LiteRtAmlOptionsProfiling;

    LiteRtStatus LiteRtAmlOptionsSetProfiling(
        LiteRtAmlOptions options, LiteRtAmlOptionsProfiling profiling);

    LiteRtStatus LiteRtAmlOptionsGetProfiling(
        LiteRtAmlOptions options, LiteRtAmlOptionsProfiling *profiling);

    typedef enum LiteRtAmlOptionsMemoryPolicy
    {
        kLiteRtAmlMemoryDefault = 0,
        kLiteRtAmlMemoryIOMMU,
        kLiteRtAmlMemoryDMA,
    } LiteRtAmlOptionsMemoryPolicy;

    LiteRtStatus LiteRtAmlOptionsSetMemoryPolicy(
        LiteRtAmlOptions options, LiteRtAmlOptionsMemoryPolicy policy);

    LiteRtStatus LiteRtAmlOptionsGetMemoryPolicy(
        LiteRtAmlOptions options, LiteRtAmlOptionsMemoryPolicy *policy);

    // model path/name
    LiteRtStatus LiteRtAmlOptionsSetModelPath(
        LiteRtAmlOptions options, const char *model_path);

    LiteRtStatus LiteRtAmlOptionsGetModelPath(
        LiteRtAmlOptions options, const char **model_path);

    LiteRtStatus LiteRtAmlOptionsSetModelName(
        LiteRtAmlOptions options, const char *model_name);

    LiteRtStatus LiteRtAmlOptionsGetModelName(
        LiteRtAmlOptions options, const char **model_name);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_AML_OPTIONS_H_
