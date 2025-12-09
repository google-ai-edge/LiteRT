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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_EDGETPU_COMPILER_API_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_EDGETPU_COMPILER_API_H_

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct EdgeTpuCompilerContext EdgeTpuCompilerContext;

// Creates a EdgeTPU ODC compiler context.
// context is the output parameter for the compiler context. The caller should
// free the memory using EdgeTpuCompilerDestroy.
// Returns 0 on success, absl status code on failure.
int EdgeTpuCompilerCreate(EdgeTpuCompilerContext** context);

// Compiles a TfLite flatbuffer into a DarwiNN executable.
//
// serialized_options_data is the serialized EdgeTpuCompilerOptions proto.
// compiled_code_data and compiled_code_size are the output parameters for the
// compiled code. The caller should free the memory using
// EdgeTpuCompilerFreeCompiledCode.
// out_status_string is the output parameter for the error message. The caller
// should free the memory using EdgeTpuCompilerFreeErrorMessage.
// Returns 0 on success, absl status code on failure.
int EdgeTpuCompilerCompileFlatbuffer(
    EdgeTpuCompilerContext* context, const char* tfl_buffer_data,
    size_t tfl_buffer_size, const char* serialized_options_data,
    size_t serialized_options_size, char*** compiled_code_data,
    size_t** compiled_code_sizes, size_t* num_bytecodes,
    char** out_status_string);

// Frees the compiled code data.
void EdgeTpuCompilerFreeCompiledCode(EdgeTpuCompilerContext* context,
                                     char** compiled_code_data,
                                     size_t* compiled_code_sizes,
                                     size_t num_bytecodes);

// Frees the error message.
void EdgeTpuCompilerFreeErrorMessage(EdgeTpuCompilerContext* context,
                                     char* error_message);

// Destroys the EdgeTPU ODC compiler context.
void EdgeTpuCompilerDestroy(EdgeTpuCompilerContext* context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_EDGETPU_COMPILER_API_H_
