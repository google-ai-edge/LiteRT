#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_COMPILER_SERVICE_API_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_COMPILER_SERVICE_API_H_

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct CompilerServiceContext CompilerServiceContext;

int CompilerServiceCreate(CompilerServiceContext** context);

int CompilerServiceSubgraphFlatbuffer(
    CompilerServiceContext* context, const char* tfl_buffer_data,
    size_t tfl_buffer_size, const char* serialized_options_data,
    size_t serialized_options_size, char** compiled_code_data,
    size_t* compiled_code_size, char** out_status_string);

void CompilerServiceFreeCompiledCode(CompilerServiceContext* context,
                                     char* compiled_code_data);

void CompilerServiceFreeErrorMessage(CompilerServiceContext* context,
                                     char* error_message);

void CompilerServiceDestroy(CompilerServiceContext* context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_COMPILER_SERVICE_API_H_
