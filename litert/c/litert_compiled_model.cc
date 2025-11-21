// Copyright 2024 Google LLC.
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

#include "litert/c/litert_compiled_model.h"

#include <stddef.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/compiled_model.h"

#ifdef __cplusplus
extern "C" {
#endif

LiteRtStatus LiteRtCreateCompiledModel(LiteRtEnvironment environment,
                                       LiteRtModel model,
                                       LiteRtOptions jit_compilation_options,
                                       LiteRtCompiledModel* compiled_model) {
  if (!environment || !model || !compiled_model) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(auto created_compiled_model,
                          LiteRtCompiledModelT::Create(
                              environment, model, jit_compilation_options));
  *compiled_model = created_compiled_model.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledModelInputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex input_index,
    LiteRtTensorBufferRequirements* buffer_requirements) {
  if (!compiled_model || !buffer_requirements) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(
      LiteRtTensorBufferRequirementsConst buffer_requirements_ptr,
      compiled_model->GetInputBufferRequirements(signature_index, input_index));
  *buffer_requirements =
      const_cast<LiteRtTensorBufferRequirements>(buffer_requirements_ptr);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledModelOutputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex output_index,
    LiteRtTensorBufferRequirements* buffer_requirements) {
  if (!compiled_model || !buffer_requirements) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(
      LiteRtTensorBufferRequirementsConst buffer_requirements_ptr,
      compiled_model->GetOutputBufferRequirementsCApi(signature_index,
                                                      output_index));
  *buffer_requirements =
      const_cast<LiteRtTensorBufferRequirements>(buffer_requirements_ptr);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledModelInputTensorLayout(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex input_index, LiteRtLayout* layout) {
  if (!compiled_model || !layout) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(
      LiteRtLayout computed_layout,
      compiled_model->GetInputTensorLayout(signature_index, input_index));
  *layout = computed_layout;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledModelEnvironment(
    LiteRtCompiledModel compiled_model, LiteRtEnvironment* environment) {
  if (!compiled_model || !environment) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(*environment, compiled_model->GetEnvironment());
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledModelOutputTensorLayouts(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    size_t num_layouts, LiteRtLayout* layouts, bool update_allocation) {
  if (!compiled_model || !layouts) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  absl::Span<LiteRtLayout> output_layouts(layouts, num_layouts);
  LITERT_RETURN_IF_ERROR(compiled_model->GetOutputTensorShapes(
      signature_index, output_layouts, update_allocation));
  size_t tensors_size = output_layouts.size();
  if (tensors_size == 0) {
    LITERT_LOG(LITERT_WARNING, "No output tensors found for signature index.");
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtRunCompiledModel(LiteRtCompiledModel compiled_model,
                                    LiteRtParamIndex signature_index,
                                    size_t num_input_buffers,
                                    LiteRtTensorBuffer* input_buffers,
                                    size_t num_output_buffers,
                                    LiteRtTensorBuffer* output_buffers) {
  if (!compiled_model || (num_input_buffers > 0 && !input_buffers) ||
      (num_output_buffers > 0 && !output_buffers)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  bool async = false;
  auto res =
      compiled_model->RunCApi(signature_index, num_input_buffers, input_buffers,
                              num_output_buffers, output_buffers, &async);
  if (!res) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().c_str());
    return res.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtRunCompiledModelAsync(LiteRtCompiledModel compiled_model,
                                         LiteRtParamIndex signature_index,
                                         size_t num_input_buffers,
                                         LiteRtTensorBuffer* input_buffers,
                                         size_t num_output_buffers,
                                         LiteRtTensorBuffer* output_buffers,
                                         bool* async) {
  if (!compiled_model || (num_input_buffers > 0 && !input_buffers) ||
      (num_output_buffers > 0 && !output_buffers)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (async) {
    *async = true;
  }
  bool async_ = true;
  bool* async_ptr = async ? async : &async_;

  auto res =
      compiled_model->RunCApi(signature_index, num_input_buffers, input_buffers,
                              num_output_buffers, output_buffers, async_ptr);
  if (!res) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().c_str());
    return res.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetCompiledModelCancellationFunction(
    LiteRtCompiledModel compiled_model, void* data,
    bool (*check_cancelled_func)(void*)) {
  if (!compiled_model || !check_cancelled_func) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  compiled_model->SetCancellationFunction(data, check_cancelled_func);
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledModel(LiteRtCompiledModel compiled_model) {
  delete compiled_model;
}

LiteRtStatus LiteRtCompiledModelStartMetricsCollection(
    LiteRtCompiledModel compiled_model, int detail_level) {
  if (!compiled_model) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(compiled_model->StartMetricsCollection(detail_level));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledModelStopMetricsCollection(
    LiteRtCompiledModel compiled_model, LiteRtMetrics metrics) {
  if (!compiled_model || !metrics) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(*metrics, compiled_model->StopMetricsCollection());
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledModelIsFullyAccelerated(
    LiteRtCompiledModel compiled_model, bool* fully_accelerated) {
  LITERT_RETURN_IF_ERROR(
      compiled_model != nullptr && fully_accelerated != nullptr,
      kLiteRtStatusErrorInvalidArgument);

  LITERT_ASSIGN_OR_RETURN(bool has_non_delegated_ops,
                          compiled_model->HasNonDelegatedOps());
  *fully_accelerated = !has_non_delegated_ops;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledModelGetProfiler(LiteRtCompiledModel compiled_model,
                                            LiteRtProfiler* profiler) {
  LITERT_RETURN_IF_ERROR(compiled_model != nullptr && profiler != nullptr,
                         kLiteRtStatusErrorInvalidArgument);
  LITERT_ASSIGN_OR_RETURN(LiteRtProfilerT * profiler_ptr,
                          compiled_model->GetProfiler());
  *profiler = profiler_ptr;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledModelResizeInputTensorNonStrict(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex input_index, const int* dims, size_t dims_size) {
  LITERT_RETURN_IF_ERROR(compiled_model != nullptr,
                         kLiteRtStatusErrorInvalidArgument);
  LITERT_RETURN_IF_ERROR(compiled_model->ResizeInputTensorNonStrict(
      signature_index, input_index, absl::MakeConstSpan(dims, dims_size)));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledModelResizeInputTensor(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex input_index, const int* dims, size_t dims_size) {
  LITERT_RETURN_IF_ERROR(compiled_model != nullptr,
                         kLiteRtStatusErrorInvalidArgument);
  LITERT_RETURN_IF_ERROR(compiled_model->ResizeInputTensor(
      signature_index, input_index, absl::MakeConstSpan(dims, dims_size)));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledModelSetDispatchAnnotation(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    const char* key, const char* value) {
  if (!compiled_model || !key || !value) {
    LITERT_LOG(LITERT_ERROR, "Invalid arguments: null pointers provided");
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Validate signature index
  if (signature_index >= compiled_model->GetNumSignatures()) {
    LITERT_LOG(LITERT_ERROR, "Invalid signature index: %zu (max: %zu)",
               signature_index, compiled_model->GetNumSignatures() - 1);
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Get the buffer context and set the annotation
  auto* buffer_context = compiled_model->GetBufferContext();
  if (!buffer_context) {
    LITERT_LOG(LITERT_ERROR, "Buffer context not initialized");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  buffer_context->SetSignatureDispatchAnnotation(signature_index, key, value);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledModelGetDispatchAnnotation(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    const char* key, const char** value) {
  if (!compiled_model || !key || !value) {
    LITERT_LOG(LITERT_ERROR, "Invalid arguments: null pointers provided");
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Validate signature index
  if (signature_index >= compiled_model->GetNumSignatures()) {
    LITERT_LOG(LITERT_ERROR, "Invalid signature index: %zu (max: %zu)",
               signature_index, compiled_model->GetNumSignatures() - 1);
    *value = nullptr;
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Get the buffer context and retrieve the annotation
  auto* buffer_context = compiled_model->GetBufferContext();
  if (!buffer_context) {
    LITERT_LOG(LITERT_ERROR, "Buffer context not initialized");
    *value = nullptr;
    return kLiteRtStatusErrorRuntimeFailure;
  }

  const auto* annotation_value =
      buffer_context->GetSignatureDispatchAnnotation(signature_index, key);
  if (annotation_value) {
    *value = annotation_value->c_str();
  } else {
    *value = nullptr;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledModelRemoveDispatchAnnotation(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    const char* key) {
  if (!compiled_model || !key) {
    LITERT_LOG(LITERT_ERROR, "Invalid arguments: null pointers provided");
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Validate signature index
  if (signature_index >= compiled_model->GetNumSignatures()) {
    LITERT_LOG(LITERT_ERROR, "Invalid signature index: %zu (max: %zu)",
               signature_index, compiled_model->GetNumSignatures() - 1);
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Get the buffer context and remove the annotation
  auto* buffer_context = compiled_model->GetBufferContext();
  if (!buffer_context) {
    LITERT_LOG(LITERT_ERROR, "Buffer context not initialized");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  buffer_context->RemoveSignatureDispatchAnnotation(signature_index, key);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledModelReportError(LiteRtCompiledModel compiled_model,
                                            const char* format, ...) {
  LITERT_RETURN_IF_ERROR(compiled_model != nullptr && format != nullptr,
                         kLiteRtStatusErrorInvalidArgument);

#if defined(LITERT_WINDOWS_OS)
  LITERT_LOG(LITERT_ERROR, "Report error not implemented");
  return kLiteRtStatusErrorUnsupported;
#else
  va_list args;
  va_start(args, format);
  // Create a formatted string since ReportError expects format and variadic
  // args
  char* buffer = nullptr;
  int len = vasprintf(&buffer, format, args);
  if (len < 0 || buffer == nullptr) {
    va_end(args);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  compiled_model->ReportError("%s", buffer);
  va_end(args);
  free(buffer);

  return kLiteRtStatusOk;
#endif
}

LiteRtStatus LiteRtCompiledModelClearErrors(
    LiteRtCompiledModel compiled_model) {
  LITERT_RETURN_IF_ERROR(compiled_model != nullptr,
                         kLiteRtStatusErrorInvalidArgument);

  auto result = compiled_model->ClearErrors();
  if (!result) {
    return result.Error().Status();
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledModelGetErrorMessages(
    LiteRtCompiledModel compiled_model, char** error_messages) {
  LITERT_RETURN_IF_ERROR(compiled_model != nullptr && error_messages != nullptr,
                         kLiteRtStatusErrorInvalidArgument);

  auto result = compiled_model->GetErrorMessages();
  if (!result) {
    return result.Error().Status();
  }

  // Allocate and copy the string
  size_t len = result->size();
  *error_messages = static_cast<char*>(malloc(len + 1));
  if (*error_messages == nullptr) {
    return kLiteRtStatusErrorRuntimeFailure;
  }

  memcpy(*error_messages, result->c_str(), len);
  (*error_messages)[len] = '\0';

  return kLiteRtStatusOk;
}

#ifdef __cplusplus
}  // extern "C"
#endif
