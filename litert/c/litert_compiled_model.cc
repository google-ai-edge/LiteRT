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

#include <memory>

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_logging.h"
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

  auto res = compiled_model->GetInputBufferRequirementsCApi(signature_index,
                                                            input_index);
  if (!res) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().c_str());
    return res.Error().Status();
  }
  *buffer_requirements = res.Value();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledModelOutputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex output_index,
    LiteRtTensorBufferRequirements* buffer_requirements) {
  if (!compiled_model || !buffer_requirements) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto res = compiled_model->GetOutputBufferRequirementsCApi(signature_index,
                                                             output_index);
  if (!res) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().c_str());
    return res.Error().Status();
  }
  *buffer_requirements = res.Value();
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

#ifdef __cplusplus
}  // extern "C"
#endif
