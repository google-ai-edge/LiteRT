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

#include "ml_drift_delegate/delegate/delegate_utils.h"

#include "ml_drift/common/precision.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "ml_drift_delegate/delegate/composite/ir/custom_parsers.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/support.h"
#include "tflite/core/c/common.h"

namespace litert::ml_drift {

bool IsAsyncExecutionMode(TfLiteContext* context,
                          const LiteRtRuntimeContext* runtime_context) {
  bool is_async_execution_mode = false;
  auto* buffer_context = reinterpret_cast<LiteRtExternalLiteRtBufferContext>(
      context->GetExternalContext(context, kTfLiteLiteRtBufferContext));
  if (buffer_context != nullptr &&
      runtime_context->external_litert_buffer_context_is_async_execution_mode(
          buffer_context, &is_async_execution_mode) == kLiteRtStatusOk) {
    return is_async_execution_mode;
  }
  return false;
}

TfLiteIntArray* GetIrModelOpsToReplace(TfLiteContext* context,
                                       const MlDriftDelegateData& delegate_data,
                                       int start_node_index,
                                       int end_node_index) {
  ir::IrModelBuilderOptions ir_options;
  ir_options.enable_infinite_float_capping =
      delegate_data.options->enable_infinite_float_capping;
  ir_options.enable_reduced_precision = delegate_data.calculation_precision !=
                                        ::ml_drift::CalculationsPrecision::F32;
  ir_options.allow_quant_ops = true;
  ir_options.start_node_index = start_node_index;
  ir_options.end_node_index = end_node_index;
  ir_options.max_delegated_partitions = 1;
  auto custom_parsers = ir::GetCustomParsers();
  return ir::GetOpsToReplace(context, ir_options, &custom_parsers);
}

}  // namespace litert::ml_drift
