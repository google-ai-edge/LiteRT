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

#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "ml_drift_delegate/delegate/composite/ir/custom_parsers.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/precision.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/support.h"
#include "tflite/core/c/common.h"

namespace litert::ml_drift {

::ml_drift::CalculationsPrecision ResolveOpenGlCalculationPrecision(
    const ::ml_drift::GpuInfo& gpu_info,
    const MlDriftDelegateOptions& options) {
  ::ml_drift::CalculationsPrecision precision;
  switch (options.precision) {
    case kDefault: {
      const bool is_old_powervr =
          gpu_info.IsPowerVR() &&
          !gpu_info.powervr_info.IsBetterThan(
              ::ml_drift::PowerVRGpu::kRogueGm9xxx);
      if (is_old_powervr || !gpu_info.SupportsFP16()) {
        // PowerVR Rogue GE8xxx and older have precision issues (RTZ) with FP16
        // leading to accuracy failures (b/274571359).
        precision = ::ml_drift::CalculationsPrecision::kF32;
      } else if (gpu_info.IsRoundToNearestSupported()) {
        precision = ::ml_drift::CalculationsPrecision::kF16;
      } else {
        // GPUs without round-to-nearest support (e.g. Adreno 1xx-3xx and newer
        // PowerVR) use FP32 accumulators with FP16 storage, avoiding pure FP16
        // shaders on legacy shader compilers.
        precision = ::ml_drift::CalculationsPrecision::kF32F16;
      }
      break;
    }
    case kFp16:
      precision = ::ml_drift::CalculationsPrecision::kF16;
      break;
    case kFp32:
      precision = ::ml_drift::CalculationsPrecision::kF32;
      break;
  }
  if (options.use_f32_accum_for_fp16) {
    precision = ::ml_drift::CalculationsPrecision::kF32F16;
  }
  return precision;
}

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
                                        ::ml_drift::CalculationsPrecision::kF32;
  ir_options.allow_quant_ops = true;
  ir_options.start_node_index = start_node_index;
  ir_options.end_node_index = end_node_index;
  ir_options.max_delegated_partitions = 1;
  auto custom_parsers = ir::GetCustomParsers();
  return ir::GetOpsToReplace(context, ir_options, &custom_parsers);
}

}  // namespace litert::ml_drift
