// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_IR_MODEL_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_IR_MODEL_BUILDER_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/c/common.h"
#include "tflite/core/api/op_resolver.h"
#include "tflite/model_builder.h"

namespace litert::ml_drift::ir {

::ml_drift::ir::IrModel* BuildIrModel(
    const TfLiteContext& context, const TfLiteDelegateParams& delegate_params,
    const IrModelBuilderOptions& options,
    const CustomIrOpMap* custom_parsers = nullptr,
    SharedConstTensorsMap* shared_tensors = nullptr,
    const TensorIndexToBufferIdMap* tensor_to_shared_buffer_id_map = nullptr,
    const TensorIndexToExternalBufferIdMap* tensor_to_external_buffer_id_map =
        nullptr);

absl::Status BuildFromFlatBuffer(
    const ::tflite::FlatBufferModel& flatbuffer,
    const ::tflite::OpResolver& op_resolver,
    const IrModelBuilderOptions& options,
    ::ml_drift::ir::IrModel* ir_model);

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_IR_MODEL_BUILDER_H_
