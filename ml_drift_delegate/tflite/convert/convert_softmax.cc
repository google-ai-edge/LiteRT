// Copyright 2026 Google LLC.
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

#include "ml_drift_delegate/tflite/convert/convert_softmax.h"

#include <limits>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

using ::ml_drift::Axis;
using ::ml_drift::BHWDC;
using ::ml_drift::DataType;
using ::ml_drift::ElementwiseAttributes;
using ::ml_drift::OperationType;
using ::ml_drift::SoftmaxAttributes;
using ::ml_drift::kMaxHalf;
using ::ml_drift::ir::IrModel;
using ::ml_drift::ir::IrOp;
using ::ml_drift::ir::IrTensor;
using ::ml_drift::ir::IrTensorId;

void ConvertSoftmax(const TfLiteContext& context, const TfLiteNode& node,
                    const TfLiteRegistration& registration,
                    const IrModelBuilderOptions& options,
                    absl::flat_hash_map<int, IrTensorId>& tensor_map,
                    IrModel& ir_model) {
  const int input_id = node.inputs->data[0];
  const int output_id = node.outputs->data[0];

  auto input_it = tensor_map.find(input_id);
  ABSL_CHECK(input_it != tensor_map.end()) << "Input tensor missing from map.";
  IrTensorId current_tensor_id = input_it->second;

  if (options.enable_infinite_float_capping) {
    const TfLiteTensor& input_tensor = context.tensors[input_id];
    const bool input_is_fp16 = input_tensor.type == kTfLiteFloat16;
    const bool use_half = options.enable_reduced_precision || input_is_fp16;
    const float cap_value =
        use_half ? kMaxHalf : std::numeric_limits<float>::max();

    const IrTensor& operand_tensor = *ir_model.tensor(current_tensor_id);
    const BHWDC& shape = operand_tensor.desc.GetBHWDCShape();
    const DataType dtype = operand_tensor.desc.GetDataType();

    // Create MAXIMUM op: max(operand, -cap_value)
    IrOp& max_op = *ir_model.add_op();
    max_op.name = ToString(OperationType::MAXIMUM);
    max_op.attr = ElementwiseAttributes{/*param=*/-cap_value};
    ir_model.AddConsumer(current_tensor_id, max_op.id);

    IrTensor& max_output_tensor = *ir_model.add_tensor(dtype, shape);
    ir_model.SetProducer(max_output_tensor.id, max_op.id);
    current_tensor_id = max_output_tensor.id;

    // Create MINIMUM op: min(operand, cap_value)
    IrOp& min_op = *ir_model.add_op();
    min_op.name = ToString(OperationType::MINIMUM);
    min_op.attr = ElementwiseAttributes{/*param=*/cap_value};
    ir_model.AddConsumer(current_tensor_id, min_op.id);

    IrTensor& min_output_tensor = *ir_model.add_tensor(dtype, shape);
    ir_model.SetProducer(min_output_tensor.id, min_op.id);
    current_tensor_id = min_output_tensor.id;
  }

  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;  // Always by channels as per model_builder.cc

  IrOp& op = *ir_model.add_op();
  op.name = ToString(OperationType::SOFTMAX);
  op.attr = attr;

  ir_model.AddConsumer(current_tensor_id, op.id);
  auto output_it = tensor_map.find(output_id);
  ABSL_CHECK(output_it != tensor_map.end())
      << "Output tensor missing from map.";
  ir_model.SetProducer(output_it->second, op.id);
}

}  // namespace litert::ml_drift::ir
