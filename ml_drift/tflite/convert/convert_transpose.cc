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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_transpose.h"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/kernels/internal/tensor_ctypes.h"

namespace litert::ml_drift::ir {

void ConvertTranspose(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const int input_id = node.inputs->data[0];
  const int perm_id = node.inputs->data[1];
  const int output_id = node.outputs->data[0];

  const TfLiteTensor& perm_tensor = context.tensors[perm_id];
  const int num_elements = tflite::NumElements(&perm_tensor);
  const int* perm_data = tflite::GetTensorData<int32_t>(&perm_tensor);

  ::ml_drift::ir::IrOp* transpose_op = ir_model.add_op();
  transpose_op->name = ToString(::ml_drift::OperationType::TRANSPOSE);

  ir_model.AddConsumer(tensor_map[input_id], transpose_op->id);
  ir_model.SetProducer(tensor_map[output_id], transpose_op->id);

  if (num_elements == 5) {
    ::ml_drift::Transpose3DAttributes attr;
    attr.perm = ::ml_drift::BHWDC(perm_data[0], perm_data[1], perm_data[2],
                                  perm_data[3], perm_data[4]);
    transpose_op->attr = std::move(attr);
  } else {
    ::ml_drift::TransposeAttributes attr;
    std::map<::ml_drift::Axis, int> axis_to_index = {
        {::ml_drift::Axis::BATCH, 0},
        {::ml_drift::Axis::HEIGHT, 1},
        {::ml_drift::Axis::WIDTH, 2},
        {::ml_drift::Axis::CHANNELS, 3}};

    if (num_elements == 4) {
      attr.perm = ::ml_drift::BHWC(perm_data[0], perm_data[1], perm_data[2],
                                   perm_data[3]);
    } else if (num_elements == 3) {
      std::vector<::ml_drift::Axis> index_to_axis = {
          ::ml_drift::Axis::BATCH, ::ml_drift::Axis::WIDTH,
          ::ml_drift::Axis::CHANNELS};
      attr.perm.b = axis_to_index[index_to_axis[perm_data[0]]];
      attr.perm.h = 1;
      attr.perm.w = axis_to_index[index_to_axis[perm_data[1]]];
      attr.perm.c = axis_to_index[index_to_axis[perm_data[2]]];
    } else if (num_elements == 2) {
      std::vector<::ml_drift::Axis> index_to_axis = {
          ::ml_drift::Axis::BATCH, ::ml_drift::Axis::CHANNELS};
      attr.perm.b = axis_to_index[index_to_axis[perm_data[0]]];
      attr.perm.h = 1;
      attr.perm.w = 2;
      attr.perm.c = axis_to_index[index_to_axis[perm_data[1]]];
    }

    transpose_op->attr = std::move(attr);
  }
}

}  // namespace litert::ml_drift::ir
