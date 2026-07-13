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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_broadcast_in_dim.h"

#include <array>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

// Returns map from tensor dimension index to BHWDC dimension index for ranks
// 1-5, based on how low-rank tensors are mapped to 5D BHWDC shapes.
absl::flat_hash_map<int, int> GetTensorToBhwdcMap(int rank) {
  if (rank == 1) return {{0, 4}};                          // C
  if (rank == 2) return {{0, 1}, {1, 2}};                  // HW
  if (rank == 3) return {{0, 1}, {1, 2}, {2, 4}};          // HWC
  if (rank == 4) return {{0, 0}, {1, 1}, {2, 2}, {3, 4}};  // BHWC
  return {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}};         // BHWDC
}

}  // namespace

void ConvertBroadcastInDim(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const TfLiteTensor* input_tensor = &context.tensors[node.inputs->data[0]];
  const TfLiteTensor* broadcast_tensor = &context.tensors[node.inputs->data[1]];
  const TfLiteTensor* output_tensor = &context.tensors[node.outputs->data[0]];
  std::vector<int> broadcast_map(
      broadcast_tensor->data.i32,
      broadcast_tensor->data.i32 + broadcast_tensor->dims->data[0]);

  // Add transpose op
  ::ml_drift::ir::IrOp* transpose_op = ir_model.add_op();
  transpose_op->name = ToString(::ml_drift::OperationType::TRANSPOSE);
  const int input_id = node.inputs->data[0];
  ir_model.AddConsumer(tensor_map[input_id], transpose_op->id);

  // Determine transpose permutation
  // In ML Drift, tensors are converted to 5D BHWDC. broadcast_in_dim is
  // converted to a transpose + optional tile. The transpose op permutes
  // dimensions to match output dimension order.
  // The permutation `perm` for transpose is such that
  // output_shape[i] = input_shape[perm[i]].
  // We need to compute this permutation based on broadcast_map which tells
  // which output tensor dimension corresponds to which input tensor dimension.
  const int input_rank = input_tensor->dims->size;
  const int output_rank = output_tensor->dims->size;
  auto in_map = GetTensorToBhwdcMap(input_rank);
  auto out_map = GetTensorToBhwdcMap(output_rank);

  std::array<int, 5> perm_arr;
  std::vector<bool> in_bhwdc_dim_used(5, false);
  std::vector<bool> out_bhwdc_dim_filled(5, false);

  for (int i = 0; i < input_rank; ++i) {
    perm_arr[out_map[broadcast_map[i]]] = in_map[i];
    in_bhwdc_dim_used[in_map[i]] = true;
    out_bhwdc_dim_filled[out_map[broadcast_map[i]]] = true;
  }
  // Fill in permutation for dimensions not specified by broadcast_map,
  // which correspond to degenerate dimensions (size 1) added during
  // conversion to BHWDC.
  int current_in_dim = 0;
  for (int i = 0; i < 5; ++i) {
    if (!out_bhwdc_dim_filled[i]) {
      while (current_in_dim < 5 && in_bhwdc_dim_used[current_in_dim]) {
        current_in_dim++;
      }
      if (current_in_dim < 5) {
        perm_arr[i] = current_in_dim;
        out_bhwdc_dim_filled[i] = true;
        in_bhwdc_dim_used[current_in_dim] = true;
      }
    }
  }

  ::ml_drift::BHWDC perm(perm_arr[0], perm_arr[1], perm_arr[2], perm_arr[3],
                         perm_arr[4]);
  ::ml_drift::Transpose3DAttributes attr;
  attr.perm = perm;
  transpose_op->attr = std::move(attr);

  // Get intermediate shape
  const ::ml_drift::BHWDC input_shape =
      ir_model.tensor(tensor_map[input_id])->desc.GetBHWDCShape();
  const std::vector<int> input_shape_vec = {input_shape.b, input_shape.h,
                                            input_shape.w, input_shape.d,
                                            input_shape.c};
  std::array<int, 5> interim_shape_vec;
  for (int d = 0; d < 5; ++d) {
    interim_shape_vec[d] = input_shape_vec[perm_arr[d]];
  }
  const ::ml_drift::BHWDC interim_shape = ::ml_drift::BHWDC(
      interim_shape_vec[0], interim_shape_vec[1], interim_shape_vec[2],
      interim_shape_vec[3], interim_shape_vec[4]);

  const int output_id = node.outputs->data[0];
  const ::ml_drift::BHWDC output_shape =
      ir_model.tensor(tensor_map[output_id])->desc.GetBHWDCShape();

  if (interim_shape != output_shape) {
    ::ml_drift::ir::IrTensor* transpose_output_tensor = ir_model.add_tensor(
        ir_model.tensor(tensor_map[input_id])->desc.GetDataType(),
        interim_shape);
    ir_model.SetProducer(transpose_output_tensor->id, transpose_op->id);
    ::ml_drift::ir::IrOp* tile_op = ir_model.add_op();
    tile_op->name = ToString(::ml_drift::OperationType::TILE);
    ir_model.AddConsumer(transpose_output_tensor->id, tile_op->id);
    ir_model.SetProducer(tensor_map[output_id], tile_op->id);
  } else {
    ir_model.SetProducer(tensor_map[output_id], transpose_op->id);
  }
}

}  // namespace litert::ml_drift::ir
