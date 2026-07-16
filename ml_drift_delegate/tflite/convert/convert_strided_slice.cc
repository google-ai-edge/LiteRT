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

#include "ml_drift_delegate/tflite/convert/convert_strided_slice.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertStridedSlice(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* slice_op = ir_model.add_op();
  slice_op->name = ToString(::ml_drift::OperationType::SLICE);

  const int input_id = node.inputs->data[0];
  const int begin_id = node.inputs->data[1];
  const int end_id = node.inputs->data[2];
  const int strides_id = node.inputs->data[3];

  ir_model.AddConsumer(tensor_map[input_id], slice_op->id);

  const TfLiteTensor& input_tensor = context.tensors[input_id];
  const TfLiteTensor& begin_tensor = context.tensors[begin_id];
  const TfLiteTensor& end_tensor = context.tensors[end_id];
  const TfLiteTensor& strides_tensor = context.tensors[strides_id];

  const auto* params =
      static_cast<const TfLiteStridedSliceParams*>(node.builtin_data);

  std::vector<int> starts(begin_tensor.data.i32,
                          begin_tensor.data.i32 + begin_tensor.dims->data[0]);
  std::vector<int> ends(end_tensor.data.i32,
                        end_tensor.data.i32 + end_tensor.dims->data[0]);
  std::vector<int> strides_vec(
      strides_tensor.data.i32,
      strides_tensor.data.i32 + strides_tensor.dims->data[0]);

  int begin_mask = params->begin_mask;
  int end_mask = params->end_mask;

  const int input_rank = input_tensor.dims->size;
  const int params_rank = starts.size();

  if (params_rank == input_rank - 1) {
    starts.insert(starts.begin(), 0);
    ends.insert(ends.begin(), input_tensor.dims->data[0]);
    strides_vec.insert(strides_vec.begin(), 1);
    begin_mask <<= 1;
    end_mask <<= 1;
  }

  ResolveNegativeIndices(*input_tensor.dims, starts);
  ResolveNegativeIndices(*input_tensor.dims, ends);
  UpdateWithMask(begin_mask, end_mask, *input_tensor.dims, starts, ends);

  if (input_rank == 5) {
    ::ml_drift::Slice3DAttributes attr;
    attr.starts = ::ml_drift::BHWDC(starts[0], starts[1], starts[2], starts[3],
                                    starts[4]);
    attr.ends = ::ml_drift::BHWDC(ends[0], ends[1], ends[2], ends[3], ends[4]);
    attr.strides =
        ::ml_drift::BHWDC(strides_vec[0], strides_vec[1], strides_vec[2],
                          strides_vec[3], strides_vec[4]);
    slice_op->attr = std::move(attr);
  } else if (input_rank == 4) {
    ::ml_drift::SliceAttributes attr;
    attr.starts = ::ml_drift::BHWC(starts[0], starts[1], starts[2], starts[3]);
    attr.ends = ::ml_drift::BHWC(ends[0], ends[1], ends[2], ends[3]);
    attr.strides = ::ml_drift::BHWC(strides_vec[0], strides_vec[1],
                                    strides_vec[2], strides_vec[3]);
    slice_op->attr = std::move(attr);
  } else if (input_rank == 3) {
    ::ml_drift::SliceAttributes attr;
    attr.starts = ::ml_drift::BHWC(starts[0], 0, starts[1], starts[2]);
    attr.ends = ::ml_drift::BHWC(ends[0], 1, ends[1], ends[2]);
    attr.strides =
        ::ml_drift::BHWC(strides_vec[0], 1, strides_vec[1], strides_vec[2]);
    slice_op->attr = std::move(attr);
  } else if (input_rank == 2) {
    ::ml_drift::SliceAttributes attr;
    attr.starts = ::ml_drift::BHWC(starts[0], 0, 0, starts[1]);
    attr.ends = ::ml_drift::BHWC(ends[0], 1, 1, ends[1]);
    attr.strides = ::ml_drift::BHWC(strides_vec[0], 1, 1, strides_vec[1]);
    slice_op->attr = std::move(attr);
  } else if (input_rank == 1) {
    ::ml_drift::SliceAttributes attr;
    attr.starts = ::ml_drift::BHWC(starts[0], 0, 0, 0);
    attr.ends = ::ml_drift::BHWC(ends[0], 1, 1, 1);
    attr.strides = ::ml_drift::BHWC(strides_vec[0], 1, 1, 1);
    slice_op->attr = std::move(attr);
  } else {
    ::ml_drift::SliceAttributes attr;
    attr.starts = ::ml_drift::BHWC(0, 0, 0, 0);
    attr.ends = ir_model.tensor(tensor_map[input_id])->desc.GetBHWCShape();
    attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
    slice_op->attr = std::move(attr);
  }
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], slice_op->id);
}

}  // namespace litert::ml_drift::ir
