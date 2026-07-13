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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_slice.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

namespace {

::ml_drift::BHWDC MapToBHWDC(const std::vector<int32_t>& values,
                             int32_t start_val) {
  const int size = values.size();
  if (size == 0) {
    return ::ml_drift::BHWDC(start_val, start_val, start_val, start_val,
                             start_val);
  } else if (size == 1) {
    return ::ml_drift::BHWDC(values[0], start_val, start_val, start_val,
                             start_val);
  } else if (size == 2) {
    return ::ml_drift::BHWDC(values[0], start_val, start_val, start_val,
                             values[1]);
  } else if (size == 3) {
    return ::ml_drift::BHWDC(values[0], start_val, values[1], start_val,
                             values[2]);
  } else if (size == 4) {
    return ::ml_drift::BHWDC(values[0], values[1], values[2], start_val,
                             values[3]);
  } else {
    return ::ml_drift::BHWDC(values[0], values[1], values[2], values[3],
                             values[4]);
  }
}

}  // namespace

void ConvertSlice(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* ir_op = ir_model.add_op();
  ir_op->name = ToString(::ml_drift::OperationType::SLICE);

  const int tfl_input_id = node.inputs->data[0];
  ::ml_drift::ir::IrTensorId input_id = tensor_map[tfl_input_id];
  ir_model.AddConsumer(input_id, ir_op->id);
  const int output_id = node.outputs->data[0];
  ir_model.SetProducer(tensor_map[output_id], ir_op->id);

  ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>
      starts_tensor;
  PopulateTensor(&context.tensors[node.inputs->data[1]], node.inputs->data[1],
                 &starts_tensor, PopulateTensorFlags::kNoExtraBytes);

  ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>
      sizes_tensor;
  PopulateTensor(&context.tensors[node.inputs->data[2]], node.inputs->data[2],
                 &sizes_tensor, PopulateTensorFlags::kNoExtraBytes);

  const ::ml_drift::BHWDC starts =
      MapToBHWDC(starts_tensor.data, /*start_val=*/0);
  ::ml_drift::BHWDC sizes = MapToBHWDC(sizes_tensor.data, /*start_val=*/1);

  const ::ml_drift::BHWDC in_shape =
      ir_model.tensor(input_id)->desc.GetBHWDCShape();

  if (sizes.b == -1) sizes.b = in_shape.b - starts.b;
  if (sizes.h == -1) sizes.h = in_shape.h - starts.h;
  if (sizes.w == -1) sizes.w = in_shape.w - starts.w;
  if (sizes.d == -1) sizes.d = in_shape.d - starts.d;
  if (sizes.c == -1) sizes.c = in_shape.c - starts.c;

  if (in_shape.d > 1 || starts.d != 0 || sizes.d != 1) {
    ::ml_drift::Slice3DAttributes attr;
    attr.starts =
        ::ml_drift::BHWDC(starts.b, starts.h, starts.w, starts.d, starts.c);
    attr.ends = ::ml_drift::BHWDC(starts.b + sizes.b, starts.h + sizes.h,
                                  starts.w + sizes.w, starts.d + sizes.d,
                                  starts.c + sizes.c);
    attr.strides = ::ml_drift::BHWDC(1, 1, 1, 1, 1);
    ir_op->attr = std::move(attr);
  } else {
    ::ml_drift::SliceAttributes attr;
    attr.starts = ::ml_drift::BHWC(starts.b, starts.h, starts.w, starts.c);
    attr.ends = ::ml_drift::BHWC(starts.b + sizes.b, starts.h + sizes.h,
                                 starts.w + sizes.w, starts.c + sizes.c);
    attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
    ir_op->attr = std::move(attr);
  }
}

}  // namespace litert::ml_drift::ir
