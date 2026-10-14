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

#include "ml_drift_delegate/tflite/convert/convert_pad.h"

#include <limits>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

void ConvertPad(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& options, ::ml_drift::ir::IrModel& ir_model) {
  const int tfl_input_id = node.inputs->data[0];
  const TfLiteTensor* input_tensor = context.tensors + tfl_input_id;
  ::ml_drift::ir::IrTensorId input_id;
  if (::tflite::IsConstantTensor(input_tensor)) {
    const bool read_float16_constants_as_float32 =
        !options.enable_reduced_precision &&
        context.tensors[node.outputs->data[0]].type == kTfLiteFloat32 &&
        input_tensor->type == kTfLiteFloat16;
    input_id = read_float16_constants_as_float32
                   ? AddFloat16ConstAsFloat32Input(context, tfl_input_id,
                                                   ir_model, {})
                         ->id
                   : AddConstInput(context, tfl_input_id, ir_model, {})->id;
  } else {
    input_id = tensor_map[tfl_input_id];
  }

  ::ml_drift::ir::IrOp* pad_op = ir_model.add_op();
  pad_op->name = ToString(::ml_drift::OperationType::kPad);

  ir_model.AddConsumer(input_id, pad_op->id);
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], pad_op->id);

  ::ml_drift::PadAttributes attr;
  if (registration.builtin_code == kTfLiteBuiltinMirrorPad) {
    attr.type = ::ml_drift::PaddingContentType::kReflect;
  } else {
    attr.type = ::ml_drift::PaddingContentType::kZeros;
  }

  ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::kInt32> paddings;
  PopulateTensor(&context.tensors[node.inputs->data[1]], node.inputs->data[1],
                 &paddings, PopulateTensorFlags::kNoExtraBytes);

  if (registration.builtin_code == kTfLiteBuiltinPadv2 &&
      node.inputs->size == 3) {
    ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::kFloat32>
        const_tensor;
    PopulateTensor(&context.tensors[node.inputs->data[2]], node.inputs->data[2],
                   &const_tensor, PopulateTensorFlags::kNoExtraBytes);
    attr.constant_values = const_tensor.data[0];
    if (options.enable_infinite_float_capping) {
      const bool input_is_fp16 =
          context.tensors[node.inputs->data[0]].type == kTfLiteFloat16;
      const bool use_half = options.enable_reduced_precision || input_is_fp16;
      if (attr.constant_values == std::numeric_limits<float>::infinity()) {
        attr.constant_values =
            use_half ? ::ml_drift::kMaxHalf : std::numeric_limits<float>::max();
      } else if (attr.constant_values ==
                 -std::numeric_limits<float>::infinity()) {
        attr.constant_values = use_half ? -::ml_drift::kMaxHalf
                                        : std::numeric_limits<float>::lowest();
      }
    }
  }

  if (paddings.shape.h == 4 && paddings.shape.w == 2) {
    // 4x2 tensor with paddings.
    attr.prepended = ::ml_drift::BHWC(paddings.data[0], paddings.data[2],
                                      paddings.data[4], paddings.data[6]);
    attr.appended = ::ml_drift::BHWC(paddings.data[1], paddings.data[3],
                                     paddings.data[5], paddings.data[7]);
  } else if (paddings.shape.h == 3 && paddings.shape.w == 2) {
    // 3x2 tensor with paddings.
    attr.prepended = ::ml_drift::BHWC(1, paddings.data[0], paddings.data[2],
                                      paddings.data[4]);
    attr.appended = ::ml_drift::BHWC(1, paddings.data[1], paddings.data[3],
                                     paddings.data[5]);
  }
  pad_op->attr = std::move(attr);
}

}  // namespace litert::ml_drift::ir
