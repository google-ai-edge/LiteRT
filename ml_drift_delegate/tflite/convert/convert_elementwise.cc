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

#include "ml_drift_delegate/tflite/convert/convert_elementwise.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"
namespace litert::ml_drift::ir {
namespace {
bool IsUnaryOp(const ::ml_drift::OperationType op_type) {
  switch (op_type) {
    case ::ml_drift::OperationType::ABS:
    case ::ml_drift::OperationType::CAST:
    case ::ml_drift::OperationType::CEIL:
    case ::ml_drift::OperationType::COPY:
    case ::ml_drift::OperationType::COS:
    case ::ml_drift::OperationType::ELU:
    case ::ml_drift::OperationType::EXP:
    case ::ml_drift::OperationType::FLOOR:
    case ::ml_drift::OperationType::GELU:
    case ::ml_drift::OperationType::HARD_SWISH:
    case ::ml_drift::OperationType::LOG:
    case ::ml_drift::OperationType::LOGICAL_NOT:
    case ::ml_drift::OperationType::NEG:
    case ::ml_drift::OperationType::ROUND:
    case ::ml_drift::OperationType::RSQRT:
    case ::ml_drift::OperationType::SIGMOID:
    case ::ml_drift::OperationType::SIGN:
    case ::ml_drift::OperationType::SIN:
    case ::ml_drift::OperationType::SQRT:
    case ::ml_drift::OperationType::SQUARE:
    case ::ml_drift::OperationType::TANH:
      return true;
    default:
      return false;
  }
}

bool IsBinaryOp(const ::ml_drift::OperationType op_type) {
  switch (op_type) {
    case ::ml_drift::OperationType::ADD:
    case ::ml_drift::OperationType::ATAN2:
    case ::ml_drift::OperationType::DIV:
    case ::ml_drift::OperationType::EQUAL:
    case ::ml_drift::OperationType::FLOOR_DIV:
    case ::ml_drift::OperationType::FLOOR_MOD:
    case ::ml_drift::OperationType::GREATER:
    case ::ml_drift::OperationType::GREATER_EQUAL:
    case ::ml_drift::OperationType::LESS:
    case ::ml_drift::OperationType::LESS_EQUAL:
    case ::ml_drift::OperationType::LOGICAL_AND:
    case ::ml_drift::OperationType::LOGICAL_OR:
    case ::ml_drift::OperationType::LOGICAL_XOR:
    case ::ml_drift::OperationType::MAXIMUM:
    case ::ml_drift::OperationType::MINIMUM:
    case ::ml_drift::OperationType::MUL:
    case ::ml_drift::OperationType::NOT_EQUAL:
    case ::ml_drift::OperationType::POW:
    case ::ml_drift::OperationType::REMAINDER:
    case ::ml_drift::OperationType::SHIFT_LEFT:
    case ::ml_drift::OperationType::SHIFT_RIGHT:
    case ::ml_drift::OperationType::SQUARED_DIFF:
    case ::ml_drift::OperationType::SUB:
      return true;
    default:
      return false;
  }
}
// Extracts shape from TfLiteTensor. And expand the shape to 5D right-aligned.
static ::ml_drift::BHWDC ExtractTensorShapeRightAligned(
    const TfLiteTensor* tflite_tensor) {
  const TfLiteIntArray* dims = tflite_tensor->dims;
  std::vector<int32_t> shape(dims->data, dims->data + dims->size);
  return GetRightAlignedBHWDC(shape, 1);
}

// Swap the inputs for MUL and ADD operations.
void SwapInputs(::ml_drift::OperationType operation_type,
                const TfLiteTensor* input0, const TfLiteTensor* input1,
                int* input_tensor0, int* input_tensor1) {
  if (operation_type != ::ml_drift::OperationType::MUL &&
      operation_type != ::ml_drift::OperationType::ADD) {
    return;
  }

  // The "larger" input tensor must be bound to 1st input and the
  // "smaller" input tensor must be bound to 2nd input.
  const ::ml_drift::BHWDC shape0 = ExtractTensorShapeRightAligned(input0);
  const ::ml_drift::BHWDC shape1 = ExtractTensorShapeRightAligned(input1);
  if (shape0.b <= shape1.b && shape0.d <= shape1.d && shape0.h <= shape1.h &&
      shape0.w <= shape1.w && shape0.c == shape1.c) {
    *input_tensor0 = 1;
    *input_tensor1 = 0;
  }
}

// Check if broadcast is needed.
bool IsBroadcastNeeded(const TfLiteContext& context,
                       const TfLiteNode& tflite_node,
                       const ::ml_drift::OperationType op_type) {
  if (!IsBinaryOp(op_type)) return false;
  const TfLiteTensor* input0 = context.tensors + tflite_node.inputs->data[0];
  const TfLiteTensor* input1 = context.tensors + tflite_node.inputs->data[1];
  if (::tflite::IsConstantTensor(input0) ^ ::tflite::IsConstantTensor(input1)) {
    return false;
  }
  if (input0 == input1) return false;

  // Check if the inputs are broadcastable and dimensions are
  // synchronized. If true, add reshapes nodes for the inputs and then
  // create the node. e.g. [1, 128, 512] + [128, 512] => [1, 1, 128, 512]
  // + [1, 1, 128, 512]
  if (!IsBroadcastable(input0->dims, input1->dims) ||
      input0->dims->size == input1->dims->size) {
    return false;
  }
  return true;
}
void ElementwiseFusedActivation(
    ::ml_drift::ir::IrModel& ir_model, const TfLiteNode& tflite_node,
    ::ml_drift::ir::IrOp* op,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    int output_id) {
  ::ml_drift::OperationType op_type =
      ::ml_drift::OperationTypeFromString(op->name);
  TfLiteFusedActivation activation = kTfLiteActNone;
  switch (op_type) {
    case ::ml_drift::OperationType::ADD:
      if (const auto* params =
              static_cast<const TfLiteAddParams*>(tflite_node.builtin_data)) {
        activation = params->activation;
      }
      break;
    case ::ml_drift::OperationType::DIV:
      if (const auto* params =
              static_cast<const TfLiteDivParams*>(tflite_node.builtin_data)) {
        activation = params->activation;
      }
      break;
    case ::ml_drift::OperationType::MUL:
      if (const auto* params =
              static_cast<const TfLiteMulParams*>(tflite_node.builtin_data)) {
        activation = params->activation;
      }
      break;
    case ::ml_drift::OperationType::SUB:
      if (const auto* params =
              static_cast<const TfLiteSubParams*>(tflite_node.builtin_data)) {
        activation = params->activation;
      }
      break;
    default:
      break;
  }
  HandleFusedActivation(activation, ir_model, op, tensor_map, output_id);
}

// Should only be called if IsBroadcastNeeded returns true.
// Inserts additional reshapes for broadcast.
void AddOpWithBroadcastReshape(
    const TfLiteContext& context, const TfLiteNode& tflite_node,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const ::ml_drift::OperationType operation_type,
    ::ml_drift::ir::IrModel& ir_model) {
  const TfLiteTensor* input0 = context.tensors + tflite_node.inputs->data[0];
  const TfLiteTensor* input1 = context.tensors + tflite_node.inputs->data[1];

  int input_tensor0 = 0;
  int input_tensor1 = 1;
  SwapInputs(operation_type, input0, input1, &input_tensor0, &input_tensor1);
  ::ml_drift::ir::IrTensorId input_id0 =
      tensor_map[tflite_node.inputs->data[0]];
  ::ml_drift::ir::IrTensorId input_id1 =
      tensor_map[tflite_node.inputs->data[1]];
  ::ml_drift::BHWDC input0_shape = ExtractTensorShapeRightAligned(input0);
  ::ml_drift::BHWDC input1_shape = ExtractTensorShapeRightAligned(input1);
  // Add reshape node for input0
  ::ml_drift::ir::IrOp* reshape_node0 = ir_model.add_op();
  reshape_node0->name = ToString(::ml_drift::OperationType::RESHAPE);
  ::ml_drift::Reshape3DAttributes reshape_attr;
  reshape_attr.new_shape = input0_shape;
  reshape_node0->attr = std::move(reshape_attr);
  ir_model.AddConsumer(input_id0, reshape_node0->id);
  ::ml_drift::ir::IrTensor* reshape_output_tensor = ir_model.add_tensor(
      ir_model.tensor(input_id0)->desc.GetDataType(), input0_shape);
  ir_model.SetProducer(reshape_output_tensor->id, reshape_node0->id);

  // Add reshape node for input1
  ::ml_drift::ir::IrOp* reshape_node1 = ir_model.add_op();
  reshape_node1->name = ToString(::ml_drift::OperationType::RESHAPE);
  ::ml_drift::Reshape3DAttributes reshape_attr1;
  reshape_attr1.new_shape = input1_shape;
  reshape_node1->attr = std::move(reshape_attr1);
  ir_model.AddConsumer(input_id1, reshape_node1->id);
  ::ml_drift::ir::IrTensor* reshape_output_tensor1 = ir_model.add_tensor(
      ir_model.tensor(input_id1)->desc.GetDataType(), input1_shape);
  ir_model.SetProducer(reshape_output_tensor1->id, reshape_node1->id);

  // Link the reshape node to the original node
  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(operation_type);
  ::ml_drift::ElementwiseAttributes attr;
  op->attr = std::move(attr);
  ir_model.AddConsumer(reshape_output_tensor->id, op->id);
  ir_model.AddConsumer(reshape_output_tensor1->id, op->id);

  // Create output value for the node.
  // Set the output shape to the tflite output tensor shape.
  const TfLiteTensor* output_tensor =
      context.tensors + tflite_node.outputs->data[0];
  const ::ml_drift::ir::IrTensorId output_id =
      tensor_map[tflite_node.outputs->data[0]];
  ::ml_drift::TensorDescriptor output_desc = ir_model.tensor(output_id)->desc;
  ::ml_drift::BHWDC tflite_style_output_shape =
      ExtractTensorShapeRightAligned(output_tensor);
  ::ml_drift::ir::IrTensor* output =
      ir_model.add_tensor(output_desc.GetDataType(), tflite_style_output_shape);

  // Add activation
  const int output_tflite_id = tflite_node.outputs->data[0];
  const ::ml_drift::ir::IrTensorId final_output_id =
      tensor_map[output_tflite_id];
  tensor_map[output_tflite_id] = output->id;
  ElementwiseFusedActivation(ir_model, tflite_node, op, tensor_map,
                             output_tflite_id);
  tensor_map[output_tflite_id] = final_output_id;

  // Reshape the output to the ml_drift output tensor shape.
  ::ml_drift::ir::IrOp* reshape = ir_model.add_op();
  reshape->name = ToString(::ml_drift::OperationType::RESHAPE);
  ir_model.AddConsumer(output->id, reshape->id);
  ir_model.SetProducer(output_id, reshape->id);
  ::ml_drift::Reshape3DAttributes output_reshape_attr;
  output_reshape_attr.new_shape = output_desc.GetBHWDCShape();
  reshape->attr = std::move(output_reshape_attr);
}

// Specialization of TfLiteTensorToTensor<Tensor<Scalar, DataType::FLOAT32>>.
static ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::FLOAT32>
ConvertToScalarFloat32Tensor(const TfLiteTensor* tfl_tensor) {
  ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::FLOAT32>
      mld_tensor;
  mld_tensor.data.resize(1);
  CopyFloat32Data(tfl_tensor, &mld_tensor.data[0]);
  mld_tensor.shape.v = 1;
  return mld_tensor;
}

// Specialization of TfLiteTensorToTensor<Tensor<Scalar, DataType::INT32>>.
static ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::INT32>
ConvertToScalarInt32Tensor(const TfLiteTensor* tfl_tensor) {
  const TfLiteType dtype = tfl_tensor->type;
  ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::INT32>
      mld_tensor;
  mld_tensor.data.push_back(dtype == kTfLiteFloat32  ? tfl_tensor->data.f[0]
                            : dtype == kTfLiteBool   ? tfl_tensor->data.b[0]
                            : dtype == kTfLiteInt8   ? tfl_tensor->data.int8[0]
                            : dtype == kTfLiteUInt8  ? tfl_tensor->data.uint8[0]
                            : dtype == kTfLiteInt16  ? tfl_tensor->data.i16[0]
                            : dtype == kTfLiteUInt16 ? tfl_tensor->data.ui16[0]
                                                     : tfl_tensor->data.i32[0]);
  mld_tensor.shape.v = 1;
  return mld_tensor;
}

// Specialization of TfLIteTensorToTensor<Tensor<Linear, DataType::FLOAT32>>.
static ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
ConvertToLinearFloat32Tensor(const TfLiteTensor* tfl_tensor) {
  ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
      mld_tensor;
  const int n = tflite::NumElements(tfl_tensor);
  mld_tensor.data.resize(n);
  CopyFloat32Data(tfl_tensor, &mld_tensor.data[0]);
  mld_tensor.shape.v = n;
  return mld_tensor;
}

// Specialization of TfLIteTensorToTensor<Tensor<::ml_drift::BHWC,
// DataType::FLOAT32>>.
static ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>
ConvertToBhwcFloat32Tensor(const TfLiteTensor* tfl_tensor) {
  ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>
      mld_tensor;
  mld_tensor.data.resize(tflite::NumElements(tfl_tensor));
  CopyFloat32Data(tfl_tensor, &mld_tensor.data[0]);
  const TfLiteIntArray* dims = tfl_tensor->dims;
  std::vector<int32_t> shape(dims->data, dims->data + dims->size);
  mld_tensor.shape = GetRightAlignedBHWC(shape, 1);
  return mld_tensor;
}

static ::ml_drift::Tensor<::ml_drift::BHWDC, ::ml_drift::DataType::FLOAT32>
ConvertToBhwdcFloat32Tensor(const TfLiteTensor* tfl_tensor) {
  ::ml_drift::Tensor<::ml_drift::BHWDC, ::ml_drift::DataType::FLOAT32>
      mld_tensor;
  mld_tensor.data.resize(tflite::NumElements(tfl_tensor));
  CopyFloat32Data(tfl_tensor, &mld_tensor.data[0]);
  const TfLiteIntArray* dims = tfl_tensor->dims;
  std::vector<int32_t> shape(dims->data, dims->data + dims->size);
  mld_tensor.shape = GetRightAlignedBHWDC(shape, 1);
  return mld_tensor;
}

void ParseInputsWithConstTensor(
    const TfLiteNode& tflite_node, ::ml_drift::ir::IrOp* op,
    const TfLiteContext& context, ::ml_drift::ir::IrModel& ir_model,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::TensorOrScalar* tensor_or_scalar) {
  // Determine runtime/constant tensors.
  const TfLiteTensor* input0 = context.tensors + tflite_node.inputs->data[0];
  const TfLiteTensor* input1 = context.tensors + tflite_node.inputs->data[1];
  const bool constant_tensor0 = tflite::IsConstantTensor(input0);

  // Create aliases for constant and runtime tensors.
  int runtime_tensor_index;
  int constant_tensor_index;
  const TfLiteTensor* constant_tensor;
  if (constant_tensor0) {
    runtime_tensor_index = 1;
    constant_tensor_index = 0;
    constant_tensor = input0;
  } else {
    runtime_tensor_index = 0;
    constant_tensor_index = 1;
    constant_tensor = input1;
  }

  ir_model.AddConsumer(
      tensor_map[tflite_node.inputs->data[runtime_tensor_index]], op->id);
  const TfLiteIntArray* constant_dims = constant_tensor->dims;
  const bool convertible_to_f32 =
      constant_tensor->type == kTfLiteFloat32 ||
      constant_tensor->type == kTfLiteFloat16 ||
      (constant_tensor->quantization.type ==
           TfLiteQuantizationType::kTfLiteAffineQuantization &&
       (constant_tensor->type == kTfLiteInt8 ||
        constant_tensor->type == kTfLiteUInt8 ||
        constant_tensor->type == kTfLiteInt4 ||
        constant_tensor->type == kTfLiteInt2));
  if (constant_dims->size < 1 || tflite::NumElements(constant_dims) == 1) {
    if (convertible_to_f32) {
      const ::ml_drift::Tensor<::ml_drift::Scalar,
                               ::ml_drift::DataType::FLOAT32>
          t = ConvertToScalarFloat32Tensor(constant_tensor);
      *tensor_or_scalar = t.data[0];
      return;
    }
    if (constant_tensor->type == kTfLiteInt32) {
      const ::ml_drift::Tensor<::ml_drift::Scalar, ::ml_drift::DataType::INT32>
          t = ConvertToScalarInt32Tensor(constant_tensor);
      *tensor_or_scalar = t.data[0];
      return;
    }
  }
  if (!convertible_to_f32) {
    ::ml_drift::ir::IrTensor* const_ir_tensor = AddConstInput(
        context, tflite_node.inputs->data[constant_tensor_index], ir_model,
        /*layout=*/{});
    ir_model.AddConsumer(const_ir_tensor->id, op->id);
    return;
  }
  if (IsLinearConvertible(constant_dims)) {
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>
        tensor = ConvertToLinearFloat32Tensor(constant_tensor);
    *tensor_or_scalar = std::move(tensor);
    return;
  }
  if (constant_dims->size <= 5) {
    const TfLiteTensor* output_tensor =
        context.tensors + tflite_node.outputs->data[0];
    if (output_tensor->dims->size == 5) {
      ::ml_drift::Tensor<::ml_drift::BHWDC, ::ml_drift::DataType::FLOAT32>
          tensor = ConvertToBhwdcFloat32Tensor(constant_tensor);
      *tensor_or_scalar = std::move(tensor);
    } else {
      ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>
          tensor = ConvertToBhwcFloat32Tensor(constant_tensor);
      *tensor_or_scalar = std::move(tensor);
    }
  }
}
}  // namespace

::ml_drift::OperationType GetElementwiseOperationType(int32_t builtin_code) {
  switch (builtin_code) {
    case kTfLiteBuiltinAbs:
      return ::ml_drift::OperationType::ABS;
    case kTfLiteBuiltinAtan2:
      return ::ml_drift::OperationType::ATAN2;
    case kTfLiteBuiltinCast:
      return ::ml_drift::OperationType::CAST;
    case kTfLiteBuiltinCeil:
      return ::ml_drift::OperationType::CEIL;
    case kTfLiteBuiltinCos:
      return ::ml_drift::OperationType::COS;
    case kTfLiteBuiltinElu:
      return ::ml_drift::OperationType::ELU;
    case kTfLiteBuiltinExp:
      return ::ml_drift::OperationType::EXP;
    case kTfLiteBuiltinFloor:
      return ::ml_drift::OperationType::FLOOR;
    case kTfLiteBuiltinGelu:
      return ::ml_drift::OperationType::GELU;
    case kTfLiteBuiltinHardSwish:
      return ::ml_drift::OperationType::HARD_SWISH;
    case kTfLiteBuiltinLog:
      return ::ml_drift::OperationType::LOG;
    case kTfLiteBuiltinLogistic:
      return ::ml_drift::OperationType::SIGMOID;
    case kTfLiteBuiltinNeg:
      return ::ml_drift::OperationType::NEG;
    case kTfLiteBuiltinRound:
      return ::ml_drift::OperationType::ROUND;
    case kTfLiteBuiltinRsqrt:
      return ::ml_drift::OperationType::RSQRT;
    case kTfLiteBuiltinSign:
      return ::ml_drift::OperationType::SIGN;
    case kTfLiteBuiltinSin:
      return ::ml_drift::OperationType::SIN;
    case kTfLiteBuiltinSqrt:
      return ::ml_drift::OperationType::SQRT;
    case kTfLiteBuiltinSquare:
      return ::ml_drift::OperationType::SQUARE;
    case kTfLiteBuiltinTanh:
      return ::ml_drift::OperationType::TANH;
    case kTfLiteBuiltinLogicalNot:
      return ::ml_drift::OperationType::LOGICAL_NOT;
    case kTfLiteBuiltinAdd:
      return ::ml_drift::OperationType::ADD;
    case kTfLiteBuiltinSub:
      return ::ml_drift::OperationType::SUB;
    case kTfLiteBuiltinMul:
      return ::ml_drift::OperationType::MUL;
    case kTfLiteBuiltinDiv:
      return ::ml_drift::OperationType::DIV;
    case kTfLiteBuiltinMaximum:
      return ::ml_drift::OperationType::MAXIMUM;
    case kTfLiteBuiltinMinimum:
      return ::ml_drift::OperationType::MINIMUM;
    case kTfLiteBuiltinPow:
      return ::ml_drift::OperationType::POW;
    case kTfLiteBuiltinEqual:
      return ::ml_drift::OperationType::EQUAL;
    case kTfLiteBuiltinNotEqual:
      return ::ml_drift::OperationType::NOT_EQUAL;
    case kTfLiteBuiltinGreater:
      return ::ml_drift::OperationType::GREATER;
    case kTfLiteBuiltinGreaterEqual:
      return ::ml_drift::OperationType::GREATER_EQUAL;
    case kTfLiteBuiltinLess:
      return ::ml_drift::OperationType::LESS;
    case kTfLiteBuiltinLessEqual:
      return ::ml_drift::OperationType::LESS_EQUAL;
    case kTfLiteBuiltinLogicalAnd:
      return ::ml_drift::OperationType::LOGICAL_AND;
    case kTfLiteBuiltinLogicalOr:
      return ::ml_drift::OperationType::LOGICAL_OR;
    case kTfLiteBuiltinBitwiseXor:
      return ::ml_drift::OperationType::LOGICAL_XOR;
    case kTfLiteBuiltinFloorDiv:
      return ::ml_drift::OperationType::FLOOR_DIV;
    case kTfLiteBuiltinFloorMod:
      return ::ml_drift::OperationType::FLOOR_MOD;
    case kTfLiteBuiltinStablehloRemainder:
      return ::ml_drift::OperationType::REMAINDER;
    case kTfLiteBuiltinRightShift:
      return ::ml_drift::OperationType::SHIFT_RIGHT;
    case kTfLiteBuiltinStablehloShiftLeft:
      return ::ml_drift::OperationType::SHIFT_LEFT;
    case kTfLiteBuiltinSquaredDifference:
      return ::ml_drift::OperationType::SQUARED_DIFF;
    default:
      ABSL_LOG(FATAL) << "Unsupported elementwise builtin_code "
                      << builtin_code;
      return ::ml_drift::OperationType::UNKNOWN;
  }
}

void ConvertElementwise(
    const TfLiteContext& context, const TfLiteNode& tflite_node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::OperationType op_type =
      GetElementwiseOperationType(registration.builtin_code);
  if (IsBroadcastNeeded(context, tflite_node, op_type)) {
    AddOpWithBroadcastReshape(context, tflite_node, tensor_map, op_type,
                              ir_model);
    return;
  }
  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(op_type);

  if (IsUnaryOp(op_type)) {
    if (op_type == ::ml_drift::OperationType::GELU &&
        tflite_node.builtin_data) {
      auto tflite_options =
          reinterpret_cast<const TfLiteGeluParams*>(tflite_node.builtin_data);
      if (tflite_options->approximate) {
        op->name = ToString(::ml_drift::OperationType::GELU_TANH_APPROX);
      }
    }
    ir_model.AddConsumer(tensor_map[tflite_node.inputs->data[0]], op->id);
  } else if (IsBinaryOp(op_type)) {
    const TfLiteTensor* input0 = context.tensors + tflite_node.inputs->data[0];
    const TfLiteTensor* input1 = context.tensors + tflite_node.inputs->data[1];
    if (!::tflite::IsConstantTensor(input0) &&
        !::tflite::IsConstantTensor(input1)) {  // both runtime inputs
      if (input0 == input1) {
        if (op_type == ::ml_drift::OperationType::MUL) {
          // replace MUL(A, A) with SQUARE(A)
          op->name = ToString(::ml_drift::OperationType::SQUARE);
          ir_model.AddConsumer(tensor_map[tflite_node.inputs->data[0]], op->id);
        } else if (op_type == ::ml_drift::OperationType::ADD) {
          // replace ADD(A, A) with MUL(A, 2.0)
          op->name = ToString(::ml_drift::OperationType::MUL);
          ::ml_drift::ElementwiseAttributes attr;
          attr.param = 2.0f;
          op->attr = std::move(attr);
          ir_model.AddConsumer(tensor_map[tflite_node.inputs->data[0]], op->id);
        }
      } else {
        int input_tensor0 = 0;
        int input_tensor1 = 1;
        SwapInputs(op_type, input0, input1, &input_tensor0, &input_tensor1);
        ir_model.AddConsumer(
            tensor_map[tflite_node.inputs->data[input_tensor0]], op->id);
        ir_model.AddConsumer(
            tensor_map[tflite_node.inputs->data[input_tensor1]], op->id);
      }
    } else {  // binary with one constant input
      ::ml_drift::ElementwiseAttributes attr;
      ParseInputsWithConstTensor(tflite_node, op, context, ir_model, tensor_map,
                                 &attr.param);
      attr.runtime_tensor_is_second = tflite::IsConstantTensor(input0);
      op->attr = std::move(attr);
    }
  }
  ElementwiseFusedActivation(ir_model, tflite_node, op, tensor_map,
                             tflite_node.outputs->data[0]);
}

}  // namespace litert::ml_drift::ir
