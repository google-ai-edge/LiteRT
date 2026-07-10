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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"

#include <any>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

TEST(ConvertAuxTest, HandleFusedActivationNone) {
  ::ml_drift::ir::IrModel model;
  ::ml_drift::ir::IrTensor* t =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  t->id = 0;
  ::ml_drift::ir::IrOp* op = model.add_op();
  op->id = 0;
  absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId> tensor_map;
  tensor_map[1] = t->id;
  HandleFusedActivation(kTfLiteActNone, model, op, tensor_map, 1);
  EXPECT_EQ(t->producer, op->id);
  EXPECT_EQ(model.ops().size(), 1);
  EXPECT_EQ(model.tensors().size(), 1);
}

// input->op->output
// transformed to:
// input->op->activation_input->activation_op->output
TEST(ConvertAuxTest, HandleFusedActivationRelu) {
  ::ml_drift::ir::IrModel model;
  ::ml_drift::ir::IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrOp* op = model.add_op();
  model.AddConsumer(input->id, op->id);
  absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId> tensor_map;
  tensor_map[1] = output->id;
  HandleFusedActivation(kTfLiteActRelu, model, op, tensor_map, 1);
  EXPECT_EQ(model.ops().size(), 2);
  EXPECT_EQ(model.tensors().size(), 3);
  // Check input tensor.
  EXPECT_EQ(*input->consumers.begin(), op->id);
  // Check op
  EXPECT_EQ(op->inputs[0], input->id);
  const ::ml_drift::ir::IrTensor* activation_input = model.tensor(2);
  EXPECT_EQ(op->outputs[0], activation_input->id);
  // Check activation_input tensor
  const ::ml_drift::ir::IrOp* activation_op = model.op(1);
  EXPECT_EQ(activation_op->name, "relu");
  EXPECT_EQ(activation_input->producer, op->id);
  EXPECT_EQ(*activation_input->consumers.begin(), activation_op->id);
  // Check activation op
  EXPECT_EQ(activation_op->inputs[0], activation_input->id);
  EXPECT_EQ(activation_op->outputs[0], output->id);
  // Check output tensor.
  EXPECT_EQ(output->producer, activation_op->id);
  // Check activation op attributes.
  const ::ml_drift::ReLUAttributes* attr =
      std::any_cast<::ml_drift::ReLUAttributes>(&activation_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->activation_min, 0.0f);
  EXPECT_EQ(attr->activation_max, 0.0f);
}

TEST(ConvertAuxTest, HandleFusedActivationRelu6) {
  ::ml_drift::ir::IrModel model;
  ::ml_drift::ir::IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrOp* op = model.add_op();
  model.AddConsumer(input->id, op->id);
  absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId> tensor_map;
  tensor_map[1] = output->id;
  HandleFusedActivation(kTfLiteActRelu6, model, op, tensor_map, 1);
  EXPECT_EQ(model.ops().size(), 2);
  EXPECT_EQ(model.tensors().size(), 3);
  // Check input tensor.
  EXPECT_EQ(*input->consumers.begin(), op->id);
  // Check op
  EXPECT_EQ(op->inputs[0], input->id);
  const ::ml_drift::ir::IrTensor* activation_input = model.tensor(2);
  EXPECT_EQ(op->outputs[0], activation_input->id);
  // Check activation_input tensor
  const ::ml_drift::ir::IrOp* activation_op = model.op(1);
  EXPECT_EQ(activation_op->name, "relu");
  EXPECT_EQ(activation_input->producer, op->id);
  EXPECT_EQ(*activation_input->consumers.begin(), activation_op->id);
  // Check activation op
  EXPECT_EQ(activation_op->inputs[0], activation_input->id);
  EXPECT_EQ(activation_op->outputs[0], output->id);
  // Check output tensor.
  EXPECT_EQ(output->producer, activation_op->id);
  // Check activation op attributes.
  const ::ml_drift::ReLUAttributes* attr =
      std::any_cast<::ml_drift::ReLUAttributes>(&activation_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->activation_min, 0.0f);
  EXPECT_EQ(attr->activation_max, 6.0f);
}

TEST(ConvertAuxTest, HandleFusedActivationReluN1To1) {
  ::ml_drift::ir::IrModel model;
  ::ml_drift::ir::IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrOp* op = model.add_op();
  model.AddConsumer(input->id, op->id);
  absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId> tensor_map;
  tensor_map[1] = output->id;
  HandleFusedActivation(kTfLiteActReluN1To1, model, op, tensor_map, 1);
  EXPECT_EQ(model.ops().size(), 2);
  EXPECT_EQ(model.tensors().size(), 3);
  // Check input tensor.
  EXPECT_EQ(*input->consumers.begin(), op->id);
  // Check op
  EXPECT_EQ(op->inputs[0], input->id);
  const ::ml_drift::ir::IrTensor* activation_input = model.tensor(2);
  EXPECT_EQ(op->outputs[0], activation_input->id);
  // Check activation_input tensor
  const ::ml_drift::ir::IrOp* activation_op = model.op(1);
  EXPECT_EQ(activation_op->name, "relu");
  EXPECT_EQ(activation_input->producer, op->id);
  EXPECT_EQ(*activation_input->consumers.begin(), activation_op->id);
  // Check activation op
  EXPECT_EQ(activation_op->inputs[0], activation_input->id);
  EXPECT_EQ(activation_op->outputs[0], output->id);
  // Check output tensor.
  EXPECT_EQ(output->producer, activation_op->id);
  // Check activation op attributes.
  const ::ml_drift::ReLUAttributes* attr =
      std::any_cast<::ml_drift::ReLUAttributes>(&activation_op->attr);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr->activation_min, -1.0f);
  EXPECT_EQ(attr->activation_max, 1.0f);
}

TEST(ConvertAuxTest, HandleFusedActivationTanh) {
  ::ml_drift::ir::IrModel model;
  ::ml_drift::ir::IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrOp* op = model.add_op();
  model.AddConsumer(input->id, op->id);
  absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId> tensor_map;
  tensor_map[1] = output->id;
  HandleFusedActivation(kTfLiteActTanh, model, op, tensor_map, 1);
  EXPECT_EQ(model.ops().size(), 2);
  EXPECT_EQ(model.tensors().size(), 3);
  // Check input tensor.
  EXPECT_EQ(*input->consumers.begin(), op->id);
  // Check op
  EXPECT_EQ(op->inputs[0], input->id);
  const ::ml_drift::ir::IrTensor* activation_input = model.tensor(2);
  EXPECT_EQ(op->outputs[0], activation_input->id);
  // Check activation_input tensor
  const ::ml_drift::ir::IrOp* activation_op = model.op(1);
  EXPECT_EQ(activation_op->name, "tanh");
  EXPECT_EQ(activation_input->producer, op->id);
  EXPECT_EQ(*activation_input->consumers.begin(), activation_op->id);
  // Check activation op
  EXPECT_EQ(activation_op->inputs[0], activation_input->id);
  EXPECT_EQ(activation_op->outputs[0], output->id);
  // Check output tensor.
  EXPECT_EQ(output->producer, activation_op->id);
}

TEST(ConvertAuxTest, HandleFusedActivationSigmoid) {
  ::ml_drift::ir::IrModel model;
  ::ml_drift::ir::IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrOp* op = model.add_op();
  model.AddConsumer(input->id, op->id);
  absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId> tensor_map;
  tensor_map[1] = output->id;
  HandleFusedActivation(kTfLiteActSigmoid, model, op, tensor_map, 1);
  EXPECT_EQ(model.ops().size(), 2);
  EXPECT_EQ(model.tensors().size(), 3);
  // Check input tensor.
  EXPECT_EQ(*input->consumers.begin(), op->id);
  // Check op
  EXPECT_EQ(op->inputs[0], input->id);
  const ::ml_drift::ir::IrTensor* activation_input = model.tensor(2);
  EXPECT_EQ(op->outputs[0], activation_input->id);
  // Check activation_input tensor
  const ::ml_drift::ir::IrOp* activation_op = model.op(1);
  EXPECT_EQ(activation_op->name, "sigmoid");
  EXPECT_EQ(activation_input->producer, op->id);
  EXPECT_EQ(*activation_input->consumers.begin(), activation_op->id);
  // Check activation op
  EXPECT_EQ(activation_op->inputs[0], activation_input->id);
  EXPECT_EQ(activation_op->outputs[0], output->id);
  // Check output tensor.
  EXPECT_EQ(output->producer, activation_op->id);
}

TEST(ConvertAuxTest, HandleFusedActivationSignBit) {
  ::ml_drift::ir::IrModel model;
  ::ml_drift::ir::IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  ::ml_drift::ir::IrOp* op = model.add_op();
  model.AddConsumer(input->id, op->id);
  absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId> tensor_map;
  tensor_map[1] = output->id;
  HandleFusedActivation(kTfLiteActSignBit, model, op, tensor_map, 1);
  EXPECT_EQ(model.ops().size(), 2);
  EXPECT_EQ(model.tensors().size(), 3);
  // Check input tensor.
  EXPECT_EQ(*input->consumers.begin(), op->id);
  // Check op
  EXPECT_EQ(op->inputs[0], input->id);
  const ::ml_drift::ir::IrTensor* activation_input = model.tensor(2);
  EXPECT_EQ(op->outputs[0], activation_input->id);
  // Check activation_input tensor
  const ::ml_drift::ir::IrOp* activation_op = model.op(1);
  EXPECT_EQ(activation_op->name, "sign");
  EXPECT_EQ(activation_input->producer, op->id);
  EXPECT_EQ(*activation_input->consumers.begin(), activation_op->id);
  // Check activation op
  EXPECT_EQ(activation_op->inputs[0], activation_input->id);
  EXPECT_EQ(activation_op->outputs[0], output->id);
  // Check output tensor.
  EXPECT_EQ(output->producer, activation_op->id);
}

TEST(ConvertAuxTest, PopulateTensorCopy) {
  ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>
      dst_tensor;
  TfLiteTensor tfl_tensor;
  tfl_tensor.type = kTfLiteFloat32;
  tfl_tensor.dims = TfLiteIntArrayCreate(4);
  tfl_tensor.dims->data[0] = 1;
  tfl_tensor.dims->data[1] = 2;
  tfl_tensor.dims->data[2] = 1;
  tfl_tensor.dims->data[3] = 2;
  std::vector<float> tensor_data = {1.0f, 2.0f, 3.0f, 4.0f};
  tfl_tensor.data.f = tensor_data.data();
  tfl_tensor.bytes = tensor_data.size() * sizeof(float);

  TfLiteContext context;
  context.tensors = &tfl_tensor;
  TfLiteNode node;
  node.inputs = TfLiteIntArrayCreate(1);
  node.inputs->data[0] = 0;

  PopulateTensor(&tfl_tensor, 0, &dst_tensor,
                 PopulateTensorFlags::kNoExtraBytes,
                 /*enable_spanned_weights=*/false);

  EXPECT_EQ(dst_tensor.shape, ::ml_drift::BHWC(1, 2, 1, 2));
  EXPECT_EQ(dst_tensor.id, 0);
  EXPECT_THAT(dst_tensor.data, testing::ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteIntArrayFree(node.inputs);
}

TEST(ConvertAuxTest, AddConstInput) {
  TfLiteTensor tfl_tensor;
  tfl_tensor.type = kTfLiteFloat32;
  tfl_tensor.dims = TfLiteIntArrayCreate(4);
  tfl_tensor.dims->data[0] = 1;
  tfl_tensor.dims->data[1] = 2;
  tfl_tensor.dims->data[2] = 1;
  tfl_tensor.dims->data[3] = 2;
  std::vector<float> tensor_data = {1.0f, 2.0f, 3.0f, 4.0f};
  tfl_tensor.data.f = tensor_data.data();
  tfl_tensor.bytes = tensor_data.size() * sizeof(float);
  tfl_tensor.allocation_type = kTfLiteMmapRo;

  TfLiteContext context;
  context.tensors = &tfl_tensor;

  ::ml_drift::ir::IrModel model;
  SizedLayout layout;
  ::ml_drift::ir::IrTensor* tensor = AddConstInput(context, 0, model, layout);

  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->desc.GetDataType(), ::ml_drift::DataType::FLOAT32);
  EXPECT_EQ(tensor->desc.GetBHWCShape(), ::ml_drift::BHWC(1, 2, 1, 2));

  // Check the op produced it
  ASSERT_EQ(model.ops().size(), 1);
  const ::ml_drift::ir::IrOp* op = model.op(0);
  EXPECT_EQ(op->name, "const");
  EXPECT_EQ(op->outputs[0], tensor->id);
  EXPECT_EQ(tensor->producer, op->id);

  const ::ml_drift::ConstTensorAttributes* attr =
      std::any_cast<::ml_drift::ConstTensorAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* t = std::get_if<::ml_drift::TensorFloat32>(&attr->tensor);
  ASSERT_TRUE(t);
  EXPECT_THAT(t->data, testing::ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));

  TfLiteIntArrayFree(tfl_tensor.dims);
}

TEST(ConvertAuxTest, AddConstInputInt32) {
  TfLiteTensor tfl_tensor;
  tfl_tensor.type = kTfLiteInt32;
  tfl_tensor.dims = TfLiteIntArrayCreate(1);
  tfl_tensor.dims->data[0] = 2;
  std::vector<int32_t> tensor_data = {10, 20};
  tfl_tensor.data.i32 = tensor_data.data();
  tfl_tensor.bytes = tensor_data.size() * sizeof(int32_t);
  tfl_tensor.allocation_type = kTfLiteMmapRo;

  TfLiteContext context;
  context.tensors = &tfl_tensor;

  ::ml_drift::ir::IrModel model;
  SizedLayout layout;
  ::ml_drift::ir::IrTensor* tensor = AddConstInput(context, 0, model, layout);

  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->desc.GetDataType(), ::ml_drift::DataType::INT32);
  EXPECT_EQ(tensor->desc.GetBHWCShape(), ::ml_drift::BHWC(2, 1, 1, 1));

  const ::ml_drift::ir::IrOp* op = model.op(0);
  const ::ml_drift::ConstTensorAttributes* attr =
      std::any_cast<::ml_drift::ConstTensorAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* t = std::get_if<::ml_drift::TensorInt32>(&attr->tensor);
  ASSERT_TRUE(t);
  EXPECT_THAT(t->data, testing::ElementsAre(10, 20));

  TfLiteIntArrayFree(tfl_tensor.dims);
}

TEST(ConvertAuxTest, AddConstInputBool) {
  TfLiteTensor tfl_tensor;
  tfl_tensor.type = kTfLiteBool;
  tfl_tensor.dims = TfLiteIntArrayCreate(1);
  tfl_tensor.dims->data[0] = 2;
  std::vector<bool> tensor_data = {true, false};
  // TfLiteBool is often char or bool.
  std::vector<uint8_t> tfl_bool_data = {1, 0};
  tfl_tensor.data.uint8 = tfl_bool_data.data();
  tfl_tensor.bytes = tfl_bool_data.size() * sizeof(uint8_t);
  tfl_tensor.allocation_type = kTfLiteMmapRo;

  TfLiteContext context;
  context.tensors = &tfl_tensor;

  ::ml_drift::ir::IrModel model;
  SizedLayout layout;
  ::ml_drift::ir::IrTensor* tensor = AddConstInput(context, 0, model, layout);

  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->desc.GetDataType(), ::ml_drift::DataType::BOOL);
  EXPECT_EQ(tensor->desc.GetBHWCShape(), ::ml_drift::BHWC(2, 1, 1, 1));

  const ::ml_drift::ir::IrOp* op = model.op(0);
  const ::ml_drift::ConstTensorAttributes* attr =
      std::any_cast<::ml_drift::ConstTensorAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* t = std::get_if<::ml_drift::TensorBool>(&attr->tensor);
  ASSERT_TRUE(t);
  EXPECT_THAT(t->data, testing::ElementsAre(1, 0));

  TfLiteIntArrayFree(tfl_tensor.dims);
}

TEST(ConvertAuxTest, AddConstInputInt8) {
  TfLiteTensor tfl_tensor;
  tfl_tensor.type = kTfLiteInt8;
  tfl_tensor.dims = TfLiteIntArrayCreate(1);
  tfl_tensor.dims->data[0] = 2;
  std::vector<int8_t> tensor_data = {10, 20};
  tfl_tensor.data.int8 = tensor_data.data();
  tfl_tensor.bytes = tensor_data.size() * sizeof(int8_t);
  tfl_tensor.allocation_type = kTfLiteMmapRo;
  tfl_tensor.params.scale = 1.0f;
  tfl_tensor.params.zero_point = 0;

  tfl_tensor.quantization.type = kTfLiteAffineQuantization;
  TfLiteAffineQuantization quant_params_storage;
  tfl_tensor.quantization.params = &quant_params_storage;
  quant_params_storage.scale = TfLiteFloatArrayCreate(1);
  quant_params_storage.scale->data[0] = 1.0f;
  quant_params_storage.zero_point = TfLiteIntArrayCreate(1);
  quant_params_storage.zero_point->data[0] = 0;
  quant_params_storage.quantized_dimension = 0;

  TfLiteContext context;
  context.tensors = &tfl_tensor;

  ::ml_drift::ir::IrModel model;
  SizedLayout layout;
  ::ml_drift::ir::IrTensor* tensor = AddConstInput(context, 0, model, layout);

  ASSERT_NE(tensor, nullptr);
  // Note: kTfLiteInt8 is currently read as TensorFloat32.
  EXPECT_EQ(tensor->desc.GetDataType(), ::ml_drift::DataType::FLOAT32);
  EXPECT_EQ(tensor->desc.GetBHWCShape(), ::ml_drift::BHWC(2, 1, 1, 1));

  const ::ml_drift::ir::IrOp* op = model.op(0);
  const ::ml_drift::ConstTensorAttributes* attr =
      std::any_cast<::ml_drift::ConstTensorAttributes>(&op->attr);
  ASSERT_TRUE(attr);
  const auto* t = std::get_if<::ml_drift::TensorFloat32>(&attr->tensor);
  ASSERT_TRUE(t);
  EXPECT_THAT(t->data, testing::ElementsAre(10.0f, 20.0f));

  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteFloatArrayFree(quant_params_storage.scale);
  TfLiteIntArrayFree(quant_params_storage.zero_point);
}

TEST(ConvertAuxTest, PopulateTensorZeroCopy) {
  ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>
      dst_tensor;
  TfLiteTensor tfl_tensor;
  tfl_tensor.type = kTfLiteFloat32;
  tfl_tensor.dims = TfLiteIntArrayCreate(4);
  tfl_tensor.dims->data[0] = 1;
  tfl_tensor.dims->data[1] = 2;
  tfl_tensor.dims->data[2] = 1;
  tfl_tensor.dims->data[3] = 2;
  std::vector<float> tensor_data = {1.0f, 2.0f, 3.0f, 4.0f};
  tfl_tensor.data.f = tensor_data.data();
  tfl_tensor.bytes = tensor_data.size() * sizeof(float);

  TfLiteContext context;
  context.tensors = &tfl_tensor;
  TfLiteNode node;
  node.inputs = TfLiteIntArrayCreate(1);
  node.inputs->data[0] = 0;

  PopulateTensor(&tfl_tensor, 0, &dst_tensor,
                 PopulateTensorFlags::kNoExtraBytes,
                 /*enable_spanned_weights=*/true);

  EXPECT_EQ(dst_tensor.shape, ::ml_drift::BHWC(1, 2, 1, 2));
  EXPECT_EQ(dst_tensor.id, 0);
  EXPECT_TRUE(dst_tensor.data.empty());
  EXPECT_THAT(dst_tensor.spanned_data,
              testing::ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteIntArrayFree(node.inputs);
}

TEST(ConvertAuxTest, PopulateTensorCopyF16) {
  ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT16>
      dst_tensor;
  TfLiteTensor tfl_tensor;
  tfl_tensor.type = kTfLiteFloat16;
  tfl_tensor.dims = TfLiteIntArrayCreate(4);
  tfl_tensor.dims->data[0] = 1;
  tfl_tensor.dims->data[1] = 1;
  tfl_tensor.dims->data[2] = 1;
  tfl_tensor.dims->data[3] = 2;
  std::vector<::ml_drift::half> tensor_data = {::ml_drift::half(1.0f),
                                               ::ml_drift::half(2.0f)};
  tfl_tensor.data.f16 = reinterpret_cast<TfLiteFloat16*>(tensor_data.data());
  tfl_tensor.bytes = tensor_data.size() * sizeof(::ml_drift::half);

  TfLiteContext context;
  context.tensors = &tfl_tensor;
  TfLiteNode node;
  node.inputs = TfLiteIntArrayCreate(1);
  node.inputs->data[0] = 0;

  PopulateTensor(&tfl_tensor, 0, &dst_tensor,
                 PopulateTensorFlags::kNoExtraBytes,
                 /*enable_spanned_weights=*/false);

  EXPECT_EQ(dst_tensor.shape, ::ml_drift::BHWC(1, 1, 1, 2));
  EXPECT_EQ(dst_tensor.id, 0);
  EXPECT_THAT(dst_tensor.data, testing::ElementsAre(::ml_drift::half(1.0f),
                                                    ::ml_drift::half(2.0f)));
  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteIntArrayFree(node.inputs);
}

TEST(ConvertAuxTest, PopulateTensorZeroCopyF16DeathTest) {
  ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT16>
      dst_tensor;
  TfLiteTensor tfl_tensor;
  tfl_tensor.type = kTfLiteFloat16;
  tfl_tensor.dims = TfLiteIntArrayCreate(4);
  tfl_tensor.dims->data[0] = 1;
  tfl_tensor.dims->data[1] = 1;
  tfl_tensor.dims->data[2] = 1;
  tfl_tensor.dims->data[3] = 2;
  std::vector<::ml_drift::half> tensor_data = {::ml_drift::half(1.0f),
                                               ::ml_drift::half(2.0f)};
  tfl_tensor.data.f16 = reinterpret_cast<TfLiteFloat16*>(tensor_data.data());
  tfl_tensor.bytes = tensor_data.size() * sizeof(::ml_drift::half);

  TfLiteContext context;
  context.tensors = &tfl_tensor;
  TfLiteNode node;
  node.inputs = TfLiteIntArrayCreate(1);
  node.inputs->data[0] = 0;

  EXPECT_DEATH(PopulateTensor(&tfl_tensor, 0, &dst_tensor,
                              PopulateTensorFlags::kNoExtraBytes,
                              /*enable_spanned_weights=*/true),
               "Unsupported type for zero-copy: float16");
  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteIntArrayFree(node.inputs);
}

template <::ml_drift::DataType T>
void RunQuantizationCopyTest() {
  ::ml_drift::Tensor<::ml_drift::BHWC, T> dst_tensor;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32> scale;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32> zero_point;
  TfLiteTensor tfl_tensor;
  tfl_tensor.type = kTfLiteInt8;
  tfl_tensor.dims = TfLiteIntArrayCreate(4);
  tfl_tensor.dims->data[0] = 1;
  tfl_tensor.dims->data[1] = 1;
  tfl_tensor.dims->data[2] = 1;
  tfl_tensor.dims->data[3] = 2;
  std::vector<std::int8_t> tensor_data = {1, 2};
  tfl_tensor.data.int8 = tensor_data.data();
  tfl_tensor.bytes = tensor_data.size() * sizeof(std::int8_t);

  tfl_tensor.quantization.type = kTfLiteAffineQuantization;
  TfLiteAffineQuantization quant_params_storage;
  tfl_tensor.quantization.params = &quant_params_storage;
  quant_params_storage.scale = TfLiteFloatArrayCreate(2);
  quant_params_storage.scale->data[0] = 0.1f;
  quant_params_storage.scale->data[1] = 0.2f;
  quant_params_storage.zero_point = TfLiteIntArrayCreate(2);
  quant_params_storage.zero_point->data[0] = 1;
  quant_params_storage.zero_point->data[1] = 2;
  quant_params_storage.quantized_dimension = 0;

  TfLiteContext context;
  context.tensors = &tfl_tensor;
  TfLiteNode node;
  node.inputs = TfLiteIntArrayCreate(1);
  node.inputs->data[0] = 0;

  PopulateTensor(&tfl_tensor, 0, &dst_tensor,
                 PopulateTensorFlags::kNoExtraBytes,
                 /*enable_spanned_weights=*/false, &scale, &zero_point);

  EXPECT_EQ(dst_tensor.shape, ::ml_drift::BHWC(1, 1, 1, 2));
  EXPECT_EQ(dst_tensor.id, 0);
  EXPECT_THAT(dst_tensor.data, testing::ElementsAre(1, 2));
  EXPECT_EQ(scale.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_THAT(scale.data, testing::ElementsAre(0.1f, 0.2f));
  EXPECT_EQ(zero_point.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_THAT(zero_point.data, testing::ElementsAre(1, 2));

  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteIntArrayFree(node.inputs);
  TfLiteFloatArrayFree(quant_params_storage.scale);
  TfLiteIntArrayFree(quant_params_storage.zero_point);
}

template <::ml_drift::DataType T>
void RunQuantizationZeroCopyTest() {
  ::ml_drift::Tensor<::ml_drift::BHWC, T> dst_tensor;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32> scale;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32> zero_point;
  TfLiteTensor tfl_tensor;
  tfl_tensor.type = kTfLiteInt8;
  tfl_tensor.dims = TfLiteIntArrayCreate(4);
  tfl_tensor.dims->data[0] = 1;
  tfl_tensor.dims->data[1] = 1;
  tfl_tensor.dims->data[2] = 1;
  tfl_tensor.dims->data[3] = 2;
  std::vector<std::int8_t> tensor_data = {1, 2};
  tfl_tensor.data.int8 = tensor_data.data();
  tfl_tensor.bytes = tensor_data.size() * sizeof(std::int8_t);

  tfl_tensor.quantization.type = kTfLiteAffineQuantization;
  TfLiteAffineQuantization quant_params_storage;
  tfl_tensor.quantization.params = &quant_params_storage;
  quant_params_storage.scale = TfLiteFloatArrayCreate(2);
  quant_params_storage.scale->data[0] = 0.1f;
  quant_params_storage.scale->data[1] = 0.2f;
  quant_params_storage.zero_point = TfLiteIntArrayCreate(2);
  quant_params_storage.zero_point->data[0] = 1;
  quant_params_storage.zero_point->data[1] = 2;
  quant_params_storage.quantized_dimension = 0;

  TfLiteContext context;
  context.tensors = &tfl_tensor;
  TfLiteNode node;
  node.inputs = TfLiteIntArrayCreate(1);
  node.inputs->data[0] = 0;

  PopulateTensor(&tfl_tensor, 0, &dst_tensor,
                 PopulateTensorFlags::kNoExtraBytes,
                 /*enable_spanned_weights=*/true, &scale, &zero_point);

  EXPECT_EQ(dst_tensor.shape, ::ml_drift::BHWC(1, 1, 1, 2));
  EXPECT_EQ(dst_tensor.id, 0);
  EXPECT_TRUE(dst_tensor.data.empty());
  EXPECT_THAT(dst_tensor.spanned_data, testing::ElementsAre(1, 2));
  EXPECT_EQ(scale.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_TRUE(scale.data.empty());
  EXPECT_THAT(scale.spanned_data, testing::ElementsAre(0.1f, 0.2f));
  EXPECT_EQ(zero_point.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_TRUE(zero_point.data.empty());
  EXPECT_THAT(zero_point.spanned_data, testing::ElementsAre(1, 2));

  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteIntArrayFree(node.inputs);
  TfLiteFloatArrayFree(quant_params_storage.scale);
  TfLiteIntArrayFree(quant_params_storage.zero_point);
}

struct QuantizationTestName {
  template <class ParamType>
  std::string operator()(
      const ::testing::TestParamInfo<ParamType>& info) const {
    return std::string(ToString(info.param));
  }
};

class PopulateTensorQuantizationTest
    : public ::testing::TestWithParam<::ml_drift::DataType> {};

INSTANTIATE_TEST_SUITE_P(PopulateTensorQuantizationTests,
                         PopulateTensorQuantizationTest,
                         ::testing::Values(::ml_drift::DataType::INT8,
                                           ::ml_drift::DataType::INT4),
                         QuantizationTestName());

TEST_P(PopulateTensorQuantizationTest, Copy) {
  if (GetParam() == ::ml_drift::DataType::INT8) {
    RunQuantizationCopyTest<::ml_drift::DataType::INT8>();
  } else {
    RunQuantizationCopyTest<::ml_drift::DataType::INT4>();
  }
}

TEST_P(PopulateTensorQuantizationTest, ZeroCopy) {
  if (GetParam() == ::ml_drift::DataType::INT8) {
    RunQuantizationZeroCopyTest<::ml_drift::DataType::INT8>();
  } else {
    RunQuantizationZeroCopyTest<::ml_drift::DataType::INT4>();
  }
}

}  // namespace
}  // namespace litert::ml_drift::ir
