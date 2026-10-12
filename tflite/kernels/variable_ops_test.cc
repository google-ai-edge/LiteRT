/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <stdint.h>

#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/interpreter.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/core/kernels/register.h"
#include "tflite/core/model_builder.h"
#include "tflite/core/subgraph.h"
#include "tflite/experimental/resource/mock_resource.h"
#include "tflite/experimental/resource/resource_base.h"
#include "tflite/experimental/resource/resource_variable.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/version.h"

namespace tflite {
namespace {
const char kContainer[] = "c";
const char kSharedName[] = "a";

class VariableOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    assign_registration_ = ::tflite::ops::builtin::Register_ASSIGN_VARIABLE();
    ASSERT_NE(assign_registration_, nullptr);
    read_registration_ = ::tflite::ops::builtin::Register_READ_VARIABLE();
    ASSERT_NE(read_registration_, nullptr);
    var_handle_registration_ = ::tflite::ops::builtin::Register_VAR_HANDLE();
    ASSERT_NE(var_handle_registration_, nullptr);

    ConstructGraph();
  }

  void ConstructInvalidGraph() {
    interpreter_ = std::make_unique<Interpreter>();
    // Invalid graph, variable is read before it is assigned a value.

    // Construct a graph like this:
    //   Input: %0
    //   Output: %2
    //   %1 = var_handle()
    //   %2 = read(%1)

    TfLiteVarHandleParams* var_handle_params = GetVarHandleParams();

    int first_new_tensor_index;
    ASSERT_EQ(interpreter_->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs({2}), kTfLiteOk);
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteResource, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    int node_index;
    interpreter_->AddNodeWithParameters({}, {1}, nullptr, 0, var_handle_params,
                                        var_handle_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                        read_registration_, &node_index);
  }

  TfLiteVarHandleParams* GetVarHandleParams() {
    TfLiteVarHandleParams* var_handle_params =
        reinterpret_cast<TfLiteVarHandleParams*>(
            malloc(sizeof(TfLiteVarHandleParams)));
    var_handle_params->container = kContainer;
    var_handle_params->shared_name = kSharedName;
    return var_handle_params;
  }

  void ConstructGraph() {
    interpreter_ = std::make_unique<Interpreter>();
    // Construct a graph like this:
    //   Input: %0
    //   Output: %2
    //   %1 = var_handle()
    //   variable_assign(%1, %0)
    //   %2 = read(%1)

    int first_new_tensor_index;
    ASSERT_EQ(interpreter_->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs({2}), kTfLiteOk);
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteResource, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    int node_index;

    TfLiteVarHandleParams* var_handle_params = GetVarHandleParams();
    interpreter_->AddNodeWithParameters({}, {1}, nullptr, 0, var_handle_params,
                                        var_handle_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1, 0}, {}, nullptr, 0, nullptr,
                                        assign_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                        read_registration_, &node_index);
  }

  // Similar with `ConstructGraph`, but with static tensor shapes.
  void ConstructGraphWithKnownShape() {
    interpreter_ = std::make_unique<Interpreter>();
    // Construct a graph like this:
    //   Input: %0
    //   Output: %2
    //   %1 = var_handle()
    //   variable_assign(%1, %0)
    //   %2 = read(%1)

    int first_new_tensor_index;
    ASSERT_EQ(interpreter_->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs({2}), kTfLiteOk);
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {2, 2},
                                               TfLiteQuantization());
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteResource, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {2, 2},
                                               TfLiteQuantization());
    int node_index;

    TfLiteVarHandleParams* var_handle_params = GetVarHandleParams();
    interpreter_->AddNodeWithParameters({}, {1}, nullptr, 0, var_handle_params,
                                        var_handle_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1, 0}, {}, nullptr, 0, nullptr,
                                        assign_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                        read_registration_, &node_index);
  }

  void ConstructSingleVariableOp(bool read) {
    interpreter_ = std::make_unique<Interpreter>();
    int first_new_tensor_index;
    ASSERT_EQ(interpreter_->AddTensors(2, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetInputs(read ? std::vector<int>{0}
                                           : std::vector<int>{0, 1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs(read ? std::vector<int>{1}
                                            : std::vector<int>{}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->SetTensorParametersReadWrite(
                  0, kTfLiteResource, "", {1}, TfLiteQuantization()),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->SetTensorParametersReadWrite(
                  1, kTfLiteFloat32, "", {1}, TfLiteQuantization()),
              kTfLiteOk);
    int node_index;
    ASSERT_EQ(interpreter_->AddNodeWithParameters(
                  read ? std::vector<int>{0} : std::vector<int>{0, 1},
                  read ? std::vector<int>{1} : std::vector<int>{}, nullptr, 0,
                  nullptr, read ? read_registration_ : assign_registration_,
                  &node_index),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    auto* resource_id = interpreter_->tensor(0);
    resource_id->data.i32 = &resource_id_storage_;
    resource_id->bytes = sizeof(resource_id_storage_);
    resource_id->allocation_type = kTfLiteCustom;
  }

  TfLiteRegistration* assign_registration_;
  TfLiteRegistration* read_registration_;
  TfLiteRegistration* var_handle_registration_;
  int32_t resource_id_storage_ = 0;
  std::unique_ptr<Interpreter> interpreter_;
};

// The external RESOURCE buffer follows the FlatBuffer. The offset and size
// arguments let each test exercise one branch of ParseTensors' range check.
std::vector<uint8_t> MakeVariableModel(BuiltinOperator op, uint64_t offset,
                                       uint64_t size, bool append_resource_id) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<int32_t> one = {1};
  const std::vector<uint8_t> value = {0, 160, 214, 68};  // 1717.0f.
  const bool read = op == BuiltinOperator_READ_VARIABLE;
  const auto id =
      CreateTensorDirect(builder, &one, TensorType_RESOURCE, 1, "resource_id");
  const auto data = CreateTensorDirect(
      builder, &one, TensorType_FLOAT32, read ? 0 : 2,
      read ? "output" : "value");
  const std::vector<flatbuffers::Offset<Tensor>> tensors = {id, data};
  const std::vector<int32_t> inputs = read ? std::vector<int32_t>{0}
                                           : std::vector<int32_t>{0, 1};
  const std::vector<int32_t> outputs = read ? std::vector<int32_t>{1}
                                            : std::vector<int32_t>{};
  const auto operation =
      CreateOperatorDirect(builder, 0, &inputs, &outputs);
  const std::vector<flatbuffers::Offset<Operator>> operations = {operation};
  const auto subgraph = CreateSubGraphDirect(builder, &tensors, nullptr,
                                            &outputs, &operations, "main");
  const std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {subgraph};
  const std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBufferDirect(builder), CreateBufferDirect(builder, nullptr, offset,
                                                      size),
      CreateBufferDirect(builder, &value)};
  const std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
      CreateOperatorCodeDirect(builder, 127, nullptr, 1, op)};
  const auto model = CreateModelDirect(builder, TFLITE_SCHEMA_VERSION, &opcodes,
                                       &subgraphs, "variable bounds", &buffers);
  FinishModelBuffer(builder, model);
  std::vector<uint8_t> bytes(builder.GetBufferPointer(),
                             builder.GetBufferPointer() + builder.GetSize());
  if (append_resource_id) {
    bytes.insert(bytes.end(), {0, 0, 0, 0});
  }
  return bytes;
}

std::vector<uint8_t> MakeTailResourceModel(BuiltinOperator op, uint64_t size,
                                           bool append_resource_id) {
  // Both nonzero offsets occupy the same scalar field width in FlatBuffers.
  const size_t model_size = MakeVariableModel(op, 2, size, false).size();
  auto bytes = MakeVariableModel(op, model_size, size, append_resource_id);
  return bytes;
}

TfLiteStatus NoOpCustomPrepare(TfLiteContext*, TfLiteNode*) {
  return kTfLiteOk;
}

TfLiteStatus NoOpCustomInvoke(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }

// The large custom options payload follows the FlatBuffer when in bounds.
std::vector<uint8_t> MakeCustomOptionsModel(uint64_t offset, uint64_t size,
                                            bool append_options) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<int32_t> empty;
  const std::vector<flatbuffers::Offset<Tensor>> tensors;
  const auto operation = CreateOperatorDirect(
      builder, 0, &empty, &empty, BuiltinOptions_NONE, 0, nullptr,
      CustomOptionsFormat_FLEXBUFFERS, nullptr, nullptr, offset, size);
  const std::vector<flatbuffers::Offset<Operator>> operations = {operation};
  const auto subgraph = CreateSubGraphDirect(builder, &tensors, nullptr, &empty,
                                             &operations, "custom bounds");
  const std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {subgraph};
  const std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBufferDirect(builder)};
  const std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
      CreateOperatorCodeDirect(builder,
                               static_cast<int8_t>(BuiltinOperator_CUSTOM),
                               "BoundsStub", 1, BuiltinOperator_CUSTOM)};
  const auto model = CreateModelDirect(builder, TFLITE_SCHEMA_VERSION, &opcodes,
                                       &subgraphs, "custom bounds", &buffers);
  FinishModelBuffer(builder, model);
  std::vector<uint8_t> bytes(builder.GetBufferPointer(),
                             builder.GetBufferPointer() + builder.GetSize());
  if (append_options) {
    bytes.insert(bytes.end(), size, 0);
  }
  return bytes;
}

TfLiteStatus BuildCustomOptionsModel(const std::vector<uint8_t>& bytes) {
  auto model = FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (model == nullptr) return kTfLiteError;
  TfLiteRegistration registration = {nullptr, nullptr, NoOpCustomPrepare,
                                     NoOpCustomInvoke};
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  resolver.AddCustom("BoundsStub", &registration);
  std::unique_ptr<Interpreter> interpreter;
  return InterpreterBuilder(*model, resolver)(&interpreter);
}

TEST_F(VariableOpsTest, TestAssignThenReadVariable) {
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_data_index = interpreter_->tensor(0);
  GetTensorData<float>(input_data_index)[0] = 1717;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  // Verify output.
  TfLiteTensor* output = interpreter_->tensor(2);
  ASSERT_EQ(output->dims->size, 0);
  EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
}

TEST_F(VariableOpsTest, TestAssignThenReadVariableWithKnownShape) {
  ConstructGraphWithKnownShape();
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_data_index = interpreter_->tensor(0);
  GetTensorData<float>(input_data_index)[0] = 1.0;
  GetTensorData<float>(input_data_index)[1] = 2.0;
  GetTensorData<float>(input_data_index)[2] = 3.0;
  GetTensorData<float>(input_data_index)[3] = 4.0;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  // Verify output.
  TfLiteTensor* output = interpreter_->tensor(2);
  ASSERT_EQ(output->dims->size, 2);
  EXPECT_EQ(GetTensorData<float>(output)[0], 1.0);
  EXPECT_EQ(GetTensorData<float>(output)[1], 2.0);
  EXPECT_EQ(GetTensorData<float>(output)[2], 3.0);
  EXPECT_EQ(GetTensorData<float>(output)[3], 4.0);
}

TEST_F(VariableOpsTest, TestReadVariableBeforeAssign) {
  ConstructInvalidGraph();
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_data_index = interpreter_->tensor(0);
  GetTensorData<float>(input_data_index)[0] = 1717;

  // Error because variable 2 is never initialized.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteError);
}

TEST_F(VariableOpsTest, TestReassignToDifferentSize) {
  // 1st invocation. The variable is assigned as a scalar.
  {
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    TfLiteTensor* input_data_index = interpreter_->tensor(0);
    GetTensorData<float>(input_data_index)[0] = 1717;
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

    // Verify output.
    TfLiteTensor* output = interpreter_->tensor(2);
    ASSERT_EQ(output->dims->size, 0);
    EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
  }

  // 2nd invocation. The variable is assigned as a 1D vector with 2 elements.
  {
    interpreter_->ResizeInputTensor(0, {2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    TfLiteTensor* input_data_index = interpreter_->tensor(0);
    GetTensorData<float>(input_data_index)[0] = 1717;
    GetTensorData<float>(input_data_index)[1] = 2121;
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

    // Verify output.
    TfLiteTensor* output = interpreter_->tensor(2);
    ASSERT_EQ(output->dims->size, 1);
    ASSERT_EQ(output->dims->data[0], 2);
    EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
    EXPECT_EQ(GetTensorData<float>(output)[1], 2121);
  }
}

TEST_F(VariableOpsTest, TestAssignVariableTypeCollision) {
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_data_index = interpreter_->tensor(0);
  GetTensorData<float>(input_data_index)[0] = 1717;

  // Insert a conflicting resource type into the subgraph resources map at id 0.
  interpreter_->primary_subgraph().resources().emplace(
      0, std::make_unique<resource::MockHashTableResource>());

  EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
}

TEST_F(VariableOpsTest, TestReadRejectsMissingResourceIdData) {
  interpreter_ = std::make_unique<Interpreter>();
  int first_new_tensor_index;
  ASSERT_EQ(interpreter_->AddTensors(2, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(interpreter_->SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter_->SetOutputs({1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->SetTensorParametersReadWrite(
                0, kTfLiteResource, "", {1}, TfLiteQuantization()),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->SetTensorParametersReadWrite(
                1, kTfLiteFloat32, "", {}, TfLiteQuantization()),
            kTfLiteOk);
  int node_index;
  ASSERT_EQ(interpreter_->AddNodeWithParameters(
                {0}, {1}, nullptr, 0, nullptr, read_registration_, &node_index),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  TfLiteTensor* resource_id = interpreter_->tensor(0);
  int32_t* saved_data = resource_id->data.i32;
  resource_id->data.i32 = nullptr;
  EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  resource_id->data.i32 = saved_data;
}

TEST_F(VariableOpsTest, TestAssignRejectsMissingResourceIdData) {
  interpreter_ = std::make_unique<Interpreter>();
  int first_new_tensor_index;
  ASSERT_EQ(interpreter_->AddTensors(2, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(interpreter_->SetInputs({0, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->SetOutputs({}), kTfLiteOk);
  ASSERT_EQ(interpreter_->SetTensorParametersReadWrite(
                0, kTfLiteResource, "", {1}, TfLiteQuantization()),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->SetTensorParametersReadWrite(
                1, kTfLiteFloat32, "", {}, TfLiteQuantization()),
            kTfLiteOk);
  int node_index;
  ASSERT_EQ(interpreter_->AddNodeWithParameters(
                {0, 1}, {}, nullptr, 0, nullptr, assign_registration_,
                &node_index),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  TfLiteTensor* resource_id = interpreter_->tensor(0);
  int32_t* saved_data = resource_id->data.i32;
  resource_id->data.i32 = nullptr;
  EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  resource_id->data.i32 = saved_data;
}

TEST_F(VariableOpsTest, TestReadRejectsEmptyResourceIdAfterAllocate) {
  ASSERT_NO_FATAL_FAILURE(ConstructSingleVariableOp(true));
  auto* resource_id = interpreter_->tensor(0);
  ASSERT_NE(resource_id->data.i32, nullptr);
  resource_id->data.i32[0] = 0;
  auto& resources = interpreter_->primary_subgraph().resources();
  ASSERT_EQ(resource::CreateResourceVariableIfNotAvailable(&resources, 0),
            kTfLiteOk);
  auto* variable = resource::GetResourceVariable(&resources, 0);
  ASSERT_NE(variable, nullptr);
  GetTensorData<float>(interpreter_->tensor(1))[0] = 1717.0f;
  ASSERT_EQ(variable->AssignFrom(interpreter_->tensor(1)), kTfLiteOk);

  resource_id->dims->data[0] = 0;
  EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  resource_id->dims->data[0] = 1;
}

TEST_F(VariableOpsTest, TestAssignRejectsEmptyResourceIdAfterAllocate) {
  ASSERT_NO_FATAL_FAILURE(ConstructSingleVariableOp(false));
  auto* resource_id = interpreter_->tensor(0);
  ASSERT_NE(resource_id->data.i32, nullptr);
  resource_id->data.i32[0] = 0;

  resource_id->dims->data[0] = 0;
  EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  resource_id->dims->data[0] = 1;
}

TEST_F(VariableOpsTest, TestReadRejectsShortResourceIdBacking) {
  ASSERT_NO_FATAL_FAILURE(ConstructSingleVariableOp(true));
  auto* resource_id = interpreter_->tensor(0);
  ASSERT_NE(resource_id->data.i32, nullptr);
  resource_id->data.i32[0] = 0;
  auto& resources = interpreter_->primary_subgraph().resources();
  ASSERT_EQ(resource::CreateResourceVariableIfNotAvailable(&resources, 0),
            kTfLiteOk);
  auto* variable = resource::GetResourceVariable(&resources, 0);
  ASSERT_NE(variable, nullptr);
  GetTensorData<float>(interpreter_->tensor(1))[0] = 1717.0f;
  ASSERT_EQ(variable->AssignFrom(interpreter_->tensor(1)), kTfLiteOk);

  const size_t original_bytes = resource_id->bytes;
  for (size_t short_bytes = 1; short_bytes < sizeof(int32_t); ++short_bytes) {
    SCOPED_TRACE(short_bytes);
    resource_id->bytes = short_bytes;
    EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  }
  resource_id->bytes = original_bytes;
}

TEST_F(VariableOpsTest, TestAssignRejectsShortResourceIdBacking) {
  ASSERT_NO_FATAL_FAILURE(ConstructSingleVariableOp(false));
  auto* resource_id = interpreter_->tensor(0);
  ASSERT_NE(resource_id->data.i32, nullptr);
  resource_id->data.i32[0] = 0;

  const size_t original_bytes = resource_id->bytes;
  for (size_t short_bytes = 1; short_bytes < sizeof(int32_t); ++short_bytes) {
    SCOPED_TRACE(short_bytes);
    resource_id->bytes = short_bytes;
    EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  }
  resource_id->bytes = original_bytes;
}

TEST_F(VariableOpsTest, TestAssignAcceptsFourByteResourceIdBacking) {
  ASSERT_NO_FATAL_FAILURE(ConstructSingleVariableOp(false));
  auto* resource_id = interpreter_->tensor(0);
  ASSERT_EQ(resource_id->bytes, sizeof(int32_t));
  GetTensorData<float>(interpreter_->tensor(1))[0] = 1717.0f;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  auto& resources = interpreter_->primary_subgraph().resources();
  auto* variable = resource::GetResourceVariable(&resources, 0);
  ASSERT_NE(variable, nullptr);
  ASSERT_NE(variable->GetTensor(), nullptr);
  EXPECT_EQ(GetTensorData<float>(variable->GetTensor())[0], 1717.0f);
}

TEST_F(VariableOpsTest, TestReadRejectsOutputLargerThanAssignedValue) {
  ConstructGraphWithKnownShape();
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  auto* input = interpreter_->tensor(0);
  GetTensorData<float>(input)[0] = 1717.0f;
  const size_t original_bytes = input->bytes;
  ASSERT_EQ(original_bytes, 4 * sizeof(float));
  input->bytes = sizeof(float);
  EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  input->bytes = original_bytes;
}

TEST(VariableModelBuilderBoundsTest, RejectsWrappedCustomOptionsOffset) {
  const auto bytes = MakeCustomOptionsModel(
      std::numeric_limits<uint64_t>::max() - 3, 8, false);
  EXPECT_EQ(BuildCustomOptionsModel(bytes), kTfLiteError);
}

TEST(VariableModelBuilderBoundsTest, RejectsCustomOptionsSizePastEnd) {
  const auto bytes =
      MakeCustomOptionsModel(4, std::numeric_limits<uint64_t>::max(), false);
  EXPECT_EQ(BuildCustomOptionsModel(bytes), kTfLiteError);
}

TEST(VariableModelBuilderBoundsTest, AcceptsInBoundsCustomOptions) {
  const size_t model_size = MakeCustomOptionsModel(2, 8, false).size();
  const auto bytes = MakeCustomOptionsModel(model_size, 8, true);
  EXPECT_EQ(BuildCustomOptionsModel(bytes), kTfLiteOk);
}

TEST(VariableModelBuilderBoundsTest, RejectsOffsetPastAllocation) {
  const auto bytes = MakeVariableModel(
      BuiltinOperator_ASSIGN_VARIABLE, std::numeric_limits<uint64_t>::max() - 3,
      8, false);
  auto model = FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ASSERT_NE(model, nullptr);
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  std::unique_ptr<Interpreter> interpreter;
  EXPECT_EQ(InterpreterBuilder(*model, resolver)(&interpreter), kTfLiteError);
}

TEST(VariableModelBuilderBoundsTest, RejectsSizePastRemainingAllocation) {
  const auto bytes = MakeVariableModel(
      BuiltinOperator_ASSIGN_VARIABLE, 4,
      std::numeric_limits<uint64_t>::max(), false);
  auto model = FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ASSERT_NE(model, nullptr);
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  std::unique_ptr<Interpreter> interpreter;
  EXPECT_EQ(InterpreterBuilder(*model, resolver)(&interpreter), kTfLiteError);
}

TEST(VariableModelBuilderBoundsTest, RejectsZeroByteReadResourceBacking) {
  const auto bytes =
      MakeTailResourceModel(BuiltinOperator_READ_VARIABLE, 0, false);
  auto model = FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ASSERT_NE(model, nullptr);
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model, resolver)(&interpreter), kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(interpreter->Invoke(), kTfLiteError);
}

TEST(VariableModelBuilderBoundsTest, RejectsZeroByteAssignResourceBacking) {
  const auto bytes =
      MakeTailResourceModel(BuiltinOperator_ASSIGN_VARIABLE, 0, false);
  auto model = FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ASSERT_NE(model, nullptr);
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model, resolver)(&interpreter), kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(interpreter->Invoke(), kTfLiteError);
}

TEST(VariableModelBuilderBoundsTest, AcceptsInBoundsResourceBacking) {
  const auto bytes = MakeTailResourceModel(BuiltinOperator_ASSIGN_VARIABLE,
                                           sizeof(int32_t), true);
  auto model = FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ASSERT_NE(model, nullptr);
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model, resolver)(&interpreter), kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  auto& resources = interpreter->primary_subgraph().resources();
  auto* variable = resource::GetResourceVariable(&resources, 0);
  ASSERT_NE(variable, nullptr);
  ASSERT_NE(variable->GetTensor(), nullptr);
  EXPECT_EQ(GetTensorData<float>(variable->GetTensor())[0], 1717.0f);
}

}  // namespace
}  // namespace tflite
