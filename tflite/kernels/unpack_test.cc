/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <initializer_list>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tflite/c/c_api_types.h"
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/types/half.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class UnpackOpModel : public SingleOpModel {
 public:
  UnpackOpModel(const TensorData& input, int axis) {
    if (axis < 0) {
      axis += input.shape.size();
    }
    const int num_outputs = input.shape[axis];
    input_ = AddInput(input);
    for (int i = 0; i < num_outputs; ++i) {
      outputs_.push_back(AddOutput(input.type));
    }
    SetBuiltinOp(BuiltinOperator_UNPACK, BuiltinOptions_UnpackOptions,
                 CreateUnpackOptions(builder_, num_outputs, axis).Union());
    BuildInterpreter({GetShape(input_)});
  }

  // In this test, we represent all input and output test data as integers, and
  // convert it to the intended type here.
  void SetInput(std::vector<int> data) {
    std::vector<T> typed_data(data.begin(), data.end());
    PopulateTensor(input_, typed_data);
  }

  std::vector<std::vector<int>> GetOutputDatas() {
    std::vector<std::vector<int>> output_datas;
    for (const int output : outputs_) {
      std::vector<T> typed_output_data = ExtractVector<T>(output);
      output_datas.emplace_back(typed_output_data.begin(),
                                typed_output_data.end());
    }
    return output_datas;
  }

  std::vector<std::vector<int>> GetOutputShapes() {
    std::vector<std::vector<int>> output_shapes;
    for (const int output : outputs_) {
      output_shapes.push_back(GetTensorShape(output));
    }
    return output_shapes;
  }

 private:
  int input_;
  std::vector<int> outputs_;
};

class PrepareOnlyUnpackOpModel : public SingleOpModel {
 public:
  explicit PrepareOnlyUnpackOpModel(const TensorData& input, int num_outputs,
                                    int axis) {
    input_ = AddInput(input);
    for (int i = 0; i < num_outputs; ++i) {
      AddOutput(input.type);
    }
    SetBuiltinOp(BuiltinOperator_UNPACK, BuiltinOptions_UnpackOptions,
                 CreateUnpackOptions(builder_, num_outputs, axis).Union());
    BuildInterpreter({GetShape(input_)}, /*num_threads=*/1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false,
                     /*allocate_and_delegate=*/false);
  }

 private:
  int input_;
};

TEST(UnpackPrepareTest, RejectsAxisOutsideInt16Range) {
  PrepareOnlyUnpackOpModel too_large(
      /*input=*/{TensorType_FLOAT32, {1}}, /*num_outputs=*/1,
      std::numeric_limits<int16_t>::max() + 1);
  EXPECT_EQ(too_large.AllocateTensors(), kTfLiteError);

  PrepareOnlyUnpackOpModel too_small(
      /*input=*/{TensorType_FLOAT32, {1}}, /*num_outputs=*/1,
      std::numeric_limits<int16_t>::min() - 1);
  EXPECT_EQ(too_small.AllocateTensors(), kTfLiteError);
}

}  // namespace
namespace ops {
namespace builtin {
TfLiteRegistration* Register_UNPACK();
}  // namespace builtin
}  // namespace ops
namespace {

class CustomOutputsUnpackOpModel : public SingleOpModel {
 public:
  CustomOutputsUnpackOpModel(const TensorData& input,
                             const std::vector<TensorData>& outputs, int axis) {
    input_ = AddInput(input);
    for (const TensorData& output : outputs) {
      outputs_.push_back(AddOutput(output));
    }
    SetBuiltinOp(
        BuiltinOperator_UNPACK, BuiltinOptions_UnpackOptions,
        CreateUnpackOptions(builder_, static_cast<int>(outputs.size()), axis)
            .Union());
    BuildInterpreter({GetShape(input_)}, /*num_threads=*/1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false,
                     /*allocate_and_delegate=*/false);
  }

  std::vector<int> GetOutputShapeAt(int index) {
    return GetTensorShape(outputs_[index]);
  }

 private:
  int input_;
  std::vector<int> outputs_;
};

TfLiteStatus FailingResizeTensor(TfLiteContext*, TfLiteTensor*,
                                 TfLiteIntArray* new_size) {
  TfLiteIntArrayFree(new_size);
  return kTfLiteError;
}

void SilentReportError(TfLiteContext*, const char*, ...) {}

TEST(UnpackPrepareTest, ValidQuantizedOutputsSucceed) {
  CustomOutputsUnpackOpModel valid_quantized(
      /*input=*/{TensorType_INT8, {2, 3}, 0, 0, 0.5f, 4},
      /*outputs=*/
      {{TensorType_INT8, {3}, 0, 0, 0.5f, 4},
       {TensorType_INT8, {3}, 0, 0, 0.5f, 4}},
      /*axis=*/0);
  ASSERT_EQ(valid_quantized.AllocateTensors(), kTfLiteOk);
  EXPECT_THAT(valid_quantized.GetOutputShapeAt(0), ElementsAreArray({3}));
  EXPECT_THAT(valid_quantized.GetOutputShapeAt(1), ElementsAreArray({3}));
}

TEST(UnpackPrepareTest,
     RejectsMismatchedAxisDimensionAndOutputMetadataWithoutPartialResize) {
  PrepareOnlyUnpackOpModel mismatched_num(
      /*input=*/{TensorType_FLOAT32, {3, 2}}, /*num_outputs=*/2, /*axis=*/0);
  EXPECT_EQ(mismatched_num.AllocateTensors(), kTfLiteError);

  CustomOutputsUnpackOpModel mismatched_second_output_zero_point(
      /*input=*/{TensorType_INT8, {2, 3}, 0, 0, 0.5f, 4},
      /*outputs=*/
      {{TensorType_INT8, {9}, 0, 0, 0.5f, 4},
       {TensorType_INT8, {9}, 0, 0, 0.5f, 7}},
      /*axis=*/0);
  EXPECT_EQ(mismatched_second_output_zero_point.AllocateTensors(),
            kTfLiteError);
  EXPECT_THAT(mismatched_second_output_zero_point.GetOutputShapeAt(0),
              ElementsAreArray({9}));

  // Exercise Prepare() directly when ResizeTensor fails to ensure output_shape
  // cleanup and error propagation.
  TfLiteIntArray* in_dims = TfLiteIntArrayCreate(2);
  in_dims->data[0] = 2;
  in_dims->data[1] = 3;
  TfLiteIntArray* out0_dims = TfLiteIntArrayCreate(1);
  out0_dims->data[0] = 3;
  TfLiteIntArray* out1_dims = TfLiteIntArrayCreate(1);
  out1_dims->data[0] = 3;
  TfLiteTensor tensors[3] = {};
  tensors[0].type = kTfLiteInt32;
  tensors[0].dims = in_dims;
  tensors[1].type = kTfLiteInt32;
  tensors[1].dims = out0_dims;
  tensors[2].type = kTfLiteInt32;
  tensors[2].dims = out1_dims;
  TfLiteIntArray* node_inputs = TfLiteIntArrayCreate(1);
  node_inputs->data[0] = 0;
  TfLiteIntArray* node_outputs = TfLiteIntArrayCreate(2);
  node_outputs->data[0] = 1;
  node_outputs->data[1] = 2;
  TfLiteUnpackParams params = {2, 0};
  TfLiteNode node = {};
  node.inputs = node_inputs;
  node.outputs = node_outputs;
  node.builtin_data = &params;
  TfLiteContext ctx = {};
  ctx.tensors = tensors;
  ctx.tensors_size = 3;
  ctx.ResizeTensor = FailingResizeTensor;
  ctx.ReportError = SilentReportError;
  TfLiteRegistration* reg = ops::builtin::Register_UNPACK();
  EXPECT_EQ(reg->prepare(&ctx, &node), kTfLiteError);
  EXPECT_EQ(tensors[1].dims, out0_dims);
  EXPECT_EQ(tensors[1].dims->size, 1);
  EXPECT_EQ(tensors[1].dims->data[0], 3);
  TfLiteIntArrayFree(in_dims);
  TfLiteIntArrayFree(out0_dims);
  TfLiteIntArrayFree(out1_dims);
  TfLiteIntArrayFree(node_inputs);
  TfLiteIntArrayFree(node_outputs);
}

TEST(UnpackPrepareTest, RejectsInconsistentInputOrOutputShapeInEval) {
  UnpackOpModel<int32_t> mismatched_output_elements(
      {TensorType_INT32, {2, 3}}, /*axis=*/0);
  mismatched_output_elements.SetInput({1, 2, 3, 4, 5, 6});
  mismatched_output_elements.GetOutputTensor(1)->dims->data[0] = 1;
  const TfLiteStatus mismatched_status = mismatched_output_elements.Invoke();
  EXPECT_EQ(mismatched_status, kTfLiteError);

  UnpackOpModel<int32_t> mismatched_axis_dim({TensorType_INT32, {2, 3}},
                                             /*axis=*/0);
  mismatched_axis_dim.SetInput({1, 2, 3, 4, 5, 6});
  mismatched_axis_dim.GetInputTensor(0)->dims->data[0] = 1;
  EXPECT_EQ(mismatched_axis_dim.Invoke(), kTfLiteError);

  UnpackOpModel<int32_t> negative_output_dimension(
      {TensorType_INT32, {2, 3}}, /*axis=*/0);
  negative_output_dimension.SetInput({1, 2, 3, 4, 5, 6});
  negative_output_dimension.GetOutputTensor(0)->dims->data[0] = -1;
  EXPECT_EQ(negative_output_dimension.Invoke(), kTfLiteError);

  // Input element-count overflow in CheckedNumElements(input): {1073741824, 2}.
  UnpackOpModel<int32_t> overflowing_input_elements(
      {TensorType_INT32, {2, 2}}, /*axis=*/0);
  overflowing_input_elements.SetInput({1, 2, 3, 4});
  TfLiteTensor* ovf_in = overflowing_input_elements.GetInputTensor(0);
  ovf_in->dims->data[0] = 1073741824;
  ovf_in->dims->data[1] = 2;
  EXPECT_EQ(overflowing_input_elements.Invoke(), kTfLiteError);

  // Output element-count overflow in CheckedNumElements(output): {46341, 46341}.
  UnpackOpModel<int32_t> overflowing_output_elements(
      {TensorType_INT32, {2, 2, 1}}, /*axis=*/0);
  overflowing_output_elements.SetInput({1, 2, 3, 4});
  TfLiteTensor* ovf_out = overflowing_output_elements.GetOutputTensor(0);
  ovf_out->dims->data[0] = 46341;
  ovf_out->dims->data[1] = 46341;
  EXPECT_EQ(overflowing_output_elements.Invoke(), kTfLiteError);

  UnpackOpModel<int32_t> insufficient_input_bytes(
      {TensorType_INT32, {2, 3}}, /*axis=*/0);
  insufficient_input_bytes.SetInput({1, 2, 3, 4, 5, 6});
  TfLiteTensor* short_in = insufficient_input_bytes.GetInputTensor(0);
  const size_t orig_in_bytes = short_in->bytes;
  short_in->bytes = 1;
  EXPECT_EQ(insufficient_input_bytes.Invoke(), kTfLiteError);
  short_in->bytes = orig_in_bytes;

  UnpackOpModel<int32_t> insufficient_output_bytes(
      {TensorType_INT32, {2, 3}}, /*axis=*/0);
  insufficient_output_bytes.SetInput({1, 2, 3, 4, 5, 6});
  TfLiteTensor* short_out = insufficient_output_bytes.GetOutputTensor(0);
  const size_t orig_out_bytes = short_out->bytes;
  short_out->bytes = 1;
  EXPECT_EQ(insufficient_output_bytes.Invoke(), kTfLiteError);
  short_out->bytes = orig_out_bytes;

  if (mismatched_status == kTfLiteError) {
    UnpackOpModel<int32_t> null_input_data({TensorType_INT32, {2, 3}},
                                           /*axis=*/0);
    null_input_data.SetInput({1, 2, 3, 4, 5, 6});
    TfLiteTensor* in_tensor = null_input_data.GetInputTensor(0);
    char* original_input_data = in_tensor->data.raw;
    in_tensor->data.raw = nullptr;
    EXPECT_EQ(null_input_data.Invoke(), kTfLiteError);
    in_tensor->data.raw = original_input_data;

    UnpackOpModel<int32_t> null_output_data({TensorType_INT32, {2, 3}},
                                            /*axis=*/0);
    null_output_data.SetInput({1, 2, 3, 4, 5, 6});
    TfLiteTensor* output = null_output_data.GetOutputTensor(0);
    char* original_output_data = output->data.raw;
    output->data.raw = nullptr;
    EXPECT_EQ(null_output_data.Invoke(), kTfLiteError);
    output->data.raw = original_output_data;
  }
}

template <typename T>
void Check(int axis, const std::initializer_list<int>& input_shape,
           const std::initializer_list<int>& input_data,
           const std::vector<std::vector<int>>& exp_output_shape,
           const std::vector<std::vector<int>>& exp_output_data,
           const TensorType& type = TensorType_FLOAT32) {
  UnpackOpModel<T> m({type, input_shape}, axis);
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Check outputs shapes.
  EXPECT_THAT(m.GetOutputShapes(), ElementsAreArray(exp_output_shape));

  // Check outputs values.
  EXPECT_THAT(m.GetOutputDatas(), ElementsAreArray(exp_output_data));
}

template <typename InputType>
struct UnpackOpTest : public ::testing::Test {
  using TypeToTest = InputType;
  TensorType TENSOR_TYPE = GetTensorType<InputType>();
};

using TestTypes = testing::Types<float, half, Eigen::bfloat16, int32_t, int8_t,
                                 uint8_t, int16_t>;
TYPED_TEST_CASE(UnpackOpTest, TestTypes);

TYPED_TEST(UnpackOpTest, ThreeOutputs) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/0, /*input_shape=*/{3, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{2}, {2}, {2}},
      /*exp_output_data=*/{{1, 2}, {3, 4}, {5, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, ThreeOutputsAxisOne) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/1, /*input_shape=*/{3, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{1, 3, 5}, {2, 4, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, ThreeOutputsNegativeAxisOne) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/-1, /*input_shape=*/{3, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{1, 3, 5}, {2, 4, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, OneOutput) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/0, /*input_shape=*/{1, 6},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{6}},
      /*exp_output_data=*/{{1, 2, 3, 4, 5, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, ThreeDimensionsOutputs) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8},
      /*exp_output_shape=*/{{2, 2}, {2, 2}},
      /*exp_output_data=*/{{1, 3, 5, 7}, {2, 4, 6, 8}},
      TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, FiveDimensionsOutputs) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2, 2, 1},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      /*exp_output_shape=*/{{2, 2, 2, 1}, {2, 2, 2, 1}},
      /*exp_output_data=*/
      {{1, 2, 5, 6, 9, 10, 13, 14}, {3, 4, 7, 8, 11, 12, 15, 16}},
      /*type=*/TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, VectorToScalar) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/0, /*input_shape=*/{5},
      /*input_data=*/{1, 2, 3, 4, 5},
      /*exp_output_shape=*/{{}, {}, {}, {}, {}},
      /*exp_output_data=*/{{1}, {2}, {3}, {4}, {5}}, TestFixture::TENSOR_TYPE);
}

#if defined(TFLITE_ENABLE_EXTRA_REFERENCE_KERNELS)
TEST(UnpackOpTest, Float8) {
  for (TensorType tensor_type :
       {TensorType_FLOAT8_E4M3FN, TensorType_FLOAT8_E5M2}) {
    Check<uint8_t>(/*axis=*/0, /*input_shape=*/{2, 2},
                   /*input_data=*/{0x00, 0x38, 0xbc, 0x7e},
                   /*exp_output_shape=*/{{2}, {2}},
                   /*exp_output_data=*/{{0x00, 0x38}, {0xbc, 0x7e}},
                   tensor_type);
  }
}
#endif

// bool tests.
TEST(UnpackOpTestBool, BoolThreeOutputs) {
  Check<bool>(
      /*axis=*/0, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{2}, {2}, {2}},
      /*exp_output_data=*/{{true, false}, {true, false}, {true, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeOutputsAxisOne) {
  Check<bool>(
      /*axis=*/1, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{true, true, true}, {false, false, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeOutputsNegativeAxisOne) {
  Check<bool>(
      /*axis=*/-1, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{true, true, true}, {false, false, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeOutputsNegativeAxisTwo) {
  Check<bool>(
      /*axis=*/-2, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{2}, {2}, {2}},
      /*exp_output_data=*/{{true, false}, {true, false}, {true, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolOneOutput) {
  Check<bool>(
      /*axis=*/0, /*input_shape=*/{1, 6},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{6}},
      /*exp_output_data=*/{{true, false, true, false, true, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeDimensionsOutputs) {
  Check<bool>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2},
      /*input_data=*/{true, false, true, false, true, false, true, false},
      /*exp_output_shape=*/{{2, 2}, {2, 2}},
      /*exp_output_data=*/
      {{true, true, true, true}, {false, false, false, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTest, BoolFiveDimensionsOutputs) {
  Check<bool>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2, 2, 1},
      /*input_data=*/
      {true, false, true, false, true, false, true, false, true, true, true,
       true, true, true, true, true},
      /*exp_output_shape=*/{{2, 2, 2, 1}, {2, 2, 2, 1}},
      /*exp_output_data=*/
      {{true, false, true, false, true, true, true, true},
       {true, false, true, false, true, true, true, true}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolVectorToScalar) {
  Check<bool>(/*axis=*/0, /*input_shape=*/{5},
              /*input_data=*/{true, false, true, false, true},
              /*exp_output_shape=*/{{}, {}, {}, {}, {}},
              /*exp_output_data=*/{{true}, {false}, {true}, {false}, {true}},
              /*type=*/TensorType_BOOL);
}

}  // namespace
}  // namespace tflite
