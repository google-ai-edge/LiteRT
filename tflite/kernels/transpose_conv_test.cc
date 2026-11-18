/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tflite/core/interpreter.h"
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/string_type.h"
#include "tflite/types/half.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_DEQUANTIZE();
TfLiteRegistration* Register_TRANSPOSECONV_REF();
TfLiteRegistration* Register_TRANSPOSECONV_GENERIC_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename InputType>
class BaseTransposeConvOpModel : public SingleOpModel {
 public:
  BaseTransposeConvOpModel(TfLiteRegistration* registration,
                           std::initializer_list<int> output_shape_data,
                           const TensorData& filter,
                           std::initializer_list<InputType> filter_data,
                           const TensorData& input, const TensorData& output,
                           Padding padding, int stride_w, int stride_h,
                           tflite::ActivationFunctionType fused_activation,
                           TestType test_type, int version = 1,
                           const TensorType& bias_type = TensorType_INT32) {
    // Just to be confusing, transpose_conv has an _input_ named "output_shape"
    // that sets the shape of the output tensor of the op :). It must always be
    // an int32 1D four element tensor.
    if (test_type == TestType::kDynamic) {
      output_shape_ = AddInput({TensorType_INT32, {4}});
      filter_ = AddInput(filter);
    } else {
      output_shape_ = AddConstInput(TensorType_INT32, output_shape_data, {4});
      filter_ = AddConstInput(filter, filter_data);
    }
    input_ = AddInput(input);

    output_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_TRANSPOSE_CONV, BuiltinOptions_TransposeConvOptions,
        CreateTransposeConvOptions(builder_, padding, stride_w, stride_h,
                                   fused_activation, bias_type)
            .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_TRANSPOSE_CONV, registration, version);
    BuildInterpreter(
        {GetShape(output_shape_), GetShape(filter_), GetShape(input_)});

    if (test_type == TestType::kDynamic) {
      PopulateTensor<int32_t>(output_shape_, output_shape_data);
      if (!std::is_same_v<InputType, int16_t> &&
          !std::is_same_v<InputType, int8_t>) {
        PopulateTensor<InputType>(filter_, filter_data);
      }
    }
  }

  void SetInput(std::initializer_list<float> data) {
    if (std::is_same_v<InputType, uint8_t>) {
      QuantizeAndPopulate<uint8_t>(input_, data);
    } else if (std::is_same_v<InputType, int8_t>) {
      QuantizeAndPopulate<int8_t>(input_, data);
    } else if (std::is_same_v<InputType, int16_t>) {
      QuantizeAndPopulate<int16_t>(input_, data);
    } else {
      PopulateTensor(input_, data);
    }
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int output_shape_;
  int filter_;
  int input_;
  int output_;
};

class TransposeConvOpModel : public BaseTransposeConvOpModel<float> {
 public:
  using BaseTransposeConvOpModel::BaseTransposeConvOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

template <typename InputType>
class PrepareOnlyTransposeConvOpModel : public SingleOpModel {
 public:
  PrepareOnlyTransposeConvOpModel(
      TfLiteRegistration* registration,
      std::initializer_list<int> output_shape_data, const TensorData& filter,
      const TensorData& input, const TensorData& output, Padding padding,
      int stride_w, int stride_h,
      tflite::ActivationFunctionType fused_activation, int version = 1,
      const TensorType& bias_type = TensorType_INT32) {
    output_shape_ = AddConstInput(TensorType_INT32, output_shape_data, {4});
    filter_ = AddInput(filter);
    input_ = AddInput(input);
    output_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_TRANSPOSE_CONV, BuiltinOptions_TransposeConvOptions,
        CreateTransposeConvOptions(builder_, padding, stride_w, stride_h,
                                   fused_activation, bias_type)
            .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_TRANSPOSE_CONV, registration, version);
    BuildInterpreter(
        {GetShape(output_shape_), GetShape(filter_), GetShape(input_)},
        /*num_threads=*/1, /*allow_fp32_relax_to_fp16=*/false,
        /*apply_delegate=*/false,
        /*allocate_and_delegate=*/false);
  }

 private:
  int output_shape_;
  int filter_;
  int input_;
  int output_;
};

const auto kKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_TRANSPOSECONV_REF()},
    {"GenericOptimized", ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT()},
});

class TransposeConvOpTest
    : public ::testing::TestWithParam<std::tuple<string, TestType>> {
 public:
  TfLiteRegistration* GetRegistration() {
    return kKernelMap->at(std::get<0>(GetParam()));
  }
  TestType GetTestType() { return std::get<1>(GetParam()); }
};

TEST(TransposeConvPrepareSecurityTest, RejectsCol2ImOverflow) {
  constexpr int kHugeDim = 46341;
  PrepareOnlyTransposeConvOpModel<uint8_t> m(
      ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT(), {1, 1, 1, 0},
      {TensorType_UINT8, {1, 1, 1, 1}, 0.0f, 255.0f},
      {TensorType_UINT8, {1, kHugeDim, kHugeDim, 1}, 0.0f, 255.0f},
      {TensorType_UINT8, {}}, Padding_SAME, /*stride_w=*/1, /*stride_h=*/1,
      ActivationFunctionType_NONE);

  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

TEST(TransposeConvPrepareSecurityTest, RejectsCol2ImTotalElementsOverflow) {
  if (sizeof(void*) <= 4) {
    GTEST_SKIP() << "Interpreter construction overflows before kernel Prepare "
                    "on 32-bit.";
  }
  // col2im_rows = 512 * 512 = 262,144 and col2im_columns = 2048 * 2 * 2 = 8192
  // both fit in int32_t, but col2im_rows * col2im_columns = 2^31 overflows
  // 32-bit signed int.
  PrepareOnlyTransposeConvOpModel<uint8_t> m(
      ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT(), {1, 512, 512, 2048},
      {TensorType_UINT8, {2048, 2, 2, 1}, 0.0f, 255.0f},
      {TensorType_UINT8, {1, 512, 512, 1}, 0.0f, 255.0f},
      {TensorType_UINT8, {}, 0.0f, 255.0f}, Padding_SAME, /*stride_w=*/1,
      /*stride_h=*/1, ActivationFunctionType_NONE);

  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

TEST(TransposeConvPrepareSecurityTest, RejectsOutputElementCountOverflow) {
  if (sizeof(void*) <= 4) {
    GTEST_SKIP() << "Interpreter construction overflows before kernel Prepare "
                    "on 32-bit.";
  }
  // With stride 2, input has 23171 * 23171 = 536,895,241 <= INT32_MAX elements,
  // while output_shape has 46342 * 46342 = 2,147,580,964 > INT32_MAX elements.
  for (TfLiteRegistration* registration :
       {ops::builtin::Register_TRANSPOSECONV_REF(),
        ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT()}) {
    PrepareOnlyTransposeConvOpModel<int8_t> m(
        registration, {1, 46342, 46342, 1},
        {TensorType_INT8, {1, 1, 1, 1}, -1.0f, 1.0f},
        {TensorType_INT8, {1, 23171, 23171, 1}, -1.0f, 1.0f},
        {TensorType_INT8, {}, -1.0f, 1.0f}, Padding_SAME, /*stride_w=*/2,
        /*stride_h=*/2, ActivationFunctionType_NONE);

    EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
  }
}

TEST(TransposeConvPrepareSecurityTest, RejectsHybridInputOverflow) {
  if (sizeof(void*) <= 4) {
    GTEST_SKIP() << "Interpreter construction overflows before kernel Prepare "
                    "on 32-bit.";
  }
  constexpr int kHugeDim = 46341;
  PrepareOnlyTransposeConvOpModel<float> m(
      ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT(), {1, 1, 1, 1},
      {TensorType_INT8, {1, 1, 1, 1}, -1.0f, 1.0f},
      {TensorType_FLOAT32, {kHugeDim, kHugeDim, 1, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, /*stride_w=*/1,
      /*stride_h=*/1, ActivationFunctionType_NONE);

  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

TEST(TransposeConvPrepareSecurityTest, RejectsUnsupportedWeightsType) {
  PrepareOnlyTransposeConvOpModel<float> m(
      ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT(), {1, 1, 1, 1},
      {TensorType_UINT8, {1, 1, 1, 1}}, {TensorType_FLOAT32, {1, 1, 1, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, /*stride_w=*/1, /*stride_h=*/1,
      ActivationFunctionType_NONE);

  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

TEST(TransposeConvPrepareSecurityTest, RejectsStrideOutsideInt16Range) {
  constexpr int kTooLarge = std::numeric_limits<int16_t>::max() + 1;
  PrepareOnlyTransposeConvOpModel<float> m(
      ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT(), {1, 1, 1, 1},
      {TensorType_FLOAT32, {1, 1, 1, 1}}, {TensorType_FLOAT32, {1, 1, 1, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, /*stride_w=*/kTooLarge,
      /*stride_h=*/1, ActivationFunctionType_NONE);

  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

TEST(TransposeConvPrepareSecurityTest, RejectsPaddingOutsideInt16Range) {
  constexpr int kFilterWidth =
      2 * (std::numeric_limits<int16_t>::max() + 1) + 1;
  PrepareOnlyTransposeConvOpModel<float> m(
      ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT(), {1, 1, 1, 1},
      {TensorType_FLOAT32, {1, 1, kFilterWidth, 1}},
      {TensorType_FLOAT32, {1, 1, 1, 1}}, {TensorType_FLOAT32, {}},
      Padding_SAME, /*stride_w=*/1, /*stride_h=*/1,
      ActivationFunctionType_NONE);

  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

TEST(TransposeConvPrepareSecurityTest, RejectsMismatchedOutputChannels) {
  PrepareOnlyTransposeConvOpModel<float> m(
      ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT(), {1, 1, 1, 2},
      {TensorType_FLOAT32, {1, 1, 1, 1}}, {TensorType_FLOAT32, {1, 1, 1, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, /*stride_w=*/1, /*stride_h=*/1,
      ActivationFunctionType_NONE);

  ASSERT_EQ(m.AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(m.Invoke(), kTfLiteError);
}

TEST(TransposeConvPrepareSecurityTest, RejectsInconsistentSpatialShape) {
  PrepareOnlyTransposeConvOpModel<float> m(
      ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT(), {1, 3, 3, 1},
      {TensorType_FLOAT32, {1, 1, 1, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, /*stride_w=*/1, /*stride_h=*/1,
      ActivationFunctionType_NONE);

  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

class PrepareOnlyInvalidOutputShapeTransposeConvOpModel : public SingleOpModel {
 public:
  explicit PrepareOnlyInvalidOutputShapeTransposeConvOpModel(
      std::initializer_list<int> output_shape_data,
      std::initializer_list<int> output_shape_dims) {
    output_shape_ =
        AddConstInput(TensorType_INT32, output_shape_data, output_shape_dims);
    filter_ = AddInput({TensorType_FLOAT32, {1, 1, 1, 1}});
    input_ = AddInput({TensorType_FLOAT32, {1, 1, 1, 1}});
    output_ = AddOutput({TensorType_FLOAT32, {}});

    SetBuiltinOp(
        BuiltinOperator_TRANSPOSE_CONV, BuiltinOptions_TransposeConvOptions,
        CreateTransposeConvOptions(builder_, Padding_SAME, /*stride_w=*/1,
                                   /*stride_h=*/1, ActivationFunctionType_NONE)
            .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_TRANSPOSE_CONV,
        ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT(), /*version=*/1);
    BuildInterpreter(
        {GetShape(output_shape_), GetShape(filter_), GetShape(input_)},
        /*num_threads=*/1, /*allow_fp32_relax_to_fp16=*/false,
        /*apply_delegate=*/false,
        /*allocate_and_delegate=*/false);
  }

 private:
  int output_shape_;
  int filter_;
  int input_;
  int output_;
};

TEST(TransposeConvPrepareSecurityTest, RejectsInvalidOutputShapeTensorLength) {
  PrepareOnlyInvalidOutputShapeTransposeConvOpModel m({1}, {1});
  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

// Test case:
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 4, 4, 1 ]),
//     tf.constant(np.arange(1, 10), shape=[ 3, 3, 1, 1 ], dtype=tf.float32),
//     tf.constant(np.arange(1, 17), shape=[ 1, 4, 4, 1 ], dtype=tf.float32),
//     [1, 1, 1, 1 ],
//     "SAME")
TEST_P(TransposeConvOpTest, SimpleTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1}, {TensorType_FLOAT32, {1, 3, 3, 1}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {TensorType_FLOAT32, {1, 4, 4, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, 1, 1, ActivationFunctionType_NONE,
      GetTestType());
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      model.GetOutput(),
      Pointwise(FloatingPointEq(), {29, 62, 83, 75, 99, 192, 237, 198, 207, 372,
                                    417, 330, 263, 446, 485, 365}));
  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

// Test case:
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 4, 4, 1 ]),
//     tf.constant(np.arange(1, 10), shape=[ 3, 3, 1, 1 ], dtype=tf.float32),
//     tf.constant(np.arange(1, 17), shape=[ 1, 4, 4, 1 ], dtype=tf.float32),
//     [1, 1, 1, 1 ],
//     "SAME")
TEST_P(TransposeConvOpTest, fusedRELUTest) {
  // NNAPI can not support for the new kernel behaviors at the moment.
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  TransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1}, {TensorType_FLOAT32, {1, 3, 3, 1}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {TensorType_FLOAT32, {1, 4, 4, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, 1, 1, ActivationFunctionType_RELU,
      GetTestType());
  model.SetInput({1, 2, -3, -4, 5, 6, -7, -8, 9, 10, -11, -12, 13, 14, 15, 16});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              Pointwise(FloatingPointEq(), {29, 24, 0, 0, 99, 72, 0, 0, 207,
                                            186, 0, 0, 263, 292, 141, 0}));
  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 19),
//                      shape=[ 3, 3, 1, 2 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 4, 4, 1 ]),
//     filter,
//     tf.constant(np.arange(1, 33), shape=[ 1, 4, 4, 2 ], dtype=tf.float32),
//     [1, 1, 1, 1 ],
//     "SAME")
// And filter value is derived by:
// filter = tf.reshape(tf.transpose(filter, perm=[3, 0, 1, 2]), shape=[18, 1])
TEST_P(TransposeConvOpTest, TwoFiltersTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1}, {TensorType_FLOAT32, {1, 3, 3, 2}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
      {TensorType_FLOAT32, {1, 4, 4, 2}}, {TensorType_FLOAT32, {}},
      Padding_SAME, 1, 1, ActivationFunctionType_NONE, GetTestType());
  model.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              Pointwise(FloatingPointEq(),
                        {184, 412, 568, 528, 678, 1347, 1689, 1434, 1494, 2715,
                         3057, 2442, 1968, 3352, 3652, 2760}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 19),
//                      shape=[ 3, 3, 1, 2 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 6, 6, 1 ]),
//     filter,
//     tf.constant(np.arange(1, 33), shape=[ 1, 4, 4, 2 ], dtype=tf.float32),
//     [1, 1, 1, 1 ],
//     "VALID")
// And filter value is derived by:
// filter = tf.reshape(tf.transpose(filter, perm=[3, 0, 1, 2]), shape=[1, 18])
TEST_P(TransposeConvOpTest, PaddingValidTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 6, 6, 1}, {TensorType_FLOAT32, {1, 3, 3, 2}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
      {TensorType_FLOAT32, {1, 4, 4, 2}}, {TensorType_FLOAT32, {}},
      Padding_VALID, 1, 1, ActivationFunctionType_NONE, GetTestType());
  model.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      model.GetOutput(),
      Pointwise(FloatingPointEq(),
                {5,    22,   59,   101,  114,  83,   52,   184,  412,
                 568,  528,  344,  237,  678,  1347, 1689, 1434, 879,
                 597,  1494, 2715, 3057, 2442, 1431, 856,  1968, 3352,
                 3652, 2760, 1548, 689,  1534, 2543, 2729, 2010, 1103}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6, 6, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 10),
//                      shape=[ 3, 3, 1, 1 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 5, 5, 1 ]),
//     filter,
//     tf.constant(np.arange(1, 5), shape=[ 1, 2, 2, 1 ], dtype=tf.float32),
//     [1, 2, 2, 1 ],
//     "VALID")
TEST_P(TransposeConvOpTest, StrideValidTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 5, 5, 1}, {TensorType_FLOAT32, {1, 3, 3, 1}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, Padding_VALID, 2, 2,
      ActivationFunctionType_NONE, GetTestType());
  model.SetInput({1, 2, 3, 4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              Pointwise(FloatingPointEq(),
                        {1,  2,  5,  4,  6,  4,  5,  14, 10, 12, 10, 14, 36,
                         24, 30, 12, 15, 34, 20, 24, 21, 24, 55, 32, 36}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 19),
//                      shape=[ 3, 3, 2, 1 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 5, 5, 2 ]),
//     filter,
//     tf.constant(np.arange(1, 5), shape=[ 1, 2, 2, 1 ], dtype=tf.float32),
//     [1, 2, 2, 1 ],
//     "VALID")
TEST_P(TransposeConvOpTest, MultiChannelTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 5, 5, 2}, {TensorType_FLOAT32, {2, 3, 3, 1}},
      {1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18},
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {}},
      Padding_VALID, 2, 2, ActivationFunctionType_NONE, GetTestType());
  model.SetInput({1, 2, 3, 4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              Pointwise(FloatingPointEq(),
                        {1,  2,  3,  4,  7,  10,  6,   8,  10, 12, 7,  8,  9,
                         10, 25, 28, 18, 20, 22,  24,  16, 20, 24, 28, 62, 72,
                         42, 48, 54, 60, 21, 24,  27,  30, 61, 68, 36, 40, 44,
                         48, 39, 42, 45, 48, 103, 110, 60, 64, 68, 72}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

// Test case:
// filter = tf.constant(np.random.randint(1, 10, size=9),
//                      shape=[ 3, 3, 1, 1 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 3, 4, 1 ]),
//     filter,
//     tf.constant([323, 521], shape=[ 1, 1, 2, 1], dtype=tf.float32),
//     [1, 3, 3, 1 ],
//     "SAME")
// And filter value is derived by:
// filter = tf.reshape(tf.transpose(filter, perm=[3, 0, 1, 2]), shape=[-1])
TEST_P(TransposeConvOpTest, AccuracyTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 3, 4, 1}, {TensorType_FLOAT32, {1, 3, 3, 1}},
      {9, 5, 6, 9, 8, 5, 3, 1, 4}, {TensorType_FLOAT32, {1, 1, 2, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, 3, 3, ActivationFunctionType_NONE,
      GetTestType());
  model.SetInput({323, 521});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({1615., 1938., 4689., 2605., 2584., 1615.,
                                  4689., 4168., 323., 1292., 1563., 521.})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3, 4, 1}));
}

// Test case:
// filter = tf.constant(np.random.randint(1, 10, size=9),
//                      shape=[ 3, 3, 1, 1 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 3, 4, 1 ]),
//     filter,
//     tf.constant([323, 521], shape=[ 1, 1, 2, 1], dtype=tf.float32),
//     [1, 3, 3, 1 ],
//     "SAME")
// And filter value is derived by:
// filter = tf.reshape(tf.transpose(filter, perm=[3, 0, 1, 2]), shape=[-1])
TEST_P(TransposeConvOpTest, AccuracyWithFusedActivationTest) {
  // NNAPI can not support for the new kernel behaviors at the moment.
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  TransposeConvOpModel model(
      GetRegistration(), {1, 3, 4, 1}, {TensorType_FLOAT32, {1, 3, 3, 1}},
      {9, 5, 6, 9, 8, 5, 3, 1, 4}, {TensorType_FLOAT32, {1, 1, 2, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, 3, 3, ActivationFunctionType_RELU,
      GetTestType());
  model.SetInput({323, -521});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {1615, 1938, 0, 0, 2584, 1615, 0, 0, 323, 1292, 0, 0})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3, 4, 1}));
}

class QuantizedTransposeConvOpModel : public BaseTransposeConvOpModel<uint8_t> {
 public:
  using BaseTransposeConvOpModel::BaseTransposeConvOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST_P(TransposeConvOpTest, SimpleTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137,
                                                139, 141, 143, 145};
  QuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_UINT8, {1, 3, 3, 1}, -63, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 1}, -63, 64},
      {TensorType_UINT8, {}, -508, 512}, Padding_SAME, 1, 1,
      ActivationFunctionType_NONE, GetTestType());
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({36, 72, 96, 84, 116, 216, 268, 220, 232,
                                       412, 464, 360, 284, 480, 512, 388},
                                      1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_P(TransposeConvOpTest, SimpleTestWithFusedActivationQuantized) {
  // NNAPI can not support for the new kernel behaviors at the moment.
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137,
                                                139, 141, 143, 145};
  QuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_UINT8, {1, 3, 3, 1}, -63, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 1}, -63, 64},
      {TensorType_UINT8, {}, -508, 512}, Padding_SAME, 1, 1,
      ActivationFunctionType_RELU, GetTestType());
  model.SetInput({1, 2, -3, -4, 5, 6, -7, -8, 9, 10, -11, -12, 13, 14, 15, 16});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({36, 24, 0, 0, 116, 76, 0, 0, 232,
                                               212, 0, 0, 284, 316, 156, 0})));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_P(TransposeConvOpTest, TwoFiltersTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  // 18}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137, 139,
                                                141, 143, 145, 147, 149, 151,
                                                153, 155, 157, 159, 161, 163};
  QuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_UINT8, {1, 3, 3, 2}, -63, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 2}, -63, 64},
      {TensorType_UINT8, {}, -4064, 4096}, Padding_SAME, 1, 1,
      ActivationFunctionType_NONE, GetTestType());
  model.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {224, 448, 608, 576, 736, 1440, 1792, 1504, 1600, 2880, 3232,
                   2560, 2048, 3456, 3776, 2848},
                  1e-5)));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_P(TransposeConvOpTest, PaddingValidTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  // 18}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137, 139,
                                                141, 143, 145, 147, 149, 151,
                                                153, 155, 157, 159, 161, 163};
  QuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 6, 6, 1},
      {TensorType_UINT8, {1, 3, 3, 2}, -63, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 2}, -63, 64},
      {TensorType_UINT8, {}, -4064, 4096}, Padding_VALID, 1, 1,
      ActivationFunctionType_NONE, GetTestType());
  model.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0,    32,   64,   128,  128,  96,   64,   224,  448,
                   608,  576,  352,  256,  736,  1440, 1792, 1504, 928,
                   640,  1600, 2880, 3232, 2560, 1504, 896,  2048, 3456,
                   3776, 2848, 1600, 704,  1568, 2592, 2784, 2048, 1120},
                  1e-5)));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6, 6, 1}));
}

class PerChannelQuantizedTransposeConvOpModel
    : public BaseTransposeConvOpModel<int8_t> {
 public:
  using BaseTransposeConvOpModel::BaseTransposeConvOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int8_t>(ExtractVector<int8_t>(output_), GetScale(output_),
                              GetZeroPoint(output_));
  }

  void SetFilter(const std::initializer_list<float>& data) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }
};

TEST_P(TransposeConvOpTest, SimpleTestQuantizedPerChannelSingleChannel) {
  const std::initializer_list<float> filter_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::initializer_list<int8_t> const_filter_data = {14, 28, 42,  56, 71,
                                                           85, 99, 113, 127};
  PerChannelQuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_INT8, {1, 3, 3, 1}, 0, 0, 0, 0, true, {9.0 / 127}, {0}, 0},
      const_filter_data,
      {TensorType_INT8, {1, 4, 4, 1}, 0, 0, 16.0 / 127, -128},
      {TensorType_INT8, {}, 0, 0, 2, -128}, Padding_SAME, 1, 1,
      ActivationFunctionType_NONE, GetTestType(),
      /* version */ 2);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({30, 62, 84, 76, 100, 192, 238, 198, 206,
                                       372, 416, 330, 262, 446, 484, 366},
                                      1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

// Test data copied from the float multi-channel test above.
TEST_P(TransposeConvOpTest, TestQuantizedPerChannelMultiChannel) {
  const std::initializer_list<float> filter_data = {
      1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  const std::initializer_list<int8_t> const_filter_data = {
      7,  22, 37, 52, 67, 82, 97, 112, 127,
      14, 28, 42, 56, 71, 85, 99, 113, 127};
  PerChannelQuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 5, 5, 2},
      {TensorType_INT8,
       {2, 3, 3, 1},
       0,
       0,
       0,
       0,
       true,
       {17.0 / 127, 18.0 / 127},
       {0, 0},
       0},
      const_filter_data, {TensorType_INT8, {1, 2, 2, 1}, 0, 0, 4.0 / 255, -128},
      {TensorType_INT8, {}, 0, 0, 1, -128}, Padding_VALID, 2, 2,
      ActivationFunctionType_NONE, GetTestType(),
      /* version */ 2);
  model.SetInput({1, 2, 3, 4});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(
          {1,  2,  3,  4,  7,  10, 6,  8,  10, 12, 7,   8,   9,  10, 25, 28, 18,
           20, 22, 24, 16, 20, 24, 28, 62, 72, 42, 48,  54,  60, 21, 24, 27, 30,
           61, 68, 36, 40, 44, 48, 39, 42, 45, 48, 103, 110, 60, 64, 68, 72},
          1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

// Test data copied from the float multi-channel test above.
TEST_P(TransposeConvOpTest, TestQuantizedPerTensorMultiChannel) {
  const std::initializer_list<float> filter_data = {
      1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  const std::initializer_list<int8_t> const_filter_data = {
      7,  21, 35, 49, 64, 78, 92, 106, 120,
      14, 28, 42, 56, 71, 85, 99, 113, 127};
  PerChannelQuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 5, 5, 2},
      {TensorType_INT8,
       {2, 3, 3, 1},
       0,
       0,
       0,
       0,
       true,
       {18.0 / 127, 18.0 / 127},
       {0, 0},
       0},
      const_filter_data, {TensorType_INT8, {1, 2, 2, 1}, 0, 0, 4.0 / 255, -128},
      {TensorType_INT8, {}, 0, 0, 1, -128}, Padding_VALID, 2, 2,
      ActivationFunctionType_NONE, GetTestType(),
      /* version */ 2);
  model.SetInput({1, 2, 3, 4});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(
          {1,  2,  3,  4,  7,  10, 6,  8,  10, 12, 7,   8,   9,  10, 25, 28, 18,
           20, 22, 24, 16, 20, 24, 28, 62, 72, 42, 48,  54,  60, 21, 24, 27, 30,
           61, 68, 36, 40, 44, 48, 39, 42, 45, 48, 103, 110, 60, 64, 68, 72},
          1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

class PerChannelQuantizedTransposeConvOpModel16x8
    : public BaseTransposeConvOpModel<int16_t> {
 public:
  using BaseTransposeConvOpModel::BaseTransposeConvOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int16_t>(ExtractVector<int16_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

  void SetFilter(const std::initializer_list<float>& data) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }
};

TEST_P(TransposeConvOpTest, SimpleTestQuantizedPerChannel16x8NoBiasInt32) {
  const std::initializer_list<float> filter_data = {
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      1, 2,  // out channel = 0, y = 0, x = 0
      3, 4,  // out channel = 0, y = 0, x = 1
      3, 4,  // out channel = 0, y = 1, x = 0
      5, 6,  // out channel = 0, y = 1, x = 1
      7, 8,  // out channel = 1, y = 0, x = 0
      5, 6,  // out channel = 1, y = 0, x = 1
      3, 4,  // out channel = 1, y = 1, x = 0
      1, 2,  // out channel = 1, y = 1, x = 1
  };
  PerChannelQuantizedTransposeConvOpModel16x8 model(
      GetRegistration(),
      /*output_shape_data=*/{1, 2, 3, 2},
      /*filter=*/
      {TensorType_INT8,
       /*shape=*/{2, 2, 2, 2},
       /*min=*/-64, /*max=*/64,
       /*scale=*/0, /*zero_point=*/0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{7.0 / 127, 8.0 / 127},
       /*per_channel_quantization_offsets=*/{0, 0},
       /*channel_index=*/0},
      /*filter_data=*/{},
      /*input=*/
      {TensorType_INT16,
       /*shape=*/{1, 2, 3, 2},
       /*min=*/0, /*max=*/0,
       /*scale=*/4.0 / 127,
       /*zero_point=*/0},
      /*output=*/
      {TensorType_INT16,
       /*shape=*/{},
       /*min=*/0, /*max=*/0,
       /*scale=*/1.0,
       /*zero_point=*/0},
      /*padding=*/Padding_SAME,
      /*stride_w=*/1, /*stride_h=*/1,
      /*fused_activation=*/ActivationFunctionType_NONE, GetTestType(),
      /*bias_type=*/TensorType_INT32);
  model.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  model.SetFilter(filter_data);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {7, 37, 16, 26, -9, -39, 27, 69, 48, 42, -32, -74}, 1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 3, 2}));
}

TEST_P(TransposeConvOpTest,
       SimpleTestWithFusedActivationQuantizedPerChannel16x8) {
  // NNAPI can not support for the new kernel behaviors at the moment.
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  const std::initializer_list<float> filter_data = {
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      1, 2,  // out channel = 0, y = 0, x = 0
      3, 4,  // out channel = 0, y = 0, x = 1
      3, 4,  // out channel = 0, y = 1, x = 0
      5, 6,  // out channel = 0, y = 1, x = 1
      7, 8,  // out channel = 1, y = 0, x = 0
      5, 6,  // out channel = 1, y = 0, x = 1
      3, 4,  // out channel = 1, y = 1, x = 0
      1, 2,  // out channel = 1, y = 1, x = 1
  };
  PerChannelQuantizedTransposeConvOpModel16x8 model(
      GetRegistration(),
      /*output_shape_data=*/{1, 2, 3, 2},
      /*filter=*/
      {TensorType_INT8,
       /*shape=*/{2, 2, 2, 2},
       /*min=*/-64, /*max=*/64,
       /*scale=*/0, /*zero_point=*/0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{7.0 / 127, 8.0 / 127},
       /*per_channel_quantization_offsets=*/{0, 0},
       /*channel_index=*/0},
      /*filter_data=*/{},
      /*input=*/
      {TensorType_INT16,
       /*shape=*/{1, 2, 3, 2},
       /*min=*/0, /*max=*/0,
       /*scale=*/4.0 / 127,
       /*zero_point=*/0},
      /*output=*/
      {TensorType_INT16,
       /*shape=*/{},
       /*min=*/0, /*max=*/0,
       /*scale=*/1.0,
       /*zero_point=*/0},
      /*padding=*/Padding_SAME,
      /*stride_w=*/1, /*stride_h=*/1,
      /*fused_activation=*/ActivationFunctionType_RELU, GetTestType());
  model.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  model.SetFilter(filter_data);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {7, 37, 16, 26, 0, 0, 27, 69, 48, 42, 0, 0}, 1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 3, 2}));
}

TEST_P(TransposeConvOpTest, SimpleTestQuantizedPerChannel16x8NoBiasInt64) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9}
  const std::initializer_list<float> filter_data = {
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      1, 2,  // out channel = 0, y = 0, x = 0
      3, 4,  // out channel = 0, y = 0, x = 1
      3, 4,  // out channel = 0, y = 1, x = 0
      5, 6,  // out channel = 0, y = 1, x = 1
      7, 8,  // out channel = 1, y = 0, x = 0
      5, 6,  // out channel = 1, y = 0, x = 1
      3, 4,  // out channel = 1, y = 1, x = 0
      1, 2,  // out channel = 1, y = 1, x = 1
  };
  PerChannelQuantizedTransposeConvOpModel16x8 model(
      GetRegistration(),
      /*output_shape_data=*/{1, 2, 3, 2},
      /*filter=*/
      {TensorType_INT8,
       /*shape=*/{2, 2, 2, 2},
       /*min=*/-64, /*max=*/64,
       /*scale=*/0, /*zero_point=*/0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{7.0 / 127, 8.0 / 127},
       /*per_channel_quantization_offsets=*/{0, 0},
       /*channel_index=*/0},
      /*filter_data=*/{},
      /*input=*/
      {TensorType_INT16,
       /*shape=*/{1, 2, 3, 2},
       /*min=*/0, /*max=*/0,
       /*scale=*/4.0 / 127,
       /*zero_point=*/0},
      /*output=*/
      {TensorType_INT16,
       /*shape=*/{},
       /*min=*/0, /*max=*/0,
       /*scale=*/1.0,
       /*zero_point=*/0},
      /*padding=*/Padding_SAME,
      /*stride_w=*/1, /*stride_h=*/1,
      /*fused_activation=*/ActivationFunctionType_NONE, GetTestType(),
      /*version=*/1,
      /*bias_type=*/TensorType_INT64);
  model.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  model.SetFilter(filter_data);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {7, 37, 16, 26, -9, -39, 27, 69, 48, 42, -32, -74}, 1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 3, 2}));
}

template <typename InputType, typename FilterType>
class BaseTransposeConvBiasOpModel : public SingleOpModel {
 public:
  BaseTransposeConvBiasOpModel(
      TfLiteRegistration* registration,
      std::initializer_list<int> output_shape_data, const TensorData& filter,
      std::initializer_list<FilterType> filter_data, const TensorData& input,
      const TensorData& output, Padding padding, int stride_w, int stride_h,
      tflite::ActivationFunctionType fused_activation, TestType test_type,
      int version = 3, const TensorType& bias_type = TensorType_INT32,
      std::optional<TensorType> quantized_bias_type = std::nullopt) {
    bias_type_ = bias_type;
    if (test_type == TestType::kDynamic) {
      output_shape_ = AddInput({TensorType_INT32, {4}});
      filter_ = AddInput(filter);
    } else {
      output_shape_ = AddConstInput(TensorType_INT32, output_shape_data, {4});
      filter_ = AddConstInput(filter, filter_data);
    }
    input_ = AddInput(input);
    int bias_size = GetShape(filter_)[0];
    if (input.type == TensorType_FLOAT32) {
      bias_type_ = TensorType_FLOAT32;
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    } else {
      if (filter.per_channel_quantization) {
        // per channel quantization.
        std::vector<float> bias_scale(
            filter.per_channel_quantization_scales.size());
        std::vector<int64_t> bias_zero_points(
            filter.per_channel_quantization_scales.size());
        for (size_t i = 0; i < filter.per_channel_quantization_scales.size();
             ++i) {
          bias_scale[i] =
              input.scale * filter.per_channel_quantization_scales[i];
          bias_zero_points[i] = 0;
        }
        TensorData bias{bias_type,
                        {bias_size},
                        /*min=*/0,
                        /*max=*/0,
                        /*scale=*/0,
                        /*zero_point=*/0,
                        true,
                        /*per_channel_quantization_scales=*/bias_scale,
                        /*per_channel_quantization_offsets=*/bias_zero_points,
                        /*channel_index==*/0};
        bias_ = AddInput(bias);
      } else {
        // per tensor quantization.
        auto bias_scale = GetScale(input_) * GetScale(filter_);
        TensorData bias{bias_type, {bias_size}, 0, 0, bias_scale};
        bias_ = AddInput(bias);
      }
    }

    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_TRANSPOSE_CONV,
                 BuiltinOptions_TransposeConvOptions,
                 CreateTransposeConvOptions(
                     builder_, padding, stride_w, stride_h, fused_activation,
                     quantized_bias_type.value_or(bias_type))
                     .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_TRANSPOSE_CONV, registration, version);
    BuildInterpreter({GetShape(output_shape_), GetShape(filter_),
                      GetShape(input_), GetShape(bias_)});
    if (test_type == TestType::kDynamic) {
      PopulateTensor<int32_t>(output_shape_, output_shape_data);
      if (!std::is_same_v<InputType, int16_t> &&
          !std::is_same_v<InputType, int8_t>) {
        PopulateTensor<FilterType>(filter_, filter_data);
      }
    }
  }

  void SetInput(std::initializer_list<float> data) {
    if (std::is_same_v<InputType, uint8_t>) {
      QuantizeAndPopulate<uint8_t>(input_, data);
    } else if (std::is_same_v<InputType, int8_t>) {
      QuantizeAndPopulate<int8_t>(input_, data);
    } else if (std::is_same_v<InputType, int16_t>) {
      QuantizeAndPopulate<int16_t>(input_, data);
    } else {
      PopulateTensor(input_, data);
    }
  }

  void SetBias(std::initializer_list<float> bias) {
    if (std::is_same_v<InputType, uint8_t>) {
      QuantizeAndPopulate<int32_t>(bias_, bias);
    } else if (std::is_same_v<FilterType, int8_t>) {
      PerChannelQuantizeBias(bias_, bias);
    } else {
      PopulateTensor(bias_, bias);
    }
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int output_shape_;
  int filter_;
  int input_;
  int bias_;
  int output_;
  TensorType bias_type_;
};

class TransposeConvOpBiasModel
    : public BaseTransposeConvBiasOpModel<float, float> {
 public:
  using BaseTransposeConvBiasOpModel::BaseTransposeConvBiasOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

// Test case:
// input_data = np.arange(1, 5).reshape(1,2,2,1).astype(np.float32)
// filter_data = np.arange(1, 19).reshape(3,3,2,1).astype(np.float32)
// bias_data = np.array([3,4])
// input = tf.keras.layers.Input(shape=(2, 2, 1))
// output = tf.keras.layers.Convolution2DTranspose(filters=2,
//                                                 kernel_size=[3, 3],
//                                                 strides=[2, 2],
//                                                 padding="valid")(input)
// model = tf.keras.models.Model(input, output)
// model.layers[1].set_weights([filter_data, bias_data])
// output = model.predict(input_data)
TEST_P(TransposeConvOpTest, MultiChannelBiasTest) {
  TransposeConvOpBiasModel model(
      GetRegistration(), /*output_shape=*/{1, 5, 5, 2},
      /*filter=*/{TensorType_FLOAT32, {2, 3, 3, 1}},
      /*filter_data=*/
      {1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18},
      /*input=*/{TensorType_FLOAT32, {1, 2, 2, 1}},
      /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID,
      /*stride_w=*/2, /*stride_h=*/2,
      /*fused_activation=*/ActivationFunctionType_NONE, GetTestType(),
      /* version */ 3);
  model.SetInput({1, 2, 3, 4});
  model.SetBias({3, 4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              Pointwise(FloatingPointEq(),
                        {4,  6,  6,  8,  10, 14,  9,   12, 13, 16, 10, 12, 12,
                         14, 28, 32, 21, 24, 25,  28,  19, 24, 27, 32, 65, 76,
                         45, 52, 57, 64, 24, 28,  30,  34, 64, 72, 39, 44, 47,
                         52, 42, 46, 48, 52, 106, 114, 63, 68, 71, 76}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

// Test case:
// input_data = np.arange(1, 5).reshape(1,2,2,1).astype(np.float32)
// filter_data = np.arange(1, 19).reshape(3,3,2,1).astype(np.float32)
// bias_data = np.array([3,4])
// input = tf.keras.layers.Input(shape=(2, 2, 1))
// output = tf.keras.layers.Convolution2DTranspose(filters=2,
//                                                 kernel_size=[3, 3],
//                                                 strides=[2, 2],
//                                                 padding="valid")(input)
// model = tf.keras.models.Model(input, output)
// model.layers[1].set_weights([filter_data, bias_data])
// output = model.predict(input_data)
TEST_P(TransposeConvOpTest, MultiChannelBiasWithFusedActivationTest) {
  // NNAPI can not support for the new kernel behaviors at the moment.
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  TransposeConvOpBiasModel model(
      GetRegistration(), /*output_shape=*/{1, 5, 5, 2},
      /*filter=*/{TensorType_FLOAT32, {2, 3, 3, 1}},
      /*filter_data=*/
      {1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18},
      /*input=*/{TensorType_FLOAT32, {1, 2, 2, 1}},
      /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID,
      /*stride_w=*/2, /*stride_h=*/2,
      /*fused_activation=*/ActivationFunctionType_RELU, GetTestType(),
      /* version */ 3);
  model.SetInput({1, 2, -3, 4});
  model.SetBias({3, 4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              Pointwise(FloatingPointEq(),
                        {4,  6,  6,  8,  10, 14, 9,  12, 13, 16, 10, 12, 12,
                         14, 28, 32, 21, 24, 25, 28, 13, 12, 9,  8,  35, 40,
                         45, 52, 57, 64, 0,  0,  0,  0,  0,  0,  39, 44, 47,
                         52, 0,  0,  0,  0,  4,  6,  63, 68, 71, 76}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

class QuantizedTransposeConvBiasOpModel
    : public BaseTransposeConvBiasOpModel<uint8_t, uint8_t> {
 public:
  using BaseTransposeConvBiasOpModel::BaseTransposeConvBiasOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST_P(TransposeConvOpTest, SimpleBiasTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137,
                                                139, 141, 143, 145};
  QuantizedTransposeConvBiasOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_UINT8, {1, 3, 3, 1}, -63.5, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {}, -508, 512}, Padding_SAME, 1, 1,
      ActivationFunctionType_NONE, GetTestType(), /* version */ 3);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  model.SetBias({1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({32, 64, 84, 76, 100, 192, 240, 200, 208,
                                       372, 420, 332, 264, 448, 488, 368},
                                      1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_P(TransposeConvOpTest, SimpleBiasWithFusedActivationTestQuantized) {
  // NNAPI can not support for the new kernel behaviors at the moment.
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137,
                                                139, 141, 143, 145};
  QuantizedTransposeConvBiasOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_UINT8, {1, 3, 3, 1}, -63.5, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {}, -508, 512}, Padding_SAME, 1, 1,
      ActivationFunctionType_RELU, GetTestType(), /* version */ 3);
  model.SetInput(
      {1, 2, -3, -4, 5, 6, -7, -8, 9, 10, -11, -12, 13, 14, -15, -16});
  model.SetBias({1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {32, 24, 0, 0, 100, 72, 0, 0, 208, 156, 0, 0, 264, 172, 0, 0},
                  1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

class PerChannelQuantizedTransposeConvBiasOpModel
    : public BaseTransposeConvBiasOpModel<int8_t, int8_t> {
 public:
  using BaseTransposeConvBiasOpModel::BaseTransposeConvBiasOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int8_t>(ExtractVector<int8_t>(output_), GetScale(output_),
                              GetZeroPoint(output_));
  }

  void SetInput(const std::initializer_list<float>& data) {
    QuantizeAndPopulate<int8_t>(input_, data);
  }

  void SetFilter(const std::initializer_list<float>& data) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }
};

TEST_P(TransposeConvOpTest, SimpleBiasTestQuantizedPerChannelSingleChannel) {
  const std::initializer_list<float> filter_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::initializer_list<int8_t> const_filter_data = {14, 28, 42,  56, 71,
                                                           85, 99, 113, 127};
  PerChannelQuantizedTransposeConvBiasOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_INT8, {1, 3, 3, 1}, 0, 0, 0, 0, true, {9.0 / 127}, {0}, 0},
      const_filter_data,
      {TensorType_INT8, {1, 4, 4, 1}, 0, 0, 16.0 / 255, -128},
      {TensorType_INT8, {}, 0, 0, 2, -128}, Padding_SAME, 1, 1,
      ActivationFunctionType_NONE, GetTestType(), /* version */ 3);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }
  model.SetBias({1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({30, 62, 84, 76, 100, 194, 238, 200, 208,
                                       372, 418, 330, 264, 446, 486, 366},
                                      1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

class PerChannel16x8TransposeConvBiasOpModel
    : public BaseTransposeConvBiasOpModel<int16_t, int8_t> {
 public:
  using BaseTransposeConvBiasOpModel::BaseTransposeConvBiasOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int16_t>(ExtractVector<int16_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

  void SetFilter(const std::initializer_list<float>& data) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }
};

// The int32 bias selects the int32 path whether `quantized_bias_type` says so
// or keeps its FLOAT32 default, as in models written before the option existed.
TEST_P(TransposeConvOpTest, SimpleBiasTestQuantizedPerChannel16x8Bias32) {
  for (TensorType quantized_bias_type :
       {TensorType_INT32, TensorType_FLOAT32}) {
    SCOPED_TRACE(EnumNameTensorType(quantized_bias_type));
    const float scale = 128.0 / 65536;

    const std::initializer_list<float> filter_data = {
        // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
        1, 2,  // out channel = 0, y = 0, x = 0
        3, 4,  // out channel = 0, y = 0, x = 1
        3, 4,  // out channel = 0, y = 1, x = 0
        5, 6,  // out channel = 0, y = 1, x = 1
        7, 8,  // out channel = 1, y = 0, x = 0
        5, 6,  // out channel = 1, y = 0, x = 1
        3, 4,  // out channel = 1, y = 1, x = 0
        1, 2,  // out channel = 1, y = 1, x = 1
    };
    PerChannel16x8TransposeConvBiasOpModel model(
        GetRegistration(),
        /*output_shape_data=*/{1, 2, 3, 2},
        /*filter=*/
        {TensorType_INT8,
         /*shape=*/{2, 2, 2, 2},
         /*min=*/-64, /*max=*/64,
         /*scale=*/0, /*zero_point=*/0,
         /*per_channel_quantization=*/true,
         /*per_channel_quantization_scales=*/{7.0 / 127, 8.0 / 127},
         /*per_channel_quantization_offsets=*/{0, 0},
         /*channel_index=*/0},
        /*filter_data=*/{},
        /*input=*/
        {TensorType_INT16,
         /*shape=*/{1, 2, 3, 2},
         /*min=*/0, /*max=*/0,
         /*scale=*/4.0 / 127,
         /*zero_point=*/0},
        /*output=*/
        {TensorType_INT16,
         /*shape=*/{},
         /*min=*/0, /*max=*/0,
         /*scale=*/scale,
         /*zero_point=*/0},
        /*padding=*/Padding_SAME,
        /*stride_w=*/1, /*stride_h=*/1,
        /*fused_activation=*/ActivationFunctionType_NONE, GetTestType(),
        /*version=*/3, /*bias_type=*/TensorType_INT32, quantized_bias_type);
    model.SetInput({
        // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
        3,
        2,  // batch = 0, y = 0, x = 0
        1,
        -1,  // batch = 0, y = 0, x = 1
        -2,
        -3,  // batch = 0, y = 0, x = 2
        4,
        3,  // batch = 0, y = 1, x = 0
        2,
        -2,  // batch = 0, y = 1, x = 1
        -3,
        -4,  // batch = 0, y = 1, x = 2
    });
    model.SetFilter(filter_data);
    model.SetBias({3, -2});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    EXPECT_THAT(
        model.GetDequantizedOutput(),
        ElementsAreArray(ArrayFloatNear(
            {10, 35, 19, 24, -6, -41, 30, 64, 51, 40, -29, -64}, 0.19)));

    // GetOutputShape() should always be same as model.SetOutputShape(...);
    EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 3, 2}));
  }
}

TEST_P(TransposeConvOpTest, SimpleBiasTestQuantizedPerChannel16x8Bias64) {
  const float scale = 128.0 / 65536;

  const std::initializer_list<float> filter_data = {
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      1, 2,  // out channel = 0, y = 0, x = 0
      3, 4,  // out channel = 0, y = 0, x = 1
      3, 4,  // out channel = 0, y = 1, x = 0
      5, 6,  // out channel = 0, y = 1, x = 1
      7, 8,  // out channel = 1, y = 0, x = 0
      5, 6,  // out channel = 1, y = 0, x = 1
      3, 4,  // out channel = 1, y = 1, x = 0
      1, 2,  // out channel = 1, y = 1, x = 1
  };
  PerChannel16x8TransposeConvBiasOpModel model(
      GetRegistration(),
      /*output_shape_data=*/{1, 2, 3, 2},
      /*filter=*/
      {TensorType_INT8,
       /*shape=*/{2, 2, 2, 2},
       /*min=*/-64, /*max=*/64,
       /*scale=*/0, /*zero_point=*/0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{7.0 / 127, 8.0 / 127},
       /*per_channel_quantization_offsets=*/{0, 0},
       /*channel_index=*/0},
      /*filter_data=*/{},
      /*input=*/
      {TensorType_INT16,
       /*shape=*/{1, 2, 3, 2},
       /*min=*/0, /*max=*/0,
       /*scale=*/4.0 / 127,
       /*zero_point=*/0},
      /*output=*/
      {TensorType_INT16,
       /*shape=*/{},
       /*min=*/0, /*max=*/0,
       /*scale=*/scale,
       /*zero_point=*/0},
      /*padding=*/Padding_SAME,
      /*stride_w=*/1, /*stride_h=*/1,
      /*fused_activation=*/ActivationFunctionType_NONE, GetTestType(),
      /*version=*/3, /*bias_type=*/TensorType_INT64);
  model.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  model.SetFilter(filter_data);
  model.SetBias({3, -2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {10, 35, 19, 24, -6, -41, 30, 64, 51, 40, -29, -64}, 0.19)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 3, 2}));
}

class HybridTransposeConvOpModel
    : public BaseTransposeConvBiasOpModel<float, int8_t> {
 public:
  using BaseTransposeConvBiasOpModel::BaseTransposeConvBiasOpModel;

  void SetFilter(std::initializer_list<float> f) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, f);
  }

  void SetBias(std::initializer_list<float> b) { PopulateTensor(bias_, b); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

TEST_P(TransposeConvOpTest, SimpleTestHybridInt8) {
  const std::initializer_list<float> filter_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::initializer_list<int8_t> const_filter_data = {14, 28, 42,  56, 71,
                                                           85, 99, 113, 127};
  HybridTransposeConvOpModel model(
      /*registration=*/GetRegistration(), /*output_shape_data=*/{1, 4, 4, 1},
      /*filter=*/
      {TensorType_INT8, {1, 3, 3, 1}, 0, 0, 0, 0, true, {9.0 / 127}, {0}, 0},
      /*filter_data=*/const_filter_data,
      /*input=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
      /*output=*/{TensorType_FLOAT32, {}},
      /*padding=*/Padding_SAME, /*stride_w=*/1, /*stride_h=*/1,
      /*fused_activation=*/ActivationFunctionType_NONE,
      /*test_type=*/GetTestType(),
      /*version=*/3,
      /*bias_type=*/TensorType_FLOAT32);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }

  model.SetBias({1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  // The values are taken from float model "SimpleTest".
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                     {30, 63, 84, 76, 100, 193, 238, 199, 208,
                                      373, 417.5, 331, 263.7, 447, 486, 366.5},
                                     0.19)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_P(TransposeConvOpTest, SimpleTestHybridInt8MultiChannel) {
  const std::initializer_list<float> filter_data = {
      1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  const std::initializer_list<int8_t> const_filter_data = {
      7,  22, 37, 52, 67, 82, 97, 112, 127,
      14, 28, 42, 56, 71, 85, 99, 113, 127};
  HybridTransposeConvOpModel model(
      /*registration=*/GetRegistration(), /*output_shape_data=*/{1, 5, 5, 2},
      /*filter=*/
      {TensorType_INT8,
       {2, 3, 3, 1},
       0,
       0,
       0,
       0,
       true,
       {17.0 / 127, 18.0 / 127},
       {0, 0},
       0},
      /*filter_data=*/const_filter_data,
      /*input=*/{TensorType_FLOAT32, {1, 2, 2, 1}},
      /*output=*/{TensorType_FLOAT32, {}},
      /*padding=*/Padding_VALID, /*stride_w=*/2, /*stride_h=*/2,
      /*fused_activation=*/ActivationFunctionType_NONE,
      /*test_type=*/GetTestType(),
      /*version=*/3,
      /*bias_type=*/TensorType_FLOAT32);

  model.SetInput({1, 2, 3, 4});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }
  model.SetBias({3, 4});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  // The values are taken from float model "MultiChannelBiasTest".
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {4,    6,  6,    8,    10, 14,  9,   12, 13,   16, 10, 12,   12,
           14,   28, 32,   21,   24, 25,  28,  19, 24,   27, 32, 64.5, 76,
           44.5, 52, 56.5, 63.5, 24, 28,  30,  34, 63.5, 72, 39, 44,   47,
           52,   42, 46,   48,   52, 106, 114, 63, 68,   71, 76},
          0.26)));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

enum class BiasMode {
  kNoBiasInput = 0,
  kNullBiasInput = 1,
  kConstBias = 2,
  kDynamicBias = 3,
};

class Float16DequantizeOpModel : public SingleOpModel {
 public:
  Float16DequantizeOpModel(const std::vector<int>& shape,
                           const std::vector<half>& data) {
    SetBypassDefaultDelegates();
    input_ = AddConstInput(TensorData{TensorType_FLOAT16, shape}, data);
    output_ = AddOutput({TensorType_FLOAT32, shape});
    SetBuiltinOp(BuiltinOperator_DEQUANTIZE, BuiltinOptions_DequantizeOptions,
                 CreateDequantizeOptions(builder_).Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_DEQUANTIZE, ops::builtin::Register_DEQUANTIZE(),
        /*version=*/2);
    BuildInterpreter({GetShape(input_)}, /*num_threads=*/1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/true);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

class Float16WeightsTransposeConvOpModel : public SingleOpModel {
 public:
  Float16WeightsTransposeConvOpModel(
      TfLiteRegistration* registration, const std::vector<int>& output_shape,
      const std::vector<int>& filter_shape,
      const std::vector<half>& filter_data, const std::vector<int>& input_shape,
      Padding padding, int stride_w, int stride_h,
      ActivationFunctionType fused_activation, TestType output_shape_type,
      BiasMode bias_mode = BiasMode::kNoBiasInput,
      const std::vector<float>& bias_data = {}) {
    SetBypassDefaultDelegates();
    if (output_shape_type == TestType::kDynamic) {
      output_shape_ = AddInput({TensorType_INT32, {4}});
    } else {
      output_shape_ =
          AddConstInput(TensorData{TensorType_INT32, {4}}, output_shape);
    }
    filter_ = AddConstInput(TensorData{TensorType_FLOAT16, filter_shape},
                            filter_data);
    input_ = AddInput({TensorType_FLOAT32, input_shape});

    const int out_channels = filter_shape[0];
    if (bias_mode == BiasMode::kNullBiasInput) {
      bias_ = AddNullInput();
    } else if (bias_mode == BiasMode::kConstBias) {
      bias_ = AddConstInput(TensorData{TensorType_FLOAT32, {out_channels}},
                            bias_data);
    } else if (bias_mode == BiasMode::kDynamicBias) {
      bias_ = AddInput({TensorType_FLOAT32, {out_channels}});
    }

    output_ = AddOutput({TensorType_FLOAT32, {}});

    SetBuiltinOp(BuiltinOperator_TRANSPOSE_CONV,
                 BuiltinOptions_TransposeConvOptions,
                 CreateTransposeConvOptions(builder_, padding, stride_w,
                                            stride_h, fused_activation)
                     .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_TRANSPOSE_CONV, registration, /*version=*/6);

    std::vector<std::vector<int>> input_shapes = {
        GetShape(output_shape_), GetShape(filter_), GetShape(input_)};
    if (bias_mode == BiasMode::kConstBias ||
        bias_mode == BiasMode::kDynamicBias) {
      input_shapes.push_back(GetShape(bias_));
    }
    BuildInterpreter(input_shapes, /*num_threads=*/1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/true);

    if (output_shape_type == TestType::kDynamic) {
      PopulateTensor<int32_t>(output_shape_, output_shape);
    }
    if (bias_mode == BiasMode::kDynamicBias) {
      PopulateTensor<float>(bias_, bias_data);
    }
  }

  void SetOutputShape(const std::vector<int32_t>& output_shape) {
    PopulateTensor<int32_t>(output_shape_, output_shape);
  }

  void SetInput(const std::vector<float>& data) {
    PopulateTensor<float>(input_, data);
  }

  void ResizeInputAndAllocate(const std::vector<int>& new_input_shape) {
    ASSERT_EQ(interpreter_->ResizeInputTensor(input_, new_input_shape),
              kTfLiteOk);
    ASSERT_EQ(AllocateTensors(), kTfLiteOk);
  }

  int GetTemporaryCount() const {
    const auto* node_and_reg = interpreter_->node_and_registration(0);
    if (node_and_reg == nullptr || node_and_reg->first.temporaries == nullptr) {
      return 0;
    }
    return node_and_reg->first.temporaries->size;
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int output_shape_ = -1;
  int filter_ = -1;
  int input_ = -1;
  int bias_ = -1;
  int output_ = -1;
};

std::vector<half> ToHalfVector(std::initializer_list<float> values) {
  std::vector<half> result;
  result.reserve(values.size());
  for (float v : values) {
    result.push_back(half(v));
  }
  return result;
}

std::vector<float> RunDequantizeAndFloat32TransposeConv(
    TfLiteRegistration* registration, const std::vector<int>& output_shape,
    const std::vector<int>& filter_shape, const std::vector<half>& filter_data,
    const std::vector<int>& input_shape, const std::vector<float>& input_data,
    Padding padding, int stride_w, int stride_h,
    ActivationFunctionType fused_activation, TestType output_shape_type,
    const std::vector<float>& bias_data = {}) {
  Float16DequantizeOpModel dequant(filter_shape, filter_data);
  EXPECT_EQ(dequant.Invoke(), kTfLiteOk);
  const std::vector<float> dequantized_weights = dequant.GetOutput();

  class Float32RefModel : public SingleOpModel {
   public:
    Float32RefModel(TfLiteRegistration* registration,
                    const std::vector<int>& output_shape,
                    const std::vector<int>& filter_shape,
                    const std::vector<float>& filter_data,
                    const std::vector<int>& input_shape, Padding padding,
                    int stride_w, int stride_h,
                    ActivationFunctionType fused_activation,
                    TestType output_shape_type,
                    const std::vector<float>& bias_data) {
      SetBypassDefaultDelegates();
      if (output_shape_type == TestType::kDynamic) {
        output_shape_ = AddInput({TensorType_INT32, {4}});
      } else {
        output_shape_ =
            AddConstInput(TensorData{TensorType_INT32, {4}}, output_shape);
      }
      filter_ = AddConstInput(TensorData{TensorType_FLOAT32, filter_shape},
                              filter_data);
      input_ = AddInput({TensorType_FLOAT32, input_shape});
      if (!bias_data.empty()) {
        bias_ = AddConstInput(TensorData{TensorType_FLOAT32, {filter_shape[0]}},
                              bias_data);
      }
      output_ = AddOutput({TensorType_FLOAT32, {}});
      SetBuiltinOp(BuiltinOperator_TRANSPOSE_CONV,
                   BuiltinOptions_TransposeConvOptions,
                   CreateTransposeConvOptions(builder_, padding, stride_w,
                                              stride_h, fused_activation)
                       .Union());
      resolver_ = std::make_unique<SingleOpResolver>(
          BuiltinOperator_TRANSPOSE_CONV, registration, /*version=*/6);
      std::vector<std::vector<int>> shapes = {
          GetShape(output_shape_), GetShape(filter_), GetShape(input_)};
      if (!bias_data.empty()) {
        shapes.push_back(GetShape(bias_));
      }
      BuildInterpreter(shapes, /*num_threads=*/1,
                       /*allow_fp32_relax_to_fp16=*/false,
                       /*apply_delegate=*/false,
                       /*allocate_and_delegate=*/true);
      if (output_shape_type == TestType::kDynamic) {
        PopulateTensor<int32_t>(output_shape_, output_shape);
      }
    }
    void SetInput(const std::vector<float>& data) {
      PopulateTensor<float>(input_, data);
    }
    std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

   private:
    int output_shape_ = -1;
    int filter_ = -1;
    int input_ = -1;
    int bias_ = -1;
    int output_ = -1;
  };

  Float32RefModel ref_model(registration, output_shape, filter_shape,
                            dequantized_weights, input_shape, padding, stride_w,
                            stride_h, fused_activation, output_shape_type,
                            bias_data);
  ref_model.SetInput(input_data);
  EXPECT_EQ(ref_model.Invoke(), kTfLiteOk);
  return ref_model.GetOutput();
}

TEST_P(TransposeConvOpTest, SimpleTestFloat16WeightsAndNoTemporaries) {
  const std::vector<half> filter_data =
      ToHalfVector({1, 2, 3, 4, 5, 6, 7, 8, 9});
  const std::vector<float> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                         9, 10, 11, 12, 13, 14, 15, 16};
  for (BiasMode bias_mode :
       {BiasMode::kNoBiasInput, BiasMode::kNullBiasInput}) {
    Float16WeightsTransposeConvOpModel model(
        GetRegistration(), {1, 4, 4, 1}, {1, 3, 3, 1}, filter_data,
        {1, 4, 4, 1}, Padding_SAME, /*stride_w=*/1, /*stride_h=*/1,
        ActivationFunctionType_NONE, GetTestType(), bias_mode);
    EXPECT_EQ(model.GetTemporaryCount(), 0);
    model.SetInput(input_data);
    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    EXPECT_THAT(
        model.GetOutput(),
        Pointwise(FloatingPointEq(), {29, 62, 83, 75, 99, 192, 237, 198, 207,
                                      372, 417, 330, 263, 446, 485, 365}));
    EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  }
}

TEST_P(TransposeConvOpTest, Float16WeightsMultiChannelStridePaddingAndBias) {
  const std::vector<half> filter_data = ToHalfVector(
      {1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18});
  const std::vector<float> input_data = {1, 2, -3, 4};
  const std::vector<float> bias_data = {3, 4};

  for (BiasMode bias_mode : {BiasMode::kConstBias, BiasMode::kDynamicBias}) {
    Float16WeightsTransposeConvOpModel model(
        GetRegistration(), {1, 5, 5, 2}, {2, 3, 3, 1}, filter_data,
        {1, 2, 2, 1}, Padding_VALID, /*stride_w=*/2, /*stride_h=*/2,
        ActivationFunctionType_RELU, GetTestType(), bias_mode, bias_data);
    EXPECT_EQ(model.GetTemporaryCount(), 0);
    model.SetInput(input_data);
    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    EXPECT_THAT(model.GetOutput(),
                Pointwise(FloatingPointEq(),
                          {4,  6,  6,  8,  10, 14, 9,  12, 13, 16, 10, 12, 12,
                           14, 28, 32, 21, 24, 25, 28, 13, 12, 9,  8,  35, 40,
                           45, 52, 57, 64, 0,  0,  0,  0,  0,  0,  39, 44, 47,
                           52, 0,  0,  0,  0,  4,  6,  63, 68, 71, 76}));
    EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
  }
}

TEST_P(TransposeConvOpTest, Float16WeightsLargerKernelAndMultiInputChannels) {
  // 2 output channels, 4x4 kernel, 2 input channels (32 weights).
  std::vector<half> filter_data;
  filter_data.reserve(64);
  for (int i = 1; i <= 64; ++i) {
    filter_data.push_back(half(static_cast<float>((i % 11) - 5) * 0.5f));
  }
  const std::vector<int> output_shape = {1, 6, 6, 2};
  const std::vector<int> filter_shape = {2, 4, 4, 2};
  const std::vector<int> input_shape = {1, 3, 3, 2};
  const std::vector<float> input_data = {
      0.5f,  -1.0f, 1.5f,  -2.0f,  2.5f, -0.5f,  1.0f, 0.25f, -0.75f,
      1.25f, -1.5f, 0.75f, -0.25f, 2.0f, -1.25f, 0.5f, 1.75f, -0.5f};
  const std::vector<float> bias_data = {0.25f, -0.75f};

  for (ActivationFunctionType act :
       {ActivationFunctionType_NONE, ActivationFunctionType_RELU_N1_TO_1,
        ActivationFunctionType_RELU6}) {
    Float16WeightsTransposeConvOpModel model(
        GetRegistration(), output_shape, filter_shape, filter_data, input_shape,
        Padding_SAME, /*stride_w=*/2, /*stride_h=*/2, act, GetTestType(),
        BiasMode::kConstBias, bias_data);
    EXPECT_EQ(model.GetTemporaryCount(), 0);
    model.SetInput(input_data);
    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    std::vector<float> expected = RunDequantizeAndFloat32TransposeConv(
        GetRegistration(), output_shape, filter_shape, filter_data, input_shape,
        input_data, Padding_SAME, /*stride_w=*/2, /*stride_h=*/2, act,
        GetTestType(), bias_data);
    EXPECT_THAT(model.GetOutput(), Pointwise(FloatingPointEq(), expected));
  }
}

TEST_P(TransposeConvOpTest, Float16WeightsRepeatedInvocationsAndDynamicResize) {
  const std::vector<half> filter_data =
      ToHalfVector({1, 2, 3, 4, 5, 6, 7, 8, 9});
  Float16WeightsTransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1}, {1, 3, 3, 1}, filter_data, {1, 4, 4, 1},
      Padding_SAME, /*stride_w=*/1, /*stride_h=*/1, ActivationFunctionType_NONE,
      GetTestType());

  const std::vector<float> input1 = {1, 2,  3,  4,  5,  6,  7,  8,
                                     9, 10, 11, 12, 13, 14, 15, 16};
  const std::vector<float> input2(16, 1.0f);

  model.SetInput(input1);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  const std::vector<float> out1 = model.GetOutput();

  // Second invocation with different input must overwrite (zero) output rather
  // than accumulating onto out1.
  model.SetInput(input2);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              Pointwise(FloatingPointEq(), {12, 21, 21, 16, 27, 45, 45, 33, 27,
                                            45, 45, 33, 24, 39, 39, 28}));

  // Third invocation with input1 again produces identical results to out1.
  model.SetInput(input1);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), Pointwise(FloatingPointEq(), out1));

  if (GetTestType() == TestType::kDynamic) {
    model.ResizeInputAndAllocate({1, 2, 2, 1});
    model.SetOutputShape({1, 2, 2, 1});
    model.SetInput({1, 2, 3, 4});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 2, 1}));
    EXPECT_THAT(model.GetOutput(),
                Pointwise(FloatingPointEq(), {23, 33, 53, 63}));
  }
}

TEST_P(TransposeConvOpTest,
       Float16WeightsSpecialValuesAndFp32AccumulationParity) {
  // Verify exact parity with DEQUANTIZE + FP32 TRANSPOSE_CONV on:
  // 1) -0.0 (0x8000): scatter adds onto +0.0f initialized output_data -> +0.0f.
  // 2) Subnormals: smallest subnormal (0x0001 = 2^-24) and largest subnormal
  //    (0x03ff = 1023 * 2^-24).
  // 3) Max finite FP16 (65504.0f = 0x7bff): 2 * 65504 = 131008.0f (exceeds FP16
  //    max without overflowing to Inf because accumulation & output are FP32).
  // 4) Catastrophic cancellation in FP16: 2048 + 1 - 2048 = 1.0f (in FP16,
  //    2048 + 1 rounds to 2048, so 2048 + 1 - 2048 would be 0 if accumulated
  //    in FP16, whereas FP32 accumulation yields 1.0f).
  const half neg_zero = half::from_bits(0x8000);
  const half min_subnormal = half::from_bits(0x0001);
  const half max_subnormal = half::from_bits(0x03ff);
  const half max_fp16 = half::from_bits(0x7bff);  // 65504.0f

  // Case A: 1x3 filter with overlapping scatter into the middle output element.
  // Input shape [1, 1, 3, 1] with values [1.0f, 1.0f, 1.0f], filter shape
  // [1, 1, 3, 1] with weights [-2048.0f, 1.0f, 2048.0f], stride 1, VALID
  // padding -> output shape [1, 1, 5, 1].
  // At output x=2: in_x=0 contributes filter[2]=2048, in_x=1 contributes
  // filter[1]=1 (2048 + 1 = 2049 in FP32), in_x=2 contributes filter[0]=-2048
  // -> final value at x=2 is 1.0f.
  {
    const std::vector<half> cancel_weights = {half(-2048.0f), half(1.0f),
                                              half(2048.0f)};
    const std::vector<int> output_shape = {1, 1, 5, 1};
    const std::vector<int> filter_shape = {1, 1, 3, 1};
    const std::vector<int> input_shape = {1, 1, 3, 1};
    const std::vector<float> input_data = {1.0f, 1.0f, 1.0f};

    Float16WeightsTransposeConvOpModel model(
        GetRegistration(), output_shape, filter_shape, cancel_weights,
        input_shape, Padding_VALID, /*stride_w=*/1, /*stride_h=*/1,
        ActivationFunctionType_NONE, GetTestType());
    model.SetInput(input_data);
    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    const std::vector<float> expected = RunDequantizeAndFloat32TransposeConv(
        GetRegistration(), output_shape, filter_shape, cancel_weights,
        input_shape, input_data, Padding_VALID, /*stride_w=*/1, /*stride_h=*/1,
        ActivationFunctionType_NONE, GetTestType());
    EXPECT_THAT(model.GetOutput(), Pointwise(FloatingPointEq(), expected));
    EXPECT_EQ(model.GetOutput()[2], 1.0f);
  }

  // Case B: Special FP16 values (-0.0, subnormals, 65504, 2 * 65504 = 131008).
  // Input shape [1, 1, 2, 1] with values [1.0f, 1.0f], filter shape
  // [4, 1, 2, 1] (4 output channels), VALID padding, stride 1 -> output shape
  // [1, 1, 3, 4].
  // Channel 0: [neg_zero, neg_zero] -> +0.0f
  // Channel 1: [min_subnormal, min_subnormal] -> center has 2 * 2^-24 = 2^-23
  // Channel 2: [max_subnormal, min_subnormal] -> center has 1024 * 2^-24 =
  //            2^-14.
  // Channel 3: [max_fp16, max_fp16] -> center has 65504 + 65504 = 131008.0f
  {
    const std::vector<half> special_weights = {
        neg_zero,      neg_zero,       // out_channel 0
        min_subnormal, min_subnormal,  // out_channel 1
        max_subnormal, min_subnormal,  // out_channel 2
        max_fp16,      max_fp16,       // out_channel 3
    };
    const std::vector<int> output_shape = {1, 1, 3, 4};
    const std::vector<int> filter_shape = {4, 1, 2, 1};
    const std::vector<int> input_shape = {1, 1, 2, 1};
    const std::vector<float> input_data = {1.0f, 1.0f};

    Float16WeightsTransposeConvOpModel model(
        GetRegistration(), output_shape, filter_shape, special_weights,
        input_shape, Padding_VALID, /*stride_w=*/1, /*stride_h=*/1,
        ActivationFunctionType_NONE, GetTestType());
    model.SetInput(input_data);
    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    const std::vector<float> actual = model.GetOutput();
    const std::vector<float> expected = RunDequantizeAndFloat32TransposeConv(
        GetRegistration(), output_shape, filter_shape, special_weights,
        input_shape, input_data, Padding_VALID, /*stride_w=*/1, /*stride_h=*/1,
        ActivationFunctionType_NONE, GetTestType());
    EXPECT_THAT(actual, Pointwise(FloatingPointEq(), expected));

    // Center spatial position x=1 (indices 4..7):
    EXPECT_EQ(actual[4], 0.0f);
    EXPECT_FALSE(std::signbit(actual[4]));
    EXPECT_EQ(actual[5], std::ldexp(1.0f, -23));
    EXPECT_EQ(actual[6], std::ldexp(1.0f, -14));
    EXPECT_EQ(actual[7], 131008.0f);
  }
}

TEST(TransposeConvFloat16ValidationTest, RejectsDynamicFloat16Weights) {
  for (TfLiteRegistration* registration :
       {ops::builtin::Register_TRANSPOSECONV_REF(),
        ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT()}) {
    PrepareOnlyTransposeConvOpModel<half> m(
        registration, {1, 4, 4, 1}, {TensorType_FLOAT16, {1, 3, 3, 1}},
        {TensorType_FLOAT32, {1, 4, 4, 1}}, {TensorType_FLOAT32, {}},
        Padding_SAME, /*stride_w=*/1, /*stride_h=*/1,
        ActivationFunctionType_NONE, /*version=*/6);
    EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
  }
}

class SparseFloat16WeightsTransposeConvOpModel : public SingleOpModel {
 public:
  explicit SparseFloat16WeightsTransposeConvOpModel(
      TfLiteRegistration* registration) {
    SetBypassDefaultDelegates();
    output_shape_ =
        AddConstInput(TensorData{TensorType_INT32, {4}}, {1, 2, 2, 1});
    TensorData sparse_filter{TensorType_FLOAT16,
                             {1, 2, 2, 1},
                             0.0f,
                             0.0f,
                             0.0f,
                             0,
                             false,
                             {},
                             {},
                             0,
                             {0, 1, 2, 3},
                             {kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimDense,
                              kTfLiteDimSparseCSR}};
    filter_ =
        AddConstSparseInput<float>(sparse_filter, {1.0f, 0.0f, 2.0f, 3.0f});
    input_ = AddInput({TensorType_FLOAT32, {1, 2, 2, 1}});
    output_ = AddOutput({TensorType_FLOAT32, {}});

    SetBuiltinOp(
        BuiltinOperator_TRANSPOSE_CONV, BuiltinOptions_TransposeConvOptions,
        CreateTransposeConvOptions(builder_, Padding_SAME, /*stride_w=*/1,
                                   /*stride_h=*/1, ActivationFunctionType_NONE)
            .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_TRANSPOSE_CONV, registration, /*version=*/6);
    BuildInterpreter(
        {GetShape(output_shape_), GetShape(filter_), GetShape(input_)},
        /*num_threads=*/1, /*allow_fp32_relax_to_fp16=*/false,
        /*apply_delegate=*/false, /*allocate_and_delegate=*/false);
  }

 private:
  int output_shape_;
  int filter_;
  int input_;
  int output_;
};

TEST(TransposeConvFloat16ValidationTest, RejectsSparseFloat16Weights) {
  for (TfLiteRegistration* registration :
       {ops::builtin::Register_TRANSPOSECONV_REF(),
        ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT()}) {
    SparseFloat16WeightsTransposeConvOpModel m(registration);
    EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
  }
}

class CustomTypesTransposeConvOpModel : public SingleOpModel {
 public:
  CustomTypesTransposeConvOpModel(TfLiteRegistration* registration,
                                  TensorType filter_type, TensorType input_type,
                                  TensorType output_type,
                                  std::optional<TensorType> bias_type) {
    SetBypassDefaultDelegates();
    output_shape_ =
        AddConstInput(TensorData{TensorType_INT32, {4}}, {1, 2, 2, 1});
    if (filter_type == TensorType_FLOAT16) {
      filter_ = AddConstInput(TensorData{TensorType_FLOAT16, {1, 2, 2, 1}},
                              {half(1.0f), half(2.0f), half(3.0f), half(4.0f)});
    } else {
      filter_ = AddConstInput(TensorData{TensorType_FLOAT32, {1, 2, 2, 1}},
                              {1.0f, 2.0f, 3.0f, 4.0f});
    }
    input_ = AddInput({input_type, {1, 2, 2, 1}});
    if (bias_type.has_value()) {
      bias_ = AddInput({*bias_type, {1}});
    }
    output_ = AddOutput({output_type, {}});

    SetBuiltinOp(
        BuiltinOperator_TRANSPOSE_CONV, BuiltinOptions_TransposeConvOptions,
        CreateTransposeConvOptions(builder_, Padding_SAME, /*stride_w=*/1,
                                   /*stride_h=*/1, ActivationFunctionType_NONE)
            .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_TRANSPOSE_CONV, registration, /*version=*/6);
    std::vector<std::vector<int>> shapes = {
        GetShape(output_shape_), GetShape(filter_), GetShape(input_)};
    if (bias_type.has_value()) {
      shapes.push_back(GetShape(bias_));
    }
    BuildInterpreter(shapes, /*num_threads=*/1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/false);
  }

 private:
  int output_shape_ = -1;
  int filter_ = -1;
  int input_ = -1;
  int bias_ = -1;
  int output_ = -1;
};

TEST(TransposeConvFloat16ValidationTest, RejectsMismatchedTypes) {
  for (TfLiteRegistration* registration :
       {ops::builtin::Register_TRANSPOSECONV_REF(),
        ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT()}) {
    // Rejects FP16 input with FP16 weights and FP32 output.
    {
      CustomTypesTransposeConvOpModel m(registration, TensorType_FLOAT16,
                                        TensorType_FLOAT16, TensorType_FLOAT32,
                                        std::nullopt);
      EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
    }
    // Rejects FP16 input with FP16 weights and FP16 output.
    {
      CustomTypesTransposeConvOpModel m(registration, TensorType_FLOAT16,
                                        TensorType_FLOAT16, TensorType_FLOAT16,
                                        std::nullopt);
      EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
    }
    // Rejects FP32 input with FP16 weights and FP16 output.
    {
      CustomTypesTransposeConvOpModel m(registration, TensorType_FLOAT16,
                                        TensorType_FLOAT32, TensorType_FLOAT16,
                                        std::nullopt);
      EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
    }
    // Rejects FP32 input with FP16 weights, FP32 output, and FP16 bias.
    {
      CustomTypesTransposeConvOpModel m(registration, TensorType_FLOAT16,
                                        TensorType_FLOAT32, TensorType_FLOAT32,
                                        TensorType_FLOAT16);
      EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    TransposeConvOpTest, TransposeConvOpTest,
    ::testing::Combine(
        ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)),
        ::testing::Values(TestType::kConst, TestType::kDynamic)));

}  // namespace
}  // namespace tflite
