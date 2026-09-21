/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tflite/kernels/fully_connected.h"
#include "tflite/kernels/internal/reference/e8m0_utils.h"
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {

class FullyConnected4BitOpModel : public SingleOpModel {
 public:
  FullyConnected4BitOpModel(
      int units, int batches, const TensorData& input,
      const TensorData& weights, const TensorData& output,
      std::vector<int8_t> weights_initializer, TfLiteRegistration* registration,
      ActivationFunctionType activation_func = ActivationFunctionType_RELU)
      : batches_(batches), units_(units) {
    // Calculate input_size_ from batch and input shape.
    int total_input_size = 1;
    for (size_t i = 0; i < input.shape.size(); ++i) {
      total_input_size *= input.shape[i];
    }
    input_size_ = total_input_size / batches_;
    input_ = AddInput(input);
    const std::vector<int8_t> quantized_data(weights_initializer);
    std::vector<int8_t> weight_data(quantized_data.size() / 2);
    for (int i = 0; i < quantized_data.size(); i++) {
      uint8_t val = quantized_data[i] & UINT8_C(15);
      if ((i % 2) == 0) {
        weight_data[i / 2] = val & INT8_C(15);
      } else {
        weight_data[i / 2] |= (val << 4);
      }
    }
    weights_ =
        AddConstInput<int8_t>(weights, weight_data.data(), weight_data.size());
    bias_ = AddInput({TensorType_FLOAT32, {units_}});
    output_ = AddOutput(output);
    FullyConnectedOptionsWeightsFormat weights_format =
        FullyConnectedOptionsWeightsFormat_DEFAULT;
    SetBuiltinOp(BuiltinOperator_FULLY_CONNECTED,
                 BuiltinOptions_FullyConnectedOptions,
                 CreateFullyConnectedOptions(builder_, activation_func,
                                             weights_format, true)
                     .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_FULLY_CONNECTED, registration);
    BuildInterpreter({GetShape(input_), GetShape(weights_), GetShape(bias_)});
    SetUnitScale();
  }

  void SetUnitScale() {
    TfLiteTensor* t = interpreter_->tensor(weights_);
    t->type = kTfLiteInt4;
    t->params.scale = 1.0;
    auto filter_params =
        reinterpret_cast<TfLiteAffineQuantization*>(t->quantization.params);
    if (filter_params && filter_params->scale &&
        filter_params->scale->size > 0) {
      for (int i = 0; i < filter_params->scale->size; i++) {
        filter_params->scale->data[i] = 1.0;
      }
    }
  }
  void SetInput(const std::vector<float>& f) { PopulateTensor(input_, f); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  void SetBias(const std::vector<float>& f) { PopulateTensor(bias_, f); }
  int input_size() { return input_size_; }
  int num_units() { return units_; }
  int num_batches() { return batches_; }

 protected:
  int input_;
  int weights_;
  int bias_;
  int output_;
  int batches_;
  int units_;
  int input_size_;
  bool use_native_int4_ = false;
};

TEST(Hybrid4BitFullyConnectedOpTest, SimpleTestHybridInt4) {
  int units = 5;
  int batches = 4;
  int cols = 40;
  FullyConnected4BitOpModel m(
      units, batches,
      /*input=*/{TensorType_FLOAT32, {batches, cols}},
      /*weights=*/{TensorType_INT4, {units, cols}, 0.0, 0.0, 1.0},
      /*output=*/{TensorType_FLOAT32, {units, batches}},
      {
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          1,  2, 3, 4, 5, 6, 7, 1, 2, -3, -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          1,  2, 3, 4, 5, 6, 7, 1, 2, -3, -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          1,  2, 3, 4, 5, 6, 7, 1, 2, -3, -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
      },
      ops::builtin::Register_FULLY_CONNECTED_GENERIC_OPT(),
      ActivationFunctionType_RELU);
  m.SetBias({1, 2, 3, 1, 2});
  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
  });
  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {393., 456., 457., 455., 394., 413., 476., 477., 475., 414.,
                   393., 456., 457., 455., 394., 393., 456., 457., 455., 394},
                  /*max_abs_err=*/1.3f)));
}

TEST(Hybrid4BitFullyConnectedOpTest, TestHybridInt4AllZeroBatch) {
  int units = 5;
  int batches = 4;
  int cols = 40;
  FullyConnected4BitOpModel m(
      units, batches,
      /*input=*/{TensorType_FLOAT32, {batches, cols}},
      /*weights=*/{TensorType_INT4, {units, cols}, 0.0, 0.0, 1.0},
      /*output=*/{TensorType_FLOAT32, {units, batches}},
      {
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          1,  2, 3, 4, 5, 6, 7, 1, 2, -3, -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          1,  2, 3, 4, 5, 6, 7, 1, 2, -3, -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          1,  2, 3, 4, 5, 6, 7, 1, 2, -3, -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
          -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,  -1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
      },
      ops::builtin::Register_FULLY_CONNECTED_GENERIC_OPT(),
      ActivationFunctionType_RELU);
  m.SetBias({1, 2, 3, 1, 2});
  m.SetInput({
      // 4x40
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, -9, -10,
      0, 0, 0, 0, 0, 0, 0, 0,  0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0,  0,
      0, 0, 0, 0, 0, 0, 0, 0,  0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0,  0,
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {393., 456., 457., 455., 394., 413., 476., 477., 475., 414.,
                   393., 456., 457., 455., 394., 1,    2,    3,    1,    2},
                  /*max_abs_err=*/1.3f)));
}

std::mt19937 random_engine(2023);
std::uniform_real_distribution<float> real_dist(0.f, 1.f);
std::uniform_int_distribution<int32_t> int_dist(-7, 7);

class Hybrid4BitFullyConnectedVsReferenceOpTests
    : public ::testing::TestWithParam<::testing::tuple<int, int, int>> {};

TEST_P(Hybrid4BitFullyConnectedVsReferenceOpTests, TestHybridInt4) {
  auto params = GetParam();
  int units = std::get<0>(params);
  int batches = std::get<1>(params);
  int cols = std::get<2>(params);
  std::vector<int8_t> weight_data(units * cols, 0);
  std::vector<float> input_data(batches * cols, 0);
  std::vector<float> bias_data(units, 0);
  for (int i = 0; i < units * cols; ++i) {
    weight_data[i] = int_dist(random_engine);
  }
  for (int i = 0; i < batches * cols; ++i) {
    input_data[i] = real_dist(random_engine);
  }
  for (int i = 0; i < units; ++i) {
    bias_data[i] = real_dist(random_engine);
  }
  FullyConnected4BitOpModel test(
      units, batches,
      /*input=*/{TensorType_FLOAT32, {batches, cols}},
      /*weights=*/{TensorType_INT4, {units, cols}, 0.0, 0.0, 1.0},
      /*output=*/{TensorType_FLOAT32, {units, batches}}, weight_data,
      ops::builtin::Register_FULLY_CONNECTED_GENERIC_OPT(),
      ActivationFunctionType_RELU);
  test.SetBias(bias_data);
  test.SetInput(input_data);
  test.Invoke();
  std::vector<float> test_data = test.GetOutput();
  FullyConnected4BitOpModel expected(
      units, batches,
      /*input=*/{TensorType_FLOAT32, {batches, cols}},
      /*weights=*/{TensorType_INT4, {units, cols}, 0.0, 0.0, 1.0},
      /*output=*/{TensorType_FLOAT32, {units, batches}}, weight_data,
      ops::builtin::Register_FULLY_CONNECTED_REF(),
      ActivationFunctionType_RELU);
  expected.SetBias(bias_data);
  expected.SetInput(input_data);
  expected.Invoke();
  std::vector<float> expected_data = expected.GetOutput();
  EXPECT_THAT(test_data, ElementsAreArray(ArrayFloatNear(
                             expected_data, /*max_abs_err=*/1e-3f)));
}

INSTANTIATE_TEST_SUITE_P(Hybrid4BitFullyConnectedVsReferenceOpTests,
                         Hybrid4BitFullyConnectedVsReferenceOpTests,
                         ::testing::ValuesIn({
                             std::make_tuple(4, 1, 32),
                             std::make_tuple(4, 1, 64),
                             std::make_tuple(5, 1, 128),
                             std::make_tuple(5, 4, 128),
                             std::make_tuple(5, 6, 128),
                             std::make_tuple(5, 1, 38),
                             std::make_tuple(5, 4, 72),
                             std::make_tuple(5, 6, 130),
                             std::make_tuple(4, 1, 56),
                             std::make_tuple(4, 1, 48),
                             std::make_tuple(4, 1, 120),
                         }));

TEST(E8M0UtilsTest, BasicShiftAndScaleArithmetic) {
  // Test exponent decoding: biased = 127 is 2^0 = 1.0
  EXPECT_EQ(reference_ops::DecodeE8M0Exponent(127), 0);
  EXPECT_EQ(reference_ops::DecodeE8M0Exponent(128), 1);
  EXPECT_EQ(reference_ops::DecodeE8M0Exponent(126), -1);
  EXPECT_EQ(reference_ops::DecodeE8M0Exponent(254), 127);
  EXPECT_EQ(reference_ops::DecodeE8M0Exponent(0), -127);

  // Test float scale decoding
  EXPECT_FLOAT_EQ(reference_ops::DecodeE8M0Scale(127), 1.0f);
  EXPECT_FLOAT_EQ(reference_ops::DecodeE8M0Scale(128), 2.0f);
  EXPECT_FLOAT_EQ(reference_ops::DecodeE8M0Scale(126), 0.5f);
  EXPECT_FLOAT_EQ(reference_ops::DecodeE8M0Scale(130), 8.0f);
  EXPECT_TRUE(std::isnan(reference_ops::DecodeE8M0Scale(255)));

  // Test scaling floats
  EXPECT_FLOAT_EQ(reference_ops::ScaleByE8M0(3.5f, 127), 3.5f);
  EXPECT_FLOAT_EQ(reference_ops::ScaleByE8M0(3.5f, 128), 7.0f);
  EXPECT_FLOAT_EQ(reference_ops::ScaleByE8M0(3.5f, 126), 1.75f);
  EXPECT_FLOAT_EQ(reference_ops::ScaleByE8M0(1.0f, 131), 16.0f);

  // Test bit-shift integer scaling
  EXPECT_EQ(reference_ops::ShiftByE8M0(10, 127), 10);
  EXPECT_EQ(reference_ops::ShiftByE8M0(10, 128), 20);  // 10 << 1
  EXPECT_EQ(reference_ops::ShiftByE8M0(10, 129), 40);  // 10 << 2
  EXPECT_EQ(reference_ops::ShiftByE8M0(10, 126), 5);   // 10 >> 1
  EXPECT_EQ(reference_ops::ShiftByE8M0(11, 126), 6);   // (11 + 1) >> 1
  EXPECT_EQ(reference_ops::ShiftByE8M0(10, 125), 3);   // (10 + 2) >> 2

  // Test float to E8M0 encoding
  EXPECT_EQ(reference_ops::FloatToE8M0(1.0f), 127);
  EXPECT_EQ(reference_ops::FloatToE8M0(2.0f), 128);
  EXPECT_EQ(reference_ops::FloatToE8M0(0.5f), 126);
  EXPECT_EQ(reference_ops::FloatToE8M0(8.0f), 130);
  EXPECT_EQ(reference_ops::FloatToE8M0(0.25f), 125);
  EXPECT_EQ(reference_ops::FloatToE8M0(std::numeric_limits<float>::quiet_NaN()),
            255);
  EXPECT_EQ(reference_ops::FloatToE8M0(std::numeric_limits<float>::infinity()),
            254);

  // Test float32 packing and unpacking (mantissa ignored)
  EXPECT_FLOAT_EQ(reference_ops::PackE8M0ToFloat32(127), 1.0f);
  EXPECT_FLOAT_EQ(reference_ops::PackE8M0ToFloat32(128), 2.0f);
  EXPECT_FLOAT_EQ(reference_ops::PackE8M0ToFloat32(126), 0.5f);
  EXPECT_EQ(reference_ops::UnpackFloat32ToE8M0(1.0f), 127);
  EXPECT_EQ(reference_ops::UnpackFloat32ToE8M0(2.0f), 128);
  EXPECT_EQ(reference_ops::UnpackFloat32ToE8M0(0.5f), 126);
  // Verify mantissa bits are ignored (1.75 = 1.75 * 2^0 -> exp 127,
  // 3.5 = 1.75 * 2^1 -> exp 128):
  EXPECT_EQ(reference_ops::UnpackFloat32ToE8M0(1.75f), 127);
  EXPECT_EQ(reference_ops::UnpackFloat32ToE8M0(3.5f), 128);
  EXPECT_EQ(reference_ops::DecodePackedFloat32Exponent(1.75f), 0);
  EXPECT_EQ(reference_ops::DecodePackedFloat32Exponent(3.5f), 1);
  EXPECT_FLOAT_EQ(reference_ops::DecodePackedFloat32Scale(1.75f), 1.0f);
  EXPECT_FLOAT_EQ(reference_ops::DecodePackedFloat32Scale(3.5f), 2.0f);
}

class FullyConnectedBlockwise4BitOpModel : public SingleOpModel {
 public:
  FullyConnectedBlockwise4BitOpModel(
      int units, int batches, int cols, int block_size, TensorType scale_type,
      const std::vector<int8_t>& quantized_weights,
      const std::vector<float>& scales, TfLiteRegistration* registration,
      ActivationFunctionType activation_func = ActivationFunctionType_NONE,
      bool asymmetric_quantize_inputs = false)
      : batches_(batches), units_(units), cols_(cols), block_size_(block_size) {
    input_ = AddInput({TensorType_FLOAT32, {batches_, cols_}});

    std::vector<uint8_t> packed_weights(quantized_weights.size() / 2);
    for (size_t i = 0; i < quantized_weights.size(); ++i) {
      uint8_t val = quantized_weights[i] & 0x0F;
      if (i % 2 == 0) {
        packed_weights[i / 2] = val;
      } else {
        packed_weights[i / 2] |= (val << 4);
      }
    }

    TensorData weight_tensor_data(
        TensorType_INT4, {units_, cols_}, /*min=*/0.0, /*max=*/0.0,
        /*scale=*/0.0, /*zero_point=*/0, /*per_channel_quantization=*/false,
        /*per_channel_quantization_scales=*/scales,
        /*per_channel_quantization_offsets=*/{}, /*channel_index=*/0,
        /*traversal_order=*/{}, /*format=*/{}, /*block_size=*/{},
        /*block_map=*/{}, /*shape_signature=*/{},
        /*per_block_quantization=*/block_size_,
        /*per_block_scale_type=*/scale_type);

    weights_ = AddConstInput<uint8_t>(weight_tensor_data, packed_weights.data(),
                                      packed_weights.size());
    bias_ = AddInput({TensorType_FLOAT32, {units_}});
    output_ = AddOutput({TensorType_FLOAT32, {batches_, units_}});

    SetBuiltinOp(
        BuiltinOperator_FULLY_CONNECTED, BuiltinOptions_FullyConnectedOptions,
        CreateFullyConnectedOptions(builder_, activation_func,
                                    FullyConnectedOptionsWeightsFormat_DEFAULT,
                                    /*keep_num_dims=*/true,
                                    asymmetric_quantize_inputs)
            .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_FULLY_CONNECTED, registration);
    BuildInterpreter({GetShape(input_), GetShape(weights_), GetShape(bias_)});
  }

  void SetInput(const std::vector<float>& f) { PopulateTensor(input_, f); }
  void SetBias(const std::vector<float>& f) { PopulateTensor(bias_, f); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int weights_;
  int bias_;
  int output_;
  int batches_;
  int units_;
  int cols_;
  int block_size_;
  TensorType input_type_;
};

TEST(Hybrid4BitFullyConnectedOpTest, TestBlockwiseWithE8M0Scale) {
  const int units = 2;
  const int batches = 1;
  const int cols = 64;
  const int block_size = 32;  // 2 blocks per unit

  std::vector<int8_t> weight_data(units * cols, 1);
  // Unit 0 has scale 1.0 (raw 127) for block 0 and 2.0 (raw 128) for block 1
  // Unit 1 has scale 0.5 (raw 126) for block 0 and 4.0 (raw 129) for block 1
  std::vector<float> scales = {1.0f, 2.0f, 0.5f, 4.0f};

  FullyConnectedBlockwise4BitOpModel model(
      units, batches, cols, block_size, TensorType_FLOAT32, weight_data, scales,
      ops::builtin::Register_FULLY_CONNECTED_REF(),
      ActivationFunctionType_NONE);

  std::vector<float> input_data(cols, 1.0f);
  model.SetInput(input_data);
  model.SetBias({0.0f, 0.0f});
  model.Invoke();

  std::vector<float> output = model.GetOutput();
  ASSERT_EQ(output.size(), 2);
  // Unit 0: block 0 is 32 * 1.0 * 1.0 = 32.0,
  //         block 1 is 32 * 1.0 * 2.0 = 64.0 => sum ~ 96.0
  // Unit 1: block 0 is 32 * 1.0 * 0.5 = 16.0,
  //         block 1 is 32 * 1.0 * 4.0 = 128.0 => sum ~ 144.0
  EXPECT_NEAR(output[0], 96.0f, 2.0f);
  EXPECT_NEAR(output[1], 144.0f, 2.0f);
}

TEST(Hybrid4BitFullyConnectedOpTest,
     TestBlockwiseWithPackedE8M0Float32IgnoresMantissa) {
  // Verifies that when scales are passed as packed float32, unpacking as E8M0
  // ignores the mantissa bits.
  const int units = 2;
  const int batches = 1;
  const int cols = 64;
  const int block_size = 32;

  std::vector<int8_t> weight_data(units * cols, 1);
  // Unit 0: block 0 has scale 1.75f (raw exp 127 = 2^0, mantissa 0.75)
  //         block 1 has scale 3.50f (raw exp 128 = 2^1, mantissa 0.75)
  // Unit 1: block 0 has scale 0.875f (raw exp 126 = 2^-1, mantissa 0.75)
  //         block 1 has scale 7.00f (raw exp 129 = 2^2, mantissa 0.75)
  std::vector<float> scales = {1.75f, 3.5f, 0.875f, 7.0f};

  FullyConnectedBlockwise4BitOpModel model(
      units, batches, cols, block_size, TensorType_FLOAT32, weight_data, scales,
      ops::builtin::Register_FULLY_CONNECTED_REF(), ActivationFunctionType_NONE,
      /*asymmetric_quantize_inputs=*/false);

  std::vector<float> input_data(cols, 3.5f);
  model.SetInput(input_data);
  model.SetBias({0.0f, 0.0f});
  model.Invoke();

  std::vector<float> output = model.GetOutput();
  ASSERT_EQ(output.size(), 2);
  // Unpacking ignores mantissa: decodes to powers-of-two {1.0f, 2.0f,
  // 0.5f, 4.0f}, producing identical results to
  // TestBlockwiseDynamicA4W4WithE8M0Scale:
  EXPECT_NEAR(output[0], 336.0f, 1e-3f);
  EXPECT_NEAR(output[1], 504.0f, 1e-3f);
}

}  // namespace tflite
