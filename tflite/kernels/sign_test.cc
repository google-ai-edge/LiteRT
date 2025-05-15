// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace {

template <typename T>
tflite::TensorType GetTypeEnum();

template <>
tflite::TensorType GetTypeEnum<float>() {
  return tflite::TensorType_FLOAT32;
}

template <>
tflite::TensorType GetTypeEnum<double>() {
  return tflite::TensorType_FLOAT64;
}

template <>
tflite::TensorType GetTypeEnum<int32_t>() {
  return tflite::TensorType_INT32;
}

class SignModel : public tflite::SingleOpModel {
 public:
  SignModel(tflite::TensorData x,
            tflite::TensorData output) {
    x_ = AddInput(x);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SIGN, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(x_)});
  }

  int x_;
  int output_;

  template <typename T>
  std::vector<T> GetOutput(const std::vector<T>& x) {
    PopulateTensor<T>(x_, x);
    Invoke();
    return ExtractVector<T>(output_);
  }
};

template <typename Float>
class SignTestFloat : public ::testing::Test {
 public:
  using FloatType = Float;
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(SignTestFloat, TestTypes);

TYPED_TEST(SignTestFloat, TestScalarFloat) {
  using Float = typename TestFixture::FloatType;
  tflite::TensorData x = {GetTypeEnum<Float>(), {}};
  tflite::TensorData output = {GetTypeEnum<Float>(), {}};
  SignModel m(x, output);
  auto got = m.GetOutput<Float>({0.0});
  ASSERT_EQ(got.size(), 1);
  EXPECT_FLOAT_EQ(got[0], 0.0);

  ASSERT_FLOAT_EQ(m.GetOutput<Float>({5.0})[0], 1.0);
  ASSERT_FLOAT_EQ(m.GetOutput<Float>({-3.0})[0], -1.0);
}

TYPED_TEST(SignTestFloat, TestBatchFloat) {
  using Float = typename TestFixture::FloatType;
  tflite::TensorData x = {GetTypeEnum<Float>(), {4, 2, 1}};
  tflite::TensorData output = {GetTypeEnum<Float>(), {4, 2, 1}};
  SignModel m(x, output);

  std::vector<Float> x_data = {0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, 0.0};

  auto got = m.GetOutput<Float>(x_data);

  EXPECT_EQ(got, std::vector<Float>(
      {1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.0}));
}

template <typename Int>
class SignTestInt : public ::testing::Test {
 public:
  using IntType = Int;
};
using TestTypesInt = ::testing::Types<int32_t>;

TYPED_TEST_SUITE(SignTestInt, TestTypesInt);

TYPED_TEST(SignTestInt, TestScalarInt) {
  using Int = typename TestFixture::IntType;
  tflite::TensorData x = {GetTypeEnum<Int>(), {}};
  tflite::TensorData output = {GetTypeEnum<Int>(), {}};
  SignModel m(x, output);
  auto got = m.GetOutput<Int>({0});
  ASSERT_EQ(got.size(), 1);
  EXPECT_EQ(got[0], 0);

  ASSERT_EQ(m.GetOutput<Int>({5})[0], 1);
  ASSERT_EQ(m.GetOutput<Int>({-3})[0], -1);
}

TYPED_TEST(SignTestInt, TestBatchInt) {
  using Int = typename TestFixture::IntType;
  tflite::TensorData x = {GetTypeEnum<Int>(), {4, 2, 1}};
  tflite::TensorData output = {GetTypeEnum<Int>(), {4, 2, 1}};
  SignModel m(x, output);

  EXPECT_EQ(m.GetOutput<Int>({0, -7, 6, -5, 4, -3, 2, 1}),
            std::vector<Int>({0, -1, 1, -1, 1, -1, 1, 1}));
}

}  // namespace
}  // namespace tflite
