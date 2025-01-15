/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_LEAKY_RELU_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_LEAKY_RELU_TESTER_H_

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "tflite/core/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class QuantizedLeakyReluTester {
 public:
  QuantizedLeakyReluTester() = default;
  QuantizedLeakyReluTester(const QuantizedLeakyReluTester&) = delete;
  QuantizedLeakyReluTester& operator=(const QuantizedLeakyReluTester&) = delete;

  inline QuantizedLeakyReluTester& Shape(std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    size_ = QuantizedLeakyReluTester::ComputeSize(shape_);
    return *this;
  }

  const std::vector<int32_t>& Shape() const { return shape_; }

  int32_t Size() const { return size_; }

  inline QuantizedLeakyReluTester& InputZeroPoint(int32_t input_zero_point) {
    input_zero_point_ = input_zero_point;
    return *this;
  }

  inline int32_t InputZeroPoint() const { return input_zero_point_; }

  inline QuantizedLeakyReluTester& OutputZeroPoint(int32_t output_zero_point) {
    output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int32_t OutputZeroPoint() const { return output_zero_point_; }

  inline QuantizedLeakyReluTester& InputScale(float input_scale) {
    input_scale_ = input_scale;
    return *this;
  }

  inline float InputScale() const { return input_scale_; }

  inline QuantizedLeakyReluTester& OutputScale(float output_scale) {
    output_scale_ = output_scale;
    return *this;
  }

  inline float OutputScale() const { return output_scale_; }

  inline QuantizedLeakyReluTester& NegativeSlope(float negative_slope) {
    negative_slope_ = negative_slope;
    return *this;
  }

  inline float NegativeSlope() const { return negative_slope_; }

  inline QuantizedLeakyReluTester& Unsigned(bool is_unsigned) {
    unsigned_ = is_unsigned;
    return *this;
  }

  inline bool Unsigned() const { return unsigned_; }

  template <class T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> shape_;
  int32_t size_;
  int32_t input_zero_point_ = 0;
  int32_t output_zero_point_ = 0;
  float input_scale_ = 1.0f;
  float output_scale_ = 1.0f;
  float negative_slope_ = 0.5f;
  bool unsigned_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_LEAKY_RELU_TESTER_H_
