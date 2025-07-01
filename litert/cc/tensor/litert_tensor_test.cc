// Copyright 2025 Google LLC.
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

#include "litert/cc/tensor/litert_tensor.h"

#include <gtest/gtest.h>
#include <vector>

namespace litert {
namespace tensor {
namespace {

class LiteRtTensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup common test fixtures
  }
};

TEST_F(LiteRtTensorTest, CreateZerosTensor) {
  auto tensor_result = zeros<float>({2, 3});
  ASSERT_TRUE(tensor_result.HasValue()) << "Failed to create zeros tensor";
  
  auto tensor = std::move(*tensor_result);
  EXPECT_EQ(tensor.shape().size(), 2);
  EXPECT_EQ(tensor.shape()[0], 2);
  EXPECT_EQ(tensor.shape()[1], 3);
  EXPECT_EQ(tensor.size(), 6);
  
  // All elements should be zero
  EXPECT_FLOAT_EQ(tensor.sum(), 0.0f);
}

TEST_F(LiteRtTensorTest, CreateOnesTensor) {
  auto tensor_result = ones<float>({3, 2});
  ASSERT_TRUE(tensor_result.HasValue()) << "Failed to create ones tensor";
  
  auto tensor = std::move(*tensor_result);
  EXPECT_EQ(tensor.size(), 6);
  
  // All elements should be one, so sum should equal size
  EXPECT_FLOAT_EQ(tensor.sum(), 6.0f);
  EXPECT_FLOAT_EQ(tensor.mean(), 1.0f);
}

TEST_F(LiteRtTensorTest, FillTensor) {
  auto tensor_result = zeros<float>({2, 2});
  ASSERT_TRUE(tensor_result.HasValue());
  
  auto tensor = std::move(*tensor_result);
  auto fill_result = tensor.fill(5.0f);
  ASSERT_TRUE(fill_result.HasValue()) << "Failed to fill tensor";
  
  EXPECT_FLOAT_EQ(tensor.sum(), 20.0f);  // 4 elements * 5.0
  EXPECT_FLOAT_EQ(tensor.mean(), 5.0f);
}

TEST_F(LiteRtTensorTest, ElementAccess) {
  auto tensor_result = zeros<float>({2, 3});
  ASSERT_TRUE(tensor_result.HasValue());
  
  auto tensor = std::move(*tensor_result);
  
  // Set individual elements
  tensor(0, 0) = 1.0f;
  tensor(0, 1) = 2.0f;
  tensor(1, 2) = 3.0f;
  
  // Check values
  EXPECT_FLOAT_EQ(tensor(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(tensor(0, 1), 2.0f);
  EXPECT_FLOAT_EQ(tensor(1, 2), 3.0f);
  EXPECT_FLOAT_EQ(tensor(1, 0), 0.0f);  // Should still be zero
}

TEST_F(LiteRtTensorTest, ArithmeticOperations) {
  auto a_result = full<float>({2, 2}, 2.0f);
  auto b_result = full<float>({2, 2}, 3.0f);
  ASSERT_TRUE(a_result.HasValue() && b_result.HasValue());
  
  auto a = std::move(*a_result);
  auto b = std::move(*b_result);
  
  // Test all three API styles
  auto c1 = a + b;         // Operator overloading
  auto c2 = a.add(b);      // Fluent style
  auto c3 = add(a, b);     // Functional style
  
  // All should give same result (2 + 3 = 5, sum = 20)
  EXPECT_FLOAT_EQ(c1.sum(), 20.0f);
  EXPECT_FLOAT_EQ(c2.sum(), 20.0f);
  EXPECT_FLOAT_EQ(c3.sum(), 20.0f);
  
  // Test scalar operations
  auto d = a + 1.0f;
  EXPECT_FLOAT_EQ(d.sum(), 12.0f);  // (2 + 1) * 4 = 12
  
  auto e = a * 2.0f;
  EXPECT_FLOAT_EQ(e.sum(), 16.0f);  // 2 * 2 * 4 = 16
}

TEST_F(LiteRtTensorTest, UniversalFunctions) {
  auto tensor_result = full<float>({2, 2}, 0.0f);
  ASSERT_TRUE(tensor_result.HasValue());
  
  auto tensor = std::move(*tensor_result);
  
  auto sin_tensor = tensor.sin();
  auto cos_tensor = tensor.cos();
  
  // sin(0) = 0, cos(0) = 1
  EXPECT_FLOAT_EQ(sin_tensor.sum(), 0.0f);
  EXPECT_FLOAT_EQ(cos_tensor.sum(), 4.0f);  // 4 elements * 1.0
}

TEST_F(LiteRtTensorTest, ReductionOperations) {
  auto tensor_result = zeros<float>({2, 3});
  ASSERT_TRUE(tensor_result.HasValue());
  
  auto tensor = std::move(*tensor_result);
  
  // Fill with known data [1, 2, 3, 4, 5, 6]
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto fill_result = tensor.from_vector(data);
  ASSERT_TRUE(fill_result.HasValue());
  
  EXPECT_FLOAT_EQ(tensor.sum(), 21.0f);    // 1+2+3+4+5+6
  EXPECT_FLOAT_EQ(tensor.mean(), 3.5f);    // 21/6
  EXPECT_FLOAT_EQ(tensor.max(), 6.0f);
  EXPECT_FLOAT_EQ(tensor.min(), 1.0f);
  EXPECT_EQ(tensor.argmax(), 5);           // Index of max element
  EXPECT_EQ(tensor.argmin(), 0);           // Index of min element
}

TEST_F(LiteRtTensorTest, ShapeManipulation) {
  auto tensor_result = full<float>({2, 3}, 1.0f);
  ASSERT_TRUE(tensor_result.HasValue());
  
  auto tensor = std::move(*tensor_result);
  EXPECT_EQ(tensor.size(), 6);
  
  // Reshape to different configuration
  auto reshaped = tensor.reshape({3, 2});
  EXPECT_EQ(reshaped.shape()[0], 3);
  EXPECT_EQ(reshaped.shape()[1], 2);
  EXPECT_EQ(reshaped.size(), 6);
  EXPECT_FLOAT_EQ(reshaped.sum(), 6.0f);  // Data should be preserved
  
  // Reshape to 1D
  auto flattened = tensor.reshape({6});
  EXPECT_EQ(flattened.shape().size(), 1);
  EXPECT_EQ(flattened.shape()[0], 6);
  
  // Test expand_dims
  auto expanded = tensor.expand_dims(0);
  EXPECT_EQ(expanded.shape().size(), 3);
  EXPECT_EQ(expanded.shape()[0], 1);
  EXPECT_EQ(expanded.shape()[1], 2);
  EXPECT_EQ(expanded.shape()[2], 3);
  
  // Test squeeze
  auto squeezed = expanded.squeeze();
  EXPECT_EQ(squeezed.shape().size(), 2);
  EXPECT_EQ(squeezed.shape()[0], 2);
  EXPECT_EQ(squeezed.shape()[1], 3);
}

TEST_F(LiteRtTensorTest, Transpose2D) {
  auto tensor_result = zeros<float>({2, 3});
  ASSERT_TRUE(tensor_result.HasValue());
  
  auto tensor = std::move(*tensor_result);
  
  // Set some distinguishable values
  tensor(0, 0) = 1.0f;
  tensor(0, 1) = 2.0f;
  tensor(1, 0) = 3.0f;
  tensor(1, 1) = 4.0f;
  
  auto transposed = tensor.transpose();
  EXPECT_EQ(transposed.shape()[0], 3);
  EXPECT_EQ(transposed.shape()[1], 2);
  
  // Check that values are correctly transposed
  EXPECT_FLOAT_EQ(transposed(0, 0), 1.0f);  // was (0,0)
  EXPECT_FLOAT_EQ(transposed(1, 0), 2.0f);  // was (0,1)
  EXPECT_FLOAT_EQ(transposed(0, 1), 3.0f);  // was (1,0)
  EXPECT_FLOAT_EQ(transposed(1, 1), 4.0f);  // was (1,1)
}

TEST_F(LiteRtTensorTest, ChainedOperations) {
  auto tensor_result = full<float>({2, 2}, 4.0f);
  ASSERT_TRUE(tensor_result.HasValue());
  
  auto tensor = std::move(*tensor_result);
  
  // Chain multiple operations
  auto result = tensor
      .add(1.0f)           // 4 + 1 = 5
      .mul(2.0f)           // 5 * 2 = 10
      .sqrt()              // sqrt(10) ≈ 3.16
      .sub(1.0f);          // 3.16 - 1 ≈ 2.16
  
  // Approximate check since we're using sqrt
  EXPECT_NEAR(result.sum(), 4 * (std::sqrt(10.0f) - 1.0f), 1e-5);
}

TEST_F(LiteRtTensorTest, ErrorHandling) {
  auto tensor_result = zeros<float>({2, 3});
  ASSERT_TRUE(tensor_result.HasValue());
  
  auto tensor = std::move(*tensor_result);
  
  // Test out of bounds access
  EXPECT_THROW(tensor(2, 0), std::out_of_range);
  EXPECT_THROW(tensor(0, 3), std::out_of_range);
  EXPECT_THROW(tensor({1, 2, 3}), std::invalid_argument);  // Wrong number of indices
  
  // Test invalid reshape
  EXPECT_THROW(tensor.reshape({2, 4}), std::invalid_argument);  // Size mismatch
}

}  // namespace
}  // namespace tensor
}  // namespace litert