/* Copyright 2026 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TESTING_NUMERICAL_TEST_SUITE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TESTING_NUMERICAL_TEST_SUITE_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/testing/numerical_test_bridge.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

#if !defined(ASSERT_OK)
#define ASSERT_OK(status) ASSERT_TRUE(status.ok())
#endif  // defined(ASSERT_OK)

namespace litert::tensor {

struct XnnpackMixinTag;

using ::testing::FloatNear;
using ::testing::Pointwise;

template <typename BackendTraits>
class NumericalTestSuite : public ::testing::Test {
 protected:
  void SetUp() override {
    bridge_ = BackendTraits::CreateBridge();
    auto status = bridge_->Initialize();
    if (!status.ok()) {
      GTEST_SKIP()
          << "Skipping test suite because backend initialization failed: "
          << status;
    }
  }

  std::unique_ptr<TestBackendBridge> bridge_;
};

TYPED_TEST_SUITE_P(NumericalTestSuite);

TYPED_TEST_P(NumericalTestSuite, AddOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape1;
    std::vector<float> data1;
    std::vector<int> shape2;
    std::vector<float> data2;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1, 2, 2, 1},
       .data2 = {5.0f, 6.0f, 7.0f, 8.0f},
       .expected = {6.0f, 8.0f, 10.0f, 12.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1, 1, 2, 2},
       .data2 = {5.0f, 6.0f, 7.0f, 8.0f},
       .expected = {6.0f, 8.0f, 10.0f, 12.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1},
       .data2 = {5.0f},
       .expected = {6.0f, 7.0f, 8.0f, 9.0f}},
      {.shape1 = {1, 2, 2},
       .data1 = {5.0f, 6.0f, 7.0f, 8.0f},
       .shape2 = {1},
       .data2 = {1.0f},
       .expected = {6.0f, 7.0f, 8.0f, 9.0f}},
      {.shape1 = {2, 2},
       .data1 = {5.0f, 6.0f, 7.0f, 8.0f},
       .shape2 = {1},
       .data2 = {1.0f},
       .expected = {6.0f, 7.0f, 8.0f, 9.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape1});
    Tensor<Tag> input2({.type = Type::kFP32, .shape = tc.shape2});
    Tensor<Tag> output = Add(input1, input2);

    auto status = bridge->BuildGraph({input1, input2}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Add op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data1)));
    ASSERT_OK(bridge->SetInput(input2, AsBytes(tc.data2)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, SubOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape1;
    std::vector<float> data1;
    std::vector<int> shape2;
    std::vector<float> data2;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape1 = {1, 2, 2, 1},
       .data1 = {8.0f, 7.0f, 6.0f, 5.0f},
       .shape2 = {1, 2, 2, 1},
       .data2 = {1.0f, 2.0f, 3.0f, 4.0f},
       .expected = {7.0f, 5.0f, 3.0f, 1.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {8.0f, 7.0f, 6.0f, 5.0f},
       .shape2 = {1, 1, 2, 2},
       .data2 = {1.0f, 2.0f, 3.0f, 4.0f},
       .expected = {7.0f, 5.0f, 3.0f, 1.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1},
       .data2 = {1.0f},
       .expected = {0.0f, 1.0f, 2.0f, 3.0f}},
      {.shape1 = {1, 2, 2},
       .data1 = {5.0f, 6.0f, 7.0f, 8.0f},
       .shape2 = {1},
       .data2 = {1.0f},
       .expected = {4.0f, 5.0f, 6.0f, 7.0f}},
      {.shape1 = {2, 2},
       .data1 = {5.0f, 6.0f, 7.0f, 8.0f},
       .shape2 = {1},
       .data2 = {1.0f},
       .expected = {4.0f, 5.0f, 6.0f, 7.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape1});
    Tensor<Tag> input2({.type = Type::kFP32, .shape = tc.shape2});
    Tensor<Tag> output = Sub(input1, input2);

    auto status = bridge->BuildGraph({input1, input2}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Sub op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data1)));
    ASSERT_OK(bridge->SetInput(input2, AsBytes(tc.data2)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, MulOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape1;
    std::vector<float> data1;
    std::vector<int> shape2;
    std::vector<float> data2;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1, 2, 2, 1},
       .data2 = {5.0f, 6.0f, 7.0f, 8.0f},
       .expected = {5.0f, 12.0f, 21.0f, 32.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1, 1, 2, 2},
       .data2 = {5.0f, 6.0f, 7.0f, 8.0f},
       .expected = {5.0f, 12.0f, 21.0f, 32.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1},
       .data2 = {2.0f},
       .expected = {2.0f, 4.0f, 6.0f, 8.0f}},
      {.shape1 = {1, 2, 2},
       .data1 = {5.0f, 6.0f, 7.0f, 8.0f},
       .shape2 = {1},
       .data2 = {2.0f},
       .expected = {10.0f, 12.0f, 14.0f, 16.0f}},
      {.shape1 = {2, 2},
       .data1 = {5.0f, 6.0f, 7.0f, 8.0f},
       .shape2 = {1},
       .data2 = {2.0f},
       .expected = {10.0f, 12.0f, 14.0f, 16.0f}},
      {.shape1 = {1, 1, 8},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       .shape2 = {},
       .data2 = {0.00048f},
       .expected = {0.00048f, 0.00096f, 0.00144f, 0.00192f, 0.00240f, 0.00288f,
                    0.00336f, 0.00384f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape1});
    Tensor<Tag> input2({.type = Type::kFP32, .shape = tc.shape2});
    Tensor<Tag> output = Mul(input1, input2);

    auto status = bridge->BuildGraph({input1, input2}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Mul op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data1)));
    ASSERT_OK(bridge->SetInput(input2, AsBytes(tc.data2)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, DivOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape1;
    std::vector<float> data1;
    std::vector<int> shape2;
    std::vector<float> data2;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 2, 2, 1},
       .data2 = {2.0f, 2.0f, 3.0f, 4.0f},
       .expected = {0.5f, 2.5f, 1.0f, 2.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 1, 2, 2},
       .data2 = {2.0f, 4.0f, 6.0f, 4.0f},
       .expected = {0.5f, 1.25f, 0.5f, 2.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1},
       .data2 = {2.0f},
       .expected = {0.5f, 1.0f, 1.5f, 2.0f}},
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 1, 2, 1},
       .data2 = {2.0f, 4.0f},
       .expected = {0.5f, 1.25f, 1.5f, 2.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape1});
    Tensor<Tag> input2({.type = Type::kFP32, .shape = tc.shape2});
    Tensor<Tag> output = Div(input1, input2);

    auto status = bridge->BuildGraph({input1, input2}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Div op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data1)));
    ASSERT_OK(bridge->SetInput(input2, AsBytes(tc.data2)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, GeluOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {0.841345f, -0.158655f, 1.9545f, -0.0455003f}},
      {.shape = {1, 4},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {0.841345f, -0.158655f, 1.9545f, -0.0455003f}},
      {.shape = {4},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {0.841345f, -0.158655f, 1.9545f, -0.0455003f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Gelu(input, false);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Gelu op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, ReluOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {1.0f, 0.0f, 2.0f, 0.0f}},
      {.shape = {1, 4},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {1.0f, 0.0f, 2.0f, 0.0f}},
      {.shape = {4},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {1.0f, 0.0f, 2.0f, 0.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Relu(input);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Relu op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, Relu6Op) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, -1.0f, 7.0f, -2.0f},
       .expected = {1.0f, 0.0f, 6.0f, 0.0f}},
      {.shape = {1, 4},
       .data = {1.0f, -1.0f, 7.0f, -2.0f},
       .expected = {1.0f, 0.0f, 6.0f, 0.0f}},
      {.shape = {4},
       .data = {1.0f, -1.0f, 7.0f, -2.0f},
       .expected = {1.0f, 0.0f, 6.0f, 0.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Relu6(input);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Relu6 op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, LeakyReluOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {1.0f, -0.2f, 2.0f, -0.4f}},
      {.shape = {1, 4},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {1.0f, -0.2f, 2.0f, -0.4f}},
      {.shape = {4},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {1.0f, -0.2f, 2.0f, -0.4f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = LeakyRelu(input, 0.2f);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "LeakyRelu op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, EluOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {1.0f, std::expm1(-1.0f), 2.0f, std::expm1(-2.0f)}},
      {.shape = {1, 4},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {1.0f, std::expm1(-1.0f), 2.0f, std::expm1(-2.0f)}},
      {.shape = {4},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {1.0f, std::expm1(-1.0f), 2.0f, std::expm1(-2.0f)}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Elu(input);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Elu op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, HardSwishOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {-4.0f, -2.0f, 1.0f, 4.0f},
       .expected = {0.0f, -0.33333334f, 0.6666667f, 4.0f}},
      {.shape = {1, 4},
       .data = {-4.0f, -2.0f, 1.0f, 4.0f},
       .expected = {0.0f, -0.33333334f, 0.6666667f, 4.0f}},
      {.shape = {4},
       .data = {-4.0f, -2.0f, 1.0f, 4.0f},
       .expected = {0.0f, -0.33333334f, 0.6666667f, 4.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = HardSwish(input);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "HardSwish op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, LogisticOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = Logistic(input);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Logistic op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {0.0f, 2.0f, -2.0f, 0.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {0.5f, 0.880797f, 0.1192029f, 0.5f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, TanhOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {0.761594f, -0.761594f, 0.964028f, -0.964028f}},
      {.shape = {1, 4},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {0.761594f, -0.761594f, 0.964028f, -0.964028f}},
      {.shape = {4},
       .data = {1.0f, -1.0f, 2.0f, -2.0f},
       .expected = {0.761594f, -0.761594f, 0.964028f, -0.964028f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Tanh(input);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Tanh op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, SqrtOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = Sqrt(input);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Sqrt op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 4.0f, 9.0f, 16.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, RsqrtOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 4.0f, 9.0f, 16.0f},
       .expected = {1.0f, 0.5f, 1.0f / 3.0f, 0.25f}},
      {.shape = {1, 4},
       .data = {1.0f, 4.0f, 9.0f, 16.0f},
       .expected = {1.0f, 0.5f, 1.0f / 3.0f, 0.25f}},
      {.shape = {4},
       .data = {1.0f, 4.0f, 9.0f, 16.0f},
       .expected = {1.0f, 0.5f, 1.0f / 3.0f, 0.25f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Rsqrt(input);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Rsqrt op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, SoftmaxOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 1, 1, 4},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .expected = {0.0320586f, 0.0871443f, 0.236883f, 0.643914f}},
      {.shape = {1, 4},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .expected = {0.0320586f, 0.0871443f, 0.236883f, 0.643914f}},
      {.shape = {4},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .expected = {0.0320586f, 0.0871443f, 0.236883f, 0.643914f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Softmax(input);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Softmax op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, TransposeOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int> perm;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .perm = {0, 2, 1, 3},
       .expected = {1.0f, 3.0f, 2.0f, 4.0f}},
      {.shape = {2, 3},
       .data = {1, 2, 3, 4, 5, 6},
       .perm = {1, 0},
       .expected = {1, 4, 2, 5, 3, 6}},
      {.shape = {1, 2, 3},
       .data = {1, 2, 3, 4, 5, 6},
       .perm = {0, 2, 1},
       .expected = {1, 4, 2, 5, 3, 6}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> perm({.type = Type::kI32,
                      .shape = {static_cast<int>(tc.perm.size())},
                      .buffer = OwningCpuBuffer::Copy<Type::kI32>(tc.perm)});
    Tensor<Tag> output = Transpose(input, perm);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Transpose op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, ReshapeOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int> new_shape;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .new_shape = {1, 4},
       .expected = {1.0f, 2.0f, 3.0f, 4.0f}},
      {.shape = {4},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .new_shape = {1, 2, 2, 1},
       .expected = {1.0f, 2.0f, 3.0f, 4.0f}},
      {.shape = {2, 2},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .new_shape = {4},
       .expected = {1.0f, 2.0f, 3.0f, 4.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Reshape(input, tc.new_shape);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Reshape op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, ExpandDimsOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = ExpandDims(input, 1);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "ExpandDims op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, SqueezeOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 1, 2}});
  Tensor<Tag> output = Squeeze(input);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Squeeze op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, SliceOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = Slice(input, {0, 0, 1, 0}, {1, 2, 1, 1});

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Slice op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(2);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {2.0f, 4.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, ConcatenationOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape1;
    std::vector<float> data1;
    std::vector<int> shape2;
    std::vector<float> data2;
    int axis;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1, 2, 2, 1},
       .data2 = {5.0f, 6.0f, 7.0f, 8.0f},
       .axis = 3,
       .expected = {1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f}},
      {.shape1 = {1, 2},
       .data1 = {1, 2},
       .shape2 = {1, 2},
       .data2 = {3, 4},
       .axis = 0,
       .expected = {1, 2, 3, 4}},
      {.shape1 = {2, 1},
       .data1 = {1, 2},
       .shape2 = {2, 1},
       .data2 = {3, 4},
       .axis = 1,
       .expected = {1, 3, 2, 4}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape1});
    Tensor<Tag> input2({.type = Type::kFP32, .shape = tc.shape2});
    Tensor<Tag> output = Concatenation({input1, input2}, tc.axis);

    auto status = bridge->BuildGraph({input1, input2}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Concatenation op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data1)));
    ASSERT_OK(bridge->SetInput(input2, AsBytes(tc.data2)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, FullyConnectedOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> input_shape;
    std::vector<float> input_data;
    std::vector<int> weights_shape;
    std::vector<float> weights_data;
    std::vector<int> bias_shape;
    std::vector<float> bias_data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.input_shape = {1, 4},
       .input_data = {1.0f, 2.0f, 3.0f, 4.0f},
       .weights_shape = {4, 4},
       .weights_data = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
       .bias_shape = {4},
       .bias_data = {0.1f, 0.2f, 0.3f, 0.4f},
       .expected = {1.1f, 2.2f, 3.3f, 4.4f}},
      {.input_shape = {2, 4},
       .input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       .weights_shape = {4, 4},
       .weights_data = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
       .bias_shape = {4},
       .bias_data = {0.1f, 0.2f, 0.3f, 0.4f},
       .expected = {1.1f, 2.2f, 3.3f, 4.4f, 5.1f, 6.2f, 7.3f, 8.4f}},
      {.input_shape = {1, 4},
       .input_data = {1.0f, 2.0f, 3.0f, 4.0f},
       .weights_shape = {2, 4},
       .weights_data = {1, 0, 0, 0, 0, 1, 0, 0},
       .bias_shape = {2},
       .bias_data = {0.1f, 0.2f},
       .expected = {1.1f, 2.2f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.input_shape});
    Tensor<Tag> weights({.type = Type::kFP32,
                         .shape = tc.weights_shape,
                         .buffer = tc.weights_data});
    Tensor<Tag> bias(
        {.type = Type::kFP32, .shape = tc.bias_shape, .buffer = tc.bias_data});
    Tensor<Tag> output = FullyConnected(input, weights, bias);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "FullyConnected op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.input_data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, FullyConnectedQuantizedOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    Type weights_type;
    std::vector<int> input_shape;
    std::vector<float> input_data;
    std::vector<int> weights_shape;
    std::vector<int8_t> weights_data;
    std::shared_ptr<PerChannelAffineQuantization> quantization;
    std::vector<int> bias_shape;
    std::vector<float> bias_data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.weights_type = Type::kI8,
       .input_shape = {1, 4},
       .input_data = {1.0f, 2.0f, 3.0f, 4.0f},
       .weights_shape = {4, 4},
       .weights_data = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           PerChannelAffineQuantization{/*scales=*/{1.0f, 1.0f, 1.0f, 1.0f},
                                        /*zero_points=*/{0, 0, 0, 0}}),
       .bias_shape = {4},
       .bias_data = {0.1f, 0.2f, 0.3f, 0.4f},
       .expected = {1.1f, 2.2f, 3.3f, 4.4f}},
      {.weights_type = Type::kI8,
       .input_shape = {1, 4},
       .input_data = {1.0f, 2.0f, 1.0f, 2.0f},
       .weights_shape = {2, 4},
       .weights_data = {1, 0, 0, 0, 0, 1, 0, 0},
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           PerChannelAffineQuantization{/*scales=*/{0.1f, 0.01f},
                                        /*zero_points=*/{0, 0}}),
       .bias_shape = {2},
       .bias_data = {0.1f, 0.2f},
       .expected = {0.2f, 0.22f}},
      {.weights_type = Type::kI4,
       .input_shape = {1, 4},
       .input_data = {1.0f, 2.0f, 1.0f, 2.0f},
       .weights_shape = {2, 4},
       .weights_data = {0x01, 0x00, 0x10, 0x00},
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           PerChannelAffineQuantization{/*scales=*/{0.1f, 0.01f},
                                        /*zero_points=*/{0, 0}}),
       .bias_shape = {},
       .bias_data = {},
       .expected = {0.1f, 0.02f}},
      {.weights_type = Type::kI4,
       .input_shape = {1, 4},
       .input_data = {1.0f, 2.0f, 3.0f, 4.0f},
       .weights_shape = {4, 4},
       .weights_data = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           PerChannelAffineQuantization{/*scales=*/{1.0f, 1.0f, 1.0f, 1.0f},
                                        /*zero_points=*/{0, 0, 0, 0}}),
       .bias_shape = {4},
       .bias_data = {0.1f, 0.2f, 0.3f, 0.4f},
       .expected = {1.1f, 2.2f, 3.3f, 4.4f}},
      {.weights_type = Type::kI2,
       .input_shape = {1, 4},
       .input_data = {1.0f, 2.0f, 3.0f, 4.0f},
       .weights_shape = {4, 4},
       .weights_data = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           PerChannelAffineQuantization{/*scales=*/{1.0f, 1.0f, 1.0f, 1.0f},
                                        /*zero_points=*/{0, 0, 0, 0}}),
       .bias_shape = {4},
       .bias_data = {0.1f, 0.2f, 0.3f, 0.4f},
       .expected = {1.1f, 2.2f, 3.3f, 4.4f}},
  };

  if constexpr (std::is_same_v<Tag, XnnpackMixinTag>) {
    GTEST_SKIP() << "FullyConnected with quantized weights is known to have "
                    "numerical issues on XNNPACK backend.";
  }

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.input_shape});
    Tensor<Tag> weights({.type = tc.weights_type,
                         .shape = tc.weights_shape,
                         .buffer = tc.weights_data,
                         .quantization = tc.quantization});
    Tensor<Tag> output;
    if (tc.bias_shape.empty()) {
      output = FullyConnected(input, weights);
    } else {
      Tensor<Tag> bias({.type = Type::kFP32,
                        .shape = tc.bias_shape,
                        .buffer = tc.bias_data});
      output = FullyConnected(input, weights, bias);
    }

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "FullyConnected with quantized weights is unimplemented "
                      "on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.input_data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(2e-2), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, ResizeBilinearOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int> size;
    bool align_corners;
    bool half_pixel_centers;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .size = {4, 4},
       .align_corners = false,
       .half_pixel_centers = true,
       .expected = {1.0f, 1.25f, 1.75f, 2.0f, 1.5f, 1.75f, 2.25f, 2.5f, 2.5f,
                    2.75f, 3.25f, 3.5f, 3.0f, 3.25f, 3.75f, 4.0f}},
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .size = {4, 4},
       .align_corners = true,
       .half_pixel_centers = false,
       .expected = {1.0f, 1.33333f, 1.66667f, 2.0f, 1.66667f, 2.0f, 2.33333f,
                    2.66667f, 2.33333f, 2.66667f, 3.0f, 3.33333f, 3.0f,
                    3.33333f, 3.66667f, 4.0f}},
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .size = {4, 4},
       .align_corners = false,
       .half_pixel_centers = false,
       .expected = {1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.5f,
                    4.0f, 4.0f, 3.0f, 3.5f, 4.0f, 4.0f}},
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .size = {1, 1},
       .align_corners = false,
       .half_pixel_centers = false,
       .expected = {1.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output =
        ResizeBilinear(input, tc.size, tc.align_corners, tc.half_pixel_centers);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "ResizeBilinear op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, DepthwiseConv2DOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 3, 3, 1}});
  std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor<Tag> filter(
      {.type = Type::kFP32, .shape = {1, 2, 2, 1}, .buffer = filter_data});
  std::vector<float> bias_data = {10.0f};
  Tensor<Tag> bias({.type = Type::kFP32, .shape = {1}, .buffer = bias_data});
  Tensor<Tag> output =
      DepthwiseConv2D(input, filter, bias, 1, 1, kPaddingValid);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "DepthwiseConv2D op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {47.0f, 57.0f, 77.0f, 87.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, DepthwiseConv2DPaddingSameOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 3, 3, 1}});
  std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor<Tag> filter(
      {.type = Type::kFP32, .shape = {1, 2, 2, 1}, .buffer = filter_data});
  std::vector<float> bias_data = {10.0f};
  Tensor<Tag> bias({.type = Type::kFP32, .shape = {1}, .buffer = bias_data});
  Tensor<Tag> output = DepthwiseConv2D(input, filter, bias, 1, 1, kPaddingSame);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "DepthwiseConv2D op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(9);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {47.0f, 57.0f, 31.0f, 77.0f, 87.0f,
                                        43.0f, 33.0f, 36.0f, 19.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, PowOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = Pow(input1, input2);

  auto status = this->bridge_->BuildGraph({input1, input2}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Pow op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input1_data = {1.0f, 2.0f, 3.0f, 8.0f};
  std::vector<float> input2_data = {2.0f, 2.0f, 2.0f, 2.0f};
  ASSERT_OK(this->bridge_->SetInput(input1, AsBytes(input1_data)));
  ASSERT_OK(this->bridge_->SetInput(input2, AsBytes(input2_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f, 4.0f, 9.0f, 64.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, SinOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = Sin(input1);

  auto status = this->bridge_->BuildGraph({input1}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Sin op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input1_data = {1.0f, 5.0f, 3.0f, 8.0f};
  ASSERT_OK(this->bridge_->SetInput(input1, AsBytes(input1_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {0.841471f, -0.958924294f, 0.14112f,
                                        0.989358246f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, CosOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = Cos(input1);

  auto status = this->bridge_->BuildGraph({input1}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Cos op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input1_data = {1.0f, 5.0f, 3.0f, 8.0f};
  ASSERT_OK(this->bridge_->SetInput(input1, AsBytes(input1_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {0.540302277f, 0.28366217f,
                                        -0.989992499f, -0.145500034f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, CeilOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.1f, -2.1f, 3.9f, -4.9f},
       .expected = {2.0f, -2.0f, 4.0f, -4.0f}},
      {.shape = {2, 2},
       .data = {1.1f, -2.1f, 3.9f, -4.9f},
       .expected = {2.0f, -2.0f, 4.0f, -4.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Ceil(input1);

    auto status = bridge->BuildGraph({input1}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Ceil op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, FloorOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.1f, -2.1f, 3.9f, -4.9f},
       .expected = {1.0f, -3.0f, 3.0f, -5.0f}},
      {.shape = {2, 2},
       .data = {1.1f, -2.1f, 3.9f, -4.9f},
       .expected = {1.0f, -3.0f, 3.0f, -5.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Floor(input1);

    auto status = bridge->BuildGraph({input1}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Floor op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, SignOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.1f, -2.1f, 0.0f, -0.0f},
       .expected = {1.0f, -1.0f, 0.0f, 0.0f}},
      {.shape = {2, 2},
       .data = {1.1f, -2.1f, 0.0f, -0.0f},
       .expected = {1.0f, -1.0f, 0.0f, 0.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Sign(input1);

    auto status = bridge->BuildGraph({input1}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Sign op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, RoundOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.5f, 2.5f, -1.5f, -2.5f},
       .expected = {2.0f, 2.0f, -2.0f, -2.0f}},
      {.shape = {2, 2},
       .data = {1.5f, 2.5f, -1.5f, -2.5f},
       .expected = {2.0f, 2.0f, -2.0f, -2.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Round(input1);

    auto status = bridge->BuildGraph({input1}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Round op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, SquareOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.5f, 2.0f, -1.5f, -3.0f},
       .expected = {2.25f, 4.0f, 2.25f, 9.0f}},
      {.shape = {2, 2},
       .data = {1.5f, 2.0f, -1.5f, -3.0f},
       .expected = {2.25f, 4.0f, 2.25f, 9.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Square(input1);

    auto status = bridge->BuildGraph({input1}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Square op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, NegOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, -2.0f, 3.0f, -4.0f},
       .expected = {-1.0f, 2.0f, -3.0f, 4.0f}},
      {.shape = {2, 2},
       .data = {1.0f, -2.0f, 3.0f, -4.0f},
       .expected = {-1.0f, 2.0f, -3.0f, 4.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Neg(input1);

    auto status = bridge->BuildGraph({input1}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Neg op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, AbsOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, -2.0f, 3.0f, -4.0f},
       .expected = {1.0f, 2.0f, 3.0f, 4.0f}},
      {.shape = {2, 2},
       .data = {1.0f, -2.0f, 3.0f, -4.0f},
       .expected = {1.0f, 2.0f, 3.0f, 4.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Abs(input1);

    auto status = bridge->BuildGraph({input1}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Abs op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, ExpOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, -1.0f, 0.0f},
       .expected = {2.718281828f, 7.389056099f, 0.367879441f, 1.0f}},
      {.shape = {2, 2},
       .data = {1.0f, 2.0f, -1.0f, 0.0f},
       .expected = {2.718281828f, 7.389056099f, 0.367879441f, 1.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Exp(input1);

    auto status = bridge->BuildGraph({input1}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Exp op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, LogOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.718281828f, 7.389056099f, 20.085536923f},
       .expected = {0.0f, 1.0f, 2.0f, 3.0f}},
      {.shape = {2, 2},
       .data = {1.0f, 2.718281828f, 7.389056099f, 20.085536923f},
       .expected = {0.0f, 1.0f, 2.0f, 3.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Log(input1);

    auto status = bridge->BuildGraph({input1}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Log op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, SelectOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> condition({.type = Type::kBOOL, .shape = {1, 2, 2, 1}});
  Tensor<Tag> t({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> e({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = Select(condition, t, e);

  auto status = this->bridge_->BuildGraph({condition, t, e}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Select op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<bool> condition_data = {true, false, false, true};
  std::vector<uint8_t> condition_bytes(condition_data.begin(),
                                       condition_data.end());
  std::vector<float> t_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> e_data = {5.0f, 6.0f, 7.0f, 8.0f};

  ASSERT_OK(this->bridge_->SetInput(condition, AsBytes(condition_bytes)));
  ASSERT_OK(this->bridge_->SetInput(t, AsBytes(t_data)));
  ASSERT_OK(this->bridge_->SetInput(e, AsBytes(e_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f, 6.0f, 7.0f, 4.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, SelectV2Op) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> condition({.type = Type::kBOOL, .shape = {1, 2, 2, 1}});
  Tensor<Tag> t({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> e({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = SelectV2(condition, t, e);

  auto status = this->bridge_->BuildGraph({condition, t, e}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "SelectV2 op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<bool> condition_data = {true, false, false, true};
  std::vector<uint8_t> condition_bytes(condition_data.begin(),
                                       condition_data.end());
  std::vector<float> t_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> e_data = {5.0f, 6.0f, 7.0f, 8.0f};

  ASSERT_OK(this->bridge_->SetInput(condition, AsBytes(condition_bytes)));
  ASSERT_OK(this->bridge_->SetInput(t, AsBytes(t_data)));
  ASSERT_OK(this->bridge_->SetInput(e, AsBytes(e_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f, 6.0f, 7.0f, 4.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, EqualOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = Equal(input1, input2);

  auto status = this->bridge_->BuildGraph({input1, input2}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Equal op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input1_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> input2_data = {1.0f, 6.0f, 3.0f, 8.0f};

  ASSERT_OK(this->bridge_->SetInput(input1, AsBytes(input1_data)));
  ASSERT_OK(this->bridge_->SetInput(input2, AsBytes(input2_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<uint8_t> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<uint8_t> expected_output = {1, 0, 1, 0};
  EXPECT_EQ(actual_output, expected_output);
}

TYPED_TEST_P(NumericalTestSuite, GreaterEqualOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = GreaterEqual(input1, input2);

  auto status = this->bridge_->BuildGraph({input1, input2}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "GreaterEqual op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input1_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> input2_data = {1.0f, 6.0f, 3.0f, 8.0f};

  ASSERT_OK(this->bridge_->SetInput(input1, AsBytes(input1_data)));
  ASSERT_OK(this->bridge_->SetInput(input2, AsBytes(input2_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<uint8_t> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<uint8_t> expected_output = {1, 0, 1, 0};
  EXPECT_EQ(actual_output, expected_output);
}

TYPED_TEST_P(NumericalTestSuite, NotEqualOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = NotEqual(input1, input2);

  auto status = this->bridge_->BuildGraph({input1, input2}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "NotEqual op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input1_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> input2_data = {1.0f, 6.0f, 3.0f, 8.0f};

  ASSERT_OK(this->bridge_->SetInput(input1, AsBytes(input1_data)));
  ASSERT_OK(this->bridge_->SetInput(input2, AsBytes(input2_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<uint8_t> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<uint8_t> expected_output = {0, 1, 0, 1};
  EXPECT_EQ(actual_output, expected_output);
}

TYPED_TEST_P(NumericalTestSuite, LessOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = Less(input1, input2);

  auto status = this->bridge_->BuildGraph({input1, input2}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Less op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input1_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> input2_data = {5.0f, 2.0f, 1.0f, 8.0f};

  ASSERT_OK(this->bridge_->SetInput(input1, AsBytes(input1_data)));
  ASSERT_OK(this->bridge_->SetInput(input2, AsBytes(input2_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<uint8_t> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<uint8_t> expected_output = {1, 0, 0, 1};
  EXPECT_EQ(actual_output, expected_output);
}

TYPED_TEST_P(NumericalTestSuite, LogicalAndOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input1({.type = Type::kBOOL, .shape = {1, 2, 2, 1}});
  Tensor<Tag> input2({.type = Type::kBOOL, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = LogicalAnd(input1, input2);

  auto status = this->bridge_->BuildGraph({input1, input2}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "LogicalAnd op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<bool> input1_bool = {true, true, false, false};
  std::vector<uint8_t> input1_data(input1_bool.begin(), input1_bool.end());
  std::vector<bool> input2_bool = {true, false, true, false};
  std::vector<uint8_t> input2_data(input2_bool.begin(), input2_bool.end());

  ASSERT_OK(this->bridge_->SetInput(input1, AsBytes(input1_data)));
  ASSERT_OK(this->bridge_->SetInput(input2, AsBytes(input2_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<uint8_t> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<uint8_t> expected_output = {1, 0, 0, 0};
  EXPECT_EQ(actual_output, expected_output);
}

TYPED_TEST_P(NumericalTestSuite, LogicalNotOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input1({.type = Type::kBOOL, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = LogicalNot(input1);

  auto status = this->bridge_->BuildGraph({input1}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "LogicalNot op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<bool> input1_bool = {true, false, true, false};
  std::vector<uint8_t> input1_data(input1_bool.begin(), input1_bool.end());

  ASSERT_OK(this->bridge_->SetInput(input1, AsBytes(input1_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<uint8_t> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<uint8_t> expected_output = {0, 1, 0, 1};
  EXPECT_EQ(actual_output, expected_output);
}

TYPED_TEST_P(NumericalTestSuite, SqueezeWithDimsOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 1, 2}});
  Tensor<Tag> output = Squeeze(input, {0, 2});

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "SqueezeWithDims op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, DepthToSpaceOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 2, 4}});
  Tensor<Tag> output = DepthToSpace(input, 2);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "DepthToSpace op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f,  2.0f,  5.0f,  6.0f,  3.0f,  4.0f,
                                   7.0f,  8.0f,  9.0f,  10.0f, 13.0f, 14.0f,
                                   11.0f, 12.0f, 15.0f, 16.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(16);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {
      1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, GatherOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> indices({.type = Type::kI32, .shape = {1}});
  Tensor<Tag> output = Gather(input, indices, 1, 0);

  auto status = this->bridge_->BuildGraph({input, indices}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Gather op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int32_t> indices_data = {1};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));
  ASSERT_OK(this->bridge_->SetInput(indices, AsBytes(indices_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(2);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {3.0f, 4.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, OneHotOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> indices({.type = Type::kI32, .shape = {4}});
  std::vector<int32_t> depth_data = {4};
  Tensor<Tag> depth({.type = Type::kI32, .shape = {}, .buffer = depth_data});
  std::vector<float> on_value_data = {1.0f};
  Tensor<Tag> on_value(
      {.type = Type::kFP32, .shape = {}, .buffer = on_value_data});
  std::vector<float> off_value_data = {0.0f};
  Tensor<Tag> off_value(
      {.type = Type::kFP32, .shape = {}, .buffer = off_value_data});
  Tensor<Tag> output = OneHot(indices, depth, on_value, off_value, -1);

  auto status = this->bridge_->BuildGraph({indices}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "OneHot op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<int32_t> indices_data = {0, 1, 2, 0};
  ASSERT_OK(this->bridge_->SetInput(indices, AsBytes(indices_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(16);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                        1.0f, 0.0f, 0.0f, 0.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, CumsumOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 1, 2, 2}});
  std::vector<int32_t> axis_data = {3};
  Tensor<Tag> axis({.type = Type::kI32, .shape = {}, .buffer = axis_data});
  Tensor<Tag> output = Cumsum(input, axis);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Cumsum op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f, 3.0f, 3.0f, 7.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, ReverseOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 1, 2, 2}});
  std::vector<int32_t> axes_data = {3};
  Tensor<Tag> axes({.type = Type::kI32, .shape = {1}, .buffer = axes_data});
  Tensor<Tag> output = Reverse(input, axes);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Reverse op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {2.0f, 1.0f, 4.0f, 3.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, TileOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 1, 2, 2}});
  std::vector<int32_t> multiples_data = {1, 1, 2, 2};
  Tensor<Tag> multiples(
      {.type = Type::kI32, .shape = {4}, .buffer = multiples_data});
  Tensor<Tag> output = Tile(input, multiples);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Tile op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(16);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                        3.0f, 4.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                                        3.0f, 4.0f, 3.0f, 4.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, ArgMaxOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 3, 4}});
  std::vector<int32_t> axis_data = {2};
  Tensor<Tag> axis({.type = Type::kI32, .shape = {1}, .buffer = axis_data});
  Tensor<Tag> output = ArgMax(input, axis, Type::kI32);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "ArgMax op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
      12, 13, 14, 15, 16, 17, 18, 19, 24, 21, 22, 23,
  };
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<int32_t> actual_output(8);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<int32_t> expected_output = {2, 2, 2, 2, 2, 2, 2, 2};
  EXPECT_EQ(actual_output, expected_output);
}

TYPED_TEST_P(NumericalTestSuite, CastOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  Tensor<Tag> output = Cast(input, Type::kI32);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Cast op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.1f, 2.9f, 3.5f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<int32_t> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<int32_t> expected_output = {1, 2, 3, 4};
  if constexpr (std::is_same_v<Tag, XnnpackMixinTag>) {
    expected_output = {1, 3, 4, 4};
  }
  EXPECT_EQ(actual_output, expected_output);
}

TYPED_TEST_P(NumericalTestSuite, EmbeddingLookupOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> ids({.type = Type::kI32, .shape = {1}});
  std::vector<float> weights_data = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                     7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                                     13.0f, 14.0f, 15.0f, 16.0f};
  Tensor<Tag> weights(
      {.type = Type::kFP32, .shape = {8, 2}, .buffer = weights_data});
  Tensor<Tag> output = EmbeddingLookup(ids, weights);

  auto status = this->bridge_->BuildGraph({ids}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "EmbeddingLookup op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<int32_t> ids_data = {4};
  ASSERT_OK(this->bridge_->SetInput(ids, AsBytes(ids_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(2);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {9.0f, 10.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, BatchMatmulOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input1({.type = Type::kFP32, .shape = {1, 2, 3, 4}});
  Tensor<Tag> input2({.type = Type::kFP32, .shape = {1, 2, 4, 5}});
  Tensor<Tag> output = BatchMatMul(input1, input2);

  auto status = this->bridge_->BuildGraph({input1, input2}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "BatchMatmul op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input1_data(24, 1.0f);
  std::vector<float> input2_data(40, 1.0f);
  ASSERT_OK(this->bridge_->SetInput(input1, AsBytes(input1_data)));
  ASSERT_OK(this->bridge_->SetInput(input2, AsBytes(input2_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(30);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output(30, 4.0f);
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, PReluOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 1, 2, 2}});
  std::vector<float> alpha_data = {0.1f};
  Tensor<Tag> alpha({.type = Type::kFP32, .shape = {1}, .buffer = alpha_data});
  Tensor<Tag> output = PRelu(input, alpha);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "PRelu op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {-1.0f, 2.0f, -3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {-0.1f, 2.0f, -0.3f, 4.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, L2NormalizationOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 1, 1, 4}});
  Tensor<Tag> output = L2Normalization(input);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "L2Normalization op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(4);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {0.182574f, 0.365148f, 0.547722f,
                                        0.730296f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, LstmOp) {
  using Tag = typename TypeParam::Tag;
  const int state_size = 4;
  Tensor<Tag> intermediate(
      {.type = Type::kFP32, .shape = {1, 1, 1, state_size * 4}});
  Tensor<Tag> prev_state({.type = Type::kFP32, .shape = {1, 1, 1, state_size}});
  std::vector<Tensor<Tag>> outputs = Lstm(intermediate, prev_state);

  auto status = this->bridge_->BuildGraph({intermediate, prev_state},
                                          {outputs[0], outputs[1]});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Lstm op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  const std::vector<float> intermediate_data = {
      0.1f, 0.2f, 0.3f, 0.4f,  // input gate
      0.5f, 0.6f, 0.7f, 0.8f,  // new input
      0.9f, 1.0f, 1.1f, 1.2f,  // forget gate
      1.3f, 1.4f, 1.5f, 1.6f,  // output gate
  };
  const std::vector<float> prev_state_data = {
      0.1f,
      0.2f,
      0.3f,
      0.4f,
  };

  ASSERT_OK(this->bridge_->SetInput(intermediate, AsBytes(intermediate_data)));
  ASSERT_OK(this->bridge_->SetInput(prev_state, AsBytes(prev_state_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_new_state(state_size);
  std::vector<float> actual_activation(state_size);
  ASSERT_OK(this->bridge_->GetOutput(outputs[0], AsBytes(actual_new_state)));
  ASSERT_OK(this->bridge_->GetOutput(outputs[1], AsBytes(actual_activation)));

  auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
  auto tanh = [](float x) { return std::tanh(x); };

  std::vector<float> expected_new_state(state_size);
  std::vector<float> expected_activation(state_size);

  for (int i = 0; i < state_size; ++i) {
    float input_gate = sigmoid(intermediate_data[i]);
    float new_input = tanh(intermediate_data[state_size + i]);
    float forget_gate = sigmoid(intermediate_data[2 * state_size + i]);
    float output_gate = sigmoid(intermediate_data[3 * state_size + i]);

    expected_new_state[i] =
        input_gate * new_input + forget_gate * prev_state_data[i];
    expected_activation[i] = output_gate * tanh(expected_new_state[i]);
  }

  EXPECT_THAT(actual_new_state, Pointwise(FloatNear(1e-5), expected_new_state));
  EXPECT_THAT(actual_activation,
              Pointwise(FloatNear(1e-5), expected_activation));
}

TYPED_TEST_P(NumericalTestSuite, Conv2DOp) {
  using Tag = typename TypeParam::Tag;
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor<Tag> filter(
      {.type = Type::kFP32, .shape = {1, 2, 2, 1}, .buffer = filter_data});
  std::vector<float> bias_data = {1.0f};
  Tensor<Tag> bias({.type = Type::kFP32, .shape = {1}, .buffer = bias_data});

  Tensor<Tag> output = Conv2D(input, filter, bias, /*stride_h=*/1,
                              /*stride_w=*/1, kPaddingValid);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "Conv2D op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(1);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f * 1.0f + 2.0f * 2.0f +
                                        3.0f * 3.0f + 4.0f * 4.0f + 1.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, TransposeConvOp) {
  using Tag = typename TypeParam::Tag;
  std::vector<int> output_shape_vec = {1, 3, 3, 1};
  std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor<Tag> filter(
      {.type = Type::kFP32, .shape = {1, 2, 2, 1}, .buffer = filter_data});
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  std::vector<float> bias_data = {0.0f};
  Tensor<Tag> bias({.type = Type::kFP32, .shape = {1}, .buffer = bias_data});

  Tensor<Tag> output =
      TransposeConv(filter, input, bias, output_shape_vec, kPaddingValid, 1, 1);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "TransposeConv op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(9);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f,  4.0f, 4.0f,  6.0f, 20.0f,
                                        16.0f, 9.0f, 24.0f, 16.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, TransposeConv2DOp) {
  using Tag = typename TypeParam::Tag;
  std::vector<int> output_shape_vec = {1, 3, 3, 1};
  std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor<Tag> filter(
      {.type = Type::kFP32, .shape = {1, 2, 2, 1}, .buffer = filter_data});
  Tensor<Tag> input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  std::vector<float> bias_data = {0.0f};
  Tensor<Tag> bias({.type = Type::kFP32, .shape = {1}, .buffer = bias_data});

  Tensor<Tag> output = TransposeConv2D(filter, input, bias, output_shape_vec,
                                       kPaddingValid, 1, 1);

  auto status = this->bridge_->BuildGraph({input}, {output});
  if (status.code() == absl::StatusCode::kUnimplemented) {
    GTEST_SKIP() << "TransposeConv2D op is unimplemented on this backend.";
  }
  ASSERT_OK(status);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK(this->bridge_->SetInput(input, AsBytes(input_data)));

  ASSERT_OK(this->bridge_->Execute());

  std::vector<float> actual_output(9);
  ASSERT_OK(this->bridge_->GetOutput(output, AsBytes(actual_output)));

  std::vector<float> expected_output = {1.0f,  4.0f, 4.0f,  6.0f, 20.0f,
                                        16.0f, 9.0f, 24.0f, 16.0f};
  EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), expected_output));
}

TYPED_TEST_P(NumericalTestSuite, MaximumOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape1;
    std::vector<float> data1;
    std::vector<int> shape2;
    std::vector<float> data2;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 2, 2, 1},
       .data2 = {2.0f, 4.0f, 6.0f, 7.0f},
       .expected = {2.0f, 5.0f, 6.0f, 8.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 1, 2, 2},
       .data2 = {2.0f, 4.0f, 6.0f, 7.0f},
       .expected = {2.0f, 5.0f, 6.0f, 8.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1},
       .data2 = {2.0f},
       .expected = {2.0f, 2.0f, 3.0f, 4.0f}},
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 1, 2, 1},
       .data2 = {2.0f, 7.0f},
       .expected = {2.0f, 7.0f, 3.0f, 8.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape1});
    Tensor<Tag> input2({.type = Type::kFP32, .shape = tc.shape2});
    Tensor<Tag> output = Maximum(input1, input2);

    auto status = bridge->BuildGraph({input1, input2}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Maximum op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data1)));
    ASSERT_OK(bridge->SetInput(input2, AsBytes(tc.data2)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, MinimumOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape1;
    std::vector<float> data1;
    std::vector<int> shape2;
    std::vector<float> data2;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 2, 2, 1},
       .data2 = {2.0f, 4.0f, 6.0f, 7.0f},
       .expected = {1.0f, 4.0f, 3.0f, 7.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 1, 2, 2},
       .data2 = {2.0f, 4.0f, 6.0f, 7.0f},
       .expected = {1.0f, 4.0f, 3.0f, 7.0f}},
      {.shape1 = {1, 1, 2, 2},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .shape2 = {1},
       .data2 = {2.0f},
       .expected = {1.0f, 2.0f, 2.0f, 2.0f}},
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 1, 2, 1},
       .data2 = {2.0f, 7.0f},
       .expected = {1.0f, 5.0f, 2.0f, 7.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape1});
    Tensor<Tag> input2({.type = Type::kFP32, .shape = tc.shape2});
    Tensor<Tag> output = Minimum(input1, input2);

    auto status = bridge->BuildGraph({input1, input2}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Minimum op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data1)));
    ASSERT_OK(bridge->SetInput(input2, AsBytes(tc.data2)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, FloorModOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape1;
    std::vector<float> data1;
    std::vector<int> shape2;
    std::vector<float> data2;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 2, 2, 1},
       .data2 = {2.0f, 4.0f, 6.0f, 4.0f},
       .expected = {1.0f, 1.0f, 3.0f, 0.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape1});
    Tensor<Tag> input2({.type = Type::kFP32, .shape = tc.shape2});
    Tensor<Tag> output = FloorMod(input1, input2);

    auto status = bridge->BuildGraph({input1, input2}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "FloorMod op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data1)));
    ASSERT_OK(bridge->SetInput(input2, AsBytes(tc.data2)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, FloorDivOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape1;
    std::vector<float> data1;
    std::vector<int> shape2;
    std::vector<float> data2;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape1 = {1, 2, 2, 1},
       .data1 = {1.0f, 5.0f, 3.0f, 8.0f},
       .shape2 = {1, 2, 2, 1},
       .data2 = {2.0f, 4.0f, 6.0f, 4.0f},
       .expected = {0.0f, 1.0f, 0.0f, 2.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape1});
    Tensor<Tag> input2({.type = Type::kFP32, .shape = tc.shape2});
    Tensor<Tag> output = FloorDiv(input1, input2);

    auto status = bridge->BuildGraph({input1, input2}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "FloorDiv op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data1)));
    ASSERT_OK(bridge->SetInput(input2, AsBytes(tc.data2)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, SumOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int32_t> axis;
    bool keep_dims;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .axis = {1},
       .keep_dims = false,
       .expected = {4.0f, 6.0f}},
      {.shape = {2, 3},
       .data = {1, 2, 3, 4, 5, 6},
       .axis = {0},
       .keep_dims = false,
       .expected = {5, 7, 9}},
      {.shape = {2, 3},
       .data = {1, 2, 3, 4, 5, 6},
       .axis = {1},
       .keep_dims = false,
       .expected = {6, 15}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> axis({.type = Type::kI32,
                      .shape = {static_cast<int>(tc.axis.size())},
                      .buffer = OwningCpuBuffer::Copy<Type::kI32>(tc.axis)});
    Tensor<Tag> output = Sum(input, axis, tc.keep_dims);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Sum op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, ReduceMaxOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int32_t> axis;
    bool keep_dims;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .axis = {1},
       .keep_dims = false,
       .expected = {3.0f, 4.0f}},
      {.shape = {2, 3},
       .data = {1, 2, 3, 4, 5, 6},
       .axis = {0},
       .keep_dims = false,
       .expected = {4, 5, 6}},
      {.shape = {2, 3},
       .data = {1, 2, 3, 4, 5, 6},
       .axis = {1},
       .keep_dims = false,
       .expected = {3, 6}},
      {.shape = {2, 3},
       .data = {1, 2, 3, 4, 5, 6},
       .axis = {0, 1},
       .keep_dims = false,
       .expected = {6}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> axis({.type = Type::kI32,
                      .shape = {static_cast<int>(tc.axis.size())},
                      .buffer = OwningCpuBuffer::Copy<Type::kI32>(tc.axis)});
    Tensor<Tag> output = ReduceMax(input, axis, tc.keep_dims);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "ReduceMax op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, MeanOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int32_t> axis;
    bool keep_dims;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .axis = {1},
       .keep_dims = false,
       .expected = {2.0f, 3.0f}},
      {.shape = {2, 3},
       .data = {1, 2, 3, 4, 5, 6},
       .axis = {0},
       .keep_dims = false,
       .expected = {2.5f, 3.5f, 4.5f}},
      {.shape = {2, 3},
       .data = {1, 2, 3, 4, 5, 6},
       .axis = {1},
       .keep_dims = false,
       .expected = {2.0f, 5.0f}},
      {.shape = {2, 3},
       .data = {1, 2, 3, 4, 5, 6},
       .axis = {0, 1},
       .keep_dims = false,
       .expected = {3.5f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> axis({.type = Type::kI32,
                      .shape = {static_cast<int>(tc.axis.size())},
                      .buffer = OwningCpuBuffer::Copy<Type::kI32>(tc.axis)});
    Tensor<Tag> output = Mean(input, axis, tc.keep_dims);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Mean op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, AveragePool2DOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    int filter_height;
    int filter_width;
    int stride_h;
    int stride_w;
    Padding padding;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .filter_height = 2,
       .filter_width = 2,
       .stride_h = 1,
       .stride_w = 1,
       .padding = kPaddingValid,
       .expected = {2.5f}},
      {.shape = {1, 4, 4, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
       .filter_height = 2,
       .filter_width = 2,
       .stride_h = 2,
       .stride_w = 2,
       .padding = kPaddingValid,
       .expected = {3.5f, 5.5f, 11.5f, 13.5f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = AveragePool2D(input, tc.filter_height, tc.filter_width,
                                       tc.stride_h, tc.stride_w, tc.padding);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "AveragePool2D op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, MaxPool2DOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    int filter_height;
    int filter_width;
    int stride_h;
    int stride_w;
    Padding padding;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .filter_height = 2,
       .filter_width = 2,
       .stride_h = 1,
       .stride_w = 1,
       .padding = kPaddingValid,
       .expected = {4.0f}},
      {.shape = {1, 4, 4, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
       .filter_height = 2,
       .filter_width = 2,
       .stride_h = 2,
       .stride_w = 2,
       .padding = kPaddingValid,
       .expected = {6.0f, 8.0f, 14.0f, 16.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = MaxPool2D(input, tc.filter_height, tc.filter_width,
                                   tc.stride_h, tc.stride_w, tc.padding);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "MaxPool2D op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, SplitOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    int axis;
    int num_splits;
    std::vector<std::vector<float>> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 4, 6},
       .data = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
       .axis = 1,
       .num_splits = 2,
       .expected = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                     10.0f, 11.0f, 12.0f},
                    {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                     21.0f, 22.0f, 23.0f, 24.0f}}},
      {.shape = {4, 2},
       .data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       .axis = 0,
       .num_splits = 2,
       .expected = {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}}},
      {.shape = {2, 4},
       .data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       .axis = 1,
       .num_splits = 2,
       .expected = {{1.0f, 2.0f, 5.0f, 6.0f}, {3.0f, 4.0f, 7.0f, 8.0f}}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    std::vector<Tensor<Tag>> outputs = Split(input, tc.axis, tc.num_splits);

    std::vector<TensorHandle> output_handles;
    output_handles.reserve(outputs.size());
    for (const auto& out : outputs) {
      output_handles.push_back(out);
    }

    auto status = bridge->BuildGraph({input}, output_handles);
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Split op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    for (size_t i = 0; i < outputs.size(); ++i) {
      std::vector<float> actual_output(tc.expected[i].size());
      ASSERT_OK(bridge->GetOutput(outputs[i], AsBytes(actual_output)));
      EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected[i]));
    }
  }
}

TYPED_TEST_P(NumericalTestSuite, PackOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data1;
    std::vector<float> data2;
    int axis;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {2, 2, 1},
       .data1 = {1.0f, 2.0f, 3.0f, 4.0f},
       .data2 = {5.0f, 6.0f, 7.0f, 8.0f},
       .axis = 3,
       .expected = {1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f}},
      {.shape = {1, 2},
       .data1 = {1, 2},
       .data2 = {3, 4},
       .axis = 0,
       .expected = {1, 2, 3, 4}},
      {.shape = {2, 1},
       .data1 = {1, 2},
       .data2 = {3, 4},
       .axis = 1,
       .expected = {1, 3, 2, 4}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input1({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> input2({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = Pack({input1, input2}, tc.axis);

    auto status = bridge->BuildGraph({input1, input2}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Pack op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input1, AsBytes(tc.data1)));
    ASSERT_OK(bridge->SetInput(input2, AsBytes(tc.data2)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

TYPED_TEST_P(NumericalTestSuite, UnpackOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    int num;
    int axis;
    std::vector<std::vector<float>> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {2, 2, 2},
       .data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       .num = 2,
       .axis = 2,
       .expected = {{1.0f, 3.0f, 5.0f, 7.0f}, {2.0f, 4.0f, 6.0f, 8.0f}}},
      {.shape = {2, 2},
       .data = {1, 2, 3, 4},
       .num = 2,
       .axis = 0,
       .expected = {{1, 2}, {3, 4}}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    std::vector<Tensor<Tag>> outputs = Unpack(input, tc.num, tc.axis);

    std::vector<TensorHandle> output_handles;
    output_handles.reserve(outputs.size());
    for (const auto& out : outputs) {
      output_handles.push_back(out);
    }

    auto status = bridge->BuildGraph({input}, output_handles);
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP() << "Unpack op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    for (size_t i = 0; i < outputs.size(); ++i) {
      std::vector<float> actual_output(tc.expected[i].size());
      ASSERT_OK(bridge->GetOutput(outputs[i], AsBytes(actual_output)));
      EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected[i]));
    }
  }
}

TYPED_TEST_P(NumericalTestSuite, ResizeNearestNeighborOp) {
  using Tag = typename TypeParam::Tag;
  struct TestCase {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int> size;
    bool align_corners;
    bool half_pixel_centers;
    std::vector<float> expected;
  };
  std::vector<TestCase> test_cases = {
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .size = {4, 4},
       .align_corners = false,
       .half_pixel_centers = false,
       .expected = {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f,
                    4.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f}},
      {.shape = {1, 2, 2, 1},
       .data = {1.0f, 2.0f, 3.0f, 4.0f},
       .size = {6, 6},
       .align_corners = false,
       .half_pixel_centers = false,
       .expected = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f,
                    2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                    3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f,
                    4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f}},
  };

  for (const auto& tc : test_cases) {
    auto bridge = TypeParam::CreateBridge();
    ASSERT_OK(bridge->Initialize());

    Tensor<Tag> input({.type = Type::kFP32, .shape = tc.shape});
    Tensor<Tag> output = ResizeNearestNeighbor(input, tc.size, tc.align_corners,
                                               tc.half_pixel_centers);

    auto status = bridge->BuildGraph({input}, {output});
    if (status.code() == absl::StatusCode::kUnimplemented) {
      GTEST_SKIP()
          << "ResizeNearestNeighbor op is unimplemented on this backend.";
    }
    ASSERT_OK(status);

    ASSERT_OK(bridge->SetInput(input, AsBytes(tc.data)));

    ASSERT_OK(bridge->Execute());

    std::vector<float> actual_output(tc.expected.size());
    ASSERT_OK(bridge->GetOutput(output, AsBytes(actual_output)));

    EXPECT_THAT(actual_output, Pointwise(FloatNear(1e-5), tc.expected));
  }
}

// Register all typed tests.
REGISTER_TYPED_TEST_SUITE_P(
    NumericalTestSuite, AddOp, SubOp, MulOp, DivOp, GeluOp, ReluOp, Relu6Op,
    LeakyReluOp, EluOp, HardSwishOp, LogisticOp, TanhOp, SqrtOp, RsqrtOp,
    SoftmaxOp, TransposeOp, ReshapeOp, ExpandDimsOp, SqueezeOp, SliceOp,
    ConcatenationOp, FullyConnectedOp, FullyConnectedQuantizedOp,
    ResizeBilinearOp, DepthwiseConv2DOp, DepthwiseConv2DPaddingSameOp, PowOp,
    SinOp, CosOp, CeilOp, FloorOp, SignOp, RoundOp, SquareOp, NegOp, AbsOp,
    ExpOp, LogOp, SelectOp, SelectV2Op, EqualOp, GreaterEqualOp, NotEqualOp,
    LessOp, LogicalAndOp, LogicalNotOp, SqueezeWithDimsOp, DepthToSpaceOp,
    GatherOp, OneHotOp, CumsumOp, ReverseOp, TileOp, ArgMaxOp, CastOp,
    EmbeddingLookupOp, BatchMatmulOp, PReluOp, L2NormalizationOp, LstmOp,
    Conv2DOp, TransposeConvOp, TransposeConv2DOp, MaximumOp, MinimumOp,
    FloorModOp, FloorDivOp, SumOp, ReduceMaxOp, MeanOp, AveragePool2DOp,
    MaxPool2DOp, SplitOp, PackOp, UnpackOp, ResizeNearestNeighborOp);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TESTING_NUMERICAL_TEST_SUITE_H_
