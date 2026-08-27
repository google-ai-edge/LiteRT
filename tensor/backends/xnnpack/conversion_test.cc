/* Copyright 2025 Google LLC.

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

#include "tensor/backends/xnnpack/conversion.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using XnnTensor = Tensor<XnnpackMixinTag>;

TEST(XnnpackConversionTest, MarksExternalValuesAndCopiesConstants) {
  XnnTensor runtime_input({.name = "input", .type = Type::kFP32, .shape = {3}});
  XnnTensor constant_bias({.name = "bias",
                           .type = Type::kFP32,
                           .shape = {3},
                           .buffer = std::vector<float>{1.f, 2.f, 3.f}});
  XnnTensor sum = Add(runtime_input, constant_bias);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({sum}));
  ASSERT_EQ(graph->values().size(), 3);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t input_index,
                                  graph->Lookup(runtime_input));
  const XnnpackValue& input_value = graph->values()[input_index];
  EXPECT_NE(input_value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT, 0);
  EXPECT_EQ(input_value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT, 0);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t bias_index,
                                  graph->Lookup(constant_bias));
  const XnnpackValue& bias_value = graph->values()[bias_index];
  EXPECT_EQ(bias_value.flags, 0);
  EXPECT_EQ(bias_value.data.size(), sizeof(float) * 3);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t output_index, graph->Lookup(sum));
  const XnnpackValue& output_value = graph->values()[output_index];
  EXPECT_NE(output_value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT, 0);
}

TEST(XnnpackConversionTest, SharedTensorRegisteredOnce) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1}});
  XnnTensor other({.name = "other",
                   .type = Type::kFP32,
                   .shape = {1},
                   .buffer = std::vector<float>{2.f}});
  XnnTensor first = Add(input, other);
  XnnTensor second = Add(first, input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({second}));

  int external_inputs = 0;
  for (const XnnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++external_inputs;
    }
  }
  EXPECT_EQ(external_inputs, 1);
}

TEST(XnnpackConversionTest, Relu6SetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = Relu6(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const XnnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 1);
}

TEST(XnnpackConversionTest, LeakyReluSetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = LeakyRelu(input, 0.2f);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const XnnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 1);
}

TEST(XnnpackConversionTest, EluSetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = Elu(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const XnnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 1);
}

TEST(XnnpackConversionTest, GeluSetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = Gelu(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const XnnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 1);
}

TEST(XnnpackConversionTest, HardSwishSetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = HardSwish(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const XnnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 1);
}

TEST(XnnpackConversionTest, PReluSetsCorrectNode) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor alpha({.name = "alpha", .type = Type::kFP32, .shape = {2}});
  XnnTensor output = PRelu(input, alpha);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));

  int num_external_inputs = 0;
  for (const XnnpackValue& value : graph->values()) {
    if (value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
      ++num_external_inputs;
    }
  }
  EXPECT_EQ(num_external_inputs, 2);
}

TEST(XnnpackConversionTest, L2NormalizationReturnsUnimplemented) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output = L2Normalization(input);

  EXPECT_EQ(BuildXnnpackGraph({output}).status().code(),
            absl::StatusCode::kUnimplemented);
}

// This test is disabled because the lowering should not do implicit data
// conversions.
//
// TODO: b/493560478 - Remove this.
//
// TEST(XnnpackConversionTest, DISABLED_DequantizesQuantizedInt8ConstantWeights)
// {
//   XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 4}});
//   auto quantization = std::make_shared<PerChannelAffineQuantization>(
//       std::vector<float>{0.5f, 0.25f}, std::vector<int64_t>{0, 0},
//       /*quantized_dimension=*/0);
//   XnnTensor weights({.name = "weights",
//                      .type = Type::kI8,
//                      .shape = {2, 4},
//                      .buffer = std::vector<int8_t>{2, 4, -2, -4, 4, 8, -4,
//                      -8}, .quantization = quantization});
//   XnnTensor output = FullyConnected(input, weights);
//
//   LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));
//   LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t weights_index,
//   graph->Lookup(weights)); const XnnpackValue& weights_value =
//   graph->values()[weights_index]; ASSERT_EQ(weights_value.buffer.size(),
//   sizeof(float) * 8); EXPECT_EQ(weights_value.info.type, Type::kFP32);
//
//   std::array<float, 8> expected = {1.0f, 2.0f, -1.0f, -2.0f,
//                                    1.0f, 2.0f, -1.0f, -2.0f};
//   for (size_t i = 0; i < expected.size(); ++i) {
//     float value = 0.0f;
//     std::memcpy(&value, weights_value.buffer.data() + i * sizeof(float),
//                 sizeof(float));
//     EXPECT_FLOAT_EQ(value, expected[i]);
//   }
// }

}  // namespace
}  // namespace litert::tensor
