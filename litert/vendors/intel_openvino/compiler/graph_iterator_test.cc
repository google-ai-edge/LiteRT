// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/intel_openvino/compiler/graph_iterator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "openvino/core/type/element_type.hpp"
#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/cc/litert_element_type.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/load_test_model.h"

namespace litert::openvino {
namespace {

struct ConversionTestParam {
  std::string device;
  ov::element::Type expected_type;
  bool adjusts_zero_points;
};

class GraphIteratorI2ConversionTest
    : public ::testing::TestWithParam<ConversionTestParam> {};

TEST_P(GraphIteratorI2ConversionTest, ConvertsConstantWeights) {
  auto cc_model = testing::LoadTestFileModel("FFW-2-bit.tflite");
  const LiteRtCompilerContext* context = LrtGetCompilerContext();
  litert::compiler::Model model(context, cc_model.Get());
  auto graph_or = model.Subgraph(0);
  ASSERT_TRUE(graph_or.HasValue());
  auto graph = graph_or.Value();

  GraphIteratorDelegate iterator(context, &graph, GetParam().device);
  const auto operations = graph.Ops();
  size_t operation_index = 0;
  for (; !iterator.is_end(); iterator.next()) {
    auto decoder = iterator.get_decoder();
    auto operation =
        std::dynamic_pointer_cast<DecoderOperation>(std::move(decoder));
    if (!operation) {
      continue;
    }

    ASSERT_LT(operation_index, operations.size());
    const auto& op = operations[operation_index++];
    const auto inputs = op.Inputs();
    ASSERT_EQ(operation->get_input_size(), inputs.size());
    for (size_t input_index = 0; input_index < operation->get_input_size();
         ++input_index) {
      const auto& input = inputs[input_index];
      if (!input.HasWeights() ||
          input.ElementType() != litert::ElementType::Int2) {
        continue;
      }

      const auto tensor_info = operation->get_input_tensor_info(input_index);
      ASSERT_EQ(tensor_info.m_element_type, GetParam().expected_type);
      ASSERT_NE(tensor_info.m_tensor_data, nullptr);
      ASSERT_NE(tensor_info.m_quantization_info, nullptr);

      const auto original_bytes = input.Weights().Bytes();
      const auto* converted_bytes =
          static_cast<const uint8_t*>(tensor_info.m_tensor_data);
      if (GetParam().device == "GPU") {
        const auto sign_extend = [](uint8_t value) {
          return static_cast<uint8_t>((value & 0x2) ? 0xC | value : value);
        };
        for (size_t byte_index = 0; byte_index < original_bytes.size();
             ++byte_index) {
          const uint8_t packed = original_bytes[byte_index];
          for (int element = 0; element < 4; element += 2) {
            const uint8_t low = (packed >> (2 * element)) & 0x3;
            const uint8_t high = (packed >> (2 * (element + 1))) & 0x3;
            EXPECT_EQ(converted_bytes[2 * byte_index + element / 2],
                      sign_extend(low) | (sign_extend(high) << 4));
          }
        }
      } else {
        for (size_t byte_index = 0; byte_index < original_bytes.size();
             ++byte_index) {
          EXPECT_EQ(converted_bytes[byte_index],
                    static_cast<uint8_t>(original_bytes[byte_index] ^ 0xAA));
        }
      }

      const auto zero_points =
          tensor_info.m_quantization_info->get_zero_point();
      for (int64_t zero_point : zero_points) {
        EXPECT_EQ(zero_point, GetParam().adjusts_zero_points ? 2 : 0);
      }
      return;
    }
  }
  FAIL() << "No constant int2 weights found";
}

INSTANTIATE_TEST_SUITE_P(
    Devices, GraphIteratorI2ConversionTest,
    ::testing::Values(ConversionTestParam{"GPU", ov::element::i4,
                                          /*adjusts_zero_points=*/false},
                      ConversionTestParam{"NPU", ov::element::u2,
                                          /*adjusts_zero_points=*/true}));

}  // namespace
}  // namespace litert::openvino
