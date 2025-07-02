// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/tensorflow_lite/decoder.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "decoder.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"

namespace litert {
namespace openvino {

using ::testing::Values;

TEST(TestLiteOvDecoder, ConstructDecoderOp) {
  auto model =
      testing::LoadTestFileModel("simple_conv_2d_fused_relu_op.tflite");
  auto graph = model.Subgraph(0);
  size_t index = 0;
  for (const auto& op : graph->Ops()) {
    auto sample_ov_decode_op = DecoderOperation(
        /*input_tensor_info=*/std::vector<
            ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*output_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*litert_op=*/op, index++);
  }
}

TEST(TestLiteOvDecoder, VerifyDecoderConv2dOp) {
  auto model =
      testing::LoadTestFileModel("simple_conv_2d_fused_relu_op.tflite");
  auto graph = model.Subgraph(0);
  size_t index = 0;
  for (const auto& op : graph->Ops()) {
    auto sample_ov_decode_op = DecoderOperation(
        /*input_tensor_info=*/std::vector<
            ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*output_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*litert_op=*/op, index++);
    ASSERT_EQ(sample_ov_decode_op.get_input_size(), 0);
    std::vector<int64_t> strides_vec =
        sample_ov_decode_op.get_attribute("strides").as<std::vector<int64_t>>();
    LITERT_LOG(LITERT_INFO, "Stride values : %ld %ld %ld %ld", strides_vec[0],
               strides_vec[1], strides_vec[2], strides_vec[3]);
    ASSERT_EQ(strides_vec, std::vector<int64_t>({1, 1, 1, 1}));
    std::string padding_str =
        sample_ov_decode_op.get_attribute("padding").as<std::string>();
    LITERT_LOG(LITERT_INFO, "Padding : %s", padding_str.c_str());
    ASSERT_EQ(padding_str, "SAME");
    std::vector<int64_t> dilations_vec =
        sample_ov_decode_op.get_attribute("dilations")
            .as<std::vector<int64_t>>();
    LITERT_LOG(LITERT_INFO, "Dilation values : %ld %ld %ld %ld",
               dilations_vec[0], dilations_vec[1], dilations_vec[2],
               dilations_vec[3]);
    ASSERT_EQ(dilations_vec, std::vector<int64_t>({1, 1, 1, 1}));
    std::string activation_str =
        sample_ov_decode_op.get_attribute("activation").as<std::string>();
    LITERT_LOG(LITERT_INFO, "Activation : %s", activation_str.c_str());
    ASSERT_EQ(activation_str, "RELU");
  }
}

}  // namespace openvino
}  // namespace litert
