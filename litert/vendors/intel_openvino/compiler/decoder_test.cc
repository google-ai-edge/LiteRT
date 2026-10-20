// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/vendors/intel_openvino/compiler/decoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"
#include "litert/vendors/intel_openvino/compiler/graph_iterator.h"

namespace litert {
namespace openvino {

using ::testing::Values;

TEST(TestLiteOvDecoder, ConstructDecoderOp) {
  auto cc_model =
      testing::LoadTestFileModel("simple_conv_2d_fused_relu_op.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto graph_or = model.Subgraph(0);
  ASSERT_TRUE(graph_or.HasValue());
  auto graph = graph_or.Value();
  size_t index = 0;
  for (const auto& op : graph.Ops()) {
    auto sample_ov_decode_op = DecoderOperation(
        /*ctx=*/ctx,
        /*input_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*output_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*litert_op=*/op, index++);
  }
}

TEST(TestLiteOvDecoder, VerifyDecoderConv2dOp) {
  auto cc_model =
      testing::LoadTestFileModel("simple_conv_2d_fused_relu_op.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto graph_or = model.Subgraph(0);
  ASSERT_TRUE(graph_or.HasValue());
  auto graph = graph_or.Value();
  size_t index = 0;
  for (const auto& op : graph.Ops()) {
    auto sample_ov_decode_op = DecoderOperation(
        /*ctx=*/ctx,
        /*input_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
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

TEST(TestLiteOvDecoder, VerifyDecoderTileOp) {
  auto cc_model = testing::LoadTestFileModel("simple_tile_op.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto graph_or = model.Subgraph(0);
  ASSERT_TRUE(graph_or.HasValue());
  auto graph = graph_or.Value();
  size_t index = 0;
  bool saw_tile = false;
  for (const auto& op : graph.Ops()) {
    auto sample_ov_decode_op = DecoderOperation(
        /*ctx=*/ctx,
        /*input_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*output_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*litert_op=*/op, index++);
    if (op.Code() != kLiteRtOpCodeTflTile) {
      continue;
    }
    saw_tile = true;
    LITERT_LOG(LITERT_INFO, "op_type : %s",
               sample_ov_decode_op.get_op_type().c_str());
    ASSERT_EQ(sample_ov_decode_op.get_op_type(), "TILE");
    // TILE has no op options in TFLite (the `multiples` vector is passed as an
    // input tensor, not an attribute). get_attribute for an unsupported name
    // should return an empty ov::Any rather than crash.
    auto multiples_attr = sample_ov_decode_op.get_attribute("multiples");
    ASSERT_TRUE(multiples_attr.empty());
  }
  ASSERT_TRUE(saw_tile) << "simple_tile_op.tflite did not contain a TILE op";
}

TEST(TestLiteOvDecoder, VerifyDecoderReduceAnyOp) {
  auto cc_model = testing::LoadTestFileModel("simple_reduceany_op.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto graph_or = model.Subgraph(0);
  ASSERT_TRUE(graph_or.HasValue());
  auto graph = graph_or.Value();
  size_t index = 0;
  bool saw_reduce_any = false;
  for (const auto& op : graph.Ops()) {
    auto sample_ov_decode_op = DecoderOperation(
        /*ctx=*/ctx,
        /*input_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*output_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*litert_op=*/op, index++);
    if (op.Code() != kLiteRtOpCodeTflReduceAny) {
      continue;
    }
    saw_reduce_any = true;
    ASSERT_EQ(sample_ov_decode_op.get_op_type(), "REDUCE_ANY");
    auto keep_dims_attr = sample_ov_decode_op.get_attribute("keep_dims");
    ASSERT_FALSE(keep_dims_attr.empty());
    EXPECT_FALSE(keep_dims_attr.as<bool>());
    EXPECT_TRUE(sample_ov_decode_op.get_attribute("unknown_attr").empty());
  }
  ASSERT_TRUE(saw_reduce_any)
      << "simple_reduceany_op.tflite did not contain a REDUCE_ANY op";
}

TEST(TestLiteOvDecoder, VerifyDecoderReduceAllOp) {
  auto cc_model = testing::LoadTestFileModel("simple_reduceall_op.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto graph_or = model.Subgraph(0);
  ASSERT_TRUE(graph_or.HasValue());
  auto graph = graph_or.Value();
  size_t index = 0;
  bool saw_reduce_all = false;
  for (const auto& op : graph.Ops()) {
    auto sample_ov_decode_op = DecoderOperation(
        /*ctx=*/ctx,
        /*input_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*output_tensor_info=*/
        std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>(),
        /*litert_op=*/op, index++);
    if (op.Code() != kLiteRtOpCodeTflReduceAll) {
      continue;
    }
    saw_reduce_all = true;
    ASSERT_EQ(sample_ov_decode_op.get_op_type(), "REDUCE_ALL");
    auto keep_dims_attr = sample_ov_decode_op.get_attribute("keep_dims");
    ASSERT_FALSE(keep_dims_attr.empty());
    EXPECT_FALSE(keep_dims_attr.as<bool>());
    EXPECT_TRUE(sample_ov_decode_op.get_attribute("unknown_attr").empty());
  }
  ASSERT_TRUE(saw_reduce_all)
      << "simple_reduceall_op.tflite did not contain a REDUCE_ALL op";
}

TEST(TestLiteOvDecoder, ConvertReduceAnyModelThroughFrontEnd) {
  auto cc_model = testing::LoadTestFileModel("simple_reduceany_op.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto subgraph = model.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());

  auto fe = std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
  std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> delegate =
      std::make_shared<litert::openvino::GraphIteratorDelegate>(
          ctx, &subgraph.Value());
  auto ov_model = fe->convert(fe->load(delegate));
  ASSERT_NE(ov_model, nullptr);
  bool saw_reduce_logical_or = false;
  for (const auto& node : ov_model->get_ordered_ops()) {
    if (std::string(node->get_type_name()) == "ReduceLogicalOr") {
      saw_reduce_logical_or = true;
      break;
    }
  }
  EXPECT_TRUE(saw_reduce_logical_or);
}

}  // namespace openvino
}  // namespace litert
