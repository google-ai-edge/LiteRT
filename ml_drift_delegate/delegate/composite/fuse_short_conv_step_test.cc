// Copyright 2026 Google LLC.
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

#include "ml_drift_delegate/delegate/composite/fuse_short_conv_step.h"

#include <any>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/short_conv_step_parser.h"

namespace litert::ml_drift {
namespace {

using ::testing::Eq;
using ::testing::NotNull;

TEST(FuseShortConvStepTest, FusesGatedShortConvStepSuccessfully) {
  ::ml_drift::GraphFloat32 graph;

  constexpr int kHiddenSize = 2048;
  constexpr int kStateCacheSize = 2;
  constexpr int kFilterSize = 3;

  // in_proj [1, 1, 1, 3 * kHiddenSize]
  auto* in_proj_input = graph.NewValue();
  in_proj_input->tensor.type = ::ml_drift::DataType::FLOAT32;
  in_proj_input->tensor.shape = ::ml_drift::BHWC(1, 1, 1, 3 * kHiddenSize);

  // conv_state [1, 1, kHiddenSize, kStateCacheSize]
  auto* conv_state_input = graph.NewValue();
  conv_state_input->tensor.type = ::ml_drift::DataType::FLOAT32;
  conv_state_input->tensor.shape =
      ::ml_drift::BHWC(1, 1, kHiddenSize, kStateCacheSize);

  // Dummy producers
  auto* dummy_in = graph.NewNode();
  dummy_in->operation.type = "dummy_in";
  graph.SetProducer(dummy_in->id, in_proj_input->id);

  auto* dummy_state = graph.NewNode();
  dummy_state->operation.type = "dummy_state";
  graph.SetProducer(dummy_state->id, conv_state_input->id);

  // Slices: b, c, x
  auto* slice_b_node = graph.NewNode();
  slice_b_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  auto* b_val = graph.NewValue();
  b_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  b_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(slice_b_node->id, in_proj_input->id);
  graph.SetProducer(slice_b_node->id, b_val->id);

  auto* slice_c_node = graph.NewNode();
  slice_c_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  auto* c_val = graph.NewValue();
  c_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  c_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(slice_c_node->id, in_proj_input->id);
  graph.SetProducer(slice_c_node->id, c_val->id);

  auto* slice_x_node = graph.NewNode();
  slice_x_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  auto* x_val = graph.NewValue();
  x_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  x_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(slice_x_node->id, in_proj_input->id);
  graph.SetProducer(slice_x_node->id, x_val->id);

  // bx_mul: b * x
  auto* bx_mul_node = graph.NewNode();
  bx_mul_node->operation.type = ToString(::ml_drift::OperationType::MUL);
  auto* bx_val = graph.NewValue();
  bx_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  bx_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(bx_mul_node->id, b_val->id);
  graph.AddConsumer(bx_mul_node->id, x_val->id);
  graph.SetProducer(bx_mul_node->id, bx_val->id);

  // px_reshape: [1, 1, 1, kHiddenSize] -> [1, 1, kHiddenSize, 1]
  auto* px_reshape_node = graph.NewNode();
  px_reshape_node->operation.type =
      ToString(::ml_drift::OperationType::RESHAPE);
  auto* px_reshaped_val = graph.NewValue();
  px_reshaped_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  px_reshaped_val->tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, 1);
  graph.AddConsumer(px_reshape_node->id, bx_val->id);
  graph.SetProducer(px_reshape_node->id, px_reshaped_val->id);

  // concat_win: conv_state + px_reshaped -> window [1, 1, kH, kFilterSize]
  auto* concat_win_node = graph.NewNode();
  concat_win_node->operation.type =
      ToString(::ml_drift::OperationType::CONCAT);
  auto* win_val = graph.NewValue();
  win_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  win_val->tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  graph.AddConsumer(concat_win_node->id, conv_state_input->id);
  graph.AddConsumer(concat_win_node->id, px_reshaped_val->id);
  graph.SetProducer(concat_win_node->id, win_val->id);

  // slice_state: window -> next_state [1, 1, kHiddenSize, kStateCacheSize]
  auto* slice_state_node = graph.NewNode();
  slice_state_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  ::ml_drift::SliceAttributes state_slice_attr;
  state_slice_attr.starts = ::ml_drift::BHWC(0, 0, 0, 1);
  state_slice_attr.ends = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  slice_state_node->operation.attributes = state_slice_attr;
  auto* next_state_val = graph.NewValue();
  next_state_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  next_state_val->tensor.shape =
      ::ml_drift::BHWC(1, 1, kHiddenSize, kStateCacheSize);
  graph.AddConsumer(slice_state_node->id, win_val->id);
  graph.SetProducer(slice_state_node->id, next_state_val->id);

  // mul_conv with constant weight
  auto* mul_conv_node = graph.NewNode();
  mul_conv_node->operation.type = ToString(::ml_drift::OperationType::MUL);
  ::ml_drift::ElementwiseAttributes mul_attr;
  ::ml_drift::TensorFloat32 weight_tensor;
  weight_tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  weight_tensor.data.resize(kHiddenSize * kFilterSize, 1.0f);
  mul_attr.param = std::move(weight_tensor);
  mul_conv_node->operation.attributes = std::move(mul_attr);
  auto* mul_conv_out = graph.NewValue();
  mul_conv_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  mul_conv_out->tensor.shape =
      ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  graph.AddConsumer(mul_conv_node->id, win_val->id);
  graph.SetProducer(mul_conv_node->id, mul_conv_out->id);

  // reduce_sum: window sum [1, 1, kHiddenSize, 1]
  auto* reduce_node = graph.NewNode();
  reduce_node->operation.type = ToString(::ml_drift::OperationType::REDUCE_SUM);
  auto* reduce_out = graph.NewValue();
  reduce_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  reduce_out->tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, 1);
  graph.AddConsumer(reduce_node->id, mul_conv_out->id);
  graph.SetProducer(reduce_node->id, reduce_out->id);

  // red_reshape: [1, 1, kHiddenSize, 1] -> [1, 1, 1, kHiddenSize]
  auto* red_reshape_node = graph.NewNode();
  red_reshape_node->operation.type =
      ToString(::ml_drift::OperationType::RESHAPE);
  auto* red_reshaped_val = graph.NewValue();
  red_reshaped_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  red_reshaped_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(red_reshape_node->id, reduce_out->id);
  graph.SetProducer(red_reshape_node->id, red_reshaped_val->id);

  // gating_mul: red_reshaped * c_val -> final_out [1, 1, 1, kHiddenSize]
  auto* gating_mul_node = graph.NewNode();
  gating_mul_node->operation.type = ToString(::ml_drift::OperationType::MUL);
  auto* final_out_val = graph.NewValue();
  final_out_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  final_out_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(gating_mul_node->id, red_reshaped_val->id);
  graph.AddConsumer(gating_mul_node->id, c_val->id);
  graph.SetProducer(gating_mul_node->id, final_out_val->id);

  // Consumers for outputs
  auto* dummy_consumer = graph.NewNode();
  dummy_consumer->operation.type = "dummy_consumer";
  graph.AddConsumer(dummy_consumer->id, final_out_val->id);
  graph.AddConsumer(dummy_consumer->id, next_state_val->id);

  // Run fusion
  EXPECT_TRUE(FuseShortConvStep(&graph).ok());

  // Verify fused node
  ::ml_drift::Node* fused_node = nullptr;
  for (::ml_drift::Node* node : graph.nodes()) {
    if (node && node->operation.type == kShortConvStepType) {
      fused_node = node;
      break;
    }
  }
  ASSERT_THAT(fused_node, NotNull());

  // Check attributes
  ASSERT_TRUE(fused_node->operation.attributes.has_value());
  const auto& attr = std::any_cast<const ShortConvStepAttributes&>(
      fused_node->operation.attributes);
  EXPECT_THAT(attr.conv_L_cache, Eq(kFilterSize));

  // Check inputs: in_proj, conv_state, conv_weight
  auto fused_inputs = graph.FindInputs(fused_node->id);
  ASSERT_THAT(fused_inputs.size(), Eq(3));
  EXPECT_THAT(fused_inputs[0]->id, Eq(in_proj_input->id));
  EXPECT_THAT(fused_inputs[1]->id, Eq(conv_state_input->id));

  // Check outputs: final_out, next_state
  auto fused_outputs = graph.FindOutputs(fused_node->id);
  ASSERT_THAT(fused_outputs.size(), Eq(2));
  EXPECT_THAT(fused_outputs[0]->id, Eq(final_out_val->id));
  EXPECT_THAT(fused_outputs[1]->id, Eq(next_state_val->id));

  // Check deleted intermediate nodes
  EXPECT_THAT(graph.GetNode(slice_b_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(slice_c_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(slice_x_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(bx_mul_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(px_reshape_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(concat_win_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(slice_state_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(mul_conv_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(reduce_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(red_reshape_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(gating_mul_node->id), Eq(nullptr));
}

TEST(FuseShortConvStepTest, FusesGatedShortConvStepWithBiasSuccessfully) {
  ::ml_drift::GraphFloat32 graph;

  constexpr int kHiddenSize = 2048;
  constexpr int kStateCacheSize = 2;
  constexpr int kFilterSize = 3;

  // in_proj [1, 1, 1, 3 * kHiddenSize]
  auto* in_proj_input = graph.NewValue();
  in_proj_input->tensor.type = ::ml_drift::DataType::FLOAT32;
  in_proj_input->tensor.shape = ::ml_drift::BHWC(1, 1, 1, 3 * kHiddenSize);

  // conv_state [1, 1, kHiddenSize, kStateCacheSize]
  auto* conv_state_input = graph.NewValue();
  conv_state_input->tensor.type = ::ml_drift::DataType::FLOAT32;
  conv_state_input->tensor.shape =
      ::ml_drift::BHWC(1, 1, kHiddenSize, kStateCacheSize);

  // conv_bias [1, 1, 1, kHiddenSize]
  auto* conv_bias_input = graph.NewValue();
  conv_bias_input->tensor.type = ::ml_drift::DataType::FLOAT32;
  conv_bias_input->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);

  // Dummy producers
  auto* dummy_in = graph.NewNode();
  dummy_in->operation.type = "dummy_in";
  graph.SetProducer(dummy_in->id, in_proj_input->id);

  auto* dummy_state = graph.NewNode();
  dummy_state->operation.type = "dummy_state";
  graph.SetProducer(dummy_state->id, conv_state_input->id);

  auto* dummy_bias = graph.NewNode();
  dummy_bias->operation.type = "dummy_bias";
  graph.SetProducer(dummy_bias->id, conv_bias_input->id);

  // Slices: b, c, x
  auto* slice_b_node = graph.NewNode();
  slice_b_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  auto* b_val = graph.NewValue();
  b_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  b_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(slice_b_node->id, in_proj_input->id);
  graph.SetProducer(slice_b_node->id, b_val->id);

  auto* slice_c_node = graph.NewNode();
  slice_c_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  auto* c_val = graph.NewValue();
  c_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  c_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(slice_c_node->id, in_proj_input->id);
  graph.SetProducer(slice_c_node->id, c_val->id);

  auto* slice_x_node = graph.NewNode();
  slice_x_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  auto* x_val = graph.NewValue();
  x_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  x_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(slice_x_node->id, in_proj_input->id);
  graph.SetProducer(slice_x_node->id, x_val->id);

  // bx_mul: b * x
  auto* bx_mul_node = graph.NewNode();
  bx_mul_node->operation.type = ToString(::ml_drift::OperationType::MUL);
  auto* bx_val = graph.NewValue();
  bx_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  bx_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(bx_mul_node->id, b_val->id);
  graph.AddConsumer(bx_mul_node->id, x_val->id);
  graph.SetProducer(bx_mul_node->id, bx_val->id);

  // px_reshape: [1, 1, 1, kHiddenSize] -> [1, 1, kHiddenSize, 1]
  auto* px_reshape_node = graph.NewNode();
  px_reshape_node->operation.type =
      ToString(::ml_drift::OperationType::RESHAPE);
  auto* px_reshaped_val = graph.NewValue();
  px_reshaped_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  px_reshaped_val->tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, 1);
  graph.AddConsumer(px_reshape_node->id, bx_val->id);
  graph.SetProducer(px_reshape_node->id, px_reshaped_val->id);

  // concat_win: conv_state + px_reshaped -> window [1, 1, kH, kFilterSize]
  auto* concat_win_node = graph.NewNode();
  concat_win_node->operation.type =
      ToString(::ml_drift::OperationType::CONCAT);
  auto* win_val = graph.NewValue();
  win_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  win_val->tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  graph.AddConsumer(concat_win_node->id, conv_state_input->id);
  graph.AddConsumer(concat_win_node->id, px_reshaped_val->id);
  graph.SetProducer(concat_win_node->id, win_val->id);

  // slice_state: window -> next_state [1, 1, kHiddenSize, kStateCacheSize]
  auto* slice_state_node = graph.NewNode();
  slice_state_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  ::ml_drift::SliceAttributes state_slice_attr;
  state_slice_attr.starts = ::ml_drift::BHWC(0, 0, 0, 1);
  state_slice_attr.ends = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  slice_state_node->operation.attributes = state_slice_attr;
  auto* next_state_val = graph.NewValue();
  next_state_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  next_state_val->tensor.shape =
      ::ml_drift::BHWC(1, 1, kHiddenSize, kStateCacheSize);
  graph.AddConsumer(slice_state_node->id, win_val->id);
  graph.SetProducer(slice_state_node->id, next_state_val->id);

  // mul_conv with constant weight
  auto* mul_conv_node = graph.NewNode();
  mul_conv_node->operation.type = ToString(::ml_drift::OperationType::MUL);
  ::ml_drift::ElementwiseAttributes mul_attr;
  ::ml_drift::TensorFloat32 weight_tensor;
  weight_tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  weight_tensor.data.resize(kHiddenSize * kFilterSize, 1.0f);
  mul_attr.param = std::move(weight_tensor);
  mul_conv_node->operation.attributes = std::move(mul_attr);
  auto* mul_conv_out = graph.NewValue();
  mul_conv_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  mul_conv_out->tensor.shape =
      ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  graph.AddConsumer(mul_conv_node->id, win_val->id);
  graph.SetProducer(mul_conv_node->id, mul_conv_out->id);

  // reduce_sum: window sum [1, 1, kHiddenSize, 1]
  auto* reduce_node = graph.NewNode();
  reduce_node->operation.type = ToString(::ml_drift::OperationType::REDUCE_SUM);
  auto* reduce_out = graph.NewValue();
  reduce_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  reduce_out->tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, 1);
  graph.AddConsumer(reduce_node->id, mul_conv_out->id);
  graph.SetProducer(reduce_node->id, reduce_out->id);

  // red_reshape: [1, 1, kHiddenSize, 1] -> [1, 1, 1, kHiddenSize]
  auto* red_reshape_node = graph.NewNode();
  red_reshape_node->operation.type =
      ToString(::ml_drift::OperationType::RESHAPE);
  auto* red_reshaped_val = graph.NewValue();
  red_reshaped_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  red_reshaped_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(red_reshape_node->id, reduce_out->id);
  graph.SetProducer(red_reshape_node->id, red_reshaped_val->id);

  // bias_add: red_reshaped + bias -> biased_out [1, 1, 1, kHiddenSize]
  auto* bias_add_node = graph.NewNode();
  bias_add_node->operation.type = ToString(::ml_drift::OperationType::ADD);
  auto* biased_out_val = graph.NewValue();
  biased_out_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  biased_out_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(bias_add_node->id, red_reshaped_val->id);
  graph.AddConsumer(bias_add_node->id, conv_bias_input->id);
  graph.SetProducer(bias_add_node->id, biased_out_val->id);

  // gating_mul: biased_out * c_val -> final_out [1, 1, 1, kHiddenSize]
  auto* gating_mul_node = graph.NewNode();
  gating_mul_node->operation.type = ToString(::ml_drift::OperationType::MUL);
  auto* final_out_val = graph.NewValue();
  final_out_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  final_out_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(gating_mul_node->id, biased_out_val->id);
  graph.AddConsumer(gating_mul_node->id, c_val->id);
  graph.SetProducer(gating_mul_node->id, final_out_val->id);

  // Consumers for outputs
  auto* dummy_consumer = graph.NewNode();
  dummy_consumer->operation.type = "dummy_consumer";
  graph.AddConsumer(dummy_consumer->id, final_out_val->id);
  graph.AddConsumer(dummy_consumer->id, next_state_val->id);

  // Run fusion
  EXPECT_TRUE(FuseShortConvStep(&graph).ok());

  // Verify fused node
  ::ml_drift::Node* fused_node = nullptr;
  for (::ml_drift::Node* node : graph.nodes()) {
    if (node && node->operation.type == kShortConvStepType) {
      fused_node = node;
      break;
    }
  }
  ASSERT_THAT(fused_node, NotNull());

  // Check attributes
  ASSERT_TRUE(fused_node->operation.attributes.has_value());
  const auto& attr = std::any_cast<const ShortConvStepAttributes&>(
      fused_node->operation.attributes);
  EXPECT_THAT(attr.conv_L_cache, Eq(kFilterSize));

  // Check inputs: in_proj, conv_state, conv_weight, conv_bias (4 inputs)
  auto fused_inputs = graph.FindInputs(fused_node->id);
  ASSERT_THAT(fused_inputs.size(), Eq(4));
  EXPECT_THAT(fused_inputs[0]->id, Eq(in_proj_input->id));
  EXPECT_THAT(fused_inputs[1]->id, Eq(conv_state_input->id));
  EXPECT_THAT(fused_inputs[3]->id, Eq(conv_bias_input->id));

  // Check outputs: final_out, next_state
  auto fused_outputs = graph.FindOutputs(fused_node->id);
  ASSERT_THAT(fused_outputs.size(), Eq(2));
  EXPECT_THAT(fused_outputs[0]->id, Eq(final_out_val->id));
  EXPECT_THAT(fused_outputs[1]->id, Eq(next_state_val->id));

  // Check deleted intermediate nodes including bias_add_node
  EXPECT_THAT(graph.GetNode(slice_b_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(slice_c_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(slice_x_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(bx_mul_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(px_reshape_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(concat_win_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(slice_state_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(mul_conv_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(reduce_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(red_reshape_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(bias_add_node->id), Eq(nullptr));
  EXPECT_THAT(graph.GetNode(gating_mul_node->id), Eq(nullptr));
}

TEST(FuseShortConvStepTest, DoesNotFuseWhenChannelsNotMultipleOfFour) {
  ::ml_drift::GraphFloat32 graph;

  constexpr int kHiddenSize = 7;  // Not divisible by 4
  constexpr int kStateCacheSize = 2;
  constexpr int kFilterSize = 3;

  auto* in_proj_input = graph.NewValue();
  in_proj_input->tensor.type = ::ml_drift::DataType::FLOAT32;
  in_proj_input->tensor.shape = ::ml_drift::BHWC(1, 1, 1, 3 * kHiddenSize);

  auto* conv_state_input = graph.NewValue();
  conv_state_input->tensor.type = ::ml_drift::DataType::FLOAT32;
  conv_state_input->tensor.shape =
      ::ml_drift::BHWC(1, 1, kHiddenSize, kStateCacheSize);

  auto* dummy_in = graph.NewNode();
  dummy_in->operation.type = "dummy_in";
  graph.SetProducer(dummy_in->id, in_proj_input->id);

  auto* dummy_state = graph.NewNode();
  dummy_state->operation.type = "dummy_state";
  graph.SetProducer(dummy_state->id, conv_state_input->id);

  auto* slice_b_node = graph.NewNode();
  slice_b_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  auto* b_val = graph.NewValue();
  b_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  b_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(slice_b_node->id, in_proj_input->id);
  graph.SetProducer(slice_b_node->id, b_val->id);

  auto* slice_c_node = graph.NewNode();
  slice_c_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  auto* c_val = graph.NewValue();
  c_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  c_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(slice_c_node->id, in_proj_input->id);
  graph.SetProducer(slice_c_node->id, c_val->id);

  auto* slice_x_node = graph.NewNode();
  slice_x_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  auto* x_val = graph.NewValue();
  x_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  x_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(slice_x_node->id, in_proj_input->id);
  graph.SetProducer(slice_x_node->id, x_val->id);

  auto* bx_mul_node = graph.NewNode();
  bx_mul_node->operation.type = ToString(::ml_drift::OperationType::MUL);
  auto* bx_val = graph.NewValue();
  bx_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  bx_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(bx_mul_node->id, b_val->id);
  graph.AddConsumer(bx_mul_node->id, x_val->id);
  graph.SetProducer(bx_mul_node->id, bx_val->id);

  auto* px_reshape_node = graph.NewNode();
  px_reshape_node->operation.type =
      ToString(::ml_drift::OperationType::RESHAPE);
  auto* px_reshaped_val = graph.NewValue();
  px_reshaped_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  px_reshaped_val->tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, 1);
  graph.AddConsumer(px_reshape_node->id, bx_val->id);
  graph.SetProducer(px_reshape_node->id, px_reshaped_val->id);

  auto* concat_win_node = graph.NewNode();
  concat_win_node->operation.type =
      ToString(::ml_drift::OperationType::CONCAT);
  auto* win_val = graph.NewValue();
  win_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  win_val->tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  graph.AddConsumer(concat_win_node->id, conv_state_input->id);
  graph.AddConsumer(concat_win_node->id, px_reshaped_val->id);
  graph.SetProducer(concat_win_node->id, win_val->id);

  auto* slice_state_node = graph.NewNode();
  slice_state_node->operation.type = ToString(::ml_drift::OperationType::SLICE);
  ::ml_drift::SliceAttributes state_slice_attr;
  state_slice_attr.starts = ::ml_drift::BHWC(0, 0, 0, 1);
  state_slice_attr.ends = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  slice_state_node->operation.attributes = state_slice_attr;
  auto* next_state_val = graph.NewValue();
  next_state_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  next_state_val->tensor.shape =
      ::ml_drift::BHWC(1, 1, kHiddenSize, kStateCacheSize);
  graph.AddConsumer(slice_state_node->id, win_val->id);
  graph.SetProducer(slice_state_node->id, next_state_val->id);

  auto* mul_conv_node = graph.NewNode();
  mul_conv_node->operation.type = ToString(::ml_drift::OperationType::MUL);
  ::ml_drift::ElementwiseAttributes mul_attr;
  ::ml_drift::TensorFloat32 weight_tensor;
  weight_tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  weight_tensor.data.resize(kHiddenSize * kFilterSize, 1.0f);
  mul_attr.param = std::move(weight_tensor);
  mul_conv_node->operation.attributes = std::move(mul_attr);
  auto* mul_conv_out = graph.NewValue();
  mul_conv_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  mul_conv_out->tensor.shape =
      ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  graph.AddConsumer(mul_conv_node->id, win_val->id);
  graph.SetProducer(mul_conv_node->id, mul_conv_out->id);

  auto* reduce_node = graph.NewNode();
  reduce_node->operation.type = ToString(::ml_drift::OperationType::REDUCE_SUM);
  auto* reduce_out = graph.NewValue();
  reduce_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  reduce_out->tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, 1);
  graph.AddConsumer(reduce_node->id, mul_conv_out->id);
  graph.SetProducer(reduce_node->id, reduce_out->id);

  auto* red_reshape_node = graph.NewNode();
  red_reshape_node->operation.type =
      ToString(::ml_drift::OperationType::RESHAPE);
  auto* red_reshaped_val = graph.NewValue();
  red_reshaped_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  red_reshaped_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(red_reshape_node->id, reduce_out->id);
  graph.SetProducer(red_reshape_node->id, red_reshaped_val->id);

  auto* gating_mul_node = graph.NewNode();
  gating_mul_node->operation.type = ToString(::ml_drift::OperationType::MUL);
  auto* final_out_val = graph.NewValue();
  final_out_val->tensor.type = ::ml_drift::DataType::FLOAT32;
  final_out_val->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kHiddenSize);
  graph.AddConsumer(gating_mul_node->id, red_reshaped_val->id);
  graph.AddConsumer(gating_mul_node->id, c_val->id);
  graph.SetProducer(gating_mul_node->id, final_out_val->id);

  auto* dummy_consumer = graph.NewNode();
  dummy_consumer->operation.type = "dummy_consumer";
  graph.AddConsumer(dummy_consumer->id, final_out_val->id);
  graph.AddConsumer(dummy_consumer->id, next_state_val->id);

  EXPECT_TRUE(FuseShortConvStep(&graph).ok());

  // Verify NO fused node was created
  for (::ml_drift::Node* node : graph.nodes()) {
    if (node) {
      EXPECT_NE(node->operation.type, kShortConvStepType);
    }
  }
}

TEST(FuseShortConvStepIrModelTest, FusesGatedShortConvStepSuccessfully) {
  ::ml_drift::ir::IrModel model;

  constexpr int kHiddenSize = 2048;
  constexpr int kStateCacheSize = 2;
  constexpr int kFilterSize = 3;

  // in_proj [1, 1, 1, 3 * kHiddenSize]
  auto* in_proj_input = model.add_tensor(
      ::ml_drift::DataType::FLOAT32,
      ::ml_drift::BHWC(1, 1, 1, 3 * kHiddenSize));
  auto* conv_state_input = model.add_tensor(
      ::ml_drift::DataType::FLOAT32,
      ::ml_drift::BHWC(1, 1, kHiddenSize, kStateCacheSize));
  model.add_input(in_proj_input->id);
  model.add_input(conv_state_input->id);

  auto* dummy_in = model.add_op();
  dummy_in->name = "dummy_in";
  model.SetProducer(in_proj_input->id, dummy_in->id);

  auto* dummy_state = model.add_op();
  dummy_state->name = "dummy_state";
  model.SetProducer(conv_state_input->id, dummy_state->id);

  // Slices: b, c, x
  auto* slice_b = model.add_op();
  slice_b->name = ToString(::ml_drift::OperationType::SLICE);
  auto* b_val = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                 ::ml_drift::BHWC(1, 1, 1, kHiddenSize));
  model.AddConsumer(in_proj_input->id, slice_b->id);
  model.SetProducer(b_val->id, slice_b->id);

  auto* slice_c = model.add_op();
  slice_c->name = ToString(::ml_drift::OperationType::SLICE);
  auto* c_val = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                 ::ml_drift::BHWC(1, 1, 1, kHiddenSize));
  model.AddConsumer(in_proj_input->id, slice_c->id);
  model.SetProducer(c_val->id, slice_c->id);

  auto* slice_x = model.add_op();
  slice_x->name = ToString(::ml_drift::OperationType::SLICE);
  auto* x_val = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                 ::ml_drift::BHWC(1, 1, 1, kHiddenSize));
  model.AddConsumer(in_proj_input->id, slice_x->id);
  model.SetProducer(x_val->id, slice_x->id);

  // bx_mul: b * x
  auto* bx_mul = model.add_op();
  bx_mul->name = ToString(::ml_drift::OperationType::MUL);
  auto* bx_val = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                  ::ml_drift::BHWC(1, 1, 1, kHiddenSize));
  model.AddConsumer(b_val->id, bx_mul->id);
  model.AddConsumer(x_val->id, bx_mul->id);
  model.SetProducer(bx_val->id, bx_mul->id);

  // px_reshape: [1, 1, 1, kHiddenSize] -> [1, 1, kHiddenSize, 1]
  auto* px_reshape = model.add_op();
  px_reshape->name = ToString(::ml_drift::OperationType::RESHAPE);
  auto* px_val = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                  ::ml_drift::BHWC(1, 1, kHiddenSize, 1));
  model.AddConsumer(bx_val->id, px_reshape->id);
  model.SetProducer(px_val->id, px_reshape->id);

  // concat: conv_state + px
  auto* concat_win = model.add_op();
  concat_win->name = ToString(::ml_drift::OperationType::CONCAT);
  auto* win_val = model.add_tensor(
      ::ml_drift::DataType::FLOAT32,
      ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize));
  model.AddConsumer(conv_state_input->id, concat_win->id);
  model.AddConsumer(px_val->id, concat_win->id);
  model.SetProducer(win_val->id, concat_win->id);

  // slice_state
  auto* slice_state = model.add_op();
  slice_state->name = ToString(::ml_drift::OperationType::SLICE);
  ::ml_drift::SliceAttributes slice_state_attr;
  slice_state_attr.starts = ::ml_drift::BHWC(0, 0, 0, 1);
  slice_state_attr.ends = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  slice_state->attr = slice_state_attr;
  auto* next_state_val = model.add_tensor(
      ::ml_drift::DataType::FLOAT32,
      ::ml_drift::BHWC(1, 1, kHiddenSize, kStateCacheSize));
  model.AddConsumer(win_val->id, slice_state->id);
  model.SetProducer(next_state_val->id, slice_state->id);

  // conv weight tensor
  auto* weight_op = model.add_op();
  weight_op->name = ToString(::ml_drift::OperationType::CONSTANT);
  auto* weight_val = model.add_tensor(
      ::ml_drift::DataType::FLOAT32,
      ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize));
  ::ml_drift::ConstTensorAttributes weight_attr;
  ::ml_drift::TensorFloat32 w_tensor;
  w_tensor.shape = ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize);
  w_tensor.data = std::vector<float>(kHiddenSize * kFilterSize, 1.0f);
  weight_attr.tensor = std::move(w_tensor);
  weight_op->attr = std::move(weight_attr);
  model.SetProducer(weight_val->id, weight_op->id);

  // mul_conv
  auto* mul_conv = model.add_op();
  mul_conv->name = ToString(::ml_drift::OperationType::MUL);
  auto* mul_conv_out = model.add_tensor(
      ::ml_drift::DataType::FLOAT32,
      ::ml_drift::BHWC(1, 1, kHiddenSize, kFilterSize));
  model.AddConsumer(win_val->id, mul_conv->id);
  model.AddConsumer(weight_val->id, mul_conv->id);
  model.SetProducer(mul_conv_out->id, mul_conv->id);

  // reduce_sum
  auto* reduce_sum = model.add_op();
  reduce_sum->name = ToString(::ml_drift::OperationType::REDUCE_SUM);
  auto* reduce_out = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                      ::ml_drift::BHWC(1, 1, kHiddenSize, 1));
  model.AddConsumer(mul_conv_out->id, reduce_sum->id);
  model.SetProducer(reduce_out->id, reduce_sum->id);

  // red_reshape
  auto* red_reshape = model.add_op();
  red_reshape->name = ToString(::ml_drift::OperationType::RESHAPE);
  auto* red_reshaped = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                        ::ml_drift::BHWC(1, 1, 1, kHiddenSize));
  model.AddConsumer(reduce_out->id, red_reshape->id);
  model.SetProducer(red_reshaped->id, red_reshape->id);

  // gating_mul
  auto* gating_mul = model.add_op();
  gating_mul->name = ToString(::ml_drift::OperationType::MUL);
  auto* final_out = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                     ::ml_drift::BHWC(1, 1, 1, kHiddenSize));
  model.AddConsumer(red_reshaped->id, gating_mul->id);
  model.AddConsumer(c_val->id, gating_mul->id);
  model.SetProducer(final_out->id, gating_mul->id);

  // Downstream consumer
  auto* dummy_consumer = model.add_op();
  dummy_consumer->name = "dummy_consumer";
  model.AddConsumer(final_out->id, dummy_consumer->id);
  model.AddConsumer(next_state_val->id, dummy_consumer->id);

  const auto slice_x_id = slice_x->id;
  const auto slice_c_id = slice_c->id;
  const auto reduce_sum_id = reduce_sum->id;
  const auto gating_mul_id = gating_mul->id;

  EXPECT_TRUE(ir::FuseShortConvStep(&model).ok());

  // Verify fused op
  const ::ml_drift::ir::IrOp* fused_op = nullptr;
  for (const auto& op : model.ops()) {
    if (op && op->name == "short_conv_step") {
      fused_op = op.get();
      break;
    }
  }
  ASSERT_THAT(fused_op, NotNull());

  const auto* attr =
      std::any_cast<ShortConvStepAttributes>(&fused_op->attr);
  ASSERT_THAT(attr, NotNull());
  EXPECT_THAT(attr->conv_L_cache, Eq(3));

  EXPECT_THAT(fused_op->inputs.size(), Eq(3));
  EXPECT_THAT(fused_op->inputs[0], Eq(in_proj_input->id));
  EXPECT_THAT(fused_op->inputs[1], Eq(conv_state_input->id));
  EXPECT_THAT(fused_op->outputs.size(), Eq(2));
  EXPECT_THAT(fused_op->outputs[0], Eq(final_out->id));
  EXPECT_THAT(fused_op->outputs[1], Eq(next_state_val->id));

  // Verify old ops are deleted
  EXPECT_EQ(model.op(slice_x_id), nullptr);
  EXPECT_EQ(model.op(slice_c_id), nullptr);
  EXPECT_EQ(model.op(reduce_sum_id), nullptr);
  EXPECT_EQ(model.op(gating_mul_id), nullptr);
}

}  // namespace
}  // namespace litert::ml_drift
