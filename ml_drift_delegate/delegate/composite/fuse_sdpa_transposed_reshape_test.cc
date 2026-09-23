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

#include "ml_drift_delegate/delegate/composite/fuse_sdpa_transposed_reshape.h"

#include "testing/base/public/gunit.h"
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::BHWC;
using ::ml_drift::DataType;
using ::ml_drift::GraphFloat32;
using ::ml_drift::Node;
using ::ml_drift::OperationType;
using ::ml_drift::ReshapeAttributes;
using ::ml_drift::TransposeAttributes;
using ::ml_drift::Value;
using ::ml_drift::ir::IrModel;
using ::ml_drift::ir::IrOp;
using ::ml_drift::ir::IrTensor;

TEST(FuseSdpaTransposedReshapeTest, FusesDirectReshapeOnDecodeGraphFloat32) {
  GraphFloat32 graph;
  Value* q = graph.NewValue();
  q->tensor.type = DataType::FLOAT32;
  q->tensor.shape = BHWC(1, 16, 1, 128);

  Value* k = graph.NewValue();
  k->tensor.type = DataType::FLOAT32;
  k->tensor.shape = BHWC(1, 8, 1024, 128);

  Value* v = graph.NewValue();
  v->tensor.type = DataType::FLOAT32;
  v->tensor.shape = BHWC(1, 8, 128, 1024);

  Value* mid = graph.NewValue();
  mid->tensor.type = DataType::FLOAT32;
  mid->tensor.shape = BHWC(1, 16, 1, 128);

  Value* out = graph.NewValue();
  out->tensor.type = DataType::FLOAT32;
  out->tensor.shape = BHWC(1, 1, 1, 2048);

  Node* sdpa = graph.NewNode();
  sdpa->operation.type = kSdpaTransposedType;
  graph.AddConsumer(sdpa->id, q->id);
  graph.AddConsumer(sdpa->id, k->id);
  graph.AddConsumer(sdpa->id, v->id);
  graph.SetProducer(sdpa->id, mid->id);

  Node* reshape = graph.NewNode();
  reshape->operation.type = ToString(OperationType::RESHAPE);
  ReshapeAttributes r_attr;
  r_attr.new_shape = out->tensor.shape;
  reshape->operation.attributes = r_attr;
  graph.AddConsumer(reshape->id, mid->id);
  graph.SetProducer(reshape->id, out->id);

  ASSERT_TRUE(FuseSdpaTransposedReshape(&graph).ok());
  EXPECT_EQ(graph.nodes().size(), 1u);
  const auto sdpa_outs = graph.FindOutputs(sdpa->id);
  ASSERT_EQ(sdpa_outs.size(), 1u);
  EXPECT_EQ(sdpa_outs[0]->id, out->id);
  EXPECT_EQ(sdpa_outs[0]->tensor.shape, BHWC(1, 1, 1, 2048));
}

TEST(FuseSdpaTransposedReshapeTest, FusesTransposeAndReshapeIrModel) {
  IrModel model;
  IrTensor* q = model.add_tensor(DataType::FLOAT32, BHWC(1, 16, 1, 128));
  IrTensor* k = model.add_tensor(DataType::FLOAT32, BHWC(1, 8, 1024, 128));
  IrTensor* v = model.add_tensor(DataType::FLOAT32, BHWC(1, 8, 128, 1024));
  IrTensor* mid = model.add_tensor(DataType::FLOAT32, BHWC(1, 16, 1, 128));
  IrTensor* perm = model.add_tensor(DataType::FLOAT32, BHWC(1, 1, 16, 128));
  IrTensor* out = model.add_tensor(DataType::FLOAT32, BHWC(1, 1, 1, 2048));
  model.add_input(q->id);
  model.add_input(k->id);
  model.add_input(v->id);
  model.add_output(out->id);

  IrOp* sdpa = model.add_op();
  sdpa->name = kSdpaTransposedType;
  model.AddConsumer(q->id, sdpa->id);
  model.AddConsumer(k->id, sdpa->id);
  model.AddConsumer(v->id, sdpa->id);
  model.SetProducer(mid->id, sdpa->id);

  IrOp* tr = model.add_op();
  tr->name = ToString(OperationType::TRANSPOSE);
  TransposeAttributes t_attr;
  t_attr.perm = BHWC(0, 2, 1, 3);
  tr->attr = t_attr;
  model.AddConsumer(mid->id, tr->id);
  model.SetProducer(perm->id, tr->id);

  IrOp* reshape = model.add_op();
  reshape->name = ToString(OperationType::RESHAPE);
  ReshapeAttributes r_attr;
  r_attr.new_shape = BHWC(1, 1, 1, 2048);
  reshape->attr = r_attr;
  model.AddConsumer(perm->id, reshape->id);
  model.SetProducer(out->id, reshape->id);

  const auto tr_id = tr->id;
  const auto reshape_id = reshape->id;
  const auto sdpa_id = sdpa->id;

  ASSERT_TRUE(ir::FuseSdpaTransposedReshape(&model).ok());
  EXPECT_EQ(model.op(tr_id), nullptr);
  EXPECT_EQ(model.op(reshape_id), nullptr);
  ASSERT_NE(model.op(sdpa_id), nullptr);
  ASSERT_EQ(model.op(sdpa_id)->outputs.size(), 1u);
  EXPECT_EQ(model.op(sdpa_id)->outputs[0], out->id);
}

}  // namespace
}  // namespace litert::ml_drift
