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

#include "ml_drift_delegate/delegate/shared_memory_manager/ir_graph_adapter.h"

#include <cstdint>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace ml_drift {
namespace {

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

// Host-runnable unit test for IrModelAdapter: exercises each of the nine
// GraphAdapter methods against a hand-built ir::IrModel. No GPU /
// cl::Environment is required, so this runs directly on the host. The on-device
// SMM-drives-adapter path is covered separately by the single IR integration
// test in shared_memory_manager_cl_test.cc.

// Builds a minimal graph: an op named "fully_connected" that consumes `weights`
// (BHWC(1,1,1,10), FLOAT32) and produces `output`.
struct TestGraph {
  ir::IrModel model;
  uint32_t op_id = 0;
  uint32_t weights_id = 0;
  uint32_t output_id = 0;

  TestGraph() {
    ir::IrOp* op = model.add_op();
    op->name = "fully_connected";
    op_id = static_cast<uint32_t>(op->id);

    ir::IrTensor* weights = model.add_tensor(TensorDescriptor{});
    weights->desc.SetBHWCShape(BHWC(1, 1, 1, 10));
    weights->desc.SetDataType(DataType::FLOAT32);
    weights_id = static_cast<uint32_t>(weights->id);

    ir::IrTensor* output = model.add_tensor(TensorDescriptor{});
    output_id = static_cast<uint32_t>(output->id);

    // IrModel takes (tensor_id, op_id).
    model.AddConsumer(weights_id, op_id);
    model.SetProducer(output_id, op_id);
  }
};

TEST(IrModelAdapterTest, GetValueShape) {
  TestGraph g;
  IrModelAdapter adapter(g.model);
  EXPECT_EQ(adapter.GetValueShape(g.weights_id), BHWC(1, 1, 1, 10));
}

TEST(IrModelAdapterTest, SetValueTypeChangesTypeAndPreservesShape) {
  TestGraph g;
  IrModelAdapter adapter(g.model);

  adapter.SetValueType(g.weights_id, DataType::FLOAT16);

  EXPECT_EQ(g.model.tensor(g.weights_id)->desc.GetDataType(),
            DataType::FLOAT16);
  // Shape is untouched.
  EXPECT_EQ(g.model.tensor(g.weights_id)->desc.GetBHWCShape(),
            BHWC(1, 1, 1, 10));
}

TEST(IrModelAdapterTest, SetValueShapeAndTypeMutatesInPlaceWithStableId) {
  TestGraph g;
  IrModelAdapter adapter(g.model);

  // Capture the underlying object + id before mutating.
  const ir::IrTensor* before = g.model.tensor(g.weights_id);
  const ir::IrTensorId id_before = before->id;

  adapter.SetValueShapeAndType(g.weights_id, BHWC(1, 1, 25, 8),
                               DataType::FLOAT16);

  const ir::IrTensor* after = g.model.tensor(g.weights_id);
  // Same object, same id: the descriptor was mutated in place (not replaced).
  EXPECT_EQ(after, before);
  EXPECT_EQ(after->id, id_before);
  EXPECT_EQ(after->desc.GetBHWCShape(), BHWC(1, 1, 25, 8));
  EXPECT_EQ(after->desc.GetDataType(), DataType::FLOAT16);
  // The consumer wiring still references the same stable id.
  EXPECT_THAT(adapter.FindConsumerOps(g.weights_id), ElementsAre(g.op_id));
}

TEST(IrModelAdapterTest, FindConsumerOps) {
  TestGraph g;
  IrModelAdapter adapter(g.model);

  EXPECT_THAT(adapter.FindConsumerOps(g.weights_id), ElementsAre(g.op_id));
  // `output` is produced (not consumed) by the op, so it has no consumers.
  EXPECT_THAT(adapter.FindConsumerOps(g.output_id), IsEmpty());
}

TEST(IrModelAdapterTest, GetOpTypeName) {
  TestGraph g;
  IrModelAdapter adapter(g.model);
  EXPECT_EQ(adapter.GetOpTypeName(g.op_id), "fully_connected");
}

TEST(IrModelAdapterTest, OpHasInputs) {
  TestGraph g;
  IrModelAdapter adapter(g.model);

  EXPECT_TRUE(adapter.OpHasInputs(g.op_id));

  // A freshly added op has no inputs.
  ir::IrOp* lonely = g.model.add_op();
  EXPECT_FALSE(adapter.OpHasInputs(static_cast<uint32_t>(lonely->id)));
}

TEST(IrModelAdapterTest, GetOpFirstInputShapeAndType) {
  TestGraph g;
  IrModelAdapter adapter(g.model);

  EXPECT_EQ(adapter.GetOpFirstInputShape(g.op_id), BHWC(1, 1, 1, 10));
  EXPECT_EQ(adapter.GetOpFirstInputType(g.op_id), DataType::FLOAT32);
}

TEST(IrModelAdapterTest, AddConstantInputWiresNewValueToConsumerOp) {
  TestGraph g;
  IrModelAdapter adapter(g.model);

  const uint32_t new_id = adapter.AddConstantInput(
      /*global_tensor_id=*/123, BHWC(1, 1, 1, 4), DataType::FLOAT16, g.op_id);

  // A distinct new value was created with the requested shape/type.
  EXPECT_NE(new_id, g.weights_id);
  EXPECT_NE(new_id, g.output_id);
  ASSERT_NE(g.model.tensor(new_id), nullptr);
  EXPECT_EQ(g.model.tensor(new_id)->desc.GetBHWCShape(), BHWC(1, 1, 1, 4));
  EXPECT_EQ(g.model.tensor(new_id)->desc.GetDataType(), DataType::FLOAT16);

  // It is wired as an input of the consumer op...
  EXPECT_THAT(g.model.op(g.op_id)->inputs,
              Contains(static_cast<ir::IrTensorId>(new_id)));
  // ...and the reverse lookup resolves back to that op.
  EXPECT_THAT(adapter.FindConsumerOps(new_id), ElementsAre(g.op_id));
}

}  // namespace
}  // namespace ml_drift
