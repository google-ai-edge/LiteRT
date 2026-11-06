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

#include "ml_drift_delegate/delegate/composite/symmetric_quantized_fully_connected.h"

#include <deque>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

using ::ml_drift::CreateGpuModelInfo;
using ::ml_drift::DataType;
using ::ml_drift::GpuApi;
using ::ml_drift::GpuInfo;
using ::ml_drift::GpuModelBuilder;
using ::ml_drift::GpuModelBuilderOptions;
using ::ml_drift::Node;
using ::ml_drift::OHWI;
using ::ml_drift::OperationType;
using ::ml_drift::Value;
using ::ml_drift::ValueId;

GpuInfo GetTestGpuInfo() {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  gpu_info.opencl_info.supports_fp16 = true;
  return gpu_info;
}

::ml_drift::FullyConnectedInt4Attributes GetInt4Attributes() {
  ::ml_drift::FullyConnectedInt4Attributes attr;
  ::ml_drift::Tensor<OHWI, DataType::kInt4> weights;
  weights.shape = OHWI(16, 1, 1, 64);
  attr.weights = weights;
  attr.scale.shape = OHWI(16, 1, 1, 2);
  return attr;
}

// Runs the helper on `node` with `num_inputs` inputs and one output.
absl::Status CreateFromNode(const Node& node, int num_inputs) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  CreateGpuModelInfo create_info;
  GpuModelBuilder builder(gpu_info, GpuModelBuilderOptions());
  std::deque<Value> values;
  std::vector<Value*> inputs;
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(&values.emplace_back(Value{static_cast<ValueId>(i)}));
  }
  Value& output = values.emplace_back(Value{static_cast<ValueId>(num_inputs)});
  return CreateSymmetricQuantizedFullyConnectedFromNode(
      gpu_info, create_info, inputs, {&output}, node, &builder);
}

TEST(SymmetricQuantizedFullyConnectedTest, WithZeroPointFallsBackToMlDrift) {
  // With (src, weights, scale, zero_point), ML Drift binds the zero point.
  ::ml_drift::FullyConnectedInt4Attributes attr = GetInt4Attributes();
  attr.zero_point.shape = attr.scale.shape;
  const Node node{/*id=*/0,
                  {ToString(OperationType::kFullyConnectedInt4), attr}};

  EXPECT_TRUE(absl::IsUnimplemented(CreateFromNode(node, /*num_inputs=*/4)));
}

TEST(SymmetricQuantizedFullyConnectedTest, OtherOpsFallBackToMlDrift) {
  const Node node{
      /*id=*/0,
      {ToString(OperationType::kFullyConnected), GetInt4Attributes()}};

  EXPECT_TRUE(absl::IsUnimplemented(CreateFromNode(node, /*num_inputs=*/3)));
}

TEST(SymmetricQuantizedFullyConnectedTest, ConstWeightsFallBackToMlDrift) {
  // A quantized FullyConnected with weights in its attributes has one input.
  const Node node{
      /*id=*/0,
      {ToString(OperationType::kFullyConnectedInt4), GetInt4Attributes()}};

  EXPECT_TRUE(absl::IsUnimplemented(CreateFromNode(node, /*num_inputs=*/1)));
}

TEST(SymmetricQuantizedFullyConnectedTest,
     FiveInputsWithZeroPointFallBackToMlDrift) {
  // With (src, weights, bias, scale, zero_point), ML Drift binds both bias and
  // zero point.
  ::ml_drift::FullyConnectedInt4Attributes attr = GetInt4Attributes();
  attr.zero_point.shape = attr.scale.shape;
  const Node node{/*id=*/0,
                  {ToString(OperationType::kFullyConnectedInt4), attr}};

  EXPECT_TRUE(absl::IsUnimplemented(CreateFromNode(node, /*num_inputs=*/5)));
}

TEST(SymmetricQuantizedFullyConnectedTest,
     MismatchedAttributesReturnInvalidArgument) {
  // Wrong attribute type must fail explicitly rather than falling back to
  // ML Drift's reference `std::any_cast` (which throws `bad_any_cast`).
  const Node node{/*id=*/0,
                  {ToString(OperationType::kFullyConnectedInt4),
                   ::ml_drift::FullyConnectedInt8Attributes{}}};

  EXPECT_TRUE(absl::IsInvalidArgument(CreateFromNode(node, /*num_inputs=*/3)));
}

TEST(SymmetricQuantizedFullyConnectedTest,
     ThreeInputsWithZeroPointReturnInvalidArgument) {
  // External quantized weights with a zero point require at least 4 inputs
  // (src, weights, scale, zero_point).
  ::ml_drift::FullyConnectedInt4Attributes attr = GetInt4Attributes();
  attr.zero_point.shape = attr.scale.shape;
  const Node node{/*id=*/0,
                  {ToString(OperationType::kFullyConnectedInt4), attr}};

  EXPECT_TRUE(absl::IsInvalidArgument(CreateFromNode(node, /*num_inputs=*/3)));
}

}  // namespace
}  // namespace litert::ml_drift
