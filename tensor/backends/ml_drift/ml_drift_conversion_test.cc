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

#include "tensor/backends/ml_drift/ml_drift_conversion.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/cl_test.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/backends/ml_drift/ml_drift_buffer.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/runners/ml_drift/ml_drift_dummy_weight_creator.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"

#define XNN_EXTRA_BYTES 16

namespace litert::tensor {

namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;

using TensorMLD = Tensor<MlDriftMixinTag>;

ml_drift::TensorDescriptor CreateDummyFCWeight(
    ml_drift::GpuModelBuilder& model_builder, TensorHandle& t,
    ml_drift::DataType mld_float_data_type) {
  LRT_TENSOR_ASSIGN_OR_ABORT(const graph::TensorInformation& info,
                             graph::GetInfo(t.GetRaw()));
  auto t_shape = info.shape;
  auto desc = model_builder.GetFullyConnectedWeightsDesc(
      mld_float_data_type, ml_drift::OHWI(t_shape[0], 1, 1, t_shape[1]));

  ml_drift::TensorDescriptor weights_td(desc.type,
                                        ml_drift::TensorStorageType::kBuffer,
                                        ml_drift::Layout::kLinear);

  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> weights;
  weights.shape = ml_drift::OHWI(t_shape[0], 1, 1, t_shape[1]);
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = std::sin(i);
  }
  const int elements_count =
      ml_drift::GetTotalElementsCountForLayout(desc, weights.shape);
  std::vector<uint8_t> data(elements_count * ml_drift::SizeOf(desc.type));
  RearrangeWeights(weights, desc, absl::MakeSpan(data));

  weights_td.SetBHWDCShape(
      ml_drift::BHWDC(1, 1, 1, 1, t_shape[0] * t_shape[1]));

  std::string value_str;
  for (int i = 0; i < data.size(); ++i) {
    absl::StrAppend(&value_str, static_cast<int>(data[i]), ", ");
  }

  weights_td.UploadDataRaw(absl::MakeConstSpan(data));
  return weights_td;
}
}  // namespace

class MlDriftConversionTest : public ml_drift::cl::OpenCLTest {
 public:
  MlDriftBuildContext context_;
  ml_drift::GpuModelBuilder* model_builder_;

  void SetUp() override {
    ml_drift::cl::OpenCLTest::SetUp();
    model_builder_ = new ml_drift::GpuModelBuilder(
        env_.device().GetInfo(), {}, ml_drift::CalculationsPrecision::kF32,
        ml_drift::TensorStorageType::kBuffer);
    context_.gpu_model_builder = model_builder_;
    context_.default_float_type = ml_drift::DataType::kFloat32;
  }

  void TearDown() override { delete model_builder_; }
};

TEST_F(MlDriftConversionTest, AddOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Add(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "add";
              })));
}

TEST_F(MlDriftConversionTest, GeluOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Gelu(input, false);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "gelu";
              })));
}

TEST_F(MlDriftConversionTest, ReluOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Relu(input);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "relu";
              })));
}

TEST_F(MlDriftConversionTest, Relu6Op) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Relu6(input);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "relu";
              })));
}

TEST_F(MlDriftConversionTest, LeakyReluOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = LeakyRelu(input, 0.2f);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "relu";
              })));
}

TEST_F(MlDriftConversionTest, EluOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Elu(input);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "elu";
              })));
}

TEST_F(MlDriftConversionTest, HardSwishOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = HardSwish(input);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "hardswish";
              })));
}

TEST_F(MlDriftConversionTest, PReluOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD alpha(
      {.type = Type::kFP32, .shape = {1}, .buffer = std::vector<float>{0.2f}});
  TensorMLD output = PRelu(input, alpha);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "prelu";
              })));
}

TEST_F(MlDriftConversionTest, L2NormalizationReturnsUnimplemented) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = L2Normalization(input);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  EXPECT_EQ(tensor_to_handle_or.status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST_F(MlDriftConversionTest, LogisticOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Logistic(input);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "sigmoid";
              })));
}

TEST_F(MlDriftConversionTest, TanhOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Tanh(input);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "tanh";
              })));
}

TEST_F(MlDriftConversionTest, AveragePool2DOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  TensorMLD output = AveragePool2D(input, 2, 2, 2, 2, kPaddingSame);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "pooling";
              })));
}

TEST_F(MlDriftConversionTest, MaxPool2DOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  TensorMLD output = MaxPool2D(input, 2, 2, 2, 2, kPaddingSame);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "pooling";
              })));
}

TEST_F(MlDriftConversionTest, GatherOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD indices({.type = Type::kI32, .shape = {1}});
  TensorMLD output = Gather(input, indices, 1, 0);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input, indices}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "gather";
              })));
}

TEST_F(MlDriftConversionTest, OneHotOp) {
  TensorMLD indices({.type = Type::kI32, .shape = {4}});
  TensorMLD depth({.type = Type::kI32,
                   .shape = {},
                   .buffer = OwningCpuBuffer::Copy<Type::kI32>({3})});
  TensorMLD on_value({.type = Type::kFP32,
                      .shape = {},
                      .buffer = OwningCpuBuffer::Copy<Type::kFP32>({1.0f})});
  TensorMLD off_value({.type = Type::kFP32,
                       .shape = {},
                       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({0.0f})});
  TensorMLD output = OneHot(indices, depth, on_value, off_value, -1);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {indices}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                            return node.name == "reshape";
                          }),
                          testing::Truly([](const ml_drift::GpuNode& node) {
                            return node.name == "one_hot";
                          }),
                          testing::Truly([](const ml_drift::GpuNode& node) {
                            return node.name == "reshape";
                          })));
}

TEST_F(MlDriftConversionTest, FullyConnectedOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 4}});
  TensorMLD weights({.type = Type::kFP32, .shape = {4, 4}});
  TensorMLD bias({.type = Type::kFP32, .shape = {1, 4}});
  TensorMLD output = FullyConnected(input, weights, bias);

  for (auto t : {weights}) {
    auto tensor_desc =
        CreateDummyFCWeight(*model_builder_, t, context_.default_float_type);
    context_.graph_tensor_to_handle[t.GetRaw()] =
        model_builder_->AddConstantTensor(std::move(tensor_desc));
  }

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input, weights, bias}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return absl::StartsWith(node.name, "fully_connected");
              })));
}

TEST_F(MlDriftConversionTest, ReshapeOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Reshape(input, {1, 4});

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "reshape";
              })));
}

TEST_F(MlDriftConversionTest, ExpandDimsOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = ExpandDims(input, 1);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "reshape";
              })));
}

TEST_F(MlDriftConversionTest, SubOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Sub(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "subtract";
              })));
}

TEST_F(MlDriftConversionTest, PowOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Pow(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "pow";
              })));
}

TEST_F(MlDriftConversionTest, MulOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Mul(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "mul";
              })));
}

TEST_F(MlDriftConversionTest, DivOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Div(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "div";
              })));
}

TEST_F(MlDriftConversionTest, FloorDivOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = FloorDiv(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "floor_div";
              })));
}

TEST_F(MlDriftConversionTest, FloorModOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = FloorMod(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "floor_mod";
              })));
}

TEST_F(MlDriftConversionTest, SinOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Sin(input1);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "sin";
              })));
}

TEST_F(MlDriftConversionTest, CosOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Cos(input1);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "cos";
              })));
}

TEST_F(MlDriftConversionTest, RsqrtOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Rsqrt(input);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "rsqrt";
              })));
}

TEST_F(MlDriftConversionTest, QuantizeOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Quantize(input, Type::kI8, {0.5}, {0});
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "quantize";
              })));
}

TEST_F(MlDriftConversionTest, DequantizeOp) {
  TensorMLD quant_input({.type = Type::kI8, .shape = {1, 2, 2, 1}});
  auto quant_params = std::make_shared<PerChannelAffineQuantization>();
  quant_params->scales = {0.25f};
  quant_params->zero_points = ZeroPoints::Int({0});
  quant_input.SetQuantization(quant_params);
  TensorMLD output = Dequantize(quant_input);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {quant_input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "dequantize";
              })));
}

// Per-channel dequantizations are computed in fp32, like per-tensor ones, so
// fp16 models don't round the scales.
TEST_F(MlDriftConversionTest, DequantizePerChannelOpComputesInFp32) {
  context_.default_float_type = ml_drift::DataType::kFloat16;
  TensorMLD quant_input({.type = Type::kI8, .shape = {2, 3}});
  quant_input.SetQuantization(std::make_shared<PerChannelAffineQuantization>(
      std::vector<float>{0.1f, 0.3f}, std::vector<int64_t>{1, -1},
      /*quantized_dimension=*/0));
  TensorMLD output = Dequantize(quant_input);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);
  const ml_drift::ValueId output_id =
      tensor_to_handle_or->at(output.GetRaw()).id;

  ml_drift::GpuModel model;
  ASSERT_OK(context_.gpu_model_builder->GetGpuModel(
      {tensor_to_handle_or->at(quant_input.GetRaw()).id}, {output_id}, &model));

  ASSERT_THAT(model.const_tensors, testing::SizeIs(2));
  for (const auto& [id, desc] : model.const_tensors) {
    EXPECT_EQ(desc.GetDataType(), ml_drift::DataType::kFloat32);
  }
  ASSERT_TRUE(model.tensors.contains(output_id));
  EXPECT_EQ(model.tensors.at(output_id).GetDataType(),
            ml_drift::DataType::kFloat16);
}

TEST_F(MlDriftConversionTest, CumsumOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD axis({.type = Type::kI32,
                  .shape = {},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>({-1})});
  TensorMLD output = Cumsum(input, axis);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "cumsum";
              })));
}

TEST_F(MlDriftConversionTest, ReverseOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD axes({.type = Type::kI32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>({-1})});
  TensorMLD output = Reverse(input, axes);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "reverse";
              })));
}

TEST_F(MlDriftConversionTest, DepthToSpaceOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 4}});
  TensorMLD output = DepthToSpace(input, 2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "depth_to_space";
              })));
}

TEST_F(MlDriftConversionTest, SumOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD axis({.type = Type::kI32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>({0})});
  TensorMLD output = Sum(input, axis, true);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return absl::StartsWith(node.name, "reduce_sum");
              })));
}

TEST_F(MlDriftConversionTest, ReduceMaxOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD axis({.type = Type::kI32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>({0})});
  TensorMLD output = ReduceMax(input, axis, true);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return absl::StartsWith(node.name, "reduce_max");
              })));
}

TEST_F(MlDriftConversionTest, ArgMaxOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 3, 4}});
  TensorMLD axis({.type = Type::kI32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>({2})});
  TensorMLD output = ArgMax(input, axis);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              Contains(testing::Truly([](const ml_drift::GpuNode& node) {
                return absl::StartsWith(node.name, "reduce_maximum_index");
              })));
}

TEST_F(MlDriftConversionTest, MeanOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD axis({.type = Type::kI32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>({0})});
  TensorMLD output = Mean(input, axis, true);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return absl::StartsWith(node.name, "reduce_mean");
              })));
}

TEST_F(MlDriftConversionTest, SelectOp) {
  TensorMLD condition({.type = Type::kBOOL, .shape = {1, 2, 2, 1}});
  condition.SetName("condition");
  TensorMLD t({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  t.SetName("t");
  TensorMLD e({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  e.SetName("e");
  TensorMLD output = Select(condition, t, e);
  output.SetName("output");
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {condition, t, e}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "selectv2";
              })));
}

TEST_F(MlDriftConversionTest, SelectV2Op) {
  TensorMLD condition({.type = Type::kBOOL, .shape = {1, 2, 2, 1}});
  condition.SetName("condition");
  TensorMLD t({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  t.SetName("t");
  TensorMLD e({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  e.SetName("e");
  TensorMLD output = SelectV2(condition, t, e);
  output.SetName("output");
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {condition, t, e}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "selectv2";
              })));
}

TEST_F(MlDriftConversionTest, SliceOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 4, 4, 1}});
  TensorMLD output = Slice(input, {0, 1, 1, 0}, {1, 2, 2, 1});
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "strided_slice";
              })));
}

TEST_F(MlDriftConversionTest, ConcatenationOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Concatenation({input1, input2}, 3);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return absl::StartsWith(node.name, "concat_");
              })));
}

TEST_F(MlDriftConversionTest, PackOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {2, 2, 1}});
  TensorMLD output = Pack({input1, input2}, 0);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              (ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                             return node.name == "reshape";
                           }),
                           testing::Truly([](const ml_drift::GpuNode& node) {
                             return node.name == "reshape";
                           }),
                           testing::Truly([](const ml_drift::GpuNode& node) {
                             return absl::StartsWith(node.name, "concat_");
                           }))));
}

TEST_F(MlDriftConversionTest, UnpackOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {2, 2, 2, 1}});
  std::vector<TensorMLD> outputs = Unpack(input, 2, 0);
  auto tensor_to_handle_or =
      AppendGpuModelBuilder(context_, absl::MakeConstSpan(outputs));
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : outputs) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              (ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return absl::StartsWith(node.name, "split");
              }))));
}

TEST_F(MlDriftConversionTest, SplitOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {2, 4, 1, 1}});
  std::vector<TensorMLD> outputs = Split(input, 1, 2);
  auto tensor_to_handle_or =
      AppendGpuModelBuilder(context_, absl::MakeConstSpan(outputs));
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : outputs) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              (ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return absl::StartsWith(node.name, "split");
              }))));
}

TEST_F(MlDriftConversionTest, BatchMatmulOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 3, 4}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 4, 5}});
  TensorMLD output = BatchMatMul(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(
      model.nodes,
      testing::Contains(testing::Truly([](const ml_drift::GpuNode& node) {
        return absl::StartsWith(node.name, "bhwc_as_");
      })));
}

TEST_F(MlDriftConversionTest, TransposeOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 3, 4}});
  TensorMLD perm({.type = Type::kI32,
                  .shape = {4},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>({0, 2, 1, 3})});
  TensorMLD output = Transpose(input, perm);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "transpose";
              })));
}

TEST_F(MlDriftConversionTest, DynamicUpdateSliceOp) {
  TensorMLD operand({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD update({.type = Type::kFP32, .shape = {1, 1, 1, 1}});
  TensorMLD start_indices(
      {.type = Type::kI32,
       .shape = {4},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 1, 1, 1})});
  TensorMLD output = DynamicUpdateSlice(operand, update, start_indices);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {operand, update, start_indices}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "dynamic_update_slice";
              })));
}

TEST_F(MlDriftConversionTest, DynamicUpdateSliceOpBatch) {
  TensorMLD operand({.type = Type::kFP32, .shape = {2, 2, 2, 1}});
  TensorMLD update({.type = Type::kFP32, .shape = {2, 1, 1, 1}});
  TensorMLD start_indices(
      {.type = Type::kI32,
       .shape = {4},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 1, 1, 1})});
  TensorMLD output = DynamicUpdateSlice(operand, update, start_indices);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {operand, update, start_indices}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "dynamic_update_slice";
              })));
}

TEST_F(MlDriftConversionTest, SoftmaxOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 4}});
  TensorMLD output = Softmax(input);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(
      model.nodes,
      testing::Contains(testing::Truly([](const ml_drift::GpuNode& node) {
        return absl::StartsWith(node.name, "softmax");
      })));
}

TEST_F(MlDriftConversionTest, EqualOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Equal(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "equal";
              })));
}

TEST_F(MlDriftConversionTest, GreaterEqualOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = GreaterEqual(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "greater_equal";
              })));
}

TEST_F(MlDriftConversionTest, NotEqualOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = NotEqual(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "not_equal";
              })));
}

TEST_F(MlDriftConversionTest, CastOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Cast(input, Type::kI32);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "cast";
              })));
}

TEST_F(MlDriftConversionTest, LessOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Less(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "less";
              })));
}

TEST_F(MlDriftConversionTest, LogicalAndOp) {
  TensorMLD input1({.type = Type::kBOOL, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kBOOL, .shape = {1, 2, 2, 1}});
  TensorMLD output = LogicalAnd(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "logical_and";
              })));
}

TEST_F(MlDriftConversionTest, LogicalNotOp) {
  TensorMLD input1({.type = Type::kBOOL, .shape = {1, 2, 2, 1}});
  TensorMLD output = LogicalNot(input1);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "logical_not";
              })));
}

TEST_F(MlDriftConversionTest, SqrtOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Sqrt(input);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "sqrt";
              })));
}

TEST_F(MlDriftConversionTest, MaximumOp) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Maximum(input1, input2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input1, input2}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "maximum";
              })));
}

// ML Drift has no float zero point support outside the FullyConnected
// weights, so it rejects them instead of silently dropping them.
TEST_F(MlDriftConversionTest, DequantizeRejectsFloatZeroPoints) {
  TensorMLD quant_input({.type = Type::kI8, .shape = {1, 2, 2, 1}});
  auto quant_params = std::make_shared<PerChannelAffineQuantization>();
  quant_params->scales = {0.25f};
  quant_params->zero_points = ZeroPoints::Float({-0.5f});
  quant_input.SetQuantization(quant_params);
  TensorMLD output = Dequantize(quant_input);
  EXPECT_EQ(AppendGpuModelBuilder(context_, {output}).status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST_F(MlDriftConversionTest, EmbeddingLookupOp) {
  TensorMLD ids({.type = Type::kI32, .shape = {1, 4}});
  TensorMLD weights({.type = Type::kFP32, .shape = {8, 4}});
  TensorMLD output = EmbeddingLookup(ids, weights);
  MlDriftDummyWeightCreator dummy_weight_creator(context_.gpu_model_builder);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto tensor_handle,
      dummy_weight_creator.CreateFcWeights<float>(
          weights, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}));
  context_.graph_tensor_to_handle[weights.GetRaw()] = tensor_handle;
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {ids, weights}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(
      model.nodes,
      testing::Contains(testing::Truly([](const ml_drift::GpuNode& node) {
        return absl::StartsWith(node.name, "embedding_lookup");
      })));
}
TEST_F(MlDriftConversionTest, ResizeBilinearOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 10, 10, 3}});
  TensorMLD size({.type = Type::kI32,
                  .shape = {2},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>({20, 20})});
  TensorMLD output = ResizeBilinear(input, size);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "resize_bilinear";
              })));
}

TEST_F(MlDriftConversionTest, ResizeNearestNeighborOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 10, 10, 3}});
  TensorMLD size({.type = Type::kI32,
                  .shape = {2},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>({20, 20})});
  TensorMLD output = ResizeNearestNeighbor(input, size);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "resize";
              })));
}

TEST_F(MlDriftConversionTest, LstmOp) {
  TensorMLD intermediate({.type = Type::kFP32, .shape = {1, 1, 1, 8}});
  TensorMLD prev_state({.type = Type::kFP32, .shape = {1, 1, 1, 2}});
  std::vector<TensorMLD> outputs = Lstm(intermediate, prev_state);
  auto tensor_to_handle_or =
      AppendGpuModelBuilder(context_, {outputs[0], outputs[1]});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {intermediate, prev_state}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {outputs[0], outputs[1]}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return node.name == "lstm";
              })));
}

TEST_F(MlDriftConversionTest, DepthwiseConv2DOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};
  TensorMLD filter({.type = Type::kFP32,
                    .shape = {1, 2, 2, 1},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>(filter_data)});
  std::vector<float> bias_data = {1.0f};
  TensorMLD bias({.type = Type::kFP32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(bias_data)});

  TensorMLD output = DepthwiseConv2D(input, filter, bias, /*stride_h=*/1,
                                     /*stride_w=*/1, kPaddingValid);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return absl::StartsWith(node.name, "depthwise_conv");
              })));
}

TEST_F(MlDriftConversionTest, Conv2DOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};
  TensorMLD filter({.type = Type::kFP32,
                    .shape = {1, 2, 2, 1},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>(filter_data)});
  std::vector<float> bias_data = {1.0f};
  TensorMLD bias({.type = Type::kFP32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(bias_data)});

  TensorMLD output = Conv2D(input, filter, bias, /*stride_h=*/1, /*stride_w=*/1,
                            kPaddingValid);

  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(model.nodes,
              ElementsAre(testing::Truly([](const ml_drift::GpuNode& node) {
                return absl::StartsWith(node.name, "convolution");
              })));
}

TEST_F(MlDriftConversionTest, TransposeConvOp) {
  std::vector<float> input_data(10 * 10 * 3, 1.0f);
  std::vector<float> filter_data(3 * 2 * 2 * 3, 1.0f);
  TensorMLD input({.type = Type::kFP32, .shape = {1, 10, 10, 3}});
  TensorMLD filter({.type = Type::kFP32,
                    .shape = {3, 2, 2, 3},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>(filter_data)});
  std::vector<float> bias_data(3, 0.0f);
  TensorMLD bias({.type = Type::kFP32,
                  .shape = {3},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(bias_data)});

  std::vector<int> output_shape = {1, 20, 20, 3};
  TensorMLD output =
      TransposeConv(filter, input, bias, output_shape, kPaddingSame, 2, 2);
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, {output});
  ASSERT_OK(tensor_to_handle_or);

  ml_drift::GpuModel model;
  std::vector<ml_drift::ValueId> input_ids, output_ids;

  for (auto t : {input}) {
    input_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }
  for (auto t : {output}) {
    output_ids.push_back(tensor_to_handle_or->at(t.GetRaw()).id);
  }

  ASSERT_OK(
      context_.gpu_model_builder->GetGpuModel(input_ids, output_ids, &model));

  EXPECT_THAT(
      model.nodes,
      testing::Contains(testing::Truly([](const ml_drift::GpuNode& node) {
        return absl::StartsWith(node.name, "conv_transposed2x2");
      })));
}

TEST_F(MlDriftConversionTest, GetTensorHandleWithMlDriftBuffer) {
  ml_drift::CreateGpuModelInfo create_info;
  create_info.precision = ml_drift::CalculationsPrecision::kF32;
  create_info.storage_type = ml_drift::TensorStorageType::kBuffer;
  create_info.hints.Add(ml_drift::ModelHints::kAllowSpecialKernels);

  auto handle = context_.gpu_model_builder->AddTensor(
      ml_drift::BHWC(1, 2, 3, 4), context_.default_float_type);
  ml_drift::GpuModel gpu_model;
  ASSERT_OK(context_.gpu_model_builder->GetGpuModel({handle.id}, {handle.id},
                                                    &gpu_model));

  ml_drift::cl::InferenceContext inference_context;
  ASSERT_OK(inference_context.InitFromGpuModel(create_info, &gpu_model, &env_));

  Tensor<MlDriftMixinTag> tensor_with_buffer(
      {.name = "foo", .type = Type::kFP32, .shape = {1, 2, 3, 4}});
  context_.default_float_type = ml_drift::DataType::kFloat32;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const graph::TensorInformation& const_tensor_info,
      graph::GetInfo(tensor_with_buffer.GetRaw()));
  auto* tensor_info = const_cast<graph::TensorInformation*>(&const_tensor_info);

  size_t buffer_size = 1 * 2 * 3 * 4 * sizeof(float);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      tensor_info->buffer, MlDriftBuffer::Create(handle.id, inference_context,
                                                 *env_.queue(), buffer_size));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto tensor_handle,
      GetTensorHandle(context_, tensor_with_buffer.GetRaw()));
  EXPECT_EQ(tensor_handle.id, handle.id);
}

TEST_F(MlDriftConversionTest, BatchMatMulRejectsBroadcastBatchDimensions) {
  // Mismatched head/batch dimension: [1, 4, 3, 8] x [1, 1, 8, 5].
  {
    TensorMLD lhs({.type = Type::kFP32, .shape = {1, 4, 3, 8}});
    TensorMLD rhs({.type = Type::kFP32, .shape = {1, 1, 8, 5}});
    TensorMLD out = BatchMatMul(lhs, rhs);
    auto status_or = AppendGpuModelBuilder(context_, {out});
    EXPECT_THAT(status_or.status(),
                absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                       testing::HasSubstr("broadcast")));
  }
  // Mismatched batch rank with implicit leading broadcast: [2, 4, 3, 8] x
  // [4, 8, 5].
  {
    TensorMLD lhs({.type = Type::kFP32, .shape = {2, 4, 3, 8}});
    TensorMLD rhs({.type = Type::kFP32, .shape = {4, 8, 5}});
    TensorMLD out = BatchMatMul(lhs, rhs);
    auto status_or = AppendGpuModelBuilder(context_, {out});
    EXPECT_THAT(status_or.status(),
                absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                       testing::HasSubstr("broadcast")));
  }
}

TEST_F(MlDriftConversionTest, DequantizeSymmetricPerChannelOpSkipsZeroPoints) {
  TensorMLD input(
      {.type = Type::kI8,
       .shape = {2, 3},
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           /*scales=*/std::vector<float>{0.1f, 0.3f},
           /*zero_points=*/ZeroPoints::Int({0, 0}),
           /*quantized_dimension=*/0)});
  TensorMLD output = Dequantize(input);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto tensor_to_handle_map,
                                  AppendGpuModelBuilder(context_, {output}));
  ml_drift::GpuModel model;
  ASSERT_OK(context_.gpu_model_builder->GetGpuModel(
      {tensor_to_handle_map.at(input.GetRaw()).id},
      {tensor_to_handle_map.at(output.GetRaw()).id}, &model));

  for (const ml_drift::GpuNode& node : model.nodes) {
    EXPECT_NE(node.name, "dequantize_sub");
  }
  EXPECT_THAT(model.const_tensors, SizeIs(1));
}

TEST_F(MlDriftConversionTest,
       BlockwiseQuantizationValidationAndAffineZeroPointLowering) {
  auto add_dummy_weight_handle = [&](TensorMLD& w) {
    context_.graph_tensor_to_handle[w.GetRaw()] = model_builder_->AddTensor(
        ml_drift::BHWC(1, 1, 1, 64), ml_drift::DataType::kUint8);
  };

  // 1. Valid affine blockwise INT4 (`Q4_1` / `Q4_K`) creates scale + zero-point
  // constant tensors.
  {
    MlDriftBuildContext local_ctx;
    ml_drift::GpuModelBuilder builder(env_.device().GetInfo(), {},
                                      ml_drift::CalculationsPrecision::kF32,
                                      ml_drift::TensorStorageType::kBuffer);
    local_ctx.gpu_model_builder = &builder;
    local_ctx.default_float_type = ml_drift::DataType::kFloat32;

    TensorMLD x({.type = Type::kFP32, .shape = {1, 2, 32}});
    TensorMLD w({.name = "q4_k.weight",
                 .type = Type::kI4,
                 .shape = {4, 32},
                 .quantization = std::make_shared<BlockwiseQuantization>(
                     std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                     ZeroPoints::Float({-2.0f, 1.5f, 0.0f, -3.25f}),
                     /*block_size=*/32, /*quantized_dimension=*/0)});
    local_ctx.graph_tensor_to_handle[w.GetRaw()] = builder.AddTensor(
        ml_drift::BHWC(1, 1, 1, 64), ml_drift::DataType::kUint8);
    TensorMLD out = FullyConnected(x, w);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto map,
                                    AppendGpuModelBuilder(local_ctx, {out}));
    ml_drift::GpuModel model;
    ASSERT_OK(
        builder.GetGpuModel({map.at(x.GetRaw()).id,
                             local_ctx.graph_tensor_to_handle[w.GetRaw()].id},
                            {map.at(out.GetRaw()).id}, &model));
    EXPECT_THAT(model.const_tensors, SizeIs(2));
  }

  // 2. Single output channel (`ohwi.o == 1`) returns UnimplementedError.
  {
    TensorMLD x({.type = Type::kFP32, .shape = {1, 2, 32}});
    TensorMLD w({.name = "single_out.weight",
                 .type = Type::kI4,
                 .shape = {1, 32},
                 .quantization = std::make_shared<BlockwiseQuantization>(
                     std::vector<float>{0.25f}, std::vector<int64_t>{},
                     /*block_size=*/32)});
    add_dummy_weight_handle(w);
    TensorMLD out = FullyConnected(x, w);
    EXPECT_THAT(AppendGpuModelBuilder(context_, {out}).status(),
                absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
  }

  // 3. Input dimension not divisible by `block_size` returns InvalidArgument.
  {
    TensorMLD x({.type = Type::kFP32, .shape = {1, 2, 48}});
    TensorMLD w({.name = "bad_block.weight",
                 .type = Type::kI8,
                 .shape = {4, 48},
                 .quantization = std::make_shared<BlockwiseQuantization>(
                     std::vector<float>(4, 0.25f), std::vector<int64_t>{},
                     /*block_size=*/32)});
    add_dummy_weight_handle(w);
    TensorMLD out = FullyConnected(x, w);
    EXPECT_THAT(AppendGpuModelBuilder(context_, {out}).status(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 4. Mismatched `zero_points.size()` returns InvalidArgument.
  {
    TensorMLD x({.type = Type::kFP32, .shape = {1, 2, 32}});
    TensorMLD w({.name = "bad_zp_count.weight",
                 .type = Type::kI4,
                 .shape = {4, 32},
                 .quantization = std::make_shared<BlockwiseQuantization>(
                     std::vector<float>(4, 0.25f),
                     ZeroPoints::Float({-1.0f, 1.0f}), /*block_size=*/32,
                     /*quantized_dimension=*/0)});
    add_dummy_weight_handle(w);
    TensorMLD out = FullyConnected(x, w);
    EXPECT_THAT(AppendGpuModelBuilder(context_, {out}).status(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 5. Mismatched blockwise `scales.size()` returns InvalidArgument.
  {
    TensorMLD x({.type = Type::kFP32, .shape = {1, 2, 32}});
    TensorMLD w({.name = "bad_scales.weight",
                 .type = Type::kI8,
                 .shape = {4, 32},
                 .quantization = std::make_shared<BlockwiseQuantization>(
                     std::vector<float>(3, 0.25f), std::vector<int64_t>{},
                     /*block_size=*/32)});
    add_dummy_weight_handle(w);
    TensorMLD out = FullyConnected(x, w);
    EXPECT_THAT(AppendGpuModelBuilder(context_, {out}).status(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 6. Non-zero `quantized_dimension` returns InvalidArgument.
  {
    TensorMLD x({.type = Type::kFP32, .shape = {1, 2, 32}});
    TensorMLD w({.name = "bad_qdim.weight",
                 .type = Type::kI8,
                 .shape = {4, 32},
                 .quantization = std::make_shared<BlockwiseQuantization>(
                     std::vector<float>(4, 0.25f), std::vector<int64_t>{},
                     /*block_size=*/32, /*quantized_dimension=*/1)});
    add_dummy_weight_handle(w);
    TensorMLD out = FullyConnected(x, w);
    EXPECT_THAT(AppendGpuModelBuilder(context_, {out}).status(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 7. `block_size % 4 != 0` when `num_blocks > 1` returns InvalidArgument.
  {
    TensorMLD x({.type = Type::kFP32, .shape = {1, 2, 12}});
    TensorMLD w({.name = "bad_block_align.weight",
                 .type = Type::kI8,
                 .shape = {4, 12},
                 .quantization = std::make_shared<BlockwiseQuantization>(
                     std::vector<float>(8, 0.25f), std::vector<int64_t>{},
                     /*block_size=*/6, /*quantized_dimension=*/0)});
    add_dummy_weight_handle(w);
    TensorMLD out = FullyConnected(x, w);
    EXPECT_THAT(AppendGpuModelBuilder(context_, {out}).status(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST_F(MlDriftConversionTest,
       LazyBlockwiseQuantizationUsesCacheWithoutMaterializing) {
  auto make_desc = [&](std::vector<float> vals) {
    ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> t;
    t.shape = ml_drift::OHWI(4, 1, 1, 2);
    t.data = std::move(vals);
    ml_drift::TensorDescriptor td = ml_drift::ScaleOrZeroPointToTensorDesc(
        context_.gpu_model_builder->gpu_info(), t,
        ml_drift::DataType::kFloat32);
    ml_drift::TensorDescriptor dataless;
    td.CopyWithoutData(&dataless);
    return dataless;
  };

  // 1. Cached `:scale:` and `:zp:` are reused without materializing `bwq`.
  {
    absl::flat_hash_map<std::string, ml_drift::TensorDescriptor> shared;
    const int dt = static_cast<int>(ml_drift::DataType::kFloat32);
    shared.emplace(absl::StrCat("lazy_cached.weight:scale:", dt),
                   make_desc(std::vector<float>(8, 0.25f)));
    shared.emplace(absl::StrCat("lazy_cached.weight:zp:", dt),
                   make_desc(std::vector<float>(8, -1.0f)));

    MlDriftBuildContext local_ctx;
    ml_drift::GpuModelBuilder builder(context_.gpu_model_builder->gpu_info(),
                                      ml_drift::ModelHints{},
                                      ml_drift::CalculationsPrecision::kF32,
                                      ml_drift::TensorStorageType::kBuffer);
    local_ctx.gpu_model_builder = &builder;
    local_ctx.default_float_type = ml_drift::DataType::kFloat32;
    local_ctx.shared_weight_descriptors = &shared;

    int materialize_calls = 0;
    using ZeroPointPresence = BlockwiseQuantization::ZeroPointPresence;
    BlockwiseQuantization::LazyOptions lazy_options{
        .zero_points = ZeroPointPresence::kMayHaveFloat,
    };
    auto bwq = BlockwiseQuantization::CreateLazily(
        /*block_size=*/32, /*quantized_dimension=*/0, /*num_blocks=*/8,
        [&]() -> BlockwiseQuantization::MaterializedBlockwiseParams {
          ++materialize_calls;
          return {.scales = std::vector<float>(8, 0.5f),
                  .float_zero_points = std::vector<float>(8, -2.0f)};
        },
        lazy_options);
    TensorMLD x({.type = Type::kFP32, .shape = {1, 2, 64}});
    TensorMLD w({.name = "lazy_cached.weight",
                 .type = Type::kI4,
                 .shape = {4, 64},
                 .buffer = std::vector<int4_t>(4 * 32, int4_t{0}),
                 .quantization = bwq});
    local_ctx.graph_tensor_to_handle[w.GetRaw()] = builder.AddTensor(
        ml_drift::BHWC(1, 1, 1, 128), ml_drift::DataType::kUint8);
    TensorMLD out = FullyConnected(x, w);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto map,
                                    AppendGpuModelBuilder(local_ctx, {out}));
    EXPECT_EQ(materialize_calls, 0);
    EXPECT_FALSE(bwq->IsMaterialized());
    ml_drift::GpuModel model;
    ASSERT_OK(
        builder.GetGpuModel({map.at(x.GetRaw()).id,
                             local_ctx.graph_tensor_to_handle[w.GetRaw()].id},
                            {map.at(out.GetRaw()).id}, &model));
    EXPECT_THAT(model.const_tensors, SizeIs(2));
  }

  // 2. Without cached descriptors, lazy `bwq` materializes on demand.
  {
    MlDriftBuildContext local_ctx;
    ml_drift::GpuModelBuilder builder(context_.gpu_model_builder->gpu_info(),
                                      ml_drift::ModelHints{},
                                      ml_drift::CalculationsPrecision::kF32,
                                      ml_drift::TensorStorageType::kBuffer);
    local_ctx.gpu_model_builder = &builder;
    local_ctx.default_float_type = ml_drift::DataType::kFloat32;

    int materialize_calls = 0;
    using ZeroPointPresence = BlockwiseQuantization::ZeroPointPresence;
    BlockwiseQuantization::LazyOptions lazy_options{
        .zero_points = ZeroPointPresence::kMayHaveFloat,
    };
    auto bwq = BlockwiseQuantization::CreateLazily(
        /*block_size=*/32, /*quantized_dimension=*/0, /*num_blocks=*/8,
        [&]() -> BlockwiseQuantization::MaterializedBlockwiseParams {
          ++materialize_calls;
          return {.scales = std::vector<float>(8, 0.5f),
                  .float_zero_points = std::vector<float>(8, -2.0f)};
        },
        lazy_options);
    TensorMLD x({.type = Type::kFP32, .shape = {1, 2, 64}});
    TensorMLD w({.name = "lazy_uncached.weight",
                 .type = Type::kI4,
                 .shape = {4, 64},
                 .buffer = std::vector<int4_t>(4 * 32, int4_t{0}),
                 .quantization = bwq});
    local_ctx.graph_tensor_to_handle[w.GetRaw()] = builder.AddTensor(
        ml_drift::BHWC(1, 1, 1, 128), ml_drift::DataType::kUint8);
    TensorMLD out = FullyConnected(x, w);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto map,
                                    AppendGpuModelBuilder(local_ctx, {out}));
    EXPECT_EQ(materialize_calls, 1);
    EXPECT_TRUE(bwq->IsMaterialized());
    ml_drift::GpuModel model;
    ASSERT_OK(
        builder.GetGpuModel({map.at(x.GetRaw()).id,
                             local_ctx.graph_tensor_to_handle[w.GetRaw()].id},
                            {map.at(out.GetRaw()).id}, &model));
    EXPECT_THAT(model.const_tensors, SizeIs(2));
  }

  // 3. Cached `:scale:` without `:zp:` for a float-ZP lazy weight falls
  // through to materialize instead of dropping zero points.
  {
    absl::flat_hash_map<std::string, ml_drift::TensorDescriptor> shared;
    const int dt = static_cast<int>(ml_drift::DataType::kFloat32);
    shared.emplace(absl::StrCat("lazy_partial.weight:scale:", dt),
                   make_desc(std::vector<float>(8, 0.25f)));

    MlDriftBuildContext local_ctx;
    ml_drift::GpuModelBuilder builder(context_.gpu_model_builder->gpu_info(),
                                      ml_drift::ModelHints{},
                                      ml_drift::CalculationsPrecision::kF32,
                                      ml_drift::TensorStorageType::kBuffer);
    local_ctx.gpu_model_builder = &builder;
    local_ctx.default_float_type = ml_drift::DataType::kFloat32;
    local_ctx.shared_weight_descriptors = &shared;

    int materialize_calls = 0;
    using ZeroPointPresence = BlockwiseQuantization::ZeroPointPresence;
    BlockwiseQuantization::LazyOptions lazy_options{
        .zero_points = ZeroPointPresence::kMayHaveFloat,
    };
    auto bwq = BlockwiseQuantization::CreateLazily(
        /*block_size=*/32, /*quantized_dimension=*/0, /*num_blocks=*/8,
        [&]() -> BlockwiseQuantization::MaterializedBlockwiseParams {
          ++materialize_calls;
          return {.scales = std::vector<float>(8, 0.5f),
                  .float_zero_points = std::vector<float>(8, -2.0f)};
        },
        lazy_options);
    TensorMLD x({.type = Type::kFP32, .shape = {1, 2, 64}});
    TensorMLD w({.name = "lazy_partial.weight",
                 .type = Type::kI4,
                 .shape = {4, 64},
                 .buffer = std::vector<int4_t>(4 * 32, int4_t{0}),
                 .quantization = bwq});
    local_ctx.graph_tensor_to_handle[w.GetRaw()] = builder.AddTensor(
        ml_drift::BHWC(1, 1, 1, 128), ml_drift::DataType::kUint8);
    TensorMLD out = FullyConnected(x, w);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto map,
                                    AppendGpuModelBuilder(local_ctx, {out}));
    EXPECT_EQ(materialize_calls, 1);
    EXPECT_TRUE(bwq->IsMaterialized());
    EXPECT_TRUE(local_ctx.weight_name_to_handle.contains(
        absl::StrCat("lazy_partial.weight:zp:", dt)));
    ml_drift::GpuModel model;
    ASSERT_OK(
        builder.GetGpuModel({map.at(x.GetRaw()).id,
                             local_ctx.graph_tensor_to_handle[w.GetRaw()].id},
                            {map.at(out.GetRaw()).id}, &model));
    EXPECT_THAT(model.const_tensors, SizeIs(2));
  }
}

}  // namespace litert::tensor
