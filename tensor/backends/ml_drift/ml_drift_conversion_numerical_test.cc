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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/any.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/compiler_options.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/backends/ml_drift/ml_drift_buffer.h"
#include "tensor/backends/ml_drift/ml_drift_conversion.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/backends/testing/numerical_test_bridge.h"
#include "tensor/backends/testing/numerical_test_suite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/internal/type_id.h"
#include "tensor/runners/ml_drift/ml_drift_dummy_weight_creator.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"

#define XNN_EXTRA_BYTES 16

namespace litert::tensor {
namespace {

// Uploads the int8 values `data` to the input tensor `id` of `ic`, which has no
// `SetInputTensor` overload for int8 tensors.
absl::Status UploadInt8Tensor(ml_drift::cl::InferenceContext& ic,
                              ml_drift::ValueId id,
                              absl::Span<const std::byte> data,
                              ml_drift::cl::CLCommandQueue* queue) {
  ml_drift::cl::Tensor* gpu_tensor = ic.GetTensor(id);
  if (gpu_tensor == nullptr) {
    return absl::NotFoundError(absl::StrCat("Tensor ", id, " not found."));
  }
  ml_drift::TensorDescriptor descriptor = gpu_tensor->GetDescriptor();
  descriptor.UploadData(reinterpret_cast<const int8_t*>(data.data()));
  return gpu_tensor->UploadDescriptorData(descriptor, queue);
}

class MlDriftTestBackendBridge : public TestBackendBridge {
 public:
  explicit MlDriftTestBackendBridge(
      ml_drift::CalculationsPrecision precision =
          ml_drift::CalculationsPrecision::kF32)
      : precision_(precision) {}

  ~MlDriftTestBackendBridge() override {
    if (env_.queue()) {
      (void)env_.queue()->WaitForCompletion();
    }
  }

  const std::vector<std::string>& last_node_names() const {
    return last_node_names_;
  }

  absl::Status Initialize() override {
    LRT_TENSOR_RETURN_IF_ERROR(ml_drift::cl::LoadOpenCL());
    LRT_TENSOR_RETURN_IF_ERROR(ml_drift::cl::CreateEnvironment(&env_));
    ml_drift::CreateGpuModelInfo ci = {
        .precision = precision_,
        .storage_type =
            ml_drift::cl::GetFastestStorageType(env_.device().GetInfo()),
    };
    model_builder_ = std::make_unique<ml_drift::GpuModelBuilder>(
        env_.device().GetInfo(), ci.hints, ci.precision, ci.storage_type);
    context_.gpu_model_builder = model_builder_.get();
    context_.default_float_type =
        precision_ == ml_drift::CalculationsPrecision::kF16
            ? ml_drift::DataType::kFloat16
            : ml_drift::DataType::kFloat32;
    dummy_weight_creator_ =
        std::make_unique<MlDriftDummyWeightCreator>(model_builder_.get());
    return absl::OkStatus();
  }

  absl::Status BuildGraph(absl::Span<const TensorHandle> inputs,
                          absl::Span<const TensorHandle> outputs) override {
    // 1. Process and pack constant FullyConnected weights automatically.
    LRT_TENSOR_ASSIGN_OR_RETURN(auto execution_plan, GetExecutionPlan(outputs));
    for (const auto* op : execution_plan) {
      if (op->GetTypeId() ==
              internal::TypeId::Get<graph::FullyConnectedOperation>() ||
          op->GetTypeId() ==
              internal::TypeId::Get<graph::EmbeddingLookupOperation>()) {
        const auto& weights_tensor = op->inputs[1];
        LRT_TENSOR_ASSIGN_OR_RETURN(auto info, graph::GetInfo(weights_tensor));
        if (info.buffer != nullptr &&
            !context_.graph_tensor_to_handle.contains(weights_tensor)) {
          auto locked_weights = info.buffer->Lock();
          ml_drift::GpuModelBuilder::TensorHandle tensor_handle;
          Tensor<MlDriftMixinTag> weights_tensor_wrapper(weights_tensor);
          if (info.type == Type::kFP32) {
            absl::Span<const float> weights_span(
                reinterpret_cast<const float*>(locked_weights.data()),
                locked_weights.size() / sizeof(float));
            LRT_TENSOR_ASSIGN_OR_RETURN(
                tensor_handle, dummy_weight_creator_->CreateFcWeights<float>(
                                   weights_tensor_wrapper, weights_span,
                                   context_.default_float_type));
          } else if (info.type == Type::kI8 || info.type == Type::kI4 ||
                     info.type == Type::kI2) {
            absl::Span<const int8_t> weights_span(
                reinterpret_cast<const int8_t*>(locked_weights.data()),
                locked_weights.size() / sizeof(int8_t));
            LRT_TENSOR_ASSIGN_OR_RETURN(
                tensor_handle, dummy_weight_creator_->CreateFcWeights<int8_t>(
                                   weights_tensor_wrapper, weights_span,
                                   context_.default_float_type));
          } else {
            return absl::UnimplementedError(
                "Unsupported weights type for FC rearrangement");
          }
          context_.graph_tensor_to_handle[weights_tensor] = tensor_handle;
        }
      }
    }

    // 2. Build the GPU model
    std::vector<TensorHandle> output_handles(outputs.begin(), outputs.end());
    auto status_or = AppendGpuModelBuilder(context_, output_handles);
    if (!status_or.ok()) {
      absl::Status s = status_or.status();
      if (s.code() == absl::StatusCode::kInvalidArgument &&
          absl::StrContains(s.message(), "does not implement")) {
        return absl::UnimplementedError(s.message());
      }
      return s;
    }
    tensor_to_handle_ = std::move(*status_or);

    std::vector<ml_drift::ValueId> graph_input_ids;
    for (const auto& input : inputs) {
      graph_input_ids.push_back(tensor_to_handle_.at(input.GetRaw()).id);
    }

    std::vector<ml_drift::ValueId> graph_output_ids;
    for (const auto& output : outputs) {
      graph_output_ids.push_back(tensor_to_handle_.at(output.GetRaw()).id);
    }

    ml_drift::GpuModel gpu_model;
    LRT_TENSOR_RETURN_IF_ERROR(model_builder_->GetGpuModel(
        graph_input_ids, graph_output_ids, &gpu_model));
    last_node_names_.clear();
    for (const auto& node : gpu_model.nodes) {
      last_node_names_.push_back(node.name);
    }

    ml_drift::CreateGpuModelInfo ci = {
        .precision = precision_,
        .storage_type =
            ml_drift::cl::GetFastestStorageType(env_.device().GetInfo()),
    };
    LRT_TENSOR_RETURN_IF_ERROR(ic_.InitFromGpuModel(ci, &gpu_model, &env_));
    return absl::OkStatus();
  }

  absl::Status SetInput(const TensorHandle& tensor,
                        absl::Span<const std::byte> data) override {
    const auto& th = tensor_to_handle_.at(tensor.GetRaw());
    if (th.tensor_desc.GetDataType() == ml_drift::DataType::kFloat32 ||
        th.tensor_desc.GetDataType() == ml_drift::DataType::kFloat16) {
      ml_drift::TensorFloat32 t;
      t.id = th.id;
      t.shape = th.tensor_desc.GetBHWCShape();
      t.data.assign(reinterpret_cast<const float*>(data.data()),
                    reinterpret_cast<const float*>(data.data() + data.size()));
      return ic_.SetInputTensor(t.id, t, env_.queue());
    } else if (th.tensor_desc.GetDataType() == ml_drift::DataType::kBool) {
      ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kBool> t;
      t.id = th.id;
      t.shape = th.tensor_desc.GetBHWCShape();
      t.data.assign(reinterpret_cast<const bool*>(data.data()),
                    reinterpret_cast<const bool*>(data.data() + data.size()));
      return ic_.SetInputTensor(t.id, t, env_.queue());
    } else if (th.tensor_desc.GetDataType() == ml_drift::DataType::kInt32) {
      ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> t;
      t.id = th.id;
      t.shape = th.tensor_desc.GetBHWCShape();
      t.data.assign(
          reinterpret_cast<const int32_t*>(data.data()),
          reinterpret_cast<const int32_t*>(data.data() + data.size()));
      return ic_.SetInputTensor(t.id, t, env_.queue());
    } else if (th.tensor_desc.GetDataType() == ml_drift::DataType::kInt8) {
      return UploadInt8Tensor(ic_, th.id, data, env_.queue());
    }
    return absl::UnimplementedError(
        "Unsupported input type for ML Drift bridge");
  }

  absl::Status Execute() override {
    LRT_TENSOR_RETURN_IF_ERROR(ic_.AddToQueue(env_.queue()));
    return env_.queue()->WaitForCompletion();
  }

  absl::Status GetOutput(const TensorHandle& tensor,
                         absl::Span<std::byte> data) override {
    const auto& output_handle = tensor_to_handle_.at(tensor.GetRaw());
    if (output_handle.tensor_desc.GetDataType() == ml_drift::DataType::kBool) {
      ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kBool>
          output_tensor_bool;
      LRT_TENSOR_RETURN_IF_ERROR(ic_.GetOutputTensor(
          output_handle.id, env_.queue(), &output_tensor_bool));
      bool* dst = reinterpret_cast<bool*>(data.data());
      for (size_t i = 0; i < output_tensor_bool.data.size(); ++i) {
        dst[i] = output_tensor_bool.data[i];
      }
    } else if (output_handle.tensor_desc.GetDataType() ==
               ml_drift::DataType::kInt32) {
      ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32>
          output_tensor_int32;
      LRT_TENSOR_RETURN_IF_ERROR(ic_.GetOutputTensor(
          output_handle.id, env_.queue(), &output_tensor_int32));
      std::memcpy(data.data(), output_tensor_int32.data.data(), data.size());
    } else {
      ml_drift::TensorFloat32 output_tensor_float;
      LRT_TENSOR_RETURN_IF_ERROR(ic_.GetOutputTensor(
          output_handle.id, env_.queue(), &output_tensor_float));
      std::memcpy(data.data(), output_tensor_float.data.data(), data.size());
    }
    return absl::OkStatus();
  }

 private:
  ml_drift::CalculationsPrecision precision_ =
      ml_drift::CalculationsPrecision::kF32;
  std::vector<std::string> last_node_names_;
  ml_drift::cl::Environment env_;
  std::unique_ptr<ml_drift::GpuModelBuilder> model_builder_;
  MlDriftBuildContext context_;
  std::unique_ptr<MlDriftDummyWeightCreator> dummy_weight_creator_;

  absl::flat_hash_map<graph::Tensor, ml_drift::GpuModelBuilder::TensorHandle>
      tensor_to_handle_;
  ml_drift::cl::InferenceContext ic_;
};

struct MlDriftTraits {
  using Tag = MlDriftMixinTag;
  static std::unique_ptr<TestBackendBridge> CreateBridge() {
    return std::make_unique<MlDriftTestBackendBridge>();
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(MlDrift, NumericalTestSuite, MlDriftTraits);

using ::litert::tensor::IsOk;
using ::testing::FloatNear;
using ::testing::Not;
using ::testing::Pointwise;

using TensorMLD = Tensor<MlDriftMixinTag>;

class MlDriftConversionNumericalTest : public ::testing::Test {
 protected:
  struct TestTensor {
    TensorMLD tensor;
    std::any data;
  };

  void SetUp() override {
    ASSERT_OK(ml_drift::cl::LoadOpenCL());
    ASSERT_OK(ml_drift::cl::CreateEnvironment(&env_));
    ml_drift::CreateGpuModelInfo ci = {
        .precision = ml_drift::CalculationsPrecision::kF32,
        .storage_type =
            ml_drift::cl::GetFastestStorageType(env_.device().GetInfo()),
    };
    model_builder_ = std::make_unique<ml_drift::GpuModelBuilder>(
        env_.device().GetInfo(), ci.hints, ci.precision, ci.storage_type);
    context_.gpu_model_builder = model_builder_.get();
    context_.default_float_type = ml_drift::DataType::kFloat32;
    dummy_weight_creator_ =
        std::make_unique<MlDriftDummyWeightCreator>(model_builder_.get());
  }

  void RunTest(TensorMLD output, const std::vector<TestTensor>& inputs,
               const std::vector<float>& expected_output);

  void RunTest(std::vector<TensorMLD> outputs,
               const std::vector<TestTensor>& inputs,
               const std::vector<std::vector<float>>& expected_outputs);

  ml_drift::cl::Environment env_;
  std::unique_ptr<ml_drift::GpuModelBuilder> model_builder_;
  MlDriftBuildContext context_;
  std::unique_ptr<MlDriftDummyWeightCreator> dummy_weight_creator_;
};

void MlDriftConversionNumericalTest::RunTest(
    TensorMLD output, const std::vector<TestTensor>& inputs,
    const std::vector<float>& expected_output) {
  RunTest(std::vector<TensorMLD>({output}), inputs, {expected_output});
}

void MlDriftConversionNumericalTest::RunTest(
    std::vector<TensorMLD> outputs, const std::vector<TestTensor>& inputs,
    const std::vector<std::vector<float>>& expected_outputs) {
  std::vector<TensorHandle> output_handles;
  for (const auto& output : outputs) {
    output_handles.push_back(output);
  }
  auto tensor_to_handle_or = AppendGpuModelBuilder(context_, output_handles);
  ASSERT_OK(tensor_to_handle_or);
  auto& tensor_to_handle = *tensor_to_handle_or;

  std::vector<ml_drift::ValueId> graph_input_ids;
  for (const auto& input : inputs) {
    graph_input_ids.push_back(tensor_to_handle.at(input.tensor.GetRaw()).id);
  }

  std::vector<ml_drift::ValueId> graph_output_ids;
  for (const auto& output : outputs) {
    graph_output_ids.push_back(tensor_to_handle.at(output.GetRaw()).id);
  }
  ml_drift::GpuModel gpu_model;
  ASSERT_OK(model_builder_->GetGpuModel(graph_input_ids, graph_output_ids,
                                        &gpu_model));

  ml_drift::cl::InferenceContext ic;
  ml_drift::CreateGpuModelInfo ci = {
      .precision = ml_drift::CalculationsPrecision::kF32,
      .storage_type =
          ml_drift::cl::GetFastestStorageType(env_.device().GetInfo()),
  };
  ASSERT_OK(ic.InitFromGpuModel(ci, &gpu_model, &env_));

  for (const auto& input : inputs) {
    const auto& th = tensor_to_handle.at(input.tensor.GetRaw());
    if (const auto* float_data =
            std::any_cast<std::vector<float>>(&input.data)) {
      ml_drift::TensorFloat32 t;
      t.id = th.id;
      t.shape = th.tensor_desc.GetBHWCShape();
      t.data = *float_data;
      ASSERT_OK(ic.SetInputTensor(t.id, t, env_.queue()));
    } else if (const auto* bool_data =
                   std::any_cast<std::vector<bool>>(&input.data)) {
      ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kBool> t;
      t.id = th.id;
      t.shape = th.tensor_desc.GetBHWCShape();
      t.data.assign(bool_data->begin(), bool_data->end());
      ASSERT_OK(ic.SetInputTensor(t.id, t, env_.queue()));
    } else if (const auto* int8_data =
                   std::any_cast<std::vector<int8_t>>(&input.data)) {
      ASSERT_OK(UploadInt8Tensor(
          ic, th.id,
          absl::MakeConstSpan(
              reinterpret_cast<const std::byte*>(int8_data->data()),
              int8_data->size()),
          env_.queue()));
    } else if (const auto* int32_data =
                   std::any_cast<std::vector<int32_t>>(&input.data)) {
      ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> t;
      t.id = th.id;
      t.shape = th.tensor_desc.GetBHWCShape();
      t.data.assign(int32_data->begin(), int32_data->end());
      ASSERT_OK(ic.SetInputTensor(t.id, t, env_.queue()));
    }
  }

  ASSERT_OK(ic.AddToQueue(env_.queue()));
  ASSERT_OK(env_.queue()->WaitForCompletion());

  ml_drift::TensorFloat32 output_tensor_float;
  ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kBool>
      output_tensor_bool;
  ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32>
      output_tensor_int32;
  for (int i = 0; i < outputs.size(); ++i) {
    const auto& output_handle = tensor_to_handle.at(outputs[i].GetRaw());
    if (output_handle.tensor_desc.GetDataType() == ml_drift::DataType::kBool) {
      ASSERT_OK(ic.GetOutputTensor(output_handle.id, env_.queue(),
                                   &output_tensor_bool));
      std::vector<float> actual_output_float;
      actual_output_float.reserve(output_tensor_bool.data.size());
      for (bool val : output_tensor_bool.data) {
        actual_output_float.push_back(val ? 1.0f : 0.0f);
      }
      EXPECT_THAT(actual_output_float,
                  Pointwise(FloatNear(1e-6), expected_outputs[i]));
    } else if (output_handle.tensor_desc.GetDataType() ==
               ml_drift::DataType::kInt32) {
      ASSERT_OK(ic.GetOutputTensor(output_handle.id, env_.queue(),
                                   &output_tensor_int32));
      std::vector<float> actual_output_float;
      actual_output_float.reserve(output_tensor_int32.data.size());
      for (int val : output_tensor_int32.data) {
        actual_output_float.push_back(static_cast<float>(val));
      }
      EXPECT_THAT(actual_output_float,
                  Pointwise(FloatNear(1e-6), expected_outputs[i]));
    } else {
      ASSERT_OK(ic.GetOutputTensor(output_handle.id, env_.queue(),
                                   &output_tensor_float));
      EXPECT_THAT(output_tensor_float.data,
                  Pointwise(FloatNear(1e-6), expected_outputs[i]));
    }
  }
}

TEST_F(MlDriftConversionNumericalTest, QuantizeThenDequantizeOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Dequantize(Quantize(input, Type::kI8, {0.5}, {0}));
  RunTest(output, {{input, std::vector<float>{0.6f, 0.8f, 1.2f, 1.6f}}},
          {0.5f, 1.0f, 1.0f, 1.5f});
}

TEST_F(MlDriftConversionNumericalTest, DequantizeOp) {
  TensorMLD quant_input({.type = Type::kI8, .shape = {1, 2, 2, 1}});
  auto quant_params = std::make_shared<PerChannelAffineQuantization>();
  quant_params->scales = {0.25f};
  quant_params->zero_points = ZeroPoints::Int({0});
  quant_input.SetQuantization(quant_params);
  TensorMLD output = Dequantize(quant_input);
  RunTest(output, {{quant_input, std::vector<int8_t>{1, 2, -1, -2}}},
          {0.25f, 0.5f, -0.25f, -0.5f});
}

TEST_F(MlDriftConversionNumericalTest, DequantizeConstantOp) {
  // [[1, 3, 5], [-7, 9, 11]] quantized along their first dimension.
  TensorMLD weights(
      {.type = Type::kI8,
       .shape = {2, 3},
       .buffer = OwningCpuBuffer::Copy<Type::kI8>({1, 3, 5, -7, 9, 11}),
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>{0.5f, 2.0f}, std::vector<int64_t>{1, -1})});
  TensorMLD input({.type = Type::kFP32, .shape = {2, 3}});
  TensorMLD output = Add(Dequantize(weights), input);
  RunTest(output, {{input, std::vector<float>{1, 1, 1, 1, 1, 1}}},
          {1, 2, 3, -11, 21, 25});
}

TEST_F(MlDriftConversionNumericalTest, GetTensorHandleWithMlDriftBuffer) {
  // 1. Create a graph and inference context to get a valid tensor ID.
  TensorMLD input({.type = Type::kFP32, .shape = {1, 1, 1, 1}});
  TensorMLD output = Add(input, input);
  std::vector<TensorHandle> output_handles = {output};
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto tensor_to_handle, AppendGpuModelBuilder(context_, output_handles));
  ml_drift::GpuModel gpu_model;
  ASSERT_OK(model_builder_->GetGpuModel(
      {tensor_to_handle.at(input.GetRaw()).id},
      {tensor_to_handle.at(output.GetRaw()).id}, &gpu_model));
  ml_drift::cl::InferenceContext ic;
  ml_drift::CreateGpuModelInfo ci = {
      .precision = ml_drift::CalculationsPrecision::kF32,
      .storage_type =
          ml_drift::cl::GetFastestStorageType(env_.device().GetInfo()),
  };
  ASSERT_OK(ic.InitFromGpuModel(ci, &gpu_model, &env_));

  // 2. Create an MlDriftBuffer associated with an ID from the context.
  const auto& tensor_id = tensor_to_handle.at(input.GetRaw()).id;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<Buffer> ml_drift_buffer,
      MlDriftBuffer::Create(tensor_id, ic, *env_.queue(), 1 * 4));
  input.SetBuffer(ml_drift_buffer);
  context_.graph_tensor_to_handle.clear();
  context_.graph_tensor_to_handle[input.GetRaw()] =
      tensor_to_handle.at(input.GetRaw());

  // 3. Call GetTensorHandle and verify the ID.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto handle,
                                  GetTensorHandle(context_, input.GetRaw()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto& mld_buffer,
                                  ml_drift_buffer->As<MlDriftBuffer>());
  EXPECT_EQ(handle.id, mld_buffer.value_id());
  EXPECT_EQ(handle.id, tensor_id);
}

class MlDriftBufferTypeIdTest : public MlDriftConversionNumericalTest {
 protected:
  void SetUp() override {
    MlDriftConversionNumericalTest::SetUp();
    TensorMLD input({.type = Type::kFP32, .shape = {1, 1, 1, 1}});
    TensorMLD output = Add(input, input);
    std::vector<TensorHandle> output_handles = {output};
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto tensor_to_handle, AppendGpuModelBuilder(context_, output_handles));
    ASSERT_OK(model_builder_->GetGpuModel(
        {tensor_to_handle.at(input.GetRaw()).id},
        {tensor_to_handle.at(output.GetRaw()).id}, &gpu_model_));
    ml_drift::CreateGpuModelInfo ci = {
        .precision = ml_drift::CalculationsPrecision::kF32,
        .storage_type =
            ml_drift::cl::GetFastestStorageType(env_.device().GetInfo()),
    };
    ASSERT_OK(ic_.InitFromGpuModel(ci, &gpu_model_, &env_));

    const auto& tensor_id = tensor_to_handle.at(input.GetRaw()).id;
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        ml_drift_buffer_,
        MlDriftBuffer::Create(tensor_id, ic_, *env_.queue(), 1 * 4));
  }

  ml_drift::GpuModel gpu_model_;
  ml_drift::cl::InferenceContext ic_;
  std::shared_ptr<Buffer> ml_drift_buffer_;
};

TEST_F(MlDriftBufferTypeIdTest, GetTypeId) {
  EXPECT_EQ(ml_drift_buffer_->GetTypeId(),
            internal::TypeId::Get<MlDriftBuffer>());
}

TEST_F(MlDriftBufferTypeIdTest, IsA) {
  EXPECT_TRUE(ml_drift_buffer_->IsA(internal::TypeId::Get<MlDriftBuffer>()));
  EXPECT_FALSE(ml_drift_buffer_->IsA(internal::TypeId::Get<SpanCpuBuffer>()));
}

TEST_F(MlDriftBufferTypeIdTest, As) {
  Buffer& buffer_ref = *ml_drift_buffer_;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(MlDriftBuffer & buffer_as,
                                  buffer_ref.As<MlDriftBuffer>());
  EXPECT_EQ(&buffer_as, ml_drift_buffer_.get());
  EXPECT_THAT(buffer_ref.As<SpanCpuBuffer>(), Not(IsOk()));

  const Buffer& const_buffer_ref = *ml_drift_buffer_;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const MlDriftBuffer& const_buffer_as,
                                  const_buffer_ref.As<MlDriftBuffer>());
  EXPECT_EQ(&const_buffer_as, ml_drift_buffer_.get());
  EXPECT_THAT(const_buffer_ref.As<SpanCpuBuffer>(), Not(IsOk()));
}

TEST_F(MlDriftConversionNumericalTest, SumAndMulOp) {
  TensorMLD input({.type = Type::kFP32, .shape = {1, 1, 4}});
  TensorMLD mul_input({.type = Type::kFP32, .shape = {1, 1, 1}});
  TensorMLD axis({.type = Type::kI32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>({2})});
  TensorMLD sum_output = Sum(input, axis, false);
  TensorMLD mul_output = Mul(sum_output, mul_input);

  RunTest(mul_output,
          {{input, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}},
           {mul_input, std::vector<float>{5.0f}}},
          {50.0f});
}

TEST_F(MlDriftConversionNumericalTest, BufferExecutionTest) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Add(input1, input2);

  std::vector<TensorHandle> output_handles = {output};
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto tensor_to_handle, AppendGpuModelBuilder(context_, output_handles));

  std::vector<ml_drift::ValueId> graph_input_ids;
  graph_input_ids.push_back(tensor_to_handle.at(input1.GetRaw()).id);
  graph_input_ids.push_back(tensor_to_handle.at(input2.GetRaw()).id);

  std::vector<ml_drift::ValueId> graph_output_ids;
  graph_output_ids.push_back(tensor_to_handle.at(output.GetRaw()).id);

  ml_drift::GpuModel gpu_model;
  ASSERT_OK(model_builder_->GetGpuModel(graph_input_ids, graph_output_ids,
                                        &gpu_model));

  ml_drift::cl::InferenceContext ic;
  ml_drift::CreateGpuModelInfo ci = {
      .precision = ml_drift::CalculationsPrecision::kF32,
      .storage_type =
          ml_drift::cl::GetFastestStorageType(env_.device().GetInfo()),
  };
  ASSERT_OK(ic.InitFromGpuModel(ci, &gpu_model, &env_));

  auto& th1 = tensor_to_handle.at(input1.GetRaw());
  auto& th2 = tensor_to_handle.at(input2.GetRaw());
  auto& th_out = tensor_to_handle.at(output.GetRaw());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto buffer1, MlDriftBuffer::Create(th1.id, ic, *env_.queue(), 4 * 4));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto buffer2, MlDriftBuffer::Create(th2.id, ic, *env_.queue(), 4 * 4));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto buffer_out,
      MlDriftBuffer::Create(th_out.id, ic, *env_.queue(), 4 * 4));

  {
    auto locked_data_1 = buffer1->LockMutable().As<float>();
    absl::c_iota(locked_data_1, 1);
    auto locked_data_2 = buffer2->LockMutable().As<float>();
    absl::c_iota(locked_data_2, 2);
  }  // Locked data is synced at then end of the scope.

  ASSERT_OK(ic.AddToQueue(env_.queue()));
  ASSERT_OK(env_.queue()->WaitForCompletion());

  EXPECT_THAT(buffer_out->Lock().As<const float>(),
              Pointwise(FloatNear(1e-6), {3.0f, 5.0f, 7.0f, 9.0f}));
}

TEST_F(MlDriftConversionNumericalTest, PersistentWeightsTest) {
  // Simulate LLM weight use case: Weights are inputs, uploaded once, reused.
  // Graph: Weights (Input 0) + Activations (Input 1) -> Output
  TensorMLD weights({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD activations({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD output = Add(weights, activations);

  std::vector<TensorHandle> output_handles = {output};
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto tensor_to_handle, AppendGpuModelBuilder(context_, output_handles));

  std::vector<ml_drift::ValueId> graph_input_ids;
  graph_input_ids.push_back(tensor_to_handle.at(weights.GetRaw()).id);
  graph_input_ids.push_back(tensor_to_handle.at(activations.GetRaw()).id);

  std::vector<ml_drift::ValueId> graph_output_ids;
  graph_output_ids.push_back(tensor_to_handle.at(output.GetRaw()).id);

  ml_drift::GpuModel gpu_model;
  ASSERT_OK(model_builder_->GetGpuModel(graph_input_ids, graph_output_ids,
                                        &gpu_model));

  ml_drift::cl::InferenceContext ic;
  ml_drift::CreateGpuModelInfo ci = {
      .precision = ml_drift::CalculationsPrecision::kF32,
      .storage_type =
          ml_drift::cl::GetFastestStorageType(env_.device().GetInfo()),
  };
  ASSERT_OK(ic.InitFromGpuModel(ci, &gpu_model, &env_));

  auto& th_weights = tensor_to_handle.at(weights.GetRaw());
  auto& th_act = tensor_to_handle.at(activations.GetRaw());
  auto& th_out = tensor_to_handle.at(output.GetRaw());

  // 1. Pre-allocate and Upload Weights (Once)
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto weight_buffer,
      MlDriftBuffer::Create(th_weights.id, ic, *env_.queue(), 4 * 4));

  {
    auto data = weight_buffer->LockMutable().As<float>();
    // Weights: 10, 10, 10, 10
    absl::c_fill(data, 10.0f);
  }  // Upload happens here via SetInputTensor

  // 2. Execution Loop (Simulated)
  for (int step = 0; step < 3; ++step) {
    // Create per-step input/output buffers
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto act_buffer,
        MlDriftBuffer::Create(th_act.id, ic, *env_.queue(), 4 * 4));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto out_buffer,
        MlDriftBuffer::Create(th_out.id, ic, *env_.queue(), 4 * 4));

    // Fill activations: step, step, step, step
    {
      auto data = act_buffer->LockMutable().As<float>();
      absl::c_fill(data, static_cast<float>(step));
    }

    // Execute (Manual logic similar to BackendModule::Execute)
    // For Weights: buffer matches context tensor -> Zero Copy (Expected)
    // For Act: buffer matches context tensor -> Zero Copy
    // For Out: buffer matches context tensor -> Zero Copy

    // Since we are manually using the buffer logic, we just ensure they are
    // attached. In a real scenario, BackendModule::Execute does this check.
    // Here we implicitly test that the buffer content persists on GPU.

    ASSERT_OK(ic.AddToQueue(env_.queue()));
    ASSERT_OK(env_.queue()->WaitForCompletion());

    // Verify Output
    // Expected: 10 + step
    EXPECT_THAT(
        out_buffer->Lock().As<const float>(),
        Pointwise(FloatNear(1e-6), std::vector<float>(4, 10.0f + step)));
  }
}

TEST(MlDriftBlockwiseQuantizedNumericalTest,
     BlockwiseQ8_0AndAffineInt4FullyConnectedAcrossDecodeTiledAndBulkRegimes) {
  constexpr int kOut = 8;
  constexpr int kIn = 64;
  constexpr int kBlockSize = 32;
  constexpr int kBlocksPerRow = kIn / kBlockSize;

  // 1. Blockwise INT8 (`Q8_0`) constant weights.
  std::vector<int8_t> q8_data(kOut * kIn);
  std::vector<float> q8_scales(kOut * kBlocksPerRow);
  std::vector<float> q8_ref(kOut * kIn);
  for (int r = 0; r < kOut; ++r) {
    for (int b = 0; b < kBlocksPerRow; ++b) {
      const int blk = r * kBlocksPerRow + b;
      q8_scales[blk] = 0.015625f * static_cast<float>((blk % 7) + 1);
      for (int j = 0; j < kBlockSize; ++j) {
        const int idx = r * kIn + b * kBlockSize + j;
        q8_data[idx] = static_cast<int8_t>((idx * 11 + 3) % 81 - 40);
        q8_ref[idx] = q8_scales[blk] * static_cast<float>(q8_data[idx]);
      }
    }
  }

  // 2. Affine blockwise INT4 (`Q4_1` / `Q4_K`) constant weights with positive
  // and negative floating-point zero-points.
  auto q4_buf = OwningCpuBuffer::Allocate<Type::kI4>(kOut * kIn);
  int4_t* q4_packed = q4_buf->Span<int4_t>().data();
  std::vector<float> q4_scales(kOut * kBlocksPerRow);
  std::vector<float> q4_zps(kOut * kBlocksPerRow);
  std::vector<float> q4_ref(kOut * kIn);
  for (int r = 0; r < kOut; ++r) {
    for (int b = 0; b < kBlocksPerRow; ++b) {
      const int blk = r * kBlocksPerRow + b;
      q4_scales[blk] = 0.03125f * static_cast<float>((blk % 5) + 1);
      q4_zps[blk] = (blk % 3 == 0) ? -3.25f : ((blk % 3 == 1) ? 2.5f : 0.0f);
      for (int j = 0; j < kBlockSize; j += 2) {
        const int elem = r * kIn + b * kBlockSize + j;
        const int qa = ((elem * 3 + 2) % 16) - 8;
        const int qb = ((elem * 7 + 5) % 16) - 8;
        q4_packed[elem / 2].a = static_cast<int8_t>(qa);
        q4_packed[elem / 2].b = static_cast<int8_t>(qb);
        q4_ref[elem] = q4_scales[blk] * (static_cast<float>(qa) - q4_zps[blk]);
        q4_ref[elem + 1] =
            q4_scales[blk] * (static_cast<float>(qb) - q4_zps[blk]);
      }
    }
  }

  // Exercise T = 1 (decode `fc1x1_int{4,8}`), T = 8 (`TiledQuantizedFC`), and
  // T = 128 (`WeightsConverter + Conv`).
  for (int seq_len : {1, 8, 128}) {
    std::vector<float> x_vals(seq_len * kIn);
    for (int i = 0; i < seq_len * kIn; ++i) {
      x_vals[i] = 0.05f * static_cast<float>((i * 13 + 7) % 19 - 9);
    }

    for (bool is_q4_affine : {false, true}) {
      MlDriftTestBackendBridge bridge;
      ASSERT_THAT(bridge.Initialize(), IsOk());

      Tensor<MlDriftMixinTag> x(
          {.name = "x", .type = Type::kFP32, .shape = {1, seq_len, kIn}});
      Tensor<MlDriftMixinTag> w =
          is_q4_affine
              ? Tensor<MlDriftMixinTag>(
                    {.name = "w_q4_k",
                     .type = Type::kI4,
                     .shape = {kOut, kIn},
                     .buffer = q4_buf,
                     .quantization = std::make_shared<BlockwiseQuantization>(
                         q4_scales, ZeroPoints::Float(q4_zps), kBlockSize,
                         /*quantized_dimension=*/0)})
              : Tensor<MlDriftMixinTag>(
                    {.name = "w_q8_0",
                     .type = Type::kI8,
                     .shape = {kOut, kIn},
                     .buffer = q8_data,
                     .quantization = std::make_shared<BlockwiseQuantization>(
                         q8_scales, std::vector<int64_t>{}, kBlockSize,
                         /*quantized_dimension=*/0)});
      Tensor<MlDriftMixinTag> y = FullyConnected(x, w);

      ASSERT_THAT(bridge.BuildGraph({TensorHandle(x)}, {TensorHandle(y)}),
                  IsOk());
      ASSERT_THAT(
          bridge.SetInput(TensorHandle(x),
                          absl::MakeConstSpan(
                              reinterpret_cast<const std::byte*>(x_vals.data()),
                              x_vals.size() * sizeof(float))),
          IsOk());
      ASSERT_THAT(bridge.Execute(), IsOk());

      std::vector<float> actual(seq_len * kOut, 0.0f);
      ASSERT_THAT(
          bridge.GetOutput(
              TensorHandle(y),
              absl::MakeSpan(reinterpret_cast<std::byte*>(actual.data()),
                             actual.size() * sizeof(float))),
          IsOk());

      const std::vector<float>& w_ref = is_q4_affine ? q4_ref : q8_ref;
      std::vector<float> expected(seq_len * kOut, 0.0f);
      for (int t = 0; t < seq_len; ++t) {
        for (int o = 0; o < kOut; ++o) {
          for (int i = 0; i < kIn; ++i) {
            expected[t * kOut + o] += x_vals[t * kIn + i] * w_ref[o * kIn + i];
          }
        }
      }
      // ML Drift stores blockwise scale/zero-point tensors in FP16 on some GPU
      // paths (`WeightsConverter` to float16 or fp16 scale textures), while our
      // chosen scales (`0.015625`, `0.03125`) and zero-points (`-3.25`, `2.5`)
      // are exact in FP16; allow 2e-3 for FP16 weight conversion at T = 128.
      EXPECT_THAT(actual,
                  testing::Pointwise(testing::FloatNear(2e-3f), expected))
          << "seq_len=" << seq_len << " is_q4_affine=" << is_q4_affine;
    }
  }
}

TEST(MlDriftBlockwiseQuantizedNumericalTest,
     QuantizedEmbeddingLookupAndTiedFullyConnectedShareWeights) {
  constexpr int kVocab = 12;
  constexpr int kDim = 64;
  constexpr int kBlockSize = 32;
  constexpr int kBlocksPerRow = kDim / kBlockSize;
  constexpr int kSeqLen = 3;

  std::vector<int8_t> q8_data(kVocab * kDim);
  std::vector<float> q8_scales(kVocab * kBlocksPerRow);
  std::vector<float> q8_ref(kVocab * kDim);
  auto q4_buf = OwningCpuBuffer::Allocate<Type::kI4>(kVocab * kDim);
  int4_t* q4_packed = q4_buf->Span<int4_t>().data();
  std::vector<float> q4_scales(kVocab * kBlocksPerRow);
  std::vector<float> q4_zps(kVocab * kBlocksPerRow);
  std::vector<float> q4_ref(kVocab * kDim);

  for (int v = 0; v < kVocab; ++v) {
    for (int b = 0; b < kBlocksPerRow; ++b) {
      const int blk = v * kBlocksPerRow + b;
      q8_scales[blk] = 0.015625f * static_cast<float>((blk % 6) + 1);
      q4_scales[blk] = 0.03125f * static_cast<float>((blk % 4) + 1);
      q4_zps[blk] = (blk % 2 == 0) ? -2.5f : 1.75f;
      for (int j = 0; j < kBlockSize; ++j) {
        const int idx = v * kDim + b * kBlockSize + j;
        q8_data[idx] = static_cast<int8_t>((idx * 9 + 5) % 61 - 30);
        q8_ref[idx] = q8_scales[blk] * static_cast<float>(q8_data[idx]);
      }
      for (int j = 0; j < kBlockSize; j += 2) {
        const int idx = v * kDim + b * kBlockSize + j;
        const int qa = ((idx * 5 + 1) % 16) - 8;
        const int qb = ((idx * 3 + 7) % 16) - 8;
        q4_packed[idx / 2].a = static_cast<int8_t>(qa);
        q4_packed[idx / 2].b = static_cast<int8_t>(qb);
        q4_ref[idx] = q4_scales[blk] * (static_cast<float>(qa) - q4_zps[blk]);
        q4_ref[idx + 1] =
            q4_scales[blk] * (static_cast<float>(qb) - q4_zps[blk]);
      }
    }
  }

  const std::vector<int32_t> token_ids = {2, 9, 5};
  for (const char* weight_name : {"token_embd.weight", "custom_emb.weight"}) {
    for (bool is_q4_affine : {false, true}) {
      MlDriftTestBackendBridge bridge;
      ASSERT_THAT(bridge.Initialize(), IsOk());

      Tensor<MlDriftMixinTag> ids(
          {.name = "ids", .type = Type::kI32, .shape = {1, kSeqLen}});
      Tensor<MlDriftMixinTag> emb_w =
          is_q4_affine
              ? Tensor<MlDriftMixinTag>(
                    {.name = weight_name,
                     .type = Type::kI4,
                     .shape = {kVocab, kDim},
                     .buffer = q4_buf,
                     .quantization = std::make_shared<BlockwiseQuantization>(
                         q4_scales, ZeroPoints::Float(q4_zps), kBlockSize,
                         /*quantized_dimension=*/0)})
              : Tensor<MlDriftMixinTag>(
                    {.name = weight_name,
                     .type = Type::kI8,
                     .shape = {kVocab, kDim},
                     .buffer = q8_data,
                     .quantization = std::make_shared<BlockwiseQuantization>(
                         q8_scales, std::vector<int64_t>{}, kBlockSize,
                         /*quantized_dimension=*/0)});
      Tensor<MlDriftMixinTag> embedded = EmbeddingLookup(ids, emb_w);
      Tensor<MlDriftMixinTag> logits = FullyConnected(embedded, emb_w);

      ASSERT_THAT(
          bridge.BuildGraph({TensorHandle(ids)},
                            {TensorHandle(embedded), TensorHandle(logits)}),
          IsOk());
      ASSERT_THAT(bridge.SetInput(
                      TensorHandle(ids),
                      absl::MakeConstSpan(
                          reinterpret_cast<const std::byte*>(token_ids.data()),
                          token_ids.size() * sizeof(int32_t))),
                  IsOk());
      ASSERT_THAT(bridge.Execute(), IsOk());

      std::vector<float> actual_emb(kSeqLen * kDim, 0.0f);
      std::vector<float> actual_logits(kSeqLen * kVocab, 0.0f);
      ASSERT_THAT(
          bridge.GetOutput(
              TensorHandle(embedded),
              absl::MakeSpan(reinterpret_cast<std::byte*>(actual_emb.data()),
                             actual_emb.size() * sizeof(float))),
          IsOk());
      ASSERT_THAT(
          bridge.GetOutput(
              TensorHandle(logits),
              absl::MakeSpan(reinterpret_cast<std::byte*>(actual_logits.data()),
                             actual_logits.size() * sizeof(float))),
          IsOk());

      const std::vector<float>& w_ref = is_q4_affine ? q4_ref : q8_ref;
      std::vector<float> expected_emb(kSeqLen * kDim, 0.0f);
      std::vector<float> expected_logits(kSeqLen * kVocab, 0.0f);
      for (int t = 0; t < kSeqLen; ++t) {
        const int tok = token_ids[t];
        for (int d = 0; d < kDim; ++d) {
          expected_emb[t * kDim + d] = w_ref[tok * kDim + d];
        }
        for (int v = 0; v < kVocab; ++v) {
          for (int d = 0; d < kDim; ++d) {
            expected_logits[t * kVocab + v] +=
                expected_emb[t * kDim + d] * w_ref[v * kDim + d];
          }
        }
      }
      EXPECT_THAT(actual_emb,
                  testing::Pointwise(testing::FloatNear(1e-4f), expected_emb));
      EXPECT_THAT(actual_logits, testing::Pointwise(testing::FloatNear(1e-3f),
                                                    expected_logits));
    }
  }
}

struct ScopedEnvOverride {
  std::string name;
  bool had_prev = false;
  std::string prev_value;

  ScopedEnvOverride(const char* env_name, const char* value) : name(env_name) {
    if (const char* old = std::getenv(env_name); old != nullptr) {
      had_prev = true;
      prev_value = old;
    }
    setenv(env_name, value, /*overwrite=*/1);
  }

  ~ScopedEnvOverride() {
    if (had_prev) {
      setenv(name.c_str(), prev_value.c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name.c_str());
    }
  }
};

std::shared_ptr<OwningCpuBuffer> PackSignedInt2(
    absl::Span<const int8_t> vals) {
  auto buf = OwningCpuBuffer::Allocate<Type::kI2>(vals.size());
  auto span = buf->Span<std::byte>();
  uint8_t* dst = reinterpret_cast<uint8_t*>(span.data());
  std::memset(dst, 0, span.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    const uint8_t q = static_cast<uint8_t>(vals[i]) & 0x03u;
    dst[i / 4] |= static_cast<uint8_t>(q << ((i % 4) * 2));
  }
  return buf;
}

std::shared_ptr<OwningCpuBuffer> PackSignedInt4(
    absl::Span<const int8_t> vals) {
  auto buf = OwningCpuBuffer::Allocate<Type::kI4>(vals.size());
  auto span = buf->Span<std::byte>();
  uint8_t* dst = reinterpret_cast<uint8_t*>(span.data());
  std::memset(dst, 0, span.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    const uint8_t q = static_cast<uint8_t>(vals[i]) & 0x0Fu;
    dst[i / 2] |= static_cast<uint8_t>(q << ((i % 2) * 4));
  }
  return buf;
}

std::shared_ptr<OwningCpuBuffer> PackSignedInt8(
    absl::Span<const int8_t> vals) {
  auto buf = OwningCpuBuffer::Allocate<Type::kI8>(vals.size());
  auto span = buf->Span<int8_t>();
  absl::c_copy(vals, span.data());
  return buf;
}

// Verifies `HoistedQuantizedFullyConnected` and FP32 promotion of
// FP16-subnormal scales across INT8 (including exact Gemma 4 E2B PLE
// subnormal scales ~1.2e-5..4.0e-5 at seq_len=1 and seq_len=10), INT4
// (per-channel + block-32 + bias), and INT2 (per-channel + lm_head softcap
// epilogue fusion + block-32 affine).
TEST(MlDriftHoistedQuantizedFcTest, Int2Int4Int8AndSubnormalScalesMatchRef) {
  struct Case {
    const char* name;
    Type weight_type;
    int seq_len;
    int out_ch;
    int in_ch;
    bool blockwise;
    bool symmetric;
    bool subnormal_scale;
    bool with_bias;
    bool with_softcap;
  };
  const Case kCases[] = {
      {"int8_subnormal_decode", Type::kI8, 1, 256, 1536, false, true, true,
       false, false},
      {"int8_subnormal_prefill", Type::kI8, 10, 256, 1536, false, true, true,
       false, false},
      {"int8_subnormal_long_prefill", Type::kI8, 256, 256, 1536, false, true,
       true, false, false},
      {"int4_block32_subnormal_long_prefill", Type::kI4, 256, 512, 256, true,
       true, true, false, false},
      {"int8_asymmetric_decode", Type::kI8, 1, 128, 64, false, false, false,
       false, false},
      {"int4_per_channel_decode_bias", Type::kI4, 1, 128, 64, false, true,
       false, true, false},
      {"int4_block32_decode", Type::kI4, 1, 128, 128, true, true, false, false,
       false},
      {"int2_per_channel_decode_wgy", Type::kI2, 1, 128, 64, false, false,
       false, false, false},
      {"int2_lm_head_softcap", Type::kI2, 1, 32768, 64, false, false, false,
       false, true},
      {"int2_block32_decode", Type::kI2, 1, 128, 128, true, false, false, false,
       false},
  };

  for (const Case& tc : kCases) {
    SCOPED_TRACE(tc.name);
    // Subnormal scales must route to the hoisted FP32 kernel by default;
    // normal-scale cases opt in via `LRT_OPENCL_HOISTED_FC=1` to exercise the
    // INT2/INT4/INT8 hoisted kernel variants.
    std::optional<ScopedEnvOverride> enable_hoisted;
    if (!tc.subnormal_scale) {
      enable_hoisted.emplace("LRT_OPENCL_HOISTED_FC", "1");
    }
    ml_drift::cl::Environment probe_env;
    ASSERT_THAT(ml_drift::cl::LoadOpenCL(), IsOk());
    ASSERT_THAT(ml_drift::cl::CreateEnvironment(&probe_env), IsOk());
    const ml_drift::CalculationsPrecision precision =
        probe_env.IsSupported(ml_drift::CalculationsPrecision::kF16)
            ? ml_drift::CalculationsPrecision::kF16
            : ml_drift::CalculationsPrecision::kF32;
    MlDriftTestBackendBridge bridge(precision);
    ASSERT_THAT(bridge.Initialize(), IsOk());

    std::vector<int8_t> q_vals(static_cast<size_t>(tc.out_ch) * tc.in_ch);
    for (size_t idx = 0; idx < q_vals.size(); ++idx) {
      if (tc.weight_type == Type::kI2) {
        q_vals[idx] =
            static_cast<int8_t>((static_cast<int>(idx * 7 + 3) % 4) - 2);
      } else if (tc.weight_type == Type::kI4) {
        q_vals[idx] =
            static_cast<int8_t>((static_cast<int>(idx * 11 + 5) % 15) - 7);
      } else {
        q_vals[idx] =
            static_cast<int8_t>((static_cast<int>(idx * 17 + 9) % 255) - 127);
      }
    }
    std::shared_ptr<OwningCpuBuffer> packed_buf =
        tc.weight_type == Type::kI2
            ? PackSignedInt2(q_vals)
            : (tc.weight_type == Type::kI4 ? PackSignedInt4(q_vals)
                                           : PackSignedInt8(q_vals));

    std::vector<float> x_data(static_cast<size_t>(tc.seq_len) * tc.in_ch);
    for (size_t idx = 0; idx < x_data.size(); ++idx) {
      const float step = tc.subnormal_scale ? 1.5f : 0.08f;
      x_data[idx] =
          step * static_cast<float>(static_cast<int>((idx * 5 + 1) % 13) - 6);
    }

    std::vector<float> bias_data;
    if (tc.with_bias) {
      bias_data.resize(tc.out_ch);
      for (int o = 0; o < tc.out_ch; ++o) {
        bias_data[o] = 0.05f * static_cast<float>((o % 9) - 4);
      }
    }

    constexpr int kBlockSize = 32;
    const int num_blocks = tc.blockwise ? (tc.in_ch / kBlockSize) : 1;
    std::vector<float> scales(static_cast<size_t>(tc.out_ch) * num_blocks);
    std::vector<float> zps(static_cast<size_t>(tc.out_ch) * num_blocks, 0.0f);
    for (int o = 0; o < tc.out_ch; ++o) {
      for (int b = 0; b < num_blocks; ++b) {
        const size_t sb = static_cast<size_t>(o) * num_blocks + b;
        if (tc.subnormal_scale) {
          // Strictly below FP16 smallest normal (2^-14 = 6.1035e-5).
          scales[sb] = 1.2e-5f + 4.0e-6f * static_cast<float>(o % 7);
        } else {
          scales[sb] = 0.02f + 0.005f * static_cast<float>((o + b * 3) % 7);
        }
        if (!tc.symmetric) {
          zps[sb] = tc.blockwise
                        ? (-0.75f + 0.25f * static_cast<float>((o + b) % 5))
                        : -0.5f;
        }
      }
    }

    ZeroPoints zero_points =
        tc.symmetric ? ZeroPoints() : ZeroPoints::Float(zps);
    std::shared_ptr<Quantization> quant =
        tc.blockwise
            ? std::static_pointer_cast<Quantization>(
                  std::make_shared<BlockwiseQuantization>(
                      scales, std::move(zero_points), kBlockSize,
                      /*quantized_dimension=*/0))
            : std::static_pointer_cast<Quantization>(
                  std::make_shared<PerChannelAffineQuantization>(
                      scales, std::move(zero_points),
                      /*quantized_dimension=*/0));

    Tensor<MlDriftMixinTag> x(
        {.name = "x", .type = Type::kFP32, .shape = {1, tc.seq_len, tc.in_ch}});
    Tensor<MlDriftMixinTag> w({.name = "w",
                               .type = tc.weight_type,
                               .shape = {tc.out_ch, tc.in_ch},
                               .buffer = packed_buf,
                               .quantization = std::move(quant)});
    Tensor<MlDriftMixinTag> out =
        tc.with_bias
            ? FullyConnected(x, w,
                             Tensor<MlDriftMixinTag>(
                                 {.name = "b",
                                  .type = Type::kFP32,
                                  .shape = {tc.out_ch},
                                  .buffer = bias_data}))
            : FullyConnected(x, w);
    if (tc.with_softcap) {
      out = Mul(Tanh(Mul(out, 1.0f / 30.0f)), 30.0f);
    }

    ASSERT_THAT(bridge.BuildGraph({TensorHandle(x)}, {TensorHandle(out)}),
                IsOk());
    ASSERT_FALSE(bridge.last_node_names().empty());
    if (precision == ml_drift::CalculationsPrecision::kF16) {
      // Single-token decode routes to the hoisted kernel; multi-token
      // subnormal-scale prefills run ML Drift's FC / convolution kernels on
      // power-of-two rescaled FP16 scales instead.
      const bool any_hoisted = absl::c_any_of(
          bridge.last_node_names(), [](const std::string& name) {
            return absl::StrContains(name, "_hoisted");
          });
      EXPECT_EQ(any_hoisted, tc.seq_len == 1)
          << "Got node: " << bridge.last_node_names().front();
    }

    ASSERT_THAT(
        bridge.SetInput(
            TensorHandle(x),
            absl::MakeConstSpan(reinterpret_cast<const std::byte*>(
                                    x_data.data()),
                                x_data.size() * sizeof(float))),
        IsOk());
    ASSERT_THAT(bridge.Execute(), IsOk());

    std::vector<float> actual(static_cast<size_t>(tc.seq_len) * tc.out_ch,
                              0.0f);
    ASSERT_THAT(bridge.GetOutput(
                    TensorHandle(out),
                    absl::MakeSpan(reinterpret_cast<std::byte*>(actual.data()),
                                   actual.size() * sizeof(float))),
                IsOk());

    std::vector<float> expected(actual.size(), 0.0f);
    for (int t = 0; t < tc.seq_len; ++t) {
      for (int o = 0; o < tc.out_ch; ++o) {
        float acc = tc.with_bias ? bias_data[o] : 0.0f;
        for (int i = 0; i < tc.in_ch; ++i) {
          const int b = tc.blockwise ? (i / kBlockSize) : 0;
          const size_t sb = static_cast<size_t>(o) * num_blocks + b;
          const float w_f =
              scales[sb] *
              (static_cast<float>(
                   q_vals[static_cast<size_t>(o) * tc.in_ch + i]) -
               zps[sb]);
          acc += x_data[static_cast<size_t>(t) * tc.in_ch + i] * w_f;
        }
        if (tc.with_softcap) {
          acc = std::tanh(acc / 30.0f) * 30.0f;
        }
        expected[static_cast<size_t>(t) * tc.out_ch + o] = acc;
      }
    }
    const float tol = tc.subnormal_scale ? 2e-2f : 1e-2f;
    EXPECT_THAT(actual, Pointwise(FloatNear(tol), expected));
  }
}

TEST(MlDriftHoistedQuantizedFcTest, SubnormalScalePromotesToFp32AndRoutes) {
  auto build_model =
      [](const std::string& desc, ml_drift::GpuApi api, Type weight_type,
         int seq_len, float scale_val, bool bufferless_stub,
         absl::flat_hash_map<std::string, ml_drift::TensorDescriptor>*
             shared_weights,
         ml_drift::GpuModel* out_model) -> absl::Status {
    ml_drift::GpuInfo gpu_info;
    ml_drift::GetGpuInfoFromDeviceDescription(desc, api, &gpu_info);
    if (api == ml_drift::GpuApi::kOpenCl) {
      gpu_info.opencl_info.extensions.push_back("cl_khr_fp16");
      gpu_info.opencl_info.supports_fp16 = true;
      gpu_info.opencl_info.compute_units_count = 6;
    }
    ml_drift::GpuModelBuilder builder(
        gpu_info, ml_drift::ModelHints(), ml_drift::CalculationsPrecision::kF16,
        ml_drift::TensorStorageType::kBuffer);
    MlDriftBuildContext ctx;
    ctx.gpu_model_builder = &builder;
    ctx.default_float_type = ml_drift::DataType::kFloat16;
    ctx.shared_weight_descriptors = shared_weights;
    MlDriftDummyWeightCreator weight_creator(&builder);

    constexpr int kOut = 128;
    constexpr int kIn = 64;
    std::vector<int8_t> q_vals(kOut * kIn, 1);
    std::shared_ptr<OwningCpuBuffer> packed_buf =
        weight_type == Type::kI2 ? PackSignedInt2(q_vals)
                                 : PackSignedInt8(q_vals);
    std::vector<float> scales(kOut, scale_val);
    Tensor<MlDriftMixinTag> x(
        {.name = "x", .type = Type::kFP32, .shape = {1, seq_len, kIn}});
    Tensor<MlDriftMixinTag> w(
        {.name = "w",
         .type = weight_type,
         .shape = {kOut, kIn},
         .buffer = bufferless_stub ? nullptr : packed_buf,
         .quantization = std::make_shared<PerChannelAffineQuantization>(
             scales, ZeroPoints(), /*quantized_dimension=*/0)});
    Tensor<MlDriftMixinTag> out = Tanh(FullyConnected(x, w));

    auto locked = packed_buf->Lock();
    absl::Span<const int8_t> w_span(
        reinterpret_cast<const int8_t*>(locked.data()), locked.size());
    LRT_TENSOR_ASSIGN_OR_RETURN(
        ctx.graph_tensor_to_handle[TensorHandle(w).GetRaw()],
        weight_creator.CreateFcWeights<int8_t>(w, w_span,
                                               ctx.default_float_type));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto handles, AppendGpuModelBuilder(ctx, {TensorHandle(out)}));
    return builder.GetGpuModel({handles.at(TensorHandle(x).GetRaw()).id},
                               {handles.at(TensorHandle(out).GetRaw()).id},
                               out_model);
  };

  // Subnormal INT8 scale (`1.5e-5 < 2^-13`) promotes the scale tensor to FP32.
  // Multi-token prefills (seq_len=10) run ML Drift's FC kernels on an FP16
  // scale rescaled by 2^4 (`w:scale_p2_4:<kFloat16>`, then a `2^-4` multiply),
  // and single-token decode routes to the hoisted FP32-math kernel. Both keep
  // working across warm-cache bufferless per-channel stubs carrying dummy
  // `1.0f` scales.
  absl::flat_hash_map<std::string, ml_drift::TensorDescriptor> shared_weights;
  const std::string pow2_key = absl::StrCat(
      "w:scale_p2_4:", static_cast<int>(ml_drift::DataType::kFloat16));
  {
    ml_drift::GpuModel prefill_model;
    ASSERT_THAT(build_model("QUALCOMM Adreno(TM) 750",
                            ml_drift::GpuApi::kOpenCl, Type::kI8,
                            /*seq_len=*/10, /*scale_val=*/1.5e-5f,
                            /*bufferless_stub=*/false, &shared_weights,
                            &prefill_model),
                IsOk());
    for (const auto& node : prefill_model.nodes) {
      EXPECT_FALSE(absl::StrContains(node.name, "_hoisted")) << node.name;
    }
    ASSERT_TRUE(shared_weights.contains(pow2_key));
    EXPECT_EQ(shared_weights.at(pow2_key).GetDataType(),
              ml_drift::DataType::kFloat16);
  }
  {
    ml_drift::GpuModel decode_model;
    ASSERT_THAT(build_model("QUALCOMM Adreno(TM) 750",
                            ml_drift::GpuApi::kOpenCl, Type::kI8,
                            /*seq_len=*/1, /*scale_val=*/1.5e-5f,
                            /*bufferless_stub=*/false, &shared_weights,
                            &decode_model),
                IsOk());
    ASSERT_EQ(decode_model.nodes.size(), 1u);
    const auto& node = decode_model.nodes.front();
    EXPECT_TRUE(absl::StrContains(node.name, "fc1x1_int8_weights_hoisted"))
        << node.name;
    EXPECT_FALSE(absl::c_linear_search(
        node.gpu_operation->compiler_options_,
        ml_drift::CompilerOptions::kClFastRelaxedMath));
    ASSERT_GE(node.inputs.size(), 3u);
    const ml_drift::ValueId scale_id = node.inputs[2];
    ASSERT_TRUE(decode_model.const_tensors.contains(scale_id));
    EXPECT_EQ(decode_model.const_tensors.at(scale_id).GetDataType(),
              ml_drift::DataType::kFloat32);
  }
  {
    ml_drift::GpuModel warm_stub_prefill;
    ASSERT_THAT(build_model("QUALCOMM Adreno(TM) 750",
                            ml_drift::GpuApi::kOpenCl, Type::kI8,
                            /*seq_len=*/10, /*scale_val=*/1.0f,
                            /*bufferless_stub=*/true, &shared_weights,
                            &warm_stub_prefill),
                IsOk());
    for (const auto& node : warm_stub_prefill.nodes) {
      EXPECT_FALSE(absl::StrContains(node.name, "_hoisted")) << node.name;
    }
  }
  // The runners lower the decode signature first and may release the weights'
  // host data before lowering prefill: promoting the scale on decode publishes
  // the rescaled FP16 scale, so a later dataless prefill still avoids the
  // hoisted kernel.
  {
    absl::flat_hash_map<std::string, ml_drift::TensorDescriptor>
        decode_first_weights;
    ml_drift::GpuModel decode_model;
    ASSERT_THAT(build_model("QUALCOMM Adreno(TM) 750",
                            ml_drift::GpuApi::kOpenCl, Type::kI8,
                            /*seq_len=*/1, /*scale_val=*/1.5e-5f,
                            /*bufferless_stub=*/false, &decode_first_weights,
                            &decode_model),
                IsOk());
    EXPECT_TRUE(decode_first_weights.contains(pow2_key));
    ml_drift::GpuModel stub_prefill;
    ASSERT_THAT(build_model("QUALCOMM Adreno(TM) 750",
                            ml_drift::GpuApi::kOpenCl, Type::kI8,
                            /*seq_len=*/10, /*scale_val=*/1.0f,
                            /*bufferless_stub=*/true, &decode_first_weights,
                            &stub_prefill),
                IsOk());
    for (const auto& node : stub_prefill.nodes) {
      EXPECT_FALSE(absl::StrContains(node.name, "_hoisted")) << node.name;
    }
  }
  {
    ml_drift::GpuModel warm_stub_decode;
    ASSERT_THAT(build_model("QUALCOMM Adreno(TM) 750",
                            ml_drift::GpuApi::kOpenCl, Type::kI8,
                            /*seq_len=*/1, /*scale_val=*/1.0f,
                            /*bufferless_stub=*/true, &shared_weights,
                            &warm_stub_decode),
                IsOk());
    ASSERT_EQ(warm_stub_decode.nodes.size(), 1u);
    const auto& node = warm_stub_decode.nodes.front();
    EXPECT_TRUE(absl::StrContains(node.name, "fc1x1_int8_weights_hoisted"))
        << node.name;
    ASSERT_GE(node.inputs.size(), 3u);
    const ml_drift::ValueId scale_id = node.inputs[2];
    ASSERT_TRUE(warm_stub_decode.const_tensors.contains(scale_id));
    EXPECT_EQ(warm_stub_decode.const_tensors.at(scale_id).GetDataType(),
              ml_drift::DataType::kFloat32);
  }

  // Normal INT2 scale (`0.05f`) stays FP16 and keeps ML Drift's default
  // `fc1x1_int2_weights` kernel (no regression on normal scales), and opts
  // into `fc1x1_int2_weights_hoisted` when `LRT_OPENCL_HOISTED_FC=1`.
  {
    ml_drift::GpuModel adreno840_default;
    ASSERT_THAT(build_model("QUALCOMM Adreno(TM) 840",
                            ml_drift::GpuApi::kOpenCl, Type::kI2,
                            /*seq_len=*/1, /*scale_val=*/0.05f,
                            /*bufferless_stub=*/false, nullptr,
                            &adreno840_default),
                IsOk());
    ASSERT_EQ(adreno840_default.nodes.size(), 1u);
    const auto& node = adreno840_default.nodes.front();
    EXPECT_FALSE(absl::StrContains(node.name, "_hoisted")) << node.name;
    ASSERT_GE(node.inputs.size(), 3u);
    const ml_drift::ValueId scale_id = node.inputs[2];
    ASSERT_TRUE(adreno840_default.const_tensors.contains(scale_id));
    EXPECT_EQ(adreno840_default.const_tensors.at(scale_id).GetDataType(),
              ml_drift::DataType::kFloat16);
  }
  {
    ScopedEnvOverride enable_hoisted("LRT_OPENCL_HOISTED_FC", "1");
    ml_drift::GpuModel adreno840_optin;
    ASSERT_THAT(build_model("QUALCOMM Adreno(TM) 840",
                            ml_drift::GpuApi::kOpenCl, Type::kI2,
                            /*seq_len=*/1, /*scale_val=*/0.05f,
                            /*bufferless_stub=*/false, nullptr,
                            &adreno840_optin),
                IsOk());
    ASSERT_EQ(adreno840_optin.nodes.size(), 1u);
    const auto& node840 = adreno840_optin.nodes.front();
    EXPECT_TRUE(absl::StrContains(node840.name, "fc1x1_int2_weights_hoisted"))
        << node840.name;
    EXPECT_FALSE(absl::c_linear_search(
        node840.gpu_operation->compiler_options_,
        ml_drift::CompilerOptions::kClFastRelaxedMath));
    ASSERT_GE(node840.inputs.size(), 3u);
    const ml_drift::ValueId scale840_id = node840.inputs[2];
    ASSERT_TRUE(adreno840_optin.const_tensors.contains(scale840_id));
    EXPECT_EQ(adreno840_optin.const_tensors.at(scale840_id).GetDataType(),
              ml_drift::DataType::kFloat16);
  }
}

}  // namespace
}  // namespace litert::tensor
