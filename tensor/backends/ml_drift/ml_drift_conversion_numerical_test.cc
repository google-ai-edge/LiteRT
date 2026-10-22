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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
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
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
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
  ~MlDriftTestBackendBridge() override {
    if (env_.queue()) {
      (void)env_.queue()->WaitForCompletion();
    }
  }

  absl::Status Initialize() override {
    LRT_TENSOR_RETURN_IF_ERROR(ml_drift::cl::LoadOpenCL());
    LRT_TENSOR_RETURN_IF_ERROR(ml_drift::cl::CreateEnvironment(&env_));
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

    ml_drift::CreateGpuModelInfo ci = {
        .precision = ml_drift::CalculationsPrecision::kF32,
        .storage_type =
            ml_drift::cl::GetFastestStorageType(env_.device().GetInfo()),
    };
    LRT_TENSOR_RETURN_IF_ERROR(ic_.InitFromGpuModel(ci, &gpu_model, &env_));
    return absl::OkStatus();
  }

  absl::Status SetInput(const TensorHandle& tensor,
                        absl::Span<const std::byte> data) override {
    const auto& th = tensor_to_handle_.at(tensor.GetRaw());
    if (th.tensor_desc.GetDataType() == ml_drift::DataType::kFloat32) {
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
  quant_params->zero_points = {0};
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

}  // namespace
}  // namespace litert::tensor
