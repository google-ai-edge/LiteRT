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
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/backends/ml_drift/metal_decode_quantized_fc.h"
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
  explicit MlDriftTestBackendBridge(ml_drift::CalculationsPrecision precision =
                                        ml_drift::CalculationsPrecision::kF32)
      : precision_(precision) {}

  ~MlDriftTestBackendBridge() override {
    if (env_.queue()) {
      (void)env_.queue()->WaitForCompletion();
    }
  }

  const std::vector<std::string>& node_names() const { return node_names_; }

  absl::Status Initialize() override {
    LRT_TENSOR_RETURN_IF_ERROR(ml_drift::cl::LoadOpenCL());
    LRT_TENSOR_RETURN_IF_ERROR(ml_drift::cl::CreateEnvironment(&env_));
    if (precision_ == ml_drift::CalculationsPrecision::kF16 &&
        !env_.device().GetInfo().SupportsFP16()) {
      precision_ = ml_drift::CalculationsPrecision::kF32;
    }
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
    node_names_.clear();
    for (const auto& node : gpu_model.nodes) {
      node_names_.push_back(node.name);
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
  std::vector<std::string> node_names_;
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

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name); old != nullptr) {
      old_value_ = std::string(old);
    }
    if (value != nullptr) {
      setenv(name_.c_str(), value, 1);
    } else {
      unsetenv(name_.c_str());
    }
  }
  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;
  ~ScopedEnvVar() {
    if (old_value_.has_value()) {
      setenv(name_.c_str(), old_value_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> old_value_;
};

TEST(MetalDecodeQuantizedFcNumericalTest,
     MatchesReferenceAcrossInt2Int4Int8AndReductionModes) {
  ScopedEnvVar force_on("LRT_METAL_DECODE_FC", "1");

  struct Case {
    Type wtype;
    int o;
    int i;
    int block_size;  // 0 = per-channel
    bool affine_zp;
    bool has_bias;
    int outs;
    int wg_x;
    int wg_y;
    ml_drift::CalculationsPrecision precision;
    int int2_mode = 0;
    bool fuse_residual_add = false;
    float x_scale = 0.0625f;
    float w_scale_base = 0.015625f;
  };
  const std::vector<Case> cases = {
      // INT2 per-channel: floor (mode=0) & bitmask (mode=1), outs=1/2/4,
      // wg_y=1/2/4, FP32 & FP16, sym & affine, plus residual epilogue fusion.
      {Type::kI2, 64, 128, 0, false, false, 4, 4, 1,
       ml_drift::CalculationsPrecision::kF16, 0, false},
      {Type::kI2, 64, 128, 0, true, true, 4, 4, 2,
       ml_drift::CalculationsPrecision::kF16, 0, true},
      {Type::kI2, 64, 128, 0, false, false, 4, 4, 1,
       ml_drift::CalculationsPrecision::kF16, 1, false},
      {Type::kI2, 48, 96, 0, false, true, 2, 4, 4,
       ml_drift::CalculationsPrecision::kF32, 0, false},
      {Type::kI2, 20, 64, 0, true, false, 1, 4, 1,
       ml_drift::CalculationsPrecision::kF16, 1, false},
      // Large-K / large-activation per-channel INT2/INT4/INT8 + large-block
      // INT4 (`block_size = 128 > 32`) exercising chunked scale flushing.
      {Type::kI2, 32, 4096, 0, false, false, 2, 4, 4,
       ml_drift::CalculationsPrecision::kF16, 0, false, 0.5f},
      {Type::kI4, 32, 4096, 0, false, false, 1, 4, 4,
       ml_drift::CalculationsPrecision::kF16, 0, false, 0.5f},
      {Type::kI4, 32, 512, 128, true, false, 1, 4, 4,
       ml_drift::CalculationsPrecision::kF16, 0, false, 0.5f},
      {Type::kI8, 32, 4096, 0, false, false, 1, 4, 4,
       ml_drift::CalculationsPrecision::kF16, 0, false, 0.5f},
      // Subnormal-scale INT8 PLE (`scale in [1.2e-5, 4.8e-5] < 2^-14`) check.
      {Type::kI8, 64, 256, 0, false, false, 1, 4, 2,
       ml_drift::CalculationsPrecision::kF16, 0, false, 0.25f, 1.2e-5f},
      // INT4 blockwise (Q4_0 / Q4_1) and per-channel: wg_y=1/4/8/16,
      // plus residual epilogue fusion (`dst_tensor.Write(res_value, ...)`).
      {Type::kI4, 64, 512, 32, false, false, 1, 4, 8,
       ml_drift::CalculationsPrecision::kF16, 0, true},
      {Type::kI4, 64, 512, 32, true, true, 1, 4, 16,
       ml_drift::CalculationsPrecision::kF16, 0, false},
      {Type::kI4, 32, 128, 32, false, false, 1, 8, 1,
       ml_drift::CalculationsPrecision::kF32, 0, false},
      {Type::kI4, 48, 128, 0, true, true, 1, 4, 4,
       ml_drift::CalculationsPrecision::kF16, 0, false},
      // INT8 blockwise Q8_0 & per-channel, including non-multiple-of-4 shapes.
      {Type::kI8, 32, 128, 32, false, true, 1, 4, 4,
       ml_drift::CalculationsPrecision::kF16, 0, true},
      {Type::kI8, 37, 69, 0, true, true, 1, 4, 2,
       ml_drift::CalculationsPrecision::kF32, 0, false},
      {Type::kI8, 37, 69, 0, false, false, 1, 8, 1,
       ml_drift::CalculationsPrecision::kF16, 0, false},
  };

  for (const Case& c : cases) {
    const std::string outs_s = absl::StrCat(c.outs);
    const std::string wgx_s = absl::StrCat(c.wg_x);
    const std::string wgy_s = absl::StrCat(c.wg_y);
    const std::string int2_s = absl::StrCat(c.int2_mode);
    ScopedEnvVar env_outs("LRT_METAL_DECODE_FC_OUTS", outs_s.c_str());
    ScopedEnvVar env_wgx("LRT_METAL_DECODE_FC_WG_X", wgx_s.c_str());
    ScopedEnvVar env_wgy("LRT_METAL_DECODE_FC_WG_Y", wgy_s.c_str());
    ScopedEnvVar env_int2("LRT_METAL_DECODE_FC_INT2", int2_s.c_str());

    MlDriftTestBackendBridge bridge(c.precision);
    ASSERT_THAT(bridge.Initialize(), IsOk());

    std::vector<float> x_vals(c.i);
    for (int k = 0; k < c.i; ++k) {
      x_vals[k] = c.x_scale * static_cast<float>((k * 13 + 5) % 19 - 9);
    }
    const int num_blocks = c.block_size > 0 ? c.i / c.block_size : 1;
    std::vector<float> scales(static_cast<size_t>(c.o) * num_blocks);
    std::vector<float> zps(static_cast<size_t>(c.o) * num_blocks, 0.0f);
    for (size_t k = 0; k < scales.size(); ++k) {
      scales[k] = c.w_scale_base * static_cast<float>((k % 7) + 1);
      if (c.affine_zp) {
        zps[k] = (k % 3 == 0) ? -1.25f : ((k % 3 == 1) ? 1.5f : 0.0f);
      }
    }

    std::vector<float> w_ref(static_cast<size_t>(c.o) * c.i);
    std::shared_ptr<Buffer> w_buf;
    if (c.wtype == Type::kI2) {
      auto buf = OwningCpuBuffer::Allocate<Type::kI2>(c.o * c.i);
      uint8_t* bytes = reinterpret_cast<uint8_t*>(buf->Span<int2_t>().data());
      std::memset(bytes, 0, (static_cast<size_t>(c.o) * c.i + 3) / 4);
      for (int r = 0; r < c.o; ++r) {
        for (int col = 0; col < c.i; ++col) {
          const int idx = r * c.i + col;
          const int q = ((idx * 5 + 1) % 4) - 2;  // [-2, 1]
          const uint8_t bits = static_cast<uint8_t>(q & 0x3);
          bytes[idx / 4] |= static_cast<uint8_t>(bits << ((idx % 4) * 2));
          const int blk =
              c.block_size > 0 ? r * num_blocks + (col / c.block_size) : r;
          w_ref[idx] = scales[blk] * (static_cast<float>(q) - zps[blk]);
        }
      }
      w_buf = std::move(buf);
    } else if (c.wtype == Type::kI4) {
      auto buf = OwningCpuBuffer::Allocate<Type::kI4>(c.o * c.i);
      int4_t* packed = buf->Span<int4_t>().data();
      for (int r = 0; r < c.o; ++r) {
        for (int col = 0; col < c.i; col += 2) {
          const int idx = r * c.i + col;
          const int qa = ((idx * 3 + 2) % 16) - 8;
          const int qb = (((idx + 1) * 7 + 5) % 16) - 8;
          packed[idx / 2].a = static_cast<int8_t>(qa);
          packed[idx / 2].b = static_cast<int8_t>(qb);
          const int blk0 =
              c.block_size > 0 ? r * num_blocks + (col / c.block_size) : r;
          const int blk1 = c.block_size > 0
                               ? r * num_blocks + ((col + 1) / c.block_size)
                               : r;
          w_ref[idx] = scales[blk0] * (static_cast<float>(qa) - zps[blk0]);
          w_ref[idx + 1] = scales[blk1] * (static_cast<float>(qb) - zps[blk1]);
        }
      }
      w_buf = std::move(buf);
    } else {
      auto buf = OwningCpuBuffer::Allocate<Type::kI8>(c.o * c.i);
      int8_t* raw = buf->Span<int8_t>().data();
      for (int r = 0; r < c.o; ++r) {
        for (int col = 0; col < c.i; ++col) {
          const int idx = r * c.i + col;
          const int q = ((idx * 11 + 3) % 81) - 40;
          raw[idx] = static_cast<int8_t>(q);
          const int blk =
              c.block_size > 0 ? r * num_blocks + (col / c.block_size) : r;
          w_ref[idx] = scales[blk] * (static_cast<float>(q) - zps[blk]);
        }
      }
      w_buf = std::move(buf);
    }

    std::shared_ptr<Quantization> quant;
    if (c.block_size > 0) {
      quant = std::make_shared<BlockwiseQuantization>(
          scales, c.affine_zp ? ZeroPoints::Float(zps) : ZeroPoints(),
          c.block_size, /*quantized_dimension=*/0);
    } else {
      quant = std::make_shared<PerChannelAffineQuantization>(
          scales, c.affine_zp ? ZeroPoints::Float(zps) : ZeroPoints(),
          /*quantized_dimension=*/0);
    }

    Tensor<MlDriftMixinTag> x(
        {.name = "x", .type = Type::kFP32, .shape = {1, 1, c.i}});
    Tensor<MlDriftMixinTag> w({.name = "w",
                               .type = c.wtype,
                               .shape = {c.o, c.i},
                               .buffer = w_buf,
                               .quantization = quant});
    std::vector<float> bias_vals(c.o, 0.0f);
    std::vector<float> residual_vals(c.o, 0.0f);
    Tensor<MlDriftMixinTag> res(
        {.name = "res", .type = Type::kFP32, .shape = {1, 1, c.o}});
    Tensor<MlDriftMixinTag> y = [&]() {
      Tensor<MlDriftMixinTag> fc = [&]() {
        if (!c.has_bias) return FullyConnected(x, w);
        for (int r = 0; r < c.o; ++r) {
          bias_vals[r] = 0.125f * static_cast<float>((r % 9) - 4);
        }
        Tensor<MlDriftMixinTag> b({.name = "b",
                                   .type = Type::kFP32,
                                   .shape = {c.o},
                                   .buffer = bias_vals});
        return FullyConnected(x, w, b);
      }();
      if (!c.fuse_residual_add) return fc;
      for (int r = 0; r < c.o; ++r) {
        residual_vals[r] = 0.25f * static_cast<float>((r % 7) - 3);
      }
      return Add(fc, res);
    }();

    std::vector<TensorHandle> inputs = {TensorHandle(x)};
    if (c.fuse_residual_add) {
      inputs.push_back(TensorHandle(res));
    }
    ASSERT_THAT(bridge.BuildGraph(inputs, {TensorHandle(y)}), IsOk());
    EXPECT_THAT(bridge.node_names(),
                testing::Contains(testing::HasSubstr("_metal_decode")));
    ASSERT_THAT(
        bridge.SetInput(TensorHandle(x),
                        absl::MakeConstSpan(
                            reinterpret_cast<const std::byte*>(x_vals.data()),
                            x_vals.size() * sizeof(float))),
        IsOk());
    if (c.fuse_residual_add) {
      ASSERT_THAT(
          bridge.SetInput(
              TensorHandle(res),
              absl::MakeConstSpan(
                  reinterpret_cast<const std::byte*>(residual_vals.data()),
                  residual_vals.size() * sizeof(float))),
          IsOk());
    }
    ASSERT_THAT(bridge.Execute(), IsOk());

    std::vector<float> actual(c.o, 0.0f);
    ASSERT_THAT(bridge.GetOutput(
                    TensorHandle(y),
                    absl::MakeSpan(reinterpret_cast<std::byte*>(actual.data()),
                                   actual.size() * sizeof(float))),
                IsOk());

    std::vector<float> expected(c.o, 0.0f);
    for (int r = 0; r < c.o; ++r) {
      float sum = bias_vals[r] + residual_vals[r];
      for (int col = 0; col < c.i; ++col) {
        sum += x_vals[col] * w_ref[static_cast<size_t>(r) * c.i + col];
      }
      expected[r] = sum;
    }
    const float tol =
        c.i >= 1024
            ? 0.25f
            : (c.precision == ml_drift::CalculationsPrecision::kF16 ? 2e-2f
                                                                    : 1e-3f);
    EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(tol), expected))
        << "o=" << c.o << " i=" << c.i << " blk=" << c.block_size
        << " outs=" << c.outs << " wg=" << c.wg_x << "x" << c.wg_y;
  }
}

TEST(MetalDecodeQuantizedFcNumericalTest,
     RejectsInvalidPreconditionsAndF32F16) {
  ml_drift::GpuInfo gpu_info;
  gpu_info.gpu_api = ml_drift::GpuApi::kMetal;
  gpu_info.vendor = ml_drift::GpuVendor::kApple;
  gpu_info.apple_info = ml_drift::AppleInfo("Apple A17 Pro");

  auto make_valid = [](ml_drift::GpuModelBuilder* builder,
                       ml_drift::TensorStorageType w_storage,
                       ml_drift::WeightsLayout w_layout, int src_c, int w_i) {
    auto src = builder->AddTensor(ml_drift::BHWC(1, 1, 1, src_c),
                                  ml_drift::DataType::kFloat16);
    ml_drift::TensorDescriptor w_td(ml_drift::DataType::kUint4, w_storage,
                                    ml_drift::Layout::kLinear);
    w_td.SetBHWDCShape(ml_drift::BHWDC(1, 1, 1, 1, 32 * w_i / 2));
    auto w_handle = builder->AddConstantTensor(std::move(w_td));
    ml_drift::TensorDescriptor s_td(ml_drift::DataType::kFloat16,
                                    ml_drift::TensorStorageType::kBuffer,
                                    ml_drift::Layout::kLinear);
    s_td.SetBHWDCShape(ml_drift::BHWDC(1, 1, 1, 1, 32));
    auto s_handle = builder->AddConstantTensor(std::move(s_td));
    ml_drift::WeightsDescription desc;
    desc.type = ml_drift::DataType::kUint4;
    desc.layout = w_layout;
    desc.output_group_size = 8;  // 32 / 4
    ml_drift::GpuModelBuilder::Weights weights(
        w_handle, desc, ml_drift::OHWI(32, 1, 1, w_i),
        ml_drift::OHWI(32, 1, 1, 1), s_handle, std::nullopt);
    return TryAddMetalDecodeQuantizedFullyConnected(builder, src, weights,
                                                    nullptr, "test_fc");
  };

  // Valid Metal FP16 config lowers cleanly.
  {
    ml_drift::GpuModelBuilder builder(gpu_info, ml_drift::ModelHints(),
                                      ml_drift::CalculationsPrecision::kF16,
                                      ml_drift::TensorStorageType::kBuffer);
    EXPECT_TRUE(make_valid(&builder, ml_drift::TensorStorageType::kBuffer,
                           ml_drift::WeightsLayout::kOSpatialIOGroupI4O4, 64,
                           64)
                    .has_value());
  }
  // Rejects kF32F16 so FP32 accumulation is honored.
  {
    ml_drift::GpuModelBuilder builder(gpu_info, ml_drift::ModelHints(),
                                      ml_drift::CalculationsPrecision::kF32F16,
                                      ml_drift::TensorStorageType::kBuffer);
    EXPECT_FALSE(make_valid(&builder, ml_drift::TensorStorageType::kBuffer,
                            ml_drift::WeightsLayout::kOSpatialIOGroupI4O4, 64,
                            64)
                     .has_value());
  }
  // Rejects non-I4O4 linear layout (kOSpatialIOGroupO4I4).
  {
    ml_drift::GpuModelBuilder builder(gpu_info, ml_drift::ModelHints(),
                                      ml_drift::CalculationsPrecision::kF16,
                                      ml_drift::TensorStorageType::kBuffer);
    EXPECT_FALSE(make_valid(&builder, ml_drift::TensorStorageType::kBuffer,
                            ml_drift::WeightsLayout::kOSpatialIOGroupO4I4, 64,
                            64)
                     .has_value());
  }
  // Rejects non-buffer storage.
  {
    ml_drift::GpuModelBuilder builder(gpu_info, ml_drift::ModelHints(),
                                      ml_drift::CalculationsPrecision::kF16,
                                      ml_drift::TensorStorageType::kBuffer);
    EXPECT_FALSE(make_valid(&builder, ml_drift::TensorStorageType::kTexture2D,
                            ml_drift::WeightsLayout::kOSpatialIOGroupI4O4, 64,
                            64)
                     .has_value());
  }
  // Rejects mismatched input channel count.
  {
    ml_drift::GpuModelBuilder builder(gpu_info, ml_drift::ModelHints(),
                                      ml_drift::CalculationsPrecision::kF16,
                                      ml_drift::TensorStorageType::kBuffer);
    EXPECT_FALSE(make_valid(&builder, ml_drift::TensorStorageType::kBuffer,
                            ml_drift::WeightsLayout::kOSpatialIOGroupI4O4, 128,
                            64)
                     .has_value());
  }
}

}  // namespace
}  // namespace litert::tensor
