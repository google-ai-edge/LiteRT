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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/runtime/compiled_model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "tflite/interpreter.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/version.h"

namespace litert {
namespace {

using ::testing::ElementsAre;

std::vector<uint8_t> CreateAddThenResizeBilinearModel() {
  flatbuffers::FlatBufferBuilder builder;

  const std::array<flatbuffers::Offset<tflite::OperatorCode>, 2> operator_codes{
      {
          tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD),
          tflite::CreateOperatorCode(builder,
                                     tflite::BuiltinOperator_RESIZE_BILINEAR),
      }};
  const auto add_options =
      tflite::CreateAddOptions(builder, tflite::ActivationFunctionType_NONE);
  const auto resize_options =
      tflite::CreateResizeBilinearOptions(builder, false, false);

  const std::array<int32_t, 2> resize_size{{4, 4}};
  const std::array<flatbuffers::Offset<tflite::Buffer>, 2> buffers{{
      tflite::CreateBuffer(builder, builder.CreateVector({})),
      tflite::CreateBuffer(
          builder, builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(resize_size.data()),
                       resize_size.size() * sizeof(int32_t))),
  }};

  const std::array<int32_t, 4> input_shape{{1, 2, 2, 1}};
  const std::array<int32_t, 4> output_shape{{1, 4, 4, 1}};
  const std::array<int32_t, 1> size_shape{{2}};
  const std::array<flatbuffers::Offset<tflite::Tensor>, 5> tensors{{
      tflite::CreateTensor(
          builder, builder.CreateVector(input_shape.data(), input_shape.size()),
          tflite::TensorType_FLOAT32),
      tflite::CreateTensor(
          builder, builder.CreateVector(input_shape.data(), input_shape.size()),
          tflite::TensorType_FLOAT32),
      tflite::CreateTensor(
          builder, builder.CreateVector(input_shape.data(), input_shape.size()),
          tflite::TensorType_FLOAT32),
      tflite::CreateTensor(
          builder, builder.CreateVector(size_shape.data(), size_shape.size()),
          tflite::TensorType_INT32, /*buffer=*/1),
      tflite::CreateTensor(
          builder,
          builder.CreateVector(output_shape.data(), output_shape.size()),
          tflite::TensorType_FLOAT32),
  }};

  const std::array<int32_t, 2> add_inputs{{0, 1}};
  const std::array<int32_t, 1> add_outputs{{2}};
  const std::array<int32_t, 2> resize_inputs{{2, 3}};
  const std::array<int32_t, 1> resize_outputs{{4}};
  const std::array<flatbuffers::Offset<tflite::Operator>, 2> operators{{
      tflite::CreateOperator(
          builder, /*opcode_index=*/0,
          builder.CreateVector(add_inputs.data(), add_inputs.size()),
          builder.CreateVector(add_outputs.data(), add_outputs.size()),
          tflite::BuiltinOptions_AddOptions, add_options.Union()),
      tflite::CreateOperator(
          builder, /*opcode_index=*/1,
          builder.CreateVector(resize_inputs.data(), resize_inputs.size()),
          builder.CreateVector(resize_outputs.data(), resize_outputs.size()),
          tflite::BuiltinOptions_ResizeBilinearOptions, resize_options.Union()),
  }};

  const std::array<int32_t, 2> subgraph_inputs{{0, 1}};
  const std::array<int32_t, 1> subgraph_outputs{{4}};
  const auto subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector(subgraph_inputs.data(), subgraph_inputs.size()),
      builder.CreateVector(subgraph_outputs.data(), subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));
  const auto description = builder.CreateString("ADD then RESIZE_BILINEAR");
  const auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));
  tflite::FinishModelBuffer(builder, model);

  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

void CompileWithYnnpackEnabled(LiteRtEnvironment environment, LiteRtModel model,
                               std::vector<std::string>* delegate_names) {
  ASSERT_NE(environment, nullptr);
  ASSERT_NE(model, nullptr);
  ASSERT_NE(delegate_names, nullptr);

  LiteRtOptions options = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&options));
  LITERT_ASSERT_OK(
      LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu));

  LITERT_ASSERT_OK_AND_ASSIGN(auto cpu_options, CpuOptions::Create());
  LITERT_ASSERT_OK(cpu_options.SetEnableYNNPack(true));
  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueCpuOptionsData(cpu_options.Get(), &identifier,
                                              &payload, &payload_deleter));

  LiteRtOpaqueOptions opaque_cpu_options = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(
      identifier, payload, payload_deleter, &opaque_cpu_options));
  LITERT_ASSERT_OK(LiteRtAddOpaqueOptions(options, opaque_cpu_options));

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        LiteRtCompiledModelT::Ptr compiled_model,
        LiteRtCompiledModelT::Create(environment, model, options));
    LITERT_ASSERT_OK_AND_ASSIGN(tflite::Interpreter * interpreter,
                                GetInterpreter(compiled_model.get()));

    const auto* subgraph = interpreter->subgraph(0);
    ASSERT_NE(subgraph, nullptr);
    const auto& execution_plan = subgraph->execution_plan();
    const auto& nodes_and_registration = subgraph->nodes_and_registration();
    for (int node_index : execution_plan) {
      const TfLiteRegistration& registration =
          nodes_and_registration[node_index].second;
      if (registration.custom_name != nullptr) {
        delegate_names->emplace_back(registration.custom_name);
      }
    }
  }

  LiteRtDestroyOptions(options);
}

TEST(YnnpackAcceleratorTest, EnableOptionUsesAvailableCpuDelegates) {
  LiteRtEnvironment environment = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateEnvironment(0, nullptr, &environment));

  const std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model = nullptr;
  LITERT_ASSERT_OK(
      LiteRtCreateModelFromFile(environment, path.c_str(), &model));

  std::vector<std::string> delegate_names;
  CompileWithYnnpackEnabled(environment, model, &delegate_names);

#if defined(LITERT_TEST_EXPECT_YNNPACK)
  EXPECT_THAT(delegate_names, ElementsAre("YNNPackDelegate"));
#else
  EXPECT_THAT(delegate_names, ElementsAre("TfLiteXNNPackDelegate"));
#endif

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

TEST(YnnpackAcceleratorTest, DelegatesSupportedNodesBeforeXnnpack) {
  LiteRtEnvironment environment = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateEnvironment(0, nullptr, &environment));

  const std::vector<uint8_t> model_buffer = CreateAddThenResizeBilinearModel();
  LiteRtModel model = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateModelFromBuffer(environment, model_buffer.data(),
                                               model_buffer.size(), &model));

  std::vector<std::string> delegate_names;
  CompileWithYnnpackEnabled(environment, model, &delegate_names);

#if defined(LITERT_TEST_EXPECT_YNNPACK)
  EXPECT_THAT(delegate_names,
              ElementsAre("YNNPackDelegate", "TfLiteXNNPackDelegate"));
#else
  EXPECT_THAT(delegate_names, ElementsAre("TfLiteXNNPackDelegate"));
#endif

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

}  // namespace
}  // namespace litert
