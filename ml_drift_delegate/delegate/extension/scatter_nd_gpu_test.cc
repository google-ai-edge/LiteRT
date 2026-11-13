// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/version.h"

namespace litert::test {

// Selects the graph pipeline. The setter is not public.
class GpuOptionsTestPeer {
 public:
  static LiteRtStatus SetUseIrModel(GpuOptions& options, bool use_ir_model) {
    return options.SetUseIrModel(use_ir_model);
  }
};

}  // namespace litert::test

namespace litert::ml_drift {
namespace {

// Builds a model with one builtin SCATTER_ND node:
//   indices: int32 [n, 1], updates: float32 [n, h, w, c],
//   shape: constant int32 [4] = {b, h, w, c}, output: float32 [b, h, w, c].
std::vector<uint8_t> CreateScatterNdModel(int n, int b, int h, int w, int c) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes = {
      tflite::CreateOperatorCode(
          builder,
          /*deprecated_builtin_code=*/tflite::BuiltinOperator_SCATTER_ND,
          /*custom_code=*/0, /*version=*/1,
          /*builtin_code=*/tflite::BuiltinOperator_SCATTER_ND)};

  const std::vector<int32_t> shape_data = {b, h, w, c};
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));  // Buffer 0 is empty.
  buffers.push_back(tflite::CreateBuffer(
      builder,
      builder.CreateVector(reinterpret_cast<const uint8_t*>(shape_data.data()),
                           shape_data.size() * sizeof(int32_t))));

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  tensors.push_back(tflite::CreateTensor(
      builder, builder.CreateVector<int32_t>({n, 1}), tflite::TensorType_INT32,
      /*buffer=*/0, builder.CreateString("indices")));
  tensors.push_back(
      tflite::CreateTensor(builder, builder.CreateVector<int32_t>({n, h, w, c}),
                           tflite::TensorType_FLOAT32, /*buffer=*/0,
                           builder.CreateString("updates")));
  tensors.push_back(tflite::CreateTensor(
      builder, builder.CreateVector<int32_t>({4}), tflite::TensorType_INT32,
      /*buffer=*/1, builder.CreateString("shape")));
  tensors.push_back(
      tflite::CreateTensor(builder, builder.CreateVector<int32_t>({b, h, w, c}),
                           tflite::TensorType_FLOAT32, /*buffer=*/0,
                           builder.CreateString("output")));

  std::vector<flatbuffers::Offset<tflite::Operator>> operators = {
      tflite::CreateOperator(builder, /*opcode_index=*/0,
                             builder.CreateVector<int32_t>({0, 1, 2}),
                             builder.CreateVector<int32_t>({3}),
                             tflite::BuiltinOptions_ScatterNdOptions,
                             tflite::CreateScatterNdOptions(builder).Union())};

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
      tflite::CreateSubGraph(builder, builder.CreateVector(tensors),
                             builder.CreateVector<int32_t>({0, 1}),
                             builder.CreateVector<int32_t>({3}),
                             builder.CreateVector(operators),
                             builder.CreateString("main"))};

  builder.Finish(
      tflite::CreateModel(builder, TFLITE_SCHEMA_VERSION,
                          builder.CreateVector(opcodes),
                          builder.CreateVector(subgraphs),
                          builder.CreateString("ScatterND test model"),
                          builder.CreateVector(buffers)),
      tflite::ModelIdentifier());
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

void RunScatterNdTest(bool use_ir_model, bool use_fp32) {
  const int n = 3, b = 4, h = 2, w = 3, c = 5;
  // Index 2 appears twice, so its two updates are summed. Rows 1 and 3 of the
  // output receive no update and stay zero.
  const std::vector<int32_t> indices = {2, 0, 2};

  std::srand(0);
  std::vector<float> updates(n * h * w * c);
  for (float& v : updates) {
    v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
  }

  const int row = h * w * c;
  std::vector<float> expected(b * row, 0.0f);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < row; ++j) {
      expected[indices[i] * row + j] += updates[i * row + j];
    }
  }

  const std::vector<uint8_t> model = CreateScatterNdModel(n, b, h, w, c);

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);
  auto options = litert::Options::Create();
  ASSERT_TRUE(options);
  options->SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  auto gpu_options = options->GetGpuOptions();
  ASSERT_TRUE(gpu_options);
  gpu_options->SetPrecision(use_fp32 ? litert::GpuOptions::Precision::kFp32
                                     : litert::GpuOptions::Precision::kFp16);
  ASSERT_EQ(litert::test::GpuOptionsTestPeer::SetUseIrModel(*gpu_options,
                                                            use_ir_model),
            kLiteRtStatusOk);

  auto compiled_model = litert::CompiledModel::Create(
      *env, litert::BufferRef<uint8_t>(model.data(), model.size()), *options);
  ASSERT_TRUE(compiled_model);

  auto input_buffers = compiled_model->CreateInputBuffers();
  ASSERT_TRUE(input_buffers);
  auto output_buffers = compiled_model->CreateOutputBuffers();
  ASSERT_TRUE(output_buffers);
  ASSERT_EQ(input_buffers->size(), 2);
  ASSERT_EQ(output_buffers->size(), 1);

  ASSERT_TRUE((*input_buffers)[0].Write<int32_t>(absl::MakeConstSpan(indices)));
  ASSERT_TRUE((*input_buffers)[1].Write<float>(absl::MakeConstSpan(updates)));
  ASSERT_TRUE(compiled_model->Run(*input_buffers, *output_buffers));

  std::vector<float> actual(expected.size());
  ASSERT_TRUE((*output_buffers)[0].Read<float>(absl::MakeSpan(actual)));

  const float tolerance = use_fp32 ? 1e-5f : 2e-3f;
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
  }
}

TEST(ScatterNdGpuTest, GraphFloat32Fp32) {
  RunScatterNdTest(/*use_ir_model=*/false, /*use_fp32=*/true);
}

TEST(ScatterNdGpuTest, GraphFloat32Fp16) {
  RunScatterNdTest(/*use_ir_model=*/false, /*use_fp32=*/false);
}

TEST(ScatterNdGpuTest, IrModelFp32) {
  RunScatterNdTest(/*use_ir_model=*/true, /*use_fp32=*/true);
}

TEST(ScatterNdGpuTest, IrModelFp16) {
  RunScatterNdTest(/*use_ir_model=*/true, /*use_fp32=*/false);
}

}  // namespace
}  // namespace litert::ml_drift
