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
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"                 // from @com_google_absl
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "testing/base/public/gunit.h"
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
//   indices: int32 [n, 3] (or [1, n, 1, 3] if `four_d`),
//   updates: float32 [n, c] (or [1, n, 1, c] if `four_d`; bool if `is_bool`),
//   shape: constant int32 [4] = {b, h, w, c}, output: [b, h, w, c] with the
//   type of updates.
// If `const_updates` is set, `updates` holds the given raw bytes (of the
// updates type) as a constant and the model has only the indices as an input.
std::vector<uint8_t> CreateScatterNdModel(
    int n, int b, int h, int w, int c, bool four_d, bool is_bool = false,
    const std::vector<uint8_t>& const_updates = {}) {
  const bool has_const_updates = !const_updates.empty();
  const tflite::TensorType data_type =
      is_bool ? tflite::TensorType_BOOL : tflite::TensorType_FLOAT32;
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
  if (has_const_updates) {
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(const_updates)));
  }

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  tensors.push_back(tflite::CreateTensor(
      builder,
      builder.CreateVector<int32_t>(four_d ? std::vector<int32_t>{1, n, 1, 3}
                                           : std::vector<int32_t>{n, 3}),
      tflite::TensorType_INT32,
      /*buffer=*/0, builder.CreateString("indices")));
  tensors.push_back(tflite::CreateTensor(
      builder,
      builder.CreateVector<int32_t>(four_d ? std::vector<int32_t>{1, n, 1, c}
                                           : std::vector<int32_t>{n, c}),
      data_type, /*buffer=*/has_const_updates ? 2 : 0,
      builder.CreateString("updates")));
  tensors.push_back(tflite::CreateTensor(
      builder, builder.CreateVector<int32_t>({4}), tflite::TensorType_INT32,
      /*buffer=*/1, builder.CreateString("shape")));
  tensors.push_back(tflite::CreateTensor(
      builder, builder.CreateVector<int32_t>({b, h, w, c}), data_type,
      /*buffer=*/0, builder.CreateString("output")));

  std::vector<flatbuffers::Offset<tflite::Operator>> operators = {
      tflite::CreateOperator(builder, /*opcode_index=*/0,
                             builder.CreateVector<int32_t>({0, 1, 2}),
                             builder.CreateVector<int32_t>({3}),
                             tflite::BuiltinOptions_ScatterNdOptions,
                             tflite::CreateScatterNdOptions(builder).Union())};

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
      tflite::CreateSubGraph(
          builder, builder.CreateVector(tensors),
          has_const_updates ? builder.CreateVector<int32_t>({0})
                            : builder.CreateVector<int32_t>({0, 1}),
          builder.CreateVector<int32_t>({3}), builder.CreateVector(operators),
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

// A compiled model with its input and output buffers.
struct CompiledScatterNd {
  // Declared first so that it outlives everything that uses it.
  std::optional<litert::Environment> env;
  std::optional<litert::CompiledModel> compiled_model;
  std::vector<litert::TensorBuffer> inputs;
  std::vector<litert::TensorBuffer> outputs;
};

// Compiles `model` for the GPU and creates its buffers. Expects the model to
// have `num_inputs` inputs and one output.
void CompileScatterNd(const std::vector<uint8_t>& model, bool use_ir_model,
                      bool use_fp32, size_t num_inputs,
                      CompiledScatterNd& result) {
  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);
  result.env.emplace(std::move(*env));
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
      *result.env, litert::BufferRef<uint8_t>(model.data(), model.size()),
      *options);
  ASSERT_TRUE(compiled_model);
  result.compiled_model.emplace(std::move(*compiled_model));

  auto input_buffers = result.compiled_model->CreateInputBuffers();
  ASSERT_TRUE(input_buffers);
  auto output_buffers = result.compiled_model->CreateOutputBuffers();
  ASSERT_TRUE(output_buffers);
  ASSERT_EQ(input_buffers->size(), num_inputs);
  ASSERT_EQ(output_buffers->size(), 1);
  result.inputs = std::move(*input_buffers);
  result.outputs = std::move(*output_buffers);
}

// `b` is the output batch (b == 1 gives an output without a batch axis) and `n`
// the number of indices (1 <= n <= 4). If `out_of_range` is set (n >= 3), two
// index rows are out of range and must be ignored.
void RunScatterNdTest(bool use_ir_model, bool use_fp32, bool four_d = false,
                      int b = 2, int n = 4, bool out_of_range = false) {
  const int h = 3, w = 4, c = 5;
  // Each row is a (b, h, w) index. With n == 4, the first and last rows address
  // the same output cell, so their updates are summed. Cells that no row
  // addresses stay zero.
  const std::vector<int32_t> all_indices = {b - 1, 2, 3,  //
                                            0,     0, 0,  //
                                            b - 1, 1, 2,  //
                                            b - 1, 2, 3};
  std::vector<int32_t> indices(all_indices.begin(),
                               all_indices.begin() + n * 3);
  if (out_of_range) {
    // Height equal to the output height, and a negative batch.
    indices[3 + 1] = h;
    indices[6] = -1;
  }

  std::srand(0);
  std::vector<float> updates(n * c);
  for (float& v : updates) {
    v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
  }

  std::vector<float> expected(b * h * w * c, 0.0f);
  for (int i = 0; i < n; ++i) {
    if (indices[i * 3] < 0 || indices[i * 3] >= b || indices[i * 3 + 1] < 0 ||
        indices[i * 3 + 1] >= h || indices[i * 3 + 2] < 0 ||
        indices[i * 3 + 2] >= w) {
      continue;
    }
    const int cell =
        (indices[i * 3] * h + indices[i * 3 + 1]) * w + indices[i * 3 + 2];
    for (int j = 0; j < c; ++j) {
      expected[cell * c + j] += updates[i * c + j];
    }
  }

  const std::vector<uint8_t> model =
      CreateScatterNdModel(n, b, h, w, c, four_d);

  CompiledScatterNd compiled;
  ASSERT_NO_FATAL_FAILURE(CompileScatterNd(model, use_ir_model, use_fp32,
                                           /*num_inputs=*/2, compiled));

  ASSERT_TRUE(compiled.inputs[0].Write<int32_t>(absl::MakeConstSpan(indices)));
  ASSERT_TRUE(compiled.inputs[1].Write<float>(absl::MakeConstSpan(updates)));
  ASSERT_TRUE(compiled.compiled_model->Run(compiled.inputs, compiled.outputs));

  std::vector<float> actual(expected.size());
  ASSERT_TRUE(compiled.outputs[0].Read<float>(absl::MakeSpan(actual)));

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

// Bool updates that share an output cell are OR-ed. The first and last rows
// address the same cell. If `const_updates` is set, the updates are a constant
// of the model and only the indices are an input.
void RunScatterNdBoolTest(bool use_ir_model, bool four_d,
                          bool const_updates = false) {
  const int n = 4, b = 2, h = 3, w = 4, c = 5;
  const std::vector<int32_t> indices = {1, 2, 3,  //
                                        0, 0, 0,  //
                                        1, 1, 2,  //
                                        1, 2, 3};
  auto updates = std::make_unique<bool[]>(n * c);
  for (int i = 0; i < n * c; ++i) updates[i] = (i % 3) != 0;

  std::vector<char> expected(b * h * w * c, 0);
  for (int i = 0; i < n; ++i) {
    const int cell =
        (indices[i * 3] * h + indices[i * 3 + 1]) * w + indices[i * 3 + 2];
    for (int j = 0; j < c; ++j) {
      expected[cell * c + j] |= updates[i * c + j] ? 1 : 0;
    }
  }

  const std::vector<uint8_t> model = CreateScatterNdModel(
      n, b, h, w, c, four_d, /*is_bool=*/true,
      /*const_updates=*/
      const_updates ? std::vector<uint8_t>(updates.get(), updates.get() + n * c)
                    : std::vector<uint8_t>());

  CompiledScatterNd compiled;
  ASSERT_NO_FATAL_FAILURE(CompileScatterNd(model, use_ir_model,
                                           /*use_fp32=*/true,
                                           /*num_inputs=*/const_updates ? 1 : 2,
                                           compiled));

  ASSERT_TRUE(compiled.inputs[0].Write<int32_t>(absl::MakeConstSpan(indices)));
  if (!const_updates) {
    ASSERT_TRUE(compiled.inputs[1].Write<bool>(
        absl::MakeConstSpan(updates.get(), n * c)));
  }
  ASSERT_TRUE(compiled.compiled_model->Run(compiled.inputs, compiled.outputs));

  auto actual = std::make_unique<bool[]>(expected.size());
  ASSERT_TRUE(compiled.outputs[0].Read<bool>(
      absl::MakeSpan(actual.get(), expected.size())));
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i] != 0) << "at index " << i;
  }
}

TEST(ScatterNdGpuTest, GraphFloat32Bool) {
  RunScatterNdBoolTest(/*use_ir_model=*/false, /*four_d=*/false);
}

TEST(ScatterNdGpuTest, IrModelBool) {
  RunScatterNdBoolTest(/*use_ir_model=*/true, /*four_d=*/false);
}

TEST(ScatterNdGpuTest, GraphFloat32BoolFourD) {
  RunScatterNdBoolTest(/*use_ir_model=*/false, /*four_d=*/true);
}

TEST(ScatterNdGpuTest, IrModelBoolFourD) {
  RunScatterNdBoolTest(/*use_ir_model=*/true, /*four_d=*/true);
}

// 4D shapes: indices [1, n, 1, 3] and updates [1, n, 1, c].
TEST(ScatterNdGpuTest, GraphFloat32Fp32FourD) {
  RunScatterNdTest(/*use_ir_model=*/false, /*use_fp32=*/true, /*four_d=*/true);
}

TEST(ScatterNdGpuTest, GraphFloat32Fp16FourD) {
  RunScatterNdTest(/*use_ir_model=*/false, /*use_fp32=*/false, /*four_d=*/true);
}

TEST(ScatterNdGpuTest, IrModelFp32FourD) {
  RunScatterNdTest(/*use_ir_model=*/true, /*use_fp32=*/true, /*four_d=*/true);
}

TEST(ScatterNdGpuTest, IrModelFp16FourD) {
  RunScatterNdTest(/*use_ir_model=*/true, /*use_fp32=*/false, /*four_d=*/true);
}

// Output batch of 1, so the output has no batch axis (HWC layout), as in
// [1, 8, 1, 3] -> [1, 8, 127, 64].
TEST(ScatterNdGpuTest, GraphFloat32Fp32BatchOne) {
  RunScatterNdTest(/*use_ir_model=*/false, /*use_fp32=*/true,
                   /*four_d=*/false, /*b=*/1);
}

TEST(ScatterNdGpuTest, IrModelFp32BatchOne) {
  RunScatterNdTest(/*use_ir_model=*/true, /*use_fp32=*/true,
                   /*four_d=*/false, /*b=*/1);
}

TEST(ScatterNdGpuTest, GraphFloat32Fp32BatchOneFourD) {
  RunScatterNdTest(/*use_ir_model=*/false, /*use_fp32=*/true,
                   /*four_d=*/true, /*b=*/1);
}

TEST(ScatterNdGpuTest, IrModelFp32BatchOneFourD) {
  RunScatterNdTest(/*use_ir_model=*/true, /*use_fp32=*/true,
                   /*four_d=*/true, /*b=*/1);
}

// A single index, so the 2D indices tensor is [1, 3].
TEST(ScatterNdGpuTest, GraphFloat32Fp32SingleIndex) {
  RunScatterNdTest(/*use_ir_model=*/false, /*use_fp32=*/true,
                   /*four_d=*/false, /*b=*/2, /*n=*/1);
}

TEST(ScatterNdGpuTest, IrModelFp32SingleIndex) {
  RunScatterNdTest(/*use_ir_model=*/true, /*use_fp32=*/true,
                   /*four_d=*/false, /*b=*/2, /*n=*/1);
}

TEST(ScatterNdGpuTest, GraphFloat32Fp32SingleIndexBatchOne) {
  RunScatterNdTest(/*use_ir_model=*/false, /*use_fp32=*/true,
                   /*four_d=*/false, /*b=*/1, /*n=*/1);
}

TEST(ScatterNdGpuTest, IrModelFp32SingleIndexBatchOne) {
  RunScatterNdTest(/*use_ir_model=*/true, /*use_fp32=*/true,
                   /*four_d=*/false, /*b=*/1, /*n=*/1);
}

// Out-of-range indices are ignored, unlike the CPU kernel, which fails.
TEST(ScatterNdGpuTest, GraphFloat32Fp32OutOfRangeIndices) {
  RunScatterNdTest(/*use_ir_model=*/false, /*use_fp32=*/true,
                   /*four_d=*/false, /*b=*/2, /*n=*/4, /*out_of_range=*/true);
}

TEST(ScatterNdGpuTest, IrModelFp32OutOfRangeIndices) {
  RunScatterNdTest(/*use_ir_model=*/true, /*use_fp32=*/true,
                   /*four_d=*/false, /*b=*/2, /*n=*/4, /*out_of_range=*/true);
}

TEST(ScatterNdGpuTest, GraphFloat32Fp32OutOfRangeIndicesBatchOne) {
  RunScatterNdTest(/*use_ir_model=*/false, /*use_fp32=*/true,
                   /*four_d=*/false, /*b=*/1, /*n=*/4, /*out_of_range=*/true);
}

TEST(ScatterNdGpuTest, IrModelFp32OutOfRangeIndicesBatchOne) {
  RunScatterNdTest(/*use_ir_model=*/true, /*use_fp32=*/true,
                   /*four_d=*/false, /*b=*/1, /*n=*/4, /*out_of_range=*/true);
}

// The updates are a constant of the model. Rows 0 and 3 address the same output
// cell, so their updates are summed.
void RunScatterNdConstUpdatesTest(bool use_ir_model, bool use_fp32,
                                  bool four_d) {
  const int n = 4, b = 2, h = 3, w = 4, c = 5;
  const std::vector<int32_t> indices = {1, 2, 3,  //
                                        0, 0, 0,  //
                                        1, 1, 2,  //
                                        1, 2, 3};

  std::srand(0);
  std::vector<float> updates(n * c);
  for (float& v : updates) {
    v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
  }

  std::vector<float> expected(b * h * w * c, 0.0f);
  for (int i = 0; i < n; ++i) {
    const int cell =
        (indices[i * 3] * h + indices[i * 3 + 1]) * w + indices[i * 3 + 2];
    for (int j = 0; j < c; ++j) {
      expected[cell * c + j] += updates[i * c + j];
    }
  }

  const std::vector<uint8_t> model = CreateScatterNdModel(
      n, b, h, w, c, four_d, /*is_bool=*/false,
      /*const_updates=*/
      std::vector<uint8_t>(
          reinterpret_cast<const uint8_t*>(updates.data()),
          reinterpret_cast<const uint8_t*>(updates.data() + updates.size())));

  CompiledScatterNd compiled;
  // Only the indices are an input.
  ASSERT_NO_FATAL_FAILURE(CompileScatterNd(model, use_ir_model, use_fp32,
                                           /*num_inputs=*/1, compiled));

  ASSERT_TRUE(compiled.inputs[0].Write<int32_t>(absl::MakeConstSpan(indices)));
  ASSERT_TRUE(compiled.compiled_model->Run(compiled.inputs, compiled.outputs));

  std::vector<float> actual(expected.size());
  ASSERT_TRUE(compiled.outputs[0].Read<float>(absl::MakeSpan(actual)));

  const float tolerance = use_fp32 ? 1e-5f : 2e-3f;
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
  }
}

TEST(ScatterNdGpuTest, GraphFloat32ConstUpdatesFp32) {
  RunScatterNdConstUpdatesTest(/*use_ir_model=*/false, /*use_fp32=*/true,
                               /*four_d=*/false);
}

TEST(ScatterNdGpuTest, GraphFloat32ConstUpdatesFp16) {
  RunScatterNdConstUpdatesTest(/*use_ir_model=*/false, /*use_fp32=*/false,
                               /*four_d=*/false);
}

TEST(ScatterNdGpuTest, IrModelConstUpdatesFp32) {
  RunScatterNdConstUpdatesTest(/*use_ir_model=*/true, /*use_fp32=*/true,
                               /*four_d=*/false);
}

TEST(ScatterNdGpuTest, IrModelConstUpdatesFp16) {
  RunScatterNdConstUpdatesTest(/*use_ir_model=*/true, /*use_fp32=*/false,
                               /*four_d=*/false);
}

TEST(ScatterNdGpuTest, GraphFloat32ConstUpdatesFourD) {
  RunScatterNdConstUpdatesTest(/*use_ir_model=*/false, /*use_fp32=*/true,
                               /*four_d=*/true);
}

TEST(ScatterNdGpuTest, IrModelConstUpdatesFourD) {
  RunScatterNdConstUpdatesTest(/*use_ir_model=*/true, /*use_fp32=*/true,
                               /*four_d=*/true);
}

TEST(ScatterNdGpuTest, GraphFloat32ConstUpdatesBool) {
  RunScatterNdBoolTest(/*use_ir_model=*/false, /*four_d=*/false,
                       /*const_updates=*/true);
}

TEST(ScatterNdGpuTest, IrModelConstUpdatesBool) {
  RunScatterNdBoolTest(/*use_ir_model=*/true, /*four_d=*/false,
                       /*const_updates=*/true);
}

TEST(ScatterNdGpuTest, GraphFloat32ConstUpdatesBoolFourD) {
  RunScatterNdBoolTest(/*use_ir_model=*/false, /*four_d=*/true,
                       /*const_updates=*/true);
}

TEST(ScatterNdGpuTest, IrModelConstUpdatesBoolFourD) {
  RunScatterNdBoolTest(/*use_ir_model=*/true, /*four_d=*/true,
                       /*const_updates=*/true);
}

}  // namespace
}  // namespace litert::ml_drift
