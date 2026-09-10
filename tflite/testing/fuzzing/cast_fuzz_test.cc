/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "tflite/c/common.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/testing/fuzzing/fuzzing_util.h"
#include "tflite/testing/fuzzing/one_op_fuzz_model.h"

namespace tflite {
namespace ops {
namespace builtin {

TfLiteRegistration* Register_CAST();

}  // namespace builtin
}  // namespace ops

namespace {

using fuzzing::RunResult;

struct CastCase {
  std::vector<int32_t> input_shape;
  std::vector<uint8_t> input_data;
  TensorType input_type;
  TensorType output_type;
  bool invoke;
};

constexpr size_t kMaxInputElements = 512;
constexpr size_t kMaxLiveAllocationBytes = 64 * 1024 * 1024;

bool IsMaterializableTensorType(TensorType type) {
  return fuzzing::TypeSize(type) != 0;
}

RunResult RunCastCase(const CastCase& test_case) {
  if (!IsMaterializableTensorType(test_case.input_type) ||
      !IsMaterializableTensorType(test_case.output_type)) {
    return RunResult::kRejected;
  }
  if (test_case.invoke && !test_case.input_data.empty() &&
      (test_case.input_type == TensorType_FLOAT32 ||
       test_case.input_type == TensorType_FLOAT16 ||
       test_case.input_type == TensorType_FLOAT64) &&
      test_case.output_type != TensorType_FLOAT32 &&
      test_case.output_type != TensorType_FLOAT16 &&
      test_case.output_type != TensorType_FLOAT64 &&
      test_case.output_type != TensorType_BOOL) {
    return RunResult::kRejected;
  }

  size_t input_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.input_shape,
                                         &input_elements) ||
      input_elements > kMaxInputElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> input_bytes =
      fuzzing::MakeValues(test_case.input_type, input_elements, 37);
  fuzzing::OverlayBytes(test_case.input_data, &input_bytes);
  fuzzing::ApplyCentralTensorInputInvariants(test_case.input_type,
                                             &input_bytes);

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};

  const auto input_shape = builder.CreateVector(test_case.input_shape);
  const auto output_shape = builder.CreateVector(test_case.input_shape);
  const auto input_tensor =
      CreateTensor(builder, input_shape, test_case.input_type, /*buffer=*/0);
  const auto output_tensor =
      CreateTensor(builder, output_shape, test_case.output_type);

  const auto cast_options =
      CreateCastOptions(builder, test_case.input_type, test_case.output_type)
          .Union();

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "cast_fuzz";
  model_spec.builtin_operator = BuiltinOperator_CAST;
  model_spec.version = 9;
  model_spec.builtin_options_type = BuiltinOptions_CastOptions;
  model_spec.builtin_options = cast_options;
  model_spec.tensors = {input_tensor, output_tensor};
  model_spec.buffers = std::move(buffers);
  model_spec.model_inputs = {0};
  model_spec.model_outputs = {1};
  model_spec.op_inputs = {0};
  model_spec.op_outputs = {1};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_CAST();
  run_spec.min_version = 1;
  run_spec.max_version = 9;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.input_shape, std::move(input_bytes)});
  run_spec.invoke = test_case.invoke;

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

auto SmallShapeDimDomain() {
  return fuzztest::OneOf(fuzztest::InRange<int32_t>(0, 4),
                         fuzztest::Just<int32_t>(32768),
                         fuzztest::Just<int32_t>(46341));
}

auto CastTensorTypeDomain() {
  return fuzztest::ElementOf<TensorType>(
      {TensorType_FLOAT32, TensorType_FLOAT16, TensorType_INT32,
       TensorType_UINT8, TensorType_INT64, TensorType_BOOL, TensorType_INT16,
       TensorType_COMPLEX64, TensorType_INT8, TensorType_FLOAT64,
       TensorType_UINT32, TensorType_UINT16, TensorType_INT4,
       TensorType_BFLOAT16, TensorType_INT2, TensorType_UINT4});
}

auto CastCaseDomain() {
  return fuzztest::StructOf<CastCase>(
      fuzztest::VectorOf(SmallShapeDimDomain()).WithMinSize(0).WithMaxSize(8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      CastTensorTypeDomain(), CastTensorTypeDomain(),
      fuzztest::Arbitrary<bool>());
}

void CastNeverCrashes(const CastCase& test_case) {
  EXPECT_NE(RunCastCase(test_case), RunResult::kHarnessFailure);
}

TEST(CastFuzzTest, Float32ToInt32Smoke) {
  EXPECT_EQ(
      RunCastCase({/*input_shape=*/{2, 3},
                   /*input_data=*/{}, TensorType_FLOAT32, TensorType_INT32,
                   /*invoke=*/true}),
      RunResult::kSuccess);
}

TEST(CastFuzzTest, Int4ToFloat32Smoke) {
  EXPECT_EQ(RunCastCase({/*input_shape=*/{7},
                         /*input_data=*/{}, TensorType_INT4, TensorType_FLOAT32,
                         /*invoke=*/true}),
            RunResult::kSuccess);
}

TEST(CastFuzzTest, ZeroElementSmoke) {
  EXPECT_EQ(
      RunCastCase({/*input_shape=*/{2, 0, 3},
                   /*input_data=*/{}, TensorType_INT64, TensorType_FLOAT32,
                   /*invoke=*/true}),
      RunResult::kSuccess);
}

FUZZ_TEST(CastFuzzTest, CastNeverCrashes).WithDomains(CastCaseDomain());

}  // namespace
}  // namespace tflite
