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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"
#include "fuzztest/fuzztest.h"
#include "tflite/array.h"
#include "tflite/c/common.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/kernels/internal/reshape_utils.h"
#include "tflite/kernels/internal/runtime_shape.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/testing/fuzzing/fuzzing_util.h"
#include "tflite/testing/fuzzing/one_op_fuzz_model.h"

#if defined(TFLITE_RESHAPE_FUZZ_ENABLE_XNNPACK)
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#endif

namespace tflite {
namespace ops {
namespace builtin {

TfLiteRegistration* Register_RESHAPE();

}  // namespace builtin
}  // namespace ops

namespace {

using fuzzing::RunResult;

enum class ShapeSpecKind {
  kOptionsOnly,
  kConstantTensor,
  kDynamicTensor,
};
enum class ExecutionMode { kBuiltin, kXnnpack };

struct ReshapeCase {
  std::vector<int32_t> input_shape;
  std::vector<int32_t> target_shape;
  std::vector<uint8_t> input_data;
  TensorType input_type;
  ShapeSpecKind shape_spec_kind;
  bool invoke;
};

struct ReshapeShapeCase {
  std::vector<int> input_shape;
  std::vector<int> target_shape;
};

constexpr size_t kMaxInputElements = 512;
constexpr size_t kMaxLiveAllocationBytes = 64 * 1024 * 1024;

void SilentReportError(TfLiteContext*, const char*, ...) {}

TfLiteContext MakeSilentContext() {
  TfLiteContext context{};
  context.ReportError = SilentReportError;
  return context;
}

RuntimeShape MakeRuntimeShape(const std::vector<int>& dims) {
  RuntimeShape shape(static_cast<int>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    shape.SetDim(i, dims[i]);
  }
  return shape;
}

bool CheckedElementCount(const std::vector<int>& shape, size_t& count) {
  size_t result = 1;
  for (const int dim : shape) {
    if (dim < 0 ||
        !fuzzing::CheckedMultiply(result, static_cast<size_t>(dim), &result)) {
      return false;
    }
  }
  count = result;
  return true;
}

bool IsSupportedInputType(TensorType type) {
  switch (type) {
    case TensorType_FLOAT32:
    case TensorType_UINT8:
    case TensorType_INT8:
    case TensorType_INT4:
    case TensorType_INT16:
    case TensorType_INT32:
    case TensorType_INT64:
    case TensorType_BOOL:
      return true;
    default:
      return false;
  }
}

TfLiteStatus ApplyXnnpackDelegate(Interpreter* interpreter) {
#if defined(TFLITE_RESHAPE_FUZZ_ENABLE_XNNPACK)
  TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
  options.num_threads = 1;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate(
      TfLiteXNNPackDelegateCreate(&options), TfLiteXNNPackDelegateDelete);
  if (delegate == nullptr) return kTfLiteError;
  return interpreter->ModifyGraphWithDelegate(std::move(delegate));
#else
  (void)interpreter;
  return kTfLiteError;
#endif
}

bool HasDelegateNode(const Interpreter& interpreter) {
  for (const int node_index : interpreter.execution_plan()) {
    const auto* node_and_registration =
        interpreter.node_and_registration(node_index);
    if (node_and_registration != nullptr &&
        node_and_registration->second.builtin_code ==
            BuiltinOperator_DELEGATE) {
      return true;
    }
  }
  return false;
}

RunResult RunReshapeCase(
    const ReshapeCase& test_case,
    ExecutionMode execution_mode = ExecutionMode::kBuiltin) {
  if (!IsSupportedInputType(test_case.input_type)) {
    return RunResult::kRejected;
  }

  size_t input_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.input_shape,
                                         &input_elements) ||
      input_elements > kMaxInputElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> input_bytes =
      fuzzing::MakeValues(test_case.input_type, input_elements, 43);
  fuzzing::OverlayBytes(test_case.input_data, &input_bytes);

  std::vector<int64_t> target_shape_values;
  target_shape_values.reserve(test_case.target_shape.size());
  for (const int32_t dim : test_case.target_shape) {
    target_shape_values.push_back(dim);
  }
  std::vector<uint8_t> target_shape_bytes =
      fuzzing::MakeIntegerValues(TensorType_INT32, target_shape_values);

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};

  const bool has_shape_tensor =
      test_case.shape_spec_kind != ShapeSpecKind::kOptionsOnly;
  const bool dynamic_shape_tensor =
      test_case.shape_spec_kind == ShapeSpecKind::kDynamicTensor;
  const uint32_t shape_buffer =
      has_shape_tensor && !dynamic_shape_tensor ? buffers.size() : 0;
  if (has_shape_tensor && !dynamic_shape_tensor) {
    buffers.push_back(
        fuzzing::CreateAlignedBuffer(&builder, target_shape_bytes));
  }

  const auto input_shape = builder.CreateVector(test_case.input_shape);
  const auto shape_tensor_shape = builder.CreateVector(std::vector<int32_t>{
      static_cast<int32_t>(test_case.target_shape.size())});
  const auto empty_output_shape = builder.CreateVector(std::vector<int32_t>{});
  flatbuffers::Offset<QuantizationParameters> quantization = 0;
  if (test_case.input_type == TensorType_INT8 ||
      test_case.input_type == TensorType_UINT8) {
    quantization = CreateQuantizationParameters(
        builder, 0, 0, builder.CreateVector<float>({0.25f}),
        builder.CreateVector<int64_t>({0}));
  }
  const auto input_tensor =
      CreateTensor(builder, input_shape, test_case.input_type, /*buffer=*/0,
                   /*name=*/0, quantization);
  const auto output_tensor =
      CreateTensor(builder, empty_output_shape, test_case.input_type,
                   /*buffer=*/0, /*name=*/0, quantization);

  std::vector<flatbuffers::Offset<Tensor>> tensors = {input_tensor};
  std::vector<int32_t> op_inputs = {0};
  std::vector<int32_t> model_inputs = {0};
  if (has_shape_tensor) {
    tensors.push_back(CreateTensor(builder, shape_tensor_shape,
                                   TensorType_INT32, shape_buffer));
    op_inputs.push_back(1);
    if (dynamic_shape_tensor) {
      model_inputs.push_back(1);
    }
  }
  tensors.push_back(output_tensor);
  const int32_t output_tensor_index = tensors.size() - 1;

  const auto reshape_options =
      CreateReshapeOptions(builder,
                           builder.CreateVector(test_case.target_shape))
          .Union();

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "reshape_fuzz";
  model_spec.builtin_operator = BuiltinOperator_RESHAPE;
  model_spec.version = 1;
  model_spec.builtin_options_type = BuiltinOptions_ReshapeOptions;
  model_spec.builtin_options = reshape_options;
  model_spec.tensors = std::move(tensors);
  model_spec.buffers = std::move(buffers);
  model_spec.model_inputs = std::move(model_inputs);
  model_spec.model_outputs = {output_tensor_index};
  model_spec.op_inputs = std::move(op_inputs);
  model_spec.op_outputs = {output_tensor_index};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_RESHAPE();
  run_spec.min_version = 1;
  run_spec.max_version = 1;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.input_shape, std::move(input_bytes)});
  if (dynamic_shape_tensor) {
    run_spec.runtime_tensors.push_back(
        {/*tensor_index=*/1,
         std::vector<int32_t>{
             static_cast<int32_t>(test_case.target_shape.size())},
         std::move(target_shape_bytes)});
  }
  if (execution_mode == ExecutionMode::kXnnpack) {
    run_spec.pre_allocate = [](Interpreter* interpreter) {
      return ApplyXnnpackDelegate(interpreter) == kTfLiteOk &&
                     HasDelegateNode(*interpreter)
                 ? RunResult::kSuccess
                 : RunResult::kRejected;
    };
  }

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

TfLiteStatus ResolveReshapeShapeCase(const ReshapeShapeCase& test_case,
                                     std::vector<int>& resolved_shape) {
  TfLiteContext context = MakeSilentContext();
  RuntimeShape input_shape = MakeRuntimeShape(test_case.input_shape);
  IntArrayUniquePtr output_shape = BuildTfLiteArray(test_case.target_shape);
  const TfLiteStatus status = reshape_internal::ResolveOutputShape(
      &context, input_shape, *output_shape);
  resolved_shape.assign(output_shape->data,
                        output_shape->data + output_shape->size);
  return status;
}

auto ValidReshapeInputShapeDomain() {
  return fuzztest::VectorOf(fuzztest::InRange<int32_t>(1, 4))
      .WithMinSize(0)
      .WithMaxSize(4);
}

auto ValidReshapeCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> input_shape, uint8_t target_kind,
         std::vector<uint8_t> input_data, TensorType input_type,
         ShapeSpecKind shape_spec_kind) {
        size_t element_count = 0;
        fuzzing::CheckedShapeElementCount(input_shape, &element_count);
        std::vector<int32_t> target_shape;
        switch (target_kind % 4) {
          case 0:
            target_shape = input_shape;
            break;
          case 1:
            target_shape = {static_cast<int32_t>(element_count)};
            break;
          case 2:
            target_shape = {1, static_cast<int32_t>(element_count)};
            break;
          case 3:
            target_shape.assign(input_shape.rbegin(), input_shape.rend());
            break;
        }
        return ReshapeCase{std::move(input_shape), std::move(target_shape),
                           std::move(input_data),  input_type,
                           shape_spec_kind,
                           /*invoke=*/true};
      },
      ValidReshapeInputShapeDomain(), fuzztest::Arbitrary<uint8_t>(),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_UINT8,
                                       TensorType_INT8, TensorType_INT4,
                                       TensorType_INT16, TensorType_INT32,
                                       TensorType_INT64, TensorType_BOOL}),
      fuzztest::ElementOf<ShapeSpecKind>({ShapeSpecKind::kOptionsOnly,
                                          ShapeSpecKind::kConstantTensor,
                                          ShapeSpecKind::kDynamicTensor}));
}

auto InvalidReshapeElementCountCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> input_shape, TensorType input_type,
         ShapeSpecKind shape_spec_kind) {
        size_t element_count = 0;
        fuzzing::CheckedShapeElementCount(input_shape, &element_count);
        return ReshapeCase{
            std::move(input_shape),
            /*target_shape=*/{static_cast<int32_t>(element_count + 1)},
            /*input_data=*/{},
            input_type,
            shape_spec_kind,
            /*invoke=*/true};
      },
      ValidReshapeInputShapeDomain(),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_UINT8,
                                       TensorType_INT8, TensorType_INT4,
                                       TensorType_INT16, TensorType_INT32,
                                       TensorType_INT64, TensorType_BOOL}),
      fuzztest::ElementOf<ShapeSpecKind>({ShapeSpecKind::kOptionsOnly,
                                          ShapeSpecKind::kConstantTensor,
                                          ShapeSpecKind::kDynamicTensor}));
}

#if defined(TFLITE_RESHAPE_FUZZ_ENABLE_XNNPACK)
auto XnnpackReshapeCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> input_shape, uint8_t target_kind,
         std::vector<uint8_t> input_data, TensorType input_type,
         ShapeSpecKind shape_spec_kind) {
        size_t element_count = 0;
        fuzzing::CheckedShapeElementCount(input_shape, &element_count);
        std::vector<int32_t> target_shape;
        switch (target_kind % 4) {
          case 0:
            target_shape = input_shape;
            break;
          case 1:
            target_shape = {static_cast<int32_t>(element_count)};
            break;
          case 2:
            target_shape = {1, static_cast<int32_t>(element_count)};
            break;
          case 3:
            target_shape.assign(input_shape.rbegin(), input_shape.rend());
            break;
        }
        return ReshapeCase{std::move(input_shape), std::move(target_shape),
                           std::move(input_data),  input_type,
                           shape_spec_kind,
                           /*invoke=*/true};
      },
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(1, 3))
          .WithMinSize(1)
          .WithMaxSize(5),
      fuzztest::Arbitrary<uint8_t>(),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_UINT8, TensorType_INT8}),
      fuzztest::ElementOf<ShapeSpecKind>(
          {ShapeSpecKind::kOptionsOnly, ShapeSpecKind::kConstantTensor}));
}
#endif

auto ValidReshapeShapeCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int> input_shape, uint8_t target_kind) {
        size_t element_count = 1;
        for (const int dim : input_shape) element_count *= dim;
        std::vector<int> target_shape;
        switch (target_kind % 3) {
          case 0:
            target_shape = input_shape;
            break;
          case 1:
            target_shape = {static_cast<int>(element_count)};
            break;
          case 2:
            target_shape = {1, static_cast<int>(element_count)};
            break;
        }
        return ReshapeShapeCase{std::move(input_shape),
                                std::move(target_shape)};
      },
      fuzztest::VectorOf(fuzztest::InRange<int>(1, 4))
          .WithMinSize(0)
          .WithMaxSize(4),
      fuzztest::Arbitrary<uint8_t>());
}

auto InvalidReshapeShapeCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int> input_shape) {
        size_t element_count = 1;
        for (const int dim : input_shape) element_count *= dim;
        return ReshapeShapeCase{
            std::move(input_shape),
            /*target_shape=*/{static_cast<int>(element_count + 1)}};
      },
      fuzztest::VectorOf(fuzztest::InRange<int>(1, 4))
          .WithMinSize(0)
          .WithMaxSize(4));
}

void ReshapeExecutesValidCases(const ReshapeCase& test_case) {
  SCOPED_TRACE(::testing::Message()
               << "input_shape="
               << ::testing::PrintToString(test_case.input_shape)
               << ", target_shape="
               << ::testing::PrintToString(test_case.target_shape)
               << ", type=" << static_cast<int>(test_case.input_type)
               << ", spec=" << static_cast<int>(test_case.shape_spec_kind));
  ASSERT_EQ(RunReshapeCase(test_case), RunResult::kSuccess);
}

void ReshapeRejectsMismatchedElementCount(const ReshapeCase& test_case) {
  ASSERT_EQ(RunReshapeCase(test_case), RunResult::kRejected);
}

void ReshapeShapeResolverAcceptsValidCases(const ReshapeShapeCase& test_case) {
  std::vector<int> resolved_shape;
  const TfLiteStatus status =
      ResolveReshapeShapeCase(test_case, resolved_shape);
  ASSERT_EQ(status, kTfLiteOk);

  for (const int dim : resolved_shape) {
    EXPECT_GE(dim, 0);
  }
  size_t input_element_count = 0;
  size_t output_element_count = 0;
  EXPECT_TRUE(CheckedElementCount(test_case.input_shape, input_element_count));
  EXPECT_TRUE(CheckedElementCount(resolved_shape, output_element_count));
  EXPECT_EQ(input_element_count, output_element_count);
}

void ReshapeShapeResolverRejectsMismatchedElementCount(
    const ReshapeShapeCase& test_case) {
  std::vector<int> resolved_shape;
  ASSERT_NE(ResolveReshapeShapeCase(test_case, resolved_shape), kTfLiteOk);
}

TEST(ReshapeFuzzTest, ScalarOutputSmoke) {
  EXPECT_EQ(RunReshapeCase({/*input_shape=*/{1},
                            /*target_shape=*/{},
                            /*input_data=*/{}, TensorType_INT32,
                            ShapeSpecKind::kOptionsOnly,
                            /*invoke=*/true}),
            RunResult::kSuccess);
}

TEST(ReshapeFuzzTest, ZeroDimWithInferredDimensionSmoke) {
  EXPECT_EQ(RunReshapeCase({/*input_shape=*/{4, 0},
                            /*target_shape=*/{2, 0, -1},
                            /*input_data=*/{}, TensorType_FLOAT32,
                            ShapeSpecKind::kConstantTensor,
                            /*invoke=*/true}),
            RunResult::kSuccess);
}

TEST(ReshapeFuzzTest, DynamicShapeTensorSmoke) {
  EXPECT_EQ(RunReshapeCase({/*input_shape=*/{2, 3},
                            /*target_shape=*/{3, 2},
                            /*input_data=*/{}, TensorType_UINT8,
                            ShapeSpecKind::kDynamicTensor,
                            /*invoke=*/true}),
            RunResult::kSuccess);
}

TEST(ReshapeShapeResolverFuzzTest, RejectsOutputShapeProductOverflow) {
  std::vector<int> resolved_shape;
  EXPECT_NE(ResolveReshapeShapeCase(
                {/*input_shape=*/{1},
                 /*target_shape=*/{std::numeric_limits<int>::max(),
                                   std::numeric_limits<int>::max(), 5}},
                resolved_shape),
            kTfLiteOk);
}

TEST(ReshapeShapeResolverFuzzTest, RejectsInputShapeProductOverflow) {
  std::vector<int> resolved_shape;
  EXPECT_NE(ResolveReshapeShapeCase(
                {/*input_shape=*/{std::numeric_limits<int>::max(),
                                  std::numeric_limits<int>::max(), 5},
                 /*target_shape=*/{1}},
                resolved_shape),
            kTfLiteOk);
}

FUZZ_TEST(ReshapeFuzzTest, ReshapeExecutesValidCases)
    .WithDomains(ValidReshapeCaseDomain());
FUZZ_TEST(ReshapeFuzzTest, ReshapeRejectsMismatchedElementCount)
    .WithDomains(InvalidReshapeElementCountCaseDomain());
FUZZ_TEST(ReshapeShapeResolverFuzzTest, ReshapeShapeResolverAcceptsValidCases)
    .WithDomains(ValidReshapeShapeCaseDomain());
FUZZ_TEST(ReshapeShapeResolverFuzzTest,
          ReshapeShapeResolverRejectsMismatchedElementCount)
    .WithDomains(InvalidReshapeShapeCaseDomain());

#if defined(TFLITE_RESHAPE_FUZZ_ENABLE_XNNPACK)
TEST(ReshapeFuzzTest, ReshapeXnnpackRankSixSmokeDelegates) {
  EXPECT_EQ(RunReshapeCase({/*input_shape=*/{1, 1, 1, 1, 2, 2},
                            /*target_shape=*/{1, 1, 1, 2, 1, 2},
                            /*input_data=*/{}, TensorType_FLOAT32,
                            ShapeSpecKind::kConstantTensor,
                            /*invoke=*/true},
                           ExecutionMode::kXnnpack),
            RunResult::kSuccess);
}

void ReshapeXnnpackExecutesValidCases(const ReshapeCase& test_case) {
  SCOPED_TRACE(::testing::Message()
               << "input_shape="
               << ::testing::PrintToString(test_case.input_shape)
               << ", target_shape="
               << ::testing::PrintToString(test_case.target_shape)
               << ", type=" << static_cast<int>(test_case.input_type)
               << ", spec=" << static_cast<int>(test_case.shape_spec_kind));
  ASSERT_EQ(RunReshapeCase(test_case, ExecutionMode::kXnnpack),
            RunResult::kSuccess);
}

FUZZ_TEST(ReshapeFuzzTest, ReshapeXnnpackExecutesValidCases)
    .WithDomains(XnnpackReshapeCaseDomain());
#endif

}  // namespace
}  // namespace tflite
