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

#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/matchers.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/core/kernels/register.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/schema/mutable/schema_generated.h"
#include "tflite/test_util.h"

namespace litert::tensor {
namespace {

using ::litert::tensor::IsOk;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::IsNull;
using ::testing::Lt;
using ::testing::Not;
using ::testing::Pointee;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::testing::UnorderedElementsAre;
using ::testing::litert::AlignmentIs;
using ::tflite::TfLiteArrayIs;

using TensorTf = Tensor<TfLiteMixinTag>;

// Retrieves the tensor with the given `name`.
//
// `tensors` must be the list of tensor indices to search in
// `interpreter.tensors()`. Usually this is either `interpreter.inputs()` or
// `interpreter.outputs()`.
absl::StatusOr<TfLiteTensor&> GetTensor(const absl::string_view name,
                                        tflite::Interpreter& interpreter,
                                        const std::vector<int>& tensors) {
  if (name.empty()) {
    return absl::InvalidArgumentError("No name to look up.");
  }
  for (int t : tensors) {
    TfLiteTensor* tensor = interpreter.tensor(t);
    if (tensor && tensor->name == name) {
      return *tensor;
    }
  }
  return absl::NotFoundError("Named input tensor not found");
}

// Retrieves the input tensor with the given `name` in the `interpreter`.
absl::StatusOr<TfLiteTensor&> GetInputTensor(const absl::string_view name,
                                             tflite::Interpreter& interpreter) {
  return GetTensor(name, interpreter, interpreter.inputs());
}

// Retrieves the output tensor with the given `name` in the `interpreter`.
absl::StatusOr<TfLiteTensor&> GetOutputTensor(
    const absl::string_view name, tflite::Interpreter& interpreter) {
  return GetTensor(name, interpreter, interpreter.outputs());
}

TEST(SerializationTest, BuildOneSubgraphAndRunIt) {
  {
    TensorTf a({.name = "a", .type = Type::kI32, .shape = {3, 3}});
    TensorTf b({.name = "b", .type = Type::kI32, .shape = {3, 3}});
    TensorTf c({.name = "c", .type = Type::kI32, .shape = {3, 3}});
    TensorTf d = Mul(a, b);
    TensorTf e = Add(c, d);
    d.SetName("d");
    e.SetName("e");
    ASSERT_THAT(Save({e, d}, "/tmp/fma.tflite"), IsOk());
  }

  auto model = tflite::FlatBufferModel::BuildFromFile("/tmp/fma.tflite");
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->inputs().size(), 3);
  ASSERT_EQ(interpreter->outputs().size(), 2);

  {  // We are scoping these tests because these references may go stale when
    // calling `Invoke`.
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TfLiteTensor & a,
                                    GetInputTensor("a", *interpreter));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TfLiteTensor & b,
                                    GetInputTensor("b", *interpreter));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TfLiteTensor & c,
                                    GetInputTensor("c", *interpreter));

    EXPECT_EQ(a.type, kTfLiteInt32);
    EXPECT_THAT(a.dims, TfLiteArrayIs({3, 3}));

    EXPECT_EQ(b.type, kTfLiteInt32);
    EXPECT_THAT(b.dims, TfLiteArrayIs({3, 3}));

    EXPECT_EQ(c.type, kTfLiteInt32);
    EXPECT_THAT(c.dims, TfLiteArrayIs({3, 3}));

    {  // Setting input 0.
      const int32_t input_data_ref[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      std::memcpy(reinterpret_cast<int32_t*>(a.data.data), input_data_ref,
                  sizeof(input_data_ref));
    }
    {  // Setting input 1.
      const int32_t input_data_ref[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
      std::memcpy(reinterpret_cast<int32_t*>(b.data.data), input_data_ref,
                  sizeof(input_data_ref));
    }
    {  // Setting input 2.
      const int32_t input_data_ref[9] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
      std::memcpy(reinterpret_cast<int32_t*>(c.data.data), input_data_ref,
                  sizeof(input_data_ref));
    }
  }

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TfLiteTensor & d,
                                    GetOutputTensor("d", *interpreter));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TfLiteTensor & e,
                                    GetOutputTensor("e", *interpreter));

    EXPECT_EQ(d.type, kTfLiteInt32);
    EXPECT_THAT(d.dims, TfLiteArrayIs({3, 3}));

    EXPECT_EQ(e.type, kTfLiteInt32);
    EXPECT_THAT(e.dims, TfLiteArrayIs({3, 3}));

    {  // Checking output 0.
      absl::Span<const int32_t> output_data(
          reinterpret_cast<int32_t*>(d.data.data), 9);
      EXPECT_THAT(output_data, ElementsAre(9, 16, 21, 24, 25, 24, 21, 16, 9));
    }
    {  // Checking output 1.
      absl::Span<const int32_t> output_data(
          reinterpret_cast<int32_t*>(e.data.data), 9);
      EXPECT_THAT(output_data, ElementsAre(11, 18, 23, 26, 27, 26, 23, 18, 11));
    }
  }
}

TEST(SerializationTest, BuildTwoSubgraphs) {
  const std::string model_path =
      testing::TempDir() + "/" +
      testing::UnitTest::GetInstance()->current_test_info()->name() + ".tflite";
  ModelFactory model_builder;
  {
    TensorTf a({.name = "a", .type = Type::kI32, .shape = {3, 3}});
    TensorTf b({.name = "b", .type = Type::kI32, .shape = {3, 3}});
    TensorTf c({.name = "c", .type = Type::kI32, .shape = {3, 3}});
    TensorTf d = Mul(a, b).SetName("d");
    TensorTf e = Add(c, d).SetName("e");
    EXPECT_THAT(model_builder.AddSubgraph({e, d}), IsOk());
  }
  {
    TensorTf f({.name = "f", .type = Type::kI32, .shape = {3, 3}});
    TensorTf g({.name = "g", .type = Type::kI32, .shape = {3, 3}});
    TensorTf h = Add(f, g).SetName("h");
    EXPECT_THAT(model_builder.AddSubgraph({h}), IsOk());
  }
  EXPECT_THAT(model_builder.Save(model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->subgraphs_size(), 2);
  EXPECT_THAT(interpreter->subgraph(0)->inputs(), SizeIs(3));
  EXPECT_THAT(interpreter->subgraph(0)->outputs(), SizeIs(2));
  EXPECT_THAT(interpreter->subgraph(1)->inputs(), SizeIs(2));
  EXPECT_THAT(interpreter->subgraph(1)->outputs(), SizeIs(1));
}

TEST(SerializationTest, BuildTwoSignatures) {
  const std::string model_path =
      testing::TempDir() + "/" +
      testing::UnitTest::GetInstance()->current_test_info()->name() + ".tflite";
  const char kSignature1[] = "signature1";
  const char kSignature2[] = "signature2";
  ModelFactory model_builder;
  {
    TensorTf a({.name = "a", .type = Type::kI32, .shape = {3, 3}});
    TensorTf b({.name = "b", .type = Type::kI32, .shape = {3, 3}});
    TensorTf c({.name = "c", .type = Type::kI32, .shape = {3, 3}});
    TensorTf d = Mul(a, b).SetName("d");
    TensorTf e = Add(c, d).SetName("e");
    EXPECT_THAT(model_builder.AddSignature({e, d}, /*name=*/kSignature1),
                IsOk());
  }
  {
    TensorTf f({.name = "f", .type = Type::kI32, .shape = {3, 3}});
    TensorTf g({.name = "g", .type = Type::kI32, .shape = {3, 3}});
    TensorTf h = Add(f, g).SetName("h");
    EXPECT_THAT(model_builder.AddSignature({h}, /*name=*/kSignature2), IsOk());
  }
  EXPECT_THAT(model_builder.Save(model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_THAT(interpreter->signature_keys(),
              UnorderedElementsAre(Pointee(StrEq(kSignature1)),
                                   Pointee(StrEq(kSignature2))));
  EXPECT_THAT(interpreter->signature_inputs(kSignature1), SizeIs(3));
  EXPECT_THAT(interpreter->signature_outputs(kSignature1), SizeIs(2));
  EXPECT_THAT(interpreter->input_tensor_by_signature("a", kSignature1),
              Not(nullptr));
  EXPECT_THAT(interpreter->input_tensor_by_signature("b", kSignature1),
              Not(nullptr));
  EXPECT_THAT(interpreter->input_tensor_by_signature("c", kSignature1),
              Not(nullptr));
  EXPECT_THAT(interpreter->output_tensor_by_signature("d", kSignature1),
              Not(nullptr));
  EXPECT_THAT(interpreter->output_tensor_by_signature("e", kSignature1),
              Not(nullptr));

  EXPECT_THAT(interpreter->signature_inputs(kSignature2), SizeIs(2));
  EXPECT_THAT(interpreter->signature_outputs(kSignature2), SizeIs(1));
  EXPECT_THAT(interpreter->input_tensor_by_signature("f", kSignature2),
              Not(nullptr));
  EXPECT_THAT(interpreter->input_tensor_by_signature("g", kSignature2),
              Not(nullptr));
  EXPECT_THAT(interpreter->output_tensor_by_signature("h", kSignature2),
              Not(nullptr));
}

TEST(SerializationTest, AddingAnEmptySubgraphFails) {
  ModelFactory model_builder;
  EXPECT_THAT(model_builder.AddSubgraph({}), Not(IsOk()));
}

TEST(SerializationTest, AddingAnEmptySignatureFails) {
  ModelFactory model_builder;
  EXPECT_THAT(model_builder.AddSignature({}, /*name=*/"sig-name"), Not(IsOk()));
}

TEST(SerializationTest, ConstantTensorWorks) {
  const std::string model_path = testing::TempDir() + "/mul.tflite";
  const std::vector<int32_t> a_data = {1, 2, 3, 4};
  const std::vector<int32_t> b_data = {8, 7, 6, 5};
  {
    TensorTf a({.type = Type::kI32, .shape = {2, 2}, .buffer = a_data});
    TensorTf b({.type = Type::kI32, .shape = {2, 2}, .buffer = b_data});
    TensorTf c = Mul(a, b).SetName("c");
    ASSERT_THAT(Save({c}, model_path), IsOk());
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->inputs().size(), 0);
  ASSERT_EQ(interpreter->outputs().size(), 1);

  ASSERT_EQ(interpreter->tensors_size(), 3);
  ASSERT_THAT(interpreter->tensor(0)->data.raw_const,
              AlignmentIs(alignof(double)));
  ASSERT_THAT(interpreter->tensor(1)->data.raw_const,
              AlignmentIs(alignof(double)));

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TfLiteTensor & c,
                                    GetOutputTensor("c", *interpreter));
    EXPECT_EQ(c.type, kTfLiteInt32);
    EXPECT_THAT(c.dims, TfLiteArrayIs({2, 2}));
    absl::Span<const int32_t> output_data(
        reinterpret_cast<int32_t*>(c.data.data), 4);
    EXPECT_THAT(output_data, ElementsAre(8, 14, 18, 20));
  }
}

TEST(SerializationTest, SharedConstantTensorAcrossSubgraphs) {
  const std::string model_path = testing::TempDir() + "/shared_const.tflite";
  const std::vector<int32_t> shared_data = {1, 2, 3, 4};
  ModelFactory model_builder;

  std::shared_ptr<Buffer> shared_buffer =
      OwningCpuBuffer::Copy<Type::kI32>(shared_data);

  {
    TensorTf a({.type = Type::kI32, .shape = {2, 2}, .buffer = shared_buffer});
    TensorTf b({.type = Type::kI32,
                .shape = {2, 2},
                .buffer = std::vector<int32_t>{8, 7, 6, 5}});
    TensorTf c = Mul(a, b).SetName("c");
    ASSERT_THAT(model_builder.AddSignature({c}, /*name=*/"sig1"), IsOk());
  }
  {
    TensorTf a({.type = Type::kI32, .shape = {2, 2}, .buffer = shared_buffer});
    TensorTf d({.type = Type::kI32,
                .shape = {2, 2},
                .buffer = std::vector<int32_t>{1, 1, 1, 1}});
    TensorTf e = Add(a, d).SetName("e");
    ASSERT_THAT(model_builder.AddSignature({e}, /*name=*/"sig2"), IsOk());
  }

  ASSERT_THAT(model_builder.Save(model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);

  // Verify that the model has only 4 buffers:
  // 0: empty, 1: shared_data, 2: {8,7,6,5}, 3: {1,1,1,1}
  EXPECT_THAT(model->GetModel()->buffers(), Pointee(SizeIs(4)));
}

TEST(SerializationTest, CreateFlatbufferWorks) {
  const std::vector<int32_t> a_data = {1, 2, 3, 4};
  const std::vector<int32_t> b_data = {8, 7, 6, 5};
  std::vector<char> model_data;
  {
    TensorTf a(
        {.name = "a", .type = Type::kI32, .shape = {2, 2}, .buffer = a_data});
    TensorTf b(
        {.name = "b", .type = Type::kI32, .shape = {2, 2}, .buffer = b_data});
    TensorTf c = Mul(a, b).SetName("c");
    ModelFactory model_builder;
    ASSERT_THAT(model_builder.AddSubgraph({c}), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(model_data,
                                    model_builder.CreateFlatbuffer());
  }
  auto model = tflite::FlatBufferModel::BuildFromBuffer(model_data.data(),
                                                        model_data.size());
  ASSERT_THAT(model->GetModel()->buffers(), Pointee(SizeIs(3)));
  auto buffer_0 = model->GetModel()->buffers()->Get(0);
  EXPECT_THAT(buffer_0->offset(), Lt(model_data.size()));
  EXPECT_THAT(buffer_0->offset() + buffer_0->size(), Lt(model_data.size()));
  EXPECT_THAT(buffer_0->offset(), AlignmentIs(kCpuBufferAlignment));
  auto buffer_1 = model->GetModel()->buffers()->Get(1);
  ASSERT_THAT(buffer_1->offset(), Lt(model_data.size()));
  ASSERT_THAT(buffer_1->offset() + buffer_1->size(), Lt(model_data.size()));
  EXPECT_THAT(buffer_1->offset(), AlignmentIs(kCpuBufferAlignment));
  auto buffer_2 = model->GetModel()->buffers()->Get(2);
  ASSERT_THAT(buffer_2->offset(), Lt(model_data.size()));
  ASSERT_THAT(buffer_2->offset() + buffer_2->size(), Lt(model_data.size()));
  EXPECT_THAT(buffer_2->offset(), AlignmentIs(kCpuBufferAlignment));

  ASSERT_THAT(model->GetModel()->subgraphs(), Pointee(SizeIs(1)));
  auto subgraph = model->GetModel()->subgraphs()->Get(0);

  EXPECT_THAT(subgraph->tensors(), Pointee(SizeIs(3)));
  std::unordered_map<std::string, const tflite::Buffer*> named_buffers;
  ASSERT_THAT(subgraph->tensors()->Get(0)->name(), Not(IsNull()));
  named_buffers.emplace(
      subgraph->tensors()->Get(0)->name()->str(),
      model->GetModel()->buffers()->Get(subgraph->tensors()->Get(0)->buffer()));
  ASSERT_THAT(subgraph->tensors()->Get(1)->name(), Not(IsNull()));
  named_buffers.emplace(
      subgraph->tensors()->Get(1)->name()->str(),
      model->GetModel()->buffers()->Get(subgraph->tensors()->Get(1)->buffer()));
  ASSERT_THAT(subgraph->tensors()->Get(2)->name(), Not(IsNull()));
  named_buffers.emplace(
      subgraph->tensors()->Get(2)->name()->str(),
      model->GetModel()->buffers()->Get(subgraph->tensors()->Get(2)->buffer()));
  EXPECT_THAT(absl::Span<const int32_t>(
                  reinterpret_cast<int32_t*>(model_data.data() +
                                             named_buffers["a"]->offset()),
                  named_buffers["a"]->size() / sizeof(int32_t)),
              ElementsAreArray(a_data));
  EXPECT_THAT(absl::Span<const int32_t>(
                  reinterpret_cast<int32_t*>(model_data.data() +
                                             named_buffers["b"]->offset()),
                  named_buffers["b"]->size() / sizeof(int32_t)),
              ElementsAreArray(b_data));

  EXPECT_THAT(subgraph->inputs(), IsNull());
  EXPECT_THAT(subgraph->outputs(), Pointee(SizeIs(1)));
  EXPECT_THAT(subgraph->operators(), Pointee(SizeIs(1)));
  auto op = subgraph->operators()->Get(0);
  EXPECT_THAT(op->opcode_index(), Eq(0));
  EXPECT_THAT(op->inputs(), Pointee(SizeIs(2)));
  EXPECT_THAT(op->outputs(), Pointee(SizeIs(1)));

  ASSERT_THAT(model->GetModel()->operator_codes(), Pointee(SizeIs(1)));
  EXPECT_THAT(model->GetModel()->operator_codes()->Get(0)->builtin_code(),
              Eq(tflite::BuiltinOperator_MUL));
}

TEST(SerializationTest, UnnamedInputsAndOutputsAreGivenAName) {
  std::vector<char> model_data;
  {
    TensorTf a({.type = Type::kI32, .shape = {2, 2}});
    TensorTf b({.type = Type::kI32, .shape = {2, 2}});
    TensorTf c = Abs(a);
    TensorTf d = Abs(b);
    ModelFactory model_builder;
    ASSERT_THAT(model_builder.AddSubgraph({c, d}), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(model_data,
                                    model_builder.CreateFlatbuffer());

    EXPECT_THAT((std::vector{a.GetName(), b.GetName()}),
                UnorderedElementsAre(StrEq("unnamed_input_0"),
                                     StrEq("unnamed_input_1")));
    EXPECT_THAT((std::vector{c.GetName(), d.GetName()}),
                UnorderedElementsAre(StrEq("unnamed_output_0"),
                                     StrEq("unnamed_output_1")));
  }
  auto model = tflite::FlatBufferModel::BuildFromBuffer(model_data.data(),
                                                        model_data.size());

  auto GetInputName = [&](int idx) {
    return model->GetModel()
        ->subgraphs()
        ->Get(0)
        ->tensors()
        ->Get(model->GetModel()->subgraphs()->Get(0)->inputs()->Get(idx))
        ->name()
        ->str();
  };

  auto GetOutputName = [&](int idx) {
    return model->GetModel()
        ->subgraphs()
        ->Get(0)
        ->tensors()
        ->Get(model->GetModel()->subgraphs()->Get(0)->outputs()->Get(idx))
        ->name()
        ->str();
  };

  ASSERT_THAT(GetInputName(0), StrEq("unnamed_input_0"));
  ASSERT_THAT(GetInputName(1), StrEq("unnamed_input_1"));
  ASSERT_THAT(GetOutputName(0), StrEq("unnamed_output_0"));
  ASSERT_THAT(GetOutputName(1), StrEq("unnamed_output_1"));
}

TEST(SerializationTest, CanSerializeAdd) {
  const std::string model_path = testing::TempDir() + "/add.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Add(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());
  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  const auto* add_options = reinterpret_cast<const TfLiteAddParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(add_options, nullptr);
  EXPECT_EQ(add_options->activation, kTfLiteActNone);
}

TEST(SerializationTest, CanSerializeSoftmax) {
  const std::string model_path = testing::TempDir() + "/softmax.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Softmax(a, 2.0f);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  const auto* softmax_options = reinterpret_cast<const TfLiteSoftmaxParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(softmax_options, nullptr);
  EXPECT_EQ(softmax_options->beta, 2.0f);
}

TEST(SerializationTest, CanSerializeLogSoftmax) {
  const std::string model_path = testing::TempDir() + "/log_softmax.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = LogSoftmax(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_LOG_SOFTMAX);
}

TEST(SerializationTest, CanSerializeRelu) {
  const std::string model_path = testing::TempDir() + "/relu.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Relu(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_RELU);
}

TEST(SerializationTest, CanSerializeRelu6) {
  const std::string model_path = testing::TempDir() + "/relu6.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Relu6(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_RELU6);
}

TEST(SerializationTest, CanSerializeLeakyRelu) {
  const std::string model_path = testing::TempDir() + "/leaky_relu.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = LeakyRelu(a, 0.2f);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_LEAKY_RELU);
  const auto* leaky_relu_options =
      reinterpret_cast<const TfLiteLeakyReluParams*>(
          node_and_reg->first.builtin_data);
  ASSERT_NE(leaky_relu_options, nullptr);
  EXPECT_EQ(leaky_relu_options->alpha, 0.2f);
}

TEST(SerializationTest, CanSerializeElu) {
  const std::string model_path = testing::TempDir() + "/elu.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Elu(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_ELU);
}

TEST(SerializationTest, CanSerializeHardSwish) {
  const std::string model_path = testing::TempDir() + "/hardswish.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = HardSwish(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_HARD_SWISH);
}

TEST(SerializationTest, CanSerializePRelu) {
  const std::string model_path = testing::TempDir() + "/prelu.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf alpha({.type = Type::kFP32, .shape = {5}});
  TensorTf c = PRelu(a, alpha);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_PRELU);
}

TEST(SerializationTest, CanSerializeL2Normalization) {
  const std::string model_path =
      testing::TempDir() + "/l2_normalization.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = L2Normalization(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_L2_NORMALIZATION);
}

TEST(SerializationTest, CanSerializeFullyConnected) {
  const std::string model_path = testing::TempDir() + "/fully_connected.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {1, 5}});
  TensorTf c({.type = Type::kFP32, .shape = {1}});
  TensorTf d =
      FullyConnected(a, b, c, litert::tensor::FusedActivation::kActNone, true);
  ASSERT_THAT(Save({d}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  const auto* fc_options = reinterpret_cast<const TfLiteFullyConnectedParams*>(
      node_and_reg->first.builtin_data);
  EXPECT_EQ(fc_options->activation, kActNone);
  EXPECT_EQ(fc_options->keep_num_dims, true);
  EXPECT_EQ(fc_options->weights_format,
            kTfLiteFullyConnectedWeightsFormatDefault);
  EXPECT_EQ(fc_options->asymmetric_quantize_inputs, false);
  EXPECT_EQ(fc_options->quantized_bias_type, kTfLiteFloat32);
}

TEST(SerializationTest, CanSerializeFullyConnectedWithoutBias) {
  const std::string model_path =
      testing::TempDir() + "/fully_connected_no_bias.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {1, 5}});
  TensorTf d =
      FullyConnected(a, b, litert::tensor::FusedActivation::kActRelu, true);
  ASSERT_THAT(Save({d}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  const auto* fc_options = reinterpret_cast<const TfLiteFullyConnectedParams*>(
      node_and_reg->first.builtin_data);
  EXPECT_EQ(fc_options->activation, kTfLiteActRelu);
  EXPECT_EQ(fc_options->keep_num_dims, true);
  EXPECT_EQ(fc_options->weights_format,
            kTfLiteFullyConnectedWeightsFormatDefault);
  EXPECT_EQ(fc_options->asymmetric_quantize_inputs, false);
  EXPECT_EQ(fc_options->quantized_bias_type, kTfLiteFloat32);
}

TEST(SerializationTest, CanSerializeConv2D) {
  const std::string model_path = testing::TempDir() + "/conv2d.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  TensorTf b({.type = Type::kFP32, .shape = {1, 3, 3, 1}});
  TensorTf c({.type = Type::kFP32, .shape = {1}});
  TensorTf d = Conv2D(a, b, c, 2, 2, kPaddingSame, 1, 1,
                      litert::tensor::FusedActivation::kActRelu);
  ASSERT_THAT(Save({d}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  const auto* conv_options = reinterpret_cast<const TfLiteConvParams*>(
      node_and_reg->first.builtin_data);
  EXPECT_EQ(conv_options->activation, kTfLiteActRelu);
  EXPECT_EQ(conv_options->padding, kTfLitePaddingSame);
  EXPECT_EQ(conv_options->stride_height, 2);
  EXPECT_EQ(conv_options->stride_width, 2);
  EXPECT_EQ(conv_options->dilation_height_factor, 1);
  EXPECT_EQ(conv_options->dilation_width_factor, 1);
}

TEST(SerializationTest, CanSerializeDepthwiseConv2D) {
  const std::string model_path =
      testing::TempDir() + "/depthwise_conv2d.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  TensorTf b({.type = Type::kFP32, .shape = {1, 3, 3, 1}});
  TensorTf c({.type = Type::kFP32, .shape = {1}});
  TensorTf d = DepthwiseConv2D(a, b, c, 2, 2, kPaddingSame, 1, 1, 1,
                               litert::tensor::FusedActivation::kActRelu);
  ASSERT_THAT(Save({d}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  const auto* conv_options = reinterpret_cast<const TfLiteDepthwiseConvParams*>(
      node_and_reg->first.builtin_data);
  EXPECT_EQ(conv_options->activation, kTfLiteActRelu);
  EXPECT_EQ(conv_options->padding, kTfLitePaddingSame);
  EXPECT_EQ(conv_options->stride_height, 2);
  EXPECT_EQ(conv_options->stride_width, 2);
  EXPECT_EQ(conv_options->dilation_height_factor, 1);
  EXPECT_EQ(conv_options->dilation_width_factor, 1);
  EXPECT_EQ(conv_options->depth_multiplier, 1);
}

TEST(SerializationTest, CanSerializePad) {
  const std::string model_path = testing::TempDir() + "/pad.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  const std::vector<int32_t> b_data = {1, 1, 1, 1, 0, 0, 0, 0};
  TensorTf b({.type = Type::kI32, .shape = {4, 2}, .buffer = b_data});
  TensorTf d = Pad(a, b);
  ASSERT_THAT(Save({d}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_PAD);
}

TEST(SerializationTest, CanSerializePadV2) {
  const std::string model_path = testing::TempDir() + "/pad_v2.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  TensorTf b(
      {.type = Type::kI32,
       .shape = {4, 2},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 1, 1, 1, 0, 0, 0, 0})});
  TensorTf c({.type = Type::kFP32, .shape = {}});
  TensorTf d = PadV2(a, b, c);
  ASSERT_THAT(Save({d}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_PADV2);
}

TEST(SerializationTest, CanSerializeBatchMatMul) {
  const std::string model_path = testing::TempDir() + "/batch_mat_mul.tflite";
  TensorTf x({.type = Type::kFP32, .shape = {2, 3, 4}});
  TensorTf y({.type = Type::kFP32, .shape = {2, 3, 5}});
  TensorTf z = BatchMatMul(x, y, true, false);
  ASSERT_THAT(Save({z}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  const auto* bmm_options = reinterpret_cast<const TfLiteBatchMatMulParams*>(
      node_and_reg->first.builtin_data);
  EXPECT_EQ(bmm_options->adj_x, true);
  EXPECT_EQ(bmm_options->adj_y, false);
}

TEST(SerializationTest, CanSerializeConcatenation) {
  const std::string model_path = testing::TempDir() + "/concatenation.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {1, 5}});
  TensorTf c =
      Concatenation({a, b}, 0, litert::tensor::FusedActivation::kActNone);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  const auto* concatenation_options =
      reinterpret_cast<const TfLiteConcatenationParams*>(
          node_and_reg->first.builtin_data);
  ASSERT_NE(concatenation_options, nullptr);
  EXPECT_EQ(concatenation_options->axis, 0);
  EXPECT_EQ(concatenation_options->activation, kTfLiteActNone);
}

TEST(SerializationTest, CanSerializePack) {
  const std::string model_path = testing::TempDir() + "/pack.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {1, 5}});
  TensorTf c = Pack({a, b}, 0);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_PACK);
  const auto* pack_options = reinterpret_cast<const TfLitePackParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(pack_options, nullptr);
  EXPECT_EQ(pack_options->values_count, 2);
  EXPECT_EQ(pack_options->axis, 0);
}

TEST(SerializationTest, CanSerializeUnpack) {
  const std::string model_path = testing::TempDir() + "/unpack.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  std::vector<TensorTf> c = Unpack(a, 2, 0);
  ASSERT_THAT(Save(c, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_UNPACK);
  const auto* unpack_options = reinterpret_cast<const TfLiteUnpackParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(unpack_options, nullptr);
  EXPECT_EQ(unpack_options->num, 2);
  EXPECT_EQ(unpack_options->axis, 0);
}

TEST(SerializationTest, CanSerializeSplit) {
  const std::string model_path = testing::TempDir() + "/split.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 4}});
  std::vector<TensorTf> c = Split(a, 1, 2);
  ASSERT_THAT(Save(c, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_SPLIT);
  const auto* split_options = reinterpret_cast<const TfLiteSplitParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(split_options, nullptr);
  EXPECT_EQ(split_options->num_splits, 2);
}

TEST(SerializationTest, CanSerializeTranspose) {
  const std::string model_path = testing::TempDir() + "/transpose.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {5, 5}});
  TensorTf b({.type = Type::kI32, .shape = {5}});
  TensorTf d = Transpose(a, b);
  ASSERT_THAT(Save({d}, model_path), IsOk());
}

TEST(SerializationTest, CanSerializeTransposeWithVector) {
  const std::string model_path =
      testing::TempDir() + "/transpose_with_vector.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {5, 5}});
  TensorTf d = Transpose(a, {1, 0});
  ASSERT_THAT(Save({d}, model_path), IsOk());
}

TEST(SerializationTest, RunAddHasCorrectResults) {
  TensorTf a({.type = Type::kFP32,
              .shape = {2, 2},
              .buffer = std::vector<float>({1, 2, 3, 4})});
  TensorTf b({.type = Type::kFP32,
              .shape = {2, 2},
              .buffer = std::vector<float>({5, 6, 7, 8})});
  TensorTf c = Add(a, b);
  ASSERT_THAT(tensor::Run({c}), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, c.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const float>(), ElementsAre(6, 8, 10, 12));
}

TEST(SerializationTest, CanSerializeGelu) {
  const std::string model_path = testing::TempDir() + "/gelu.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Gelu(a, true);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  const auto* gelu_options = reinterpret_cast<const TfLiteGeluParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(gelu_options, nullptr);
  EXPECT_EQ(gelu_options->approximate, true);
}

TEST(SerializationTest, CanSerializeTanh) {
  const std::string model_path = testing::TempDir() + "/tanh.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Tanh(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_TANH);
}

TEST(SerializationTest, CanSerializeSin) {
  const std::string model_path = testing::TempDir() + "/sin.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Sin(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_SIN);
}

TEST(SerializationTest, CanSerializeCos) {
  const std::string model_path = testing::TempDir() + "/cos.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Cos(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_COS);
}

TEST(SerializationTest, CanSerializeLogicalAnd) {
  const std::string model_path = testing::TempDir() + "/logical_and.tflite";
  TensorTf a({.type = Type::kBOOL, .shape = {2, 5}});
  TensorTf b({.type = Type::kBOOL, .shape = {2, 5}});
  TensorTf c = LogicalAnd(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_LOGICAL_AND);
}

TEST(SerializationTest, CanSerializeLogicalOr) {
  const std::string model_path = testing::TempDir() + "/logical_or.tflite";
  TensorTf a({.type = Type::kBOOL, .shape = {2, 5}});
  TensorTf b({.type = Type::kBOOL, .shape = {2, 5}});
  TensorTf c = LogicalOr(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_LOGICAL_OR);
}

TEST(SerializationTest, CanSerializeLogicalNot) {
  const std::string model_path = testing::TempDir() + "/logical_not.tflite";
  TensorTf a({.type = Type::kBOOL, .shape = {2, 5}});
  TensorTf c = LogicalNot(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_LOGICAL_NOT);
}

TEST(SerializationTest, CanSerializeBitwiseXor) {
  const std::string model_path = testing::TempDir() + "/bitwise_xor.tflite";
  TensorTf a({.type = Type::kI32, .shape = {2, 5}});
  TensorTf b({.type = Type::kI32, .shape = {2, 5}});
  TensorTf c = BitwiseXor(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_BITWISE_XOR);
}

TEST(SerializationTest, CanSerializeRightShift) {
  const std::string model_path = testing::TempDir() + "/right_shift.tflite";
  TensorTf a({.type = Type::kI32, .shape = {2, 5}});
  TensorTf b({.type = Type::kI32, .shape = {2, 5}});
  TensorTf c = RightShift(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_RIGHT_SHIFT);
}

TEST(SerializationTest, CanSerializeFloorDiv) {
  const std::string model_path = testing::TempDir() + "/floor_div.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = FloorDiv(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_FLOOR_DIV);
}

TEST(SerializationTest, CanSerializeFloorMod) {
  const std::string model_path = testing::TempDir() + "/floor_mod.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = FloorMod(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_FLOOR_MOD);
}

TEST(SerializationTest, CanSerializeMinimum) {
  const std::string model_path = testing::TempDir() + "/minimum.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Minimum(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_MINIMUM);
}

TEST(SerializationTest, CanSerializeMaximum) {
  const std::string model_path = testing::TempDir() + "/maximum.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Maximum(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_MAXIMUM);
}

TEST(SerializationTest, CanSerializeLess) {
  const std::string model_path = testing::TempDir() + "/less.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Less(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_LESS);
}

TEST(SerializationTest, CanSerializeGreater) {
  const std::string model_path = testing::TempDir() + "/greater.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Greater(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_GREATER);
}

TEST(SerializationTest, CanSerializeGreaterEqual) {
  const std::string model_path = testing::TempDir() + "/greater_equal.tflite";
  const std::vector<int32_t> a_data = {1, 2, 3, 4};
  const std::vector<int32_t> b_data = {8, 7, 6, 5};
  TensorTf a({.type = Type::kI32, .shape = {2, 2}, .buffer = a_data});
  TensorTf b({.type = Type::kI32, .shape = {2, 2}, .buffer = b_data});
  TensorTf c = GreaterEqual(a, b);
  c.SetName("c");
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_GREATER_EQUAL);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TfLiteTensor & c,
                                    GetOutputTensor("c", *interpreter));
    EXPECT_EQ(c.type, kTfLiteBool);
    EXPECT_THAT(c.dims, TfLiteArrayIs({2, 2}));
    absl::Span<const uint8_t> output_data(
        reinterpret_cast<uint8_t*>(c.data.data), 4);
    EXPECT_THAT(output_data, ElementsAre(0, 0, 0, 0));
  }
}

TEST(SerializationTest, CanSerializeSlice) {
  const std::string model_path = testing::TempDir() + "/slice.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {4, 4}});
  TensorTf b = Slice(a, {1, 1}, {2, 2});
  ASSERT_THAT(Save({b}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_SLICE);
}

TEST(SerializationTest, CanSerializeSelect) {
  const std::string model_path = testing::TempDir() + "/select.tflite";
  TensorTf condition({.type = Type::kBOOL, .shape = {2, 5}});
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Select(condition, a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_SELECT);
}

TEST(SerializationTest, CanSerializeSelectV2) {
  const std::string model_path = testing::TempDir() + "/select_v2.tflite";
  TensorTf condition({.type = Type::kBOOL, .shape = {2, 5}});
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = SelectV2(condition, a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_SELECT_V2);
}

TEST(SerializationTest, CanSerializeCast) {
  const std::string model_path = testing::TempDir() + "/cast.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b = Cast(a, Type::kI32);
  ASSERT_THAT(Save({b}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_CAST);
}

TEST(SerializationTest, CanSerializeReshape) {
  const std::string model_path = testing::TempDir() + "/reshape.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 5, 1}});
  TensorTf b = Reshape(a, {5});
  ASSERT_THAT(Save({b}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  const auto* reshape_options = reinterpret_cast<const TfLiteReshapeParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(reshape_options, nullptr);
  EXPECT_THAT(
      absl::MakeSpan(reshape_options->shape, reshape_options->num_dimensions),
      ElementsAre(5));
}

TEST(SerializationTest, CanSerializeExpandDims) {
  const std::string model_path = testing::TempDir() + "/expand_dims.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 5, 1}});
  TensorTf b = ExpandDims(a, 0);
  ASSERT_THAT(Save({b}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_EXPAND_DIMS);
}

TEST(SerializationTest, CanSerializeLogistic) {
  const std::string model_path = testing::TempDir() + "/logistic.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Logistic(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_LOGISTIC);
}

TEST(SerializationTest, CanSerializeDynamicUpdateSlice) {
  const std::string model_path =
      testing::TempDir() + "/dynamic_update_slice.tflite";
  TensorTf operand({.type = Type::kFP32, .shape = {10, 10}});
  TensorTf update({.type = Type::kFP32, .shape = {2, 2}});
  TensorTf start_indices({.type = Type::kI32, .shape = {2}});
  TensorTf result = DynamicUpdateSlice(operand, update, start_indices);
  ASSERT_THAT(Save({result}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE);
}

TEST(SerializationTest, CanSerializeDynamicUpdateSliceWithVector) {
  const std::string model_path =
      testing::TempDir() + "/dynamic_update_slice_with_vector.tflite";
  TensorTf operand({.type = Type::kFP32, .shape = {10, 10}});
  TensorTf update({.type = Type::kFP32, .shape = {2, 2}});
  TensorTf result = DynamicUpdateSlice(operand, update, {0, 0});
  ASSERT_THAT(Save({result}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE);
}

TEST(SerializationTest, CanSerializeEmbeddingLookup) {
  const std::string model_path =
      testing::TempDir() + "/embedding_lookup.tflite";
  TensorTf value({.type = Type::kFP32, .shape = {10, 4}});
  TensorTf lookup({.type = Type::kI32, .shape = {2}});
  TensorTf result = EmbeddingLookup(lookup, value);
  ASSERT_THAT(Save({result}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_EMBEDDING_LOOKUP);
}

TEST(SerializationTest, CanSerializeEmbeddingLookupWithVector) {
  const std::string model_path =
      testing::TempDir() + "/embedding_lookup_with_vector.tflite";
  TensorTf value({.type = Type::kFP32, .shape = {10, 4}});
  TensorTf result = EmbeddingLookup({1, 2}, value);
  ASSERT_THAT(Save({result}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_EMBEDDING_LOOKUP);
}

TEST(SerializationTest, CanSerializeCustom) {
  const std::string model_path = testing::TempDir() + "/custom.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  std::vector<TensorTf> outputs =
      Custom({a}, "MyCustomOp", {1, 2, 3}, {{2, 5}}, {Type::kFP32});
  ASSERT_THAT(Save(outputs, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteUnresolvedOps);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_CUSTOM);
  ASSERT_THAT(node_and_reg->second.custom_name, StrEq("MyCustomOp"));
  const auto* custom_options = node_and_reg->first.custom_initial_data;
  const size_t custom_size = node_and_reg->first.custom_initial_data_size;
  EXPECT_THAT(absl::MakeSpan((const uint8_t*)custom_options, custom_size),
              ElementsAre(1, 2, 3));
}

TEST(SerializationTest, CanSerializeTile) {
  const std::string model_path = testing::TempDir() + "/tile.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 3}});
  TensorTf multiples({.type = Type::kI32, .shape = {2}});
  TensorTf b = Tile(a, multiples);
  ASSERT_THAT(Save({b}, model_path), IsOk());
}

TEST(SerializationTest, CanSerializeTileWithVector) {
  const std::string model_path =
      testing::TempDir() + "/tile_with_vector.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 3}});
  TensorTf b = Tile(a, {2, 1});
  ASSERT_THAT(Save({b}, model_path), IsOk());
}

TEST(SerializationTest, CanSerializeSum) {
  const std::string model_path = testing::TempDir() + "/sum_op.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 3}});
  TensorTf b(
      {.type = Type::kI32, .shape = {1}, .buffer = std::vector<int32_t>({1})});
  TensorTf c = Sum(a, b, true);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  const auto* gelu_options = reinterpret_cast<const TfLiteReducerParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(gelu_options, nullptr);
  EXPECT_EQ(gelu_options->keep_dims, true);
}

TEST(SerializationTest, CanSerializeReduceMax) {
  const std::string model_path = testing::TempDir() + "/reduce_max.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 3}});
  TensorTf b(
      {.type = Type::kI32, .shape = {1}, .buffer = std::vector<int32_t>({1})});
  TensorTf c = ReduceMax(a, b, true);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_REDUCE_MAX);
  const auto* reducer_options = reinterpret_cast<const TfLiteReducerParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(reducer_options, nullptr);
  EXPECT_EQ(reducer_options->keep_dims, true);
}

TEST(SerializationTest, CanSerializeReduceMaxWithoutKeepDims) {
  const std::string model_path =
      testing::TempDir() + "/reduce_max_no_keep_dims.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 3}});
  TensorTf b(
      {.type = Type::kI32, .shape = {1}, .buffer = std::vector<int32_t>({1})});
  TensorTf c = ReduceMax(a, b, false);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_REDUCE_MAX);
  const auto* reducer_options = reinterpret_cast<const TfLiteReducerParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(reducer_options, nullptr);
  EXPECT_EQ(reducer_options->keep_dims, false);
}

TEST(SerializationTest, CanSerializeMean) {
  const std::string model_path = testing::TempDir() + "/mean_op.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 3}});
  TensorTf b(
      {.type = Type::kI32, .shape = {1}, .buffer = std::vector<int32_t>({1})});
  TensorTf c = Mean(a, b, true);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_MEAN);
  const auto* reducer_options = reinterpret_cast<const TfLiteReducerParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(reducer_options, nullptr);
  EXPECT_EQ(reducer_options->keep_dims, true);
}

TEST(SerializationTest, CanSerializeMeanWithoutKeepDims) {
  const std::string model_path =
      testing::TempDir() + "/mean_op_no_keep_dims.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 3}});
  TensorTf b(
      {.type = Type::kI32, .shape = {1}, .buffer = std::vector<int32_t>({1})});
  TensorTf c = Mean(a, b, false);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_MEAN);
  const auto* reducer_options = reinterpret_cast<const TfLiteReducerParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(reducer_options, nullptr);
  EXPECT_EQ(reducer_options->keep_dims, false);
}

TEST(SerializationTest, CanSerializeTopK) {
  const std::string model_path = testing::TempDir() + "/topk.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  std::vector<TensorTf> c = TopK(a, 2);
  ASSERT_THAT(Save(c, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_TOPK_V2);
}

TEST(SerializationTest, CanSerializeTopKWithDifferentShapes) {
  const std::string model_path =
      testing::TempDir() + "/topk_different_shapes.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {3, 10}});
  std::vector<TensorTf> c = TopK(a, 5);
  ASSERT_THAT(Save(c, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_TOPK_V2);
}

TEST(SerializationTest, CanSerializeTopKWithSingleOutput) {
  const std::string model_path =
      testing::TempDir() + "/topk_different_shapes.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {3, 10}});
  std::vector<TensorTf> c = TopK(a, 5);
  ASSERT_THAT(Save({c[0]}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_TOPK_V2);
}

TEST(SerializationTest, CanSerializeQuantize) {
  const std::string model_path = testing::TempDir() + "/quantize.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Quantize(a, Type::kI8, {0.5}, {128});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c.GetRaw()));
  ASSERT_EQ(c_info.type, Type::kI8);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_QUANTIZE);
}

TEST(SerializationTest, CanSerializeDequantize) {
  const std::string model_path = testing::TempDir() + "/dequantize.tflite";
  TensorTf a({.type = Type::kI8, .shape = {2, 5}});
  TensorTf c = Dequantize(a);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c.GetRaw()));
  ASSERT_EQ(c_info.type, Type::kFP32);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_DEQUANTIZE);
}

TEST(SerializationTest, CanSerializeCumsum) {
  const std::string model_path = testing::TempDir() + "/cumsum.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf axis({.type = Type::kI32, .shape = {}});
  TensorTf c = Cumsum(a, axis, /*exclusive=*/true, /*reverse=*/true);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_CUMSUM);
  const auto* cumsum_options = reinterpret_cast<const TfLiteCumsumParams*>(
      node_and_reg->first.builtin_data);
  EXPECT_EQ(cumsum_options->exclusive, true);
  EXPECT_EQ(cumsum_options->reverse, true);
}

TEST(SerializationTest, CanSerializeReverse) {
  const std::string model_path = testing::TempDir() + "/reverse.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf axes({.type = Type::kI32, .shape = {1}});
  TensorTf c = Reverse(a, axes);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_REVERSE_V2);
}

TEST(SerializationTest, CanSerializeSpaceToDepth) {
  const std::string model_path = testing::TempDir() + "/space_to_depth.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 4, 4, 1}});
  TensorTf c = SpaceToDepth(a, 2);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_SPACE_TO_DEPTH);
  const auto* options = reinterpret_cast<const TfLiteSpaceToDepthParams*>(
      node_and_reg->first.builtin_data);
  EXPECT_EQ(options->block_size, 2);
}

TEST(SerializationTest, CanSerializeDepthToSpace) {
  const std::string model_path = testing::TempDir() + "/depth_to_space.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {1, 2, 2, 4}});
  TensorTf c = DepthToSpace(a, 2);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_DEPTH_TO_SPACE);
  const auto* options = reinterpret_cast<const TfLiteDepthToSpaceParams*>(
      node_and_reg->first.builtin_data);
  EXPECT_EQ(options->block_size, 2);
}

TEST(SerializationTest, CanSerializeGather) {
  const std::string model_path = testing::TempDir() + "/gather.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kI32, .shape = {2}});
  TensorTf c = Gather(a, b, 0);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_GATHER);
  const auto* gather_options = reinterpret_cast<const TfLiteGatherParams*>(
      node_and_reg->first.builtin_data);
  EXPECT_EQ(gather_options->axis, 0);
}

TEST(SerializationTest, CanSerializeGatherWithBatchDims) {
  const std::string model_path =
      testing::TempDir() + "/gather_with_batch_dims.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 3, 4}});
  TensorTf b({.type = Type::kI32, .shape = {2, 5}});
  TensorTf c = Gather(a, b, 1, 1);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_GATHER);
  const auto* gather_options = reinterpret_cast<const TfLiteGatherParams*>(
      node_and_reg->first.builtin_data);
  EXPECT_EQ(gather_options->axis, 1);
  EXPECT_EQ(gather_options->batch_dims, 1);
}

TEST(SerializationTest, CanSerializeGatherNd) {
  const std::string model_path = testing::TempDir() + "/gather_nd.tflite";
  TensorTf input({.type = Type::kFP32, .shape = {3, 2, 2}});
  TensorTf indices({.type = Type::kI32, .shape = {2, 2}});
  TensorTf output = GatherNd(input, indices);

  ASSERT_THAT(Save({output}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_GATHER_ND);
}

TEST(SerializationTest, CanSerializeOneHot) {
  const std::string model_path = testing::TempDir() + "/one_hot.tflite";
  TensorTf indices({.type = Type::kI32, .shape = {4}});
  TensorTf depth(
      {.type = Type::kI32, .shape = {}, .buffer = std::vector<int32_t>{3}});
  TensorTf on_value(
      {.type = Type::kFP32, .shape = {}, .buffer = std::vector<float>{1.0f}});
  TensorTf off_value(
      {.type = Type::kFP32, .shape = {}, .buffer = std::vector<float>{0.0f}});
  TensorTf output = OneHot(indices, depth, on_value, off_value, -1);

  ASSERT_THAT(Save({output}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_ONE_HOT);
}

TEST(SerializationTest, CanSerializeEqual) {
  const std::string model_path = testing::TempDir() + "/equal.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = Equal(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_EQUAL);
}

TEST(SerializationTest, CanSerializeNotEqual) {
  const std::string model_path = testing::TempDir() + "/not_equal.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf b({.type = Type::kFP32, .shape = {2, 5}});
  TensorTf c = NotEqual(a, b);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_NOT_EQUAL);
}

TEST(SerializationTest, CanSerializeResizeBilinear) {
  const std::string model_path = testing::TempDir() + "/resize_bilinear.tflite";
  TensorTf input({.type = Type::kFP32, .shape = {1, 10, 10, 3}});
  TensorTf size({.type = Type::kI32,
                 .shape = {2},
                 .buffer = std::vector<int32_t>({20, 20})});
  TensorTf output = ResizeBilinear(input, size, /*align_corners=*/true,
                                   /*half_pixel_centers=*/false);
  ASSERT_THAT(Save({output}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_RESIZE_BILINEAR);
  const auto* options = reinterpret_cast<const TfLiteResizeBilinearParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(options->align_corners, true);
  EXPECT_EQ(options->half_pixel_centers, false);
}

TEST(SerializationTest, CanSerializeResizeNearestNeighbor) {
  const std::string model_path =
      testing::TempDir() + "/resize_nearest_neighbor.tflite";
  TensorTf input({.type = Type::kFP32, .shape = {1, 10, 10, 3}});
  TensorTf size({.type = Type::kI32,
                 .shape = {2},
                 .buffer = std::vector<int32_t>({20, 20})});
  TensorTf output = ResizeNearestNeighbor(input, size, /*align_corners=*/true,
                                          /*half_pixel_centers=*/false);
  ASSERT_THAT(Save({output}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR);
  const auto* options =
      reinterpret_cast<const TfLiteResizeNearestNeighborParams*>(
          node_and_reg->first.builtin_data);
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(options->align_corners, true);
  EXPECT_EQ(options->half_pixel_centers, false);
}

TEST(SerializationTest, CanSerializeLstm) {
  const std::string model_path = testing::TempDir() + "/lstm.tflite";
  TensorTf intermediate({.type = Type::kFP32, .shape = {1, 8}});
  TensorTf prev_state({.type = Type::kFP32, .shape = {1, 2}});
  std::vector<TensorTf> outputs = Lstm(intermediate, prev_state);
  ASSERT_THAT(Save({outputs[0], outputs[1]}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteUnresolvedOps);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_CUSTOM);
  ASSERT_THAT(node_and_reg->second.custom_name, StrEq("LSTM_BASIC"));
}

TEST(SerializationTest, CanSerializeTransposeConv) {
  const std::string model_path = testing::TempDir() + "/transpose_conv.tflite";
  TensorTf filter({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorTf input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorTf bias({.type = Type::kFP32, .shape = {1}});
  std::vector<int> output_shape_vec = {1, 4, 4, 1};

  TensorTf d =
      TransposeConv(filter, input, bias, output_shape_vec, kPaddingSame, 2, 2);
  ASSERT_THAT(Save({d}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 2);
  const auto* node_and_reg_0 = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg_0, nullptr);
  EXPECT_EQ(node_and_reg_0->second.builtin_code,
            tflite::BuiltinOperator_TRANSPOSE_CONV);

  const auto* options = reinterpret_cast<const TfLiteTransposeConvParams*>(
      node_and_reg_0->first.builtin_data);
  EXPECT_EQ(options->padding, kTfLitePaddingSame);
  EXPECT_EQ(options->stride_height, 2);
  EXPECT_EQ(options->stride_width, 2);

  const auto* node_and_reg_1 = interpreter->node_and_registration(1);
  ASSERT_NE(node_and_reg_1, nullptr);
  EXPECT_EQ(node_and_reg_1->second.builtin_code, tflite::BuiltinOperator_ADD);
}

TEST(SerializationTest, CanSerializeTransposeConv2D) {
  const std::string model_path =
      testing::TempDir() + "/transpose_conv2d.tflite";
  TensorTf filter({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorTf input({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorTf bias({.type = Type::kFP32, .shape = {1}});
  std::vector<int> output_shape_vec = {1, 4, 4, 1};

  TensorTf d = TransposeConv2D(filter, input, bias, output_shape_vec,
                               kPaddingSame, 2, 2);
  ASSERT_THAT(Save({d}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 2);
  const auto* node_and_reg_0 = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg_0, nullptr);
  EXPECT_EQ(node_and_reg_0->second.builtin_code,
            tflite::BuiltinOperator_TRANSPOSE_CONV);

  const auto* options = reinterpret_cast<const TfLiteTransposeConvParams*>(
      node_and_reg_0->first.builtin_data);
  EXPECT_EQ(options->padding, kTfLitePaddingSame);
  EXPECT_EQ(options->stride_height, 2);
  EXPECT_EQ(options->stride_width, 2);

  const auto* node_and_reg_1 = interpreter->node_and_registration(1);
  ASSERT_NE(node_and_reg_1, nullptr);
  EXPECT_EQ(node_and_reg_1->second.builtin_code, tflite::BuiltinOperator_ADD);
}

TEST(SerializationTest, CanSerializeArgMax) {
  const std::string model_path = testing::TempDir() + "/arg_max.tflite";
  TensorTf a({.type = Type::kFP32, .shape = {2, 3}});
  TensorTf b(
      {.type = Type::kI32, .shape = {1}, .buffer = std::vector<int32_t>({1})});
  TensorTf c = ArgMax(a, b, Type::kI64);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_ARG_MAX);
  const auto* argmax_options = reinterpret_cast<const TfLiteArgMaxParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(argmax_options, nullptr);
  EXPECT_EQ(argmax_options->output_type, kTfLiteInt64);
}

TEST(SerializationTest, MultipleSerializationsGiveTheSameFile) {
  // We are rebuilding the graph multiple times because what we want to ensure
  // is that the same code will produce the same data.
  auto build_model = []() -> absl::StatusOr<std::vector<char>> {
    std::vector<char> buffer;
    TensorTf a({.name = "aaa", .type = Type::kI32, .shape = {3, 3}});
    TensorTf b({.name = "bbb", .type = Type::kI32, .shape = {3, 3}});
    TensorTf c({.name = "ccc", .type = Type::kI32, .shape = {3, 3}});
    TensorTf d = Mul(a, b).SetName("ddd");
    TensorTf e = Add(c, d).SetName("eee");
    if (auto status = Save({e, d}, buffer); !status.ok()) {
      return status;
    }
    return buffer;
  };
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<char> buffer1, build_model());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<char> buffer2, build_model());
  EXPECT_THAT(buffer1, ElementsAreArray(buffer2));
}

TEST(SerializationTest, TensorsShareBufferAcrossSignatures) {
  const std::string model_path =
      testing::TempDir() + "/" +
      testing::UnitTest::GetInstance()->current_test_info()->name() + ".tflite";
  const char kSignature1[] = "signature1";
  const char kSignature2[] = "signature2";
  ModelFactory model_builder;

  // Create a shared buffer using OwningCpuBuffer
  auto buffer = OwningCpuBuffer::Allocate<Type::kFP32>(4);
  auto span = buffer->Span<float>();
  span[0] = 1.0f;
  span[1] = 2.0f;
  span[2] = 3.0f;
  span[3] = 4.0f;

  {
    TensorTf input1({.name = "input1", .type = Type::kFP32, .shape = {2, 2}});
    TensorTf a({.name = "a", .type = Type::kFP32, .shape = {2, 2}});
    ASSERT_THAT(graph::SetBuffer(a.GetRaw(), buffer), IsOk());

    TensorTf b = Add(input1, a).SetName("b");
    EXPECT_THAT(model_builder.AddSignature({b}, /*name=*/kSignature1), IsOk());
  }
  {
    TensorTf input2({.name = "input2", .type = Type::kFP32, .shape = {2, 2}});
    TensorTf c({.name = "c", .type = Type::kFP32, .shape = {2, 2}});
    ASSERT_THAT(graph::SetBuffer(c.GetRaw(), buffer), IsOk());

    TensorTf d = Add(input2, c).SetName("d");
    EXPECT_THAT(model_builder.AddSignature({d}, /*name=*/kSignature2), IsOk());
  }
  EXPECT_THAT(model_builder.Save(model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  int sg1_idx = interpreter->GetSubgraphIndexFromSignature(kSignature1);
  int sg2_idx = interpreter->GetSubgraphIndexFromSignature(kSignature2);

  tflite::Subgraph* sg1 = interpreter->subgraph(sg1_idx);
  tflite::Subgraph* sg2 = interpreter->subgraph(sg2_idx);

  ASSERT_NE(sg1, nullptr);
  ASSERT_NE(sg2, nullptr);

  ASSERT_GE(sg1->nodes_size(), 1);
  ASSERT_GE(sg2->nodes_size(), 1);

  const auto* node_reg1 = sg1->node_and_registration(0);
  const auto* node_reg2 = sg2->node_and_registration(0);

  ASSERT_NE(node_reg1, nullptr);
  ASSERT_NE(node_reg2, nullptr);

  const TfLiteNode& node1 = node_reg1->first;
  const TfLiteNode& node2 = node_reg2->first;

  ASSERT_GE(node1.inputs->size, 2);
  ASSERT_GE(node2.inputs->size, 2);

  int a_idx = node1.inputs->data[1];
  int c_idx = node2.inputs->data[1];

  const TfLiteTensor* t1 = sg1->tensor(a_idx);
  const TfLiteTensor* t2 = sg2->tensor(c_idx);

  ASSERT_NE(t1, nullptr);
  ASSERT_NE(t2, nullptr);

  // Verify that they share the same data pointer!
  EXPECT_EQ(t1->data.raw, t2->data.raw);
}

}  // namespace
}  // namespace litert::tensor
