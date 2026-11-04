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

#include "litert/python/litert_model/litert_model_lib.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/build_stamp.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "tflite/schema/schema_generated.h"

namespace litert::litert_model {
namespace {

using ::litert::internal::GetTflOpCodes;
using ::litert::internal::GetTflOptions;
using ::litert::internal::LoadModelFromBuffer;
using ::litert::internal::LoadModelFromFile;
using ::litert::internal::SerializeModel;
using ::litert::internal::SetTflOpCodeInd;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::litert::IsError;

// Returns the value of `result`; aborts on error.
template <class T>
T Unwrap(Expected<T> result) {
  ABSL_CHECK(result) << result.Error().Message();
  return *result;
}

LiteRtTensorT& NewTensor(LiteRtSubgraphT& subgraph, absl::string_view name,
                         std::vector<int32_t> dims = {2}) {
  LiteRtTensorT& tensor = subgraph.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, dims));
  tensor.SetName(std::string(name));
  return tensor;
}

// main(x, y) = relu(x + y), signature "serving".
struct TestGraph {
  std::unique_ptr<LiteRtModelT> model = std::make_unique<LiteRtModelT>();
  LiteRtSubgraphT* subgraph = nullptr;
  LiteRtTensorT* x = nullptr;
  LiteRtTensorT* y = nullptr;
  LiteRtTensorT* sum = nullptr;
  LiteRtTensorT* out = nullptr;
  LiteRtOpT* add = nullptr;
  LiteRtOpT* relu = nullptr;
};

TestGraph BuildTestGraph() {
  TestGraph graph;
  graph.subgraph = &graph.model->EmplaceSubgraph();
  graph.x = &NewTensor(*graph.subgraph, "x");
  graph.y = &NewTensor(*graph.subgraph, "y");
  graph.sum = &NewTensor(*graph.subgraph, "sum");
  graph.out = &NewTensor(*graph.subgraph, "out");
  graph.add = Unwrap(AddOp(*graph.model, *graph.subgraph, kLiteRtOpCodeTflAdd,
                           {graph.x, graph.y}, {graph.sum}, {}));
  graph.relu = Unwrap(AddOp(*graph.model, *graph.subgraph, kLiteRtOpCodeTflRelu,
                            {graph.sum}, {graph.out}, {}));
  const std::vector<NamedTensor> inputs = {{"x", graph.x}, {"y", graph.y}};
  const std::vector<NamedTensor> outputs = {{"out", graph.out}};
  EXPECT_TRUE(
      AddSignature(*graph.model, "serving", *graph.subgraph, inputs, outputs));
  return graph;
}

TEST(ElementTypeTest, BitWidthAndDenseByteSize) {
  EXPECT_EQ(ElementTypeBitWidth(kLiteRtElementTypeInt4), 4);
  EXPECT_EQ(ElementTypeBitWidth(kLiteRtElementTypeBool), 8);
  EXPECT_EQ(ElementTypeBitWidth(kLiteRtElementTypeComplex128), 128);
  EXPECT_EQ(ElementTypeBitWidth(kLiteRtElementTypeTfString), 0);
  EXPECT_EQ(Unwrap(DenseByteSize(kLiteRtElementTypeFloat32, {2, 3})), 24);
  EXPECT_EQ(Unwrap(DenseByteSize(kLiteRtElementTypeInt4, {3})), 2);
  EXPECT_EQ(Unwrap(DenseByteSize(kLiteRtElementTypeInt2, {5})), 2);
  EXPECT_EQ(Unwrap(DenseByteSize(kLiteRtElementTypeFloat32, {})), 4);
  EXPECT_EQ(Unwrap(DenseByteSize(kLiteRtElementTypeFloat32, {-1, 3})),
            std::nullopt);
  EXPECT_EQ(Unwrap(DenseByteSize(kLiteRtElementTypeTfString, {3})),
            std::nullopt);
}

TEST(ElementTypeTest, DenseByteSizeOverflow) {
  constexpr int32_t kMaxDim = std::numeric_limits<int32_t>::max();
  EXPECT_THAT(
      DenseByteSize(kLiteRtElementTypeFloat32, {kMaxDim, kMaxDim, kMaxDim}),
      IsError(kLiteRtStatusErrorInvalidArgument));
  // 2^62 elements fit in a 64-bit size_t but 2^62 * 32 bits do not.
  EXPECT_THAT(DenseByteSize(kLiteRtElementTypeFloat32, {1 << 30, 1 << 30, 4}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_EQ(Unwrap(DenseByteSize(kLiteRtElementTypeFloat32,
                                 {kMaxDim, kMaxDim, kMaxDim, 0})),
            0);
  // A dynamic or unsized type is not an overflow, however large the shape.
  EXPECT_EQ(
      Unwrap(DenseByteSize(kLiteRtElementTypeFloat32, {kMaxDim, kMaxDim, -1})),
      std::nullopt);
  EXPECT_EQ(Unwrap(DenseByteSize(kLiteRtElementTypeTfString,
                                 {kMaxDim, kMaxDim, kMaxDim})),
            std::nullopt);
}

TEST(OpCodeTableTest, ReusesAndAppendsEntries) {
  LiteRtModelT model;
  EXPECT_EQ(Unwrap(GetOrAddTflOpCodeIndex(model, kLiteRtOpCodeTflAdd, "")), 0);
  EXPECT_EQ(Unwrap(GetOrAddTflOpCodeIndex(model, kLiteRtOpCodeTflAdd, "")), 0);
  EXPECT_EQ(Unwrap(GetOrAddTflOpCodeIndex(model, kLiteRtOpCodeTflCustom, "a")),
            1);
  EXPECT_EQ(Unwrap(GetOrAddTflOpCodeIndex(model, kLiteRtOpCodeTflCustom, "b")),
            2);
  EXPECT_EQ(Unwrap(GetOrAddTflOpCodeIndex(model, kLiteRtOpCodeTflCustom, "a")),
            1);
  EXPECT_EQ(
      Unwrap(GetOrAddTflOpCodeIndex(model, kLiteRtOpCodeShloComposite, "")), 3);
  const auto& codes = GetTflOpCodes(model);
  ASSERT_EQ(codes.size(), 4);
  EXPECT_EQ(codes[2]->custom_code, "b");
  EXPECT_EQ(codes[3]->builtin_code,
            tflite::BuiltinOperator_STABLEHLO_COMPOSITE);
  EXPECT_EQ(codes[3]->deprecated_builtin_code,
            tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
  EXPECT_EQ(codes[0]->version, 1);
}

TEST(AddOpTest, BuildsValidGraphThatSerializes) {
  TestGraph graph = BuildTestGraph();
  LITERT_EXPECT_OK(ValidateModel(*graph.model));
  EXPECT_THAT(graph.subgraph->Inputs(), ElementsAre(graph.x, graph.y));
  EXPECT_THAT(graph.subgraph->Outputs(), ElementsAre(graph.out));
  EXPECT_EQ(graph.sum->DefiningOp(), graph.relu->Inputs()[0]->DefiningOp());

  LITERT_ASSERT_OK_AND_ASSIGN(auto serialized,
                              SerializeModel(std::move(*graph.model)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto reloaded,
                              LoadModelFromBuffer(std::move(serialized)));
  LITERT_EXPECT_OK(ValidateModel(*reloaded));
  ASSERT_EQ(reloaded->Subgraph(0).Ops().size(), 2);
  EXPECT_EQ(reloaded->Subgraph(0).Op(0).OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(reloaded->Subgraph(0).Op(1).OpCode(), kLiteRtOpCodeTflRelu);
  LITERT_ASSERT_OK_AND_ASSIGN(auto signature,
                              reloaded->FindSignature("serving"));
  EXPECT_THAT(signature.get().InputNames(), ElementsAre("x", "y"));
}

TEST(AddOpTest, RejectsInvalidEdits) {
  TestGraph graph = BuildTestGraph();
  LiteRtTensorT& fresh = NewTensor(*graph.subgraph, "fresh");
  EXPECT_THAT(AddOp(*graph.model, *graph.subgraph, kLiteRtOpCodeTflRelu,
                    {graph.out}, {&fresh}, {.index = 0}),
              IsError(kLiteRtStatusErrorInvalidGraphInvariant));
  EXPECT_THAT(AddOp(*graph.model, *graph.subgraph, kLiteRtOpCodeTflRelu,
                    {graph.x}, {graph.sum}, {}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(AddOp(*graph.model, *graph.subgraph, kLiteRtOpCodeTflRelu,
                    {graph.x}, {&fresh}, {.index = 7}),
              IsError(kLiteRtStatusErrorIndexOOB));
  EXPECT_THAT(AddOp(*graph.model, *graph.subgraph, kLiteRtOpCodeTflCustom,
                    {graph.x}, {&fresh}, {}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_EQ(graph.subgraph->Ops().size(), 2);
  LITERT_EXPECT_OK(ValidateModel(*graph.model));
}

TEST(AddOpTest, InsertsAtPosition) {
  TestGraph graph = BuildTestGraph();
  LiteRtTensorT& negated = NewTensor(*graph.subgraph, "negated");
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtOpT * neg, AddOp(*graph.model, *graph.subgraph, kLiteRtOpCodeTflNeg,
                             {graph.y}, {&negated}, {.index = 0}));
  LITERT_ASSERT_OK(SetOpInput(*graph.subgraph, *graph.add, 1, negated));
  EXPECT_EQ(Unwrap(FindOpIndex(*graph.subgraph, *neg)), 0);
  EXPECT_EQ(Unwrap(FindOpIndex(*graph.subgraph, *graph.add)), 1);
  EXPECT_THAT(graph.y->Users(), ElementsAre(neg));
  EXPECT_THAT(negated.Users(), ElementsAre(graph.add));
  LITERT_EXPECT_OK(ValidateModel(*graph.model));
  // The consumer must come after the producer.
  EXPECT_THAT(SetOpInput(*graph.subgraph, *graph.add, 0, *graph.out),
              IsError(kLiteRtStatusErrorInvalidGraphInvariant));
}

TEST(BuiltinOptionsTest, PackUnpackRoundTrip) {
  TestGraph graph = BuildTestGraph();
  EXPECT_EQ(GetBuiltinOptionsTypes(*graph.add), std::make_pair(0, 0));

  tflite::OperatorT op_t;
  auto add_options = std::make_unique<tflite::AddOptionsT>();
  add_options->fused_activation_function = tflite::ActivationFunctionType_RELU;
  op_t.builtin_options.Set(std::move(*add_options));
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(tflite::Operator::Pack(builder, &op_t));
  LITERT_ASSERT_OK(UnpackBuiltinOptions(
      BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()),
      *graph.add));
  EXPECT_EQ(GetBuiltinOptionsTypes(*graph.add),
            std::make_pair(
                static_cast<int32_t>(tflite::BuiltinOptions_AddOptions), 0));

  for (int i = 0; i < 3; ++i) {
    OwningBufferRef<uint8_t> packed = PackBuiltinOptions(*graph.add);
    const auto* op = flatbuffers::GetRoot<tflite::Operator>(packed.Data());
    ASSERT_NE(op->builtin_options_as_AddOptions(), nullptr);
    EXPECT_EQ(op->builtin_options_as_AddOptions()->fused_activation_function(),
              tflite::ActivationFunctionType_RELU);
    EXPECT_EQ(op->builtin_options_2_type(), tflite::BuiltinOptions2_NONE);
  }
  // Packing leaves the op unchanged.
  ASSERT_NE(GetTflOptions(*graph.add).AsAddOptions(), nullptr);

  const uint8_t garbage[] = {1, 2, 3};
  EXPECT_THAT(UnpackBuiltinOptions(BufferRef<uint8_t>(garbage, 3), *graph.add),
              IsError(kLiteRtStatusErrorInvalidFlatbuffer));
  EXPECT_THAT(UnpackBuiltinOptions(BufferRef<uint8_t>(garbage, 0), *graph.add),
              IsError(kLiteRtStatusErrorInvalidFlatbuffer));
  // Failed unpacks leave the options unchanged.
  EXPECT_NE(GetTflOptions(*graph.add).AsAddOptions(), nullptr);
}

TEST(EliminateDeadCodeTest, RemovesDeadOpsToFixpoint) {
  TestGraph graph = BuildTestGraph();
  // A dead chain y -> neg -> neg2 and an unused constant.
  LiteRtTensorT& t1 = NewTensor(*graph.subgraph, "t1");
  LiteRtTensorT& t2 = NewTensor(*graph.subgraph, "t2");
  ASSERT_TRUE(AddOp(*graph.model, *graph.subgraph, kLiteRtOpCodeTflNeg,
                    {graph.y}, {&t1}, {}));
  ASSERT_TRUE(AddOp(*graph.model, *graph.subgraph, kLiteRtOpCodeTflNeg, {&t1},
                    {&t2}, {}));
  LiteRtTensorT& cst = NewTensor(*graph.subgraph, "cst");
  const float data[2] = {1.0f, 2.0f};
  SetWeightsFromOwnedBuffer(
      cst.Weights(), OwningBufferRef<uint8_t>(
                         reinterpret_cast<const uint8_t*>(data), sizeof(data)));
  EXPECT_TRUE(EliminateDeadCode(*graph.model, *graph.subgraph));
  EXPECT_THAT(graph.subgraph->Ops(), ElementsAre(graph.add, graph.relu));
  EXPECT_EQ(graph.subgraph->Tensors().size(), 4);
  EXPECT_FALSE(EliminateDeadCode(*graph.model, *graph.subgraph));
  LITERT_EXPECT_OK(ValidateModel(*graph.model));
}

TEST(ReplaceAllUsesTest, RemapsOutputsAndSignatures) {
  TestGraph graph = BuildTestGraph();
  LiteRtTensorT& negated = NewTensor(*graph.subgraph, "negated");
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtOpT * neg, AddOp(*graph.model, *graph.subgraph, kLiteRtOpCodeTflNeg,
                             {graph.out}, {&negated}, {}));
  LITERT_ASSERT_OK(ReplaceAllUses(*graph.model, *graph.subgraph, *graph.out,
                                  negated, {neg}));
  EXPECT_THAT(graph.subgraph->Outputs(), ElementsAre(&negated));
  LITERT_ASSERT_OK_AND_ASSIGN(auto signature,
                              graph.model->FindSignature("serving"));
  EXPECT_EQ(signature.get().GetOutputTensor(0), &negated);
  EXPECT_THAT(signature.get().OutputNames(), ElementsAre("out"));
  LITERT_EXPECT_OK(ValidateModel(*graph.model));
}

TEST(SignatureTest, SyncsNamesWithSubgraphIo) {
  TestGraph graph = BuildTestGraph();
  LiteRtTensorT& extra = NewTensor(*graph.subgraph, "");
  LITERT_ASSERT_OK(
      SetSubgraphInputs(*graph.model, *graph.subgraph, {graph.y, graph.x}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto signature,
                              graph.model->FindSignature("serving"));
  EXPECT_THAT(signature.get().InputNames(), ElementsAre("y", "x"));
  LITERT_ASSERT_OK(SetSubgraphInputs(*graph.model, *graph.subgraph,
                                     {graph.y, graph.x, &extra}));
  EXPECT_THAT(signature.get().InputNames(), ElementsAre("y", "x", "input_2"));
  EXPECT_THAT(SetSubgraphInputs(*graph.model, *graph.subgraph, {graph.sum}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  const std::vector<NamedTensor> outputs = {{"o", graph.out}};
  EXPECT_THAT(
      AddSignature(*graph.model, "serving", *graph.subgraph, {}, outputs),
      IsError(kLiteRtStatusErrorAlreadyExists));
  LITERT_ASSERT_OK_AND_ASSIGN(const LiteRtSignatureT* removed,
                              RemoveSignature(*graph.model, "serving"));
  EXPECT_NE(removed, nullptr);
  EXPECT_TRUE(graph.model->Signatures().empty());
  EXPECT_THAT(RemoveSignature(*graph.model, "serving"),
              IsError(kLiteRtStatusErrorNotFound));
}

TEST(ValidateModelTest, ReportsViolations) {
  TestGraph graph = BuildTestGraph();
  LITERT_ASSERT_OK(RemoveOp(*graph.model, *graph.subgraph, *graph.add));
  auto result = ValidateModel(*graph.model);
  ASSERT_THAT(result, IsError(kLiteRtStatusErrorInvalidGraphInvariant));
  EXPECT_THAT(result.Error().Message(), HasSubstr("neither constant"));

  TestGraph other = BuildTestGraph();
  SetTflOpCodeInd(*other.relu, 0);  // Points at the ADD entry.
  auto mismatch = ValidateModel(*other.model);
  ASSERT_FALSE(mismatch);
  EXPECT_THAT(mismatch.Error().Message(), HasSubstr("operator code entry"));
}

TEST(ValidateModelTest, SerializableLevelAcceptsLoaderModels) {
  TestGraph graph = BuildTestGraph();
  LITERT_ASSERT_OK(RemoveOp(*graph.model, *graph.subgraph, *graph.add));
  EXPECT_FALSE(ValidateModel(*graph.model));
  LITERT_EXPECT_OK(ValidateModel(*graph.model, ValidationLevel::kSerializable));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, LoadModelFromFile(
                      testing::GetTestFilePath("blockwise_quantized.tflite")));
  EXPECT_FALSE(ValidateModel(*model));
  LITERT_EXPECT_OK(ValidateModel(*model, ValidationLevel::kSerializable));
}

TEST(DispatchTest, OutlineRenamesRepeatedPlaceholderOutputs) {
  // A model without signature returning `out` twice: the loader's placeholder
  // signature repeats the output name.
  TestGraph graph = BuildTestGraph();
  LITERT_ASSERT_OK(RemoveSignature(*graph.model, "serving"));
  LITERT_ASSERT_OK(SetSubgraphOutputs(*graph.model, *graph.subgraph,
                                      {graph.out, graph.out}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto serialized,
                              SerializeModel(std::move(*graph.model)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                              LoadModelFromBuffer(std::move(serialized)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto placeholder,
                              model->FindSignature(kDefaultSignatureKey));
  EXPECT_THAT(placeholder.get().OutputNames(), ElementsAre("out", "out"));
  LITERT_EXPECT_OK(ValidateModel(*model));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtOpT * dispatch,
      OutlineSubgraphToDispatchOp(*model, model->Subgraph(0)));
  EXPECT_EQ(dispatch->NumOutputs(), 2);
  EXPECT_NE(dispatch->Outputs()[0], dispatch->Outputs()[1]);
  EXPECT_THAT(placeholder.get().OutputNames(), ElementsAre("out", "out_1"));
  LITERT_EXPECT_OK(ValidateModel(*model));
  LITERT_ASSERT_OK_AND_ASSIGN(auto reserialized,
                              SerializeModel(std::move(*model)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto reloaded,
                              LoadModelFromBuffer(std::move(reserialized)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto signature,
                              reloaded->FindSignature(kDefaultSignatureKey));
  EXPECT_THAT(signature.get().OutputNames(), ElementsAre("out", "out_1"));
  EXPECT_EQ(reloaded->Subgraph(0).Op(0).NumOutputs(), 2);
}

TEST(RemoveTensorTest, OnlyRemovesUnusedTensors) {
  TestGraph graph = BuildTestGraph();
  LiteRtTensorT& spare = NewTensor(*graph.subgraph, "spare");
  EXPECT_THAT(RemoveTensor(*graph.subgraph, *graph.sum),
              IsError(kLiteRtStatusErrorInvalidArgument));
  LITERT_ASSERT_OK(RemoveTensor(*graph.subgraph, spare));
  EXPECT_EQ(graph.subgraph->Tensors().size(), 4);
}

TEST(RemoveTensorTest, KeepsBlockWiseQuantizationTensors) {
  TestGraph graph = BuildTestGraph();
  LiteRtTensorT& scales = NewTensor(*graph.subgraph, "scales");
  LiteRtTensorT& zero_points = NewTensor(*graph.subgraph, "zero_points");
  graph.sum->SetQarams(
      MakeBlockWiseQuantization(&scales, &zero_points, /*block_size=*/2));
  EXPECT_THAT(RemoveTensor(*graph.subgraph, scales),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(RemoveTensor(*graph.subgraph, zero_points),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_EQ(graph.subgraph->Tensors().size(), 6);
  graph.sum->SetQarams(MakeEmptyQuantization());
  LITERT_EXPECT_OK(RemoveTensor(*graph.subgraph, scales));
}

TEST(DispatchTest, OutlineAttachSerializeReload) {
  TestGraph graph = BuildTestGraph();
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtOpT * dispatch,
      OutlineSubgraphToDispatchOp(*graph.model, *graph.subgraph));
  EXPECT_TRUE(IsDispatchOp(*dispatch));
  EXPECT_THAT(graph.subgraph->Ops(), ElementsAre(dispatch));
  EXPECT_THAT(dispatch->Inputs(), ElementsAre(graph.x, graph.y));
  EXPECT_THAT(dispatch->Outputs(), ElementsAre(graph.out));
  EXPECT_EQ(graph.subgraph->Tensors().size(), 3);
  EXPECT_THAT(ReadDispatchOpOptions(*dispatch),
              IsError(kLiteRtStatusErrorNotFound));

  const std::string bytecode = "compiled-bytecode";  // 17 bytes.
  const auto buffer_id = graph.model->Buffers()->RegisterOwnedBuffer(
      OwningBufferRef<uint8_t>(bytecode));
  EXPECT_THAT(AttachAsset(*graph.model, *dispatch, 9999, "serving"),
              IsError(kLiteRtStatusErrorInvalidArgument));
  LITERT_ASSERT_OK(AttachAsset(*graph.model, *dispatch, buffer_id, "serving"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto stamp, internal::MakeBuildStamp("TestVendor", "TestSocModel"));
  ASSERT_EQ(graph.model->PushMetadata(internal::kLiteRtBuildStampKey,
                                      std::move(stamp)),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK(ValidateModel(*graph.model));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto serialized,
      SerializeModel(std::move(*graph.model), /*bytecode_alignment=*/8));
  const std::string serialized_copy(
      reinterpret_cast<const char*>(serialized.Data()), serialized.Size());
  LITERT_ASSERT_OK_AND_ASSIGN(auto reloaded,
                              LoadModelFromBuffer(std::move(serialized)));
  LITERT_EXPECT_OK(ValidateModel(*reloaded));
  const LiteRtOpT& op = reloaded->Subgraph(0).Op(0);
  EXPECT_TRUE(IsDispatchOp(op));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, ReadDispatchOpOptions(op));
  EXPECT_EQ(options.name, "serving");
  EXPECT_EQ(options.bytecode_size, bytecode.size());
  EXPECT_EQ(options.bytecode_offset % 8, 0);
  EXPECT_EQ(
      serialized_copy.substr(options.bytecode_offset, options.bytecode_size),
      bytecode);
  const auto build_stamp = GetBuildStamp(*reloaded);
  ASSERT_TRUE(build_stamp.has_value());
  EXPECT_EQ(build_stamp->soc_manufacturer, "TestVendor");
  EXPECT_EQ(build_stamp->soc_model, "TestSocModel");
}

TEST(DispatchTest, OutlineRejectsSignatureOutOfSyncWithSubgraph) {
  TestGraph graph = BuildTestGraph();
  LITERT_ASSERT_OK(SetSubgraphOutputs(*graph.model, *graph.subgraph,
                                      {graph.out, graph.sum}));
  // A signature listing the subgraph outputs in another order.
  graph.model->EmplaceSignature(
      graph.subgraph, std::vector<std::string>{"x", "y"},
      std::vector<LiteRtTensor>{graph.x, graph.y},
      std::vector<std::string>{"sum", "out"},
      std::vector<LiteRtTensor>{graph.sum, graph.out}, "swapped");
  EXPECT_THAT(OutlineSubgraphToDispatchOp(*graph.model, *graph.subgraph),
              IsError(kLiteRtStatusErrorInvalidGraphInvariant));
  // Nothing was changed.
  EXPECT_THAT(graph.subgraph->Ops(), ElementsAre(graph.add, graph.relu));
  EXPECT_THAT(graph.subgraph->Outputs(), ElementsAre(graph.out, graph.sum));
  EXPECT_EQ(graph.subgraph->Tensors().size(), 4);
  EXPECT_EQ(graph.out->DefiningOp(), graph.relu);
}

TEST(DispatchTest, RejectsMalformedDispatchOptions) {
  TestGraph graph = BuildTestGraph();
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtOpT * dispatch,
      OutlineSubgraphToDispatchOp(*graph.model, *graph.subgraph));
  const uint8_t garbage[] = {0xff, 0x01, 0x02, 0x03};
  dispatch->SetCustomOptions(garbage, sizeof(garbage));
  EXPECT_THAT(ReadDispatchOpOptions(*dispatch),
              IsError(kLiteRtStatusErrorInvalidFlatbuffer));
}

TEST(DispatchTest, OutlineGivesForwardedAndRepeatedOutputsFreshTensors) {
  TestGraph graph = BuildTestGraph();
  LiteRtTensorT& cst = NewTensor(*graph.subgraph, "cst");
  const float data[2] = {1.0f, 2.0f};
  SetWeightsFromOwnedBuffer(
      cst.Weights(), OwningBufferRef<uint8_t>(
                         reinterpret_cast<const uint8_t*>(data), sizeof(data)));
  cst.SetQarams(MakePerTensorQuantization(0.5f, 3));
  // `x` is both forwarded to an output and used by ADD; `out` is listed twice.
  LITERT_ASSERT_OK(SetSubgraphOutputs(*graph.model, *graph.subgraph,
                                      {graph.out, graph.x, &cst, graph.out}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto signature,
                              graph.model->FindSignature("serving"));
  EXPECT_THAT(signature.get().OutputNames(),
              ElementsAre("out", "x", "cst", "out_1"));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtOpT * dispatch,
      OutlineSubgraphToDispatchOp(*graph.model, *graph.subgraph));
  const auto& outputs = graph.subgraph->Outputs();
  ASSERT_EQ(outputs.size(), 4);
  EXPECT_THAT(dispatch->Outputs(),
              ElementsAre(outputs[0], outputs[1], outputs[2], outputs[3]));
  EXPECT_THAT(dispatch->Inputs(), ElementsAre(graph.x, graph.y));
  EXPECT_EQ(outputs[0], graph.out);
  EXPECT_NE(outputs[1], graph.x);
  EXPECT_EQ(outputs[1]->Name(), "x");
  EXPECT_EQ(outputs[2]->Name(), "cst");
  EXPECT_EQ(outputs[2]->Weights().Buffer().Size(), 0);
  EXPECT_EQ(outputs[2]->Qparams().first, kLiteRtQuantizationPerTensor);
  EXPECT_FLOAT_EQ(outputs[2]->Qparams().second.per_tensor.scale, 0.5f);
  EXPECT_NE(outputs[3], graph.out);
  EXPECT_EQ(outputs[3]->Name(), "out");
  for (LiteRtTensor output : outputs) {
    EXPECT_EQ(output->DefiningOp(), dispatch);
  }
  // x, y and the four dispatch outputs; the unused constant was removed.
  EXPECT_EQ(graph.subgraph->Tensors().size(), 6);
  EXPECT_THAT(graph.subgraph->Inputs(), ElementsAre(graph.x, graph.y));
  for (size_t i = 0; i < outputs.size(); ++i) {
    EXPECT_EQ(signature.get().GetOutputTensor(i), outputs[i]);
  }
  EXPECT_THAT(signature.get().OutputNames(),
              ElementsAre("out", "x", "cst", "out_1"));
  LITERT_ASSERT_OK(ValidateModel(*graph.model));

  LITERT_ASSERT_OK_AND_ASSIGN(auto serialized,
                              SerializeModel(std::move(*graph.model)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto reloaded,
                              LoadModelFromBuffer(std::move(serialized)));
  LITERT_EXPECT_OK(ValidateModel(*reloaded));
  const LiteRtSubgraphT& subgraph = reloaded->Subgraph(0);
  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Op(0).NumOutputs(), 4);
  EXPECT_EQ(subgraph.Op(0).NumInputs(), 2);
  LITERT_ASSERT_OK_AND_ASSIGN(auto reloaded_signature,
                              reloaded->FindSignature("serving"));
  EXPECT_THAT(reloaded_signature.get().OutputNames(),
              ElementsAre("out", "x", "cst", "out_1"));
}

TEST(LoadedModelTest, TestdataModelsValidate) {
  for (const char* name :
       {"simple_model.tflite", "simple_conv_2d_op.tflite",
        "simple_shlo_composite.tflite", "multi_subgraph.tflite"}) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto model, LoadModelFromFile(testing::GetTestFilePath(name)));
    LITERT_EXPECT_OK(ValidateModel(*model)) << name;
  }
}

TEST(PruneUnreachableSubgraphsTest, PrunesAndRemapsComposites) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                              LoadModelFromFile(testing::GetTestFilePath(
                                  "simple_shlo_composite.tflite")));
  ASSERT_EQ(model->NumSubgraphs(), 2);
  EXPECT_EQ(Unwrap(PruneUnreachableSubgraphs(*model)), 0);
  EXPECT_EQ(model->NumSubgraphs(), 2);

  ASSERT_TRUE(OutlineSubgraphToDispatchOp(*model, model->Subgraph(0)));
  EXPECT_EQ(Unwrap(PruneUnreachableSubgraphs(*model)), 1);
  EXPECT_EQ(model->NumSubgraphs(), 1);
  LITERT_EXPECT_OK(ValidateModel(*model));
}

TEST(PruneUnreachableSubgraphsTest, SignaturelessAndUnsupportedControlFlow) {
  LiteRtModelT empty_model;
  EXPECT_EQ(Unwrap(PruneUnreachableSubgraphs(empty_model)), 0);

  LiteRtModelT signatureless;
  LiteRtSubgraphT& main = signatureless.EmplaceSubgraph();
  signatureless.EmplaceSubgraph();
  EXPECT_EQ(Unwrap(PruneUnreachableSubgraphs(signatureless)), 1);
  EXPECT_THAT(signatureless.Subgraphs(), ElementsAre(&main));
  EXPECT_TRUE(signatureless.Signatures().empty());

  LiteRtModelT while_model;
  LiteRtSubgraphT& entry = while_model.EmplaceSubgraph();
  while_model.EmplaceSubgraph();
  LiteRtTensorT& x = NewTensor(entry, "x");
  LiteRtTensorT& y = NewTensor(entry, "y");
  ASSERT_TRUE(AddOp(while_model, entry, kLiteRtOpCodeTflWhile, {&x}, {&y}, {}));
  EXPECT_EQ(Unwrap(PruneUnreachableSubgraphs(while_model)), 0);
  EXPECT_EQ(while_model.NumSubgraphs(), 2);
  EXPECT_TRUE(while_model.Signatures().empty());
}

}  // namespace
}  // namespace litert::litert_model
