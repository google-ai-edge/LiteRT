// Copyright 2024 Google LLC.
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

#include "litert/core/model/model.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/build_stamp.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/matchers.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAreArray;

//
// Model
//

TEST(ModelTest, GetMetadata) {
  static constexpr absl::string_view kMetadata = "VALUE";
  static constexpr absl::string_view kKey = "KEY";

  LiteRtModelT model;
  LITERT_ASSERT_OK(model.PushMetadata(kKey, kMetadata));
  auto found_metadata = model.FindMetadata(kKey);
  ASSERT_TRUE(found_metadata);
  EXPECT_EQ(found_metadata->StrView(), kMetadata);
}

TEST(ModelTest, MetadataDNE) {
  LiteRtModelT model;
  auto res = model.FindMetadata("FOO");
  ASSERT_FALSE(res.HasValue());
}

TEST(ModelTest, GetBuildStamp) {
  static constexpr absl::string_view kSocManufacturer = "honda";
  static constexpr absl::string_view kSocModel = "accord";

  LiteRtModelT model;

  LITERT_ASSERT_OK_AND_ASSIGN(auto build_stamp_ref,
                              MakeBuildStamp(kSocManufacturer, kSocModel));
  LITERT_ASSERT_OK(model.PushMetadata(kLiteRtBuildStampKey, build_stamp_ref));
  auto build_stamp = GetBuildStamp(model);
  ASSERT_TRUE(build_stamp);
  EXPECT_TRUE(IsCompiled(model));
  EXPECT_EQ(build_stamp->soc_manufacturer, kSocManufacturer);
  EXPECT_EQ(build_stamp->soc_model, kSocModel);
}

TEST(ModelTest, EmplaceSubgraph) {
  LiteRtModelT model;
  auto& sg = model.EmplaceSubgraph();
  EXPECT_EQ(model.Subgraphs().size(), 1);
  auto& tensor = sg.EmplaceTensor();
  EXPECT_EQ(tensor.Weights().GetBufferManager(), model.Buffers());
}

TEST(ModelTest, Signature) {
  static constexpr absl::string_view kSignatureName = "MY_SIGNATURE";

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  std::vector<std::string> inputs = {"input_1", "input_2"};
  std::vector<LiteRtTensor> input_tensors;
  input_tensors.reserve(inputs.size());
  for (const auto& name : inputs) {
    auto& tensor = subgraph.EmplaceTensor();
    tensor.SetName(name);
    subgraph.Inputs().push_back(&tensor);
    input_tensors.push_back(&tensor);
  }

  std::vector<std::string> outputs = {"output_1"};
  std::vector<LiteRtTensor> output_tensors;
  output_tensors.reserve(outputs.size());
  for (const auto& name : outputs) {
    auto& tensor = subgraph.EmplaceTensor();
    tensor.SetName(name);
    subgraph.Outputs().push_back(&tensor);
    output_tensors.push_back(&tensor);
  }

  auto& signature =
      model.EmplaceSignature(&subgraph, inputs, input_tensors, outputs,
                             output_tensors, std::string(kSignatureName));

  auto found_signature = model.FindSignature(kSignatureName);
  ASSERT_TRUE(found_signature);
  EXPECT_EQ(found_signature->get(), signature);
}

TEST(ModelTest, SignatureDNE) {
  static constexpr absl::string_view kSignatureName = "MY_SIGNATURE";
  LiteRtModelT model;
  auto found_signature = model.FindSignature(kSignatureName);
  EXPECT_FALSE(found_signature);
}

TEST(ModelTest, SignatureAllowsDistinctInputOutputAliases) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetName("tensor_internal");
  subgraph.Inputs().push_back(&tensor);
  subgraph.Outputs().push_back(&tensor);

  std::vector<std::string> input_names = {"alias_in"};
  std::vector<LiteRtTensor> input_tensors = {&tensor};
  std::vector<std::string> output_names = {"alias_out"};
  std::vector<LiteRtTensor> output_tensors = {&tensor};

  auto& signature = model.EmplaceSignature(
      &subgraph, std::move(input_names), std::move(input_tensors),
      std::move(output_names), std::move(output_tensors), "sig");

  const auto& sig_inputs = signature.InputNames();
  ASSERT_EQ(sig_inputs.size(), 1);
  EXPECT_EQ(sig_inputs[0], "alias_in");
  const auto& sig_outputs = signature.OutputNames();
  ASSERT_EQ(sig_outputs.size(), 1);
  EXPECT_EQ(sig_outputs[0], "alias_out");

  auto input_tensor = signature.FindInputTensor("alias_in");
  ASSERT_TRUE(input_tensor.HasValue());
  auto output_tensor = signature.FindOutputTensor("alias_out");
  ASSERT_TRUE(output_tensor.HasValue());
  EXPECT_EQ(*input_tensor, *output_tensor);
}

TEST(ModelTest, SignatureMaintainsAliasesPerSignature) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetName("tensor_internal");
  subgraph.Inputs().push_back(&tensor);

  std::vector<std::string> input_names_sig1 = {"alias_sig1"};
  std::vector<LiteRtTensor> input_tensors_sig1 = {&tensor};
  std::vector<std::string> output_names_sig1 = {};
  std::vector<LiteRtTensor> output_tensors_sig1 = {};
  model.EmplaceSignature(
      &subgraph, std::move(input_names_sig1), std::move(input_tensors_sig1),
      std::move(output_names_sig1), std::move(output_tensors_sig1), "sig1");

  std::vector<std::string> input_names_sig2 = {"alias_sig2"};
  std::vector<LiteRtTensor> input_tensors_sig2 = {&tensor};
  std::vector<std::string> output_names_sig2 = {};
  std::vector<LiteRtTensor> output_tensors_sig2 = {};
  model.EmplaceSignature(
      &subgraph, std::move(input_names_sig2), std::move(input_tensors_sig2),
      std::move(output_names_sig2), std::move(output_tensors_sig2), "sig2");

  auto sig1 = model.FindSignature("sig1");
  ASSERT_TRUE(sig1);
  auto sig2 = model.FindSignature("sig2");
  ASSERT_TRUE(sig2);

  const auto& sig1_inputs = sig1->get().InputNames();
  ASSERT_EQ(sig1_inputs.size(), 1);
  EXPECT_EQ(sig1_inputs[0], "alias_sig1");

  const auto& sig2_inputs = sig2->get().InputNames();
  ASSERT_EQ(sig2_inputs.size(), 1);
  EXPECT_EQ(sig2_inputs[0], "alias_sig2");

  auto tensor_sig1 = sig1->get().FindInputTensor("alias_sig1");
  ASSERT_TRUE(tensor_sig1.HasValue());
  auto tensor_sig2 = sig2->get().FindInputTensor("alias_sig2");
  ASSERT_TRUE(tensor_sig2.HasValue());
  EXPECT_EQ(*tensor_sig1, *tensor_sig2);
}

TEST(ModelTest, AttachExternalBufferToOp) {
  static constexpr absl::string_view kBufferData = "BUFFER_DATA";
  static constexpr absl::string_view kOpName = "OP1";
  static constexpr absl::string_view kOp2Name = "OP2";

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  auto& op2 = subgraph.EmplaceOp();

  OwningBufferRef<uint8_t> external_buf(kBufferData);

  auto buf1_id = model.Buffers()->RegisterOwnedBuffer(std::move(external_buf));

  model.AttachAssetToOp(&op, buf1_id, std::string(kOpName));
  model.AttachAssetToOp(&op2, buf1_id, std::string(kOp2Name));

  auto op_1_res = model.FindOpAsset(&op);
  ASSERT_TRUE(op_1_res);
  EXPECT_EQ(op_1_res->second, kOpName);
  EXPECT_EQ(op_1_res->first, buf1_id);

  auto op_2_res = model.FindOpAsset(&op2);
  ASSERT_TRUE(op_2_res);
  EXPECT_EQ(op_2_res->second, kOp2Name);
  EXPECT_EQ(op_2_res->first, buf1_id);
}

TEST(ModelTest, InsertOpAtIndex) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  auto& op2 = subgraph.EmplaceOp();
  auto& op3 = subgraph.EmplaceOpAt(1);

  EXPECT_EQ(subgraph.Ops().size(), 3);
  EXPECT_TRUE(&subgraph.Op(0) == &op);
  EXPECT_TRUE(&subgraph.Op(1) == &op3);
  EXPECT_TRUE(&subgraph.Op(2) == &op2);
}

TEST(ModelTest, ExternalBufferNotFound) {
  LiteRtModelT model;
  LiteRtOpT op;
  ASSERT_FALSE(model.FindOpAsset(&op));
}

//
// Subgraph
//

TEST(ModelSubgraphTest, Input) {
  LiteRtTensorT tensor;
  LiteRtSubgraphT subgraph;
  subgraph.Inputs().push_back(&tensor);
  EXPECT_EQ(&subgraph.Input(0), subgraph.Inputs().front());
  EXPECT_EQ(tensor.NumElements(), 0);
}

TEST(ModelSubgraphTest, Output) {
  LiteRtTensorT tensor;
  LiteRtSubgraphT subgraph;
  subgraph.Outputs().push_back(&tensor);
  EXPECT_EQ(&subgraph.Output(0), subgraph.Outputs().front());
}

TEST(ModelSubgraphTest, EmplaceTensor) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  ASSERT_EQ(subgraph.Tensors().size(), 1);
  EXPECT_THAT(subgraph.Tensors(), ElementsAreArray({&tensor}));
}

TEST(ModelSubgraphTest, EmplaceOp) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_THAT(subgraph.Ops(), ElementsAreArray({&op}));
}

TEST(ModelSubgraphTest, TransferOpsFrom) {
  LiteRtSubgraphT subgraph;
  LiteRtSubgraphT other_subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  other_subgraph.TransferOpsFrom(subgraph.OpsAllocation(), 0);
  EXPECT_EQ(subgraph.Ops().size(), 0);
  EXPECT_EQ(other_subgraph.Ops().size(), 1);
  EXPECT_EQ(other_subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflAdd);
}

TEST(ModelSubgraphTest, TransferTensorsFrom) {
  LiteRtSubgraphT subgraph;
  LiteRtSubgraphT other_subgraph;
  subgraph.EmplaceTensor();
  other_subgraph.TransferTensorsFrom(subgraph.TensorsAllocation());
  EXPECT_EQ(subgraph.Tensors().size(), 0);
  EXPECT_EQ(other_subgraph.Tensors().size(), 1);
}

//
// Op
//

TEST(ModelOpTest, Input) {
  LiteRtOpT op;
  LiteRtTensorT tensor;
  op.Inputs().push_back(&tensor);
  EXPECT_EQ(&op.Input(0), op.Inputs().front());
}

TEST(ModelOpTest, Output) {
  LiteRtOpT op;
  LiteRtTensorT tensor;
  op.Outputs().push_back(&tensor);
  EXPECT_EQ(&op.Output(0), op.Outputs().front());
}

TEST(ModelOpTest, CustomOptions) {
  static constexpr absl::string_view kOpts = "OPTIONS";

  LiteRtOpT op;
  op.SetCustomOptions(kOpts);
  EXPECT_EQ(op.CustomOptions().StrView(), kOpts);
}

TEST(ModelOpTest, Options) {
  static constexpr auto kOptsType = ::tflite::BuiltinOptions_AddOptions;

  TflOptions options;
  options.type = kOptsType;
  options.Set(::tflite::AddOptionsT());

  LiteRtOpT op;
  litert::internal::SetTflOptions(op, std::move(options));

  ASSERT_EQ(litert::internal::GetTflOptions(op).type, kOptsType);
}

TEST(ModelOpTest, OpCode) {
  constexpr static auto kOpCode = kLiteRtOpCodeTflMul;

  LiteRtOpT op;
  op.SetOpCode(kOpCode);
  EXPECT_EQ(op.OpCode(), kOpCode);
}

//
// Tensor
//

TEST(ModelTensorTypeTest, MakeRankedTensorType) {
  static constexpr const int32_t kDims[] = {2, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);
  static constexpr auto kElementType = kLiteRtElementTypeFloat32;
  const auto tensor_type = MakeRankedTensorType(kElementType, kDimsSpan);
  ASSERT_EQ(tensor_type.first, kLiteRtRankedTensorType);
  EXPECT_EQ(tensor_type.second.ranked_tensor_type.element_type, kElementType);
  const auto& layout = tensor_type.second.ranked_tensor_type.layout;
  ASSERT_EQ(layout.rank, kDimsSpan.size());
  EXPECT_THAT(absl::MakeConstSpan(layout.dimensions, kDimsSpan.size()),
              ElementsAreArray(kDimsSpan));
}

TEST(ModelQuantizationTypeTest, MakePerTensor) {
  static constexpr auto kScale = 1.0f;
  static constexpr auto kZero = 1L;
  const auto quant = MakePerTensorQuantization(kScale, kZero);
  ASSERT_EQ(quant.first, kLiteRtQuantizationPerTensor);
  const auto& per_tensor = quant.second.per_tensor;
  EXPECT_EQ(per_tensor.scale, kScale);
  EXPECT_EQ(per_tensor.zero_point, kZero);
}

TEST(ModelQuantizationTypeTest, MakePerChannel) {
  static constexpr std::array kScale = {1.0f, 2.0f};
  static constexpr std::array kZero = {1L, 2L};
  static constexpr int32_t kQdim = 0;

  LiteRtTensorT tensor;
  const auto quant = MakePerChannelQuantization(
      kScale, kZero, kQdim,
      [&tensor](auto s) { return tensor.RequestScratchBuffer(s); });

  ASSERT_EQ(quant.first, kLiteRtQuantizationPerChannel);
  const auto& per_channel = quant.second.per_channel;

  const auto size = per_channel.num_channels;
  ASSERT_EQ(size, 2);
  EXPECT_EQ(per_channel.quantized_dimension, 0);

  auto scales = absl::MakeConstSpan(per_channel.scales, size);
  auto zeros = absl::MakeConstSpan(per_channel.zero_points, size);

  EXPECT_THAT(scales, ElementsAreArray(kScale));
  EXPECT_THAT(zeros, ElementsAreArray(kZero));
}

TEST(ModelWeightsTest, EmptyWeights) {
  LiteRtWeightsT weights;
  EXPECT_EQ(weights.Buffer().Size(), 0);
}

TEST(ModelWeightsTest, WeightsWithExternalBufferManager) {
  static constexpr absl::string_view kData = "some_data";
  BufferManager manager;

  LiteRtWeightsT weights;
  weights.SetBufferManager(&manager);

  BufferRef<uint8_t> buf(kData.data(), kData.size());
  SetWeightsFromUnownedBuffer(weights, buf);

  LITERT_ASSERT_OK_AND_ASSIGN(auto weights_buffer,
                              manager.GetBuffer(weights.GetBufferId()));
  EXPECT_EQ(weights_buffer.StrView(), kData);
  EXPECT_EQ(weights.Buffer().StrView(), kData);
}

TEST(ModelWeightsTest, WeightsFromUnownedBuffer) {
  static constexpr absl::string_view kData = "some_data";

  LiteRtWeightsT weights;
  BufferRef<uint8_t> buf(kData.data(), kData.size());
  SetWeightsFromUnownedBuffer(weights, buf);

  EXPECT_EQ(weights.Buffer().StrView(), kData);
}

TEST(ModelWeightsTest, WeightsFromOwnedBuffer) {
  static constexpr absl::string_view kData = "some_data";

  LiteRtWeightsT weights;

  OwningBufferRef<uint8_t> buf(kData);
  SetWeightsFromUnownedBuffer(weights, std::move(buf));

  EXPECT_EQ(weights.Buffer().StrView(), kData);
}

TEST(ModelWeightsTest, OverwriteBuffer) {
  static constexpr absl::string_view kData = "some_data";
  static constexpr absl::string_view kData2 = "some_data2";

  LiteRtWeightsT weights;

  {
    OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }

  {
    OwningBufferRef<uint8_t> buf(kData2);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }

  EXPECT_EQ(weights.Buffer().StrView(), kData2);
}

TEST(ModelTensorTest, Name) {
  static constexpr absl::string_view kName = "TENSOR_NAME";

  LiteRtTensorT tensor;
  tensor.SetName(std::string(kName.begin(), kName.end()));
  EXPECT_EQ(tensor.Name(), kName);
}

TEST(ModelTensorTest, Use) {
  LiteRtTensorT tensor;
  tensor.Users().emplace_back();
  tensor.UserArgInds().push_back(0);
  auto [user, ind] = tensor.GetUse(0);
  EXPECT_EQ(user, tensor.Users().front());
  EXPECT_EQ(ind, 0);
}

TEST(ModelTensorTest, DefiningOp) {
  LiteRtTensorT tensor;
  LiteRtOpT op;
  tensor.SetDefiningOp(op, 0);
  EXPECT_EQ(tensor.DefiningOp(), &op);
  EXPECT_EQ(tensor.DefiningOpOutInd(), 0);
}

TEST(ModelTest, TransferSubgraphToReindexComposite) {
  LiteRtModelT model;

  auto& subgraph = model.EmplaceSubgraph();
  auto& other_subgraph = model.EmplaceSubgraph();
  auto& decomp_subgraph = model.EmplaceSubgraph();

  auto& composite = subgraph.EmplaceOp();
  composite.SetOpCode(kLiteRtOpCodeShloComposite);
  ::tflite::StableHLOCompositeOptionsT opts;
  opts.name = "composite";
  opts.decomposition_subgraph_index = 2;
  TflOptions2 options;
  options.type = tflite::BuiltinOptions2_StableHLOCompositeOptions;
  options.Set(std::move(opts));
  litert::internal::SetTflOptions2(composite, std::move(options));

  LiteRtSubgraphT::Alloc dest;
  std::vector<size_t> indices = {1};
  model.TransferSubgraphTo(dest, std::move(indices));

  EXPECT_THAT(model.Subgraphs(),
              ElementsAreArray({&subgraph, &decomp_subgraph}));
  EXPECT_THAT(dest.Elements(), ElementsAreArray({&other_subgraph}));

  const auto& new_opts = litert::internal::GetTflOptions2(composite);
  const auto new_decomp_ind =
      new_opts.AsStableHLOCompositeOptions()->decomposition_subgraph_index;
  EXPECT_EQ(new_decomp_ind, 1);
}

TEST(ModelTest, TransferSubgraphToReindexCompositeNoChange) {
  LiteRtModelT model;

  auto& subgraph = model.EmplaceSubgraph();
  auto& decomp_subgraph = model.EmplaceSubgraph();
  auto& other_subgraph = model.EmplaceSubgraph();

  auto& composite = subgraph.EmplaceOp();
  composite.SetOpCode(kLiteRtOpCodeShloComposite);
  ::tflite::StableHLOCompositeOptionsT opts;
  opts.name = "composite";
  opts.decomposition_subgraph_index = 1;
  TflOptions2 options;
  options.type = tflite::BuiltinOptions2_StableHLOCompositeOptions;
  ;
  options.Set(std::move(opts));
  litert::internal::SetTflOptions2(composite, std::move(options));

  LiteRtSubgraphT::Alloc dest;
  std::vector<size_t> indices = {2};
  model.TransferSubgraphTo(dest, std::move(indices));

  EXPECT_THAT(model.Subgraphs(),
              ElementsAreArray({&subgraph, &decomp_subgraph}));
  EXPECT_THAT(dest.Elements(), ElementsAreArray({&other_subgraph}));

  const auto& new_opts = litert::internal::GetTflOptions2(composite);
  const auto new_decomp_ind =
      new_opts.AsStableHLOCompositeOptions()->decomposition_subgraph_index;
  EXPECT_EQ(new_decomp_ind, 1);
}

TEST(ModelTest, TransferSubgraphToReindexCompositeMultiple) {
  LiteRtModelT model;

  auto& subgraph = model.EmplaceSubgraph();
  auto& other_subgraph = model.EmplaceSubgraph();
  auto& other_subgraph2 = model.EmplaceSubgraph();
  auto& other_subgraph3 = model.EmplaceSubgraph();
  auto& decomp_subgraph = model.EmplaceSubgraph();
  auto& other_subgraph4 = model.EmplaceSubgraph();

  auto& composite = subgraph.EmplaceOp();
  composite.SetOpCode(kLiteRtOpCodeShloComposite);
  ::tflite::StableHLOCompositeOptionsT opts;
  opts.name = "composite";
  opts.decomposition_subgraph_index = 4;
  TflOptions2 options;
  options.type = tflite::BuiltinOptions2_StableHLOCompositeOptions;
  ;
  options.Set(std::move(opts));
  litert::internal::SetTflOptions2(composite, std::move(options));

  LiteRtSubgraphT::Alloc dest;
  std::vector<size_t> indices = {1, 3, 5};
  model.TransferSubgraphTo(dest, std::move(indices));

  EXPECT_THAT(model.Subgraphs(), ElementsAreArray({&subgraph, &other_subgraph2,
                                                   &decomp_subgraph}));
  EXPECT_THAT(
      dest.Elements(),
      ElementsAreArray({&other_subgraph, &other_subgraph3, &other_subgraph4}));

  const auto& new_opts = litert::internal::GetTflOptions2(composite);
  const auto new_decomp_ind =
      new_opts.AsStableHLOCompositeOptions()->decomposition_subgraph_index;
  EXPECT_EQ(new_decomp_ind, 2);
}

//
// Misc Ir Containers
//

TEST(ModelOpListTest, Push) {
  LiteRtOpListT op_list;
  LiteRtOpT op;
  op_list.Push(&op);
  auto vec = op_list.Values();
  EXPECT_EQ(vec.front().first, &op);
}

TEST(ModelOpListTest, PushWithIndex) {
  LiteRtOpListT op_list;
  LiteRtOpT op;
  op_list.Push(&op, 1);
  auto vec = op_list.Values();
  EXPECT_EQ(vec.front().first, &op);
  EXPECT_EQ(vec.front().second, 1);
}

//
// Traversal Utils
//

TEST(CcForEachIrTest, OpF3) {
  LiteRtModelT model;
  model.EmplaceSubgraph().EmplaceOp();

  int count = 0;
  ForEachIr(&model, [&](LiteRtSubgraph subgraph, int32_t subgraph_index,
                        LiteRtOp op) { count++; });
  EXPECT_EQ(count, 1);
}

TEST(CcForEachIrTest, OpF1) {
  LiteRtModelT model;
  model.EmplaceSubgraph().EmplaceOp();

  int count = 0;
  ForEachIr(&model, [&](LiteRtOp op) { count++; });
  EXPECT_EQ(count, 1);
}

TEST(CcForEachIrTest, OpF2) {
  LiteRtModelT model;
  model.EmplaceSubgraph().EmplaceOp();

  int count = 0;
  ForEachIr(&model, [&](LiteRtSubgraph subgraph, LiteRtOp op) { count++; });
  EXPECT_EQ(count, 1);
}

TEST(CcForEachIrTest, SgF1) {
  LiteRtModelT model;
  model.EmplaceSubgraph().EmplaceOp();

  int count = 0;
  ForEachIr(&model, [&](LiteRtSubgraph subgraph) { count++; });
  EXPECT_EQ(count, 1);
}

TEST(CcForEachIrTest, SgF2) {
  LiteRtModelT model;
  model.EmplaceSubgraph().EmplaceOp();

  int count = 0;
  ForEachIr(&model,
            [&](LiteRtSubgraph subgraph, int32_t subgraph_index) { count++; });
  EXPECT_EQ(count, 1);
}

//
// Printing
//

TEST(PrintingTest, RankedTensorType) {
  EXPECT_EQ(absl::StrFormat(
                "%v", MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 2})),
            "2d_i32<1x2>");
}

TEST(PrintingTest, Tensor) {
  LiteRtTensorT tensor;
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
  EXPECT_EQ(absl::StrFormat("%v", tensor), "3d_i32<2x2x2>");
}

TEST(PrintingTest, ConstTensor) {
  OwningBufferRef<uint8_t> buf(8);
  LiteRtTensorT tensor;
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
  SetWeightsFromOwnedBuffer(tensor.Weights(), std::move(buf));
  EXPECT_EQ(absl::StrFormat("%v", tensor), "3d_i32<2x2x2>_cst[8B]");
}

TEST(PrintingTest, TensoVector) {
  LiteRtTensorT tensor;
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));

  LiteRtTensorT tensor2;
  tensor2.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));

  std::vector<LiteRtTensor> tensors = {&tensor, &tensor2};
  EXPECT_EQ(absl::StrFormat("%v", tensors), "(3d_i32<2x2x2>,3d_i32<2x2x2>)");
}

TEST(PrintingTest, Op) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  {
    ::tflite::AddOptionsT add_opts;
    add_opts.fused_activation_function = ::tflite::ActivationFunctionType_RELU;
    add_opts.pot_scale_int16 = false;
    TflOptions opts;
    opts.type = ::tflite::BuiltinOptions_AddOptions;
    opts.Set(std::move(add_opts));
    litert::internal::SetTflOptions(op, std::move(opts));
  }

  LiteRtTensorT tensor;
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
  op.Inputs().push_back(&tensor);

  LiteRtTensorT tensor2;
  tensor2.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2}));
  op.Inputs().push_back(&tensor2);

  LiteRtTensorT tensor3;
  tensor3.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
  op.Outputs().push_back(&tensor3);

  EXPECT_EQ(absl::StrFormat("%v", op),
            "tfl.add{fa=RELU}(3d_i32<2x2x2>,1d_i32<2>)->(3d_i32<2x2x2>)");
}

TEST(PrintingTest, TflOptions) {
  TflOptions opts;
  opts.type = ::tflite::BuiltinOptions_AddOptions;
  ::tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = ::tflite::ActivationFunctionType_RELU;
  add_opts.pot_scale_int16 = false;
  opts.Set(std::move(add_opts));
  EXPECT_EQ(absl::StrFormat("%v", opts), "{fa=RELU}");
}

TEST(PrintingTest, TflOptionsNoPrinter) {
  TflOptions opts;
  opts.type = ::tflite::BuiltinOptions_SubOptions;
  ::tflite::SubOptionsT add_opts;
  opts.Set(std::move(add_opts));
  EXPECT_EQ(absl::StrFormat("%v", opts), "{!no_printer}");
}

TEST(PrintingTest, TflOptions2NoPrinter) {
  TflOptions2 opts;
  opts.type = ::tflite::BuiltinOptions2_StableHLOCompositeOptions;
  ::tflite::StableHLOCompositeOptionsT comp_opts;
  opts.Set(std::move(comp_opts));
  EXPECT_EQ(absl::StrFormat("%v", opts), "{!no_printer}");
}

TEST(PrintingTest, FusedActivationFunction) {
  EXPECT_EQ(absl::StrFormat("%v", ::tflite::ActivationFunctionType_RELU),
            "RELU");
}

TEST(PrintingTest, TflNullOptions) {
  ::tflite::AddOptionsT* add_opts = nullptr;
  EXPECT_EQ(absl::StrFormat("%v", add_opts), "{null}");
}

TEST(PrintingTest, TflAddOptions) {
  ::tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = ::tflite::ActivationFunctionType_RELU6;
  add_opts.pot_scale_int16 = true;
  EXPECT_EQ(absl::StrFormat("%v", add_opts), "{fa=RELU6,pot=true}");
}

TEST(PrintingTest, TflAddOptionsPointer) {
  ::tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = ::tflite::ActivationFunctionType_RELU6;
  add_opts.pot_scale_int16 = true;
  EXPECT_EQ(absl::StrFormat("%v", &add_opts), "{fa=RELU6,pot=true}");
}

TEST(PrintingTest, Subgraph) {
  LiteRtSubgraphT subgraph;

  {
    auto& op = subgraph.EmplaceOp();

    op.SetOpCode(kLiteRtOpCodeTflAdd);

    ::tflite::AddOptionsT add_opts;
    add_opts.fused_activation_function = ::tflite::ActivationFunctionType_RELU;
    add_opts.pot_scale_int16 = false;
    TflOptions opts;
    opts.type = ::tflite::BuiltinOptions_AddOptions;
    opts.Set(std::move(add_opts));
    litert::internal::SetTflOptions(op, std::move(opts));

    auto& tensor = subgraph.EmplaceTensor();
    tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
    op.Inputs().push_back(&tensor);

    auto& tensor2 = subgraph.EmplaceTensor();
    tensor2.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2}));
    op.Inputs().push_back(&tensor2);

    auto& tensor3 = subgraph.EmplaceTensor();
    tensor3.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
    op.Outputs().push_back(&tensor3);
  }

  {
    auto& op = subgraph.EmplaceOp();

    op.SetOpCode(kLiteRtOpCodeTflAdd);

    ::tflite::AddOptionsT add_opts;
    add_opts.fused_activation_function = ::tflite::ActivationFunctionType_RELU;
    add_opts.pot_scale_int16 = false;
    TflOptions opts;
    opts.type = ::tflite::BuiltinOptions_AddOptions;
    opts.Set(std::move(add_opts));
    litert::internal::SetTflOptions(op, std::move(opts));

    auto& tensor = subgraph.EmplaceTensor();
    tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
    op.Inputs().push_back(&tensor);

    auto& tensor2 = subgraph.EmplaceTensor();
    tensor2.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2}));
    op.Inputs().push_back(&tensor2);

    auto& tensor3 = subgraph.EmplaceTensor();
    tensor3.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
    op.Outputs().push_back(&tensor3);
  }

  EXPECT_EQ(absl::StrFormat("%v", subgraph.Ops()),
            "tfl.add{fa=RELU}(3d_i32<2x2x2>,1d_i32<2>)->(3d_i32<2x2x2>)/"
            "tfl.add{fa=RELU}(3d_i32<2x2x2>,1d_i32<2>)->(3d_i32<2x2x2>)");
}

}  // namespace
}  // namespace litert::internal
