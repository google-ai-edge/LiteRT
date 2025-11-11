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

#include "litert/c/litert_model.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/test/matchers.h"

namespace {

using ::litert::BufferRef;
using ::litert::OwningBufferRef;
using ::testing::ElementsAreArray;
using ::testing::litert::IsError;

TEST(LiteRtWeightsTest, GetNullWeights) {
  LiteRtWeightsT weights = {};

  const void* addr;
  size_t size;
  LITERT_ASSERT_OK(LiteRtGetWeightsBytes(&weights, &addr, &size));

  EXPECT_EQ(addr, nullptr);
  EXPECT_EQ(size, 0);
}

TEST(LiteRtWeightsTest, GetWeights) {
  static constexpr std::array kData = {1, 2, 3};
  const uint8_t* kDataPtr = reinterpret_cast<const uint8_t*>(kData.data());
  const auto kDataSize = kData.size() * sizeof(int32_t);

  LiteRtWeightsT weights;
  SetWeightsFromOwnedBuffer(weights,
                            OwningBufferRef<uint8_t>(kDataPtr, kDataSize));

  const void* addr;
  size_t size;
  LITERT_ASSERT_OK(LiteRtGetWeightsBytes(&weights, &addr, &size));

  EXPECT_NE(addr, nullptr);
  EXPECT_EQ(size, 3 * sizeof(int32_t));

  EXPECT_THAT(absl::MakeConstSpan(reinterpret_cast<const int32_t*>(addr), 3),
              ElementsAreArray(kData));
}

TEST(LiteRtWeightsTest, GetBufferId) {
  static constexpr std::array kData = {1, 2, 3};
  const uint8_t* kDataPtr = reinterpret_cast<const uint8_t*>(kData.data());
  const auto kDataSize = kData.size() * sizeof(int32_t);

  LiteRtWeightsT weights;
  SetWeightsFromOwnedBuffer(weights,
                            OwningBufferRef<uint8_t>(kDataPtr, kDataSize));

  int32_t buffer_id;
  LITERT_ASSERT_OK(LiteRtGetWeightsBufferId(&weights, &buffer_id));
  EXPECT_EQ(buffer_id, 1);
}

TEST(LiteRtTensorTest, GetUnrankedType) {
  static constexpr auto kElementType = kLiteRtElementTypeFloat32;
  static constexpr auto kId = kLiteRtUnrankedTensorType;

  TensorType type;
  type.first = kId;
  type.second.unranked_tensor_type.element_type = kElementType;

  LiteRtTensorT tensor;
  tensor.SetType(std::move(type));

  LiteRtTensorTypeId id;
  LITERT_ASSERT_OK(LiteRtGetTensorTypeId(&tensor, &id));
  ASSERT_EQ(id, kId);

  LiteRtUnrankedTensorType unranked;
  LITERT_ASSERT_OK(LiteRtGetUnrankedTensorType(&tensor, &unranked));
  EXPECT_EQ(unranked.element_type, kElementType);
}

TEST(LiteRtTensorTest, GetRankedTensorType) {
  static constexpr auto kElementType = kLiteRtElementTypeFloat32;
  static constexpr auto kId = kLiteRtRankedTensorType;

  LiteRtTensorT tensor;
  tensor.SetType(MakeRankedTensorType(kElementType, {3, 3}));

  LiteRtTensorTypeId id;
  LITERT_ASSERT_OK(LiteRtGetTensorTypeId(&tensor, &id));
  ASSERT_EQ(id, kId);

  LiteRtRankedTensorType ranked;
  LITERT_ASSERT_OK(LiteRtGetRankedTensorType(&tensor, &ranked));
  EXPECT_EQ(ranked.element_type, kElementType);
  ASSERT_EQ(ranked.layout.rank, 2);
  EXPECT_THAT(absl::MakeConstSpan(ranked.layout.dimensions, 2),
              ElementsAreArray({3, 3}));
}

TEST(LiteRtTensorTest, GetUses) {
  LiteRtTensorT tensor;

  LiteRtOpT user;
  tensor.Users().push_back(&user);
  tensor.UserArgInds().push_back(0);

  LiteRtOpT other_user;
  tensor.Users().push_back(&other_user);
  tensor.UserArgInds().push_back(1);

  LiteRtParamIndex num_uses;
  LITERT_ASSERT_OK(LiteRtGetNumTensorUses(&tensor, &num_uses));
  ASSERT_EQ(num_uses, 2);

  LiteRtOp actual_user;
  LiteRtParamIndex actual_user_arg_index;
  LITERT_ASSERT_OK(LiteRtGetTensorUse(&tensor, /*use_index=*/0, &actual_user,
                                      &actual_user_arg_index));
  ASSERT_EQ(actual_user, &user);
  ASSERT_EQ(actual_user_arg_index, 0);

  LITERT_ASSERT_OK(LiteRtGetTensorUse(&tensor, /*use_index=*/1, &actual_user,
                                      &actual_user_arg_index));
  ASSERT_EQ(actual_user, &other_user);
  ASSERT_EQ(actual_user_arg_index, 1);
}

TEST(LiteRtTensorTest, GetDefiningOp) {
  LiteRtTensorT tensor;

  LiteRtOpT def_op;
  tensor.SetDefiningOp(def_op, 0);

  LiteRtTensorDefiningOp actual_def_op;
  bool has_defining_op;
  LITERT_ASSERT_OK(
      LiteRtGetTensorDefiningOp(&tensor, &has_defining_op, &actual_def_op));
  ASSERT_TRUE(has_defining_op);
  EXPECT_EQ(actual_def_op.op, &def_op);
  EXPECT_EQ(actual_def_op.op_output_index, 0);
}

TEST(LiteRtTensorTest, NoDefiningOp) {
  LiteRtTensorT tensor;

  LiteRtTensorDefiningOp actual_def_op;
  bool has_defining_op;
  LITERT_ASSERT_OK(
      LiteRtGetTensorDefiningOp(&tensor, &has_defining_op, &actual_def_op));
  ASSERT_FALSE(has_defining_op);
}

TEST(LiteRtTensorTest, Name) {
  static constexpr const char kName[] = "foo";

  LiteRtTensorT tensor;
  tensor.SetName(std::string(kName));

  const char* name;
  LITERT_ASSERT_OK(LiteRtGetTensorName(&tensor, &name));
  EXPECT_STREQ(name, kName);
}

TEST(LiteRtTensorTest, Index) {
  static constexpr const std::uint32_t kTensorIndex = 1;

  LiteRtTensorT tensor;
  tensor.SetTensorIndex(kTensorIndex);

  std::uint32_t index;
  LITERT_ASSERT_OK(LiteRtGetTensorIndex(&tensor, &index));
  EXPECT_EQ(index, kTensorIndex);
}

TEST(LiteRtTensorTest, QuantizationNone) {
  LiteRtTensorT tensor;

  LiteRtQuantizationTypeId q_type_id;
  LITERT_ASSERT_OK(LiteRtGetQuantizationTypeId(&tensor, &q_type_id));
  EXPECT_EQ(q_type_id, kLiteRtQuantizationNone);

  LiteRtQuantizationPerTensor per_tensor_quantization;
  EXPECT_NE(LiteRtGetPerTensorQuantization(&tensor, &per_tensor_quantization),
            kLiteRtStatusOk);
}

TEST(LiteRtTensorTest, QuantizationPerTensor) {
  static constexpr auto kScale = 1.0;
  static constexpr auto kZeroPoint = 1;

  LiteRtTensorT tensor;
  tensor.SetQarams(MakePerTensorQuantization(kScale, kZeroPoint));

  LiteRtQuantizationTypeId q_type_id;
  LITERT_ASSERT_OK(LiteRtGetQuantizationTypeId(&tensor, &q_type_id));
  ASSERT_EQ(q_type_id, kLiteRtQuantizationPerTensor);

  LiteRtQuantizationPerTensor per_tensor_quantization;
  LITERT_ASSERT_OK(
      LiteRtGetPerTensorQuantization(&tensor, &per_tensor_quantization));

  EXPECT_EQ(per_tensor_quantization.scale, kScale);
  EXPECT_EQ(per_tensor_quantization.zero_point, kZeroPoint);
}

TEST(LiteRtTensorTest, QuantizationPerChannel) {
  static constexpr size_t kNumChannels = 2;
  static constexpr size_t kQuantizedDimension = 0;
  static constexpr float kScales[kNumChannels] = {1.0, 2.0};
  static constexpr int64_t kZps[kNumChannels] = {2, 3};

  LiteRtTensorT tensor;

  {
    auto per_channel =
        MakePerChannelQuantization(kScales, kZps, kQuantizedDimension, tensor);
    tensor.SetQarams(per_channel);
  }

  LiteRtQuantizationTypeId q_type_id;
  LITERT_ASSERT_OK(LiteRtGetQuantizationTypeId(&tensor, &q_type_id));
  ASSERT_EQ(q_type_id, kLiteRtQuantizationPerChannel);

  LiteRtQuantizationPerChannel per_channel_quantization;
  LITERT_ASSERT_OK(
      LiteRtGetPerChannelQuantization(&tensor, &per_channel_quantization));

  EXPECT_THAT(
      absl::MakeConstSpan(per_channel_quantization.scales, kNumChannels),
      testing::ElementsAreArray(kScales));
  EXPECT_THAT(
      absl::MakeConstSpan(per_channel_quantization.zero_points, kNumChannels),
      testing::ElementsAreArray(kZps));
  ASSERT_EQ(per_channel_quantization.num_channels, kNumChannels);
  ASSERT_EQ(per_channel_quantization.quantized_dimension, kQuantizedDimension);
}

TEST(LiteRtOpTest, GetOpCode) {
  static constexpr auto kCode = kLiteRtOpCodeTflCustom;

  LiteRtOpT op;
  op.SetOpCode(kCode);

  LiteRtOpCode code;
  LITERT_ASSERT_OK(LiteRtGetOpCode(&op, &code));
  EXPECT_EQ(code, kCode);
}

TEST(LiteRtOpTest, GetInputs) {
  LiteRtTensorT input1;
  LiteRtTensorT input2;

  LiteRtOpT op;
  op.Inputs().push_back(&input1);
  op.Inputs().push_back(&input2);

  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_OK(LiteRtGetNumOpInputs(&op, &num_inputs));
  ASSERT_EQ(num_inputs, 2);

  LiteRtTensor actual_input;
  LITERT_ASSERT_OK(LiteRtGetOpInput(&op, /*input_index=*/0, &actual_input));
  EXPECT_EQ(actual_input, &input1);

  LITERT_ASSERT_OK(LiteRtGetOpInput(&op, /*input_index=*/1, &actual_input));
  EXPECT_EQ(actual_input, &input2);
}

TEST(LiteRtOpTest, GetOutputs) {
  LiteRtTensorT output1;
  LiteRtTensorT output2;

  LiteRtOpT op;
  op.Outputs().push_back(&output1);
  op.Outputs().push_back(&output2);

  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_OK(LiteRtGetNumOpOutputs(&op, &num_outputs));
  ASSERT_EQ(num_outputs, 2);

  LiteRtTensor actual_output;
  LITERT_ASSERT_OK(LiteRtGetOpOutput(&op, /*output_index=*/0, &actual_output));
  EXPECT_EQ(actual_output, &output1);

  LITERT_ASSERT_OK(LiteRtGetOpOutput(&op, /*output_index=*/1, &actual_output));
  EXPECT_EQ(actual_output, &output2);
}

TEST(LiteRtOpTest, GetCustomCode) {
  LiteRtOpT op;
  op.SetCustomCode("custom_code");
  op.SetOpCode(kLiteRtOpCodeTflCustom);
  const char* code;
  LITERT_ASSERT_OK(LiteRtGetCustomCode(&op, &code));
  EXPECT_STREQ(code, "custom_code");
}

TEST(LiteRtOpTest, GetCustomCodeNotCustom) {
  LiteRtOpT op;
  op.SetCustomCode("custom_code");
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  const char* code;
  EXPECT_NE(LiteRtGetCustomCode(&op, &code), kLiteRtStatusOk);
}

TEST(LiteRtSubgraphTest, GetInputs) {
  LiteRtTensorT input1;
  LiteRtTensorT input2;

  LiteRtSubgraphT subgraph;
  subgraph.Inputs().push_back(&input1);
  subgraph.Inputs().push_back(&input2);

  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphInputs(&subgraph, &num_inputs));

  LiteRtTensor actual_input;
  LITERT_ASSERT_OK(
      LiteRtGetSubgraphInput(&subgraph, /*input_index=*/0, &actual_input));
  EXPECT_EQ(actual_input, &input1);

  LITERT_ASSERT_OK(
      LiteRtGetSubgraphInput(&subgraph, /*input_index=*/1, &actual_input));
  EXPECT_EQ(actual_input, &input2);
}

TEST(LiteRtSubgraphTest, GetOutputs) {
  LiteRtTensorT output1;
  LiteRtTensorT output2;

  LiteRtSubgraphT subgraph;
  subgraph.Outputs().push_back(&output1);
  subgraph.Outputs().push_back(&output2);

  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphOutputs(&subgraph, &num_outputs));

  LiteRtTensor actual_output;
  LITERT_ASSERT_OK(
      LiteRtGetSubgraphOutput(&subgraph, /*output_index=*/0, &actual_output));
  EXPECT_EQ(actual_output, &output1);

  LITERT_ASSERT_OK(
      LiteRtGetSubgraphOutput(&subgraph, /*output_index=*/1, &actual_output));
  EXPECT_EQ(actual_output, &output2);
}

TEST(LiteRtSubgraphTest, GetOps) {
  LiteRtSubgraphT subgraph;
  auto& op1 = subgraph.EmplaceOp();
  auto& op2 = subgraph.EmplaceOp();

  LiteRtParamIndex num_ops;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphOps(&subgraph, &num_ops));
  ASSERT_EQ(num_ops, 2);

  LiteRtOp actual_op;
  LITERT_ASSERT_OK(LiteRtGetSubgraphOp(&subgraph, /*op_index=*/0, &actual_op));
  ASSERT_EQ(actual_op, &op1);

  LITERT_ASSERT_OK(LiteRtGetSubgraphOp(&subgraph, /*op_index=*/1, &actual_op));
  ASSERT_EQ(actual_op, &op2);
}

TEST(LiteRtModelTest, GetMetadata) {
  static constexpr absl::string_view kKey = "KEY";
  static constexpr absl::string_view kData = "DATA";

  LiteRtModelT model;
  model.PushMetadata(kKey, kData);

  const void* metadata;
  size_t metadata_size;
  LITERT_ASSERT_OK(
      LiteRtGetModelMetadata(&model, kKey.data(), &metadata, &metadata_size));
  EXPECT_EQ(BufferRef(metadata, metadata_size).StrView(), kData);
}

TEST(LiteRtModelTest, AddMetadataSuccess) {
  static constexpr absl::string_view kKey = "KEY";
  static constexpr absl::string_view kData = "DATA";

  LiteRtModelT model;
  LITERT_ASSERT_OK(
      LiteRtAddModelMetadata(&model, kKey.data(), kData.data(), kData.size()));

  const void* metadata;
  size_t metadata_size;
  LITERT_ASSERT_OK(
      LiteRtGetModelMetadata(&model, kKey.data(), &metadata, &metadata_size));
  EXPECT_EQ(BufferRef(metadata, metadata_size).StrView(), kData);
}

TEST(LiteRtModelTest, AddMetadataGetMetadataOutsideOfScopeSuccess) {
  LiteRtModelT model;
  std::string kExpectedKey = "KEY";
  std::string kExpectedData = "DATA";
  // Scope to add metadata.
  {
    std::string kKey = "KEY";
    std::string kData = "DATA";
    LITERT_ASSERT_OK(LiteRtAddModelMetadata(&model, kKey.c_str(), kData.c_str(),
                                            kData.size()));
  }
  const void* metadata;
  size_t metadata_size;
  LITERT_ASSERT_OK(LiteRtGetModelMetadata(&model, kExpectedKey.data(),
                                          &metadata, &metadata_size));
  EXPECT_EQ(BufferRef(metadata, metadata_size).StrView(), kExpectedData.data());
}

TEST(LiteRtModelTest, AddMetadataExistingFails) {
  static constexpr absl::string_view kKey = "KEY";
  static constexpr absl::string_view kData = "DATA";

  LiteRtModelT model;
  LITERT_ASSERT_OK(
      LiteRtAddModelMetadata(&model, kKey.data(), kData.data(), kData.size()));

  // Adding again should fail.
  EXPECT_THAT(
      LiteRtAddModelMetadata(&model, kKey.data(), kData.data(), kData.size()),
      IsError(kLiteRtStatusErrorAlreadyExists));
}

TEST(LiteRtModelTest, GetSubgraph) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  LiteRtSubgraph actual_subgraph;
  LITERT_ASSERT_OK(LiteRtGetModelSubgraph(&model, 0, &actual_subgraph));
  EXPECT_EQ(actual_subgraph, &subgraph);
}

TEST(LiteRtModelTest, GetSubgraphOOB) {
  LiteRtModelT model;

  LiteRtSubgraph actual_subgraph;
  EXPECT_THAT(LiteRtGetModelSubgraph(&model, 0, &actual_subgraph),
              IsError(kLiteRtStatusErrorIndexOOB));
}

TEST(LiteRtModelTest, SerializeModelWithSignaturesWithOneSignature) {
  // This test checks that the serialization succeeds and the signature is
  // added correctly even if the model has only one subgraph.
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& input_tensor = subgraph.EmplaceTensor();
  input_tensor.SetName("input");
  subgraph.Inputs().push_back(&input_tensor);

  auto& output_tensor = subgraph.EmplaceTensor();
  output_tensor.SetName("output");
  subgraph.Outputs().push_back(&output_tensor);

  const char* signature_key = "serving_default";
  char* signatures[] = {const_cast<char*>(signature_key)};

  uint8_t* buf = nullptr;
  size_t size = 0;
  size_t offset = 0;
  const LiteRtModelSerializationOptions options = {/*bytecode_alignment=*/64};

  // We expect this to fail on serialization if NPU is disabled, but signature
  // should be added regardless.
  const LiteRtStatus status =
      LiteRtSerializeModelWithSignatures(&model, &buf, &size, &offset,
                                         /*destroy_model=*/false, signatures,
                                         /*num_signatures=*/1, options);

#ifdef LITERT_BUILD_INCLUDE_NPU
  LITERT_ASSERT_OK(status);
  EXPECT_NE(buf, nullptr);
  EXPECT_GT(size, 0);
#else
  EXPECT_NE(status, kLiteRtStatusOk);
  EXPECT_EQ(buf, nullptr);
  EXPECT_EQ(size, 0);
#endif

  // The model should now have one signature.
  LiteRtParamIndex num_signatures;
  LITERT_ASSERT_OK(LiteRtGetNumModelSignatures(&model, &num_signatures));
  ASSERT_EQ(num_signatures, 1);

  LiteRtSignature signature;
  LITERT_ASSERT_OK(LiteRtGetModelSignature(&model, 0, &signature));

  const char* key;
  LITERT_ASSERT_OK(LiteRtGetSignatureKey(signature, &key));
  EXPECT_STREQ(key, "serving_default");

  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_OK(LiteRtGetNumSignatureInputs(signature, &num_inputs));
  ASSERT_EQ(num_inputs, 1);

  const char* input_name;
  LITERT_ASSERT_OK(LiteRtGetSignatureInputName(signature, 0, &input_name));
  EXPECT_STREQ(input_name, "input");

  LiteRtTensor sig_input_tensor;
  LITERT_ASSERT_OK(
      LiteRtGetSignatureInputTensorByIndex(signature, 0, &sig_input_tensor));
  EXPECT_EQ(sig_input_tensor, &input_tensor);

  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_OK(LiteRtGetNumSignatureOutputs(signature, &num_outputs));
  ASSERT_EQ(num_outputs, 1);

  const char* output_name;
  LITERT_ASSERT_OK(LiteRtGetSignatureOutputName(signature, 0, &output_name));
  EXPECT_STREQ(output_name, "output");

  LiteRtTensor sig_output_tensor;
  LITERT_ASSERT_OK(
      LiteRtGetSignatureOutputTensorByIndex(signature, 0, &sig_output_tensor));
  EXPECT_EQ(sig_output_tensor, &output_tensor);

  // Clean up buffer if serialization succeeded.
  free(buf);
}

TEST(LiteRtModelTest, SerializeModelWithSignaturesMultipleSubgraphs) {
  // This test checks that the serialization succeeds and the signatures are
  // added correctly even if the model has multiple subgraphs.
  LiteRtModelT model;
  auto& subgraph1 = model.EmplaceSubgraph();
  auto& input_tensor1 = subgraph1.EmplaceTensor();
  input_tensor1.SetName("input1");
  subgraph1.Inputs().push_back(&input_tensor1);
  auto& output_tensor1 = subgraph1.EmplaceTensor();
  output_tensor1.SetName("output1");
  subgraph1.Outputs().push_back(&output_tensor1);

  auto& subgraph2 = model.EmplaceSubgraph();
  auto& input_tensor2 = subgraph2.EmplaceTensor();
  input_tensor2.SetName("input2");
  subgraph2.Inputs().push_back(&input_tensor2);
  auto& output_tensor2 = subgraph2.EmplaceTensor();
  output_tensor2.SetName("output2");
  subgraph2.Outputs().push_back(&output_tensor2);

  const char* signature_key1 = "sig1";
  const char* signature_key2 = "sig2";
  char* signatures[] = {const_cast<char*>(signature_key1),
                        const_cast<char*>(signature_key2)};

  uint8_t* buf = nullptr;
  size_t size = 0;
  size_t offset = 0;
  const LiteRtModelSerializationOptions options = {/*bytecode_alignment=*/64};

  const LiteRtStatus status =
      LiteRtSerializeModelWithSignatures(&model, &buf, &size, &offset,
                                         /*destroy_model=*/false, signatures,
                                         /*num_signatures=*/2, options);

#ifdef LITERT_BUILD_INCLUDE_NPU
  LITERT_ASSERT_OK(status);
  EXPECT_NE(buf, nullptr);
  EXPECT_GT(size, 0);
#else
  EXPECT_NE(status, kLiteRtStatusOk);
  EXPECT_EQ(buf, nullptr);
  EXPECT_EQ(size, 0);
#endif

  // The model should now have two signatures.
  LiteRtParamIndex num_signatures;
  LITERT_ASSERT_OK(LiteRtGetNumModelSignatures(&model, &num_signatures));
  ASSERT_EQ(num_signatures, 2);

  // Check first signature.
  {
    LiteRtSignature signature;
    LITERT_ASSERT_OK(LiteRtGetModelSignature(&model, 0, &signature));

    const char* key;
    LITERT_ASSERT_OK(LiteRtGetSignatureKey(signature, &key));
    EXPECT_STREQ(key, "sig1");

    LiteRtTensor sig_input_tensor;
    LITERT_ASSERT_OK(
        LiteRtGetSignatureInputTensorByIndex(signature, 0, &sig_input_tensor));
    EXPECT_EQ(sig_input_tensor, &input_tensor1);

    LiteRtTensor sig_output_tensor;
    LITERT_ASSERT_OK(LiteRtGetSignatureOutputTensorByIndex(signature, 0,
                                                           &sig_output_tensor));
    EXPECT_EQ(sig_output_tensor, &output_tensor1);
  }

  // Check second signature.
  {
    LiteRtSignature signature;
    LITERT_ASSERT_OK(LiteRtGetModelSignature(&model, 1, &signature));

    const char* key;
    LITERT_ASSERT_OK(LiteRtGetSignatureKey(signature, &key));
    EXPECT_STREQ(key, "sig2");

    LiteRtTensor sig_input_tensor;
    LITERT_ASSERT_OK(
        LiteRtGetSignatureInputTensorByIndex(signature, 0, &sig_input_tensor));
    EXPECT_EQ(sig_input_tensor, &input_tensor2);

    LiteRtTensor sig_output_tensor;
    LITERT_ASSERT_OK(LiteRtGetSignatureOutputTensorByIndex(signature, 0,
                                                           &sig_output_tensor));
    EXPECT_EQ(sig_output_tensor, &output_tensor2);
  }

  // Clean up buffer if serialization succeeded.
  free(buf);
}

TEST(LiteRtOpListTest, PushOps) {
  LiteRtOpListT op_list;
  LiteRtOpT op;

  LITERT_ASSERT_OK(LiteRtPushOp(&op_list, &op, 0));
  auto vec = op_list.Values();
  ASSERT_EQ(vec.size(), 1);
  EXPECT_EQ(vec.front().first, &op);
}
TEST(LiteRtModelTest, TestCheckSameUnrankedType) {
  LiteRtUnrankedTensorType type1;
  type1.element_type = kLiteRtElementTypeFloat32;
  LiteRtUnrankedTensorType type2;
  type2.element_type = kLiteRtElementTypeFloat32;
  EXPECT_TRUE(LiteRtIsSameUnrankedTensorType(&type1, &type2));

  LiteRtUnrankedTensorType type3;
  type3.element_type = kLiteRtElementTypeFloat32;
  LiteRtUnrankedTensorType type4;
  type4.element_type = kLiteRtElementTypeFloat16;
  EXPECT_FALSE(LiteRtIsSameUnrankedTensorType(&type3, &type4));
}

}  // namespace
