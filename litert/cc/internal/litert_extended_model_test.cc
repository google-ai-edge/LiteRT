// Copyright 2025 Google LLC.
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

#include "litert/cc/internal/litert_extended_model.h"

#include <cstdint>
#include <memory>
#include <optional>
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
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

// Tests for CC Wrapper classes around public C api.

namespace litert {

namespace {

//===----------------------------------------------------------------------===//
//                                CC Op                                       //
//===----------------------------------------------------------------------===//

TEST(CcOpTest, SimpleSupportedOp) {
  auto litert_model = testing::LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();
  const auto ops = subgraph->Ops();
  const auto& op = ops.front();

  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflMul);
  EXPECT_EQ(op.Inputs().size(), 2);
  EXPECT_EQ(op.Outputs().size(), 1);
  EXPECT_FALSE(op.CustomCode().HasValue());
}

TEST(CcOpTest, CustomCode) {
  auto litert_model = testing::LoadTestFileModel("MTKEXT_CONV_2D.tflite");
  auto subgraph = litert_model.MainSubgraph();
  const auto ops = subgraph->Ops();
  const auto& op = ops.front();
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflCustom);
  LITERT_ASSERT_OK_AND_ASSIGN(auto custom_code, op.CustomCode());
  EXPECT_EQ(custom_code, "MTKEXT_CONV_2D");
}

//===----------------------------------------------------------------------===//
//                                CC Tensor                                   //
//===----------------------------------------------------------------------===//

TEST(CcTensorTest, SimpleModel) {
  auto litert_model = testing::LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();

  auto inputs = subgraph->Inputs();
  ASSERT_EQ(inputs.size(), 2);

  {
    const Tensor& input_tensor = inputs.front();
    ASSERT_EQ(input_tensor.TypeId(), kLiteRtRankedTensorType);

    auto input_ranked_tensor_type = input_tensor.RankedTensorType();
    EXPECT_TRUE(input_ranked_tensor_type);
    ASSERT_EQ(input_ranked_tensor_type->ElementType(), ElementType::Float32);

    EXPECT_FALSE(input_tensor.HasWeights());

    auto input_weights = input_tensor.Weights();
    ASSERT_EQ(input_weights.Bytes().size(), 0);
    ASSERT_EQ(input_weights.BufferId(), 1);

    ASSERT_EQ(input_tensor.DefiningOp(), std::nullopt);

    const auto uses = input_tensor.Uses();
    ASSERT_EQ(uses.size(), 1);
  }

  auto outputs = subgraph->Outputs();
  ASSERT_EQ(outputs.size(), 1);

  {
    const Tensor& output_tensor = outputs.front();
    ASSERT_EQ(output_tensor.TypeId(), kLiteRtRankedTensorType);

    auto output_defining_op = output_tensor.DefiningOp();
    EXPECT_TRUE(output_defining_op.has_value());

    ASSERT_TRUE(output_tensor.Uses().empty());
  }
}

TEST(CcTensorTest, WeightsData) {
  auto litert_model = testing::LoadTestFileModel("add_cst.tflite");
  auto subgraph = litert_model.MainSubgraph();

  auto data = subgraph->Ops().front().Inputs().back().WeightsData<float>();
  ASSERT_TRUE(data.HasValue());
  EXPECT_THAT(data.Value(), ::testing::ElementsAreArray({1.0, 2.0, 3.0, 4.0}));
}

TEST(CcTensorTest, Name) {
  constexpr absl::string_view kName = "foo";
  LiteRtTensorT tensor;
  tensor.SetName(std::string(kName));

  Tensor cc_tensor(&tensor);
  EXPECT_EQ(cc_tensor.Name(), kName);
}

TEST(CcTensorTest, Index) {
  constexpr std::uint32_t kIndex = 1;
  LiteRtTensorT tensor;
  tensor.SetTensorIndex(kIndex);

  Tensor cc_tensor(&tensor);
  EXPECT_EQ(cc_tensor.TensorIndex(), kIndex);
}

TEST(CcTensorTest, QuantizationNone) {
  LiteRtTensorT litert_tensor;
  litert_tensor.Qparams().first = kLiteRtQuantizationNone;

  Tensor tensor(&litert_tensor);
  EXPECT_EQ(tensor.QTypeId(), kLiteRtQuantizationNone);
  EXPECT_FALSE(tensor.HasQuantization());
}

TEST(CcTensorTest, QuantizationPerTensor) {
  constexpr auto kScale = 1.0;
  constexpr auto kZeroPoint = 1;

  LiteRtTensorT litert_tensor;
  litert_tensor.SetQarams(MakePerTensorQuantization(kScale, kZeroPoint));

  Tensor tensor(&litert_tensor);
  ASSERT_EQ(tensor.QTypeId(), kLiteRtQuantizationPerTensor);
  ASSERT_TRUE(tensor.HasQuantization());

  const auto per_tensor_quantization = tensor.PerTensorQuantization();
  EXPECT_EQ(per_tensor_quantization.scale, kScale);
  EXPECT_EQ(per_tensor_quantization.zero_point, kZeroPoint);
}

TEST(CcTensorTest, QuantizationPerChannel) {
  constexpr auto kNumChannels = 2;
  constexpr auto kQuantizedDimension = 0;
  constexpr float kScales[kNumChannels] = {1.0, 2.0};
  constexpr int64_t kZeroPoints[kNumChannels] = {0, 0};

  LiteRtTensorT litert_tensor;
  auto per_channel = MakePerChannelQuantization(
      kScales, kZeroPoints, kQuantizedDimension, litert_tensor);
  litert_tensor.SetQarams(per_channel);

  Tensor tensor(&litert_tensor);
  ASSERT_EQ(tensor.QTypeId(), kLiteRtQuantizationPerChannel);
  ASSERT_TRUE(tensor.HasQuantization());

  const auto per_channel_quantization = tensor.PerChannelQuantization();
  EXPECT_THAT(
      absl::MakeConstSpan(per_channel_quantization.scales, kNumChannels),
      ::testing::ElementsAreArray(kScales));
  EXPECT_THAT(
      absl::MakeConstSpan(per_channel_quantization.zero_points, kNumChannels),
      ::testing::ElementsAreArray(kZeroPoints));
  EXPECT_EQ(per_channel_quantization.num_channels, kNumChannels);
  EXPECT_EQ(per_channel_quantization.quantized_dimension, kQuantizedDimension);
}

TEST(CcTensorTest, ZeroSizeTensorTest) {
  auto litert_model = testing::LoadTestFileModel("scala_reshape.tflite");
  auto subgraph = litert_model.MainSubgraph();
  const auto ops = subgraph->Ops();
  const auto& op = ops.front();
  EXPECT_FALSE(op.Inputs().at(1).IsSubgraphInput());
}

//===----------------------------------------------------------------------===//
//                               CC Subgraph                                  //
//===----------------------------------------------------------------------===//

TEST(CcSubgraphTest, SimpleModel) {
  auto model = testing::LoadTestFileModel("one_mul.tflite");
  auto subgraph = model.MainSubgraph();

  ASSERT_EQ(subgraph->Inputs().size(), 2);
  ASSERT_EQ(subgraph->Outputs().size(), 1);
  ASSERT_EQ(subgraph->Ops().size(), 1);

  auto input0_tensor = subgraph->Input("arg0");
  ASSERT_TRUE(input0_tensor.HasValue());
  auto input1_tensor = subgraph->Input("arg1");
  ASSERT_TRUE(input1_tensor.HasValue());

  auto output_tensor = subgraph->Output("tfl.mul");
  ASSERT_TRUE(output_tensor.HasValue());
  ASSERT_EQ(output_tensor->TypeId(), kLiteRtRankedTensorType);
  auto output_ranked_tensor_type = output_tensor->RankedTensorType();
  EXPECT_TRUE(output_ranked_tensor_type);
  ASSERT_EQ(output_ranked_tensor_type->ElementType(), ElementType::Float32);
}

//===----------------------------------------------------------------------===//
//                               CC Model                                     //
//===----------------------------------------------------------------------===//

TEST(CcModelTest, AddMetadataSuccess) {
  auto model = testing::LoadTestFileModel("one_mul.tflite");
  constexpr absl::string_view kKey = "KEY";
  constexpr absl::string_view kData = "DATA";
  LITERT_ASSERT_OK(model.AddMetadata(kKey.data(), kData.data()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto metadata, model.Metadata(kKey.data()));
  EXPECT_EQ(absl::string_view(reinterpret_cast<const char*>(metadata.data()),
                              metadata.size()),
            kData);
}

TEST(CcModelTest, AddMetadataGetMetadataOutsideOfScopeSuccess) {
  auto model = testing::LoadTestFileModel("one_mul.tflite");
  constexpr absl::string_view kExpectedKey = "KEY";
  constexpr absl::string_view kExpectedData = "DATA";
  {
    constexpr absl::string_view kKey = "KEY";
    constexpr absl::string_view kData = "DATA";
    LITERT_ASSERT_OK(model.AddMetadata(kKey.data(), kData.data()));
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto metadata,
                              model.Metadata(kExpectedKey.data()));
  EXPECT_EQ(absl::string_view(reinterpret_cast<const char*>(metadata.data()),
                              metadata.size()),
            kExpectedData);
}

TEST(CcModelTest, SerializeModelSuccess) {
  Expected<std::unique_ptr<internal::FlatbufferWrapper>> flatbuffer =
      internal::FlatbufferWrapper::CreateFromTflFile(
          testing::GetTestFilePath("one_mul.tflite"));

  ExtendedModel model = testing::LoadTestFileModel("one_mul.tflite");
  LiteRtModelSerializationOptions serialization_options =
      SerializationOptions::Defaults();
  serialization_options.bytecode_alignment = 1;
  Expected<OwningBufferRef<uint8_t>> serialized =
      ExtendedModel::Serialize(std::move(model), serialization_options);
  ASSERT_TRUE(serialized.HasValue());
  EXPECT_GT(serialized->Size(), 0);
  // TODO:(yunandrew) add check for serialized size
}

TEST(CcModelTest, SerializePreCompiledModelHasSameSizeAsOriginal) {
  Expected<std::unique_ptr<internal::FlatbufferWrapper>> flatbuffer =
      internal::FlatbufferWrapper::CreateFromTflFile(
          testing::GetTestFilePath("simple_add_op.sm8750.tflite"));

  ExtendedModel model =
      testing::LoadTestFileModel("simple_add_op.sm8750.tflite");
  LiteRtModelSerializationOptions serialization_options =
      SerializationOptions::Defaults();
  serialization_options.bytecode_alignment = 1;
  Expected<OwningBufferRef<uint8_t>> serialized =
      ExtendedModel::Serialize(std::move(model), serialization_options);
  ASSERT_TRUE(serialized.HasValue());
  EXPECT_EQ(serialized->Size(), flatbuffer->get()->Buf().Size());
}

}  // namespace
}  // namespace litert
