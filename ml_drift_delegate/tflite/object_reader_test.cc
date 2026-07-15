// Copyright 2025 Google LLC.
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

#include "ml_drift_delegate/tflite/object_reader.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "ml_drift_delegate/tflite/stub_tflite_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/testing/matchers.h"
#include "tflite/util.h"

using ::testing::Not;
using ::testing::tflite::SimpleConstTensor;

namespace litert::ml_drift {

template <typename Sink>
void AbslStringify(Sink& sink,
                   const ObjectReader::ConstantInputSharingInfo& i) {
  if (i.IsShared()) {
    std::string external_repr = i.HasExternalBufferId()
                                    ? absl::StrCat(i.external_buffer_id)
                                    : std::string("none");
    absl::Format(&sink, "{Shared buffer: %zu, external: %s}", i.buffer_id,
                 external_repr.c_str());
  } else {
    absl::Format(&sink, "{Not shared}");
  }
}

namespace {

MATCHER(IsShared, (negation ? "isn't shared" : "is shared")) {
  return arg.IsShared();
}

TEST(IsSharedMatcherTest, Works) {
  EXPECT_THAT(ObjectReader::ConstantInputSharingInfo::BuildNotShared(),
              Not(IsShared()));
  EXPECT_THAT(ObjectReader::ConstantInputSharingInfo{.buffer_id = 0},
              IsShared());
  EXPECT_THAT(ObjectReader::ConstantInputSharingInfo{.external_buffer_id = 0},
              IsShared());
}

TEST(ObjectBuilderBasicTest, Construction) {
  auto context = std::make_unique<StubTfLiteContext>(
      kTfLiteBuiltinAdd, /*op_version=*/3, /*num_inputs=*/2,
      /*shape=*/std::vector<int>({1, 1, 1, 1}));

  ::ml_drift::GraphFloat32 graph;
  absl::flat_hash_map<int, ::ml_drift::Value*> tensor_to_value;

  ObjectReader reader(&graph, context.get(), context->node(), &tensor_to_value,
                      /*quant_conversion_map=*/nullptr,
                      /*tensor_to_buffer_id_map=*/nullptr,
                      /*tensor_to_external_buffer_id_map=*/nullptr,
                      /*shared_tensor_map=*/nullptr);
}

class ObjectBuilderSharingTest : public testing::Test {
 public:
  enum { kPrivateTensorIdx = 1, kSharedTensorIdx = 2, kInvalidTensorIdx = 100 };
  enum { kPrivateInputPos = 0, kSharedInputPos = 1, kInvalidInputPos = 101 };
  enum { kGlobalBufferId = 4 };

 protected:
  std::unique_ptr<StubTfLiteContext> context_ =
      std::make_unique<StubTfLiteContext>(
          kTfLiteBuiltinAdd, /*op_version=*/3,
          /*num_inputs=*/2,
          /*shape=*/std::vector<int>({1, 1, 1, 1}));
  ::ml_drift::GraphFloat32 graph_;
  absl::flat_hash_map<int, ::ml_drift::Value*> tensor_to_value_;
  TensorIndexToBufferIdMap tensor_to_buffer_id_map_ = {
      {kSharedTensorIdx, kGlobalBufferId}};
  SharedConstTensorsMap shared_tensor_map_;
  ObjectReader reader_ = ObjectReader(
      &graph_, context_.get(), context_->node(), &tensor_to_value_,
      /*quant_conversion_map=*/nullptr, &tensor_to_buffer_id_map_,
      /*tensor_to_external_buffer_id_map=*/nullptr, &shared_tensor_map_);
};

TEST_F(ObjectBuilderSharingTest,
       SharingEnabledWhenBuiltWithBufferMapAndSharedTensorsMap) {
  EXPECT_TRUE(reader_.SharingEnabled());
  reader_.AllowSharingInput(kSharedInputPos);
  EXPECT_THAT(ObjectReader::ConstantInputSharingInfo::BuildNotShared(),
              Not(IsShared()));
}

TEST_F(ObjectBuilderSharingTest, SharingDisabledIfSharedTensorsMapIsNull) {
  reader_ =
      ObjectReader(&graph_, context_.get(), context_->node(), &tensor_to_value_,
                   /*quant_conversion_map=*/nullptr, &tensor_to_buffer_id_map_,
                   /*tensor_to_external_buffer_id_map=*/nullptr,
                   /*shared_tensor_map=*/nullptr);
  EXPECT_FALSE(reader_.SharingEnabled());
}

TEST_F(ObjectBuilderSharingTest, SharingDisabledIfBufferMapIsNull) {
  reader_ = ObjectReader(
      &graph_, context_.get(), context_->node(), &tensor_to_value_,
      /*quant_conversion_map=*/nullptr,
      /*tensor_to_buffer_id_map=*/nullptr,
      /*tensor_to_external_buffer_id_map=*/nullptr, &shared_tensor_map_);
  EXPECT_FALSE(reader_.SharingEnabled());
}

TEST_F(ObjectBuilderSharingTest, SharingDisabledByTensorIndexReturnsNotShared) {
  reader_ =
      ObjectReader(&graph_, context_.get(), context_->node(), &tensor_to_value_,
                   /*quant_conversion_map=*/nullptr,
                   /*tensor_to_buffer_id_map=*/nullptr,
                   /*tensor_to_external_buffer_id_map=*/nullptr,
                   /*shared_tensor_map=*/nullptr);
  reader_.AllowSharingInput(kSharedInputPos);
  EXPECT_THAT(reader_.GetSharingInfoByTensorIndex(kSharedTensorIdx),
              Not(IsShared()));
}

TEST_F(ObjectBuilderSharingTest, SharedTensorByTensorIndexReturnsShared) {
  reader_.AllowSharingInput(kSharedInputPos);
  EXPECT_THAT(reader_.GetSharingInfoByTensorIndex(kSharedTensorIdx),
              IsShared());
}

TEST_F(ObjectBuilderSharingTest,
       NotAllowedSharedTensorByTensorIndexReturnsNotShared) {
  // kSharedTensor has not been explicitely marked shared.
  EXPECT_THAT(reader_.GetSharingInfoByTensorIndex(kSharedTensorIdx),
              Not(IsShared()));
}

TEST_F(ObjectBuilderSharingTest,
       AllowedPrivateTensorByTensorIndexReturnsNotShared) {
  reader_.AllowSharingInput(kPrivateInputPos);
  EXPECT_THAT(reader_.GetSharingInfoByTensorIndex(kPrivateTensorIdx),
              Not(IsShared()));
}

TEST_F(ObjectBuilderSharingTest, SharingDisabledByInputPosReturnsNotShared) {
  reader_ =
      ObjectReader(&graph_, context_.get(), context_->node(), &tensor_to_value_,
                   /*quant_conversion_map=*/nullptr,
                   /*tensor_to_buffer_id_map=*/nullptr,
                   /*tensor_to_external_buffer_id_map=*/nullptr,
                   /*shared_tensor_map=*/nullptr);
  reader_.AllowSharingInput(kSharedInputPos);
  EXPECT_THAT(reader_.GetSharingInfoByNodeInputIndex(kSharedInputPos),
              Not(IsShared()));
}

TEST_F(ObjectBuilderSharingTest, SharedTensorByInputPosReturnsShared) {
  reader_.AllowSharingInput(kSharedInputPos);
  EXPECT_THAT(reader_.GetSharingInfoByNodeInputIndex(kSharedInputPos),
              IsShared());
}

TEST_F(ObjectBuilderSharingTest,
       NotAllowedSharedTensorByInputPosReturnsNotShared) {
  // kSharedTensor has not been explicitely marked shared.
  EXPECT_THAT(reader_.GetSharingInfoByNodeInputIndex(kSharedInputPos),
              Not(IsShared()));
}

TEST_F(ObjectBuilderSharingTest,
       AllowedPrivateTensorByInputPosReturnsNotShared) {
  reader_.AllowSharingInput(kPrivateInputPos);
  EXPECT_THAT(reader_.GetSharingInfoByNodeInputIndex(kPrivateInputPos),
              Not(IsShared()));
}

TEST_F(ObjectBuilderSharingTest, SetSharedTensorWorks) {
  reader_.AllowSharingInput(kSharedInputPos);
  constexpr ::ml_drift::ValueId kFakeGraphNodeId = 5;
  reader_.SetSharedTensor(kFakeGraphNodeId, kGlobalBufferId, kSharedTensorIdx,
                          /*dequant_forced=*/false, /*layout=*/std::nullopt);
  ASSERT_EQ(shared_tensor_map_.size(), 1);
  ASSERT_TRUE(shared_tensor_map_.contains(kFakeGraphNodeId));

  const SharedTfliteTensor kExpected{.tflite_tensor_id = kSharedTensorIdx,
                                     .global_id = kGlobalBufferId,
                                     .dequant_forced = false};
  EXPECT_EQ(shared_tensor_map_.at(kFakeGraphNodeId), kExpected);
}

TEST(TfLiteTensorToTensorTest, Float2D) {
  float data[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  SimpleConstTensor tfl_tensor(TfLiteType::kTfLiteFloat32, {1, 5},
                               absl::MakeSpan(data));
  ::ml_drift::TensorFloat32 t;
  TfLiteTensorToTensorCopyData(&tfl_tensor, &t, ReadTensorFlags::kNoExtraBytes);
  EXPECT_EQ(t.shape, ::ml_drift::BHWC(1, 1, 1, 5));
}

TEST(TfLiteTensorToTensorTest, FloatScalar) {
  float data[1] = {1.0};
  SimpleConstTensor tfl_tensor(TfLiteType::kTfLiteFloat32, {},
                               absl::MakeSpan(data));
  ::ml_drift::TensorFloat32 t;
  TfLiteTensorToTensorCopyData(&tfl_tensor, &t, ReadTensorFlags::kNoExtraBytes);
  EXPECT_EQ(t.shape, ::ml_drift::BHWC(1, 1, 1, 1));
}

TEST(TfLiteTensorToTensorTest, Float162D) {
  ::ml_drift::half data[5] = {::ml_drift::half(1.0f), ::ml_drift::half(2.0f),
                              ::ml_drift::half(3.0f), ::ml_drift::half(4.0f),
                              ::ml_drift::half(5.0f)};
  TfLiteTensor tfl_tensor{};
  tfl_tensor.type = kTfLiteFloat16;
  tfl_tensor.dims = tflite::ConvertVectorToTfLiteIntArray({1, 5});
  tfl_tensor.data.raw_const = reinterpret_cast<const char*>(data);
  tfl_tensor.bytes = sizeof(data);
  tfl_tensor.allocation_type = kTfLiteMmapRo;

  ::ml_drift::TensorFloat16 t;
  TfLiteTensorToTensorCopyData(&tfl_tensor, &t, ReadTensorFlags::kNoExtraBytes);
  EXPECT_EQ(t.shape, ::ml_drift::BHWC(1, 1, 1, 5));
  EXPECT_EQ(t.data.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(static_cast<float>(t.data[i]), static_cast<float>(data[i]));
  }
  TfLiteIntArrayFree(tfl_tensor.dims);
}

TEST(TfLiteTensorToTensorTest, Float16Scalar) {
  ::ml_drift::half data[1] = {::ml_drift::half(1.0f)};
  TfLiteTensor tfl_tensor{};
  tfl_tensor.type = kTfLiteFloat16;
  tfl_tensor.dims = tflite::ConvertVectorToTfLiteIntArray({});
  tfl_tensor.data.raw_const = reinterpret_cast<const char*>(data);
  tfl_tensor.bytes = sizeof(data);
  tfl_tensor.allocation_type = kTfLiteMmapRo;

  ::ml_drift::TensorFloat16 t;
  TfLiteTensorToTensorCopyData(&tfl_tensor, &t, ReadTensorFlags::kNoExtraBytes);
  EXPECT_EQ(t.shape, ::ml_drift::BHWC(1, 1, 1, 1));
  EXPECT_EQ(t.data.size(), 1);
  EXPECT_EQ(static_cast<float>(t.data[0]), static_cast<float>(data[0]));
  TfLiteIntArrayFree(tfl_tensor.dims);
}

TEST(TfLiteTensorToTensorTest, Int8QuantizedPerTensor) {
  int8_t data[4] = {10, 20, 30, 40};
  TfLiteTensor tfl_tensor{};
  tfl_tensor.type = kTfLiteInt8;
  tfl_tensor.dims = tflite::ConvertVectorToTfLiteIntArray({1, 1, 1, 4});
  tfl_tensor.data.int8 = data;
  tfl_tensor.bytes = sizeof(data);
  tfl_tensor.allocation_type = kTfLiteArenaRw;

  tfl_tensor.quantization.type = kTfLiteAffineQuantization;
  auto* quant_params = static_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  quant_params->scale = TfLiteFloatArrayCreate(1);
  quant_params->scale->data[0] = 0.5f;
  quant_params->zero_point = TfLiteIntArrayCreate(1);
  quant_params->zero_point->data[0] = 5;
  quant_params->quantized_dimension = 0;
  tfl_tensor.quantization.params = quant_params;
  tfl_tensor.params.scale = 0.5f;
  tfl_tensor.params.zero_point = 5;

  // Test deep copy approach.
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8> t;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32> scale;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32> zero_point;
  TfLiteTensorToTensorCopyData(&tfl_tensor, &t, ReadTensorFlags::kNoExtraBytes,
                               &scale, &zero_point);
  EXPECT_EQ(t.shape, ::ml_drift::OHWI(1, 1, 1, 4));
  EXPECT_THAT(t.data, testing::ElementsAreArray(data));
  EXPECT_EQ(scale.data.size(), 1);
  EXPECT_EQ(scale.data[0], 0.5f);
  EXPECT_EQ(zero_point.data.size(), 1);
  EXPECT_EQ(zero_point.data[0], 5);

  // Test zero copy approach.
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8> t_view;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
      scale_view;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>
      zero_point_view;
  TfLiteTensorToTensorZeroCopy(&tfl_tensor, &t_view, &scale_view,
                               &zero_point_view);
  EXPECT_EQ(t_view.shape, ::ml_drift::OHWI(1, 1, 1, 4));
  EXPECT_THAT(t_view.spanned_data, testing::ElementsAreArray(data));
  EXPECT_EQ(scale_view.data.size(), 1);
  EXPECT_EQ(scale_view.data[0], 0.5f);
  EXPECT_EQ(zero_point_view.data.size(), 1);
  EXPECT_EQ(zero_point_view.data[0], 5);

  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteFloatArrayFree(quant_params->scale);
  TfLiteIntArrayFree(quant_params->zero_point);
  free(quant_params);
}

TEST(TfLiteTensorToTensorTest, Int8QuantizedPerChannel) {
  int8_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  TfLiteTensor tfl_tensor{};
  tfl_tensor.type = kTfLiteInt8;
  tfl_tensor.dims = tflite::ConvertVectorToTfLiteIntArray({2, 1, 1, 4});
  tfl_tensor.data.int8 = data;
  tfl_tensor.bytes = sizeof(data);
  tfl_tensor.allocation_type = kTfLiteArenaRw;

  tfl_tensor.quantization.type = kTfLiteAffineQuantization;
  auto* quant_params = static_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  float scales_data[] = {0.5f, 1.5f};
  int zero_points_data[] = {1, 2};
  quant_params->scale = TfLiteFloatArrayCreate(2);
  quant_params->scale->data[0] = scales_data[0];
  quant_params->scale->data[1] = scales_data[1];
  quant_params->zero_point = TfLiteIntArrayCreate(2);
  quant_params->zero_point->data[0] = zero_points_data[0];
  quant_params->zero_point->data[1] = zero_points_data[1];
  quant_params->quantized_dimension = 0;
  tfl_tensor.quantization.params = quant_params;

  // Test deep copy approach.
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8> t;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32> scale;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32> zero_point;
  TfLiteTensorToTensorCopyData(&tfl_tensor, &t, ReadTensorFlags::kNoExtraBytes,
                               &scale, &zero_point);
  EXPECT_EQ(t.shape, ::ml_drift::OHWI(2, 1, 1, 4));
  EXPECT_THAT(t.data, testing::ElementsAreArray(data));
  EXPECT_EQ(scale.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_THAT(scale.data, testing::ElementsAreArray(scales_data));
  EXPECT_EQ(zero_point.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_THAT(zero_point.data, testing::ElementsAreArray(zero_points_data));

  // Test zero copy approach.
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8> t_view;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
      scale_view;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>
      zero_point_view;
  TfLiteTensorToTensorZeroCopy(&tfl_tensor, &t_view, &scale_view,
                               &zero_point_view);

  EXPECT_EQ(t_view.shape, ::ml_drift::OHWI(2, 1, 1, 4));
  EXPECT_THAT(t_view.spanned_data, testing::ElementsAreArray(data));
  EXPECT_EQ(scale_view.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_THAT(scale_view.spanned_data, testing::ElementsAreArray(scales_data));
  EXPECT_EQ(zero_point_view.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_THAT(zero_point_view.spanned_data,
              testing::ElementsAreArray(zero_points_data));

  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteFloatArrayFree(quant_params->scale);
  TfLiteIntArrayFree(quant_params->zero_point);
  free(quant_params);
}

TEST(TfLiteTensorToTensorTest, Int4QuantizedPerTensor) {
  int8_t data[2] = {0x21, 0x43};  // Packed int4 data. {1, 2} -> 0x21, {3, 4}
  TfLiteTensor tfl_tensor{};
  tfl_tensor.type = kTfLiteInt4;
  tfl_tensor.dims = tflite::ConvertVectorToTfLiteIntArray({1, 1, 1, 4});
  tfl_tensor.data.int8 = data;
  tfl_tensor.bytes = sizeof(data);
  tfl_tensor.allocation_type = kTfLiteArenaRw;

  tfl_tensor.quantization.type = kTfLiteAffineQuantization;
  auto* quant_params = static_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  quant_params->scale = TfLiteFloatArrayCreate(1);
  quant_params->scale->data[0] = 0.5f;
  quant_params->zero_point = TfLiteIntArrayCreate(1);
  quant_params->zero_point->data[0] = -8;
  quant_params->quantized_dimension = 0;
  tfl_tensor.quantization.params = quant_params;
  tfl_tensor.params.scale = 0.5f;
  tfl_tensor.params.zero_point = -8;

  // Test deep copy approach.
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4> t;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32> scale;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32> zero_point;
  TfLiteTensorToTensorCopyData(&tfl_tensor, &t, ReadTensorFlags::kNoExtraBytes,
                               &scale, &zero_point);
  EXPECT_EQ(t.shape, ::ml_drift::OHWI(1, 1, 1, 4));
  EXPECT_EQ(t.data.size(), 2);
  EXPECT_EQ(t.data[0], data[0]);
  EXPECT_EQ(t.data[1], data[1]);

  EXPECT_EQ(scale.data.size(), 1);
  EXPECT_EQ(scale.data[0], 0.5f);
  EXPECT_EQ(zero_point.data.size(), 1);
  EXPECT_EQ(zero_point.data[0], -8);

  // Test zero copy approach.
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4> t_view;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
      scale_view;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>
      zero_point_view;
  TfLiteTensorToTensorZeroCopy(&tfl_tensor, &t_view, &scale_view,
                               &zero_point_view);

  EXPECT_EQ(t_view.shape, ::ml_drift::OHWI(1, 1, 1, 4));
  EXPECT_EQ(t_view.spanned_data.size(), 2);
  EXPECT_EQ(t_view.spanned_data[0], data[0]);
  EXPECT_EQ(t_view.spanned_data[1], data[1]);

  EXPECT_EQ(scale_view.data.size(), 1);
  EXPECT_EQ(scale_view.data[0], 0.5f);
  EXPECT_EQ(zero_point_view.data.size(), 1);
  EXPECT_EQ(zero_point_view.data[0], -8);

  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteFloatArrayFree(quant_params->scale);
  TfLiteIntArrayFree(quant_params->zero_point);
  free(quant_params);
}

TEST(TfLiteTensorToTensorTest, Int4QuantizedPerChannel) {
  int8_t data[4] = {0x21, 0x43, 0x65, 0x78};
  TfLiteTensor tfl_tensor{};
  tfl_tensor.type = kTfLiteInt4;
  tfl_tensor.dims = tflite::ConvertVectorToTfLiteIntArray({2, 1, 1, 4});
  tfl_tensor.data.int8 = data;
  tfl_tensor.bytes = sizeof(data);
  tfl_tensor.allocation_type = kTfLiteArenaRw;

  tfl_tensor.quantization.type = kTfLiteAffineQuantization;
  auto* quant_params = static_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  float scales_data[] = {0.5f, 1.5f};
  int zero_points_data[] = {-8, 0};
  quant_params->scale = TfLiteFloatArrayCreate(2);
  quant_params->scale->data[0] = scales_data[0];
  quant_params->scale->data[1] = scales_data[1];
  quant_params->zero_point = TfLiteIntArrayCreate(2);
  quant_params->zero_point->data[0] = zero_points_data[0];
  quant_params->zero_point->data[1] = zero_points_data[1];
  quant_params->quantized_dimension = 0;
  tfl_tensor.quantization.params = quant_params;

  // Test deep copy approach.
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4> t;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32> scale;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32> zero_point;
  TfLiteTensorToTensorCopyData(&tfl_tensor, &t, ReadTensorFlags::kNoExtraBytes,
                               &scale, &zero_point);
  EXPECT_EQ(t.shape, ::ml_drift::OHWI(2, 1, 1, 4));
  EXPECT_THAT(t.data, testing::ElementsAreArray(data));
  EXPECT_EQ(scale.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_THAT(scale.data, testing::ElementsAreArray(scales_data));
  EXPECT_EQ(zero_point.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_THAT(zero_point.data, testing::ElementsAreArray(zero_points_data));

  // Test zero copy approach.
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4> t_view;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
      scale_view;
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>
      zero_point_view;
  TfLiteTensorToTensorZeroCopy(&tfl_tensor, &t_view, &scale_view,
                               &zero_point_view);

  EXPECT_EQ(t_view.shape, ::ml_drift::OHWI(2, 1, 1, 4));
  EXPECT_THAT(t_view.spanned_data, testing::ElementsAreArray(data));
  EXPECT_EQ(scale_view.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_THAT(scale_view.spanned_data, testing::ElementsAreArray(scales_data));
  EXPECT_EQ(zero_point_view.shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_THAT(zero_point_view.spanned_data,
              testing::ElementsAreArray(zero_points_data));

  TfLiteIntArrayFree(tfl_tensor.dims);
  TfLiteFloatArrayFree(quant_params->scale);
  TfLiteIntArrayFree(quant_params->zero_point);
  free(quant_params);
}

}  // namespace
}  // namespace litert::ml_drift
