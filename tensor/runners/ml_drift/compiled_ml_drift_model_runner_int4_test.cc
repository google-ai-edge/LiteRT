/* Copyright 2026 Google LLC.

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

// Host-only tests (no GPU) of the INT4 FullyConnected weight preparation of
// `CompiledMlDriftModelRunner`: the packed rearrangement must produce the same
// bytes and `sum_i` as unpacking to INT8 and rearranging that.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "tensor/runners/ml_drift/compiled_ml_drift_model_runner.h"
#include "tensor/runners/ml_drift/ml_drift_dummy_weight_creator.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::ml_drift_runner_internal {
namespace {

using ::testing::ElementsAreArray;

struct Int4Case {
  int o;
  int i;
  ml_drift::WeightsLayout layout;
};

class RearrangeInt4WeightsTest : public ::testing::TestWithParam<Int4Case> {};

// Dense signed INT4 values (two per byte, low nibble first) covering -8..7.
std::vector<int8_t> PackedInt4(int num_elements) {
  std::vector<int8_t> packed((num_elements + 1) / 2, 0);
  for (int k = 0; k < num_elements; ++k) {
    const int value = (k * 5 + 3) % 16 - 8;
    const uint8_t nibble = static_cast<uint8_t>(value) & 0xF;
    uint8_t& byte = reinterpret_cast<uint8_t&>(packed[k / 2]);
    byte |= (k % 2 == 0) ? nibble : static_cast<uint8_t>(nibble << 4);
  }
  return packed;
}

TEST_P(RearrangeInt4WeightsTest, MatchesUnpackingToInt8) {
  const Int4Case& c = GetParam();
  const ml_drift::OHWI ohwi(c.o, 1, 1, c.i);
  ml_drift::WeightsDescription desc;
  desc.type = ml_drift::DataType::kUint4;
  desc.layout = c.layout;
  desc.output_group_size = ml_drift::DivideRoundUp(c.o, 4);
  const int elements_count =
      ml_drift::GetTotalElementsCountForLayout(desc, ohwi);
  const int num_elements = c.o * c.i;
  const std::vector<int8_t> packed = PackedInt4(num_elements);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const RearrangedInt4Weights actual,
      RearrangeInt4Weights(packed, ohwi, desc, elements_count,
                           /*want_sum_i=*/true));

  // The previous path: unpack to INT8, rearrange, sum the unpacked rows.
  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> unpacked;
  unpacked.shape = ohwi;
  unpacked.data.assign(num_elements + 64, 0);
  ml_drift::UnpackDenseInt4IntoInt8(packed.data(), num_elements,
                                    unpacked.data.data());
  std::vector<uint8_t> expected((static_cast<size_t>(elements_count) * 4 + 7) /
                                8);
  ml_drift::RearrangeWeightsInt8AsUint4(unpacked, desc,
                                        absl::MakeSpan(expected), 8, 8u);
  EXPECT_THAT(actual.data, ElementsAreArray(expected));
  ASSERT_TRUE(actual.sum_i.has_value());
  const absl::Span<const int8_t> unpacked_rows =
      absl::MakeConstSpan(unpacked.data).first(num_elements);
  EXPECT_THAT(*actual.sum_i,
              ElementsAreArray(ComputeWeightsSumI(ohwi, unpacked_rows)));
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, RearrangeInt4WeightsTest,
    ::testing::Values(
        // Odd input sizes (rows straddle bytes) and output counts that are not
        // a multiple of 4, for both packed layouts.
        Int4Case{6, 7, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int4Case{5, 9, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int4Case{8, 32, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int4Case{6, 12, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int4Case{6, 7,
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
        Int4Case{5, 9,
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
        Int4Case{8, 32,
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
        Int4Case{6, 12,
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4}),
    [](const ::testing::TestParamInfo<Int4Case>& info) {
      return absl::StrCat(
          info.param.layout == ml_drift::WeightsLayout::kOSpatialIOGroupI4O4
              ? "Buffer"
              : "Texture",
          "O", info.param.o, "I", info.param.i);
    });

TEST(RearrangeInt4WeightsTest, RejectsAShortBuffer) {
  const ml_drift::OHWI ohwi(4, 1, 1, 8);
  ml_drift::WeightsDescription desc;
  desc.type = ml_drift::DataType::kUint4;
  desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  desc.output_group_size = 1;
  const std::vector<int8_t> packed(15, 0);  // 16 bytes needed.
  EXPECT_THAT(
      RearrangeInt4Weights(packed, ohwi, desc,
                           ml_drift::GetTotalElementsCountForLayout(desc, ohwi),
                           /*want_sum_i=*/false),
      ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

// GPU weight conversion (`SetConvertWeightsOnGpu`) writes ML Drift's canonical
// weight tensors (`GetTensorDescriptorsForWeightsLayout`) in place of the CPU
// packed bytes: they must hold exactly the packed INT4 bytes.
TEST_P(RearrangeInt4WeightsTest, CanonicalTensorHoldsThePackedBytes) {
  const Int4Case& c = GetParam();
  if (c.i % 4 != 0) GTEST_SKIP() << "Kept on the CPU path";
  const ml_drift::OHWI ohwi(c.o, 1, 1, c.i);
  ml_drift::WeightsDescription desc;
  desc.type = ml_drift::DataType::kUint4;
  desc.layout = c.layout;
  desc.output_group_size = ml_drift::DivideRoundUp(c.o, 4);
  const int elements_count =
      ml_drift::GetTotalElementsCountForLayout(desc, ohwi);
  ASSERT_OK_AND_ASSIGN(
      const RearrangedInt4Weights packed,
      RearrangeInt4Weights(PackedInt4(c.o * c.i), ohwi, desc, elements_count,
                           /*want_sum_i=*/false));
  const std::vector<ml_drift::TensorDescriptor> tensors =
      ml_drift::GetTensorDescriptorsForWeightsLayout(ohwi, desc);
  ASSERT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].GetMemorySizeInBytes(), packed.data.size());

  const ml_drift::TensorDescriptor packed_td =
      CreatePackedWeightsTensorDescriptor(desc, ohwi, elements_count,
                                          /*is_embedding=*/false);
  EXPECT_EQ(packed_td.GetMemorySizeInBytes(), packed.data.size());
  const ml_drift::TensorDescriptor emb_td = CreatePackedWeightsTensorDescriptor(
      desc, ohwi, elements_count, /*is_embedding=*/true);
  EXPECT_EQ(emb_td.GetMemorySizeInBytes(), packed.data.size());
  // Linear buffers hold exactly the packed bytes (`GetMemorySizeInBytes()`
  // rounds up to whole 4-channel slices).
  for (const ml_drift::TensorDescriptor* td : {&packed_td, &emb_td}) {
    EXPECT_EQ(td->GetStorageType(),
              desc.IsLinearLayout() ? ml_drift::TensorStorageType::kBuffer
                                    : ml_drift::TensorStorageType::kTexture2D);
    if (td->GetStorageType() == ml_drift::TensorStorageType::kBuffer) {
      EXPECT_EQ(static_cast<size_t>(td->GetBHWDCShape().c), packed.data.size());
    }
  }
}

// Embedding weights keep a linear buffer for 2D texture layouts other than
// `k2DYIsSpatialIOAndXIsOGroupI4O4`; other weights use a 2D texture.
TEST(CreatePackedWeightsTensorDescriptorTest,
     EmbeddingsUseABufferForUnsupportedTextureLayouts) {
  const ml_drift::OHWI ohwi(8, 1, 1, 32);
  ml_drift::WeightsDescription desc;
  desc.type = ml_drift::DataType::kUint4;
  desc.layout = ml_drift::WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  desc.output_group_size = ml_drift::DivideRoundUp(ohwi.o, 4);
  ASSERT_FALSE(desc.IsLinearLayout());
  const size_t elements_count =
      static_cast<size_t>(ml_drift::GetTotalElementsCountForLayout(desc, ohwi));

  const ml_drift::TensorDescriptor emb_td = CreatePackedWeightsTensorDescriptor(
      desc, ohwi, elements_count, /*is_embedding=*/true);
  EXPECT_EQ(emb_td.GetStorageType(), ml_drift::TensorStorageType::kBuffer);
  EXPECT_EQ(static_cast<size_t>(emb_td.GetBHWDCShape().c),
            (elements_count * 4 + 7) / 8);

  const ml_drift::TensorDescriptor fc_td = CreatePackedWeightsTensorDescriptor(
      desc, ohwi, elements_count, /*is_embedding=*/false);
  EXPECT_EQ(fc_td.GetStorageType(), ml_drift::TensorStorageType::kTexture2D);
}

// Linear buffers of 8-bit and wider types hold one element per channel.
TEST(CreatePackedWeightsTensorDescriptorTest, WideTypesAreNotPacked) {
  const ml_drift::OHWI ohwi(8, 1, 1, 32);
  for (const ml_drift::DataType type :
       {ml_drift::DataType::kFloat16, ml_drift::DataType::kFloat32}) {
    ml_drift::WeightsDescription desc;
    desc.type = type;
    desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
    desc.output_group_size = ml_drift::DivideRoundUp(ohwi.o, 4);
    ASSERT_TRUE(desc.IsLinearLayout());
    const size_t elements_count = static_cast<size_t>(
        ml_drift::GetTotalElementsCountForLayout(desc, ohwi));
    const ml_drift::TensorDescriptor td = CreatePackedWeightsTensorDescriptor(
        desc, ohwi, elements_count, /*is_embedding=*/false);
    EXPECT_EQ(td.GetStorageType(), ml_drift::TensorStorageType::kBuffer);
    EXPECT_EQ(static_cast<size_t>(td.GetBHWDCShape().c), elements_count);
  }
}

struct Int8Case {
  int o;
  int i;
  int output_group_size;
  ml_drift::WeightsLayout layout;
};

class RearrangeInt8WeightsTest : public ::testing::TestWithParam<Int8Case> {};

TEST_P(RearrangeInt8WeightsTest, DirectFastPathMatchesMlDriftReference) {
  const Int8Case& c = GetParam();
  const ml_drift::OHWI ohwi(c.o, 1, 1, c.i);
  ml_drift::WeightsDescription desc;
  desc.type = ml_drift::DataType::kUint8;
  desc.layout = c.layout;
  desc.output_group_size = c.output_group_size;
  const size_t elements_count =
      static_cast<size_t>(ml_drift::GetTotalElementsCountForLayout(desc, ohwi));

  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> weights;
  weights.shape = ohwi;
  weights.data.resize(static_cast<size_t>(c.o) * c.i);
  for (size_t k = 0; k < weights.data.size(); ++k) {
    weights.data[k] = static_cast<int8_t>((k * 37 + 11) % 256 - 128);
  }

  std::vector<uint8_t> expected(elements_count, 0xA5u);
  ml_drift::RearrangeWeightsInt8AsUint8(weights, desc, absl::MakeSpan(expected),
                                        128, 128u);

  std::vector<uint8_t> actual(elements_count, 0x5Au);
  ASSERT_TRUE(TryDirectRearrangeInt8AsUint8I4O4(weights.data.data(), ohwi, desc,
                                                absl::MakeSpan(actual)));
  EXPECT_EQ(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
    ShapesAndLayouts, RearrangeInt8WeightsTest,
    ::testing::Values(
        // Non-multiple-of-4 dimensions across both layouts and group sizes.
        Int8Case{6, 7, 1, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int8Case{5, 9, 2, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int8Case{13, 19, 3, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int8Case{6, 7, 1,
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
        Int8Case{5, 9, 2,
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
        // >= 512K elements with non-multiple-of-4 boundaries to cover the
        // `ParallelFor` path in both layouts.
        Int8Case{515, 1027, 4, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int8Case{515, 1027, 4,
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4}),
    [](const ::testing::TestParamInfo<Int8Case>& info) {
      return absl::StrCat(
          info.param.layout == ml_drift::WeightsLayout::kOSpatialIOGroupI4O4
              ? "Buffer"
              : "Texture",
          "O", info.param.o, "I", info.param.i, "G",
          info.param.output_group_size);
    });

TEST(RearrangeInt8WeightsTest, ReturnsFalseForUnsupportedShapeOrSmallBuffer) {
  const ml_drift::OHWI ohwi(8, 1, 1, 8);
  ml_drift::WeightsDescription desc;
  desc.type = ml_drift::DataType::kUint8;
  desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  desc.output_group_size = 1;
  const std::vector<int8_t> weights(64, 1);
  std::vector<uint8_t> short_dst(16, 0);
  EXPECT_FALSE(TryDirectRearrangeInt8AsUint8I4O4(
      weights.data(), ohwi, desc, absl::MakeSpan(short_dst)));
  std::vector<uint8_t> valid_dst(64, 0);
  EXPECT_FALSE(TryDirectRearrangeInt8AsUint8I4O4(
      weights.data(), ml_drift::OHWI(8, 2, 1, 8), desc,
      absl::MakeSpan(valid_dst)));
}

}  // namespace
}  // namespace litert::tensor::ml_drift_runner_internal
