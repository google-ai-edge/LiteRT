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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
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
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
        // Large aligned shapes exercising the cache-tiled direct INT4 path.
        Int4Case{96, 160, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int4Case{96, 160,
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

TEST(RearrangeInt4WeightsTest, DirectTiledMatchesReferenceAcrossMultiGroups) {
  for (const ml_drift::WeightsLayout layout :
       {ml_drift::WeightsLayout::kOSpatialIOGroupI4O4,
        ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4}) {
    for (const int group_size : {1, 4, 8}) {
      // `o = 84` (21 slices) is not a multiple of `group_size * 4` for 4/8,
      // exercising padded trailing `d_group` slices inside multi-group mode.
      const ml_drift::OHWI ohwi(84, 1, 1, 160);
      ml_drift::WeightsDescription desc;
      desc.type = ml_drift::DataType::kUint4;
      desc.layout = layout;
      desc.output_group_size = group_size;
      const int elements_count =
          ml_drift::GetTotalElementsCountForLayout(desc, ohwi);
      const std::vector<int8_t> packed = PackedInt4(ohwi.o * ohwi.i);

      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          const RearrangedInt4Weights actual_with_sum,
          RearrangeInt4Weights(packed, ohwi, desc, elements_count,
                               /*want_sum_i=*/true));
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          const RearrangedInt4Weights actual_no_sum,
          RearrangeInt4Weights(packed, ohwi, desc, elements_count,
                               /*want_sum_i=*/false));
      EXPECT_FALSE(actual_no_sum.sum_i.has_value());
      EXPECT_THAT(actual_no_sum.data, ElementsAreArray(actual_with_sum.data));

      ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> unpacked;
      unpacked.shape = ohwi;
      unpacked.data.assign(ohwi.o * ohwi.i + 64, 0);
      ml_drift::UnpackDenseInt4IntoInt8(packed.data(), ohwi.o * ohwi.i,
                                        unpacked.data.data());
      std::vector<uint8_t> expected(
          (static_cast<size_t>(elements_count) * 4 + 7) / 8);
      ml_drift::RearrangeWeightsInt8AsUint4(unpacked, desc,
                                            absl::MakeSpan(expected), 8, 8u);
      EXPECT_THAT(actual_with_sum.data, ElementsAreArray(expected));
      ASSERT_TRUE(actual_with_sum.sum_i.has_value());
      EXPECT_THAT(*actual_with_sum.sum_i,
                  ElementsAreArray(ComputeWeightsSumI(
                      ohwi, absl::MakeConstSpan(unpacked.data)
                                .first(ohwi.o * ohwi.i))));
    }
  }
}

TEST(RearrangeInt8WeightsTest, DirectMatchesReferenceAcrossLayoutsAndGroups) {
  struct Int8Case {
    int o;
    int i;
    int group_size;
    ml_drift::WeightsLayout layout;
  };
  const std::vector<Int8Case> cases = {
      {6, 7, 2, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {5, 9, 2, ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
      {84, 160, 21, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {84, 160, 8, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {84, 160, 21, ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
      {84, 160, 8, ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
      {70, 99, 5, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {70, 99, 5, ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
  };
  for (const Int8Case& c : cases) {
    const ml_drift::OHWI ohwi(c.o, 1, 1, c.i);
    ml_drift::WeightsDescription desc;
    desc.type = ml_drift::DataType::kUint8;
    desc.layout = c.layout;
    desc.output_group_size = c.group_size;
    const size_t elements_count = static_cast<size_t>(
        ml_drift::GetTotalElementsCountForLayout(desc, ohwi));

    ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> ref_in;
    ref_in.shape = ohwi;
    ref_in.data.resize(static_cast<size_t>(c.o) * c.i);
    for (size_t k = 0; k < ref_in.data.size(); ++k) {
      ref_in.data[k] = static_cast<int8_t>((k * 37 + 13) % 256 - 128);
    }

    std::vector<uint8_t> expected(elements_count, 0xCC);
    ml_drift::RearrangeWeightsInt8AsUint8(ref_in, desc,
                                          absl::MakeSpan(expected), 128, 128u);

    std::vector<uint8_t> actual(elements_count, 0x33);
    ASSERT_TRUE(TryRearrangeInt8AsUint8Direct(ref_in.data, ohwi, desc,
                                              absl::MakeSpan(actual)));
    EXPECT_THAT(actual, ElementsAreArray(expected))
        << "o=" << c.o << " i=" << c.i << " group=" << c.group_size;
  }
}

TEST(BuildBlockwiseScaleOrZpDescriptorTest, MatchesReferenceOnMetalAndOpenCl) {
  ml_drift::GpuInfo apple_metal;
  apple_metal.gpu_api = ml_drift::GpuApi::kMetal;
  apple_metal.vendor = ml_drift::GpuVendor::kApple;
  apple_metal.apple_info.gpu_family = ml_drift::AppleInfo::Family::kApple10;
  apple_metal.metal_info.buffer_max_size = 1ULL << 30;
  apple_metal.metal_info.image2d_max_width = 16384;
  apple_metal.metal_info.image2d_max_height = 16384;

  ml_drift::GpuInfo adreno_cl;
  adreno_cl.gpu_api = ml_drift::GpuApi::kOpenCl;
  adreno_cl.vendor = ml_drift::GpuVendor::kQualcomm;
  adreno_cl.opencl_info.supports_images = true;
  adreno_cl.opencl_info.image2d_max_width = 16384;
  adreno_cl.opencl_info.image2d_max_height = 16384;
  adreno_cl.opencl_info.buffer_max_size = 1ULL << 30;

  for (const ml_drift::GpuInfo* gpu_info : {&apple_metal, &adreno_cl}) {
    for (const ml_drift::DataType dtype :
         {ml_drift::DataType::kFloat16, ml_drift::DataType::kFloat32}) {
      for (const std::pair<int, int>& oi :
           {std::pair{8, 4}, std::pair{7, 5}, std::pair{96, 64},
            std::pair{70, 37}}) {
        const ml_drift::OHWI shape(oi.first, 1, 1, oi.second);
        std::vector<float> scales(shape.DimensionsProduct());
        for (size_t k = 0; k < scales.size(); ++k) {
          scales[k] = static_cast<float>(static_cast<int>(k % 19) - 9) * 0.125f;
        }
        ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> ref_src;
        ref_src.shape = shape;
        ref_src.data = scales;
        ml_drift::TensorDescriptor expected =
            ml_drift::ScaleOrZeroPointToTensorDesc(*gpu_info, ref_src, dtype);
        expected
            .UpdateToSupportedStorageType(*gpu_info, expected.GetBHWDCShape())
            .IgnoreError();

        const ml_drift::TensorDescriptor actual =
            BuildBlockwiseScaleOrZpDescriptor(*gpu_info, scales, shape, dtype);
        EXPECT_EQ(actual.GetStorageType(), expected.GetStorageType());
        EXPECT_EQ(actual.GetDataType(), expected.GetDataType());
        EXPECT_EQ(actual.GetLayout(), expected.GetLayout());
        EXPECT_EQ(actual.GetBHWDCShape(), expected.GetBHWDCShape());
        EXPECT_THAT(actual.GetData(), ElementsAreArray(expected.GetData()));
      }
    }
  }
}

std::vector<uint8_t> PackedInt2(int num_elements) {
  std::vector<uint8_t> packed((num_elements + 3) / 4, 0);
  for (int k = 0; k < num_elements; ++k) {
    const int value = (k * 3 + 1) % 4 - 2;  // -2..1
    const uint8_t crumb = static_cast<uint8_t>(value) & 0x03u;
    packed[k / 4] |= static_cast<uint8_t>(crumb << ((k % 4) * 2));
  }
  return packed;
}

std::vector<int8_t> UnpackInt2ForReference(absl::Span<const uint8_t> packed,
                                           int num_elements) {
  std::vector<int8_t> unpacked(num_elements + 64, 0);
  for (int k = 0; k < num_elements; ++k) {
    int8_t q = (packed[k / 4] >> ((k % 4) * 2)) & 0x03;
    if (q & 0x02) q |= 0xFC;
    unpacked[k] = q;
  }
  return unpacked;
}

TEST(RearrangeInt4WeightsTest, ParallelDirectMatchesLegacyAndSingleThread) {
  struct ParallelInt4Case {
    int o;
    int i;
    int group_size;
    ml_drift::WeightsLayout layout;
  };
  const std::vector<ParallelInt4Case> cases = {
      {1024, 1024, 256, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {1036, 516, 259, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {1024, 1024, 32, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {1024, 1024, 256,
       ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
      {1036, 516, 259,
       ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
      {1036, 516, 32,
       ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
  };
  for (const ParallelInt4Case& c : cases) {
    const ml_drift::OHWI ohwi(c.o, 1, 1, c.i);
    ml_drift::WeightsDescription desc;
    desc.type = ml_drift::DataType::kUint4;
    desc.layout = c.layout;
    desc.output_group_size = c.group_size;
    const int elements_count =
        ml_drift::GetTotalElementsCountForLayout(desc, ohwi);
    const std::vector<int8_t> packed = PackedInt4(c.o * c.i);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        const RearrangedInt4Weights parallel,
        RearrangeInt4Weights(packed, ohwi, desc, elements_count,
                             /*want_sum_i=*/true, /*max_threads=*/8));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        const RearrangedInt4Weights single_thread,
        RearrangeInt4Weights(packed, ohwi, desc, elements_count,
                             /*want_sum_i=*/true, /*max_threads=*/1));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        const RearrangedInt4Weights no_sum,
        RearrangeInt4Weights(packed, ohwi, desc, elements_count,
                             /*want_sum_i=*/false, /*max_threads=*/8));

    ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> unpacked;
    unpacked.shape = ohwi;
    unpacked.data.assign(c.o * c.i + 64, 0);
    ml_drift::UnpackDenseInt4IntoInt8(packed.data(), c.o * c.i,
                                      unpacked.data.data());
    std::vector<uint8_t> expected(
        (static_cast<size_t>(elements_count) * 4 + 7) / 8);
    ml_drift::RearrangeWeightsInt8AsUint4(unpacked, desc,
                                          absl::MakeSpan(expected), 8, 8u);

    EXPECT_THAT(parallel.data, ElementsAreArray(expected))
        << "o=" << c.o << " i=" << c.i << " group=" << c.group_size;
    EXPECT_THAT(single_thread.data, ElementsAreArray(expected));
    EXPECT_THAT(no_sum.data, ElementsAreArray(expected));
    ASSERT_TRUE(parallel.sum_i.has_value());
    ASSERT_TRUE(single_thread.sum_i.has_value());
    EXPECT_FALSE(no_sum.sum_i.has_value());
    const std::vector<int32_t> expected_sum = ComputeWeightsSumI(
        ohwi, absl::MakeConstSpan(unpacked.data).first(c.o * c.i));
    EXPECT_THAT(*parallel.sum_i, ElementsAreArray(expected_sum));
    EXPECT_THAT(*single_thread.sum_i, ElementsAreArray(expected_sum));
  }
}

TEST(RearrangeInt2WeightsTest, MatchesLegacyUnpackAcrossShapesAndParallelism) {
  struct Int2Case {
    int o;
    int i;
    int group_size;
    ml_drift::WeightsLayout layout;
  };
  const std::vector<Int2Case> cases = {
      {6, 7, 2, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {5, 9, 2, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {8, 32, 2, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {84, 160, 1, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {84, 160, 4, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {84, 160, 8, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {6, 7, 2, ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
      {5, 9, 2, ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
      {84, 160, 4, ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
      // Large parallel shapes (>= 512K elements) across aligned,
      // non-tile-multiple, odd, and multi-group variants.
      {1024, 1024, 256, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {1036, 516, 259, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {1027, 517, 257, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {1024, 1024, 32, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
      {1024, 1024, 256,
       ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
      {1027, 517, 32,
       ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
      // Non-I4O4 layout exercising the RearrangeInt2Weights fallback path.
      {8, 32, 2, ml_drift::WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4},
  };
  for (const Int2Case& c : cases) {
    const ml_drift::OHWI ohwi(c.o, 1, 1, c.i);
    ml_drift::WeightsDescription desc;
    desc.type = ml_drift::DataType::kUint2;
    desc.layout = c.layout;
    desc.output_group_size = c.group_size;
    const int elements_count =
        ml_drift::GetTotalElementsCountForLayout(desc, ohwi);
    const std::vector<uint8_t> packed = PackedInt2(c.o * c.i);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        const RearrangedInt4Weights parallel,
        RearrangeInt2Weights(packed, ohwi, desc, elements_count,
                             /*want_sum_i=*/true, /*max_threads=*/8));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        const RearrangedInt4Weights single_thread,
        RearrangeInt2Weights(packed, ohwi, desc, elements_count,
                             /*want_sum_i=*/true, /*max_threads=*/1));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        const RearrangedInt4Weights no_sum,
        RearrangeInt2Weights(packed, ohwi, desc, elements_count,
                             /*want_sum_i=*/false, /*max_threads=*/8));

    ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> unpacked;
    unpacked.shape = ohwi;
    unpacked.data = UnpackInt2ForReference(packed, c.o * c.i);
    std::vector<uint8_t> expected(
        (static_cast<size_t>(elements_count) * 2 + 7) / 8);
    ml_drift::RearrangeWeightsInt8AsUint2(unpacked, desc,
                                          absl::MakeSpan(expected), 2, 2u);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        const std::vector<int32_t> expected_sum,
        ComputePackedInt2WeightsSumI(packed, ohwi));

    EXPECT_THAT(parallel.data, ElementsAreArray(expected))
        << "o=" << c.o << " i=" << c.i << " group=" << c.group_size;
    EXPECT_THAT(single_thread.data, ElementsAreArray(expected));
    EXPECT_THAT(no_sum.data, ElementsAreArray(expected));
    ASSERT_TRUE(parallel.sum_i.has_value());
    ASSERT_TRUE(single_thread.sum_i.has_value());
    EXPECT_FALSE(no_sum.sum_i.has_value());
    EXPECT_THAT(*parallel.sum_i, ElementsAreArray(expected_sum));
    EXPECT_THAT(*single_thread.sum_i, ElementsAreArray(expected_sum));
  }
}

TEST(RearrangeInt2WeightsTest, RejectsAShortBuffer) {
  const ml_drift::OHWI ohwi(4, 1, 1, 8);
  ml_drift::WeightsDescription desc;
  desc.type = ml_drift::DataType::kUint2;
  desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  desc.output_group_size = 1;
  const std::vector<uint8_t> packed(7, 0);  // 8 bytes needed.
  EXPECT_THAT(
      RearrangeInt2Weights(packed, ohwi, desc,
                           ml_drift::GetTotalElementsCountForLayout(desc, ohwi),
                           /*want_sum_i=*/false),
      ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::tensor::ml_drift_runner_internal
