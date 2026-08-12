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

#include "litert/vendors/nvidia/bytecode.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace litert::nvidia {
namespace {

void AppendLe32(std::vector<uint8_t>& bytes, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<uint8_t>(value >> shift));
  }
}

void AppendLe64(std::vector<uint8_t>& bytes, uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<uint8_t>(value >> shift));
  }
}

void AppendFixedString(std::vector<uint8_t>& bytes, const std::string& value) {
  AppendLe32(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

// This intentionally does not call PackTensorRtBytecode: it fixes the legacy
// wire layout independently so a simultaneous packer/parser drift is caught.
std::vector<uint8_t> MakeHandAuthoredHeadBytecode(uint32_t version,
                                                  uint32_t weight_format) {
  constexpr uint32_t kWireMagic = 0x4e52544c;
  constexpr uint32_t kK = 64;
  constexpr uint32_t kN = 64;
  const uint64_t weight_bytes =
      weight_format ==
              static_cast<uint32_t>(
                  TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved)
          ? static_cast<uint64_t>(kK) * kN / 2
          : static_cast<uint64_t>(kK) * kN / 4;

  std::vector<uint8_t> bytes;
  AppendLe32(bytes, kWireMagic);
  AppendLe32(bytes, version);
  AppendFixedString(bytes, "partition_0");
  AppendLe32(bytes, 0);  // Inputs.
  AppendLe32(bytes, 2);  // Outputs.
  AppendFixedString(bytes, "hidden");
  AppendFixedString(bytes, "");
  AppendLe64(bytes, 1);  // Engine size.
  bytes.push_back(0xa5);
  AppendLe32(bytes, 0);  // Hidden port.
  AppendLe32(bytes, 1);  // Logits port.
  AppendLe32(bytes, kK);
  AppendLe32(bytes, kN);
  AppendLe32(bytes, 0x41f00000);  // 30.0f.
  if (version == kTensorRtBytecodeVersionWithTypedHead) {
    AppendLe32(bytes, weight_format);
  }
  AppendLe64(bytes, weight_bytes);
  bytes.insert(bytes.end(), weight_bytes, 0x87);
  AppendLe64(bytes, static_cast<uint64_t>(kN) * sizeof(uint16_t));
  bytes.insert(bytes.end(), kN * sizeof(uint16_t), 0x3f);
  return bytes;
}

TEST(TensorRtBytecodeTest, TensorRtLlmHeadArchitectureEligibility) {
  EXPECT_FALSE(IsTensorRtLlmHeadComputeCapabilitySupported(75));
  EXPECT_TRUE(IsTensorRtLlmHeadComputeCapabilitySupported(80));
  EXPECT_TRUE(IsTensorRtLlmHeadComputeCapabilitySupported(89));
  EXPECT_FALSE(IsTensorRtLlmHeadComputeCapabilitySupported(90));
  EXPECT_FALSE(IsTensorRtLlmHeadComputeCapabilitySupported(100));
  EXPECT_TRUE(IsTensorRtLlmHeadComputeCapabilitySupported(120));
  EXPECT_TRUE(IsTensorRtLlmHeadComputeCapabilitySupported(121));

  EXPECT_FALSE(IsInt2GemvComputeCapabilitySupported(79));
  EXPECT_TRUE(IsInt2GemvComputeCapabilitySupported(80));
  EXPECT_TRUE(IsInt2GemvComputeCapabilitySupported(90));
  EXPECT_TRUE(IsInt2GemvComputeCapabilitySupported(100));
  EXPECT_TRUE(IsInt2GemvComputeCapabilitySupported(120));
}

TEST(TensorRtBytecodeTest, VersionThreeRoundTripsNativeInt2HeadViews) {
  const std::vector<uint8_t> engine = {9, 7, 5, 3};
  const std::vector<uint8_t> weights(64 * 64 / 4, 0x93);
  const std::vector<uint8_t> scales(64 * sizeof(uint16_t), 0x3f);
  const TensorRtLlmHead head{
      /*hidden_output_port=*/0,
      /*logits_output_port=*/1,
      /*k=*/64,
      /*n=*/64,
      /*soft_cap=*/30.0f,
      /*weight_format=*/TensorRtLlmHeadWeightFormat::kInt2TfliteRowMajor,
      /*packed_weights=*/weights.data(),
      /*packed_weights_size=*/weights.size(),
      /*bf16_scales=*/scales.data(),
      /*bf16_scales_size=*/scales.size(),
  };
  auto packed = PackTensorRtBytecode("partition_0", {"input_0"}, {"hidden", ""},
                                     engine.data(), engine.size(), &head);
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();

  auto parsed = ParseTensorRtBytecode(packed->data(), packed->size());
  ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
  EXPECT_EQ(parsed->version, kTensorRtBytecodeVersionWithTypedHead);
  ASSERT_TRUE(parsed->trtllm_head.has_value());
  EXPECT_EQ(parsed->trtllm_head->weight_format,
            TensorRtLlmHeadWeightFormat::kInt2TfliteRowMajor);
  EXPECT_EQ(std::vector<uint8_t>(parsed->trtllm_head->packed_weights,
                                 parsed->trtllm_head->packed_weights +
                                     parsed->trtllm_head->packed_weights_size),
            weights);
  EXPECT_EQ(std::vector<uint8_t>(parsed->trtllm_head->bf16_scales,
                                 parsed->trtllm_head->bf16_scales +
                                     parsed->trtllm_head->bf16_scales_size),
            scales);
}

TEST(TensorRtBytecodeTest, PackerRejectsUnknownHeadWeightFormat) {
  const std::vector<uint8_t> engine = {1};
  const std::vector<uint8_t> weights(64 * 64 / 4, 0);
  const std::vector<uint8_t> scales(64 * sizeof(uint16_t), 0);
  const TensorRtLlmHead head{0,
                             1,
                             64,
                             64,
                             30.0f,
                             static_cast<TensorRtLlmHeadWeightFormat>(99),
                             weights.data(),
                             weights.size(),
                             scales.data(),
                             scales.size()};
  EXPECT_FALSE(PackTensorRtBytecode("partition_0", {}, {"hidden", ""},
                                    engine.data(), engine.size(), &head)
                   .HasValue());
}

TEST(TensorRtBytecodeTest, ParserRejectsUnknownVersionThreeHeadWeightFormat) {
  const auto bytes = MakeHandAuthoredHeadBytecode(
      kTensorRtBytecodeVersionWithTypedHead, /*weight_format=*/99);
  EXPECT_FALSE(ParseTensorRtBytecode(bytes.data(), bytes.size()).HasValue());
}

TEST(TensorRtBytecodeTest, VersionOneRoundTripIsStillSupported) {
  const std::vector<uint8_t> engine = {1, 3, 5, 7};
  auto packed = PackTensorRtBytecode("partition_0", {"input_0"}, {"output_0"},
                                     engine.data(), engine.size());
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();

  auto parsed = ParseTensorRtBytecode(packed->data(), packed->size());
  ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
  EXPECT_EQ(parsed->version, kTensorRtBytecodeVersion);
  EXPECT_EQ(parsed->function_name, "partition_0");
  EXPECT_EQ(parsed->input_names, std::vector<std::string>({"input_0"}));
  EXPECT_EQ(parsed->output_names, std::vector<std::string>({"output_0"}));
  EXPECT_EQ(std::vector<uint8_t>(parsed->engine_data,
                                 parsed->engine_data + parsed->engine_size),
            engine);
  EXPECT_FALSE(parsed->trtllm_head.has_value());
}

TEST(TensorRtBytecodeTest, VersionTwoRoundTripsTensorRtLlmHeadViews) {
  const std::vector<uint8_t> engine = {2, 4, 6, 8};
  const std::vector<uint8_t> weights(64 * 64 / 2, 0x87);
  const std::vector<uint8_t> scales(64 * sizeof(uint16_t), 0x3f);
  const TensorRtLlmHead head{
      /*hidden_output_port=*/0,
      /*logits_output_port=*/1,
      /*k=*/64,
      /*n=*/64,
      /*soft_cap=*/30.0f,
      /*weight_format=*/
      TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved,
      /*packed_weights=*/weights.data(),
      /*packed_weights_size=*/weights.size(),
      /*bf16_scales=*/scales.data(),
      /*bf16_scales_size=*/scales.size(),
  };
  auto packed = PackTensorRtBytecode("partition_0", {"input_0"}, {"hidden", ""},
                                     engine.data(), engine.size(), &head);
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();

  auto parsed = ParseTensorRtBytecode(packed->data(), packed->size());
  ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
  EXPECT_EQ(parsed->version, kTensorRtBytecodeVersionWithTrtLlmHead);
  ASSERT_TRUE(parsed->trtllm_head.has_value());
  EXPECT_EQ(parsed->trtllm_head->hidden_output_port, 0);
  EXPECT_EQ(parsed->trtllm_head->logits_output_port, 1);
  EXPECT_EQ(parsed->trtllm_head->k, 64);
  EXPECT_EQ(parsed->trtllm_head->n, 64);
  EXPECT_FLOAT_EQ(parsed->trtllm_head->soft_cap, 30.0f);
  EXPECT_EQ(parsed->trtllm_head->weight_format,
            TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved);
  EXPECT_EQ(std::vector<uint8_t>(parsed->trtllm_head->packed_weights,
                                 parsed->trtllm_head->packed_weights +
                                     parsed->trtllm_head->packed_weights_size),
            weights);
  EXPECT_EQ(std::vector<uint8_t>(parsed->trtllm_head->bf16_scales,
                                 parsed->trtllm_head->bf16_scales +
                                     parsed->trtllm_head->bf16_scales_size),
            scales);
}

TEST(TensorRtBytecodeTest, ParsesFixedLegacyVersionTwoWireFormat) {
  const auto bytes = MakeHandAuthoredHeadBytecode(
      kTensorRtBytecodeVersionWithTrtLlmHead,
      static_cast<uint32_t>(
          TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved));
  auto parsed = ParseTensorRtBytecode(bytes.data(), bytes.size());
  ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
  EXPECT_EQ(parsed->version, kTensorRtBytecodeVersionWithTrtLlmHead);
  EXPECT_EQ(parsed->function_name, "partition_0");
  EXPECT_EQ(parsed->output_names, std::vector<std::string>({"hidden", ""}));
  ASSERT_TRUE(parsed->trtllm_head.has_value());
  EXPECT_EQ(parsed->trtllm_head->weight_format,
            TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved);
  EXPECT_EQ(parsed->trtllm_head->packed_weights_size, 64 * 64 / 2);
  EXPECT_EQ(parsed->trtllm_head->bf16_scales_size, 64 * sizeof(uint16_t));
  EXPECT_EQ(parsed->trtllm_head->packed_weights[0], 0x87);
  EXPECT_EQ(parsed->trtllm_head->bf16_scales[0], 0x3f);
}

TEST(TensorRtBytecodeTest, VersionTwoRequiresEmptyLogitsBinding) {
  const std::vector<uint8_t> engine = {1};
  const std::vector<uint8_t> weights(64 * 64 / 2, 0);
  const std::vector<uint8_t> scales(64 * sizeof(uint16_t), 0);
  const TensorRtLlmHead head{
      0,
      1,
      64,
      64,
      30.0f,
      TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved,
      weights.data(),
      weights.size(),
      scales.data(),
      scales.size()};

  auto packed = PackTensorRtBytecode("partition_0", {}, {"hidden", "logits"},
                                     engine.data(), engine.size(), &head);
  EXPECT_FALSE(packed.HasValue());
}

TEST(TensorRtBytecodeTest, VersionTwoRejectsTruncatedScalePayload) {
  const std::vector<uint8_t> engine = {1};
  const std::vector<uint8_t> weights(64 * 64 / 2, 0);
  const std::vector<uint8_t> scales(64 * sizeof(uint16_t), 0);
  const TensorRtLlmHead head{
      0,
      1,
      64,
      64,
      30.0f,
      TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved,
      weights.data(),
      weights.size(),
      scales.data(),
      scales.size()};
  auto packed = PackTensorRtBytecode("partition_0", {}, {"hidden", ""},
                                     engine.data(), engine.size(), &head);
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();
  packed->pop_back();

  EXPECT_FALSE(
      ParseTensorRtBytecode(packed->data(), packed->size()).HasValue());
}

TEST(TensorRtBytecodeTest, VersionTwoRejectsInvalidMetadata) {
  const std::vector<uint8_t> engine = {1};
  const std::vector<uint8_t> weights(64 * 64 / 2, 0);
  const std::vector<uint8_t> scales(64 * sizeof(uint16_t), 0);

  TensorRtLlmHead invalid_ports{
      0,
      0,
      64,
      64,
      30.0f,
      TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved,
      weights.data(),
      weights.size(),
      scales.data(),
      scales.size()};
  EXPECT_FALSE(PackTensorRtBytecode("partition_0", {}, {"hidden", ""},
                                    engine.data(), engine.size(),
                                    &invalid_ports)
                   .HasValue());

  TensorRtLlmHead invalid_cap{
      0,
      1,
      64,
      64,
      NAN,
      TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved,
      weights.data(),
      weights.size(),
      scales.data(),
      scales.size()};
  EXPECT_FALSE(PackTensorRtBytecode("partition_0", {}, {"hidden", ""},
                                    engine.data(), engine.size(), &invalid_cap)
                   .HasValue());

  TensorRtLlmHead invalid_weight_size{
      0,
      1,
      64,
      64,
      30.0f,
      TensorRtLlmHeadWeightFormat::kInt4ColumnMajorInterleaved,
      weights.data(),
      weights.size() - 1,
      scales.data(),
      scales.size()};
  EXPECT_FALSE(PackTensorRtBytecode("partition_0", {}, {"hidden", ""},
                                    engine.data(), engine.size(),
                                    &invalid_weight_size)
                   .HasValue());
}

TEST(TensorRtBytecodeTest, RejectsUnexpectedTrailingData) {
  const std::vector<uint8_t> engine = {1};
  auto packed = PackTensorRtBytecode("partition_0", {}, {"output"},
                                     engine.data(), engine.size());
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();
  packed->push_back(0);
  EXPECT_FALSE(
      ParseTensorRtBytecode(packed->data(), packed->size()).HasValue());
}

TEST(TensorRtBytecodeTest, RejectsImpossibleStringCountBeforeAllocation) {
  std::vector<uint8_t> bytes;
  AppendLe32(bytes, 0x4e52544c);
  AppendLe32(bytes, kTensorRtBytecodeVersion);
  AppendFixedString(bytes, "partition_0");
  AppendLe32(bytes, std::numeric_limits<uint32_t>::max());
  EXPECT_FALSE(ParseTensorRtBytecode(bytes.data(), bytes.size()).HasValue());
}

}  // namespace
}  // namespace litert::nvidia
