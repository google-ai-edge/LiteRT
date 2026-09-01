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
#include <cstddef>
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

TEST(TensorRtBytecodeTest, SharedWeightBundleSelectsEngineAndWeightViews) {
  std::vector<TensorRtSharedWeight> shared_weights;
  shared_weights.push_back(
      {TensorRtWeightDataType::kFloat, 2, {1, 2, 3, 4, 5, 6, 7, 8}});
  shared_weights.push_back(
      {TensorRtWeightDataType::kInt4, 5, {0x12, 0x34, 0x05}});
  const std::vector<uint8_t> engine_0 = {9, 8, 7};
  const std::vector<uint8_t> engine_1 = {6, 5, 4, 3};

  TensorRtBundleEntry entry_0;
  entry_0.function_name = "partition_0";
  entry_0.input_names = {"input_0"};
  entry_0.output_names = {"output_0"};
  entry_0.engine_data = engine_0.data();
  entry_0.engine_size = engine_0.size();
  entry_0.refit_weights = {{"matrix", 0}};
  TensorRtBundleEntry entry_1;
  entry_1.function_name = "partition_1";
  entry_1.input_names = {"input_1"};
  entry_1.output_names = {"output_1"};
  entry_1.engine_data = engine_1.data();
  entry_1.engine_size = engine_1.size();
  entry_1.refit_weights = {{"matrix_alias", 0}, {"packed", 1}};

  auto packed =
      PackTensorRtSharedWeightBundle(shared_weights, {entry_0, entry_1});
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();
  EXPECT_FALSE(
      ParseTensorRtBytecode(packed->data(), packed->size()).HasValue());

  auto parsed =
      ParseTensorRtBytecode(packed->data(), packed->size(), "partition_1");
  ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
  EXPECT_EQ(parsed->version, kTensorRtBytecodeVersionWithSharedWeights);
  EXPECT_EQ(parsed->function_name, "partition_1");
  EXPECT_EQ(parsed->input_names, std::vector<std::string>({"input_1"}));
  EXPECT_EQ(parsed->output_names, std::vector<std::string>({"output_1"}));
  EXPECT_EQ(std::vector<uint8_t>(parsed->engine_data,
                                 parsed->engine_data + parsed->engine_size),
            engine_1);
  ASSERT_EQ(parsed->refit_weights.size(), 2);
  EXPECT_EQ(parsed->refit_weights[0].name, "matrix_alias");
  EXPECT_EQ(parsed->refit_weights[0].data_type, TensorRtWeightDataType::kFloat);
  EXPECT_EQ(parsed->refit_weights[0].count, 2);
  EXPECT_EQ(std::vector<uint8_t>(
                parsed->refit_weights[0].data,
                parsed->refit_weights[0].data + parsed->refit_weights[0].size),
            shared_weights[0].data);
  EXPECT_EQ(parsed->refit_weights[1].name, "packed");
  EXPECT_EQ(parsed->refit_weights[1].data_type, TensorRtWeightDataType::kInt4);
  EXPECT_EQ(parsed->refit_weights[1].count, 5);
  EXPECT_EQ(std::vector<uint8_t>(
                parsed->refit_weights[1].data,
                parsed->refit_weights[1].data + parsed->refit_weights[1].size),
            shared_weights[1].data);
  EXPECT_FALSE(
      ParseTensorRtBytecode(packed->data(), packed->size(), "missing_partition")
          .HasValue());
}

TEST(TensorRtBytecodeTest, SharedWeightBundleRejectsInvalidReferences) {
  std::vector<TensorRtSharedWeight> shared_weights = {
      {TensorRtWeightDataType::kInt8, 1, {7}}};
  const std::vector<uint8_t> engine = {1};
  TensorRtBundleEntry entry;
  entry.function_name = "partition_0";
  entry.engine_data = engine.data();
  entry.engine_size = engine.size();
  entry.refit_weights = {{"weight", 1}};
  EXPECT_FALSE(
      PackTensorRtSharedWeightBundle(shared_weights, {entry}).HasValue());
}

TEST(TensorRtBytecodeTest, SharedWeightShardKeepsAndRemapsReferencedWeights) {
  std::vector<TensorRtSharedWeight> shared_weights = {
      {TensorRtWeightDataType::kInt8, 2, {1, 2}},
      {TensorRtWeightDataType::kInt8, 3, {3, 4, 5}},
      {TensorRtWeightDataType::kInt4, 5, {0x12, 0x34, 0x05}}};
  const std::vector<uint8_t> engine = {9, 8, 7};
  TensorRtBundleEntry entry;
  entry.function_name = "partition_1";
  entry.input_names = {"input"};
  entry.output_names = {"output"};
  entry.engine_data = engine.data();
  entry.engine_size = engine.size();
  entry.refit_weights = {{"packed", 2}, {"first", 0}, {"packed_alias", 2}};

  auto shard = PackTensorRtSharedWeightShard(shared_weights, entry);
  ASSERT_TRUE(shard.HasValue()) << shard.Error().Message();
  auto full = PackTensorRtSharedWeightBundle(shared_weights, {entry});
  ASSERT_TRUE(full.HasValue()) << full.Error().Message();
  EXPECT_LT(shard->size(), full->size());

  auto parsed =
      ParseTensorRtBytecode(shard->data(), shard->size(), "partition_1");
  ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
  ASSERT_EQ(parsed->refit_weights.size(), 3);
  EXPECT_EQ(std::vector<uint8_t>(
                parsed->refit_weights[0].data,
                parsed->refit_weights[0].data + parsed->refit_weights[0].size),
            shared_weights[2].data);
  EXPECT_EQ(std::vector<uint8_t>(
                parsed->refit_weights[1].data,
                parsed->refit_weights[1].data + parsed->refit_weights[1].size),
            shared_weights[0].data);
  EXPECT_EQ(parsed->refit_weights[2].data, parsed->refit_weights[0].data);
}

TEST(TensorRtBytecodeTest, SharedWeightShardRejectsInvalidReferences) {
  const std::vector<TensorRtSharedWeight> shared_weights = {
      {TensorRtWeightDataType::kInt8, 1, {7}}};
  const std::vector<uint8_t> engine = {1};
  TensorRtBundleEntry entry;
  entry.function_name = "partition_0";
  entry.engine_data = engine.data();
  entry.engine_size = engine.size();
  entry.refit_weights = {{"weight", 1}};
  EXPECT_FALSE(PackTensorRtSharedWeightShard(shared_weights, entry).HasValue());
}

TEST(TensorRtBytecodeTest, SharedWeightBundleRejectsOverflowingInt4Count) {
  std::vector<TensorRtSharedWeight> shared_weights = {
      {TensorRtWeightDataType::kInt4,
       std::numeric_limits<uint64_t>::max(),
       {}}};
  const std::vector<uint8_t> engine = {1};
  TensorRtBundleEntry entry;
  entry.function_name = "partition_0";
  entry.engine_data = engine.data();
  entry.engine_size = engine.size();
  entry.refit_weights = {{"weight", 0}};
  EXPECT_FALSE(
      PackTensorRtSharedWeightBundle(shared_weights, {entry}).HasValue());
}

TEST(TensorRtBytecodeTest, AotLocatorRoundTrips) {
  const std::vector<uint8_t> artifact = {1, 3, 5, 7, 9};
  TensorRtAotLocator locator{
      "/tmp/tensorrt/artifact.trt_aot", artifact.size(),
      FingerprintTensorRtArtifact(artifact.data(), artifact.size()),
      TensorRtAotFileIdentity{/*device=*/3,
                              /*inode=*/5,
                              /*mtime_seconds=*/7,
                              /*mtime_nanoseconds=*/11,
                              /*ctime_seconds=*/13,
                              /*ctime_nanoseconds=*/17}};
  auto packed = PackTensorRtAotLocator(locator);
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();
  ASSERT_LT(packed->size(), 128);

  auto parsed = TryParseTensorRtAotLocator(packed->data(), packed->size());
  ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
  ASSERT_TRUE(parsed->has_value());
  EXPECT_EQ(parsed->value().path, locator.path);
  EXPECT_EQ(parsed->value().artifact_size, locator.artifact_size);
  EXPECT_EQ(parsed->value().fingerprint, locator.fingerprint);
  EXPECT_EQ(parsed->value().file_identity, locator.file_identity);
  EXPECT_EQ(parsed->value().version, kTensorRtAotLocatorVersion);
}

TEST(TensorRtBytecodeTest, LegacyAotLocatorStillParsesWithoutFileIdentity) {
  constexpr uint32_t kAotLocatorMagic = 0x414e524c;
  const std::string path = "/tmp/legacy.trt_aot";
  std::vector<uint8_t> packed;
  AppendLe32(packed, kAotLocatorMagic);
  AppendLe32(packed, 1);  // Legacy locator version.
  AppendLe64(packed, 123);
  AppendLe64(packed, 7);
  AppendLe64(packed, 11);
  AppendFixedString(packed, path);

  auto parsed = TryParseTensorRtAotLocator(packed.data(), packed.size());
  ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
  ASSERT_TRUE(parsed->has_value());
  EXPECT_EQ(parsed->value().path, path);
  EXPECT_EQ(parsed->value().artifact_size, 123);
  EXPECT_EQ(parsed->value().fingerprint, (TensorRtArtifactFingerprint{7, 11}));
  EXPECT_FALSE(parsed->value().file_identity.has_value());
  EXPECT_EQ(parsed->value().version, 1);
}

TEST(TensorRtBytecodeTest, AotFingerprintIsChunkComposable) {
  std::vector<uint8_t> bytes(kTensorRtAotFingerprintChunkBytes + 17);
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(i * 31);
  }
  TensorRtAotFingerprintBuilder builder;
  builder.Add(bytes.data(), kTensorRtAotFingerprintChunkBytes);
  builder.Add(bytes.data() + kTensorRtAotFingerprintChunkBytes, 17);
  EXPECT_EQ(builder.Finish(),
            FingerprintTensorRtAotArtifact(bytes.data(), bytes.size()));
  bytes.back() ^= 1;
  EXPECT_FALSE(builder.Finish() ==
               FingerprintTensorRtAotArtifact(bytes.data(), bytes.size()));
}

TEST(TensorRtBytecodeTest, OrdinaryBytecodeIsNotAnAotLocator) {
  const std::vector<uint8_t> engine = {1};
  auto packed = PackTensorRtBytecode("partition_0", {}, {"output"},
                                     engine.data(), engine.size());
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();
  auto parsed = TryParseTensorRtAotLocator(packed->data(), packed->size());
  ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
  EXPECT_FALSE(parsed->has_value());
}

TEST(TensorRtBytecodeTest, AotLocatorRejectsRelativeAndTruncatedData) {
  TensorRtAotLocator relative{"artifact.trt_aot", 1, {2, 3}};
  EXPECT_FALSE(PackTensorRtAotLocator(relative).HasValue());

  TensorRtAotLocator locator{"/tmp/artifact.trt_aot", 1, {2, 3}};
  auto packed = PackTensorRtAotLocator(locator);
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();
  packed->pop_back();
  EXPECT_FALSE(
      TryParseTensorRtAotLocator(packed->data(), packed->size()).HasValue());
}

TEST(TensorRtBytecodeTest, ArtifactFingerprintDependsOnAllBytes) {
  const std::vector<uint8_t> first = {1, 2, 3, 4};
  const std::vector<uint8_t> second = {1, 2, 3, 5};
  EXPECT_EQ(FingerprintTensorRtArtifact(first.data(), first.size()),
            FingerprintTensorRtArtifact(first.data(), first.size()));
  EXPECT_FALSE(FingerprintTensorRtArtifact(first.data(), first.size()) ==
               FingerprintTensorRtArtifact(second.data(), second.size()));
}

TEST(TensorRtBytecodeTest, AotManifestRoundTrips) {
  TensorRtAotLocator locator{"/tmp/artifact.trt_aot", 123, {7, 11}};
  auto packed_locator = PackTensorRtAotLocator(locator);
  ASSERT_TRUE(packed_locator.HasValue()) << packed_locator.Error().Message();
  TensorRtAotManifest manifest{
      {13, 17}, {*packed_locator}, {"partition_0", "partition_1"}, {0, 0}};
  auto packed = PackTensorRtAotManifest(manifest);
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();
  auto parsed = ParseTensorRtAotManifest(packed->data(), packed->size());
  ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
  EXPECT_EQ(parsed->cache_key, manifest.cache_key);
  EXPECT_EQ(parsed->locators, manifest.locators);
  EXPECT_EQ(parsed->call_infos, manifest.call_infos);
  EXPECT_EQ(parsed->bytecode_indices, manifest.bytecode_indices);
}

TEST(TensorRtBytecodeTest, AotManifestRejectsInvalidIndexAndTruncation) {
  TensorRtAotLocator locator{"/tmp/artifact.trt_aot", 123, {7, 11}};
  auto packed_locator = PackTensorRtAotLocator(locator);
  ASSERT_TRUE(packed_locator.HasValue()) << packed_locator.Error().Message();
  TensorRtAotManifest invalid{
      {13, 17}, {*packed_locator}, {"partition_0"}, {1}};
  EXPECT_FALSE(PackTensorRtAotManifest(invalid).HasValue());

  invalid.bytecode_indices[0] = 0;
  auto packed = PackTensorRtAotManifest(invalid);
  ASSERT_TRUE(packed.HasValue()) << packed.Error().Message();
  packed->pop_back();
  EXPECT_FALSE(
      ParseTensorRtAotManifest(packed->data(), packed->size()).HasValue());
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
