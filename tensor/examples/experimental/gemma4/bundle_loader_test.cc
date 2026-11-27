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

#include "tensor/examples/experimental/gemma4/bundle_loader.h"

#include <sys/stat.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/experimental/gemma4/kv_cache.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {
namespace {

using ::absl_testing::StatusIs;
using ::litert::tensor::examples::gemma4::Config;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::FloatEq;
using ::testing::HasSubstr;
using ::testing::Key;
using ::testing::Not;
using ::testing::SizeIs;

constexpr float kExactSoftcapReciprocal = 0.03333333507180214f;
constexpr float kDefaultKeyScale = 0.03125f;
constexpr float kDefaultValueScale = 0.0625f;

std::string BuildKvSpecsJson(const Config& config, float key_scale,
                             float value_scale, bool reverse_order) {
  const int owners = config.num_kv_heads == 2 ? 24 : 15;
  std::string text = "[";
  for (int idx = 0; idx < owners; ++idx) {
    if (idx > 0) {
      text += ',';
    }
    const int owner = reverse_order ? (owners - 1 - idx) : idx;
    const int dim =
        config.GetLayerType(owner) == Config::LayerType::kGlobal ? 512 : 256;
    absl::StrAppend(&text, R"({"owner":)", owner, R"(,"head_dim":)", dim,
                    R"(,"zero_point":0,"key_scale":)", key_scale,
                    R"(,"value_scale":)", value_scale,
                    R"(,"key_shape_template":[1,)", config.num_kv_heads,
                    R"(,"capacity",)", dim, R"(],"value_shape_template":[1,)",
                    config.num_kv_heads, ",", dim, R"(,"capacity"]})");
  }
  text += ']';
  return text;
}

std::string BuildV2Manifest(const Config& config,
                            float key_scale = kDefaultKeyScale,
                            float value_scale = kDefaultValueScale,
                            bool reverse_owners = true,
                            absl::string_view storage = "compact_int2_v1") {
  const int owners = config.num_kv_heads == 2 ? 24 : 15;
  return absl::StrCat(
      R"({"schema_version":2,"status":"complete","unmapped_float_coefficients":[],)",
      R"("experimental_storage":")", storage,
      R"(","model_config":{"variant":")",
      config.num_kv_heads == 2 ? "e4b" : "e2b", R"(","embed_dim":)",
      config.embed_dim, R"(,"hidden_dim":)", config.hidden_dim,
      R"(,"num_layers":)", config.num_layers, R"(,"num_heads":)",
      config.num_heads, R"(,"num_kv_heads":)", config.num_kv_heads,
      R"(,"vocab_size":)", config.vocab_size, R"(,"num_kv_owners":)", owners,
      R"(,"attention_pattern_size":)", config.attention_pattern_size,
      R"(,"head_dim":)", config.head_dim, R"(,"global_key_size":)",
      config.global_key_size, R"(,"per_layer_input_dim":)",
      config.per_layer_input_dim, R"(,"sliding_window_size":)",
      config.sliding_window_size, R"(},"kv_cache_specs":)",
      BuildKvSpecsJson(config, key_scale, value_scale, reverse_owners),
      R"(,"tensors":[]})");
}

std::string BuildV1Manifest(bool compact_int2) {
  std::string storage_field;
  std::string int2_count_field;
  if (compact_int2) {
    storage_field = R"("experimental_storage":"compact_int2_v1",)";
    int2_count_field = R"("int2":62,)";
  }
  return absl::StrCat(
      R"({"schema_version":1,"status":"complete","unmapped_float_coefficients":[],)",
      storage_field,
      R"("source_bundle":{"sha256":"ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42","bytes":2583085056},)",
      R"("tensor_counts":{)", int2_count_field, R"("float32":814,"int4":)",
      compact_int2 ? 146 : 208, R"(,"int8":71},"kv_cache_specs":)",
      BuildKvSpecsJson(Config::E2B(), kDefaultKeyScale, kDefaultValueScale,
                       /*reverse_order=*/false),
      R"(,"tensors":[]})");
}

std::string ReplaceFirst(std::string text, absl::string_view before,
                         absl::string_view after) {
  const std::string::size_type pos = text.find(before);
  ABSL_CHECK_NE(pos, std::string::npos) << "Missing substring: " << before;
  text.replace(pos, before.size(), after);
  return text;
}

// Small synthetic config compatible with block_size = 256 for PLE embeddings.
Config TinyBundleConfig() {
  Config config = Config::E2B();
  config.vocab_size = 2;
  config.embed_dim = 4;
  config.num_layers = 1;
  config.per_layer_input_dim = 256;
  return config;
}

std::vector<uint8_t> PackFloats(absl::Span<const float> values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(float));
  for (float value : values) {
    const uint32_t bits = absl::bit_cast<uint32_t>(value);
    bytes.push_back(static_cast<uint8_t>(bits & 0xff));
    bytes.push_back(static_cast<uint8_t>((bits >> 8) & 0xff));
    bytes.push_back(static_cast<uint8_t>((bits >> 16) & 0xff));
    bytes.push_back(static_cast<uint8_t>((bits >> 24) & 0xff));
  }
  return bytes;
}

uint8_t PackTwoInt4(int8_t lo, int8_t hi) {
  return static_cast<uint8_t>((lo & 0x0f) | ((hi & 0x0f) << 4));
}

uint8_t PackFourInt2(int8_t a, int8_t b, int8_t c, int8_t d) {
  return static_cast<uint8_t>((a & 0x03) | ((b & 0x03) << 2) |
                              ((c & 0x03) << 4) | ((d & 0x03) << 6));
}

class TempBundleDir {
 public:
  explicit TempBundleDir(absl::string_view suffix) {
    const testing::TestInfo* test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    ABSL_CHECK_NE(test_info, nullptr);
    std::string safe_name = absl::StrCat(test_info->test_suite_name(), "_",
                                         test_info->name(), "_", suffix);
    for (char& c : safe_name) {
      if (c == '/' || c == ' ') {
        c = '_';
      }
    }
    directory_ = absl::StrCat(testing::TempDir(), "/gemma4_bundle_", safe_name);
    if (mkdir(directory_.c_str(), 0755) != 0) {
      ABSL_CHECK_EQ(errno, EEXIST) << "Failed to create " << directory_;
    }
    std::remove(absl::StrCat(directory_, "/manifest.json").c_str());
  }

  TempBundleDir(const TempBundleDir&) = delete;
  TempBundleDir& operator=(const TempBundleDir&) = delete;

  ~TempBundleDir() {
    for (auto it = files_.rbegin(); it != files_.rend(); ++it) {
      std::remove(it->c_str());
    }
    for (auto it = subdirs_.rbegin(); it != subdirs_.rend(); ++it) {
      rmdir(it->c_str());
    }
    rmdir(directory_.c_str());
  }

  const std::string& path() const { return directory_; }

  absl::Status CreateSubdir(absl::string_view relative) {
    const std::string full = absl::StrCat(directory_, "/", relative);
    if (mkdir(full.c_str(), 0755) != 0 && errno != EEXIST) {
      return absl::InternalError(absl::StrCat("Failed to mkdir ", full));
    }
    subdirs_.push_back(full);
    return absl::OkStatus();
  }

  absl::Status WriteFile(absl::string_view relative,
                         absl::Span<const uint8_t> bytes) {
    return WriteRawBytes(relative, reinterpret_cast<const char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size()));
  }

  absl::Status WriteText(absl::string_view relative, absl::string_view text) {
    return WriteRawBytes(relative, text.data(),
                         static_cast<std::streamsize>(text.size()));
  }

 private:
  absl::Status WriteRawBytes(absl::string_view relative, const char* data,
                             std::streamsize size) {
    const std::string full = absl::StrCat(directory_, "/", relative);
    std::ofstream file(full, std::ios::binary);
    if (!file) {
      return absl::InternalError(absl::StrCat("Failed to open ", full));
    }
    if (size > 0) {
      file.write(data, size);
    }
    if (!file) {
      return absl::InternalError(absl::StrCat("Failed to write ", full));
    }
    files_.push_back(full);
    return absl::OkStatus();
  }

  std::string directory_;
  std::vector<std::string> files_;
  std::vector<std::string> subdirs_;
};

// Populates a minimal valid synthetic bundle on disk exercising FP32, INT8,
// INT4 per-channel, INT4 blockwise PLE, widened INT4->INT2 repacking, and
// softcap constant loading.
absl::Status PopulateSyntheticStandardBundle(
    TempBundleDir& dir, absl::flat_hash_map<std::string, ExpectedTensor>& out,
    std::string& manifest_json) {
  LRT_TENSOR_RETURN_IF_ERROR(dir.CreateSubdir("tensors"));

  // Token embeddings: int4 [2, 4] -> 4 bytes.
  // Row 0: [1, -2, 3, -4], Row 1: [-1, 2, -3, 4], scales [0.5, 0.25].
  LRT_TENSOR_RETURN_IF_ERROR(dir.WriteFile(
      "tensors/embed.bin", {PackTwoInt4(1, -2), PackTwoInt4(3, -4),
                            PackTwoInt4(-1, 2), PackTwoInt4(-3, 4)}));
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/embed_scales.bin", PackFloats({0.5f, 0.25f})));

  // Per-layer embeddings: int4 [2, 256] -> 256 bytes, blockwise [2, 1].
  std::vector<uint8_t> ple_bytes(256, PackTwoInt4(1, -1));
  LRT_TENSOR_RETURN_IF_ERROR(dir.WriteFile("tensors/ple.bin", ple_bytes));
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/ple_scales.bin", PackFloats({0.25f, 0.5f})));

  // FP32 norm weight: [4] -> 16 bytes.
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/norm.bin", PackFloats({1.0f, 1.5f, -0.5f, 2.0f})));

  // FP32 activation scale: [1] -> 4 bytes.
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/act_scale.bin", PackFloats({0.25f})));

  // INT8 linear weight: [2, 4] -> 8 bytes, per-channel scales [2].
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/q_proj.bin", {1, 2, 3, 4, 255, 254, 253, 252}));
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/q_proj_scales.bin", PackFloats({0.125f, 0.25f})));

  // Widened INT4 weight with INT2 provenance: [2, 4] -> 8 elements -> 4 bytes
  // in INT4, repacked to 2 bytes in INT2.
  // Row 0: [-2, -1, 0, 1], Row 1: [1, 0, -1, -2].
  LRT_TENSOR_RETURN_IF_ERROR(dir.WriteFile(
      "tensors/down_proj.bin", {PackTwoInt4(-2, -1), PackTwoInt4(0, 1),
                                PackTwoInt4(1, 0), PackTwoInt4(-1, -2)}));
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/down_proj_scales.bin", PackFloats({0.5f, 0.25f})));

  // Exact softcap reciprocal constant and an ignored unrelated constant.
  LRT_TENSOR_RETURN_IF_ERROR(dir.WriteFile(
      "tensors/softcap.bin", PackFloats({kExactSoftcapReciprocal})));
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/other_const.bin", PackFloats({1.0f})));

  out = {
      {"model.embed_tokens.weight", {"int4", {2, 4}}},
      {"model.embed_tokens_per_layer.weight", {"int4", {2, 256}}},
      {"model.norm.weight", {"float32", {4}}},
      {"model.layers.0.mlp.down_proj.input_scale", {"float32", {1}}},
      {"model.layers.0.self_attn.q_proj.weight", {"int8", {2, 4}}},
      {"model.layers.0.mlp.down_proj.weight", {"int4", {2, 4}}},
  };

  manifest_json =
      R"({"schema_version":2,"status":"complete","unmapped_float_coefficients":[],)"
      R"("tensors":[)"
      R"({"name":"model.embed_tokens.weight","dtype":"int4","shape":[2,4],"bytes":4,)"
      R"("encoding":"signed_twos_complement_low_nibble_first","file":"tensors/embed.bin",)"
      R"("quantization":{"kind":"per_channel","quantized_dimension":0,"zero_points":[0],)"
      R"("scales_shape":[2],"scales_dtype":"float32","scales_bytes":8,"scales_file":"tensors/embed_scales.bin"}},)"
      R"({"name":"model.embed_tokens_per_layer.weight","dtype":"int4","shape":[2,256],"bytes":256,)"
      R"("encoding":"signed_twos_complement_low_nibble_first","file":"tensors/ple.bin",)"
      R"("quantization":{"kind":"blockwise","quantized_dimension":0,"zero_points":[0],"block_size":256,)"
      R"("scales_shape":[2,1],"scales_dtype":"float32","scales_bytes":8,"scales_file":"tensors/ple_scales.bin"}},)"
      R"({"name":"model.norm.weight","dtype":"float32","shape":[4],"bytes":16,)"
      R"("encoding":"little_endian_float32","file":"tensors/norm.bin"},)"
      R"({"name":"model.layers.0.mlp.down_proj.input_scale","dtype":"float32","shape":[1],"bytes":4,)"
      R"("encoding":"little_endian_float32","file":"tensors/act_scale.bin"},)"
      R"({"name":"model.layers.0.self_attn.q_proj.weight","dtype":"int8","shape":[2,4],"bytes":8,)"
      R"("encoding":"signed_int8","file":"tensors/q_proj.bin",)"
      R"("quantization":{"kind":"per_channel","quantized_dimension":0,"zero_points":[0],)"
      R"("scales_shape":[2],"scales_dtype":"float32","scales_bytes":8,"scales_file":"tensors/q_proj_scales.bin"}},)"
      R"({"name":"model.layers.0.mlp.down_proj.weight","dtype":"int4","shape":[2,4],"bytes":4,)"
      R"("encoding":"signed_twos_complement_low_nibble_first","file":"tensors/down_proj.bin",)"
      R"("quantization":{"kind":"per_channel","quantized_dimension":0,"zero_points":[0],)"
      R"("scales_shape":[2],"scales_dtype":"float32","scales_bytes":8,"scales_file":"tensors/down_proj_scales.bin"},)"
      R"("sources":[{"dtype":"INT2","source_bytes":2}]}],)"
      R"("constants":[)"
      R"({"name":"decode.tensor_36","dtype":"float32","shape":[],"bytes":4,"file":"tensors/softcap.bin"},)"
      R"({"name":"ignored.const","dtype":"float32","shape":[],"bytes":4,"file":"tensors/other_const.bin"}]})";
  return dir.WriteText("manifest.json", manifest_json);
}

// Populates a compact INT2 synthetic bundle with INT2 PLE partition provenance,
// INT2 linear weight provenance, and `semantic_constants` indirection.
absl::Status PopulateSyntheticCompactInt2Bundle(
    TempBundleDir& dir, absl::flat_hash_map<std::string, ExpectedTensor>& out,
    std::string& manifest_json) {
  LRT_TENSOR_RETURN_IF_ERROR(dir.CreateSubdir("tensors"));

  // Token embeddings: int2 [2, 4] -> 2 bytes.
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/embed_i2.bin",
                    {PackFourInt2(1, -1, -2, 0), PackFourInt2(-2, 0, 1, -1)}));
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/embed_scales.bin", PackFloats({0.5f, 0.25f})));

  // Per-layer embeddings: int2 [2, 256] -> 128 bytes, 1 partition of [2, 256].
  std::vector<uint8_t> ple_i2(128, PackFourInt2(1, 0, -1, -2));
  LRT_TENSOR_RETURN_IF_ERROR(dir.WriteFile("tensors/ple_i2.bin", ple_i2));
  LRT_TENSOR_RETURN_IF_ERROR(
      dir.WriteFile("tensors/ple_scales.bin", PackFloats({0.25f, 0.5f})));

  LRT_TENSOR_RETURN_IF_ERROR(dir.WriteFile(
      "tensors/softcap.bin", PackFloats({kExactSoftcapReciprocal})));

  out = {
      {"model.embed_tokens.weight", {"int2", {2, 4}}},
      {"model.embed_tokens_per_layer.weight", {"int2", {2, 256}}},
  };

  manifest_json =
      R"({"schema_version":2,"status":"complete","unmapped_float_coefficients":[],)"
      R"("semantic_constants":{"logits_softcap_reciprocal":"custom.softcap"},)"
      R"("tensors":[)"
      R"({"name":"model.embed_tokens.weight","dtype":"int2","shape":[2,4],"bytes":2,)"
      R"("encoding":"signed_twos_complement_low_2bits_first","file":"tensors/embed_i2.bin",)"
      R"("quantization":{"kind":"per_channel","quantized_dimension":0,"zero_points":[0],)"
      R"("scales_shape":[2],"scales_dtype":"float32","scales_bytes":8,"scales_file":"tensors/embed_scales.bin"},)"
      R"("sources":[{"dtype":"INT2","shape":[2,4],"source_bytes":2}]},)"
      R"({"name":"model.embed_tokens_per_layer.weight","dtype":"int2","shape":[2,256],"bytes":128,)"
      R"("encoding":"signed_twos_complement_low_2bits_first","file":"tensors/ple_i2.bin",)"
      R"("quantization":{"kind":"blockwise","quantized_dimension":0,"zero_points":[0],"block_size":256,)"
      R"("scales_shape":[2,1],"scales_dtype":"float32","scales_bytes":8,"scales_file":"tensors/ple_scales.bin"},)"
      R"("sources":[{"destination_layer_partition":0,"dtype":"INT2","shape":[2,256],"source_bytes":128}]}],)"
      R"("constants":[)"
      R"({"name":"custom.softcap","dtype":"float32","shape":[],"bytes":4,"file":"tensors/softcap.bin"}]})";
  return dir.WriteText("manifest.json", manifest_json);
}

TEST(BundleLoaderKvSpecsTest, LoadsAndSortsE2BOwnersWithoutTensorFiles) {
  TempBundleDir dir("e2b_owners");
  ASSERT_THAT(
      dir.WriteText("manifest.json",
                    BuildV2Manifest(Config::E2B(), /*key_scale=*/0.03125f,
                                    /*value_scale=*/0.0625f,
                                    /*reverse_owners=*/true)),
      IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::vector<KvOwnerSpec> specs,
      LoadPublishedKvSpecs(dir.path(), Config::E2B()));

  EXPECT_THAT(specs, SizeIs(15));
  EXPECT_THAT(specs,
              Each(AllOf(Field(&KvOwnerSpec::num_heads, 1),
                         Field(&KvOwnerSpec::key_scale, FloatEq(0.03125f)),
                         Field(&KvOwnerSpec::value_scale, FloatEq(0.0625f)))));
  // E2B has attention_pattern_size = 5: global layers are 4, 9, 14.
  EXPECT_THAT(specs, ElementsAre(AllOf(Field(&KvOwnerSpec::owner, 0),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 1),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 2),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 3),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 4),
                                       Field(&KvOwnerSpec::head_dim, 512)),
                                 AllOf(Field(&KvOwnerSpec::owner, 5),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 6),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 7),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 8),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 9),
                                       Field(&KvOwnerSpec::head_dim, 512)),
                                 AllOf(Field(&KvOwnerSpec::owner, 10),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 11),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 12),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 13),
                                       Field(&KvOwnerSpec::head_dim, 256)),
                                 AllOf(Field(&KvOwnerSpec::owner, 14),
                                       Field(&KvOwnerSpec::head_dim, 512))));
}

TEST(BundleLoaderKvSpecsTest, RejectsE4BWithoutCompactOptIn) {
  TempBundleDir dir("e4b_no_optin");
  ASSERT_THAT(dir.WriteText("manifest.json", BuildV2Manifest(Config::E4B())),
              IsOk());

  EXPECT_THAT(LoadPublishedKvSpecs(dir.path(), Config::E4B()),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("E4B")));
}

TEST(BundleLoaderKvSpecsTest, LoadsE4BMultiHeadOwnersWithCompactOptIn) {
  TempBundleDir dir("e4b_compact");
  ASSERT_THAT(
      dir.WriteText("manifest.json",
                    BuildV2Manifest(Config::E4B(), /*key_scale=*/0.03125f,
                                    /*value_scale=*/0.0625f,
                                    /*reverse_owners=*/true)),
      IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::vector<KvOwnerSpec> specs,
      LoadPublishedKvSpecs(dir.path(), Config::E4B(), /*compact_int2=*/true));

  EXPECT_THAT(specs, SizeIs(24));
  EXPECT_THAT(specs,
              Each(AllOf(Field(&KvOwnerSpec::num_heads, 2),
                         Field(&KvOwnerSpec::key_scale, FloatEq(0.03125f)),
                         Field(&KvOwnerSpec::value_scale, FloatEq(0.0625f)))));
  // E4B has attention_pattern_size = 6: global owners are 5, 11, 17, 23.
  EXPECT_EQ(specs[0].owner, 0);
  EXPECT_EQ(specs[0].head_dim, 256);
  EXPECT_EQ(specs[5].owner, 5);
  EXPECT_EQ(specs[5].head_dim, 512);
  EXPECT_EQ(specs[11].head_dim, 512);
  EXPECT_EQ(specs[17].head_dim, 512);
  EXPECT_EQ(specs[23].owner, 23);
  EXPECT_EQ(specs[23].head_dim, 512);
}

TEST(BundleLoaderKvSpecsTest, LoadsLegacySchemaV1StandardE2BManifest) {
  TempBundleDir dir("v1_standard");
  ASSERT_THAT(
      dir.WriteText("manifest.json", BuildV1Manifest(/*compact_int2=*/false)),
      IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::vector<KvOwnerSpec> specs,
      LoadPublishedKvSpecs(dir.path(), Config::E2B(), /*compact_int2=*/false));

  EXPECT_THAT(specs, SizeIs(15));
  EXPECT_EQ(specs[4].head_dim, 512);
  EXPECT_EQ(specs[14].head_dim, 512);
}

TEST(BundleLoaderKvSpecsTest, LoadsLegacySchemaV1CompactInt2E2BManifest) {
  TempBundleDir dir("v1_compact");
  ASSERT_THAT(
      dir.WriteText("manifest.json", BuildV1Manifest(/*compact_int2=*/true)),
      IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::vector<KvOwnerSpec> specs,
      LoadPublishedKvSpecs(dir.path(), Config::E2B(), /*compact_int2=*/true));

  EXPECT_THAT(specs, SizeIs(15));
  EXPECT_EQ(specs[0].owner, 0);
  EXPECT_EQ(specs[14].owner, 14);
}

TEST(BundleLoaderKvSpecsTest, RejectsMissingManifestFile) {
  TempBundleDir dir("missing_manifest");

  EXPECT_THAT(
      LoadPublishedKvSpecs(dir.path(), Config::E2B()),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("cannot open")));
}

TEST(BundleLoaderKvSpecsTest, RejectsUnsupportedRunnerConfig) {
  TempBundleDir dir("unsupported_config");
  ASSERT_THAT(dir.WriteText("manifest.json", BuildV2Manifest(Config::E2B())),
              IsOk());
  Config unsupported = Config::E2B();
  unsupported.sliding_window_size = 8;

  EXPECT_THAT(
      LoadPublishedKvSpecs(dir.path(), unsupported),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("only supports")));
}

TEST(BundleLoaderKvSpecsTest, RejectsVariantMismatchBetweenBundleAndRunner) {
  TempBundleDir dir("variant_mismatch");
  ASSERT_THAT(dir.WriteText("manifest.json", BuildV2Manifest(Config::E4B())),
              IsOk());

  EXPECT_THAT(
      LoadPublishedKvSpecs(dir.path(), Config::E2B(), /*compact_int2=*/true),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("model variant differ")));
}

struct InvalidManifestCase {
  std::string name;
  std::string manifest;
  bool compact_int2 = false;
  bool use_e4b = false;
  std::string expected_error;
};

class BundleLoaderInvalidManifestTest
    : public testing::TestWithParam<InvalidManifestCase> {};

TEST_P(BundleLoaderInvalidManifestTest, RejectsInvalidManifest) {
  const InvalidManifestCase& tc = GetParam();
  TempBundleDir dir(tc.name);
  ASSERT_THAT(dir.WriteText("manifest.json", tc.manifest), IsOk());
  const Config config = tc.use_e4b ? Config::E4B() : Config::E2B();

  EXPECT_THAT(LoadPublishedKvSpecs(dir.path(), config, tc.compact_int2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(tc.expected_error)));
}

INSTANTIATE_TEST_SUITE_P(
    ManifestValidation, BundleLoaderInvalidManifestTest,
    testing::Values(
        InvalidManifestCase{"EmptyManifest", "", false, false, "manifest size"},
        InvalidManifestCase{"JsonArrayRoot", "[]", false, false,
                            "invalid JSON object"},
        InvalidManifestCase{"TruncatedJson", "{", false, false,
                            "invalid JSON object"},
        InvalidManifestCase{
            "TrailingData",
            absl::StrCat(BuildV2Manifest(Config::E2B()), "junk"), false, false,
            "trailing data"},
        InvalidManifestCase{"MissingSchemaVersion",
                            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                                         R"("schema_version":2,)", ""),
                            false, false, "missing integer schema_version"},
        InvalidManifestCase{"MissingStatusField",
                            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                                         R"("status":"complete",)", ""),
                            false, false,
                            "missing or incorrectly typed field status"},
        InvalidManifestCase{
            "UnsupportedSchemaVersion",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("schema_version":2)", R"("schema_version":3)"),
            false, false, "unsupported schema"},
        InvalidManifestCase{
            "FractionalSchemaVersion",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("schema_version":2)", R"("schema_version":2.5)"),
            false, false, "nonnegative integer"},
        InvalidManifestCase{
            "IncompleteStatus",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("status":"complete")", R"("status":"partial")"),
            false, false, "incomplete"},
        InvalidManifestCase{
            "UnmappedFloatCoefficients",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("unmapped_float_coefficients":[])",
                         R"("unmapped_float_coefficients":["unknown"])"),
            false, false, "unmapped coefficients"},
        InvalidManifestCase{
            "EmbedDimMismatch",
            ReplaceFirst(BuildV2Manifest(Config::E2B()), R"("embed_dim":1536)",
                         R"("embed_dim":1537)"),
            false, false, "incorrect model config"},
        InvalidManifestCase{
            "IncorrectCompactStorageTag",
            BuildV2Manifest(Config::E2B(), kDefaultKeyScale, kDefaultValueScale,
                            /*reverse_owners=*/true, "unknown_storage"),
            /*compact_int2=*/true, false, "compact INT2 storage"},
        InvalidManifestCase{
            "IncorrectKvOwnerCount",
            ReplaceFirst(
                BuildV2Manifest(Config::E2B(), kDefaultKeyScale,
                                kDefaultValueScale, /*reverse_owners=*/false),
                R"({"owner":0,"head_dim":256,"zero_point":0,"key_scale":0.03125,"value_scale":0.0625,"key_shape_template":[1,1,"capacity",256],"value_shape_template":[1,1,256,"capacity"]},)",
                ""),
            false, false, "incorrect KV owner count"},
        InvalidManifestCase{
            "NonObjectKvRecord",
            ReplaceFirst(
                BuildV2Manifest(Config::E2B(), kDefaultKeyScale,
                                kDefaultValueScale, /*reverse_owners=*/false),
                R"({"owner":0,"head_dim":256,"zero_point":0,"key_scale":0.03125,"value_scale":0.0625,"key_shape_template":[1,1,"capacity",256],"value_shape_template":[1,1,256,"capacity"]})",
                "42"),
            false, false, "invalid KV record"},
        InvalidManifestCase{"DuplicateOwner",
                            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                                         R"("owner":14)", R"("owner":13)"),
                            false, false, "invalid owner"},
        InvalidManifestCase{"OutOfRangeOwner",
                            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                                         R"("owner":14)", R"("owner":15)"),
                            false, false, "invalid owner"},
        InvalidManifestCase{"NegativeOwner",
                            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                                         R"("owner":14)", R"("owner":-1)"),
                            false, false, "nonnegative integer"},
        InvalidManifestCase{"FractionalOwner",
                            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                                         R"("owner":14)", R"("owner":0.5)"),
                            false, false, "nonnegative integer"},
        InvalidManifestCase{
            "ZeroKeyScale",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("key_scale":0.03125)", R"("key_scale":0)"),
            false, false, "scale"},
        InvalidManifestCase{
            "NegativeKeyScale",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("key_scale":0.03125)", R"("key_scale":-0.5)"),
            false, false, "scale"},
        InvalidManifestCase{
            "InexactKeyScale",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("key_scale":0.03125)", R"("key_scale":0.1)"),
            false, false, "scale"},
        InvalidManifestCase{
            "OverflowingKeyScale",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("key_scale":0.03125)", R"("key_scale":1e100)"),
            false, false, "scale"},
        InvalidManifestCase{
            "ZeroValueScale",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("value_scale":0.0625)", R"("value_scale":0)"),
            false, false, "scale"},
        InvalidManifestCase{
            "NegativeValueScale",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("value_scale":0.0625)", R"("value_scale":-0.5)"),
            false, false, "scale"},
        InvalidManifestCase{
            "InexactValueScale",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("value_scale":0.0625)", R"("value_scale":0.1)"),
            false, false, "scale"},
        InvalidManifestCase{
            "NonzeroZeroPoint",
            ReplaceFirst(BuildV2Manifest(Config::E2B()), R"("zero_point":0)",
                         R"("zero_point":1)"),
            false, false, "zero point"},
        InvalidManifestCase{"WrongGlobalHeadDim",
                            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                                         R"("head_dim":512,"zero_point")",
                                         R"("head_dim":256,"zero_point")"),
                            false, false, "head width"},
        InvalidManifestCase{
            "WrongKeyTemplateHeads",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("key_shape_template":[1,1,"capacity",512])",
                         R"("key_shape_template":[1,2,"capacity",512])"),
            false, false, "KV shape template dimensions"},
        InvalidManifestCase{
            "WrongKeyTemplateRank",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("key_shape_template":[1,1,"capacity",512])",
                         R"("key_shape_template":[1,1,512])"),
            false, false, "KV shape template rank"},
        InvalidManifestCase{
            "TransposedValueTemplate",
            ReplaceFirst(BuildV2Manifest(Config::E2B()),
                         R"("value_shape_template":[1,1,512,"capacity"])",
                         R"("value_shape_template":[1,1,"capacity",512])"),
            false, false, "KV capacity template"},
        InvalidManifestCase{"LegacyV1RejectsE4B",
                            BuildV1Manifest(/*compact_int2=*/true),
                            /*compact_int2=*/true, /*use_e4b=*/true,
                            "legacy bundle is E2B only"},
        InvalidManifestCase{
            "LegacyV1UnexpectedHash",
            ReplaceFirst(BuildV1Manifest(/*compact_int2=*/false),
                         "ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce"
                         "27911b27e42",
                         "0000"),
            false, false, "source bundle identity"},
        InvalidManifestCase{
            "LegacyV1UnexpectedBytes",
            ReplaceFirst(BuildV1Manifest(/*compact_int2=*/false), "2583085056",
                         "1234"),
            false, false, "source bundle identity"},
        InvalidManifestCase{"LegacyV1WrongCompactStorage",
                            ReplaceFirst(BuildV1Manifest(/*compact_int2=*/true),
                                         "compact_int2_v1", "other_storage"),
                            /*compact_int2=*/true, false,
                            "experimental compact INT2 bundle"},
        InvalidManifestCase{"LegacyV1WrongInt2Count",
                            ReplaceFirst(BuildV1Manifest(/*compact_int2=*/true),
                                         R"("int2":62)", R"("int2":61)"),
                            /*compact_int2=*/true, false,
                            "62 original INT2 tensors"},
        InvalidManifestCase{
            "LegacyV1WrongFloat32Count",
            ReplaceFirst(BuildV1Manifest(/*compact_int2=*/false),
                         R"("float32":814)", R"("float32":813)"),
            false, false, "tensor dtype coverage"}),
    [](const testing::TestParamInfo<InvalidManifestCase>& info) {
      return info.param.name;
    });

TEST(BundleLoaderPublishedBundleTest, RejectsEmptyTensorListForFullBundle) {
  TempBundleDir dir("empty_tensors_full");
  ASSERT_THAT(dir.WriteText("manifest.json", BuildV2Manifest(Config::E2B())),
              IsOk());

  EXPECT_THAT(LoadPublishedBundle(dir.path(), Config::E2B(),
                                  /*embeddings_only=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tensor record count")));
}

TEST(BundleLoaderPublishedBundleTest,
     RejectsEmptyTensorListForEmbeddingsOnlyBundle) {
  TempBundleDir dir("empty_tensors_emb");
  ASSERT_THAT(dir.WriteText("manifest.json", BuildV2Manifest(Config::E2B())),
              IsOk());

  EXPECT_THAT(LoadPublishedBundle(dir.path(), Config::E2B(),
                                  /*embeddings_only=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tensor record count")));
}

TEST(BundleLoaderTensorTest,
     LoadsStandardBundleRepacksInt2WeightsAndInitializesEmbeddings) {
  TempBundleDir dir("load_standard");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());
  const Config config = TinyBundleConfig();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      LoadedTensors loaded, LoadBundleTensors(dir.path(), config, expected,
                                              BundleLoadScope::kAllTensors));

  // 6 tensors + 1 matched softcap reciprocal constant = 7 entries.
  EXPECT_THAT(loaded.weights_handle, SizeIs(7));
  ASSERT_THAT(loaded.weights_handle,
              Contains(Key("matched.logits_softcap_reciprocal")));
  ASSERT_THAT(loaded.weights_handle, Contains(Key("model.norm.weight")));
  ASSERT_THAT(loaded.weights_handle,
              Contains(Key("model.layers.0.self_attn.q_proj.weight")));
  ASSERT_THAT(loaded.weights_handle,
              Contains(Key("model.layers.0.mlp.down_proj.weight")));

  // FP32 norm weights preserve exact little-endian coefficients.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Buffer & norm_buf,
      loaded.weights_handle.at("model.norm.weight").GetBuffer());
  LockedBufferSpan<const float> norm_vals = norm_buf.Lock().As<const float>();
  EXPECT_THAT(
      absl::MakeConstSpan(norm_vals.data(), norm_vals.size()),
      ElementsAre(FloatEq(1.0f), FloatEq(1.5f), FloatEq(-0.5f), FloatEq(2.0f)));

  // INT8 linear weights preserve type and raw bytes.
  const TensorHandle& q_proj =
      loaded.weights_handle.at("model.layers.0.self_attn.q_proj.weight");
  EXPECT_EQ(q_proj.GetType(), Type::kI8);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & q_buf, q_proj.GetBuffer());
  LockedBufferSpan<const int8_t> q_vals = q_buf.Lock().As<const int8_t>();
  EXPECT_THAT(absl::MakeConstSpan(q_vals.data(), q_vals.size()),
              ElementsAre(1, 2, 3, 4, -1, -2, -3, -4));

  // Widened INT4 down_proj with INT2 provenance must be repacked into kI2.
  const TensorHandle& down_proj =
      loaded.weights_handle.at("model.layers.0.mlp.down_proj.weight");
  EXPECT_EQ(down_proj.GetType(), Type::kI2);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & down_buf, down_proj.GetBuffer());
  LockedBufferSpan<const uint8_t> down_bytes =
      down_buf.Lock().As<const uint8_t>();
  // Pre-calculated 2-bit two's-complement bytes (LSB first):
  // Row 0 [-2, -1, 0, 1] -> 0b01001110 (0x4e);
  // Row 1 [1, 0, -1, -2] -> 0b10110001 (0xb1).
  EXPECT_THAT(absl::MakeConstSpan(down_bytes.data(), down_bytes.size()),
              ElementsAre(0x4e, 0xb1));

  // Verify dequantized token and per-layer embedding lookups from mmap-backed
  // INT4 tables: token row 0 = [1, -2, 3, -4] * 0.5 = [0.5, -1.0, 1.5, -2.0],
  // PLE row 0 repeats [1, -1] * 0.25 = [0.25, -0.25].
  ASSERT_NE(loaded.token_embedding, nullptr);
  ASSERT_NE(loaded.emb_per_layer_table, nullptr);
  std::vector<float> tok0(4, 0.0f);
  ASSERT_THAT(loaded.token_embedding->Lookup({0}, absl::MakeSpan(tok0)),
              IsOk());
  EXPECT_THAT(tok0, ElementsAre(FloatEq(0.5f), FloatEq(-1.0f), FloatEq(1.5f),
                                FloatEq(-2.0f)));
  std::vector<float> ple0(256, 0.0f);
  ASSERT_THAT(loaded.emb_per_layer_table->Lookup({0}, absl::MakeSpan(ple0)),
              IsOk());
  EXPECT_THAT(absl::MakeConstSpan(ple0).subspan(0, 4),
              ElementsAre(FloatEq(0.25f), FloatEq(-0.25f), FloatEq(0.25f),
                          FloatEq(-0.25f)));
}

TEST(BundleLoaderTensorTest, KeepsNativeInt4WeightWithoutInt2Repacking) {
  TempBundleDir dir("native_int4");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());
  ASSERT_THAT(
      dir.WriteText(
          "manifest.json",
          ReplaceFirst(manifest_json,
                       R"("sources":[{"dtype":"INT2","source_bytes":2}])",
                       R"("sources":[{"dtype":"INT4","source_bytes":4}])")),
      IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      LoadedTensors loaded,
      LoadBundleTensors(dir.path(), TinyBundleConfig(), expected,
                        BundleLoadScope::kAllTensors));

  ASSERT_THAT(loaded.weights_handle,
              Contains(Key("model.layers.0.mlp.down_proj.weight")));
  EXPECT_EQ(
      loaded.weights_handle.at("model.layers.0.mlp.down_proj.weight").GetType(),
      Type::kI4);
}

TEST(BundleLoaderTensorTest,
     EmbeddingsOnlyScopeSkipsNonEmbeddingWeightsAndLoadsTables) {
  TempBundleDir dir("load_emb_only");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      LoadedTensors loaded,
      LoadBundleTensors(dir.path(), TinyBundleConfig(), expected,
                        BundleLoadScope::kEmbeddingsOnly));

  EXPECT_THAT(loaded.weights_handle, SizeIs(3));
  EXPECT_THAT(loaded.weights_handle,
              Contains(Key("model.embed_tokens.weight")));
  EXPECT_THAT(loaded.weights_handle,
              Contains(Key("model.embed_tokens_per_layer.weight")));
  EXPECT_THAT(loaded.weights_handle,
              Contains(Key("matched.logits_softcap_reciprocal")));
  EXPECT_THAT(loaded.weights_handle, Not(Contains(Key("model.norm.weight"))));
  ASSERT_NE(loaded.token_embedding, nullptr);
  ASSERT_NE(loaded.emb_per_layer_table, nullptr);
  std::vector<float> tok0(4, 0.0f);
  ASSERT_THAT(loaded.token_embedding->Lookup({0}, absl::MakeSpan(tok0)),
              IsOk());
  EXPECT_THAT(tok0, ElementsAre(FloatEq(0.5f), FloatEq(-1.0f), FloatEq(1.5f),
                                FloatEq(-2.0f)));
}

TEST(BundleLoaderTensorTest,
     LoadsCompactInt2BundleWithPlePartitionsAndSemanticSoftcapName) {
  TempBundleDir dir("load_compact_i2");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticCompactInt2Bundle(dir, expected, manifest_json),
              IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      LoadedTensors loaded,
      LoadBundleTensors(dir.path(), TinyBundleConfig(), expected,
                        BundleLoadScope::kAllTensors));

  ASSERT_THAT(loaded.weights_handle,
              Contains(Key("model.embed_tokens.weight")));
  ASSERT_THAT(loaded.weights_handle,
              Contains(Key("model.embed_tokens_per_layer.weight")));
  EXPECT_EQ(loaded.weights_handle.at("model.embed_tokens.weight").GetType(),
            Type::kI2);
  EXPECT_EQ(
      loaded.weights_handle.at("model.embed_tokens_per_layer.weight").GetType(),
      Type::kI2);
  ASSERT_NE(loaded.token_embedding, nullptr);
  // Token 0 row = [1, -1, -2, 0] * 0.5 = [0.5, -0.5, -1.0, 0.0].
  std::vector<float> tok0(4, 0.0f);
  ASSERT_THAT(loaded.token_embedding->Lookup({0}, absl::MakeSpan(tok0)),
              IsOk());
  EXPECT_THAT(tok0, ElementsAre(FloatEq(0.5f), FloatEq(-0.5f), FloatEq(-1.0f),
                                FloatEq(0.0f)));
}

TEST(BundleLoaderTensorTest, RejectsNonPositiveConfigDimensions) {
  TempBundleDir dir("bad_config_dims");
  Config bad_config = TinyBundleConfig();
  bad_config.embed_dim = 0;

  EXPECT_THAT(
      LoadBundleTensors(dir.path(), bad_config, {}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
}

TEST(BundleLoaderTensorTest, RejectsMissingRequiredEmbeddingInExpectedMap) {
  TempBundleDir dir("missing_req_emb");
  ASSERT_THAT(dir.CreateSubdir("tensors"), IsOk());
  ASSERT_THAT(
      dir.WriteFile("tensors/norm.bin", PackFloats({1.0f, 1.0f, 1.0f, 1.0f})),
      IsOk());
  ASSERT_THAT(dir.WriteFile("tensors/softcap.bin",
                            PackFloats({kExactSoftcapReciprocal})),
              IsOk());
  ASSERT_THAT(
      dir.WriteText(
          "manifest.json",
          R"({"schema_version":2,"status":"complete","unmapped_float_coefficients":[],)"
          R"("tensors":[{"name":"model.norm.weight","dtype":"float32","shape":[4],"bytes":16,)"
          R"("encoding":"little_endian_float32","file":"tensors/norm.bin"}],)"
          R"("constants":[{"name":"decode.tensor_36","dtype":"float32","shape":[],"bytes":4,)"
          R"("file":"tensors/softcap.bin"}]})"),
      IsOk());
  const absl::flat_hash_map<std::string, ExpectedTensor> expected = {
      {"model.norm.weight", {"float32", {4}}},
  };

  EXPECT_THAT(LoadBundleTensors(dir.path(), TinyBundleConfig(), expected),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("required embedding")));
}

TEST(BundleLoaderTensorTest, RejectsDirectoryAsTensorFile) {
  TempBundleDir dir("dir_as_file");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());
  ASSERT_THAT(dir.CreateSubdir("tensors/fake_dir"), IsOk());
  ASSERT_THAT(dir.WriteText("manifest.json",
                            ReplaceFirst(manifest_json, "tensors/norm.bin",
                                         "tensors/fake_dir")),
              IsOk());

  EXPECT_THAT(LoadBundleTensors(dir.path(), TinyBundleConfig(), expected),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("file type or exact byte count")));
}

TEST(BundleLoaderTensorTest, RejectsNonFiniteFp32Coefficient) {
  TempBundleDir dir("nan_fp32");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());
  ASSERT_THAT(
      dir.WriteFile("tensors/norm.bin",
                    PackFloats({1.0f, std::numeric_limits<float>::quiet_NaN(),
                                0.0f, 1.0f})),
      IsOk());

  EXPECT_THAT(LoadBundleTensors(dir.path(), TinyBundleConfig(), expected),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("nonfinite coefficient")));
}

TEST(BundleLoaderTensorTest, RejectsNonPositiveActivationScale) {
  TempBundleDir dir("zero_act_scale");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());
  ASSERT_THAT(dir.WriteFile("tensors/act_scale.bin", PackFloats({0.0f})),
              IsOk());

  EXPECT_THAT(LoadBundleTensors(dir.path(), TinyBundleConfig(), expected),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("nonpositive activation scale")));
}

TEST(BundleLoaderTensorTest, RejectsNonPositiveQuantizationScale) {
  TempBundleDir dir("neg_quant_scale");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());
  ASSERT_THAT(
      dir.WriteFile("tensors/q_proj_scales.bin", PackFloats({0.125f, -0.25f})),
      IsOk());

  EXPECT_THAT(LoadBundleTensors(dir.path(), TinyBundleConfig(), expected),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("invalid quantization scale")));
}

TEST(BundleLoaderTensorTest, RejectsWidenedInt4CodeOutsideInt2Range) {
  TempBundleDir dir("int2_code_oob");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());
  // Value +3 is valid INT4 but outside INT2 [-2, 1].
  ASSERT_THAT(dir.WriteFile("tensors/down_proj.bin",
                            {PackTwoInt4(3, 0), PackTwoInt4(0, 0),
                             PackTwoInt4(0, 0), PackTwoInt4(0, 0)}),
              IsOk());

  EXPECT_THAT(LoadBundleTensors(dir.path(), TinyBundleConfig(), expected),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("outside [-2,1]")));
}

TEST(BundleLoaderTensorTest, RejectsUnexpectedSoftcapConstantValue) {
  TempBundleDir dir("wrong_softcap_val");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());
  ASSERT_THAT(dir.WriteFile("tensors/softcap.bin", PackFloats({0.05f})),
              IsOk());

  EXPECT_THAT(LoadBundleTensors(dir.path(), TinyBundleConfig(), expected),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("softcap reciprocal value")));
}

TEST(BundleLoaderTensorTest, RejectsUnsupportedExpectedDtype) {
  TempBundleDir dir("unsupported_dtype");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());
  expected["model.norm.weight"] = {"bfloat16", {4}};
  ASSERT_THAT(
      dir.WriteText(
          "manifest.json",
          ReplaceFirst(manifest_json,
                       R"("name":"model.norm.weight","dtype":"float32")",
                       R"("name":"model.norm.weight","dtype":"bfloat16")")),
      IsOk());

  EXPECT_THAT(LoadBundleTensors(dir.path(), TinyBundleConfig(), expected),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported dtype")));
}

TEST(BundleLoaderTensorTest, RejectsRank1QuantizedTensor) {
  TempBundleDir dir("rank1_quant");
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
              IsOk());
  expected["model.layers.0.self_attn.q_proj.weight"] = {"int8", {8}};
  ASSERT_THAT(
      dir.WriteText(
          "manifest.json",
          ReplaceFirst(
              manifest_json,
              R"("name":"model.layers.0.self_attn.q_proj.weight","dtype":"int8","shape":[2,4])",
              R"("name":"model.layers.0.self_attn.q_proj.weight","dtype":"int8","shape":[8])")),
      IsOk());

  EXPECT_THAT(
      LoadBundleTensors(dir.path(), TinyBundleConfig(), expected),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("rank >= 2")));
}

struct TensorMutationCase {
  std::string name;
  bool compact_bundle;
  std::string before;
  std::string after;
  std::string expected_error;
};

class BundleLoaderTensorValidationTest
    : public testing::TestWithParam<TensorMutationCase> {};

TEST_P(BundleLoaderTensorValidationTest, RejectsInvalidTensorRecord) {
  const TensorMutationCase& tc = GetParam();
  TempBundleDir dir(tc.name);
  absl::flat_hash_map<std::string, ExpectedTensor> expected;
  std::string manifest_json;
  if (tc.compact_bundle) {
    ASSERT_THAT(
        PopulateSyntheticCompactInt2Bundle(dir, expected, manifest_json),
        IsOk());
  } else {
    ASSERT_THAT(PopulateSyntheticStandardBundle(dir, expected, manifest_json),
                IsOk());
  }
  ASSERT_THAT(dir.WriteText("manifest.json",
                            ReplaceFirst(manifest_json, tc.before, tc.after)),
              IsOk());

  EXPECT_THAT(LoadBundleTensors(dir.path(), TinyBundleConfig(), expected),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(tc.expected_error)));
}

INSTANTIATE_TEST_SUITE_P(
    TensorValidation, BundleLoaderTensorValidationTest,
    testing::Values(
        TensorMutationCase{"AbsoluteFilePath", false, "tensors/norm.bin",
                           "/etc/passwd", "invalid relative file path"},
        TensorMutationCase{"ParentTraversalPath", false, "tensors/norm.bin",
                           "tensors/../norm.bin", "unsafe relative file path"},
        TensorMutationCase{"DotSegmentPath", false, "tensors/norm.bin",
                           "tensors/./norm.bin", "unsafe relative file path"},
        TensorMutationCase{"MissingWeightFile", false, "tensors/norm.bin",
                           "tensors/does_not_exist.bin", "open "},
        TensorMutationCase{"WrongFileByteSize", false, "tensors/norm.bin",
                           "tensors/act_scale.bin",
                           "file type or exact byte count"},
        TensorMutationCase{"RankExceedsEight", false,
                           R"("shape":[4],"bytes":16)",
                           R"("shape":[1,1,1,1,1,1,1,1,4],"bytes":16)",
                           "unsupported tensor rank"},
        TensorMutationCase{"ZeroShapeDimension", false,
                           R"("shape":[4],"bytes":16)",
                           R"("shape":[0],"bytes":16)",
                           "nonpositive or oversized shape dimension"},
        TensorMutationCase{"DtypeMismatch", false,
                           R"("name":"model.norm.weight","dtype":"float32")",
                           R"("name":"model.norm.weight","dtype":"int8")",
                           "dtype/shape mismatch"},
        TensorMutationCase{"ByteCountMismatch", false,
                           R"("shape":[4],"bytes":16)",
                           R"("shape":[4],"bytes":12)", "tensor byte count"},
        TensorMutationCase{"UnsupportedEncoding", false,
                           "little_endian_float32", "big_endian_float32",
                           "unsupported encoding"},
        TensorMutationCase{"UnexpectedFp32Quantization", false,
                           R"("file":"tensors/norm.bin")",
                           R"("file":"tensors/norm.bin","quantization":{})",
                           "unexpected FP32 quantization"},
        TensorMutationCase{
            "NonzeroQuantizedAxis", false, R"("quantized_dimension":0)",
            R"("quantized_dimension":1)", "quantization axis/zero point"},
        TensorMutationCase{"MultipleZeroPoints", false, R"("zero_points":[0])",
                           R"("zero_points":[0,0])",
                           "quantization axis/zero point"},
        TensorMutationCase{"NonzeroQuantZeroPoint", false,
                           R"("zero_points":[0])", R"("zero_points":[1])",
                           "nonzero zero point"},
        TensorMutationCase{"WrongScaleKind", false, R"("kind":"per_channel")",
                           R"("kind":"blockwise")", "scale layout/kind"},
        TensorMutationCase{"InvalidBlockSize", false, R"("block_size":256)",
                           R"("block_size":128)", "invalid block size"},
        TensorMutationCase{"ScaleByteCountMismatch", false,
                           R"("scales_bytes":8)", R"("scales_bytes":4)",
                           "quantization scale byte count"},
        TensorMutationCase{"WidenedInt4EmptySources", false,
                           R"("sources":[{"dtype":"INT2","source_bytes":2}])",
                           R"("sources":[])", "one original quantized tensor"},
        TensorMutationCase{"WidenedInt4WrongSourceBytes", false,
                           R"("sources":[{"dtype":"INT2","source_bytes":2}])",
                           R"("sources":[{"dtype":"INT2","source_bytes":3}])",
                           "original INT2 size mismatch"},
        TensorMutationCase{
            "NonObjectTensorRecord", false,
            R"({"name":"model.norm.weight","dtype":"float32","shape":[4],"bytes":16,"encoding":"little_endian_float32","file":"tensors/norm.bin"})",
            "123", "tensor record must be an object"},
        TensorMutationCase{"UnexpectedTensorName", false,
                           R"("name":"model.norm.weight")",
                           R"("name":"model.unknown.weight")",
                           "unexpected or duplicate tensor"},
        TensorMutationCase{
            "NonObjectConstantRecord", false,
            R"({"name":"ignored.const","dtype":"float32","shape":[],"bytes":4,"file":"tensors/other_const.bin"})",
            "99", "constant record must be an object"},
        TensorMutationCase{"DuplicateSoftcapConstant", false,
                           R"("name":"ignored.const")",
                           R"("name":"decode.tensor_36")",
                           "duplicate logits softcap reciprocal"},
        TensorMutationCase{
            "MissingSoftcapConstant", false, R"("name":"decode.tensor_36")",
            R"("name":"other.softcap")", "missing logits softcap reciprocal"},
        TensorMutationCase{
            "CompactInt2WrongSourceCount", true,
            R"("sources":[{"dtype":"INT2","shape":[2,4],"source_bytes":2}])",
            R"("sources":[])", "one original INT2 tensor"},
        TensorMutationCase{
            "CompactInt2ProvenanceMismatch", true,
            R"("sources":[{"dtype":"INT2","shape":[2,4],"source_bytes":2}])",
            R"("sources":[{"dtype":"INT2","shape":[2,4],"source_bytes":4}])",
            "compact INT2 provenance mismatch"},
        TensorMutationCase{
            "Int2PleWrongPartitionCount", true,
            R"("sources":[{"destination_layer_partition":0,"dtype":"INT2","shape":[2,256],"source_bytes":128}])",
            R"("sources":[])", "INT2 PLE partition count"},
        TensorMutationCase{
            "Int2PleNonObjectSource", true,
            R"("sources":[{"destination_layer_partition":0,"dtype":"INT2","shape":[2,256],"source_bytes":128}])",
            R"("sources":[7])", "INT2 PLE source record"},
        TensorMutationCase{"Int2PleProvenanceMismatch", true,
                           R"("destination_layer_partition":0)",
                           R"("destination_layer_partition":1)",
                           "INT2 PLE provenance"}),
    [](const testing::TestParamInfo<TensorMutationCase>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace litert::tensor::examples::experimental::gemma4::cpu
