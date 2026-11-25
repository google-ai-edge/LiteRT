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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {
namespace {
using ::litert::tensor::examples::gemma4::Config;
using ::testing::HasSubstr;

std::string Manifest(const Config& c) {
  const int owners = c.num_kv_heads == 2 ? 24 : 15;
  std::string text = absl::StrCat(
      R"({"schema_version":2,"status":"complete","unmapped_float_coefficients":[],)",
      R"("experimental_storage":"compact_int2_v1","model_config":{"variant":")",
      c.num_kv_heads == 2 ? "e4b" : "e2b", R"(","embed_dim":)", c.embed_dim,
      R"(,"hidden_dim":)", c.hidden_dim, R"(,"num_layers":)", c.num_layers,
      R"(,"num_heads":)", c.num_heads, R"(,"num_kv_heads":)", c.num_kv_heads,
      R"(,"vocab_size":)", c.vocab_size, R"(,"num_kv_owners":)", owners,
      R"(,"attention_pattern_size":)", c.attention_pattern_size,
      R"(,"head_dim":)", c.head_dim, R"(,"global_key_size":)",
      c.global_key_size, R"(,"per_layer_input_dim":)", c.per_layer_input_dim,
      R"(,"sliding_window_size":)", c.sliding_window_size,
      R"(},"kv_cache_specs":[)");
  // Deliberately reversed: the public result must be indexed by owner ID.
  for (int owner = owners - 1; owner >= 0; --owner) {
    if (owner != owners - 1) text += ',';
    const int dim =
        c.GetLayerType(owner) == Config::LayerType::kGlobal ? 512 : 256;
    absl::StrAppend(
        &text, R"({"owner":)", owner, R"(,"head_dim":)", dim,
        R"(,"zero_point":0,"key_scale":0.03125,"value_scale":0.0625,)",
        R"("key_shape_template":[1,)", c.num_kv_heads, R"(,"capacity",)", dim,
        R"(],"value_shape_template":[1,)", c.num_kv_heads, ",", dim,
        R"(,"capacity"]})");
  }
  return text + R"(],"tensors":[]})";
}

class BundleLoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string pattern = testing::TempDir() + "/gemma4-bundle-XXXXXX";
    ASSERT_NE(mkdtemp(pattern.data()), nullptr);
    directory_ = pattern;
  }
  void TearDown() override { std::filesystem::remove_all(directory_); }
  void Write(const std::string& text) {
    std::ofstream file(directory_ + "/manifest.json", std::ios::binary);
    ASSERT_TRUE(file);
    file << text;
    ASSERT_TRUE(file);
  }
  std::string Changed(const std::string& before, const std::string& after) {
    std::string text = Manifest(Config::E2B());
    const auto at = text.find(before);
    EXPECT_NE(at, std::string::npos);
    if (at != std::string::npos) text.replace(at, before.size(), after);
    return text;
  }
  std::string directory_;
};

TEST_F(BundleLoaderTest, LoadsAndSortsE2BOwnersWithoutTensorFiles) {
  Write(Manifest(Config::E2B()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto specs, LoadPublishedKvSpecs(directory_, Config::E2B()));
  ASSERT_EQ(specs.size(), 15);
  for (int i = 0; i < specs.size(); ++i) {
    EXPECT_EQ(specs[i].owner, i);
    EXPECT_EQ(specs[i].num_heads, 1);
    EXPECT_EQ(specs[i].head_dim, (i + 1) % 5 == 0 ? 512 : 256);
    EXPECT_FLOAT_EQ(specs[i].key_scale, 0.03125f);
    EXPECT_FLOAT_EQ(specs[i].value_scale, 0.0625f);
  }
}

TEST_F(BundleLoaderTest, LoadsE4BMultiHeadOwnersOnlyWithCompactOptIn) {
  Write(Manifest(Config::E4B()));
  EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E4B()).ok());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto specs, LoadPublishedKvSpecs(directory_, Config::E4B(), true));
  ASSERT_EQ(specs.size(), 24);
  for (int i = 0; i < specs.size(); ++i) {
    EXPECT_EQ(specs[i].owner, i);
    EXPECT_EQ(specs[i].num_heads, 2);
    EXPECT_EQ(specs[i].head_dim, (i + 1) % 6 == 0 ? 512 : 256);
  }
}

TEST_F(BundleLoaderTest, RejectsMissingMalformedAndTrailingData) {
  EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E2B()).ok());
  for (const std::string& text :
       {std::string(), std::string("[]"), std::string("{"),
        Manifest(Config::E2B()) + "garbage"}) {
    SCOPED_TRACE(text.substr(0, 80));
    Write(text);
    EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E2B()).ok());
  }
}

TEST_F(BundleLoaderTest, RejectsIncompleteOrUnsupportedSchema) {
  for (const auto& [before, after] :
       std::vector<std::pair<std::string, std::string>>{
           {R"("schema_version":2)", R"("schema_version":3)"},
           {R"("schema_version":2)", R"("schema_version":2.5)"},
           {R"("status":"complete")", R"("status":"partial")"},
           {R"("unmapped_float_coefficients":[])",
            R"("unmapped_float_coefficients":["unknown"])"}}) {
    SCOPED_TRACE(after);
    Write(Changed(before, after));
    EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E2B()).ok());
  }
}

TEST_F(BundleLoaderTest, RejectsArchitectureMismatch) {
  Write(Changed(R"("embed_dim":1536)", R"("embed_dim":1537)"));
  EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E2B()).ok());
  Write(Manifest(Config::E4B()));
  EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E2B(), true).ok());
  Config unsupported = Config::E2B();
  unsupported.sliding_window_size = 8;
  EXPECT_FALSE(LoadPublishedKvSpecs(directory_, unsupported).ok());
}

TEST_F(BundleLoaderTest, RejectsDuplicateOutOfRangeAndFractionalOwners) {
  for (const char* replacement : {"13", "15", "-1", "0.5"}) {
    SCOPED_TRACE(replacement);
    Write(Changed(R"("owner":14)", absl::StrCat(R"("owner":)", replacement)));
    EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E2B()).ok());
  }
}

TEST_F(BundleLoaderTest, RejectsInvalidOrInexactQuantization) {
  for (const char* scale : {"0", "-0.5", "0.1", "1e100"}) {
    SCOPED_TRACE(scale);
    Write(Changed(R"("key_scale":0.03125)",
                  absl::StrCat(R"("key_scale":)", scale)));
    EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E2B()).ok());
  }
  Write(Changed(R"("zero_point":0)", R"("zero_point":1)"));
  EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E2B()).ok());
}

TEST_F(BundleLoaderTest, RejectsIncorrectKeyAndTransposedValueTemplates) {
  for (const auto& [before, after] :
       std::vector<std::pair<std::string, std::string>>{
           {R"("head_dim":512,"zero_point")", R"("head_dim":256,"zero_point")"},
           {R"("key_shape_template":[1,1,"capacity",512])",
            R"("key_shape_template":[1,2,"capacity",512])"},
           {R"("key_shape_template":[1,1,"capacity",512])",
            R"("key_shape_template":[1,1,512])"},
           {R"("value_shape_template":[1,1,512,"capacity"])",
            R"("value_shape_template":[1,1,"capacity",512])"}}) {
    SCOPED_TRACE(after);
    Write(Changed(before, after));
    EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E2B()).ok());
  }
}

TEST_F(BundleLoaderTest, RejectsMissingTensorCoverageEvenForEmbeddingsOnly) {
  Write(Manifest(Config::E2B()));
  for (bool embeddings_only : {false, true}) {
    const auto loaded =
        LoadPublishedBundle(directory_, Config::E2B(), embeddings_only);
    ASSERT_FALSE(loaded.ok());
    EXPECT_THAT(loaded.status().message(), HasSubstr("tensor record count"));
  }
}

TEST_F(BundleLoaderTest, RejectsIncorrectCompactStorageTag) {
  Write(Changed("compact_int2_v1", "unknown_storage"));
  EXPECT_FALSE(LoadPublishedKvSpecs(directory_, Config::E2B(), true).ok());
}

}  // namespace
}  // namespace litert::tensor::examples::experimental::gemma4::cpu
