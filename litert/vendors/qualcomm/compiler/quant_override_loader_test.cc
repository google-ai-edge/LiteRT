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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/compiler/quant_override_loader.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>

#include "QnnTypes.h"  // from @qairt
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json

namespace litert::qnn {
namespace {

// ============================================================
// LoadQuantOverridesV1 tests
// ============================================================

TEST(LoadQuantOverridesV1Test, ScaleOffsetAsymmetricDefaultsToUint8) {
  const auto json = nlohmann::json::parse(R"({
    "activation_encodings": {
      "tensor_a": [{"scale": 0.01, "offset": -128.0}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  ASSERT_EQ(result.size(), 1);
  const auto& enc = result.at("tensor_a");
  EXPECT_FLOAT_EQ(enc.scale, 0.01f);
  // AIMET offset = -zero_point, so zero_point = -(-128) = 128
  EXPECT_EQ(enc.zero_point, 128);
  EXPECT_EQ(enc.data_type, QNN_DATATYPE_UFIXED_POINT_8);
}

TEST(LoadQuantOverridesV1Test, ScaleOffsetSymmetricIsSymmetricTrue) {
  const auto json = nlohmann::json::parse(R"({
    "activation_encodings": {
      "tensor_s": [{"scale": 0.005, "offset": 0.0, "is_symmetric": "True"}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  ASSERT_EQ(result.size(), 1);
  const auto& enc = result.at("tensor_s");
  EXPECT_FLOAT_EQ(enc.scale, 0.005f);
  EXPECT_EQ(enc.zero_point, 0);
  EXPECT_EQ(enc.data_type, QNN_DATATYPE_SFIXED_POINT_8);
}

TEST(LoadQuantOverridesV1Test, ScaleOffsetSymmetricLowercaseTrue) {
  const auto json = nlohmann::json::parse(R"({
    "param_encodings": {
      "weight": [{"scale": 0.002, "offset": 0.0, "is_symmetric": "true"}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result.at("weight").data_type, QNN_DATATYPE_SFIXED_POINT_8);
}

TEST(LoadQuantOverridesV1Test, ParamEncodingsIsLoaded) {
  const auto json = nlohmann::json::parse(R"({
    "param_encodings": {
      "bias": [{"scale": 0.0001, "offset": 0.0}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_FLOAT_EQ(result.at("bias").scale, 0.0001f);
}

TEST(LoadQuantOverridesV1Test, BothSectionsLoaded) {
  const auto json = nlohmann::json::parse(R"({
    "activation_encodings": {
      "act": [{"scale": 0.01, "offset": -128.0}]
    },
    "param_encodings": {
      "weight": [{"scale": 0.002, "offset": 0.0}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  EXPECT_EQ(result.size(), 2);
  EXPECT_TRUE(result.contains("act"));
  EXPECT_TRUE(result.contains("weight"));
}

TEST(LoadQuantOverridesV1Test, Bitwidth16AsymmetricIsUint16) {
  const auto json = nlohmann::json::parse(R"({
    "activation_encodings": {
      "tensor_16u": [{"scale": 0.0001, "offset": -32768.0, "bitwidth": 16}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result.at("tensor_16u").data_type, QNN_DATATYPE_UFIXED_POINT_16);
}

TEST(LoadQuantOverridesV1Test, Bitwidth16SymmetricIsSfixed16) {
  const auto json = nlohmann::json::parse(R"({
    "activation_encodings": {
      "tensor_16s": [{"scale": 0.0001, "offset": 0.0, "bitwidth": 16,
                      "is_symmetric": "True"}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result.at("tensor_16s").data_type, QNN_DATATYPE_SFIXED_POINT_16);
}

TEST(LoadQuantOverridesV1Test, MinMaxFormAsymmetric) {
  // min=-1.0, max=2.0 → asymmetric
  // scale = (2.0 - (-1.0)) / 255.0 = 3.0/255.0
  // zero_point = round(1.0 / (3.0/255.0)) = round(85.0) = 85
  const auto json = nlohmann::json::parse(R"({
    "activation_encodings": {
      "t": [{"min": -1.0, "max": 2.0}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  ASSERT_EQ(result.size(), 1);
  const auto& enc = result.at("t");
  const float expected_scale = 3.0f / 255.0f;
  EXPECT_NEAR(enc.scale, expected_scale, 1e-7f);
  EXPECT_EQ(enc.zero_point, 85);
  EXPECT_EQ(enc.data_type, QNN_DATATYPE_UFIXED_POINT_8);
}

TEST(LoadQuantOverridesV1Test, MinMaxFormSymmetricDetectedAutomatically) {
  // min=-1.0, max=1.0 → |min+max| = 0 < 1e-6 * |max| = 1e-6 → symmetric
  const auto json = nlohmann::json::parse(R"({
    "activation_encodings": {
      "t": [{"min": -1.0, "max": 1.0}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result.at("t").data_type, QNN_DATATYPE_SFIXED_POINT_8);
}

TEST(LoadQuantOverridesV1Test, MinMaxFormIsSymmetricOverridesAutoDetect) {
  // min=-1.0, max=2.0 would be detected as asymmetric, but is_symmetric=True
  // overrides that
  const auto json = nlohmann::json::parse(R"({
    "activation_encodings": {
      "t": [{"min": -1.0, "max": 2.0, "is_symmetric": "True"}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result.at("t").data_type, QNN_DATATYPE_SFIXED_POINT_8);
}

TEST(LoadQuantOverridesV1Test, EntryWithoutScaleOrMinIsSkipped) {
  const auto json = nlohmann::json::parse(R"({
    "activation_encodings": {
      "bad": [{"bitwidth": 8}],
      "good": [{"scale": 0.01, "offset": 0.0}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);

  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(result.contains("good"));
  EXPECT_FALSE(result.contains("bad"));
}

TEST(LoadQuantOverridesV1Test, MissingSectionIsNotAnError) {
  const auto json = nlohmann::json::parse(R"({
    "activation_encodings": {
      "t": [{"scale": 0.01, "offset": 0.0}]
    }
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);
  // Should succeed even with no "param_encodings"
  EXPECT_EQ(result.size(), 1);
}

TEST(LoadQuantOverridesV1Test, EmptyJsonProducesEmptyMap) {
  const auto json = nlohmann::json::parse(R"({})");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV1(json, result);
  EXPECT_TRUE(result.empty());
}

// ============================================================
// LoadQuantOverridesV2 tests
// ============================================================

TEST(LoadQuantOverridesV2Test, Uint8Entry) {
  const auto json = nlohmann::json::parse(R"({
    "version": "2.0.0",
    "encodings": [
      {"name": "t", "output_dtype": "uint8", "y_scale": 0.01, "y_zero_point": 128}
    ]
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV2(json, result);

  ASSERT_EQ(result.size(), 1);
  const auto& enc = result.at("t");
  EXPECT_FLOAT_EQ(enc.scale, 0.01f);
  EXPECT_EQ(enc.zero_point, 128);
  EXPECT_EQ(enc.data_type, QNN_DATATYPE_UFIXED_POINT_8);
}

TEST(LoadQuantOverridesV2Test, Int8EntryDefaultZeroPoint) {
  // y_zero_point absent → defaults to 0 (symmetric)
  const auto json = nlohmann::json::parse(R"({
    "version": "2.0.0",
    "encodings": [
      {"name": "w", "output_dtype": "int8", "y_scale": 0.005}
    ]
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV2(json, result);

  ASSERT_EQ(result.size(), 1);
  const auto& enc = result.at("w");
  EXPECT_EQ(enc.zero_point, 0);
  EXPECT_EQ(enc.data_type, QNN_DATATYPE_SFIXED_POINT_8);
}

TEST(LoadQuantOverridesV2Test, Uint16Entry) {
  const auto json = nlohmann::json::parse(R"({
    "version": "2.0.0",
    "encodings": [
      {"name": "t16", "output_dtype": "uint16", "y_scale": 0.0001, "y_zero_point": 32768}
    ]
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV2(json, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result.at("t16").data_type, QNN_DATATYPE_UFIXED_POINT_16);
  EXPECT_EQ(result.at("t16").zero_point, 32768);
}

TEST(LoadQuantOverridesV2Test, Int16Entry) {
  const auto json = nlohmann::json::parse(R"({
    "version": "2.0.0",
    "encodings": [
      {"name": "s16", "output_dtype": "int16", "y_scale": 0.0001}
    ]
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV2(json, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result.at("s16").data_type, QNN_DATATYPE_SFIXED_POINT_16);
}

TEST(LoadQuantOverridesV2Test, Int32Entry) {
  const auto json = nlohmann::json::parse(R"({
    "version": "2.0.0",
    "encodings": [
      {"name": "b", "output_dtype": "int32", "y_scale": 0.00001}
    ]
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV2(json, result);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result.at("b").data_type, QNN_DATATYPE_SFIXED_POINT_32);
}

TEST(LoadQuantOverridesV2Test, UnknownDtypeIsSkipped) {
  const auto json = nlohmann::json::parse(R"({
    "version": "2.0.0",
    "encodings": [
      {"name": "bad", "output_dtype": "float32", "y_scale": 1.0},
      {"name": "good", "output_dtype": "uint8", "y_scale": 0.01}
    ]
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV2(json, result);

  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(result.contains("good"));
  EXPECT_FALSE(result.contains("bad"));
}

TEST(LoadQuantOverridesV2Test, MissingRequiredFieldsSkipsEntry) {
  const auto json = nlohmann::json::parse(R"({
    "version": "2.0.0",
    "encodings": [
      {"output_dtype": "uint8", "y_scale": 0.01},
      {"name": "no_dtype", "y_scale": 0.01},
      {"name": "no_scale", "output_dtype": "uint8"},
      {"name": "complete", "output_dtype": "uint8", "y_scale": 0.01}
    ]
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV2(json, result);

  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(result.contains("complete"));
}

TEST(LoadQuantOverridesV2Test, MultipleEntriesAllLoaded) {
  const auto json = nlohmann::json::parse(R"({
    "version": "2.0.0",
    "encodings": [
      {"name": "a", "output_dtype": "uint8", "y_scale": 0.01, "y_zero_point": 128},
      {"name": "b", "output_dtype": "int8",  "y_scale": 0.005},
      {"name": "c", "output_dtype": "int16", "y_scale": 0.0002}
    ]
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV2(json, result);

  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result.at("a").data_type, QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_EQ(result.at("b").data_type, QNN_DATATYPE_SFIXED_POINT_8);
  EXPECT_EQ(result.at("c").data_type, QNN_DATATYPE_SFIXED_POINT_16);
}

TEST(LoadQuantOverridesV2Test, EmptyEncodingsArrayProducesEmptyMap) {
  const auto json = nlohmann::json::parse(R"({
    "version": "2.0.0",
    "encodings": []
  })");
  absl::flat_hash_map<std::string, QuantEncoding> result;
  LoadQuantOverridesV2(json, result);
  EXPECT_TRUE(result.empty());
}

// ============================================================
// LoadQuantOverrides (file-based dispatcher) tests
// ============================================================

TEST(LoadQuantOverridesTest, EmptyPathReturnsEmptyMap) {
  const auto result = LoadQuantOverrides("");
  EXPECT_TRUE(result.empty());
}

TEST(LoadQuantOverridesTest, NonExistentFileReturnsEmptyMap) {
  const auto result = LoadQuantOverrides("/nonexistent/path/file.json");
  EXPECT_TRUE(result.empty());
}

TEST(LoadQuantOverridesTest, InvalidJsonReturnsEmptyMap) {
  char tmpfile[] = "/tmp/quant_override_test_invalid_XXXXXX";
  int fd = mkstemp(tmpfile);
  ASSERT_NE(fd, -1);
  close(fd);
  std::ofstream f(tmpfile);
  f << "{not valid json}";
  f.close();

  const auto result = LoadQuantOverrides(tmpfile);
  EXPECT_TRUE(result.empty());
  std::remove(tmpfile);
}

TEST(LoadQuantOverridesTest, V2DispatchedWhenVersionIs200) {
  char tmpfile[] = "/tmp/quant_override_test_v2_XXXXXX";
  int fd = mkstemp(tmpfile);
  ASSERT_NE(fd, -1);
  close(fd);
  std::ofstream f(tmpfile);
  f << R"({
    "version": "2.0.0",
    "encodings": [
      {"name": "t", "output_dtype": "int8", "y_scale": 0.005}
    ]
  })";
  f.close();

  const auto result = LoadQuantOverrides(tmpfile);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result.at("t").data_type, QNN_DATATYPE_SFIXED_POINT_8);
  std::remove(tmpfile);
}

TEST(LoadQuantOverridesTest, V1DispatchedWhenNoVersionField) {
  char tmpfile[] = "/tmp/quant_override_test_v1_XXXXXX";
  int fd = mkstemp(tmpfile);
  ASSERT_NE(fd, -1);
  close(fd);
  std::ofstream f(tmpfile);
  f << R"({
    "activation_encodings": {
      "act": [{"scale": 0.01, "offset": -128.0}]
    }
  })";
  f.close();

  const auto result = LoadQuantOverrides(tmpfile);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result.at("act").data_type, QNN_DATATYPE_UFIXED_POINT_8);
  std::remove(tmpfile);
}

}  // namespace
}  // namespace litert::qnn
