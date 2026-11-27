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

#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>

#include "QnnTypes.h"  // from @qairt
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json

namespace litert::qnn {

void LoadQuantOverridesV1(
    const nlohmann::json& enc_data,
    absl::flat_hash_map<std::string, QuantEncoding>& result) {
  for (const char* section : {"activation_encodings", "param_encodings"}) {
    if (!enc_data.contains(section)) continue;
    for (const auto& [name, enc_list] : enc_data[section].items()) {
      if (!enc_list.is_array() || enc_list.empty()) continue;
      const auto& enc = enc_list[0];

      float scale;
      std::int32_t zero_point;
      bool is_symmetric = false;
      int bitwidth = 8;

      if (enc.contains("scale") && enc.contains("offset")) {
        scale = enc["scale"].get<float>();
        // AIMET convention: offset = -zero_point
        zero_point = static_cast<std::int32_t>(-enc["offset"].get<float>());
        is_symmetric =
            enc.contains("is_symmetric") &&
            (enc["is_symmetric"].get<std::string>() == "True" ||
             enc["is_symmetric"].get<std::string>() == "true");
        if (enc.contains("bitwidth")) bitwidth = enc["bitwidth"].get<int>();
      } else if (enc.contains("min") && enc.contains("max")) {
        const float min_val = enc["min"].get<float>();
        const float max_val = enc["max"].get<float>();
        scale = (max_val - min_val) / 255.0f;
        zero_point =
            static_cast<std::int32_t>(std::round(-min_val / scale));
        is_symmetric =
            (std::abs(min_val + max_val) < 1e-6f * std::abs(max_val));
        if (enc.contains("bitwidth")) bitwidth = enc["bitwidth"].get<int>();
        if (enc.contains("is_symmetric"))
          is_symmetric = (enc["is_symmetric"].get<std::string>() == "True" ||
                          enc["is_symmetric"].get<std::string>() == "true");
      } else {
        continue;
      }

      Qnn_DataType_t quant_dtype;
      if (bitwidth == 16) {
        quant_dtype = is_symmetric ? QNN_DATATYPE_SFIXED_POINT_16
                                   : QNN_DATATYPE_UFIXED_POINT_16;
      } else {
        quant_dtype = is_symmetric ? QNN_DATATYPE_SFIXED_POINT_8
                                   : QNN_DATATYPE_UFIXED_POINT_8;
      }
      result.emplace(name, QuantEncoding{scale, zero_point, quant_dtype});
    }
  }
}

void LoadQuantOverridesV2(
    const nlohmann::json& enc_data,
    absl::flat_hash_map<std::string, QuantEncoding>& result) {
  static const absl::flat_hash_map<std::string, Qnn_DataType_t> kDtypeMap = {
      {"uint8", QNN_DATATYPE_UFIXED_POINT_8},
      {"int8", QNN_DATATYPE_SFIXED_POINT_8},
      {"uint16", QNN_DATATYPE_UFIXED_POINT_16},
      {"int16", QNN_DATATYPE_SFIXED_POINT_16},
      {"int32", QNN_DATATYPE_SFIXED_POINT_32},
  };
  for (const auto& entry : enc_data["encodings"]) {
    if (!entry.contains("name") || !entry.contains("output_dtype") ||
        !entry.contains("y_scale"))
      continue;
    const std::string name = entry["name"].get<std::string>();
    const std::string dtype_str = entry["output_dtype"].get<std::string>();
    auto it = kDtypeMap.find(dtype_str);
    if (it == kDtypeMap.end()) continue;
    const float scale = entry["y_scale"].get<float>();
    const std::int32_t zero_point =
        entry.contains("y_zero_point")
            ? static_cast<std::int32_t>(entry["y_zero_point"].get<int>())
            : 0;
    result.emplace(name, QuantEncoding{scale, zero_point, it->second});
  }
}

absl::flat_hash_map<std::string, QuantEncoding> LoadQuantOverrides(
    absl::string_view json_path) {
  absl::flat_hash_map<std::string, QuantEncoding> result;
  if (json_path.empty()) return result;

  std::string path(json_path);
  std::ifstream f(path);
  if (!f.is_open()) {
    LITERT_LOG(LITERT_ERROR, "quantization_overrides: cannot open file '%s'",
               path.c_str());
    return result;
  }

  nlohmann::json enc_data;
  try {
    f >> enc_data;
  } catch (const nlohmann::json::exception& e) {
    LITERT_LOG(LITERT_ERROR,
               "quantization_overrides: JSON parse error in '%s': %s",
               path.c_str(), e.what());
    return result;
  }

  if (enc_data.contains("version") &&
      enc_data["version"].get<std::string>() == "2.0.0") {
    LoadQuantOverridesV2(enc_data, result);
  } else {
    LoadQuantOverridesV1(enc_data, result);
  }

  LITERT_LOG(LITERT_INFO,
             "quantization_overrides: loaded %zu encodings from '%s'",
             result.size(), path.c_str());
  return result;
}

}  // namespace litert::qnn
