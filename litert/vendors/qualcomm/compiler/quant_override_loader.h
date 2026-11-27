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

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_QUANT_OVERRIDE_LOADER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_QUANT_OVERRIDE_LOADER_H_

#include <cstdint>
#include <string>

#include "QnnTypes.h"  // from @qairt
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json

namespace litert::qnn {

// Per-tensor quantization encoding parsed from a JSON override file.
struct QuantEncoding {
  float scale;
  std::int32_t zero_point;
  Qnn_DataType_t data_type;
};

// Parses an AIMET v1/v0.6 encoding JSON object into result.
//
// Expected top-level keys: "activation_encodings", "param_encodings".
// Each value is a dict mapping tensor name → array of encoding objects:
//   {"scale": 0.01, "offset": -128}            (scale/offset form)
//   {"min": -1.0, "max": 1.0, "bitwidth": 8}   (min/max form)
// "offset" follows the AIMET sign convention: offset = -zero_point.
void LoadQuantOverridesV1(
    const nlohmann::json& enc_data,
    absl::flat_hash_map<std::string, QuantEncoding>& result);

// Parses a v2.0.0 encoding JSON object (e.g. "dlc_extracted") into result.
//
// Expected top-level key: "encodings" (array).
// Each element: {"name": "t", "output_dtype": "uint8",
//                "y_scale": 0.01, "y_zero_point": 128}
// y_zero_point is the QNN zero_point directly (not negated like AIMET offset).
// y_zero_point is optional and defaults to 0 (symmetric).
void LoadQuantOverridesV2(
    const nlohmann::json& enc_data,
    absl::flat_hash_map<std::string, QuantEncoding>& result);

// Loads a quantization encoding file into a per-tensor map.
// Dispatches to LoadQuantOverridesV1 (AIMET format) or LoadQuantOverridesV2
// (v2.0.0 format) based on the "version" field in the JSON.
// Returns an empty map on error (errors are logged).
absl::flat_hash_map<std::string, QuantEncoding> LoadQuantOverrides(
    absl::string_view json_path);

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_QUANT_OVERRIDE_LOADER_H_
