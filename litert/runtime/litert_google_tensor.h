// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_GOOGLE_TENSOR_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_GOOGLE_TENSOR_H_

#include <string>
#include <vector>

#include "litert/c/options/litert_google_tensor_options_type.h"

struct LiteRtGoogleTensorOptionsT {
  LiteRtGoogleTensorOptionsTruncationType float_truncation_type =
      kLiteRtGoogleTensorFloatTruncationTypeAuto;
  bool int64_to_int32_truncation = false;
  std::string output_dir = "";
  bool dump_op_timings = false;
  bool enable_large_model_support = false;
  bool enable_4bit_compilation = false;
  LiteRtGoogleTensorOptionsShardingIntensity sharding_intensity =
      kLiteRtGoogleTensorShardingIntensityMinimal;
  std::vector<std::vector<std::string>> testing_flags = {};
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_GOOGLE_TENSOR_H_
