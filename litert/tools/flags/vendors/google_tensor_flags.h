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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_GOOGLE_TENSOR_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_GOOGLE_TENSOR_FLAGS_H_

#include <string>

#include "absl/flags/declare.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_google_tensor_options.h"

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

#if defined(INCLUDE_GOOGLE_TENSOR_COMPILE_FLAGS)

ABSL_DECLARE_FLAG(LiteRtGoogleTensorOptionsTruncationType,
                  google_tensor_truncation_type);

bool AbslParseFlag(absl::string_view text,
                   LiteRtGoogleTensorOptionsTruncationType* options,
                   std::string* error);

std::string AbslUnparseFlag(LiteRtGoogleTensorOptionsTruncationType options);

ABSL_DECLARE_FLAG(bool, google_tensor_int64_to_int32);

ABSL_DECLARE_FLAG(bool, google_tensor_dump_op_timings);

ABSL_DECLARE_FLAG(bool, google_tensor_enable_large_model_support);

ABSL_DECLARE_FLAG(bool, google_tensor_enable_4bit_compilation);

ABSL_DECLARE_FLAG(LiteRtGoogleTensorOptionsShardingIntensity,
                  google_tensor_sharding_intensity);

bool AbslParseFlag(absl::string_view text,
                   LiteRtGoogleTensorOptionsShardingIntensity* options,
                   std::string* error);

std::string AbslUnparseFlag(LiteRtGoogleTensorOptionsShardingIntensity options);

ABSL_DECLARE_FLAG(std::string, google_tensor_testing_flags);

#endif

// PARSERS (internal) //////////////////////////////////////////////////////////

#if defined(INCLUDE_GOOGLE_TENSOR_COMPILE_FLAGS) || \
    defined(INCLUDE_GOOGLE_TENSOR_RUNTIME_FLAGS)

namespace litert::google_tensor {

Expected<GoogleTensorOptions> GoogleTensorOptionsFromFlags();

}  // namespace litert::google_tensor

#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_GOOGLE_TENSOR_FLAGS_H_
