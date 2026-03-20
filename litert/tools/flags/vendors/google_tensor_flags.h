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

ABSL_DECLARE_FLAG(LrtGoogleTensorOptionsTruncationType,
                  google_tensor_truncation_type);

bool AbslParseFlag(absl::string_view text,
                   LrtGoogleTensorOptionsTruncationType* options,
                   std::string* error);

std::string AbslUnparseFlag(LrtGoogleTensorOptionsTruncationType options);

ABSL_DECLARE_FLAG(bool, google_tensor_int64_to_int32);

ABSL_DECLARE_FLAG(bool, google_tensor_dump_op_timings);

ABSL_DECLARE_FLAG(bool, google_tensor_enable_large_model_support);

ABSL_DECLARE_FLAG(bool, google_tensor_enable_4bit_compilation);

ABSL_DECLARE_FLAG(LrtGoogleTensorOptionsShardingIntensity,
                  google_tensor_sharding_intensity);

ABSL_DECLARE_FLAG(bool, google_tensor_enable_dynamic_range_quantization);

bool AbslParseFlag(absl::string_view text,
                   LrtGoogleTensorOptionsShardingIntensity* options,
                   std::string* error);

std::string AbslUnparseFlag(LrtGoogleTensorOptionsShardingIntensity options);

ABSL_DECLARE_FLAG(std::string, google_tensor_testing_flags);

ABSL_DECLARE_FLAG(std::string, google_tensor_op_filters_proto);


namespace litert::google_tensor {
bool AbslParseFlag(::absl::string_view text,
                   GoogleTensorOptions::PerformanceMode* options,
                   ::std::string* error);

::std::string AbslUnparseFlag(GoogleTensorOptions::PerformanceMode options);
}  // namespace litert::google_tensor

ABSL_DECLARE_FLAG(litert::google_tensor::GoogleTensorOptions::PerformanceMode,
                  google_tensor_performance_mode);

// PARSERS (internal) //////////////////////////////////////////////////////////

namespace litert::google_tensor {

// Updates the provided GoogleTensorOptions based on the values of the
// GoogleTensor-specific command-line flags defined in this file.
Expected<void> UpdateGoogleTensorOptionsFromFlags(GoogleTensorOptions& options);

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_GOOGLE_TENSOR_FLAGS_H_
