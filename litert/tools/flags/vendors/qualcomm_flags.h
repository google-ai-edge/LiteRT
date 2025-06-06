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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_QUALCOMM_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_QUALCOMM_FLAGS_H_

#include <string>
#include <vector>

#include "absl/flags/declare.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_qualcomm_options.h"

// GENERAL SDK SETTINGS ////////////////////////////////////////////////////////

#if defined(INCLUDE_QUALCOMM_COMPILE_FLAGS) || \
    defined(INCLUDE_QUALCOMM_RUNTIME_FLAGS)

ABSL_DECLARE_FLAG(LiteRtQualcommOptionsLogLevel, qualcomm_log_level);
std::string AbslUnparseFlag(LiteRtQualcommOptionsLogLevel options);
bool AbslParseFlag(absl::string_view text,
                   LiteRtQualcommOptionsLogLevel* options, std::string* error);

#endif

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

#if defined(INCLUDE_QUALCOMM_COMPILE_FLAGS)

ABSL_DECLARE_FLAG(bool, qualcomm_enable_weight_sharing);

ABSL_DECLARE_FLAG(bool, qualcomm_use_htp_preference);

ABSL_DECLARE_FLAG(bool, qualcomm_use_qint16_as_quint16);

ABSL_DECLARE_FLAG(std::vector<std::string>, qualcomm_dump_tensor_ids);

ABSL_DECLARE_FLAG(std::string, qualcomm_qnn_json_path);

#endif

// DISPATCH OPTIONS ////////////////////////////////////////////////////////////

#if defined(INCLUDE_QUALCOMM_COMPILE_FLAGS)

ABSL_DECLARE_FLAG(LiteRtQualcommOptionsHtpPerformanceMode,
                  qualcomm_htp_performance_mode);
bool AbslParseFlag(absl::string_view text,
                   LiteRtQualcommOptionsHtpPerformanceMode* options,
                   std::string* error);
std::string AbslUnparseFlag(LiteRtQualcommOptionsHtpPerformanceMode options);

ABSL_DECLARE_FLAG(LiteRtQualcommOptionsProfiling, qualcomm_profiling);
bool AbslParseFlag(absl::string_view text,
                   LiteRtQualcommOptionsProfiling* options, std::string* error);
std::string AbslUnparseFlag(LiteRtQualcommOptionsProfiling options);

#endif

// TO OBJECT (internal) ////////////////////////////////////////////////////////

#if defined(INCLUDE_QUALCOMM_COMPILE_FLAGS) || \
    defined(INCLUDE_QUALCOMM_RUNTIME_FLAGS)

namespace litert::qualcomm {

Expected<QualcommOptions> QualcommOptionsFromFlags();

}  // namespace litert::qualcomm

#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_QUALCOMM_FLAGS_H_
