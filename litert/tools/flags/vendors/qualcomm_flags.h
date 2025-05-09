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

#include "absl/flags/declare.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_qualcomm_options.h"

// GENERAL SDK SETTINGS ////////////////////////////////////////////////////////

ABSL_DECLARE_FLAG(LiteRtQualcommOptionsLogLevel, qualcomm_log_level);
std::string AbslUnparseFlag(LiteRtQualcommOptionsLogLevel options);
bool AbslParseFlag(absl::string_view text,
                   LiteRtQualcommOptionsLogLevel* options, std::string* error);

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

ABSL_DECLARE_FLAG(bool, qualcomm_enable_weight_sharing);

ABSL_DECLARE_FLAG(bool, qualcomm_use_htp_preference);

ABSL_DECLARE_FLAG(bool, qualcomm_use_qint16_as_quint16);

// DISPATCH OPTIONS ////////////////////////////////////////////////////////////

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

// TO OBJECT (internal) ////////////////////////////////////////////////////////

namespace litert::qualcomm {

Expected<QualcommOptions> QualcommOptionsFromFlags();

}  // namespace litert::qualcomm

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_QUALCOMM_FLAGS_H_
