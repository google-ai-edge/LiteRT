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
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/tools/flags/flag_types.h"

// GENERAL SDK SETTINGS ////////////////////////////////////////////////////////

#if defined(INCLUDE_QUALCOMM_COMPILE_FLAGS) || \
    defined(INCLUDE_QUALCOMM_RUNTIME_FLAGS)

ABSL_DECLARE_FLAG(litert::qualcomm::QualcommOptions::LogLevel,
                  qualcomm_log_level);

namespace litert::qualcomm {

std::string AbslUnparseFlag(QualcommOptions::LogLevel opts);

bool AbslParseFlag(absl::string_view text, QualcommOptions::LogLevel* options,
                   std::string* error);

}  // namespace litert::qualcomm

ABSL_DECLARE_FLAG(litert::qualcomm::QualcommOptions::Backend, qualcomm_backend);

namespace litert::qualcomm {

bool AbslParseFlag(absl::string_view text, QualcommOptions::Backend* options,
                   std::string* error);

std::string AbslUnparseFlag(QualcommOptions::Backend options);

}  // namespace litert::qualcomm

#endif

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

#if defined(INCLUDE_QUALCOMM_COMPILE_FLAGS)

ABSL_DECLARE_FLAG(bool, qualcomm_enable_weight_sharing);

ABSL_DECLARE_FLAG(bool, qualcomm_use_htp_preference);

ABSL_DECLARE_FLAG(bool, qualcomm_use_qint16_as_quint16);

ABSL_DECLARE_FLAG(::litert::tools::IntList, qualcomm_dump_tensor_ids);

ABSL_DECLARE_FLAG(std::string, qualcomm_ir_json_dir);

ABSL_DECLARE_FLAG(std::string, qualcomm_dlc_dir);

ABSL_DECLARE_FLAG(std::string, qualcomm_saver_output_dir);

ABSL_DECLARE_FLAG(uint32_t, qualcomm_vtcm_size);

ABSL_DECLARE_FLAG(uint32_t, qualcomm_num_hvx_thread);

ABSL_DECLARE_FLAG(litert::qualcomm::QualcommOptions::OptimizationLevel,
                  qualcomm_optimization_level);
ABSL_DECLARE_FLAG(litert::qualcomm::QualcommOptions::GraphPriority,
                  qualcomm_graph_priority);

namespace litert::qualcomm {

bool AbslParseFlag(absl::string_view text,
                   QualcommOptions::OptimizationLevel* optimization_level,
                   std::string* error);

std::string AbslUnparseFlag(
    QualcommOptions::OptimizationLevel optimization_level);

bool AbslParseFlag(absl::string_view text,
                   QualcommOptions::GraphPriority* graph_priority,
                   std::string* error);

std::string AbslUnparseFlag(QualcommOptions::GraphPriority graph_priority);

}  // namespace litert::qualcomm

ABSL_DECLARE_FLAG(bool, qualcomm_use_conv_hmx);

ABSL_DECLARE_FLAG(bool, qualcomm_use_fold_relu);

#endif

// DISPATCH OPTIONS ////////////////////////////////////////////////////////////

#if defined(INCLUDE_QUALCOMM_COMPILE_FLAGS)

ABSL_DECLARE_FLAG(litert::qualcomm::QualcommOptions::HtpPerformanceMode,
                  qualcomm_htp_performance_mode);

namespace litert::qualcomm {

bool AbslParseFlag(absl::string_view text,
                   QualcommOptions::HtpPerformanceMode* options,
                   std::string* error);

std::string AbslUnparseFlag(QualcommOptions::HtpPerformanceMode options);

}  // namespace litert::qualcomm

ABSL_DECLARE_FLAG(litert::qualcomm::QualcommOptions::DspPerformanceMode,
                  qualcomm_dsp_performance_mode);

namespace litert::qualcomm {

bool AbslParseFlag(absl::string_view text,
                   QualcommOptions::DspPerformanceMode* options,
                   std::string* error);

std::string AbslUnparseFlag(QualcommOptions::DspPerformanceMode options);

}  // namespace litert::qualcomm

ABSL_DECLARE_FLAG(litert::qualcomm::QualcommOptions::Profiling,
                  qualcomm_profiling);

namespace litert::qualcomm {

bool AbslParseFlag(absl::string_view text, QualcommOptions::Profiling* options,
                   std::string* error);

std::string AbslUnparseFlag(QualcommOptions::Profiling options);

}  // namespace litert::qualcomm

#endif

// TO OBJECT (internal) ////////////////////////////////////////////////////////

#if defined(INCLUDE_QUALCOMM_COMPILE_FLAGS) || \
    defined(INCLUDE_QUALCOMM_RUNTIME_FLAGS)

namespace litert::qualcomm {

// Updates the provided QualcommOptions based on the values of the
// Qualcomm-specific command-line flags defined in this file.
Expected<void> UpdateQualcommOptionsFromFlags(QualcommOptions& opts);

}  // namespace litert::qualcomm

#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_QUALCOMM_FLAGS_H_
