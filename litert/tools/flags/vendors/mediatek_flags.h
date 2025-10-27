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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_MEDIATEK_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_MEDIATEK_FLAGS_H_

#include <string>

#include "absl/flags/declare.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_mediatek_options.h"

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

#if defined(INCLUDE_MEDIATEK_COMPILE_FLAGS)

ABSL_DECLARE_FLAG(LiteRtMediatekOptionsNeronSDKVersionType,
                  mediatek_sdk_version_type);

ABSL_DECLARE_FLAG(bool, mediatek_enable_gemma_compiler_optimizations);

ABSL_DECLARE_FLAG(LiteRtMediatekNeuronAdapterPerformanceMode,
                  mediatek_performance_mode_type);

ABSL_DECLARE_FLAG(bool, mediatek_enable_l1_cache_optimizations);

ABSL_DECLARE_FLAG(LiteRtMediatekNeuronAdapterOptimizationHint,
                  mediatek_optimization_hint);

ABSL_DECLARE_FLAG(bool, mediatek_disable_dla_dir_removal);

ABSL_DECLARE_FLAG(std::string, mediatek_dla_dir);

bool AbslParseFlag(absl::string_view text,
                   LiteRtMediatekOptionsNeronSDKVersionType* options,
                   std::string* error);

std::string AbslUnparseFlag(LiteRtMediatekOptionsNeronSDKVersionType options);

bool AbslParseFlag(absl::string_view text,
                   LiteRtMediatekNeuronAdapterPerformanceMode* options,
                   std::string* error);

std::string AbslUnparseFlag(LiteRtMediatekNeuronAdapterPerformanceMode options);

bool AbslParseFlag(absl::string_view text,
                   LiteRtMediatekNeuronAdapterPerformanceMode* options,
                   std::string* error);

std::string AbslUnparseFlag(LiteRtMediatekNeuronAdapterPerformanceMode options);

bool AbslParseFlag(absl::string_view text,
                   LiteRtMediatekNeuronAdapterOptimizationHint* options,
                   std::string* error);

std::string AbslUnparseFlag(
    LiteRtMediatekNeuronAdapterOptimizationHint options);

#endif

// PARSERS (internal) //////////////////////////////////////////////////////////

#if defined(INCLUDE_MEDIATEK_COMPILE_FLAGS) || \
    defined(INCLUDE_MEDIATEK_RUNTIME_FLAGS)

namespace litert::mediatek {

Expected<MediatekOptions> MediatekOptionsFromFlags();

}  // namespace litert::mediatek

#endif  // INCLUDE_MEDIATEK_COMPILE_FLAGS || INCLUDE_MEDIATEK_RUNTIME_FLAGS
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_MEDIATEK_FLAGS_H_
