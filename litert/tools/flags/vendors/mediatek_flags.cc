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

#include "litert/tools/flags/vendors/mediatek_flags.h"

#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/tools/flags/options_parser_registry.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

ABSL_FLAG(litert::mediatek::MediatekOptions::NeronSDKVersion,
          mediatek_sdk_version_type,
          litert::mediatek::MediatekOptions::NeronSDKVersion::kVersion8,
          "Version for neuron sdk for Mediatek.");

ABSL_FLAG(bool, mediatek_enable_gemma_compiler_optimizations, false,
          "Whether to enable Gemma Mediatek compiler optimizations.");

ABSL_FLAG(bool, mediatek_enable_l1_cache_optimizations, false,
          "Whether to enable L1 cache optimizations.");

ABSL_FLAG(litert::mediatek::MediatekOptions::PerformanceMode,
          mediatek_performance_mode_type,
          litert::mediatek::MediatekOptions::PerformanceMode::kSustainedSpeed,
          "Performance mode for Mediatek Inference.");

ABSL_FLAG(litert::mediatek::MediatekOptions::OptimizationHint,
          mediatek_optimization_hint,
          litert::mediatek::MediatekOptions::OptimizationHint::kNormal,
          "Optimization hint for Mediatek Inference.");

ABSL_FLAG(bool, mediatek_disable_dla_dir_removal, false,
          "Disable DLA directory removal for Mediatek Compilation.");

ABSL_FLAG(
    std::string, mediatek_dla_dir, "",
    "Meidatek DLA provided directory. If provided, all compiled DLA's will be "
    "stored in the provided directory path. Meant to be used in conjunction "
    "with `--mediatek_disable_dla_dir_removal` so that DLA's aren't cleaned "
    "up post compilation.");

ABSL_FLAG(std::string, mediatek_aot_compilation_options, "",
          "Aot compilation options for Mediatek Inference.");

namespace litert::mediatek {

bool AbslParseFlag(absl::string_view text,
                   MediatekOptions::NeronSDKVersion* options,
                   std::string* error) {
  if (text == "version7") {
    *options = MediatekOptions::NeronSDKVersion::kVersion7;
    return true;
  }

  if (text == "version8") {
    *options = MediatekOptions::NeronSDKVersion::kVersion8;
    return true;
  }

  if (text == "version9") {
    *options = MediatekOptions::NeronSDKVersion::kVersion9;
    return true;
  }

  *error = "Unknown sdk version type";
  return false;
}

std::string AbslUnparseFlag(MediatekOptions::NeronSDKVersion options) {
  switch (options) {
    case MediatekOptions::NeronSDKVersion::kVersion7:
      return "version7";
    case MediatekOptions::NeronSDKVersion::kVersion8:
      return "version8";
    case MediatekOptions::NeronSDKVersion::kVersion9:
      return "version9";
  }
}

bool AbslParseFlag(absl::string_view text,
                   MediatekOptions::PerformanceMode* options,
                   std::string* error) {
  if (text == "low_power") {
    *options = MediatekOptions::PerformanceMode::kLowPower;
    return true;
  }
  if (text == "fast_single_answer") {
    *options = MediatekOptions::PerformanceMode::kFastSingleAnswer;
    return true;
  }
  if (text == "sustained_speed") {
    *options = MediatekOptions::PerformanceMode::kSustainedSpeed;
    return true;
  }
  if (text == "turbo_boost") {
    *options = MediatekOptions::PerformanceMode::kTurboBoost;
    return true;
  }

  *error = "Unknown mediatek performance mode type";
  return false;
}

std::string AbslUnparseFlag(MediatekOptions::PerformanceMode options) {
  switch (options) {
    case MediatekOptions::PerformanceMode::kLowPower:
      return "low_power";
    case MediatekOptions::PerformanceMode::kSustainedSpeed:
      return "sustained_speed";
    case MediatekOptions::PerformanceMode::kFastSingleAnswer:
      return "fast_single_answer";
    case MediatekOptions::PerformanceMode::kTurboBoost:
      return "turbo_boost";
  }
}

bool AbslParseFlag(absl::string_view text,
                   MediatekOptions::OptimizationHint* options,
                   std::string* error) {
  if (text == "normal") {
    *options = MediatekOptions::OptimizationHint::kNormal;
    return true;
  }
  if (text == "low_latency") {
    *options = MediatekOptions::OptimizationHint::kLowLatency;
    return true;
  }
  if (text == "deep_fusion") {
    *options = MediatekOptions::OptimizationHint::kDeepFusion;
    return true;
  }
  if (text == "batch_processing") {
    *options = MediatekOptions::OptimizationHint::kBatchProcessing;
    return true;
  }

  *error = "Unknown mediatek optimization hint type";
  return false;
}

std::string AbslUnparseFlag(MediatekOptions::OptimizationHint options) {
  switch (options) {
    case MediatekOptions::OptimizationHint::kNormal:
      return "normal";
    case MediatekOptions::OptimizationHint::kLowLatency:
      return "low_latency";
    case MediatekOptions::OptimizationHint::kDeepFusion:
      return "deep_fusion";
    case MediatekOptions::OptimizationHint::kBatchProcessing:
      return "batch_processing";
  }
}

}  // namespace litert::mediatek

// NOLINTEND(*alien-types*)

namespace litert::mediatek {

Expected<void> UpdateMediatekOptionsFromFlags(MediatekOptions& options) {
  options.SetNeronSDKVersionType(
      absl::GetFlag(FLAGS_mediatek_sdk_version_type));
  options.SetEnableGemmaCompilerOptimizations(
      absl::GetFlag(FLAGS_mediatek_enable_gemma_compiler_optimizations));
  options.SetPerformanceMode(
      absl::GetFlag(FLAGS_mediatek_performance_mode_type));
  options.SetEnableL1CacheOptimizations(
      absl::GetFlag(FLAGS_mediatek_enable_l1_cache_optimizations));
  options.SetOptimizationHint(absl::GetFlag(FLAGS_mediatek_optimization_hint));
  options.SetDisableDlaDirRemoval(
      absl::GetFlag(FLAGS_mediatek_disable_dla_dir_removal));
  options.SetMediatekDlaDir(absl::GetFlag(FLAGS_mediatek_dla_dir));
  options.SetAotCompilationOptions(
      absl::GetFlag(FLAGS_mediatek_aot_compilation_options));
  return {};
}

}  // namespace litert::mediatek

namespace litert::mediatek {

LITERT_REGISTER_OPTIONS_PARSER([](Options& options) -> Expected<void> {
  LITERT_ASSIGN_OR_RETURN(auto& mediatek_opts, options.GetMediatekOptions());
  return UpdateMediatekOptionsFromFlags(mediatek_opts);
});

}  // namespace litert::mediatek
