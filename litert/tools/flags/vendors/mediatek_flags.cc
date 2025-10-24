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
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_mediatek_options.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

ABSL_FLAG(LiteRtMediatekOptionsNeronSDKVersionType, mediatek_sdk_version_type,
          kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8,
          "Version for neuron sdk for Mediatek.");

ABSL_FLAG(bool, mediatek_enable_gemma_compiler_optimizations, false,
          "Whether to enable Gemma Mediatek compiler optimizations.");

ABSL_FLAG(bool, mediatek_enable_l1_cache_optimizations, false,
          "Whether to enable L1 cache optimizations.");

ABSL_FLAG(LiteRtMediatekNeuronAdapterPerformanceMode,
          mediatek_performance_mode_type,
          kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed,
          "Performance mode for Mediatek Inference.");

ABSL_FLAG(LiteRtMediatekNeuronAdapterOptimizationHint,
          mediatek_optimization_hint,
          kLiteRtMediatekNeuronAdapterOptimizationHintNormal,
          "Optimization hint for Mediatek Inference.");

ABSL_FLAG(bool, mediatek_disable_dla_dir_removal, false,
          "Disable DLA directory removal for Mediatek Compilation.");

bool AbslParseFlag(absl::string_view text,
                   LiteRtMediatekOptionsNeronSDKVersionType* options,
                   std::string* error) {
  if (text == "version7") {
    *options = kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7;
    return true;
  }

  if (text == "version8") {
    *options = kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8;
    return true;
  }

  if (text == "version9") {
    *options = kLiteRtMediatekOptionsNeronSDKVersionTypeVersion9;
    return true;
  }

  *error = "Unknown sdk version type";
  return false;
}

std::string AbslUnparseFlag(LiteRtMediatekOptionsNeronSDKVersionType options) {
  switch (options) {
    case kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7:
      return "version7";
    case kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8:
      return "version8";
    case kLiteRtMediatekOptionsNeronSDKVersionTypeVersion9:
      return "version9";
  }
}

bool AbslParseFlag(absl::string_view text,
                   LiteRtMediatekNeuronAdapterPerformanceMode* options,
                   std::string* error) {
  if (text == "low_power") {
    *options = kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower;
    return true;
  }
  if (text == "fast_single_answer") {
    *options =
        kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferFastSingleAnswer;
    return true;
  }
  if (text == "sustained_speed") {
    *options =
        kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed;
    return true;
  }
  if (text == "turbo_boost") {
    *options =
        kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferTurboBoost;
    return true;
  }

  *error = "Unknown mediatek performance mode type";
  return false;
}

std::string AbslUnparseFlag(
    LiteRtMediatekNeuronAdapterPerformanceMode options) {
  switch (options) {
    case kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower:
      return "low_power";
    case kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed:
      return "sustained_speed";
    case (
        kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferFastSingleAnswer
      ):
      return "fast_single_answer";
    case kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferTurboBoost:
      return "turbo_boost";
  }
}

bool AbslParseFlag(absl::string_view text,
                   LiteRtMediatekNeuronAdapterOptimizationHint* options,
                   std::string* error) {
  if (text == "normal") {
    *options = kLiteRtMediatekNeuronAdapterOptimizationHintNormal;
    return true;
  }
  if (text == "low_latency") {
    *options = kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency;
    return true;
  }
  if (text == "deep_fusion") {
    *options = kLiteRtMediatekNeuronAdapterOptimizationHintDeepFusion;
    return true;
  }
  if (text == "batch_processing") {
    *options = kLiteRtMediatekNeuronAdapterOptimizationHintBatchProcessing;
    return true;
  }

  *error = "Unknown mediatek optimization hint type";
  return false;
}

std::string AbslUnparseFlag(
    LiteRtMediatekNeuronAdapterOptimizationHint options) {
  switch (options) {
    case kLiteRtMediatekNeuronAdapterOptimizationHintNormal:
      return "normal";
    case kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency:
      return "low_latency";
    case (kLiteRtMediatekNeuronAdapterOptimizationHintDeepFusion):
      return "deep_fusion";
    case kLiteRtMediatekNeuronAdapterOptimizationHintBatchProcessing:
      return "batch_processing";
  }
}

// NOLINTEND(*alien-types*)

namespace litert::mediatek {

Expected<MediatekOptions> MediatekOptionsFromFlags() {
  LITERT_ASSIGN_OR_RETURN(auto options, MediatekOptions::Create());
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
  return options;
}

}  // namespace litert::mediatek
