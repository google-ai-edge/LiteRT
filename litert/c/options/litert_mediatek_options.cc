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
#include "litert/c/options/litert_mediatek_options.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/cache/hash_util.h"

struct LiteRtMediatekOptionsT {
  LiteRtMediatekOptionsNeronSDKVersionType neron_sdk_version =
      kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8;
  bool gemma_compiler_optimizations = false;
  LiteRtMediatekNeuronAdapterPerformanceMode performance_mode =
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed;
  bool l1_cache_optimizations = false;
  LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint =
      kLiteRtMediatekNeuronAdapterOptimizationHintNormal;
  bool disable_dla_dir_removal = false;
  std::string mediatek_dla_dir;
};

LiteRtStatus LiteRtMediatekOptionsCreate(LiteRtOpaqueOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto options_data = std::make_unique<LiteRtMediatekOptionsT>();

  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtMediatekOptionsGetIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtMediatekOptions>(payload);
      },
      options));

  auto mtk_hash = [](const void* payload) -> uint64_t {
    const LiteRtMediatekOptionsT* options =
        reinterpret_cast<const LiteRtMediatekOptionsT*>(payload);
    uint64_t ans = 0;
    litert::HashCombine(
        ans, options->neron_sdk_version, options->gemma_compiler_optimizations,
        options->performance_mode, options->l1_cache_optimizations,
        options->optimization_hint);
    return ans;
  };
  LITERT_RETURN_IF_ERROR(LiteRtSetOpaqueOptionsHash(*options, mtk_hash));

  options_data.release();
  return kLiteRtStatusOk;
}
const char* LiteRtMediatekOptionsGetIdentifier() { return "mediatek"; }

LiteRtStatus LiteRtMediatekOptionsGet(LiteRtOpaqueOptions options,
                                      LiteRtMediatekOptions* options_data) {
  if (options_data == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const char* identifier;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  if (absl::NullSafeStringView(identifier) !=
      LiteRtMediatekOptionsGetIdentifier()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  void* payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(options, &payload));
  *options_data = reinterpret_cast<LiteRtMediatekOptionsT*>(payload);
  return kLiteRtStatusOk;
}
// COMPILATION OPTIONS /////////////////////////////////////////////////////////
// sdk_version_type ----------------------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetNeronSDKVersionType(
    LiteRtMediatekOptions options,
    LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->neron_sdk_version = sdk_version_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtMediatekOptionsGetNeronSDKVersionType(
    LiteRtMediatekOptions options,
    LiteRtMediatekOptionsNeronSDKVersionType* sdk_version_type) {
  if (options == nullptr || sdk_version_type == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sdk_version_type = options->neron_sdk_version;
  return kLiteRtStatusOk;
}

// gemma_compiler_optimizations ---------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetGemmaCompilerOptimizations(
    LiteRtMediatekOptions options, bool gemma_compiler_optimizations) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->gemma_compiler_optimizations = gemma_compiler_optimizations;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtMediatekOptionsGetGemmaCompilerOptimizations(
    LiteRtMediatekOptions options, bool* gemma_compiler_optimizations) {
  if (gemma_compiler_optimizations == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *gemma_compiler_optimizations = options->gemma_compiler_optimizations;

  return kLiteRtStatusOk;
}

// neuron_adapter_peformance_mode --------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetPerformanceMode(
    LiteRtMediatekOptions options,
    LiteRtMediatekNeuronAdapterPerformanceMode performance_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->performance_mode = performance_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtMediatekOptionsGetPerformanceMode(
    LiteRtMediatekOptions options,
    LiteRtMediatekNeuronAdapterPerformanceMode* performance_mode) {
  if (options == nullptr || performance_mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *performance_mode = options->performance_mode;

  return kLiteRtStatusOk;
}

// l1_cache_optimizations ----------------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetL1CacheOptimizations(
    LiteRtMediatekOptions options, bool l1_cache_optimizations) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->l1_cache_optimizations = l1_cache_optimizations;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtMediatekOptionsGetL1CacheOptimizations(
    LiteRtMediatekOptions options, bool* l1_cache_optimizations) {
  if (l1_cache_optimizations == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *l1_cache_optimizations = options->l1_cache_optimizations;

  return kLiteRtStatusOk;
}

// neuron_optimization_hints -------------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetOptimizationHint(
    LiteRtMediatekOptions options,
    LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->optimization_hint = optimization_hint;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtMediatekOptionsGetOptimizationHint(
    LiteRtMediatekOptions options,
    LiteRtMediatekNeuronAdapterOptimizationHint* optimization_hint) {
  if (options == nullptr || optimization_hint == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *optimization_hint = options->optimization_hint;

  return kLiteRtStatusOk;
}


// disable_dla_dir_removal ---------------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetDisableDlaDirRemoval(
    LiteRtMediatekOptions options, bool disable_dla_dir_removal) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->disable_dla_dir_removal = disable_dla_dir_removal;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtMediatekOptionsGetDisableDlaDirRemoval(
    LiteRtMediatekOptions options, bool* disable_dla_dir_removal) {
  if (disable_dla_dir_removal == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *disable_dla_dir_removal = options->disable_dla_dir_removal;

  return kLiteRtStatusOk;
}

// mediatek_dla_dir -------------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetMediatekDlaDir(
    LiteRtMediatekOptions options, const char* mediatek_dla_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->mediatek_dla_dir = mediatek_dla_dir;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtMediatekOptionsGetMediatekDlaDir(
    LiteRtMediatekOptions options, const char** mediatek_dla_dir) {
  if (options == nullptr || mediatek_dla_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *mediatek_dla_dir = options->mediatek_dla_dir.c_str();

  return kLiteRtStatusOk;
}
