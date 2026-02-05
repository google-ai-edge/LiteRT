// Copyright 2026 Google LLC.
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

#include "litert/vendors/google_tensor/dispatch/dispatch_api_config.h"

#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/options/litert_darwinn_options.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/cc/options_helper.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"
#include "litert/vendors/google_tensor/dispatch/sb_api_features.h"

namespace litert::google_tensor {

namespace {

// Indicates whether `InitializeDispatchApiConfig` has been called.
//
// NOTE: all constants are expected to be initialized in
// `InitializeDispatchApiConfig`.
bool TheDispatchApiConfigIsInitialized = false;

// Aborts the calling process if the Dispatch API config is not initialized,
// else a nop.
#define CHECK_DISPATCH_API_CONFIG_INIT()                      \
  do {                                                        \
    if (!TheDispatchApiConfigIsInitialized) {                 \
      LITERT_FATAL("Dispatch API config is not initialized"); \
    }                                                         \
  } while (0)

// Optional DarwiNN-specific options provided by the application.
absl_nullable std::unique_ptr<litert::DarwinnRuntimeOptions> TheDarwinnOptions;

// Google Tensor Dispatch API build ID.
char TheBuildId[256];

// Capabilities of the available SouthBound implementation.
int TheCapabilities = 0;

// Tensor buffer types that are supported by the available SouthBound
// implementation.
std::vector<LiteRtTensorBufferType> TheSupportedTensorBufferTypes;

}  // namespace

LiteRtStatus InitializeDispatchApiConfig(
    LiteRtEnvironmentOptions environment_options, LiteRtOptions options) {
  auto [opts, opq_opts, darwinn_opts] =
      litert::ParseOptions<litert::DarwinnRuntimeOptions>(options);

  if (darwinn_opts.HasValue()) {
    TheDarwinnOptions = std::make_unique<litert::DarwinnRuntimeOptions>(
        std::move(*darwinn_opts));
    LITERT_LOG(LITERT_INFO, "Found Darwinn runtime options");
  } else {
    LITERT_LOG(LITERT_INFO, "No Darwinn runtime options found, using defaults");
  }

  const char* sb_api_version = thrGetVendorApiVersion();
  const char* sb_vendor_id = thrGetVendorId();
  snprintf(TheBuildId, sizeof(TheBuildId),
           "GoogleTensor Dispatch API version %d.%d.%d, SB API version %s, "
           "vendor id: %s",
           LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
           LITERT_API_VERSION_PATCH, sb_api_version, sb_vendor_id);

  TheCapabilities = kLiteRtDispatchCapabilitiesBasic;
  if (GoogleTensorSouthBoundFeatureSupported(
          GoogleTensorSouthBoundFeature::kRobustFences)) {
    TheCapabilities |= kLiteRtDispatchCapabilitiesAsync;
  }
  if (GoogleTensorSouthBoundFeatureSupported(
          GoogleTensorSouthBoundFeature::kIndexedNodeBinding)) {
    TheCapabilities |= kLiteRtDispatchCapabilitiesGraph;
  }

  TheSupportedTensorBufferTypes = {
#if LITERT_HAS_AHWB_SUPPORT
      kLiteRtTensorBufferTypeAhwb,
#endif
  };
#if LITERT_HAS_DMABUF_SUPPORT
  if (GoogleTensorSouthBoundFeatureSupported(
          GoogleTensorSouthBoundFeature::kDmaBufRegistration)) {
    TheSupportedTensorBufferTypes.push_back(kLiteRtTensorBufferTypeDmaBuf);
  }
#endif
  if (TheSupportedTensorBufferTypes.empty()) {
    TheSupportedTensorBufferTypes.push_back(kLiteRtTensorBufferTypeHostMemory);
  }

  TheDispatchApiConfigIsInitialized = true;
  return kLiteRtStatusOk;
}

DarwinnRuntimeOptions* absl_nullable GetTheDarwinnOptions() {
  CHECK_DISPATCH_API_CONFIG_INIT();
  return TheDarwinnOptions.get();
}

const char* absl_nonnull GetTheBuildId() {
  CHECK_DISPATCH_API_CONFIG_INIT();
  return TheBuildId;
}

int GetTheCapabilities() {
  CHECK_DISPATCH_API_CONFIG_INIT();
  return TheCapabilities;
}

absl::Span<const LiteRtTensorBufferType> GetTheSupportedTensorBufferTypes() {
  CHECK_DISPATCH_API_CONFIG_INIT();
  return TheSupportedTensorBufferTypes;
}

}  // namespace litert::google_tensor
