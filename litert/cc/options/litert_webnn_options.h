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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_WEBNN_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_WEBNN_OPTIONS_H_

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_webnn_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert {

// Builds a WebNN option object that can be passed to LiteRT CompiledModel
// creation.
//
class WebNnOptions : public litert::OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static Expected<WebNnOptions> Create();
  static const char* GetPayloadIdentifier();

  LiteRtStatus SetDevicePreference(LiteRtWebNnDeviceType device_type);
  LiteRtStatus SetPowerPreference(LiteRtWebNnPowerPreference power_preference);
  LiteRtStatus SetPrecision(LiteRtWebNnPrecision precision);
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_WEBNN_OPTIONS_H_
