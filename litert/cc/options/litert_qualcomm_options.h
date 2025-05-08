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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::qualcomm {

// Wraps a LiteRtQualcommOptions object for convenience.
class QualcommOptions : public OpaqueOptions {
 public:
  using LogLevel = LiteRtQualcommOptionsLogLevel;
  using PowerMode = LiteRtQualcommOptionsPowerMode;

  using OpaqueOptions::OpaqueOptions;

  static const char* Discriminator() {
    return LiteRtQualcommOptionsGetIdentifier();
  }

  static Expected<QualcommOptions> Create(OpaqueOptions& options);

  static Expected<QualcommOptions> Create();

  void SetLogLevel(LogLevel log_level);
  LogLevel GetLogLevel();

  void SetPowerMode(PowerMode power_mode);
  PowerMode GetPowerMode();

  void SetEnableWeightSharing(bool weight_sharing_enabled);
  bool GetEnableWeightSharing();

 private:
  LiteRtQualcommOptions Data() const;
};

}  // namespace litert::qualcomm

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
