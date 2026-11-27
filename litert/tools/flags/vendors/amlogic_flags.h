// Copyright 2026 Amlogic, Inc.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_AML_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_AML_FLAGS_H_

#include <string>

#include "absl/flags/declare.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_amlogic_options.h"

// Compilation options consumed by the AML compiler plugin.

ABSL_DECLARE_FLAG(std::string, amlogic_model_path);
ABSL_DECLARE_FLAG(std::string, amlogic_model_name);
ABSL_DECLARE_FLAG(std::string, amlogic_log_level);

namespace litert::amlogic {

// Applies AML command-line flags onto @p options. Empty flags are left unset.
Expected<void> UpdateAmlogicOptionsFromFlags(AmlogicOptions& options);

}  // namespace litert::amlogic

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_AML_FLAGS_H_
