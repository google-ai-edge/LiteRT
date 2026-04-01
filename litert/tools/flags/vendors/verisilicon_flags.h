// Copyright 2025 Vivante Corporation.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_VERISILICON_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_VERISILICON_FLAGS_H_

#include <string>

#include "absl/flags/declare.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_verisilicon_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_verisilicon_options.h"

// GENERAL SETTINGS ////////////////////////////////////////////////////////

#if defined(INCLUDE_VERISILICON_COMPILE_FLAGS) || \
    defined(INCLUDE_VERISILICON_RUNTIME_FLAGS)

ABSL_DECLARE_FLAG(unsigned int, verisilicon_device_index);

ABSL_DECLARE_FLAG(unsigned int, verisilicon_core_index);
ABSL_DECLARE_FLAG(unsigned int, verisilicon_time_out);
ABSL_DECLARE_FLAG(unsigned int, verisilicon_profile_level);
ABSL_DECLARE_FLAG(bool, verisilicon_dump_nbg);

#endif

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

#if defined(INCLUDE_VERISILICON_COMPILE_FLAGS)

#endif

// PARSERS (internal) //////////////////////////////////////////////////////////

#if defined(INCLUDE_VERISILICON_COMPILE_FLAGS) || \
    defined(INCLUDE_VERISILICON_RUNTIME_FLAGS)

namespace litert::verisilicon {

// Updates the provided VerisiliconOptions based on the values of the
// Verisilicon-specific command-line flags defined in this file.
Expected<void> UpdateVerisiliconOptionsFromFlags(VerisiliconOptions& options);

}  // namespace litert::verisilicon

#endif  // INCLUDE_VERISILICON_COMPILE_FLAGS || INCLUDE_VERISILICON_RUNTIME_FLAGS
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_VERISILICON_FLAGS_H_
