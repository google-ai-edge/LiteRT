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

#include "litert/tools/flags/vendors/verisilicon_flags.h"

#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_verisilicon_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_verisilicon_options.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

ABSL_FLAG(unsigned int, verisilicon_device_index, 0,
          "The device index of multi-device.");

ABSL_FLAG(unsigned int, verisilicon_core_index, 0,
          "The core index of multi-core.");

ABSL_FLAG(unsigned int, verisilicon_time_out, 0,
          "Milliseconds time out of network.");

ABSL_FLAG(unsigned int, verisilicon_profile_level, 0,
          "Execute operations(commands) one by one and show profile log.");

ABSL_FLAG(bool, verisilicon_dump_nbg, false,
          "Dump NBG resource(nbg, input, output).");

// NOLINTEND(*alien-types*)

namespace litert::verisilicon {

Expected<void> UpdateVerisiliconOptionsFromFlags(VerisiliconOptions& options) {
  options.SetDeviceIndex(
      absl::GetFlag(FLAGS_verisilicon_device_index));
  options.SetCoreIndex(
      absl::GetFlag(FLAGS_verisilicon_core_index));
  options.SetTimeOut(
      absl::GetFlag(FLAGS_verisilicon_time_out));
  options.SetProfileLevel(
      absl::GetFlag(FLAGS_verisilicon_profile_level));
  options.SetDumpNBG(
      absl::GetFlag(FLAGS_verisilicon_dump_nbg));
  return {};
}

}  // namespace litert::verisilicon
