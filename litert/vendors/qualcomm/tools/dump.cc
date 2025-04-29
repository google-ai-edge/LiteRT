// Copyright 2024 Google LLC.
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

#include "litert/vendors/qualcomm/tools/dump.h"  // Include the header with the new signature

#include <string>  // Include for std::string

#include "absl/strings/str_cat.h"  // from @com_google_absl  // For concatenating strings efficiently
#include "absl/strings/str_format.h"  // from @com_google_absl  // Use StrFormat to return strings
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/qnn_manager.h"  // For QnnManager definition
#include "QnnInterface.h"  // from @qairt
#include "System/QnnSystemInterface.h"  // from @qairt

namespace litert::qnn::internal {
namespace {

static constexpr absl::string_view kNullDumpTpl = "%s : nullptr\n";

// Changed signature: Returns std::string
std::string Dump(const QnnInterface_t* interface) {
  static constexpr absl::string_view kQnnInterfaceHeader = "< QnnInterface_t >";
  // NOLINTBEGIN
  static constexpr absl::string_view kQnnInterfaceDumpTpl =
      "\
  %s\n\
  name: %s\n\
  backend_id: %u\n\
  core_api_version: %u.%u.%u\n\
  backend_api_version: %u.%u.%u\n";
  // NOLINTEND

  if (interface == nullptr) {
    // Return the formatted string directly
    return absl::StrFormat(kNullDumpTpl, kQnnInterfaceHeader);
  }

  const auto core_version = interface->apiVersion.coreApiVersion;
  const auto backend_version = interface->apiVersion.backendApiVersion;

  // Return the formatted string directly
  return absl::StrFormat(kQnnInterfaceDumpTpl, kQnnInterfaceHeader,
                         interface->providerName, interface->backendId,
                         core_version.major, core_version.minor,
                         core_version.patch, backend_version.major,
                         backend_version.minor, backend_version.patch);
}

// Changed signature: Returns std::string
std::string Dump(const QnnSystemInterface_t* interface) {
  static constexpr absl::string_view kQnnSystemInterfaceHeader =
      "< QnnSystemInterface_t >";
  // NOLINTBEGIN
  static constexpr absl::string_view kQnnSystemInterfaceDumpTpl =
      "\
  %s\n\
  name: %s\n\
  backend_id: %u\n\
  system_api_version: %u.%u.%u\n";
  // NOLINTEND

  if (interface == nullptr) {
    // Return the formatted string directly
    return absl::StrFormat(kNullDumpTpl, kQnnSystemInterfaceHeader);
  }

  const auto system_version = interface->systemApiVersion;

  // Return the formatted string directly
  return absl::StrFormat(kQnnSystemInterfaceDumpTpl, kQnnSystemInterfaceHeader,
                         interface->providerName, interface->backendId,
                         system_version.major, system_version.minor,
                         system_version.patch);
}

}  // namespace

// Changed signature: Returns std::string
std::string Dump(const QnnManager& qnn) {
  // Call the internal Dump functions, get their strings, and concatenate them.
  // absl::StrCat is generally preferred over repeated string operator+=
  return absl::StrCat(Dump(qnn.interface_), Dump(qnn.system_interface_));
}
}  // namespace litert::qnn::internal
