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

#include "litert/cc/options/litert_dispatch_delegate_vendor_options.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/core/litert_toml_parser.h"

namespace litert {

std::string DispatchDelegateVendorOptions::ToToml() const {
  std::string toml;
  for (const FunctionMapping& mapping : function_mappings_) {
    absl::StrAppendFormat(&toml, "func.%s.signature = \"%s\"\n",
                          mapping.function_name, mapping.signature_name);
    for (const TensorPortMapping& port : mapping.input_tensor_ports) {
      absl::StrAppendFormat(&toml, "func.%s.in.%d = \"%s\"\n",
                            mapping.function_name, port.port_index,
                            port.tensor_name);
    }
    for (const TensorPortMapping& port : mapping.output_tensor_ports) {
      absl::StrAppendFormat(&toml, "func.%s.out.%d = \"%s\"\n",
                            mapping.function_name, port.port_index,
                            port.tensor_name);
    }
  }
  return toml;
}

Expected<DispatchDelegateVendorOptions>
DispatchDelegateVendorOptions::CreateFromToml(absl::string_view toml_string) {
  if (toml_string.empty()) {
    return DispatchDelegateVendorOptions();
  }

  DispatchDelegateVendorOptions options;

  // Helper lambda to get a reference to an existing FunctionMapping with the
  // given name, or create a new one and append it to the options.
  auto get_or_create_mapping =
      [&options](absl::string_view function_name) -> FunctionMapping& {
    for (FunctionMapping& m : options.function_mappings_) {
      if (m.function_name == function_name) {
        return m;
      }
    }
    options.function_mappings_.push_back(
        FunctionMapping{std::string(function_name), "", {}, {}});
    return options.function_mappings_.back();
  };

  const LiteRtStatus toml_status = litert::internal::ParseToml(
      toml_string,
      [&get_or_create_mapping](absl::string_view key,
                               absl::string_view value) -> LiteRtStatus {
        absl::string_view rest = key;
        if (!absl::ConsumePrefix(&rest, "func.")) {
          return kLiteRtStatusOk;
        }

        // Check for signature: func.<name>.signature = "..."
        if (absl::ConsumeSuffix(&rest, ".signature")) {
          get_or_create_mapping(rest).signature_name = std::string(value);
          return kLiteRtStatusOk;
        }

        // Check for input: func.<name>.in.<port> = "..."
        if (size_t in_pos = rest.rfind(".in.");
            in_pos != absl::string_view::npos) {
          absl::string_view function_name = rest.substr(0, in_pos);
          absl::string_view port_str = rest.substr(in_pos + 4);
          LITERT_ASSIGN_OR_RETURN(int64_t port_idx,
                                  litert::internal::ParseTomlInt(port_str));
          get_or_create_mapping(function_name)
              .input_tensor_ports.push_back(
                  {std::string(value), static_cast<int>(port_idx)});
          return kLiteRtStatusOk;
        }

        // Check for output: func.<name>.out.<port> = "..."
        if (size_t out_pos = rest.rfind(".out.");
            out_pos != absl::string_view::npos) {
          absl::string_view function_name = rest.substr(0, out_pos);
          absl::string_view port_str = rest.substr(out_pos + 5);
          LITERT_ASSIGN_OR_RETURN(int64_t port_idx,
                                  litert::internal::ParseTomlInt(port_str));
          get_or_create_mapping(function_name)
              .output_tensor_ports.push_back(
                  {std::string(value), static_cast<int>(port_idx)});
          return kLiteRtStatusOk;
        }

        return kLiteRtStatusOk;
      });

  LITERT_RETURN_IF_ERROR(toml_status);
  return options;
}

LiteRtStatus DispatchDelegateVendorOptions::GetOpaqueOptionsData(
    const char** identifier, void** payload,
    void (**payload_deleter)(void*)) const {
  if (identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  std::string toml = ToToml();
  char* str = strdup(toml.c_str());
  if (str == nullptr) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  *identifier = kIdentifier;
  *payload = str;
  *payload_deleter = [](void* p) { ::free(p); };
  return kLiteRtStatusOk;
}

Expected<DispatchDelegateVendorOptions>
DispatchDelegateVendorOptions::FromOpaqueOptions(OpaqueOptions& options) {
  LITERT_ASSIGN_OR_RETURN(
      const char* toml_payload,
      FindOpaqueData<const char>(options, kIdentifier));
  if (toml_payload == nullptr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Null payload for dispatch delegate vendor options");
  }
  return CreateFromToml(toml_payload);
}

}  // namespace litert
