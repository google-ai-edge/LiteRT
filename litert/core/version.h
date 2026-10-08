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

#ifndef ODML_LITERT_LITERT_CORE_VERSION_H_
#define ODML_LITERT_LITERT_CORE_VERSION_H_

#include <cstdint>
#include <type_traits>

#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/litert_common.h"

namespace litert::internal {

// Return true if two API versions are the same.
inline bool IsSameVersion(const LiteRtApiVersion& v1,
                          const LiteRtApiVersion& v2) {
  return (v1.major == v2.major) && (v1.minor == v2.minor) &&
         (v1.patch == v2.patch);
}

// Return true if a given API version is the same as the current runtime.
inline bool IsSameVersionAsRuntime(const LiteRtApiVersion& v) {
  return IsSameVersion(v, {LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
                           LITERT_API_VERSION_PATCH});
}

// Return true if the vendor version is compatible with the runtime version
// under Option B (Major Version Tables + LiteRtAbiHeader at Offset 0).
// Compatibility rules:
// 1. Major version must match exactly.
// 2. If major is 0, minor version must match exactly (pre-1.0 breaking
//    changes).
// 3. If major >= 1, both backward (runtime.minor >= vendor.minor) and forward
//    (runtime.minor < vendor.minor) minor versions are ABI-compatible because
//    Option B structs are append-only and bounds-checked via struct_size.
// 4. Patch versions are ignored for compatibility.
inline bool IsCompatibleVersion(const LiteRtApiVersion& vendor_version,
                                const LiteRtApiVersion& runtime_version) {
  if (vendor_version.major != runtime_version.major) {
    return false;
  }
  if (runtime_version.major == 0) {
    return vendor_version.minor == runtime_version.minor;
  }
  return true;
}

// Generic single-call version negotiation helper implementing Option B
// (Major Version Tables + LiteRtAbiHeader at Offset 0).
//
// Queries the vendor plugin once with `runtime_version`, inspects the
// returned table's `LiteRtAbiHeader` at offset 0, verifies that
// `abi_header.major_version == expected_abi_major` and
// `abi_header.struct_size >= sizeof(LiteRtAbiHeader)`.
//
// Memory ownership: The returned interface table pointed to by
// `*out_interface` is owned by the vendor plugin (typically stored as a
// static const table in the plugin's .rodata or static storage). The caller is
// NOT responsible for freeing it and must NOT attempt to delete or free it.
// The interface table remains valid for the lifetime of the loaded plugin.
//
// Template arguments:
//   QueryFn: A callable matching `LiteRtStatus(InterfaceId, LiteRtApiVersion,
//   LiteRtInterface*)`
//   InterfaceId: The enum or type identifying the interface.
//
// Returns kLiteRtStatusOk if an interface was successfully negotiated, or an
// error code (e.g. kLiteRtStatusErrorUnsupported) otherwise.
template <typename QueryFn, typename InterfaceId>
inline LiteRtStatus NegotiateInterface(
    QueryFn query_fn, InterfaceId interface_id,
    const LiteRtApiVersion& runtime_version, uint16_t expected_abi_major,
    LiteRtInterface* out_interface,
    LiteRtApiVersion* out_negotiated_version = nullptr) {
  if (out_interface == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if constexpr (std::is_pointer_v<QueryFn>) {
    if (query_fn == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }

  *out_interface = nullptr;
  LiteRtStatus status = query_fn(interface_id, runtime_version, out_interface);
  if (status != kLiteRtStatusOk) {
    return status;
  }
  if (*out_interface == nullptr) {
    return kLiteRtStatusErrorRuntimeFailure;
  }

  const auto* header = static_cast<const LiteRtAbiHeader*>(*out_interface);
  if (header->struct_size < sizeof(LiteRtAbiHeader) ||
      header->major_version != expected_abi_major) {
    *out_interface = nullptr;
    return kLiteRtStatusErrorWrongVersion;
  }

  if (out_negotiated_version != nullptr) {
    *out_negotiated_version = {header->major_version, header->minor_version, 0};
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_VERSION_H_
