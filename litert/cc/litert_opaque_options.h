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

#ifndef ODML_LITERT_LITERT_CC_LITERT_OPAQUE_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_LITERT_OPAQUE_OPTIONS_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

class OpaqueOptions
    : public internal::Handle<LiteRtOpaqueOptions, LiteRtDestroyOpaqueOptions> {
  // TODO: lukeboyer - Work logging into this api (and derived classes).
 public:
  using Ref = std::reference_wrapper<OpaqueOptions>;

  OpaqueOptions() = default;

  // Parameter `owned` indicates if the created AcceleratorCompilationOptions
  // object should take ownership of the provided `options` handle.
  explicit OpaqueOptions(LiteRtOpaqueOptions options, OwnHandle owned)
      : Handle(options, owned) {}

  static Expected<OpaqueOptions> Create(
      const std::string& payload_identifier, void* payload_data,
      void (*payload_destructor)(void* payload_data)) {
    LiteRtOpaqueOptions options;
    LITERT_RETURN_IF_ERROR(
        LiteRtCreateOpaqueOptions(payload_identifier.c_str(), payload_data,
                                  payload_destructor, &options));
    return OpaqueOptions(options, OwnHandle::kYes);
  }

  Expected<absl::string_view> GetIdentifier() const {
    const char* payload_identifier;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetOpaqueOptionsIdentifier(Get(), &payload_identifier));
    return absl::string_view(payload_identifier);
  }

  template <typename T>
  Expected<T*> GetData() const {
    void* payload_data;
    LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(Get(), &payload_data));
    return reinterpret_cast<T*>(payload_data);
  }

  Expected<OpaqueOptions> Next() {
    auto h = Get();
    LITERT_RETURN_IF_ERROR(LiteRtGetNextOpaqueOptions(&h));
    return OpaqueOptions(h, OwnHandle::kNo);
  }

  Expected<void> Append(OpaqueOptions&& appended_options) {
    auto h = Get();
    LITERT_RETURN_IF_ERROR(
        LiteRtAppendOpaqueOptions(&h, appended_options.Release()));
    if (h != Get()) {
      // If appending a new linked list item has changed the linked list head
      // pointer, then we need to reflect that as the new handle. Note that
      // should happen only if the previous handle was null.
      assert(!Get());
      *this = OpaqueOptions(h, OwnHandle::kYes);
    }
    return {};
  }

  Expected<void> Pop() {
    auto h = Get();
    LITERT_RETURN_IF_ERROR(LiteRtPopOpaqueOptions(&h));
    if (h != Get()) {
      // If popping the last item has changed the linked list head pointer, then
      // we release the current handle since it has been already destructed by
      // the pop call, and then use the new head pointer as the new handle.
      (void)Release();
      *this = OpaqueOptions(h, OwnHandle::kYes);
    }
    return {};
  }

  Expected<void> SetHash(LiteRtOpaqueOptionsHashFunc payload_hash_func);

  Expected<uint64_t> Hash() const;
};

// Find the opaque option in the chain that matches the provided identifier.
Expected<OpaqueOptions> FindOpaqueOptions(
    OpaqueOptions& options, const std::string& payload_identifier);

// Convience wrapper on top of Find. Resolves the matching opaque options
// as an instance of the type derived from OpaqueOptions, provided it implements
// the required hooks. This is analogous to FindData, except that it returns
// the C++ wrapper associated with an opaque options type, rather than the raw
// payload data.
template <typename Discriminated,
          std::enable_if_t<std::is_base_of<OpaqueOptions, Discriminated>::value,
                           bool> = true>
Expected<Discriminated> FindOpaqueOptions(OpaqueOptions& options) {
  LITERT_ASSIGN_OR_RETURN(
      auto opq,
      FindOpaqueOptions(options, std::string(Discriminated::Discriminator())));
  return Discriminated::Create(opq);
}

// Find the raw data payload of given type from the opaque option in the chain
// with matching identifier. This is analogous to Discriminate, except that
// it returns the payload used to interopt with the C API.
template <typename T>
Expected<T*> FindOpaqueData(OpaqueOptions& options,
                            const std::string& payload_identifier) {
  void* payload_data;
  LITERT_RETURN_IF_ERROR(LiteRtFindOpaqueOptionsData(
      options.Get(), payload_identifier.c_str(), &payload_data));
  return reinterpret_cast<T*>(payload_data);
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_OPAQUE_OPTIONS_H_
