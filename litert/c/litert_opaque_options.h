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

#ifndef ODML_LITERT_LITERT_C_LITERT_OPAQUE_OPTIONS_H_
#define ODML_LITERT_LITERT_C_LITERT_OPAQUE_OPTIONS_H_

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// A linked list of versioned accelerator compilation options. List items
// include:
//
// - a unique payload identifier field (string), used to distinguish payloads of
//   different types;
//
// - a payload field and associated payload destructor callback;
//
// - a payload version field, used by the consumer code to know the structure of
//   the payload.
LITERT_DEFINE_HANDLE(LiteRtOpaqueOptions);

LiteRtStatus LiteRtCreateOpaqueOptions(
    const LiteRtApiVersion* payload_version, const char* payload_identifier,
    void* payload_data, void (*payload_destructor)(void* payload_data),
    LiteRtOpaqueOptions* options);

// Releases an entire options list starting from `options`.
//
// Warning: Once an `options` item has been appended to another `options` item,
// the user will no longer need to destoy the former `options` item manually
// with this function.
void LiteRtDestroyOpaqueOptions(LiteRtOpaqueOptions options);

// Gets the payload version field of the first item in the given `options` list.
LiteRtStatus LiteRtGetOpaqueOptionsVersion(LiteRtOpaqueOptions options,
                                           LiteRtApiVersion* payload_version);

// Gets the patload identifier field of the first item in the given `options`
// list.
LiteRtStatus LiteRtGetOpaqueOptionsIdentifier(LiteRtOpaqueOptions options,
                                              const char** payload_identifier);

// Gets the payload data field of the first item in the given `options` list.
LiteRtStatus LiteRtGetOpaqueOptionsData(LiteRtOpaqueOptions options,
                                        void** payload_data);

// Gets the payload version and data for the `options` list item with a given
// payload identifier. Return kLiteRtStatusErrorNotFound if not such item is
// found.
LiteRtStatus LiteRtFindOpaqueOptionsData(LiteRtOpaqueOptions options,
                                         const char* payload_identifier,
                                         LiteRtApiVersion* payload_version,
                                         void** payload_data);

// Iterate through the next item in the option list pointed by `options` and
// sets parameter `options` to null if there is no next item.
LiteRtStatus LiteRtGetNextOpaqueOptions(LiteRtOpaqueOptions* options);

// Appends `next_options` to the list ponted by `options` and takes ownership of
// the appended object. While parameter `options` must be non-null, `*options`
// may however be null, in which case this call is equivalent to `*options =
// appended_options`.
LiteRtStatus LiteRtAppendOpaqueOptions(LiteRtOpaqueOptions* options,
                                       LiteRtOpaqueOptions appended_options);

// Removes and deallocates the last option in the linked list pointed by
// parameter `options`.
LiteRtStatus LiteRtPopOpaqueOptions(LiteRtOpaqueOptions* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ODML_LITERT_LITERT_C_LITERT_OPAQUE_OPTIONS_H_
