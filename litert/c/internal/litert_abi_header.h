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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_ABI_HEADER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_ABI_HEADER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Header containing ABI version and size metadata.
 *
 * Must be the first member of any ABI-stable struct.
 * Size: 8 bytes. Padding: 0 bytes (on 64-bit systems).
 */
typedef struct LiteRtAbiHeader {
  // The physical size of the parent structure (including this header).
  // Supports struct sizes up to 64KB, which is safe for C API tables.
  uint16_t struct_size;

  // Bumped when introducing breaking changes or a completely new struct layout.
  uint16_t major_version;

  // Bumped when adding new methods (appended) or deprecating existing ones.
  uint16_t minor_version;

  // Reserved for future use (e.g., flags or compatibility metadata).
  uint16_t reserved;
} LiteRtAbiHeader;

/**
 * @brief Checks if the provider's ABI version is compatible with the consumer's
 * requirements.
 *
 * @param instance_ptr Pointer to the struct instance (e.g.,
 * LiteRtRuntimeContext).
 * @param req_major The major version required by the consumer.
 * @param req_minor The minimum minor version required by the consumer.
 */
#define LITERT_ABI_IS_COMPATIBLE(instance_ptr, req_major, req_minor) \
  ((instance_ptr)->abi_header.major_version == (req_major) &&        \
   (instance_ptr)->abi_header.minor_version >= (req_minor))

/**
 * @brief Safely verifies that an API member is version-compatible, physically
 * present in memory, and implemented (non-null).
 *
 * @param instance_ptr Pointer to the ABI-versioned struct instance.
 * @param req_major The expected major version of the struct layout.
 * @param api_member The name of the function pointer / member to check.
 */
#define LITERT_ABI_HAS_API(instance_ptr, req_major, api_member)              \
  ((instance_ptr) != nullptr &&                                              \
   (instance_ptr)->abi_header.major_version == (req_major) &&                \
   (instance_ptr)->abi_header.struct_size >=                                 \
       ((const char*)(&(instance_ptr)->api_member) +                         \
        sizeof((instance_ptr)->api_member) - (const char*)(instance_ptr)) && \
   (instance_ptr)->api_member != nullptr)

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_ABI_HEADER_H_
