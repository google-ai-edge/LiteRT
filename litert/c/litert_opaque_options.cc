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

#include "litert/c/litert_opaque_options.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "litert/c/litert_common.h"

struct LiteRtOpaqueOptionsT {
  std::string payload_identifier;
  std::unique_ptr<void, void (*)(void*)> payload_data;
  LiteRtOpaqueOptionsT* next = nullptr;
  LiteRtOpaqueOptionsHashFunc payload_hash_func = nullptr;

  LiteRtOpaqueOptionsT(std::string payload_identifier_, void* payload_data_,
                       void (*payload_destructor_)(void*))
      : payload_identifier(std::move(payload_identifier_)),
        payload_data(payload_data_, payload_destructor_) {}
};

LiteRtStatus LiteRtCreateOpaqueOptions(const char* payload_identifier,
                                       void* payload_data,
                                       void (*payload_destructor)(void*),
                                       LiteRtOpaqueOptions* options) {
  if (!payload_identifier || !payload_data || !payload_destructor || !options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LiteRtOpaqueOptionsT(std::string(payload_identifier),
                                      payload_data, payload_destructor);
  return kLiteRtStatusOk;
}

void LiteRtDestroyOpaqueOptions(LiteRtOpaqueOptions options) {
  while (options) {
    LiteRtOpaqueOptions next = options->next;
    delete options;
    options = next;
  }
}

LiteRtStatus LiteRtGetOpaqueOptionsIdentifier(LiteRtOpaqueOptions options,
                                              const char** payload_identifier) {
  if (!options || !payload_identifier) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *payload_identifier = options->payload_identifier.c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetOpaqueOptionsData(LiteRtOpaqueOptions options,
                                        void** payload_data) {
  if (!options || !payload_data) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *payload_data = options->payload_data.get();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFindOpaqueOptionsData(LiteRtOpaqueOptions options,
                                         const char* payload_identifier,
                                         void** payload_data) {
  if (!options || !payload_identifier || !payload_data) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  while (options) {
    if (options->payload_identifier == payload_identifier) {
      *payload_data = options->payload_data.get();
      return kLiteRtStatusOk;
    } else {
      options = options->next;
    }
  }
  return kLiteRtStatusErrorNotFound;
}

LiteRtStatus LiteRtGetNextOpaqueOptions(LiteRtOpaqueOptions* options) {
  if (!options || !*options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = (*options)->next;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAppendOpaqueOptions(LiteRtOpaqueOptions* options,
                                       LiteRtOpaqueOptions appended_options) {
  if (!options || !appended_options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  while (*options) {
    options = &((*options)->next);
  }
  *options = appended_options;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtPopOpaqueOptions(LiteRtOpaqueOptions* options) {
  if (!options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtOpaqueOptions* last = options;
  while ((*last)->next) {
    last = &(*last)->next;
  }
  if (*last) {
    LiteRtDestroyOpaqueOptions(*last);
    *last = nullptr;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetOpaqueOptionsHash(
    LiteRtOpaqueOptions options,
    LiteRtOpaqueOptionsHashFunc payload_hash_func) {
  if (!options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->payload_hash_func = payload_hash_func;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetOpaqueOptionsHash(LiteRtOpaqueOptions options,
                                        uint64_t* hash) {
  if (!options || !hash) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!options->payload_hash_func) {
    // Hash function not set for these options.
    return kLiteRtStatusErrorUnsupported;
  }
  *hash = options->payload_hash_func(options->payload_data.get());
  return kLiteRtStatusOk;
}
