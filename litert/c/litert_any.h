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

#ifndef ODML_LITERT_LITERT_C_LITERT_ANY_H_
#define ODML_LITERT_LITERT_C_LITERT_ANY_H_

#include <stdbool.h>  // NOLINT: To use bool type in C
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  kLiteRtAnyTypeNone = 0,
  kLiteRtAnyTypeBool = 1,
  kLiteRtAnyTypeInt = 2,
  kLiteRtAnyTypeReal = 3,
  kLiteRtAnyTypeString = 8,
  kLiteRtAnyTypeVoidPtr = 9,
} LiteRtAnyType;

inline const char* LiteRtAnyTypeToString(LiteRtAnyType type) {
  switch (type) {
    case kLiteRtAnyTypeNone:
      return "kLiteRtAnyTypeNone";
    case kLiteRtAnyTypeBool:
      return "kLiteRtAnyTypeBool";
    case kLiteRtAnyTypeInt:
      return "kLiteRtAnyTypeInt";
    case kLiteRtAnyTypeReal:
      return "kLiteRtAnyTypeReal";
    case kLiteRtAnyTypeString:
      return "kLiteRtAnyTypeString";
    case kLiteRtAnyTypeVoidPtr:
      return "kLiteRtAnyTypeVoidPtr";
  }
  return "Unknown";
}

typedef struct {
  LiteRtAnyType type;
  union {
    bool bool_value;
    int64_t int_value;
    double real_value;
    const char* str_value;
    const void* ptr_value;
  };
} LiteRtAny;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_ANY_H_
