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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_C_TYPES_PRINTING_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_C_TYPES_PRINTING_H_

#include <string>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"

// AbslStringify specializations for types in the litert c api.
// TODO: lukeboyer - Migrate code in tools/dump.h to leverage the abseil
// stringify framework.

template <class Sink>
void AbslStringify(Sink& sink, const LiteRtElementType& type) {
  std::string dtype_str;
  switch (type) {
    case kLiteRtElementTypeInt32:
      dtype_str = "i32";
      break;
    case kLiteRtElementTypeInt64:
      dtype_str = "i64";
      break;
    case kLiteRtElementTypeFloat32:
      dtype_str = "f32";
      break;
    case kLiteRtElementTypeFloat64:
      dtype_str = "f64";
      break;
    case kLiteRtElementTypeInt16:
      dtype_str = "i16";
      break;
    case kLiteRtElementTypeInt8:
      dtype_str = "i8";
      break;
    case kLiteRtElementTypeUInt8:
      dtype_str = "u8";
      break;
    case kLiteRtElementTypeUInt16:
      dtype_str = "u16";
      break;
    case kLiteRtElementTypeUInt32:
      dtype_str = "u32";
      break;
    case kLiteRtElementTypeUInt64:
      dtype_str = "u64";
      break;
    case kLiteRtElementTypeBool:
      dtype_str = "i1";
      break;
    default:
      dtype_str = "UNKNOWN_ELEMENT_TYPE";
      break;
  }

  absl::Format(&sink, "%s", dtype_str);
}

template <class Sink>
void AbslStringify(Sink& sink, const LiteRtLayout& layout) {
  absl::Format(
      &sink, "<%s>",
      absl::StrJoin(absl::MakeConstSpan(layout.dimensions, layout.rank), "x"));
}

template <class Sink>
void AbslStringify(Sink& sink, const LiteRtRankedTensorType& type) {
  const auto& layout = type.layout;
  absl::Format(&sink, "%ud_%v%v", layout.rank, type.element_type, layout);
}

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_C_TYPES_PRINTING_H_
