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

#include "litert/c/litert_layout.h"

#include <algorithm>
#include <cstddef>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

LiteRtStatus LiteRtGetNumLayoutElements(const LiteRtLayout* layout,
                                        size_t* num_elements) {
  if (!layout || !num_elements) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_elements = 1;
  for (size_t i = 0; i < layout->rank; ++i) {
    if (layout->dimensions[i] < 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    *num_elements *= layout->dimensions[i];
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtIsSameLayout(const LiteRtLayout* layout1,
                                const LiteRtLayout* layout2, bool* result) {
  if (!layout1 || !layout2 || !result) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  if ((layout1->rank != layout2->rank) ||
      (layout1->has_strides != layout2->has_strides)) {
    *result = false;
  } else if (!std::equal(layout1->dimensions,
                         layout1->dimensions + layout1->rank,
                         layout2->dimensions)) {
    *result = false;
  } else if (layout1->has_strides &&
             !std::equal(layout1->strides, layout1->strides + layout1->rank,
                         layout2->strides)) {
    *result = false;
  } else {
    *result = true;
  }

  return kLiteRtStatusOk;
}

#ifdef __cplusplus
}  // extern "C"
#endif
