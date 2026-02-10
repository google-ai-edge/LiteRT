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

#ifndef ODML_LITERT_LITERT_C_LITERT_SCHEDULING_INFO_H_
#define ODML_LITERT_LITERT_C_LITERT_SCHEDULING_INFO_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Bitmask values describing which fields in `LiteRtSchedulingInfo` are
// explicitly provided.
typedef enum LiteRtSchedulingInfoField {
  kLiteRtSchedulingInfoFieldOriginalUid = 1u << 0,
  kLiteRtSchedulingInfoFieldDebugFeatureId = 1u << 1,
  kLiteRtSchedulingInfoFieldJobPriority = 1u << 2,
  kLiteRtSchedulingInfoFieldGroupId = 1u << 3,
} LiteRtSchedulingInfoField;

// Scheduling information associated with an inference request. This data is
// passed from application code through LiteRT to vendor delegates (e.g. NPU
// dispatch implementations).
//
// Fields are considered set if the corresponding bit is present in
// `fields_mask`.
typedef struct LiteRtSchedulingInfo {
  uint32_t fields_mask;
  int32_t original_uid;
  int32_t debug_feature_id;
  int32_t job_priority;
  int32_t group_id;
} LiteRtSchedulingInfo;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_SCHEDULING_INFO_H_
