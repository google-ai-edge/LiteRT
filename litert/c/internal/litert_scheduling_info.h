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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_SCHEDULING_INFO_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_SCHEDULING_INFO_H_

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

enum {
  // Group id is a raw 16-byte UUID value.
  kLiteRtSchedulingInfoGroupIdSize = 16,
};

typedef enum LiteRtSchedulingJobPriority {
  // Lower numeric values indicate higher priority.
  kLiteRtSchedulingInfoJobPriorityHighest = 0,
  kLiteRtSchedulingInfoJobPriorityLowest = 1000,
} LiteRtSchedulingJobPriority;

// Scheduling information associated with an inference request. This data is
// passed from application code through LiteRT to vendor delegates (e.g. NPU
// dispatch implementations).
//
// Fields are considered set if the corresponding bit is present in
// `fields_mask`.
//
// This is reserved for 1P access only (Android NPU Manager).
// TODO (b/484042050): Apply ABI check to ensure this struct remains ABI stable.
typedef struct LiteRtSchedulingInfo {
  uint32_t fields_mask;
  int32_t original_uid;
  // Null-terminated debug feature string, e.g.
  // "com.android.aicore.text_summarization".
  const char* debug_feature_id;
  // Priority in [kLiteRtSchedulingInfoJobPriorityHighest,
  // kLiteRtSchedulingInfoJobPriorityLowest], where smaller values indicate
  // higher priority.
  int32_t job_priority;
  // Raw 16-byte UUID identifying a related scheduling group.
  uint8_t group_id[kLiteRtSchedulingInfoGroupIdSize];
} LiteRtSchedulingInfo;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_SCHEDULING_INFO_H_
