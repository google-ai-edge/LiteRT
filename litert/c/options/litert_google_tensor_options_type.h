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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_TYPE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_TYPE_H_

// float_truncation_type -------------------------------------------------------

typedef enum LiteRtGoogleTensorOptionsTruncationType {
  kLiteRtGoogleTensorFloatTruncationTypeAuto = 0,
  kLiteRtGoogleTensorFloatTruncationTypeNoTruncation = 1,
  kLiteRtGoogleTensorFloatTruncationTypeBfloat16 = 2,
  kLiteRtGoogleTensorFloatTruncationTypeHalf = 3,
} LiteRtGoogleTensorOptionsTruncationType;

typedef enum LiteRtGoogleTensorOptionsShardingIntensity {
  kLiteRtGoogleTensorShardingIntensityMinimal = 0,
  kLiteRtGoogleTensorShardingIntensityModerate = 1,
  kLiteRtGoogleTensorShardingIntensityExtensive = 2,
  kLiteRtGoogleTensorShardingIntensityMaximum = 3,
} LiteRtGoogleTensorOptionsShardingIntensity;

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_TYPE_H_
