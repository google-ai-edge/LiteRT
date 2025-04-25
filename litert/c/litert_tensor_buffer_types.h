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

#ifndef ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_TYPES_H_
#define ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_TYPES_H_

typedef enum {
  kLiteRtTensorBufferTypeUnknown = 0,
  kLiteRtTensorBufferTypeHostMemory = 1,
  kLiteRtTensorBufferTypeAhwb = 2,
  kLiteRtTensorBufferTypeIon = 3,
  kLiteRtTensorBufferTypeDmaBuf = 4,
  kLiteRtTensorBufferTypeFastRpc = 5,
  kLiteRtTensorBufferTypeGlBuffer = 6,
  kLiteRtTensorBufferTypeGlTexture = 7,

  // 10-19 are reserved for OpenCL memory objects.
  kLiteRtTensorBufferTypeOpenClBuffer = 10,
  kLiteRtTensorBufferTypeOpenClBufferFp16 = 11,
  kLiteRtTensorBufferTypeOpenClTexture = 12,
  kLiteRtTensorBufferTypeOpenClTextureFp16 = 13,
  kLiteRtTensorBufferTypeOpenClImageBuffer = 14,
  kLiteRtTensorBufferTypeOpenClImageBufferFp16 = 15,
} LiteRtTensorBufferType;

inline bool IsOpenClMemory(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeOpenClBuffer ||
         buffer_type == kLiteRtTensorBufferTypeOpenClBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeOpenClTexture ||
         buffer_type == kLiteRtTensorBufferTypeOpenClTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeOpenClImageBuffer ||
         buffer_type == kLiteRtTensorBufferTypeOpenClImageBufferFp16;
}

#endif  // ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_TYPES_H_
