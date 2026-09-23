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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ENTRY_EMBEDDING_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ENTRY_EMBEDDING_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Transforms Vision Encoder entry positional embedding from
// OneHot(depth 10240) + Select + FC(768, 10240) + Concat + Sum
// to Gather(10240, 768) + Add, saving ~51.6MB DRAM traffic and ~16ms.
//
// Before:
//   coord_x [1, 630]                    coord_y [1, 630]
//          |                                   |
//   +------v------+                     +------v------+
//   |   OneHot /  |                     |   OneHot /  |
//   |   SelectV2  |                     |   SelectV2  |
//   +------┬------+                     +------┬------+
//          |                                   |
//   +------v------+                     +------v------+
//   | FC [768,    | <-- w_x             | FC [768,    | <-- w_y
//   |     10240]  |     [768, 10240]    |     10240]  |     [768, 10240]
//   +------┬------+                     +------┬------+
//          |                                   |
//   +------v------+                     +------v------+
//   |   Reshape   |                     |   Reshape   |
//   | [1,630,768] |                     | [1,630,768] |
//   +------┬------+                     +------┬------+
//          \                                   /
//           +───────────────┬─────────────────+
//                           |
//                  +--------v--------+
//                  |  Concatenation  |
//                  +--------┬--------+
//                           |
//                  +--------v--------+
//                  |       Sum       |
//                  +--------┬--------+
//                           |
//                  sum_out [1, 630, 768]
//
// After:
//   w_x_t [10240, 768]  coord_x [1, 630]  w_y_t [10240, 768]  coord_y [1, 630]
//            \                 /                   \                 /
//          +──v───────────────v──+               +──v───────────────v──+
//          |   Gather (axis=0)   |               |   Gather (axis=0)   |
//          |    [1, 630, 768]    |               |    [1, 630, 768]    |
//          +──────────┬──────────+               +──────────┬──────────+
//                     \                                     /
//                      +─────────────────┬─────────────────+
//                                        |
//                               +────────v────────+
//                               |       Add       |
//                               +────────┬────────+
//                                        |
//                               sum_out [1, 630, 768]
LiteRtStatus EntryEmbeddingTransformation(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ENTRY_EMBEDDING_TRANSFORMATION_H_
