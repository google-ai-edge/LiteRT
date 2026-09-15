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

#ifndef ODML_LITERT_LITERT_C_LITERT_VERSION_H_
#define ODML_LITERT_LITERT_C_LITERT_VERSION_H_

#ifndef LITERT_VERSION_STRING
#ifndef LITERT_MAJOR_VERSION
#define LITERT_MAJOR_VERSION 2
#endif
#ifndef LITERT_MINOR_VERSION
#define LITERT_MINOR_VERSION 3
#endif
#ifndef LITERT_PATCH_VERSION
#define LITERT_PATCH_VERSION 0
#endif
#ifndef LITERT_VERSION_SUFFIX
#define LITERT_VERSION_SUFFIX ""
#endif
#ifndef LITERT_STR_HELPER
#define LITERT_STR_HELPER(x) #x
#define LITERT_STR(x) LITERT_STR_HELPER(x)
#endif
#define LITERT_VERSION_STRING                                         \
  (LITERT_STR(LITERT_MAJOR_VERSION) "." LITERT_STR(LITERT_MINOR_VERSION) \
   "." LITERT_STR(LITERT_PATCH_VERSION) LITERT_VERSION_SUFFIX)
#endif  // LITERT_VERSION_STRING

#endif  // ODML_LITERT_LITERT_C_LITERT_VERSION_H_
