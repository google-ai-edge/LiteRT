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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_RUNTIME_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_RUNTIME_OPTIONS_H_

// Internal LiteRt runtime options struct. This data structure is used to
// pass runtime options to the interpreter and will be used in the framework
// code.
struct LiteRtRuntimeOptionsT {
  // If true, the interpreter will inline composite ops.
  bool shlo_composite_inlining = false;

  // If true, the interpreter will enable profiling.
  bool enable_profiling = false;

  static const char* Identifier() { return "runtime"; }
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_RUNTIME_OPTIONS_H_
