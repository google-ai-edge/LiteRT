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

#ifndef ODML_LITERT_LITERT_TFLITE_SUPPORT_OP_RESOLVER_SELECTED_OPS_REGISTRY_H_
#define ODML_LITERT_LITERT_TFLITE_SUPPORT_OP_RESOLVER_SELECTED_OPS_REGISTRY_H_

namespace tflite {
class MutableOpResolver;
}  // namespace tflite

namespace litert {
namespace tflite_support {

using SelectedOpsCallback = void (*)(::tflite::MutableOpResolver*);

// Registers a callback that populates a `tflite::MutableOpResolver` with
// selected ops generated at build time (e.g. via `models = [...]` on
// `litert_cc_library`, `litert_cc_binary`, or `litert_cc_test`).
// Safe to call during C++ static initialization (`__init_array`).
void RegisterSelectedOpsCallback(SelectedOpsCallback callback);

// Returns true if at least one selected-ops callback has been registered.
bool HasSelectedOpsCallback();

// Invokes all registered selected-ops callbacks on `resolver`.
void ApplySelectedOpsCallbacks(::tflite::MutableOpResolver* resolver);

}  // namespace tflite_support
}  // namespace litert

#endif  // ODML_LITERT_LITERT_TFLITE_SUPPORT_OP_RESOLVER_SELECTED_OPS_REGISTRY_H_
