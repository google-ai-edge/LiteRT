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

#ifndef ODML_LITERT_LITERT_TFLITE_SUPPORT_OP_RESOLVER_OP_RESOLVER_H_
#define ODML_LITERT_LITERT_TFLITE_SUPPORT_OP_RESOLVER_OP_RESOLVER_H_

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"

namespace tflite {
class MutableOpResolver;
}  // namespace tflite

namespace litert {
namespace tflite_support {

/// Sets a `tflite::MutableOpResolver` on `litert::Options` to resolve built-in
/// and custom ops when creating a `CompiledModel`.
///
/// Example usage:
/// @code
/// tflite::MutableOpResolver resolver;
/// RegisterSelectedOps(&resolver);
/// litert::tflite_support::SetOpResolver(options, &resolver);
/// @endcode
///
/// Note: `resolver` must outlive `CompiledModel::Create()`. This API is
/// experimental and is designed to support selective op registration in TFLite.
/// It is subject to change or removal at any time without notice. This API is
/// not ABI stable and should only be used in static runtime builds.
Expected<void> SetOpResolver(Options& options,
                             const tflite::MutableOpResolver* resolver);

}  // namespace tflite_support
}  // namespace litert

#endif  // ODML_LITERT_LITERT_TFLITE_SUPPORT_OP_RESOLVER_OP_RESOLVER_H_
