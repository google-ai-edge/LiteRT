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

/// @internal
/// Sets a `tflite::MutableOpResolver` on `litert::Options` to resolve built-in
/// and custom ops when creating a `CompiledModel`.
/// @warning This is for internal use only.
///
/// Example usage:
/// @code
/// tflite::MutableOpResolver resolver;
/// RegisterSelectedOps(&resolver);
/// litert::tflite_support::SetOpResolver(options, &resolver);
/// @endcode
///
/// Note: `resolver` must outlive the `CompiledModel` created with these
/// options. This API is experimental and intended for internal/static runtime
/// builds only (e.g. selective op registration via `RegisterSelectedOps`).
/// It is not ABI-stable and is only supported when using the in-process builtin
/// runtime; calling it against a dynamic runtime boundary returns an
/// unsupported error. For ABI-stable op resolution across runtime boundaries,
/// use `SetOpaqueOpResolver` or `LiteRtSetOpResolverOption`.
Expected<void> SetOpResolver(Options& options,
                             const tflite::MutableOpResolver* resolver);

/// Sets an ABI-stable op resolver callback to resolve built-in and custom ops
/// across runtime boundaries.
///
/// Note: Registrations returned by `c_api` callbacks must have
/// `registration_external` (`TfLiteOperator*`) set. Non-opaque
/// `TfLiteRegistration`s without `registration_external` will be rejected.
/// (See b/580021727 for supporting `TfLiteOperator*` directly.)
Expected<void> SetOpaqueOpResolver(Options& options,
                                   const LiteRtOpResolverCApi& c_api);

}  // namespace tflite_support
}  // namespace litert

#endif  // ODML_LITERT_LITERT_TFLITE_SUPPORT_OP_RESOLVER_OP_RESOLVER_H_
