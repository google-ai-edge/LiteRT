/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_MIXIN_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_MIXIN_H_

namespace litert::tensor {

template <class Mixin>
class TensorMixin {};

namespace graph {
// Provides custom behaviour to operations.
//
// - Op is the operation that is being specialized.
// - Mixin is a tag to identify the mix-in.
template <class Op, class Mixin>
class OpMixin {};

}  // namespace graph
}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_MIXIN_H_
