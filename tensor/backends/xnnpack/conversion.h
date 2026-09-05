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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/nnpack_common/conversion.h"
#include "tensor/backends/xnnpack/arithmetic.h"  // IWYU pragma: export
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
namespace litert::tensor {

using XnnpackGraph = NnpackGraph<XnnpackTraits>;

// Builds an XNNPACK graph from the given outputs.
inline absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
    std::vector<TensorHandle> outputs) {
  return BuildNnpackGraph<XnnpackTraits>(std::move(outputs));
}

// Lowers the implementation graph of an operation to XNNPACK.
inline absl::Status InlineImplementationGraphFor(
    const graph::Operation& op, absl::Span<const graph::Tensor> inlined_inputs,
    absl::Span<const graph::Tensor> inlined_outputs, XnnpackBuildContext& ctx) {
  return InlineImplementationGraphFor<XnnpackTraits>(op, inlined_inputs,
                                                     inlined_outputs, ctx);
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_
