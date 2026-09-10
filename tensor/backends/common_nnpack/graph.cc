/* Copyright 2026 Google LLC.

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

#include "tensor/backends/common_nnpack/graph.h"

#include <cstddef>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "tensor/tensor.h"

namespace litert::tensor {

absl::StatusOr<size_t> NnpackGraph::Lookup(const TensorHandle& tensor) const {
  auto it = tensor_index_.find(tensor.GetRaw());
  if (it == tensor_index_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Tensor %s not found in graph.", tensor.GetName()));
  }
  return it->second;
}

}  // namespace litert::tensor
