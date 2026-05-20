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

#include "tensor/internal/shape.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace litert::tensor {

absl::StatusOr<std::vector<int>> BroadcastShapes(
    const std::vector<std::vector<int>>& shapes) {
  if (shapes.empty()) {
    return std::vector<int>();
  }

  size_t max_dims = 0;
  for (const auto& shape : shapes) {
    if (shape.size() > max_dims) {
      max_dims = shape.size();
    }
  }

  std::vector<int> result_shape(max_dims, 1);
  for (const auto& shape : shapes) {
    for (size_t i = 0; i < max_dims; ++i) {
      int dim1 = result_shape[i];
      int dim2 = (i < max_dims - shape.size())
                     ? 1
                     : shape[i - (max_dims - shape.size())];
      if (dim1 != 1 && dim2 != 1 && dim1 != dim2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Shapes cannot be broadcasted: ", dim1, " vs ", dim2));
      }
      result_shape[i] = std::max(dim1, dim2);
    }
  }

  return result_shape;
}

}  // namespace litert::tensor
