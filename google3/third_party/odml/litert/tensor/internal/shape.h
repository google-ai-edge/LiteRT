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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_SHAPE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_SHAPE_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::tensor {

// Calculates the broadcasted shape from a vector of shapes.
absl::StatusOr<std::vector<int>> BroadcastShapes(
    const std::vector<std::vector<int>>& shapes);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_SHAPE_H_
