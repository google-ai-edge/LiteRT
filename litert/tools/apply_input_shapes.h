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

// Reusable model-level input-shape override logic, extracted from
// apply_input_shapes_main.cc so that apply_plugin can call it inline without
// an intermediate .tflite file.
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_TOOLS_APPLY_INPUT_SHAPES_H_
#define ODML_LITERT_LITERT_TOOLS_APPLY_INPUT_SHAPES_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::tools {

using ::litert::internal::Dims;

namespace internal {

Expected<Dims> ParseShape(absl::string_view shape_str);

struct NameAndShape {
  std::string name;
  std::string shape_str;
};

Expected<NameAndShape> ParseNameAndShape(absl::string_view input);

Expected<void> UpdateTensorType(LiteRtTensor tensor, const Dims& shape);

}  // namespace internal

// Apply input shape overrides to a model in memory, then run shape inference.
//
// model                - The in-memory model to mutate.
// signature_key        - Signature to resolve tensors through (empty = first).
// positional_inputs    - Shapes by position, e.g. {"1:224:224:3", "1:10"}.
//                        Count must match the number of model inputs exactly.
// name_inputs          - Shapes by tensor name, e.g. {"arg0@1:224:224:3"}.
//
// At most one of positional_inputs or name_inputs may be non-empty.
Expected<void> ApplyInputShapes(
    LiteRtModelT* model, const std::string& signature_key,
    const std::vector<std::string>& positional_inputs,
    const std::vector<std::string>& name_inputs);

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_TOOLS_APPLY_INPUT_SHAPES_H_
