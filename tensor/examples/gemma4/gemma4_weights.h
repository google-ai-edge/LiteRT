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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_WEIGHTS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_WEIGHTS_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4 {

// Returns the mapping from HuggingFace Gemma 4 safetensor tensor names
// to model tensor names used by the Gemma 4 computation graph.
absl::flat_hash_map<std::string, std::string> GetGemma4WeightMapping(
    int n_layers);

// Mapping hooks that provide the Gemma 4 weights that checkpoints may omit.
//
// - "lm_head.weight" falls back to the tied embedding weights
//   ("model.embed_tokens.weight").
class Gemma4WeightHooks : public TensorMappingHooks {
 public:
  absl::StatusOr<TensorHandle> OnNotFound(
      TensorMapping& mapping, absl::string_view model_name) override;
};

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_WEIGHTS_H_
