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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_MODEL_BUILDER_INTERNAL_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_MODEL_BUILDER_INTERNAL_H_

#include <vector>

#include "absl/status/status.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/examples/experimental/gemma4/bundle_loader.h"
#include "tensor/examples/experimental/gemma4/model_builder.h"
#include "tensor/examples/gemma4/gemma4_config.h"

namespace litert::tensor::examples::experimental::gemma4::cpu::
    model_builder_internal {

// Assembly shared by the validated E2B entry point and small-model tests.
// Callers must supply a compatible configuration, weights, and owner specs.
absl::Status AddSignatures(ModelFactory& factory,
                           const examples::gemma4::Config& config,
                           const LoadedTensors& weights,
                           const std::vector<KvOwnerSpec>& specs,
                           const ModelAuthoringOptions& options);

}  // namespace
   // litert::tensor::examples::experimental::gemma4::cpu::model_builder_internal

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_MODEL_BUILDER_INTERNAL_H_
