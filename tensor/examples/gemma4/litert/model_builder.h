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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_MODEL_BUILDER_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_MODEL_BUILDER_H_

#include <vector>

#include "absl/status/status.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/examples/gemma4/litert/bundle_loader.h"
#include "tensor/examples/gemma4/litert/kv_cache.h"

namespace litert::tensor::examples::gemma4::cpu {

struct ModelAuthoringOptions {
  int prefill_rows = 128;
  bool dynamic_prefill_rows = false;
  bool explicit_dynamic_shapes = false;
};

// Adds the published Gemma4 E2B prefill and decode graphs to a fresh factory.
// Load all weights with LoadPublishedBundle(..., embeddings_only=false) and
// obtain specs from LoadPublishedKvSpecs. Keep the loaded tensors alive until
// the caller finishes ModelFactory::Save or CreateFlatbuffer.
//
// Prefill exposes only new KV for the 15 unique owners. Decode exposes logits
// and new KV. The caller supplies host embeddings, active history, alignment
// padding, positions and masks, following Runner's input/output contract.
absl::Status AddGemma4Signatures(ModelFactory& factory,
                                 const LoadedTensors& weights,
                                 const std::vector<KvOwnerSpec>& specs,
                                 const ModelAuthoringOptions& options = {});

}  // namespace litert::tensor::examples::gemma4::cpu

#endif  // LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_MODEL_BUILDER_H_
