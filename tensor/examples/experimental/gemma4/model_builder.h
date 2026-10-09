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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_MODEL_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_MODEL_BUILDER_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/examples/experimental/gemma4/bundle_loader.h"
#include "tensor/examples/experimental/gemma4/kv_cache.h"
#include "tensor/examples/gemma4/gemma4_config.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Configuration for authoring the Gemma4 prefill and decode TFLite signatures.
struct ModelAuthoringOptions {
  // Authoring-time query row count for the `prefill` signature; must be a power
  // of two in [32, 1024] when calling the published E2B `AddGemma4Signatures`
  // overload, or positive when supplying an explicit `Config`.
  int prefill_rows = 128;
  // If true, author dynamic leading-dimension slices/reshapes in `prefill` so
  // the host may resize the query row dimension at runtime.
  bool dynamic_prefill_rows = false;
  // If true, emit explicit `-1` TFLite shape signatures on resizable KV, mask,
  // and (when `dynamic_prefill_rows` is true) prefill-row boundaries so callers
  // can use strict resizing.
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
                                 absl::Span<const KvOwnerSpec> specs,
                                 const ModelAuthoringOptions& options = {});

// Adds `prefill` and `decode` signatures for a caller-supplied Gemma4 `config`.
// Callers must supply compatible `weights`, KV owner `specs`, and a positive
// `options.prefill_rows`.
absl::Status AddGemma4Signatures(ModelFactory& factory,
                                 const examples::gemma4::Config& config,
                                 const LoadedTensors& weights,
                                 absl::Span<const KvOwnerSpec> specs,
                                 const ModelAuthoringOptions& options = {});

}  // namespace litert::tensor::examples::experimental::gemma4::cpu

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_MODEL_BUILDER_H_
