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

// Reads validated published Gemma4 E2B/E4B weight bundles (unpacked .litertlm
// bundle directories).
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_BUNDLE_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_BUNDLE_LOADER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/examples/experimental/gemma4/kv_cache.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Holds the memory-mapped model weight tensors and initialized embedding lookup
// tables loaded from an unpacked .litertlm Gemma4 bundle.
struct LoadedTensors {
  absl::flat_hash_map<std::string, TensorHandle> weights_handle;
  std::unique_ptr<examples::gemma4::GemmaEmbeddingTable> token_embedding;
  std::unique_ptr<examples::gemma4::GemmaEmbeddingTable> emb_per_layer_table;
};

// Expected data type name and shape for a tensor entry in an unpacked bundle.
struct ExpectedTensor {
  std::string dtype;
  Shape shape;
};

// Controls which tensors `LoadBundleTensors` loads from a bundle directory.
enum class BundleLoadScope {
  kAllTensors,
  kEmbeddingsOnly,
};

namespace published_bundle_detail {

using Expected = ExpectedTensor;

// Returns the complete map of expected tensor names to their expected dtype and
// shape for a published Gemma4 E2B or E4B model configuration.
absl::flat_hash_map<std::string, Expected> ExpectedTensors(
    const examples::gemma4::Config& c, bool compact_int2 = false);

}  // namespace published_bundle_detail

// Loads and validates the per-owner KV cache specifications from
// `manifest.json` inside an unpacked .litertlm bundle `directory`.
absl::StatusOr<std::vector<KvOwnerSpec>> LoadPublishedKvSpecs(
    absl::string_view directory, const examples::gemma4::Config& config,
    bool compact_int2 = false);

// Loads, validates, and memory-maps the model weights and embedding tables from
// an unpacked .litertlm bundle `directory`. When `embeddings_only` is true,
// only the token and per-layer embedding tables (plus the logits softcap
// reciprocal constant) are loaded.
absl::StatusOr<LoadedTensors> LoadPublishedBundle(
    absl::string_view directory, const examples::gemma4::Config& config,
    bool embeddings_only = false, bool compact_int2 = false);

// Loads, validates, and memory-maps the tensor files declared in
// `manifest.json` inside `directory` against `expected_tensors`, validates the
// required logits softcap reciprocal constant, and initializes the host
// embedding tables for `config`. Unlike `LoadPublishedBundle`, this entry point
// does not restrict `config` to full E2B/E4B dimensions, allowing callers to
// load compact or synthetic Gemma4 bundles with the same validation logic.
absl::StatusOr<LoadedTensors> LoadBundleTensors(
    absl::string_view directory, const examples::gemma4::Config& config,
    const absl::flat_hash_map<std::string, ExpectedTensor>& expected_tensors,
    BundleLoadScope scope = BundleLoadScope::kAllTensors);

}  // namespace litert::tensor::examples::experimental::gemma4::cpu

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_BUNDLE_LOADER_H_
