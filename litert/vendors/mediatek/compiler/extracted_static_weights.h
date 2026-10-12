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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_EXTRACTED_STATIC_WEIGHTS_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_EXTRACTED_STATIC_WEIGHTS_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"

// Helpers for compiling a network with its static weights (constant tensors)
// extracted to a side file instead of embedded in the compiled network. The
// compiled network then takes each extracted weight buffer as an extra input
// after its regular inputs.

namespace litert::mediatek {

// Returns the Neuron compiler options that make it extract the static weights
// of the compiled network to the file at `path`. Fails if `path` is empty or
// contains whitespace, since compiler options are whitespace-separated.
Expected<std::string> StaticWeightExtractionOptions(absl::string_view path);

// Creates a new, empty file with a unique name derived from `name` for the
// Neuron compiler to extract static weights to, and returns its path. The file
// is created in `$MTKNN_ADAPTER_DLA_DIR` if that is set, and in the system
// temporary directory otherwise. The caller must delete the file.
Expected<std::string> CreateExtractedStaticWeightsFile(absl::string_view name);

// Parses `contents`, the contents of a file that the Neuron compiler extracted
// static weights to, and returns the weights in the order in which the compiled
// network expects them as inputs. The returned views point into `contents`.
// Empty `contents` mean that the compiler extracted no weights.
//
// The file consists of the weight payload, a JSON object that maps the input
// order of each weight to its `["offset", "size"]` in the payload (all values
// are quoted decimal integers), and a 12-byte footer: the little-endian
// `uint64` offset of the JSON object followed by the little-endian `uint32`
// magic number `0xbbced1ec`.
Expected<std::vector<absl::string_view>> ParseExtractedStaticWeights(
    absl::string_view contents);

}  // namespace litert::mediatek

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_EXTRACTED_STATIC_WEIGHTS_H_
