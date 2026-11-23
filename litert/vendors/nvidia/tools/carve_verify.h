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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_VERIFY_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_VERIFY_H_

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"

// The checks the carve tool runs on what it wrote.
namespace litert::tools {

// Checks that the model `carved` computes what the model `original` computes,
// from the two files alone: with the decomposition of every composite that
// `carved` has in addition put back in the composite's place, each subgraph of
// `original` must be the same graph in `carved`, down to the options of each
// operator, the type, shape and quantization of each tensor (with the scale
// and zero point tensors that blockwise quantization names, which must be
// constants of the subgraph that uses them) and the bytes of each constant.
// Names of tensors inside a graph are not compared. It also checks that every
// tensor is written before it is read and once, that the signatures, the
// metadata and the operator codes are as they were, that each new subgraph is
// the decomposition of exactly one composite, and that no tensor is left
// behind unused.
//
// Returns the first difference; `summary` says what was compared.
absl::Status VerifyCarvedStructure(litert::BufferRef<uint8_t> original,
                                   litert::BufferRef<uint8_t> carved,
                                   std::string* summary);

struct NumericCheckOptions {
  // The signatures to run; all of them if empty.
  std::vector<std::string> signatures;
  int num_threads = 4;
};

// Runs both models on the CPU the way LiteRT's CPU accelerator does (the
// TFLite interpreter with composites it has no kernel for inlined, and
// XNNPACK) on the same pseudo-random inputs, and requires every output of
// every signature to be the same, bit for bit. Floating-point and quantized
// inputs are random, integer inputs (positions, indices) are 0 and boolean
// inputs (masks) are true, so that the inputs are valid for any graph.
//
// A model whose constants live outside its own bytes (TFLite external
// buffers) cannot be run this way.
absl::Status CompareOnCpu(litert::BufferRef<uint8_t> original,
                          litert::BufferRef<uint8_t> carved,
                          const NumericCheckOptions& options,
                          std::string* summary);

}  // namespace litert::tools

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_VERIFY_H_
