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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_TEST_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_TEST_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "litert/cc/litert_buffer_ref.h"
#include "litert/vendors/nvidia/tools/carve_model.h"
#include "tflite/schema/schema_generated.h"

namespace litert::tools::testing {

// A small model with factorized feed-forwards: a chain of factorized MLPs, each
// between an F16 residual stream and an ADD, with centered INT2 weights
// (blockwise quantization with a block shape) and INT8 activations. Signature
// "main": x F32 [1, tokens, features] -> y of the same type.
struct FactorizedMlpModelOptions {
  int tokens = 3;
  // The feature width is 2 * half_features.
  int half_features = 8;
  int groups = 2;
  // The width of one group after one up projection; an up projection has
  // groups * group_features outputs.
  int group_features = 4;
  // The hidden width of one group.
  int group_hidden = 4;
  int num_mlps = 2;
  // Give the block-diagonal weights a leading dimension of 1, as a decode graph
  // may have them.
  bool rank4_block_weights = false;
  // Keep the constants behind the flatbuffer (Buffer.offset and size), the
  // way models with large weights are written, instead of inside it.
  bool appended_buffers = false;
  // With appended_buffers, the constants of at most this many bytes stay
  // inside the flatbuffer: a model with small constants inside and weights
  // behind.
  size_t keep_inline_up_to = 0;
  // Changes the model before it is serialized.
  std::function<void(tflite::ModelT&)> mutate;
};

// The serialized model.
std::string BuildFactorizedMlpModel(const FactorizedMlpModelOptions& options);

inline litert::BufferRef<uint8_t> AsBuffer(const std::string& bytes) {
  return litert::BufferRef<uint8_t>(bytes.data(), bytes.size());
}

// The bytes of a carved model as a file holds them.
std::string ToBytes(const CarvedModel& model);

// The model in `bytes` after `mutate` changed it. The data the model keeps
// behind its flatbuffer stays in place.
std::string Mutated(const std::string& bytes,
                    const std::function<void(tflite::ModelT&)>& mutate);

}  // namespace litert::tools::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_TEST_UTIL_H_
