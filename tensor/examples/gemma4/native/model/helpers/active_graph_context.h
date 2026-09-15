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

// Local experiment: explicit graph cuts for ordered persistent INT8 appends.
#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_ACTIVE_GRAPH_CONTEXT_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_ACTIVE_GRAPH_CONTEXT_H_
#include <vector>
#include "tensor/tensor.h"
namespace litert::tensor::examples::gemma4::native {
struct ActiveLayerGraph {
  int owner = -1;
  int dim = 0;
  bool global = false;
  TensorHandle hidden_input, ple_input;
  TensorHandle query, new_key, new_value, context_input, output;
};
struct ActiveGraphContext {
  int layer = 0;
  std::vector<int> owners;
  std::vector<ActiveLayerGraph> layers;
  TensorHandle initial_hidden;
  std::vector<TensorHandle> ple_outputs;
  TensorHandle final_hidden;
};
// Scoped only during graph authoring; execution owns all recorded handles.
inline thread_local ActiveGraphContext* active_graph_context = nullptr;
}
#endif
