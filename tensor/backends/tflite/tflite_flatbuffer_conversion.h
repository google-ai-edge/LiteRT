/* Copyright 2025 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TFLITE_TFLITE_FLATBUFFER_CONVERSION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TFLITE_TFLITE_FLATBUFFER_CONVERSION_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/linked_flat_hash_map.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tflite/c/c_api_types.h"
#include "tflite/schema/mutable/schema_generated.h"

namespace litert::tensor {

absl::StatusOr<Type> FromTfLite(TfLiteType type);

class ModelFactory {
 public:
  // Information about an external buffer in a group.
  struct ExternalBufferInfo {
    uint64_t offset;
    uint64_t length;
    const TensorHandle& tensor;
  };
  using ExternalBufferMap =
      LinkedFlatHashMap<std::string, std::vector<ExternalBufferInfo>>;

  ModelFactory();

  // Saves the model to `path`.
  absl::Status Save(absl::string_view path);

  // Creates an interpreter from the model.
  absl::StatusOr<std::vector<char>> CreateFlatbuffer();

  // Adds a subgraph to the model.
  absl::Status AddSubgraph(std::vector<TensorHandle> outputs);

  template <class... Mixins>
  absl::Status AddSubgraph(std::vector<Tensor<Mixins...>> outputs) {
    return AddSubgraph(
        std::vector<TensorHandle>(outputs.begin(), outputs.end()));
  }

  // Adds a new signature to the model.
  //
  // The signature inputs and outputs are named using the tensor names.
  absl::Status AddSignature(std::vector<TensorHandle> outputs,
                            std::string name);

  // Registers a set of external buffers. It must be called before
  // AddSubgraph().
  absl::Status AddExternalBufferMap(
      const ExternalBufferMap& external_buffer_map);

 protected:
  // Explores a new subgraph that is reachable from the given output tensors.
  absl::Status Explore(std::vector<TensorHandle> outputs);

  // Adds the previously explored subgraph to the flatbuffer object
  // representation.
  absl::Status Build();

  // Adds a subgraph while preserving any currently explored parent subgraph.
  absl::StatusOr<int> AddSubgraphAndReturnIndex(
      std::vector<TensorHandle> outputs);

  // Updates the FINISHED flatbuffer builder TFLite buffer data with the
  // corresponding sizes and offsets.
  absl::Status UpdateBufferData(flatbuffers::FlatBufferBuilder& fbb);

  absl::Status WriteBufferData(std::ofstream& output_file);

 private:
  struct TensorSerializationInfo {
    int index = -1;
    bool is_output = false;
    std::optional<uint32_t> external_buffer_id;
  };

  struct OpSerializationInfo {};

  struct BufferSerializationInfo {
    int index = -1;
    size_t serialization_offset = 0;
    std::optional<uint32_t> external_buffer_id;
  };

  std::optional<uint32_t> GetExternalBufferId(
      const graph::Tensor& tensor) const;

  LinkedFlatHashMap<graph::Tensor, TensorSerializationInfo> tensors_;
  LinkedFlatHashMap<std::shared_ptr<graph::Operation>, OpSerializationInfo>
      operations_;
  std::vector<const graph::Operation*> execution_plan_;
  LinkedFlatHashMap<std::shared_ptr<Buffer>, BufferSerializationInfo> buffers_;
  std::deque<std::shared_ptr<Buffer>> buffer_list_;
  tflite::ModelT model_;
  size_t allocation_size_;
  absl::flat_hash_map<graph::Tensor, uint32_t> tensor_to_external_buffer_id_;
};

// Creates a flatbuffer from the given outputs and saves it to the given path.
//
// This function is a shorthand to create a graph with a unique subgraph and
// save it.
//
// Note: For more complex graphs, use the `ModelFactory` class.
absl::Status Save(std::vector<TensorHandle> outputs, absl::string_view path);

// Creates a flatbuffer from the given outputs and saves it to the given path.
//
// This function is a shorthand to create a graph with a unique subgraph and
// save it.
//
// Note: For more complex graphs, use the `ModelFactory` class.
template <class... Mixins>
absl::Status Save(std::vector<Tensor<Mixins...>> outputs,
                  absl::string_view path) {
  std::vector<TensorHandle> erased_outputs(outputs.begin(), outputs.end());
  return Save(std::move(erased_outputs), path);
}

// Creates a flatbuffer from the given outputs.
//
// This function is a shorthand to create a graph with a unique subgraph.
//
// Note: For more complex graphs, use the `ModelFactory` class.
absl::Status Save(std::vector<TensorHandle> outputs, std::vector<char>& fb);

// Creates a flatbuffer from the given outputs.
//
// This function is a shorthand to create a graph with a unique subgraph.
//
// Note: For more complex graphs, use the `ModelFactory` class.
template <class... Mixins>
absl::Status Save(std::vector<Tensor<Mixins...>> outputs,
                  std::vector<char>& fb) {
  std::vector<TensorHandle> erased_outputs(outputs.begin(), outputs.end());
  return Save(std::move(erased_outputs), fb);
}

// Runs this primary graph that is stored in the given model factory.
absl::Status Run(ModelFactory& model_factory);

// Runs a constant model from the given outputs.
//
// This function is a shorthand to create a model with a unique subgraph, save
// it to a flatbuffer and run it.
//
// Note: Because the function doesn't let you specify the inputs tensors other
// than through the graph definition, all the tensor data must be specified when
// building the graph.
//
// If you need a more complex setup, you need to setup the TFLite interpreter
// yourself.
absl::Status Run(std::vector<TensorHandle> outputs);

// Runs a constant model from the given outputs.
//
// This function is a shorthand to save a model to a flatbuffer and run its
// primary subgraph.
template <class... Mixins>
absl::Status Run(std::vector<Tensor<Mixins...>> outputs) {
  return Run(std::vector<TensorHandle>(outputs.begin(), outputs.end()));
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TFLITE_TFLITE_FLATBUFFER_CONVERSION_H_
