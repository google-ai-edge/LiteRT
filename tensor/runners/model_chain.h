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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_MODEL_CHAIN_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_MODEL_CHAIN_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "tensor/runners/litert/litert_buffer.h"

namespace litert::tensor {

// Configuration descriptor for allocating or validating a stage's tensor
// buffer.
struct HardwareBufferDescriptor {
  std::vector<int32_t> shape;
  litert::ElementType element_type = litert::ElementType::Float32;
  size_t size_bytes = 0;
  litert::TensorBufferType buffer_type = litert::TensorBufferType::kHostMemory;
  size_t alignment = 64;
  bool gpu_readable = true;
  bool gpu_writable = true;
  bool npu_accessible = true;
  bool cpu_accessible = true;

  litert::RankedTensorType ToRankedTensorType() const {
    litert::Layout layout(litert::Dimensions(shape.begin(), shape.end()));
    return litert::RankedTensorType(element_type, std::move(layout));
  }

  size_t PackedBytes() const {
    if (size_bytes > 0) return size_bytes;
    auto ranked = ToRankedTensorType();
    auto bytes = ranked.Bytes();
    if (bytes.HasValue()) return *bytes;
    return 0;
  }
};

// Represents an abstract model execution stage in a ModelChain.
class ModelStage {
 public:
  virtual ~ModelStage() = default;

  virtual absl::string_view Name() const = 0;

  virtual std::vector<std::string> InputNames() const = 0;
  virtual std::vector<std::string> OutputNames() const = 0;

  virtual absl::StatusOr<HardwareBufferDescriptor> GetInputDescriptor(
      absl::string_view name) const = 0;
  virtual absl::StatusOr<HardwareBufferDescriptor> GetOutputDescriptor(
      absl::string_view name) const = 0;

  virtual absl::Status SetInputBuffer(
      absl::string_view name, std::shared_ptr<LitertBuffer> buffer) = 0;
  virtual absl::Status SetOutputBuffer(
      absl::string_view name, std::shared_ptr<LitertBuffer> buffer) = 0;

  virtual std::shared_ptr<LitertBuffer> GetInputBuffer(
      absl::string_view name) const = 0;
  virtual std::shared_ptr<LitertBuffer> GetOutputBuffer(
      absl::string_view name) const = 0;

  virtual void SetEnvironment(std::shared_ptr<litert::Environment> env) {}

  virtual absl::Status PrepareStageBoundary(
      const absl::flat_hash_map<std::string, HardwareBufferDescriptor>&
          negotiated_inputs,
      const absl::flat_hash_map<std::string, HardwareBufferDescriptor>&
          negotiated_outputs) {
    return absl::OkStatus();
  }

  virtual absl::Status Run() = 0;
};

// Flexible stage implementation driven by a user-provided execution function.
class FunctionalModelStage : public ModelStage {
 public:
  using BufferMap =
      absl::flat_hash_map<std::string, std::shared_ptr<LitertBuffer>>;
  using ExecuteFn = std::function<absl::Status(const BufferMap& inputs,
                                               const BufferMap& outputs)>;

  FunctionalModelStage(
      std::string name,
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>
          input_descriptors,
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>
          output_descriptors,
      ExecuteFn execute_fn);

  absl::string_view Name() const override { return name_; }

  std::vector<std::string> InputNames() const override;
  std::vector<std::string> OutputNames() const override;

  absl::StatusOr<HardwareBufferDescriptor> GetInputDescriptor(
      absl::string_view name) const override;
  absl::StatusOr<HardwareBufferDescriptor> GetOutputDescriptor(
      absl::string_view name) const override;

  absl::Status SetInputBuffer(
      absl::string_view name, std::shared_ptr<LitertBuffer> buffer) override;
  absl::Status SetOutputBuffer(
      absl::string_view name, std::shared_ptr<LitertBuffer> buffer) override;

  std::shared_ptr<LitertBuffer> GetInputBuffer(
      absl::string_view name) const override;
  std::shared_ptr<LitertBuffer> GetOutputBuffer(
      absl::string_view name) const override;

  void SetEnvironment(std::shared_ptr<litert::Environment> env) override {
    env_ = std::move(env);
  }

  absl::Status Run() override;

 private:
  std::shared_ptr<litert::Environment> env_;
  std::string name_;
  absl::flat_hash_map<std::string, HardwareBufferDescriptor>
      input_descriptors_;
  absl::flat_hash_map<std::string, HardwareBufferDescriptor>
      output_descriptors_;
  absl::flat_hash_map<std::string, std::shared_ptr<LitertBuffer>>
      input_buffers_;
  absl::flat_hash_map<std::string, std::shared_ptr<LitertBuffer>>
      output_buffers_;
  ExecuteFn execute_fn_;
};

// First-class model stage executing a compiled LiteRT model (.tflite).
// Automatically discovers input and output tensor descriptors from model
// signatures and directly registers shared LitertBuffers for zero-copy
// execution without memcpy.
class CompiledModelStage : public ModelStage {
 public:
  static absl::StatusOr<std::shared_ptr<CompiledModelStage>> Create(
      std::shared_ptr<litert::Environment> env, std::string name,
      const std::string& model_path, litert::Options options,
      size_t signature_index = 0);

  static absl::StatusOr<std::shared_ptr<CompiledModelStage>> Create(
      std::shared_ptr<litert::Environment> env, std::string name,
      const std::string& model_path,
      litert::HwAccelerators accelerators = litert::HwAccelerators::kCpu,
      size_t signature_index = 0);

  static absl::StatusOr<std::shared_ptr<CompiledModelStage>> Create(
      std::shared_ptr<litert::Environment> env, std::string name,
      absl::Span<const uint8_t> model_buffer, litert::Options options,
      size_t signature_index = 0);

  static absl::StatusOr<std::shared_ptr<CompiledModelStage>> Create(
      std::shared_ptr<litert::Environment> env, std::string name,
      CompiledModel compiled_model, size_t signature_index = 0);

  absl::string_view Name() const override { return name_; }

  std::vector<std::string> InputNames() const override;
  std::vector<std::string> OutputNames() const override;

  absl::StatusOr<HardwareBufferDescriptor> GetInputDescriptor(
      absl::string_view name) const override;
  absl::StatusOr<HardwareBufferDescriptor> GetOutputDescriptor(
      absl::string_view name) const override;

  absl::Status SetInputBuffer(
      absl::string_view name, std::shared_ptr<LitertBuffer> buffer) override;
  absl::Status SetOutputBuffer(
      absl::string_view name, std::shared_ptr<LitertBuffer> buffer) override;

  std::shared_ptr<LitertBuffer> GetInputBuffer(
      absl::string_view name) const override;
  std::shared_ptr<LitertBuffer> GetOutputBuffer(
      absl::string_view name) const override;

  void SetEnvironment(std::shared_ptr<litert::Environment> env) override {
    env_ = std::move(env);
  }

  absl::Status PrepareStageBoundary(
      const absl::flat_hash_map<std::string, HardwareBufferDescriptor>&
          negotiated_inputs,
      const absl::flat_hash_map<std::string, HardwareBufferDescriptor>&
          negotiated_outputs) override;

  CompiledModel& compiled_model() { return compiled_model_; }
  const CompiledModel& compiled_model() const { return compiled_model_; }
  size_t signature_index() const { return signature_index_; }

  absl::Status Run() override;

 private:
  CompiledModelStage(std::shared_ptr<litert::Environment> env, std::string name,
                     CompiledModel compiled_model, size_t signature_index,
                     std::vector<std::string> input_names,
                     std::vector<std::string> output_names,
                     absl::flat_hash_map<std::string, HardwareBufferDescriptor>
                         input_descriptors,
                     absl::flat_hash_map<std::string, HardwareBufferDescriptor>
                         output_descriptors,
                     std::string model_path = "",
                     std::vector<uint8_t> model_buffer = {},
                     std::optional<litert::Options> options = std::nullopt);

  absl::Status RefreshDescriptorsFromCompiledModel();

  std::shared_ptr<litert::Environment> env_;
  std::string name_;
  CompiledModel compiled_model_;
  size_t signature_index_ = 0;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  absl::flat_hash_map<std::string, HardwareBufferDescriptor>
      input_descriptors_;
  absl::flat_hash_map<std::string, HardwareBufferDescriptor>
      output_descriptors_;
  absl::flat_hash_map<std::string, std::shared_ptr<LitertBuffer>>
      input_buffers_;
  absl::flat_hash_map<std::string, std::shared_ptr<LitertBuffer>>
      output_buffers_;
  std::string model_path_;
  std::vector<uint8_t> model_buffer_;
  std::optional<litert::Options> options_;
};


// Negotiates and harmonizes data layouts, alignments, and flags across stages.
class BoundaryLayoutNegotiator {
 public:
  // Inspects upstream producer and downstream consumer descriptors and returns
  // a unified descriptor suitable for allocating a shared LiteRT buffer.
  static absl::StatusOr<HardwareBufferDescriptor> HarmonizeStageBoundary(
      const HardwareBufferDescriptor& producer_desc,
      const HardwareBufferDescriptor& consumer_desc);
};

// Represents an orchestrated pipeline of model execution stages where
// activations flow continuously across accelerators via LitertBuffer.
class ModelChain {
 public:
  struct Connection {
    std::string from_stage;
    std::string output_name;
    std::string to_stage;
    std::string input_name;
  };

  class Builder {
   public:
    Builder();
    ~Builder();

    // Sets a custom LiteRT environment for buffer allocation.
    Builder& WithEnvironment(std::shared_ptr<litert::Environment> env);
    Builder& WithEnvironment(litert::Environment env);

    // Adds a stage to the pipeline.
    Builder& AddStage(std::shared_ptr<ModelStage> stage);

    // Connects an output of from_stage to an input of to_stage.
    Builder& Connect(absl::string_view from_stage,
                     absl::string_view output_name,
                     absl::string_view to_stage, absl::string_view input_name);

    // Validates pipeline DAG, negotiates stage boundaries, pre-allocates
    // intermediate LitertBuffers, binds them to respective stages,
    // and constructs ModelChain.
    absl::StatusOr<ModelChain> Build();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
  };

  ~ModelChain();
  ModelChain(ModelChain&&) noexcept;
  ModelChain& operator=(ModelChain&&) noexcept;

  // Executes all stages in topological order with zero CPU intermediate copies.
  absl::Status Execute();

  // Binds an entry input buffer to a specific stage and port.
  absl::Status SetInputBuffer(absl::string_view stage_name,
                              absl::string_view input_name,
                              std::shared_ptr<LitertBuffer> buffer);

  // Binds an entry input buffer by input name (if name is unique).
  absl::Status SetInputBuffer(absl::string_view input_name,
                              std::shared_ptr<LitertBuffer> buffer);

  // Retrieves an output buffer from a specific stage and port.
  absl::StatusOr<std::shared_ptr<LitertBuffer>> GetOutputBuffer(
      absl::string_view stage_name, absl::string_view output_name) const;

  // Retrieves an output buffer by name (if name is unique).
  absl::StatusOr<std::shared_ptr<LitertBuffer>> GetOutputBuffer(
      absl::string_view output_name) const;

  // Retrieves a stage by name.
  absl::StatusOr<std::shared_ptr<ModelStage>> GetStage(
      absl::string_view name) const;

  // Returns all intermediate shared hardware buffers pre-allocated for chain.
  const std::vector<std::shared_ptr<LitertBuffer>>&
  GetIntermediateBuffers() const;

 private:
  struct Impl;
  explicit ModelChain(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_MODEL_CHAIN_H_
