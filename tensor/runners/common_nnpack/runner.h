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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMMON_NNPACK_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMMON_NNPACK_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

class NnpackRunner {
 public:
  explicit NnpackRunner(std::unique_ptr<NnpackGraph> graph)
      : graph_(std::move(graph)) {}

  virtual ~NnpackRunner() = default;

  NnpackRunner(NnpackRunner&&) = default;
  NnpackRunner& operator=(NnpackRunner&&) = default;
  NnpackRunner(const NnpackRunner&) = delete;
  NnpackRunner& operator=(const NnpackRunner&) = delete;

  virtual void SetNumThreads(size_t num_threads) { num_threads_ = num_threads; }

  // Sets the input data corresponding to `tensor` from `external_tensor`.
  //
  // The shape and buffer of `external_tensor` are propagated to `tensor`. The
  // types of both tensors must match.
  absl::Status SetInput(const TensorHandle& tensor,
                        const TensorHandle& external_tensor);

  // Sets the input data corresponding to `tensor`.
  //
  // If `copy_data` is `true` then the `data` is copied into a new buffer owned
  // by the runtime. Otherwise, a non-owning view is kept.
  absl::Status SetInput(const TensorHandle& tensor,
                        absl::Span<const std::byte> data,
                        bool copy_data = false);

  // Sets the input data corresponding to `tensor`.
  //
  // If `copy_data` is `true` then the `data` is copied into a new buffer owned
  // by the runtime. Otherwise, a non-owning view is kept.
  absl::Status SetInput(const TensorHandle& tensor, absl::Span<std::byte> data,
                        const bool copy_data = false) {
    return SetInput(tensor, absl::Span<const std::byte>(data), copy_data);
  }

  // Sets the input data corresponding to `tensor`.
  //
  // Warning: This **always** keeps a view over the data.
  template <class ContiguousSequence,
            class S = std::remove_reference_t<ContiguousSequence>,
            class T = typename S::value_type,
            class SFINAE = decltype(std::declval<S>().data())>
  absl::Status SetInput(const TensorHandle& tensor,
                        const ContiguousSequence& seq) {
    if (tensor.GetType() != ApiType<T>::value) {
      return absl::InvalidArgumentError(
          "The sequence type doesn't match the input tensor type.");
    }
    return SetInput(tensor,
                    absl::Span<const std::byte>(
                        reinterpret_cast<const std::byte*>(seq.data()),
                        seq.size() * sizeof(T)),
                    /*copy_data=*/false);
  }

  // Deleted overload to avoid keeping a view over a dangling sequence.
  template <class ContiguousSequence,
            class S = std::remove_reference_t<ContiguousSequence>,
            class T = typename S::value_type,
            class SFINAE = decltype(std::declval<S>().data())>
  absl::Status SetInput(const TensorHandle& tensor,
                        const ContiguousSequence&& seq) = delete;

  // Sets the input data corresponding to `tensor` by copying `seq`.
  //
  // The sequence is unconditionally copied.
  template <class ContiguousSequence,
            class S = std::remove_reference_t<ContiguousSequence>,
            class T = typename S::value_type,
            class SFINAE = decltype(std::declval<S>().data())>
  absl::Status SetInputAsCopy(const TensorHandle& tensor,
                              ContiguousSequence&& seq) {
    if (tensor.GetType() != ApiType<T>::value) {
      return absl::InvalidArgumentError(
          "The sequence type doesn't match the input tensor type.");
    }
    return SetInput(tensor,
                    absl::Span<const std::byte>(
                        reinterpret_cast<const std::byte*>(seq.data()),
                        seq.size() * sizeof(T)),
                    /*copy_data=*/true);
  }

  // Sets the output data corresponding to `tensor`.
  //
  // Note: the given `tensor` needs to be an output tensor of the graph and the
  // `data` size must match the `tensor` shape.
  //
  // Warning: the `data` is **never** copied and is only stored as a view.
  absl::Status SetOutput(const TensorHandle& tensor,
                         absl::Span<std::byte> data);

  // Reshapes the input corresponding to `tensor`.
  //
  // Warning: `tensor` is only used to identify the data that needs to be
  // updated.
  absl::Status ReshapeInput(const TensorHandle& tensor,
                            absl::Span<const int32_t> shape);

  // Writes a sub-span of the input buffer.
  absl::Status WriteInput(const TensorHandle& tensor, size_t offset_bytes,
                          absl::Span<const std::byte> data);

  // Writes a sub-span of the input buffer.
  template <typename T>
  absl::Status WriteInput(const TensorHandle& tensor, size_t offset_bytes,
                          absl::Span<const T> data) {
    return WriteInput(tensor, offset_bytes,
                      absl::Span<const std::byte>(
                          reinterpret_cast<const std::byte*>(data.data()),
                          data.size() * sizeof(T)));
  }

  // Writes a sub-span of the input buffer.
  template <typename Sequence>
  absl::Status WriteInput(const TensorHandle& tensor, size_t offset_bytes,
                          const Sequence& seq) {
    using T = typename Sequence::value_type;
    return WriteInput(tensor, offset_bytes,
                      absl::Span<const std::byte>(
                          reinterpret_cast<const std::byte*>(seq.data()),
                          seq.size() * sizeof(T)));
  }

  // Prepares the runtime for invocation.
  //
  // Note: this is lazily called by `Run()` and should only be called if the
  // runtime preparation needs to be done in advance.
  absl::Status PrepareRuntime();

  // Runs the runtime.
  //
  // This lazily prepares the runtime and invokes it on the set inputs.
  absl::Status Run();

  // Returns a span holding the output data corresponding to `tensor`.
  absl::StatusOr<LockedBufferSpan<const std::byte>> ReadOutput(
      const TensorHandle& tensor) const;

  // Returns a span holding the output data corresponding to `tensor`.
  template <class T>
  absl::StatusOr<LockedBufferSpan<const T>> ReadOutputAs(
      const TensorHandle& tensor) {
    if (tensor.GetType() != ApiType<T>::value) {
      return absl::InvalidArgumentError(
          "The read type doesn't match the output tensor type.");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(LockedBufferSpan<const std::byte> out,
                                ReadOutput(tensor));
    return std::move(out).template As<const T>();
  }

  const NnpackGraph& graph() const { return *graph_; }
  NnpackGraph& mutable_graph() { return *graph_; }
  const absl::flat_hash_map<uint32_t, std::shared_ptr<Buffer>>&
  external_buffers() const {
    return external_buffers_;
  }
  absl::flat_hash_map<uint32_t, std::shared_ptr<Buffer>>&
  mutable_external_buffers() {
    return external_buffers_;
  }

 protected:
  virtual uint32_t FlagExternalInput() const = 0;
  virtual uint32_t FlagExternalOutput() const = 0;

  virtual absl::Status CreateRuntime(size_t num_threads) = 0;
  virtual absl::Status SetExternalValueShape(uint32_t id,
                                             absl::Span<const size_t> dims) = 0;
  virtual absl::Status ReshapeRuntime() = 0;
  virtual absl::Status GetExternalValueShape(uint32_t id,
                                             std::vector<size_t>& dims) = 0;
  virtual absl::Status SetupExternalValues(
      absl::Span<NnpackValue> values,
      const absl::flat_hash_map<uint32_t, std::shared_ptr<Buffer>>&
          external_buffers,
      std::vector<LockedBufferSpan<const std::byte>>& locks) = 0;
  virtual absl::Status InvokeRuntime() = 0;

  std::unique_ptr<NnpackGraph> graph_;
  // Buffers that need to be kept alive to execute the runtime.
  absl::flat_hash_map<uint32_t, std::shared_ptr<Buffer>> external_buffers_;
  size_t num_threads_ = 1;
  bool runtime_prepared_ = false;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMMON_NNPACK_RUNNER_H_
