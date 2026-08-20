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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_XNNPACK_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_XNNPACK_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "pthreadpool.h"  // from @pthreadpool

struct xnn_runtime;

namespace litert::tensor {
class XnnpackRunnerTest_ConstantsAreNotBoundAsExternals_Test;

// XnnpackRunner is a class that runs an XNNPACK graph.
class XnnpackRunner {
  friend class XnnpackRunnerTest_ConstantsAreNotBoundAsExternals_Test;

 public:
  // Creates an XnnpackRunner from a list of output tensors.
  static absl::StatusOr<XnnpackRunner> Create(
      std::vector<TensorHandle> outputs);

  // Sets the number of threads.
  //
  // Warning: this must be called before the first call to `Run`.
  void SetNumThreads(size_t num_threads) { num_threads_ = num_threads; }

  // Sets the weights cache.
  //
  // Warning: this must be called before the first call to `Run`.
  void SetWeightsCache(xnn_weights_cache_t weights_cache) {
    weights_cache_ = weights_cache;
  }

  // Prepares the XNNPACK runtime (compiles the graph into operators and packs
  // weights).
  absl::Status PrepareRuntime();

  // Sets the input data for a given tensor.
  absl::Status SetInput(const TensorHandle& tensor,
                        absl::Span<const std::byte> data,
                        bool copy_data = false);

  absl::Status SetInput(const TensorHandle& tensor, absl::Span<std::byte> data,
                        bool copy_data = false) {
    return SetInput(tensor, absl::Span<const std::byte>(data), copy_data);
  }

  // Sets the input data for a given tensor.
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
    return SetInput(tensor, absl::Span<const std::byte>(
                                reinterpret_cast<const std::byte*>(seq.data()),
                                seq.size() * sizeof(T)));
  }

  template <class ContiguousSequence,
            class S = std::remove_reference_t<ContiguousSequence>,
            class T = typename S::value_type,
            class SFINAE = decltype(std::declval<S>().data())>
  absl::Status SetInput(const TensorHandle& tensor,
                        const ContiguousSequence&& seq) = delete;

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

  // Sets the output buffer for a given tensor.
  absl::Status SetOutput(const TensorHandle& tensor,
                         absl::Span<std::byte> data);

  // Updates the shape for an external input tensor.
  absl::Status ReshapeInput(const TensorHandle& tensor,
                            absl::Span<const int32_t> shape);

  // Writes a slice of bytes into an external input tensor's host buffer.
  absl::Status WriteInput(const TensorHandle& tensor, size_t offset_bytes,
                          absl::Span<const std::byte> data);

  // Sets the input data for a given tensor.
  template <class ContiguousSequence,
            class S = std::remove_reference_t<ContiguousSequence>,
            class T = typename S::value_type,
            class SFINAE = decltype(std::declval<S>().data())>
  absl::Status WriteInput(const TensorHandle& tensor, size_t offset_bytes,
                          ContiguousSequence&& seq) {
    return WriteInput(tensor, offset_bytes,
                      absl::Span<const std::byte>(
                          reinterpret_cast<const std::byte*>(seq.data()),
                          seq.size() * sizeof(T)));
  }

  // Runs the XNNPACK graph.
  absl::Status Run();

  // Reads the output data for a given tensor.
  absl::StatusOr<LockedBufferSpan<const std::byte>> ReadOutput(
      const TensorHandle& tensor) const;

  template <class T>
  absl::StatusOr<LockedBufferSpan<const T>> ReadOutputAs(
      const TensorHandle& tensor) {
    if (tensor.GetType() != ApiType<T>::value) {
      return absl::InvalidArgumentError(
          "The read type doesn't match the output tensor type.");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(LockedBufferSpan<const std::byte> out,
                                ReadOutput(tensor));
    return std::move(out).As<const T>();
  }

 private:
  explicit XnnpackRunner(std::unique_ptr<XnnpackGraph> graph);

  // External buffer that can be either an external view or an owned buffer.
  class ExternalBuffer {
   public:
    // Gets a span of either the external view or the owned buffer.
    absl::Span<std::byte> data();
    absl::Span<const std::byte> data() const;

    void SetExternalView(absl::Span<const std::byte> data);
    void SetOwnedBuffer(absl::Span<const std::byte> data);
    absl::Status Resize(size_t new_size);
    bool IsOwned() const { return external_view_.data() == nullptr; }

   private:
    absl::Span<std::byte> external_view_;
    std::vector<std::byte> owned_buffer_;
  };

#define TENSOR_API_UNIQUE_PTR_WITH_DELETER(NAME, TYPE, DEL_FUNC) \
  struct NAME##Deleter {                                         \
    void operator()(TYPE* data) const {                          \
      if (data) {                                                \
        DEL_FUNC(data);                                          \
      }                                                          \
    }                                                            \
  };                                                             \
  using NAME##Ptr = std::unique_ptr<TYPE, NAME##Deleter>

  TENSOR_API_UNIQUE_PTR_WITH_DELETER(Runtime, xnn_runtime, xnn_delete_runtime);
  TENSOR_API_UNIQUE_PTR_WITH_DELETER(Threadpool, pthreadpool,
                                     pthreadpool_destroy);
#undef TENSOR_API_UNIQUE_PTR_WITH_DELETER

  RuntimePtr runtime_ = nullptr;
  std::unique_ptr<XnnpackGraph> graph_;
  absl::flat_hash_map<uint32_t, ExternalBuffer> external_buffers_;
  ThreadpoolPtr threadpool_ = nullptr;
  size_t num_threads_ = 1;
  xnn_weights_cache_t weights_cache_ = nullptr;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_XNNPACK_RUNNER_H_
